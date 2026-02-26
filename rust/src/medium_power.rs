//! Handles reading CPTV frames from either a gzip-compressed Unix-socket stream
//! or a test CPTV file, and puts them onto a channel for the classifier thread.
//!
//! Mirrors the Python functions in mediumpower.py.

use std::io::{Read, Write};
use std::ops::{Deref, DerefMut};
use std::os::unix::net::UnixStream;
use std::path::Path;
use std::sync::mpsc::Sender;
use std::time::{Duration, Instant};

use anyhow::{bail, Context};
use flate2::Decompress;
use flate2::FlushDecompress;

use crate::header_info::HeaderInfo;
use codec::decode::{CptvStreamDecoder,CptvFrame};

pub const SOCKET_NAME: &str = "/var/run/lepton-frames";
const CLEAR_SIGNAL: &[u8] = b"clear";

/// Messages sent from the reader thread to the classifier thread.
pub enum FrameMessage {
    /// A decoded frame plus the wall-clock time it was queued.
    Frame(CptvFrame, Instant),
    /// Signals the classifier to shut down.
    Stop,
}

// // ── CPTV reader stub ──────────────────────────────────────────────────────────

// /// Trait for reading CPTV frames from a byte buffer.
// /// Wire up the real `cptv` Rust crate here (the same library that backs
// /// the Python `cptv_rs_python_bindings` package).
// pub trait CptvReader: Send {
//     /// Try to read the next frame from `data`.
//     /// Returns `Some((frame, bytes_consumed))` or `None` if more data is needed.
//     fn next_frame_from_data(&mut self, data: &[u8]) -> Option<(CptvFrame, usize)>;

//     /// Read a complete frame from a file-backed reader (for test mode).
//     fn next_frame(&mut self) -> Option<CptvFrame>;
// }

// /// Placeholder that always returns `None`.  Replace with the real implementation.
// pub struct StubCptvReader;

// impl CptvReader for StubCptvReader {
//     fn next_frame_from_data(&mut self, _data: &[u8]) -> Option<(CptvFrame, usize)> {
//         todo!("Wire up the cptv Rust crate to implement CptvReader")
//     }
//     fn next_frame(&mut self) -> Option<CptvFrame> {
//         todo!("Wire up the cptv Rust crate to implement CptvReader")
//     }
// }

// ── gzip decompression ────────────────────────────────────────────────────────

/// Read and skip a gzip header from the front of `data`.
/// Returns the number of bytes consumed, or `None` if the header is incomplete.
///
/// A minimal gzip header is 10 bytes:
///   ID1 ID2 CM FLG MTIME(4) XFL OS
/// Followed by optional extensions depending on FLG bits.
fn skip_gzip_header(data: &[u8]) -> Option<usize> {
    if data.len() < 10 { return None; }
    // Magic + compression method check.
    if data[0] != 0x1f || data[1] != 0x8b || data[2] != 0x08 { return None; }

    let flg   = data[3];
    let mut pos = 10usize;

    // FEXTRA
    if flg & 0x04 != 0 {
        if data.len() < pos + 2 { return None; }
        let xlen = u16::from_le_bytes([data[pos], data[pos + 1]]) as usize;
        pos += 2 + xlen;
    }
    // FNAME
    if flg & 0x08 != 0 {
        while pos < data.len() && data[pos] != 0 { pos += 1; }
        pos += 1; // skip NUL
    }
    // FCOMMENT
    if flg & 0x10 != 0 {
        while pos < data.len() && data[pos] != 0 { pos += 1; }
        pos += 1;
    }
    // FHCRC
    if flg & 0x02 != 0 { pos += 2; }

    if pos > data.len() { None } else { Some(pos) }
}

/// Decompress accumulated data using a raw-deflate decompressor.
///
/// Mirrors the Python `decompress` function.
/// Returns `(remaining_data, decompressed_chunk, header_read)`.
pub fn decompress(
    decompressor: &mut Decompress,
    data: &[u8],
    read_header: bool,
) -> anyhow::Result<(Vec<u8>, Vec<u8>, bool)> {
    let (data_slice, new_read_header) = if !read_header {
        match skip_gzip_header(data) {
            Some(offset) => (&data[offset..], true),
            None => {
                log::info!("Couldn't read gzip header");
                return Ok((data.to_vec(), Vec::new(), false));
            }
        }
    } else {
        (data, true)
    };

    // Give the decompressor up to 4× the input as a scratch buffer.
    let buf_size = (data_slice.len() * 4).max(4096);
    let mut output = vec![0u8; buf_size];

    let before_in  = decompressor.total_in();
    let before_out = decompressor.total_out();

    let status = decompressor
        .decompress(data_slice, &mut output, FlushDecompress::None)
        .context("deflate decompress")?;

    let consumed = (decompressor.total_in()  - before_in)  as usize;
    let produced = (decompressor.total_out() - before_out) as usize;

    output.truncate(produced);
    let remaining = data_slice[consumed..].to_vec();

    Ok((remaining, output, new_read_header))
}

// ── header parsing ────────────────────────────────────────────────────────────

/// Read the header block terminated by `\n\n`, then consume the optional
/// 5-byte `clear` message.
pub fn handle_headers(conn: &mut UnixStream) -> anyhow::Result<(HeaderInfo, Vec<u8>)> {
    let mut headers = Vec::<u8>::new();
    let mut left_over = Vec::<u8>::new();

    loop {
        log::info!("Getting header info");
        let mut buf = vec![0u8; 4096];
        let n = conn.read(&mut buf).context("reading header")?;
        if n == 0 { bail!("Disconnected from camera while getting headers"); }
        headers.extend_from_slice(&buf[..n]);

        if let Some(done) = find_bytes(&headers, b"\n\n") {
            left_over = headers[done + 2..].to_vec();
            headers.truncate(done);

            // Ensure we have at least 5 bytes for the optional `clear` prefix.
            while left_over.len() < 5 {
                let mut extra = vec![0u8; 5 - left_over.len()];
                let n = conn.read(&mut extra)?;
                left_over.extend_from_slice(&extra[..n]);
            }
            if left_over.starts_with(CLEAR_SIGNAL) {
                left_over = left_over[5..].to_vec();
            }
            break;
        }
    }

    let header_s = String::from_utf8(headers).context("header not valid UTF-8")?;
    log::info!("header is {}", header_s);
    let info = HeaderInfo::parse_header(&header_s)?;
    Ok((info, left_over))
}

// ── medium-power streaming loop ───────────────────────────────────────────────

/// Read a medium-power gzip-compressed CPTV stream from `conn`,
/// decode frames, and push them onto `tx`.
///
/// Mirrors the Python `medium_power` function.
pub fn medium_power_loop(
    mut conn: UnixStream,
    headers: &HeaderInfo,
    extra_b: Vec<u8>,
    tx: &Sender<FrameMessage>,
) -> anyhow::Result<()> {
    let mut decompressor = Decompress::new(false); // raw deflate, wbits = -MAX_WBITS
    let mut u8_data: Vec<u8> = Vec::new();
    let mut data:    Vec<u8> = Vec::new();
    let mut read_header = false;
    let mut finished = false;
    let frame_size = headers.frame_size as usize;
    let mut cptv_decoder = CptvStreamDecoder::new();
    conn.set_read_timeout(Some(Duration::from_secs(5)))?;

    log::info!(
        "Headers frame size is {} extra_b size is {}",
        frame_size,
        extra_b.len()
    );

    let mut extra: Option<Vec<u8>> = if extra_b.is_empty() { None } else { Some(extra_b) };

    while !finished {

        let mut buf: Vec<u8> = vec![0u8; frame_size];
        let byte_data =  match conn.read(&mut buf) {
            Ok(bytes_read) => &buf[..bytes_read],
            Err(e) if e.kind() == std::io::ErrorKind::TimedOut => {
                log::info!("Timed out");
                std::thread::sleep(Duration::from_secs(1));
                continue
            }
            Err(e) => {
                log::error!("Error receiving: {}", e);
                data.clear();
                std::thread::sleep(Duration::from_secs(1));
                continue;
            }
        };

        if !byte_data.is_empty() {
            if let Some(clear_pos) = find_bytes(&byte_data, CLEAR_SIGNAL) {
                data.extend_from_slice(&byte_data[..clear_pos]);
                log::info!("Received clear — finished file");
                finished = true;
                tx.send(FrameMessage::Stop).ok();
            } else {
                data.extend_from_slice(&byte_data);
            }
        }

        // Decompress.
        match decompress(&mut decompressor, &data, read_header) {
            Ok((remaining, chunk, rh)) => {
                data         = remaining;
                read_header  = rh;
                if chunk.is_empty() { continue; }
                u8_data.extend_from_slice(&chunk);
            }
            Err(e) => {
                log::error!("Error decompressing: {}", e);
                std::thread::sleep(Duration::from_secs(1));
                if data.len() > 40_000 {
                    log::info!("Failed to decompress {} bytes", data.len());
                }
                continue;
            }
        }

        // Parse frames from decompressed data.
        loop {
            if let Ok((frame_ref, used)) =  cptv_decoder.next_frame_from_data(&u8_data){
                u8_data.drain(..used);
                // let frame = frame_ref.deref();
                tx.send(FrameMessage::Frame(frame_ref.clone(), Instant::now())).ok();
            }else{
                    if u8_data.len() > 40_000 {
                        log::info!("Have {} bytes of decompressed data but cannot parse a frame", u8_data.len());
                    }
                    break;
            }
        }
    }

    Ok(())
}

// ── test-mode CPTV file reader ────────────────────────────────────────────────

/// Read all frames from a CPTV file and push them onto `tx` at ~9 fps.
/// Mirrors the Python `parse_cptv` function.
pub fn parse_cptv(
    cptv_file: &str,
    tx: &Sender<FrameMessage>,
) -> anyhow::Result<()> {
    use codec::decode::{CptvDecoder};
    use std::fs::File;

    log::info!("Parsing CPTV file: {}", cptv_file);
    let file = File::open(&Path::new(cptv_file))?;
    let decoder = CptvDecoder::from(file)?;

    for  (_,frame) in decoder.enumerate(){
        tx.send(FrameMessage::Frame(frame, Instant::now())).ok();
        std::thread::sleep(Duration::from_millis(111)); // ~9 fps
    }
    tx.send(FrameMessage::Stop).ok();
    Ok(())
}

// ── utility ───────────────────────────────────────────────────────────────────

fn find_bytes(haystack: &[u8], needle: &[u8]) -> Option<usize> {
    haystack
        .windows(needle.len())
        .position(|w| w == needle)
}
