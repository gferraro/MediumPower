mod rectangle;
mod region;
mod frame;
mod frame_buffer;
mod basic_clip;
mod kalman;
mod region_tracker;
mod track;
mod motion_detector;
mod clip_track_extractor;
mod tools;
mod header_info;
mod medium_power;
mod classifier;

use std::os::unix::net::UnixListener;
use std::sync::mpsc;
use std::time::Instant;
use std::thread;

use anyhow::Context;
use log::{error, info};

use basic_clip::BasicClip;
use classifier::LiteInterpreter;
use clip_track_extractor::{ClipTrackExtractor, TrackingConfig};
use medium_power::{
    FrameMessage, SOCKET_NAME,
    handle_headers, medium_power_loop, parse_cptv,
};
use track::Track;

// ── entry point ───────────────────────────────────────────────────────────────

fn main() -> anyhow::Result<()> {
    env_logger::Builder::from_env(
        env_logger::Env::default().default_filter_or("info"),
    )
    .format_timestamp_secs()
    .init();

    let (tx, rx) = mpsc::channel::<FrameMessage>();

    // Spawn classifier / tracking thread.
    let processor = thread::spawn(move || {
        if let Err(e) = run_classifier(rx) {
            error!("Classifier thread error: {}", e);
        }
    });

    // Test mode: read from a local CPTV file.
    let test_mode = false;
    if test_mode {
        info!("Parsing test.cptv");
        parse_cptv("test.cptv", &tx)?;
        drop(tx);
        processor.join().ok();
        return Ok(());
    }

    // Production mode: listen on Unix socket.
    info!("Making sock");
    let _ = std::fs::remove_file(SOCKET_NAME);
    let listener = UnixListener::bind(SOCKET_NAME)
        .with_context(|| format!("binding Unix socket {}", SOCKET_NAME))?;
    listener.set_nonblocking(false)?;

    let start = Instant::now();
    loop {
        info!("Waiting for a connection ({:.1}s elapsed)", start.elapsed().as_secs_f32());
        match listener.accept() {
            Ok((mut conn, _addr)) => {
                info!("Connection accepted");
                match handle_headers(&mut conn) {
                    Ok((headers, extra_b)) => {
                        info!("Got headers: {}", headers);
                        if let Err(e) =
                            medium_power_loop(conn, &headers, extra_b, &tx)
                        {
                            error!("medium_power_loop error: {}", e);
                        }
                    }
                    Err(e) => error!("Error parsing headers: {}", e),
                }
            }
            Err(e) => {
                error!("Accept error: {}", e);
                break;
            }
        }
    }

    drop(tx);
    processor.join().ok();
    Ok(())
}

// ── classifier thread ─────────────────────────────────────────────────────────

fn new_clip() -> anyhow::Result<(ClipTrackExtractor, BasicClip)> {
    let config = TrackingConfig::default();
    let extractor = ClipTrackExtractor::new(config);
    let mut clip = BasicClip::new();
    clip.crop_rectangle = Some(extractor.background_alg.crop_rectangle.clone());
    Ok((extractor, clip))
}

fn load_model() -> anyhow::Result<LiteInterpreter> {
    info!("Loading TFLite model");
    let path = std::path::Path::new("./tflite/converted_model.tflite");
    let interp = LiteInterpreter::new(path)?;
    info!("Loaded TFLite model");
    Ok(interp)
}

fn run_classifier(rx: mpsc::Receiver<FrameMessage>) -> anyhow::Result<()> {
    // Load model in a background thread while frames start arriving.
    let model_handle = thread::spawn(load_model);

    info!("Loading clip");
    let (mut extractor, mut clip) = new_clip()?;
    let mut tracks: Vec<Track> = Vec::new();

    info!("Waiting for frames");
    let predict_every = 20usize;
    let mut frame_i = 0usize;

    for msg in rx {
        match msg {
            FrameMessage::Stop => {
                info!("Classifier received stop signal");
                break;
            }
            FrameMessage::Frame(cptv_frame, time_sent) => {
                if let Err(e) =
                    extractor.process_frame_with_tracks(&mut clip, &cptv_frame, &mut tracks)
                {
                    error!("Error processing frame: {}", e);
                    continue;
                }

                frame_i += 1;
                if frame_i % predict_every == 0 {
                    info!(
                        "Frame {} — behind by {:.3}s",
                        frame_i,
                        time_sent.elapsed().as_secs_f32()
                    );

                    // Get active tracks with enough history.
                    let active: Vec<&Track> = tracks
                        .iter()
                        .filter(|t| {
                            clip.active_track_ids.contains(&t.get_id())
                                && t.bounds_history.len() >= 8
                        })
                        .collect();

                    if active.is_empty() {
                        info!("No active tracks (total active: {})", clip.active_track_ids.len());
                        continue;
                    }

                    // Wait for model to finish loading (blocks only on first prediction).
                    if model_handle.is_finished() || frame_i == predict_every {
                        // Nothing to do — model loading is synchronous via join below.
                    }

                    identify_last_frame(&clip, active[0], &model_handle);
                }
            }
        }
    }

    Ok(())
}

fn identify_last_frame(
    clip: &BasicClip,
    track: &Track,
    model_handle: &thread::JoinHandle<anyhow::Result<LiteInterpreter>>,
) {
    // The model handle cannot be joined multiple times; in a real implementation
    // use Arc<Mutex<Option<LiteInterpreter>>> or a once-cell.
    // For now we print a placeholder log.
    info!("Track {} — would run inference here", track.get_id());
}
