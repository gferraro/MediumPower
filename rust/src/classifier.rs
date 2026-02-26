//! Animal classification from tracked regions.
//!
//! Mirrors the Python `LiteInterpreter`, `frame_samples`, `preprocess_segments`,
//! and `get_limits` logic in mediumpower.py.
//!
//! The actual TFLite inference is stubbed via the `Classifier` trait.
//! To enable inference:
//!   1. Add `tflite = "X.Y"` (or similar) to Cargo.toml.
//!   2. Implement the `predict` method on `LiteInterpreter`.

use std::path::{Path, PathBuf};
use anyhow::Context;
use rand::seq::SliceRandom;
use opencv::prelude::*;

use crate::basic_clip::BasicClip;
use crate::frame::ProcessedFrame;
use crate::region::Region;
use crate::tools::normalize;
use crate::track::Track;

// ── Classifier trait ──────────────────────────────────────────────────────────

/// Abstraction over an inference backend.
pub trait Classifier: Send {
    /// Run inference on the most recent frames of `track` in `clip`.
    /// Returns `Some((class_probabilities, frame_numbers, mean_mass))` or `None`.
    fn predict_recent_frames(
        &mut self,
        clip: &BasicClip,
        track: &Track,
    ) -> Option<(Vec<f32>, Vec<u32>, f32)>;
}

// ── LiteInterpreter (stub) ────────────────────────────────────────────────────

/// Wraps a TFLite model file.
/// The `predict` method is a stub — wire up `tflite-rs` or `ai-edge-litert` here.
pub struct LiteInterpreter {
    pub model_file: PathBuf,
    pub labels:     Vec<String>,
    pub version:    Option<String>,
    pub mapped_labels: Option<serde_json::Value>,
}

impl LiteInterpreter {
    /// Load model metadata from the `.json` sidecar file next to the `.tflite`.
    pub fn new(model_file: &Path) -> anyhow::Result<Self> {
        let json_path = model_file.with_extension("json");
        log::info!("Loading metadata from {:?}", json_path);

        let metadata: serde_json::Value = serde_json::from_reader(
            std::fs::File::open(&json_path)
                .with_context(|| format!("opening {:?}", json_path))?,
        )
        .context("parsing model JSON metadata")?;

        let version = metadata.get("version").and_then(|v| v.as_str()).map(|s| s.to_string());
        let labels: Vec<String> = metadata["labels"]
            .as_array()
            .context("labels not an array")?
            .iter()
            .filter_map(|v| v.as_str())
            .map(|s| s.to_string())
            .collect();
        let mapped_labels = metadata.get("mapped_labels").cloned();

        Ok(Self { model_file: model_file.to_path_buf(), labels, version, mapped_labels })
    }

    /// Run inference on a preprocessed input tensor.
    ///
    /// TODO: replace `todo!()` with actual TFLite inference once a crate is added.
    pub fn predict(&mut self, _input: &opencv::core::Mat) -> anyhow::Result<Vec<f32>> {
        todo!("TFLite inference — add tflite-rs (or ai-edge-litert) crate and implement here")
    }
}

impl Classifier for LiteInterpreter {
    fn predict_recent_frames(
        &mut self,
        clip: &BasicClip,
        track: &Track,
    ) -> Option<(Vec<f32>, Vec<u32>, f32)> {
        let samples = frame_samples(clip, track, 25);
        if samples.is_empty() { return None; }

        let (frame_numbers, preprocessed, mass) =
            preprocess_segments(clip, track, &samples).ok()??;

        let prediction = self.predict(&preprocessed).ok()?;
        Some((prediction, frame_numbers, mass))
    }
}

// ── frame sampling ────────────────────────────────────────────────────────────

/// Pick up to `num_frames` random valid regions from the track's recent history.
/// Mirrors `frame_samples` in mediumpower.py.
pub fn frame_samples<'a>(
    clip: &BasicClip,
    track: &'a Track,
    num_frames: usize,
) -> Vec<&'a Region> {
    let history = &track.bounds_history;
    let last_50 = &history[history.len().saturating_sub(50)..];

    // Find the start index such that frame_number >= current_frame - 50.
    let min_frame = (clip.current_frame - 50).max(0) as u32;
    let start_idx = last_50
        .iter()
        .position(|r| r.frame_number.unwrap_or(0) >= min_frame)
        .unwrap_or(0);

    let regions: Vec<&Region> = last_50[start_idx..]
        .iter()
        .filter(|r| {
            r.mass > 0.0
                && !r.blank
                && r.width > 0.0
                && r.height > 0.0
                && !clip.ffc_frames.contains(
                    &(r.frame_number.unwrap_or(0) as i32),
                )
        })
        .collect();

    if regions.is_empty() { return Vec::new(); }

    log::info!(
        "Getting frame samples current frame {} starting at {} frame# {:?}",
        clip.current_frame,
        start_idx,
        regions.first().and_then(|r| r.frame_number),
    );

    // Random sample without replacement.
    let mut rng = rand::thread_rng();
    let n = num_frames.min(regions.len());
    let mut idxs: Vec<usize> = (0..regions.len()).collect();
    idxs.shuffle(&mut rng);
    idxs.truncate(n);
    idxs.sort();
    idxs.iter().map(|&i| regions[i]).collect()
}

// ── preprocessing ─────────────────────────────────────────────────────────────

/// Build the tiled input tensor for the classifier.
/// Returns `Some((frame_numbers, preprocessed_mat, mean_mass))`.
/// Mirrors `preprocess_segments` in mediumpower.py.
pub fn preprocess_segments(
    clip: &BasicClip,
    track: &Track,
    samples: &[&Region],
) -> anyhow::Result<Option<(Vec<u32>, opencv::core::Mat, f32)>> {
    use crate::tools::preprocess_movement;

    if samples.is_empty() { return Ok(None); }

    let mut samples_sorted: Vec<&Region> = samples.to_vec();
    samples_sorted.sort_by_key(|r| r.frame_number.unwrap_or(0));

    let filtered_limits = get_limits(clip, track);

    let mut total_mass = 0.0f32;
    let mut clip_thermals_at_zero = true;
    let mut frame_temp_medians: std::collections::HashMap<u32, f32> =
        std::collections::HashMap::new();
    let mut cropped_frames: Vec<ProcessedFrame> = Vec::new();

    for region in &samples_sorted {
        total_mass += region.mass;
        let fn_ = region.frame_number.unwrap_or(0);
        let frame = match clip.get_frame(fn_) {
            Some(f) => f,
            None => continue,
        };

        // Compute per-frame thermal median.
        let median = mat_median(&frame.thermal)?;
        frame_temp_medians.insert(fn_, median);

        // Check sign of subtracted sub-thermal.
        if clip_thermals_at_zero {
            if let Ok(sub_roi) = region.subimage(&frame.thermal) {
                let sub_median = mat_median(&sub_roi)? - median;
                if sub_median <= 0.0 {
                    clip_thermals_at_zero = false;
                }
            }
        }

        let mut cropped = frame.crop_by_region(region)?;
        // Subtract per-frame median from thermal channel.
        {
            let m = *frame_temp_medians.get(&fn_).unwrap_or(&0.0);
            let mut shifted = opencv::core::Mat::default();
            cropped.thermal.convert_to(&mut shifted, opencv::core::CV_32F, 1.0, -(m as f64))?;
            cropped.thermal = shifted;
        }
        cropped.resize_with_aspect((32, 32), clip.crop_rectangle.as_ref(), false, (0,0,0,0), None)?;
        cropped_frames.push(cropped);
    }

    if cropped_frames.is_empty() { return Ok(None); }

    let mean_mass = total_mass / samples_sorted.len() as f32;

    // Normalise filtered and thermal channels.
    for frame in &mut cropped_frames {
        let (normed_filtered, _) = normalize(
            &frame.filtered,
            filtered_limits.0.map(|v| v as f64),
            filtered_limits.1.map(|v| v as f64),
            255.0,
        )?;
        frame.filtered = normed_filtered;

        let (normed_thermal, _) = normalize(&frame.thermal, None, None, 255.0)?;
        frame.thermal = normed_thermal;
    }

    // Build tiled movement mosaic: channels = [thermal=0, filtered=1, filtered=1].
    let preprocessed = match preprocess_movement(
        &cropped_frames,
        5,          // frames_per_row
        32,         // frame_size
        &[0, 1, 1], // channels
    )? {
        Some(m) => m,
        None => {
            log::warn!("No frames to predict on");
            return Ok(None);
        }
    };

    let frame_numbers: Vec<u32> = samples_sorted
        .iter()
        .filter_map(|r| r.frame_number)
        .collect();

    Ok(Some((frame_numbers, preprocessed, mean_mass)))
}

// ── normalisation limits ──────────────────────────────────────────────────────

/// Scan the track's bounds history to find the global min/max of the filtered
/// channel, used to normalise the filtered frames consistently.
/// Mirrors `get_limits` in mediumpower.py.
pub fn get_limits(clip: &BasicClip, track: &Track) -> (Option<f32>, Option<f32>) {
    let mut min_diff: Option<f32> = None;
    let mut max_diff: f32 = 0.0;

    for region in track.bounds_history.iter().rev() {
        if region.blank || region.width <= 0.0 || region.height <= 0.0 { continue; }
        let fn_ = match region.frame_number { Some(n) => n, None => continue };
        let frame = match clip.get_frame(fn_) { Some(f) => f, None => continue };

        let roi = match region.subimage(&frame.filtered) {
            Ok(r) => r,
            Err(_) => continue,
        };

        let mut mn = 0.0f64;
        let mut mx = 0.0f64;
        opencv::core::min_max_loc(&roi, Some(&mut mn), Some(&mut mx), None, None, &opencv::core::no_array())
            .unwrap_or(());

        if min_diff.is_none() || (mn as f32) < min_diff.unwrap() {
            min_diff = Some(mn as f32);
        }
        if (mx as f32) > max_diff {
            max_diff = mx as f32;
        }
    }

    (min_diff, Some(max_diff))
}

// ── helpers ───────────────────────────────────────────────────────────────────

fn mat_median(mat: &opencv::core::Mat) -> anyhow::Result<f32> {
    let rows = mat.rows() as usize;
    let cols = mat.cols() as usize;
    let n    = rows * cols;
    if n == 0 { return Ok(0.0); }

    let mut vals: Vec<f32> = Vec::with_capacity(n);
    for r in 0..rows {
        for c in 0..cols {
            vals.push(*mat.at_2d::<f32>(r as i32, c as i32).unwrap_or(&0.0));
        }
    }
    vals.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let mid = n / 2;
    Ok(if n % 2 == 0 { (vals[mid - 1] + vals[mid]) / 2.0 } else { vals[mid] })
}
