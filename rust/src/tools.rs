use anyhow::Context;
use opencv::{core, imgproc};
use opencv::core::{Mat, Scalar, Size, CV_32F, CV_8U, no_array};
use opencv::prelude::*;
use codec::decode::CptvFrame;

use crate::frame:: ProcessedFrame;
use crate::rectangle::Rectangle;
use crate::region::Region;

// ── image statistics ──────────────────────────────────────────────────────────

/// Normalize a float32 Mat so values span [0, new_max].
/// Returns `(normalized_mat, (success, max_used, min_used))`.
/// Mirrors the Python `normalize` function in tools.py.
pub fn normalize(
    data: &Mat,
    min_val: Option<f64>,
    max_val: Option<f64>,
    new_max: f64,
) -> anyhow::Result<(Mat, (bool, f64, f64))> {
    if data.empty() {
        let zeros = Mat::zeros(data.rows(), data.cols(), CV_32F)?.to_mat()?;
        return Ok((zeros, (false, 0.0, 0.0)));
    }

    let (mut mn, mut mx) = (min_val.unwrap_or(0.0), max_val.unwrap_or(0.0));

    if min_val.is_none() || max_val.is_none() {
        let mut mn_f = 0.0f64;
        let mut mx_f = 0.0f64;
        core::min_max_loc(data, Some(&mut mn_f), Some(&mut mx_f), None, None, &no_array())?;
        if min_val.is_none() { mn = mn_f; }
        if max_val.is_none() { mx = mx_f; }
    }

    if mx == mn {
        if mx == 0.0 {
            let zeros = Mat::zeros(data.rows(), data.cols(), CV_32F)?.to_mat()?;
            return Ok((zeros, (false, mx, mn)));
        }
        let mut result = Mat::default();
        data.convert_to(&mut result, CV_32F, 1.0 / mx, 0.0)?;
        return Ok((result, (true, mx, mn)));
    }

    let scale = new_max / (mx - mn);
    let shift = -mn * scale;
    let mut result = Mat::default();
    data.convert_to(&mut result, CV_32F, scale, shift)?;
    Ok((result, (true, mx, mn)))
}

// ── motion detection helpers ──────────────────────────────────────────────────

/// Stats returned for each connected component:  x, y, w, h, area (pixel count).
#[derive(Debug, Clone)]
pub struct ComponentStats {
    pub x:      i32,
    pub y:      i32,
    pub width:  i32,
    pub height: i32,
    pub area:   i32,
}

/// Detect objects in a thresholded + morphologically-closed image.
/// Returns `(num_labels, label_mat, stats_vec, centroids_vec)`.
/// Mirrors the Python `detect_objects` function in tools.py.
pub fn detect_objects(
    image: &Mat,
    otsus: bool,
    threshold: f64,
    kernel: (i32, i32),
) -> anyhow::Result<(i32, Mat, Vec<ComponentStats>, Vec<[f32; 2]>)> {
    // Convert to u8 for OpenCV threshold operations.
    let mut u8_image = Mat::default();
    image.convert_to(&mut u8_image, CV_8U, 1.0, 0.0)?;

    // Gaussian blur.
    let ksize = Size::new(kernel.0, kernel.1);
    let mut blurred = Mat::default();
    imgproc::gaussian_blur(&u8_image, &mut blurred, ksize, 0.0, 0.0, core::BORDER_DEFAULT)?;

    // Threshold.
    let thresh_type = if otsus {
        imgproc::THRESH_BINARY + imgproc::THRESH_OTSU
    } else {
        imgproc::THRESH_BINARY
    };
    let mut threshed = Mat::default();
    imgproc::threshold(&blurred, &mut threshed, threshold, 255.0, thresh_type)?;

    // Morphological close.
    let struct_elem = imgproc::get_structuring_element(
        imgproc::MORPH_RECT,
        ksize,
        core::Point::new(-1, -1),
    )?;
    let mut closed = Mat::default();
    imgproc::morphology_ex(
        &threshed,
        &mut closed,
        imgproc::MORPH_CLOSE,
        &struct_elem,
        core::Point::new(-1, -1),
        1,
        core::BORDER_CONSTANT,
        imgproc::morphology_default_border_value()?,
    )?;

    // Connected components with stats.
    let mut labels     = Mat::default();
    let mut cv_stats   = Mat::default();
    let mut cv_cents   = Mat::default();
    let num_labels = imgproc::connected_components_with_stats(
        &closed,
        &mut labels,
        &mut cv_stats,
        &mut cv_cents,
        8,
        core::CV_32S,
    )?;

    // Convert OpenCV stats/centroids Mats to Rust vecs.
    let mut stats: Vec<ComponentStats> = Vec::with_capacity(num_labels as usize);
    let mut centroids: Vec<[f32; 2]>  = Vec::with_capacity(num_labels as usize);

    for i in 0..num_labels as usize {
        let x      = *cv_stats.at_2d::<i32>(i as i32, imgproc::CC_STAT_LEFT)?;
        let y      = *cv_stats.at_2d::<i32>(i as i32, imgproc::CC_STAT_TOP)?;
        let width  = *cv_stats.at_2d::<i32>(i as i32, imgproc::CC_STAT_WIDTH)?;
        let height = *cv_stats.at_2d::<i32>(i as i32, imgproc::CC_STAT_HEIGHT)?;
        let area   = *cv_stats.at_2d::<i32>(i as i32, imgproc::CC_STAT_AREA)?;
        stats.push(ComponentStats { x, y, width, height, area });

        let cx = *cv_cents.at_2d::<f64>(i as i32, 0)? as f32;
        let cy = *cv_cents.at_2d::<f64>(i as i32, 1)? as f32;
        centroids.push([cx, cy]);
    }

    Ok((num_labels, labels, stats, centroids))
}

/// Apply a Gaussian blur then threshold to produce a binary mask.
/// Returns `(mask_mat, pixel_count_above_threshold)`.
/// Mirrors `blur_and_return_as_mask` in region.py.
pub fn blur_and_return_as_mask(frame: &Mat, threshold: f32) -> anyhow::Result<(Mat, usize)> {
    let ksize = Size::new(5, 5);
    let mut blurred = Mat::default();

    // Convert to f32 if needed before blurring.
    let mut f32_frame = Mat::default();
    frame.convert_to(&mut f32_frame, CV_32F, 1.0, 0.0)?;

    imgproc::gaussian_blur(&f32_frame, &mut blurred, ksize, 0.0, 0.0, core::BORDER_DEFAULT)?;

    // Zero out values below threshold (in-place subtraction then clip).
    let mut threshed = blurred.clone();
    let thresh_scalar = Scalar::all(threshold as f64);
    core::subtract(&blurred, &thresh_scalar, &mut threshed, &no_array(), -1)?;

    // Count positive pixels.
    let mut count = 0usize;
    let rows = threshed.rows() as usize;
    let cols = threshed.cols() as usize;
    for r in 0..rows {
        for c in 0..cols {
            if *threshed.at_2d::<f32>(r as i32, c as i32)? > 0.0 {
                count += 1;
            }
        }
    }

    Ok((threshed, count))
}

// ── FFC detection ─────────────────────────────────────────────────────────────

/// Returns `true` if `frame` was captured during a Flat-Field Correction event.
/// Mirrors `is_affected_by_ffc` in tools.py / cliptrackextractor.py.
pub fn is_affected_by_ffc(frame: &CptvFrame) -> bool {
    // 9.9 seconds expressed in milliseconds.
    const FFC_THRESHOLD_MS: u32 = 9_900;
    if frame.time_on == 0 && frame.last_ffc_time == 0 {
        return false;
    }
    frame.time_on.saturating_sub(frame.last_ffc_time) < FFC_THRESHOLD_MS
}

// ── geometry ──────────────────────────────────────────────────────────────────

/// Squared Euclidean distance between two 2-D points.
pub fn eucl_distance_sq(first: (f32, f32), second: (f32, f32)) -> f32 {
    let dx = first.0 - second.0;
    let dy = first.1 - second.1;
    dx * dx + dy * dy
}

// ── image resizing ────────────────────────────────────────────────────────────

/// Resize `frame` to fit within `new_dim` (height, width) while preserving
/// aspect ratio, then centre-pad to exactly `new_dim`.
/// Mirrors `resize_and_pad` in tools.py.
pub fn resize_and_pad(
    frame: &Mat,
    new_dim: (i32, i32),                // (height, width)
    region: Option<&Region>,
    crop_region: Option<&Rectangle>,
    keep_edge: bool,
    pad: Option<f64>,                   // None → use channel minimum
    edge_offset: (i32, i32, i32, i32),  // (left, top, right, bottom)
    original_region: Option<&Region>,
) -> anyhow::Result<Mat> {
    let new_h = new_dim.0 as f64;
    let new_w = new_dim.1 as f64;

    let src_h = frame.rows() as f64;
    let src_w = frame.cols() as f64;

    let scale = (new_h / src_h).min(new_w / src_w);
    let w = ((src_w * scale).round() as i32).max(1).min(new_dim.1);
    let h = ((src_h * scale).round() as i32).max(1).min(new_dim.0);

    // Resize preserving aspect ratio.
    let mut resized = Mat::default();
    imgproc::resize(
        frame,
        &mut resized,
        Size::new(w, h),
        0.0, 0.0,
        imgproc::INTER_LINEAR,
    )?;

    // Determine pad value.
    let pad_val = if let Some(p) = pad {
        p
    } else {
        let mut mn = 0.0f64;
        core::min_max_loc(frame, Some(&mut mn), None, None, None, &no_array())?;
        mn
    };

    // Build output canvas filled with pad value.
    let mut canvas = Mat::new_rows_cols_with_default(
        new_dim.0,
        new_dim.1,
        frame.typ(),
        Scalar::all(pad_val),
    )?;

    // Compute offset to centre the resized image (or snap to edge).
    let mut off_x = (new_dim.1 - w) / 2;
    let mut off_y = (new_dim.0 - h) / 2;

    if keep_edge {
        if let (Some(reg), Some(crop)) = (original_region.or(region), crop_region) {
            if reg.left() <= crop.left() {
                off_x = edge_offset.0.min(new_dim.1 - w);
            } else if reg.right() >= crop.right() {
                off_x = ((new_dim.1 - edge_offset.2) - w).max(0);
            }
            if reg.top() <= crop.top() {
                off_y = edge_offset.1.min(new_dim.0 - h);
            } else if reg.bottom() >= crop.bottom() {
                off_y = (new_dim.0 - h - edge_offset.3).max(0);
            }
        }
    }

    // Copy resized image into canvas.
    let roi = opencv::core::Rect::new(off_x, off_y, w, h);
    resized.copy_to(&mut Mat::roi_mut(&mut canvas, roi)?)?;

    Ok(canvas)
}

// ── preprocessing for classification ─────────────────────────────────────────

/// Tile `frames_per_row × frames_per_row` frame patches into a single square image.
/// Mirrors `square_clip` in tools.py.
pub fn square_clip(
    data: &[Mat],
    frames_per_row: usize,
    tile_dim: (usize, usize),
    frame_samples: &[usize],
    do_normalize: bool,
) -> anyhow::Result<(Mat, bool)> {
    let h = (frames_per_row * tile_dim.0) as i32;
    let w = (frames_per_row * tile_dim.1) as i32;
    let mut canvas = Mat::zeros(h, w, CV_32F)?.to_mat()?;

    let mut success = false;
    let mut i = 0usize;

    'outer: for x in 0..frames_per_row {
        for y in 0..frames_per_row {
            if i >= frame_samples.len() { break 'outer; }
            let frame = &data[frame_samples[i]];
            i += 1;

            let tile = if do_normalize {
                let (normed, stats) = normalize(frame, None, None, 255.0)?;
                if !stats.0 { continue; }
                normed
            } else {
                let mut f32_frame = Mat::default();
                frame.convert_to(&mut f32_frame, CV_32F, 1.0, 0.0)?;
                f32_frame
            };

            // Resize tile to tile_dim.
            let mut resized = Mat::default();
            imgproc::resize(
                &tile,
                &mut resized,
                Size::new(tile_dim.1 as i32, tile_dim.0 as i32),
                0.0, 0.0,
                imgproc::INTER_LINEAR,
            )?;

            let row_start = (x * tile_dim.0) as i32;
            let col_start = (y * tile_dim.1) as i32;
            let roi = opencv::core::Rect::new(
                col_start, row_start,
                tile_dim.1 as i32, tile_dim.0 as i32,
            );
            resized.copy_to(&mut Mat::roi_mut(&mut canvas, roi)?)?;
            success = true;
        }
    }

    Ok((canvas, success))
}

/// Build the tiled movement mosaic used as classifier input.
/// Returns a 3-channel (H×W×3) float32 Mat (channels: thermal, filtered, filtered).
/// Mirrors `preprocess_movement` in tools.py.
pub fn preprocess_movement(
    frames: &[ProcessedFrame],
    frames_per_row: usize,
    frame_size: usize,
    channels: &[usize],
) -> anyhow::Result<Option<Mat>> {
    let n_needed = frames_per_row * frames_per_row;
    let n_have   = frames.len();

    // Build sample indices, repeating randomly if not enough frames.
    let mut rng = rand::thread_rng();
    let mut frame_samples: Vec<usize> = (0..n_have).collect();
    if n_have < n_needed {
        use rand::seq::SliceRandom;
        let extra: Vec<usize> = (0..(n_needed - n_have))
            .map(|_| *frame_samples.choose(&mut rng).unwrap())
            .collect();
        frame_samples.extend(extra);
        frame_samples.sort();
    }

    // Build per-channel tiled images.
    let mut channel_mats: Vec<Mat> = Vec::new();
    let mut seen: std::collections::HashMap<usize, Mat> = std::collections::HashMap::new();

    for &ch in channels {
        if let Some(m) = seen.get(&ch) {
            channel_mats.push(m.clone());
            continue;
        }
        let channel_data: Vec<Mat> = frames
            .iter()
            .map(|f| {
                let mut m = Mat::default();
                f.get_channel(ch).convert_to(&mut m, CV_32F, 1.0, 0.0)?;
                Ok(m)
            })
            .collect::<anyhow::Result<Vec<_>>>()?;

        let (tiled, ok) = square_clip(
            &channel_data,
            frames_per_row,
            (frame_size, frame_size),
            &frame_samples,
            false,
        )?;
        if !ok {
            return Ok(None);
        }
        seen.insert(ch, tiled.clone());
        channel_mats.push(tiled);
    }

    // Stack channels into a single multi-channel Mat.
    let mut merged = Mat::default();
    let channel_vec: opencv::core::Vector<Mat> = channel_mats.into_iter().collect();
    core::merge(&channel_vec, &mut merged)?;
    Ok(Some(merged))
}
