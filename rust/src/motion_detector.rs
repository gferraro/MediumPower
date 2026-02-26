use anyhow::Context;
use opencv::{core, imgproc};
use opencv::core::{Mat, Scalar, CV_32F, CV_32S, no_array};
use opencv::prelude::*;

use codec::decode::CptvFrame;
use crate::rectangle::Rectangle;

// ── SlidingWindow ─────────────────────────────────────────────────────────────

/// Fixed-size circular buffer of optional frames.
/// Mirrors the Python `SlidingWindow` class in motiondetector.py.
pub struct SlidingWindow<T: Clone> {
    frames: Vec<Option<T>>,
    pub last_index:    Option<usize>,
    pub oldest_index:  Option<usize>,
    pub non_ffc_index: Option<usize>,
    pub ffc: bool,
    size: usize,
}

impl<T: Clone> SlidingWindow<T> {
    pub fn new(size: usize) -> Self {
        Self {
            frames: vec![None; size],
            last_index: None,
            oldest_index: None,
            non_ffc_index: None,
            ffc: false,
            size,
        }
    }

    pub fn add(&mut self, frame: T, ffc: bool) {
        if self.last_index.is_none() {
            self.oldest_index  = Some(0);
            self.frames[0]     = Some(frame);
            self.last_index    = Some(0);
            if !ffc { self.non_ffc_index = Some(0); }
        } else {
            let new_idx = (self.last_index.unwrap() + 1) % self.size;
            let oldest  = self.oldest_index.unwrap();
            if new_idx == oldest {
                if oldest == self.non_ffc_index.unwrap_or(usize::MAX) && !ffc {
                    self.non_ffc_index = Some((oldest + 1) % self.size);
                }
                self.oldest_index = Some((oldest + 1) % self.size);
            }
            self.frames[new_idx] = Some(frame);
            self.last_index = Some(new_idx);
        }
        if !ffc && self.ffc {
            self.non_ffc_index = self.last_index;
        }
        self.ffc = ffc;
    }

    pub fn current(&self) -> Option<&T> {
        self.last_index.and_then(|i| self.frames[i].as_ref())
    }

    pub fn oldest(&self) -> Option<&T> {
        self.oldest_index.and_then(|i| self.frames[i].as_ref())
    }

    pub fn oldest_nonffc(&self) -> Option<&T> {
        self.non_ffc_index.and_then(|i| self.frames[i].as_ref())
    }

    pub fn get(&self, i: usize) -> Option<&T> {
        self.frames[i % self.size].as_ref()
    }

    /// Return all frames in insertion order.
    pub fn get_frames(&self) -> Vec<&T> {
        if self.last_index.is_none() { return Vec::new(); }
        let mut frames = Vec::new();
        let end = (self.last_index.unwrap() + 1) % self.size;
        let mut cur = self.oldest_index.unwrap();
        loop {
            if let Some(f) = self.frames[cur].as_ref() {
                frames.push(f);
            }
            cur = (cur + 1) % self.size;
            if cur == end { break; }
        }
        frames
    }

    pub fn reset(&mut self) {
        self.last_index = None;
        self.oldest_index = None;
    }
}

// ── RunningMean ───────────────────────────────────────────────────────────────

/// Maintains a running sum (and count) for computing a sliding-window mean.
/// Mirrors the Python `RunningMean` class in motiondetector.py.
pub struct RunningMean {
    running_sum: Mat,    // CV_32S sum of pixel values
    frames: usize,
    window_size: usize,
}

impl RunningMean {
    /// Initialise from the first batch of frames (pix is CV_16U).
    pub fn new(data: &[Mat], window_size: usize) -> anyhow::Result<Self> {
        let first = &data[0];
        let mut f32_frame = Mat::default();
        first.convert_to(&mut f32_frame, CV_32S, 1.0, 0.0)?;
        let mut sum = f32_frame;

        for frame in data.iter().skip(1) {
            let mut f32_frame = Mat::default();
            frame.convert_to(&mut f32_frame, CV_32S, 1.0, 0.0)?;
            core::add(&sum.clone(), &f32_frame, &mut sum, &no_array(), -1)?;
        }
        Ok(Self { running_sum: sum, frames: data.len(), window_size })
    }

    /// Incorporate a new frame (and remove the oldest if at capacity).
    pub fn add(&mut self, new_data: &Mat, oldest_data: Option<&Mat>) -> anyhow::Result<()> {
        if self.frames == self.window_size {
            if let Some(old) = oldest_data {
                let mut old_s = Mat::default();
                old.convert_to(&mut old_s, CV_32S, 1.0, 0.0)?;
                core::subtract(&self.running_sum.clone(), &old_s, &mut self.running_sum, &no_array(), -1)?;
            }
        } else {
            self.frames += 1;
        }
        let mut new_s = Mat::default();
        new_data.convert_to(&mut new_s, CV_32S, 1.0, 0.0)?;
        core::add(&self.running_sum.clone(), &new_s, &mut self.running_sum, &no_array(), -1)?;
        Ok(())
    }

    /// Return the element-wise mean as a float32 Mat.
    pub fn mean(&self) -> anyhow::Result<Mat> {
        let mut result = Mat::default();
        let scale = 1.0 / self.frames as f64;
        self.running_sum.convert_to(&mut result, CV_32F, scale, 0.0)?;
        Ok(result)
    }
}

// ── WeightedBackground ────────────────────────────────────────────────────────

/// Background model updated with a weighted running minimum.
/// Mirrors the Python `WeightedBackground` class in motiondetector.py.
pub struct WeightedBackground {
    pub crop_rectangle: Rectangle,
    thermal_window: SlidingWindow<Mat>,
    background: Option<Mat>,        // full-size float background
    background_weight: Mat,         // float weight per pixel (crop size)
    running_mean: Option<RunningMean>,
    pub average: f32,
    edge_pixels: usize,
    weight_add: f32,
    num_frames: usize,
    pub movement_detected: bool,
    pub triggered: u32,
    pub ffc_affected: bool,
}

const MEAN_FRAMES: usize = 45;

impl WeightedBackground {
    pub fn new(edge_pixels: usize, res_x: usize, res_y: usize, weight_add: f32, init_average: f32) -> Self {
        let crop = Rectangle::new(
            edge_pixels as f32,
            edge_pixels as f32,
            (res_x - 2 * edge_pixels) as f32,
            (res_y - 2 * edge_pixels) as f32,
        );
        let crop_h = (res_y - 2 * edge_pixels) as i32;
        let crop_w = (res_x - 2 * edge_pixels) as i32;
        let bg_weight = Mat::zeros(crop_h, crop_w, CV_32F).unwrap().to_mat().unwrap();

        Self {
            crop_rectangle: crop,
            thermal_window: SlidingWindow::new(MEAN_FRAMES + 1),
            background: None,
            background_weight: bg_weight,
            running_mean: None,
            average: init_average,
            edge_pixels,
            weight_add,
            num_frames: 0,
            movement_detected: false,
            triggered: 0,
            ffc_affected: false,
        }
    }

    pub fn background(&self) -> Option<&Mat> {
        self.background.as_ref()
    }

    pub fn get_average(&self) -> f32 { self.average }

    /// Process one CPTV frame, updating the background model.
    pub fn process_frame(&mut self, cptv_frame: &CptvFrame, ffc_affected: bool) -> anyhow::Result<()> {
        self.num_frames += 1;
        let prev_ffc = self.ffc_affected;
        self.ffc_affected = ffc_affected;

        // Convert raw pix to float.
        let mut pix_f32 = Mat::default();
        let mat = Mat::from_slice(cptv_frame.image_data.as_slice()).unwrap();
        mat.convert_to(&mut pix_f32, CV_32F, 1.0, 0.0)?;

        self.thermal_window.add(pix_f32.clone(), ffc_affected);

        let oldest_pix = self.thermal_window.oldest().cloned();

        // Initialise or update running mean.
        if self.running_mean.is_none() {
            let frames_raw: Vec<&Mat> = self.thermal_window
                .get_frames()
                .into_iter()
                .take(MEAN_FRAMES)
                .collect();
            if !frames_raw.is_empty() {
                let frames_owned: Vec<Mat> = frames_raw.iter().map(|f| (**f).clone()).collect();
                self.running_mean = Some(RunningMean::new(&frames_owned, MEAN_FRAMES)?);
            }
        } else {
            let mean = self.running_mean.as_mut().unwrap();
            mean.add(&pix_f32, oldest_pix.as_ref())?;
        }

        // Compute the mean image cropped to the interior.
        let mean_mat = match &self.running_mean {
            Some(m) => m.mean()?,
            None => return Ok(()),
        };
        let crop_roi = self.crop_rectangle.to_cv_rect();
        let frame_crop = Mat::roi(&mean_mat, crop_roi)?;
        let mut frame_i32 = Mat::default();
        frame_crop.convert_to(&mut frame_i32, CV_32F, 1.0, 0.0)?;

        if self.background.is_none() {
            // First time: initialise background from mean.
            let ep = self.edge_pixels as i32;
            let res_y = 120;
            let res_x = 160;
            let mut bg = Mat::zeros(res_y, res_x, CV_32F)?.to_mat()?;
            // Copy crop into centre.
            let roi = opencv::core::Rect::new(ep, ep, frame_i32.cols(), frame_i32.rows());
            frame_i32.copy_to(&mut Mat::roi_mut(&mut bg, roi)?)?;
            self.average = mat_average(&frame_i32)?;
            self.background = Some(bg);
            self.set_background_edges()?;
            return Ok(());
        }

        let bg = self.background.as_mut().unwrap();

        // edgeless = copy of crop_rectangle subimage of background
        let mut edgeless = Mat::default();
        Mat::roi(bg, crop_roi)?.copy_to(&mut edgeless)?;

        // cond = edgeless < frame_i32 - background_weight
        let mut diff = Mat::default();
        core::subtract(&frame_i32, &self.background_weight, &mut diff, &no_array(), -1)?;
        let mut cond = Mat::default();
        core::compare(&edgeless, &diff, &mut cond, core::CMP_LT)?;

        // new_background = where(cond, edgeless, frame_i32)
        let mut new_background: Mat = frame_i32.clone();
        edgeless.copy_to_masked(&mut new_background, &cond)?;

        // new_weight = where(cond, weight + weight_add, 0)
        let mut new_weight = Mat::zeros(
            self.background_weight.rows(),
            self.background_weight.cols(),
            CV_32F,
        )?.to_mat()?;
        let mut weight_plus = Mat::default();
        self.background_weight.convert_to(
            &mut weight_plus, CV_32F, 1.0, self.weight_add as f64
        )?;
        weight_plus.copy_to_masked(&mut new_weight, &cond)?;

        // Detect any background change.
        let mut change_mask = Mat::default();
        core::compare(&new_background, &edgeless, &mut change_mask, core::CMP_NE)?;
        let any_change = core::count_non_zero(&change_mask)? > 0;

        if any_change {
            // Write updated crop back into background.
            new_background.copy_to(&mut Mat::roi_mut(bg, crop_roi)?)?;
            let old_avg = self.average;
            let mut edgeless_updated = Mat::default();
            Mat::roi(bg, crop_roi)?.copy_to(&mut edgeless_updated)?;
            self.average = mat_average(&edgeless_updated)?;
            if self.average != old_avg {
                log::debug!(
                    "MotionDetector temp threshold changed from {} to {}",
                    old_avg,
                    self.average
                );
            }
            self.set_background_edges()?;
        }

        self.background_weight = new_weight;

        if ffc_affected || prev_ffc {
            log::debug!("{} MotionDetector FFC", self.num_frames);
            self.movement_detected = false;
            self.triggered = 0;
            if prev_ffc {
                self.thermal_window.non_ffc_index = self.thermal_window.last_index;
            }
        }

        Ok(())
    }

    fn set_background_edges(&mut self) -> anyhow::Result<()> {
        let bg = match &mut self.background {
            Some(b) => b,
            None => return Ok(()),
        };
        let rows = bg.rows() as usize;
        let cols = bg.cols() as usize;
        let ep   = self.edge_pixels;

        for i in 0..ep {
            // Top edges: copy ep-th row into i-th row.
            let src_row = ep as i32;
            let mut src = Mat::default();
            Mat::roi(bg, opencv::core::Rect::new(0, src_row, cols as i32, 1))?.copy_to(&mut src)?;
            src.copy_to(&mut Mat::roi_mut(bg, opencv::core::Rect::new(0, i as i32, cols as i32, 1))?)?;

            // Bottom edges: copy (rows-ep-1)-th row into (rows-i-1)-th row.
            let src_row = (rows - ep - 1) as i32;
            let mut src = Mat::default();
            Mat::roi(bg, opencv::core::Rect::new(0, src_row, cols as i32, 1))?.copy_to(&mut src)?;
            src.copy_to(&mut Mat::roi_mut(bg, opencv::core::Rect::new(0, (rows - i - 1) as i32, cols as i32, 1))?)?;

            // Left edges: copy ep-th column into i-th column.
            let src_col = ep as i32;
            let mut src = Mat::default();
            Mat::roi(bg, opencv::core::Rect::new(src_col, 0, 1, rows as i32))?.copy_to(&mut src)?;
            src.copy_to(&mut Mat::roi_mut(bg, opencv::core::Rect::new(i as i32, 0, 1, rows as i32))?)?;

            // Right edges: copy (cols-ep-1)-th column into (cols-i-1)-th column.
            let src_col = (cols - ep - 1) as i32;
            let mut src = Mat::default();
            Mat::roi(bg, opencv::core::Rect::new(src_col, 0, 1, rows as i32))?.copy_to(&mut src)?;
            src.copy_to(&mut Mat::roi_mut(bg, opencv::core::Rect::new((cols - i - 1) as i32, 0, 1, rows as i32))?)?;
        }
        Ok(())
    }
}

fn mat_average(mat: &Mat) -> anyhow::Result<f32> {
    let s = core::mean(mat, &no_array())?;
    Ok(s[0] as f32)
}

impl Default for WeightedBackground {
    fn default() -> Self {
        Self::new(1, 160, 120, 1.0, 28000.0)
    }
}
