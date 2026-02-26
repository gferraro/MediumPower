use anyhow::Context;
use opencv::{core, imgproc};
use opencv::core::{Mat, Scalar, CV_32F, no_array};
use opencv::prelude::*;

use crate::basic_clip::BasicClip;
use codec::decode::CptvFrame;
use crate::motion_detector::WeightedBackground;
use crate::region::Region;
use crate::rectangle::Rectangle;
use crate::tools::{normalize, detect_objects};
use crate::track::Track;

// ── Configuration ─────────────────────────────────────────────────────────────

/// Tracker-specific numerical parameters loaded from config.
/// Mirrors the `params` dict on DefaultTracking in mediumpower.py.
#[derive(Debug, Clone)]
pub struct TrackerParams {
    pub base_distance_change: Option<f32>,
    pub min_mass_change:       Option<f32>,
    pub restrict_mass_after:   Option<f32>,
    pub mass_change_percent:   Option<f32>,
    pub max_distance:          Option<f32>,
    pub max_blanks:            Option<usize>,
    pub velocity_multiplier:   Option<f32>,
    pub base_velocity:         Option<f32>,
}

impl Default for TrackerParams {
    fn default() -> Self {
        Self {
            base_distance_change: Some(450.0),
            min_mass_change:       Some(20.0),
            restrict_mass_after:   Some(1.5),
            mass_change_percent:   Some(0.55),
            max_distance:          Some(2000.0),
            max_blanks:            Some(18),
            velocity_multiplier:   Some(2.0),
            base_velocity:         Some(2.0),
        }
    }
}

/// Top-level tracking configuration.
/// Mirrors `DefaultTracking` in mediumpower.py (the class built in `init_trackers`).
#[derive(Debug, Clone)]
pub struct TrackingConfig {
    pub edge_pixels:              usize,
    pub frame_padding:            usize,
    pub min_dimension:            usize,
    pub track_smoothing:          bool,
    pub denoise:                  bool,
    pub high_quality_optical_flow: bool,
    pub max_tracks:               Option<usize>,
    pub track_overlap_ratio:      f32,
    pub min_duration_secs:        f32,
    pub track_min_offset:         f32,
    pub track_min_mass:           f32,
    pub moving_vel_thresh:        f32,
    pub min_moving_frames:        usize,
    pub max_blank_percent:        f32,
    pub max_mass_std_percent:     f32,
    pub max_jitter:               f32,
    pub tracker_type:             String,
    pub tracker_camera_type:      String,
    pub aoi_min_mass:             f32,
    pub aoi_pixel_variance:       f32,
    pub cropped_regions_strategy: String,
    pub filter_regions_pre_match: bool,
    pub min_hist_diff:            Option<f32>,
    pub min_tag_confidence:       f32,
    pub enable_track_output:      bool,
    pub params:                   TrackerParams,
}

impl Default for TrackingConfig {
    fn default() -> Self {
        Self {
            edge_pixels:               1,
            frame_padding:             4,
            min_dimension:             0,
            track_smoothing:           false,
            denoise:                   false,
            high_quality_optical_flow: false,
            max_tracks:                None,
            track_overlap_ratio:       0.5,
            min_duration_secs:         0.0,
            track_min_offset:          4.0,
            track_min_mass:            2.0,
            moving_vel_thresh:         4.0,
            min_moving_frames:         2,
            max_blank_percent:         30.0,
            max_mass_std_percent:      0.55,
            max_jitter:                20.0,
            tracker_type:              "RegionTracker".to_string(),
            tracker_camera_type:       "thermal".to_string(),
            aoi_min_mass:              4.0,
            aoi_pixel_variance:        2.0,
            cropped_regions_strategy:  "cautious".to_string(),
            filter_regions_pre_match:  true,
            min_hist_diff:             None,
            min_tag_confidence:        0.8,
            enable_track_output:       true,
            params:                    TrackerParams::default(),
        }
    }
}

// ── ClipTrackExtractor ────────────────────────────────────────────────────────

/// Runs motion detection and multi-object tracking frame by frame.
/// Mirrors the Python `ClipTrackExtractor` class in cliptrackextractor.py.
pub struct ClipTrackExtractor {
    pub background_alg:    WeightedBackground,
    pub config:            TrackingConfig,
    pub background_thresh: f32,
    version:               String,
}

const VERSION: u32 = 11;

impl ClipTrackExtractor {
    pub fn new(config: TrackingConfig) -> Self {
        let padding = config.frame_padding.max(3);
        let mut cfg = config;
        cfg.frame_padding = padding;
        Self {
            background_alg:    WeightedBackground::default(),
            config:            cfg,
            background_thresh: 50.0,
            version:           format!("PI-{}", VERSION),
        }
    }

    pub fn tracker_version(&self) -> &str { &self.version }

    // ── per-frame processing ──────────────────────────────────────────────

    /// Process one incoming CPTV frame: update background, detect motion,
    /// match regions to existing tracks.
    pub fn process_frame(
        &mut self,
        clip: &mut BasicClip,
        frame: &CptvFrame,
    ) -> anyhow::Result<()> {
        let ffc_affected = crate::tools::is_affected_by_ffc(frame);

        self.background_alg.process_frame(frame, ffc_affected)?;

        // Convert raw pix to float32 thermal.
        let mut thermal = Mat::default();
        let mat = Mat::from_slice(frame.image_data.as_slice()).unwrap();
        mat.convert_to(&mut thermal, CV_32F, 1.0, 0.0)?;

        clip.ffc_affected = ffc_affected;

        // Compute filtered frame (background subtracted).
        let filtered: Mat = match self.background_alg.background() {
            Some(bg) => {
                let mut f = Mat::default();
                core::subtract(&thermal, bg, &mut f, &no_array(), -1)?;
                f
            }
            None => Mat::zeros(thermal.rows(), thermal.cols(), CV_32F)?.to_mat()?,
        };

        // Detect objects for tracking.
        let (obj_filtered, threshold) =
            self.get_filtered_frame(&thermal, false, self.config.denoise)?;

        let (_count, _labels, stats, centroids) = detect_objects(
            &obj_filtered,
            false,
            threshold as f64,
            (5, 5),
        )?;

        _ = clip.add_frame(thermal, filtered, None, ffc_affected);

        // Update active tracks.
        if ffc_affected {
            clip.active_track_ids.clear();
        } else {
            let regions = self.get_regions_of_interest(
                clip,
                &stats[1..],
                &centroids[1..],
            )?;
            self.apply_region_matchings(clip, regions)?;
        }

        Ok(())
    }

    // ── filtered frame ────────────────────────────────────────────────────

    fn get_filtered_frame(
        &self,
        thermal: &Mat,
        sub_change: bool,
        denoise: bool,
    ) -> anyhow::Result<(Mat, f32)> {
        let bg = match self.background_alg.background() {
            Some(b) => b,
            None => {
                let zeros = Mat::zeros(thermal.rows(), thermal.cols(), CV_32F)?.to_mat()?;
                let (normed, stats) = normalize(&zeros, None, None, 255.0)?;
                return Ok((normed, self.background_thresh));
            }
        };

        let mut filtered = Mat::default();
        thermal.convert_to(&mut filtered, CV_32F, 1.0, 0.0)?;

        let avg_change = if sub_change {
            let thermal_avg = core::mean(&filtered, &no_array())?[0];
            let bg_avg = self.background_alg.get_average() as f64;
            (thermal_avg - bg_avg).round() as f64
        } else {
            0.0
        };

        // filtered = clip(thermal - background - avg_change, 0, inf)
        let mut subtracted = Mat::default();
        core::subtract(&filtered, bg, &mut subtracted, &no_array(), -1)?;
        let mut shifted = Mat::default();
        subtracted.convert_to(&mut shifted, CV_32F, 1.0, -avg_change)?;

        // Clip negatives to zero (THRESH_TOZERO with threshold=0).
        let mut clipped = Mat::default();
        imgproc::threshold(&shifted, &mut clipped, 0.0, 0.0, imgproc::THRESH_TOZERO)?;

        let (normed, stats) = normalize(&clipped, None, None, 255.0)?;

        if denoise {
            let mut denoised = Mat::default();
            let mut u8_normed = Mat::default();
            normed.convert_to(&mut u8_normed, opencv::core::CV_8U, 1.0, 0.0)?;
            let mut f32_denoised = Mat::default();
            denoised.convert_to(&mut f32_denoised, CV_32F, 1.0, 0.0)?;
            let mapped_thresh = if stats.1 == stats.2 {
                self.background_thresh
            } else {
                self.background_thresh / (stats.1 - stats.2) as f32 * 255.0
            };
            return Ok((f32_denoised, mapped_thresh));
        }

        let mapped_thresh = if stats.1 == stats.2 {
            self.background_thresh
        } else {
            self.background_thresh / (stats.1 - stats.2) as f32 * 255.0
        };

        Ok((normed, mapped_thresh))
    }

    // ── regions of interest ───────────────────────────────────────────────

    fn get_regions_of_interest(
        &self,
        clip: &BasicClip,
        stats: &[crate::tools::ComponentStats],
        centroids: &[[f32; 2]],
    ) -> anyhow::Result<Vec<Region>> {
        let crop = match &clip.crop_rectangle {
            Some(c) => c.clone(),
            None => Rectangle::new(0.0, 0.0, clip.res_x as f32, clip.res_y as f32),
        };

        let delta_filtered = self.get_delta_frame(clip);

        let padding = self.config.frame_padding as f32;
        let mut regions = Vec::new();

        for (i, comp) in stats.iter().enumerate() {
            let centroid = if i < centroids.len() {
                Some([centroids[i][0], centroids[i][1]])
            } else {
                Some([
                    comp.x as f32 + comp.width as f32 / 2.0,
                    comp.y as f32 + comp.height as f32 / 2.0,
                ])
            };

            let mut region = Region::new(
                comp.x as f32,
                comp.y as f32,
                comp.width as f32,
                comp.height as f32,
                centroid,
            );
            region.mass = comp.area as f32;
            region.id   = i as u32;
            region.frame_number = Some(clip.current_frame as u32);

            if (region.width as usize) < self.config.min_dimension
                || (region.height as usize) < self.config.min_dimension
            {
                continue;
            }

            // Pixel variance from delta frame.
            if let Some(delta) = &delta_filtered {
                if let Ok(roi) = region.subimage(delta) {
                    let mean = core::mean(&roi, &no_array())?;
                    let mean_sq_diff = core::mean(&roi, &no_array())?; // reuse for variance calc
                    // Simple variance: compute manually.
                    region.pixel_variance = mat_variance(&roi)?;
                }
            }

            let old_region = region.copy();
            region.crop(&crop);
            region.was_cropped = format!("{}", old_region.rect) != format!("{}", region.rect);

            // Apply cropped-region strategy.
            match self.config.cropped_regions_strategy.as_str() {
                "cautious" => {
                    let cw_frac = (old_region.width - region.width) / old_region.width;
                    let ch_frac = (old_region.height - region.height) / old_region.height;
                    if cw_frac > 0.25 || ch_frac > 0.25 { continue; }
                }
                "none" => {
                    if region.was_cropped { continue; }
                }
                "all" => {}
                other => {
                    anyhow::bail!("Invalid cropped_regions_strategy: {}", other);
                }
            }

            // Pre-match noise filter.
            if self.config.filter_regions_pre_match
                && region.pixel_variance < self.config.aoi_pixel_variance
                && region.mass < self.config.aoi_min_mass
            {
                log::debug!(
                    "{} filtering region {} because of variance {} and mass {}",
                    region.frame_number.unwrap_or(0),
                    region,
                    region.pixel_variance,
                    region.mass,
                );
                continue;
            }

            region.enlarge(padding, Some(&crop));

            let extra_edge = (crop.width * 0.03).ceil();
            region.set_is_along_border(&crop, extra_edge);

            regions.push(region);
        }

        Ok(regions)
    }

    fn get_delta_frame(&self, clip: &BasicClip) -> Option<Mat> {
        let cur  = clip.frame_buffer.current_frame.as_ref()?;
        let prev = clip.frame_buffer.prev_frame.as_ref()?;

        let (f, _)  = normalize(&cur.filtered,  None, None, 255.0).ok()?;
        let (pf, _) = normalize(&prev.filtered, None, None, 255.0).ok()?;

        let mut f_f32  = Mat::default();
        let mut pf_f32 = Mat::default();
        f.convert_to(&mut f_f32,  CV_32F, 1.0, 0.0).ok()?;
        pf.convert_to(&mut pf_f32, CV_32F, 1.0, 0.0).ok()?;

        let mut delta = Mat::default();
        core::absdiff(&f_f32, &pf_f32, &mut delta).ok()?;
        Some(delta)
    }

    // ── region ↔ track matching ───────────────────────────────────────────

    fn apply_region_matchings(
        &self,
        clip: &mut BasicClip,
        regions: Vec<Region>,
    ) -> anyhow::Result<()> {
        let (unmatched_idxs, matched_ids) = self.match_existing_tracks(clip, &regions)?;

        let unmatched_regions: Vec<Region> = unmatched_idxs
            .into_iter()
            .map(|i| regions[i].clone())
            .collect();

        let new_ids = self.create_new_tracks(clip, unmatched_regions)?;

        // Tracks that were active but neither matched nor new → add blank frame.
        let unactive: Vec<u32> = clip
            .active_track_ids
            .iter()
            .copied()
            .filter(|id| !matched_ids.contains(id) && !new_ids.contains(id))
            .collect();

        clip.active_track_ids = matched_ids.union(&new_ids).copied().collect();
        self.filter_inactive_tracks(clip, unactive)?;

        clip.region_history.push(regions);
        Ok(())
    }

    fn match_existing_tracks(
        &self,
        clip: &mut BasicClip,
        regions: &[Region],
    ) -> anyhow::Result<(Vec<usize>, std::collections::HashSet<u32>)> {
        // Collect scores from every active track.
        let mut scores: Vec<(f32, u32, usize)> = Vec::new(); // (dist, track_id, region_idx)

        let active_ids: Vec<u32> = {
            let mut ids: Vec<u32> = clip.active_track_ids.iter().copied().collect();
            ids.sort();
            ids
        };

        // We need access to the tracks by id — find them in clip.track_ids mapped
        // to the Track objects stored in a separate Vec.  Because BasicClip only
        // stores track IDs (to avoid circular deps), the actual Track objects are
        // managed by ClipTrackExtractor's caller (run_classifier in main).
        // For simplicity this method signals match indices back to the caller
        // via a flat score list; the caller is responsible for calling track.add_region.
        //
        // NOTE: The real Track objects live in the `tracks` vec passed into the
        // frame-processing loop.  This method works with track IDs and returns
        // indices so the caller can apply updates.  See the usage in process_frame_with_tracks.
        Ok((
            (0..regions.len()).collect(),
            std::collections::HashSet::new(),
        ))
    }

    fn create_new_tracks(
        &self,
        clip: &mut BasicClip,
        regions: Vec<Region>,
    ) -> anyhow::Result<std::collections::HashSet<u32>> {
        let mut new_ids = std::collections::HashSet::new();
        for region in regions {
            let track = Track::from_region(
                clip.get_id(),
                clip.frames_per_second,
                clip.crop_rectangle.clone(),
                Some(self.version.clone()),
                &self.config,
                region,
            );
            let id = track.get_id();
            new_ids.insert(id);
            clip.add_active_track_id(id);
            // The track itself is returned to the caller via the public API below.
        }
        Ok(new_ids)
    }

    fn filter_inactive_tracks(
        &self,
        clip: &mut BasicClip,
        unactive_ids: Vec<u32>,
    ) -> anyhow::Result<()> {
        // Inactive tracks are kept alive if still tracking (blank frame added).
        // The actual Track objects are managed by the caller.
        for _id in unactive_ids {
            // Caller must call track.add_blank_frame() and re-add id if tracking.
        }
        Ok(())
    }

    /// Full frame-processing entry point that owns the Track vector.
    /// This is the preferred API for external callers.
    pub fn process_frame_with_tracks(
        &mut self,
        clip: &mut BasicClip,
        frame: &CptvFrame,
        tracks: &mut Vec<Track>,
    ) -> anyhow::Result<()> {
        let ffc_affected = crate::tools::is_affected_by_ffc(frame);
        self.background_alg.process_frame(frame, ffc_affected)?;


                let mut thermal = Mat::default();
        let mat = Mat::from_slice(frame.image_data.as_slice()).unwrap();
        mat.convert_to(&mut thermal, CV_32F, 1.0, 0.0)?;

        clip.ffc_affected = ffc_affected;

        let filtered: Mat = match self.background_alg.background() {
            Some(bg) => {
                let mut f = Mat::default();
                core::subtract(&thermal, bg, &mut f, &no_array(), -1)?;
                f
            }
            None => Mat::zeros(thermal.rows(), thermal.cols(), CV_32F)?.to_mat()?,
        };

        let (obj_filtered, threshold) =
            self.get_filtered_frame(&thermal, false, self.config.denoise)?;
        let (_count, _labels, stats, centroids) =
            detect_objects(&obj_filtered, false, threshold as f64, (5, 5))?;

        _ = clip.add_frame(thermal, filtered, None, ffc_affected);

        if ffc_affected {
            clip.active_track_ids.clear();
            return Ok(());
        }

        let regions = self.get_regions_of_interest(clip, &stats[1..], &centroids[1..])?;

        // --- match existing tracks ---
        let mut unmatched_region_idxs: Vec<usize> = (0..regions.len()).collect();
        let mut matched_track_ids = std::collections::HashSet::new();
        let mut blanked_track_ids = std::collections::HashSet::new();

        // Build score list across all active tracks.
        let mut all_scores: Vec<(f32, u32, usize)> = Vec::new(); // (dist, track_id, region_idx)

        let active_ids: Vec<u32> = {
            let mut ids: Vec<u32> = clip.active_track_ids.iter().copied().collect();
            ids.sort();
            ids
        };

        for &tid in &active_ids {
            if let Some(track) = tracks.iter().find(|t| t.get_id() == tid) {
                for (score, region_idx) in track.match_regions(&regions) {
                    all_scores.push((score, tid, region_idx));
                }
            }
        }

        // Sort: primary by frames_since_target_seen + fractional id, secondary by score.
        all_scores.sort_by(|a, b| {
            let ta = tracks.iter().find(|t| t.get_id() == a.1);
            let tb = tracks.iter().find(|t| t.get_id() == b.1);
            let ord_a = ta.map(|t| {
                t.frames_since_target_seen() as f32 + t.get_id() as f32 * 1e-6
            }).unwrap_or(0.0);
            let ord_b = tb.map(|t| {
                t.frames_since_target_seen() as f32 + t.get_id() as f32 * 1e-6
            }).unwrap_or(0.0);
            ord_a.partial_cmp(&ord_b).unwrap()
                .then(a.0.partial_cmp(&b.0).unwrap())
        });

        let mut used_region_idxs: std::collections::HashSet<usize> = std::collections::HashSet::new();

        for (_, tid, ridx) in &all_scores {
            if matched_track_ids.contains(tid) || used_region_idxs.contains(ridx) || blanked_track_ids.contains(tid) {
                continue;
            }
            let region = &regions[*ridx];

            // Post-match variance/mass filter (when filter_regions_pre_match is false).
            if !self.config.filter_regions_pre_match {
                if region.pixel_variance < self.config.aoi_pixel_variance
                    || region.mass < self.config.aoi_min_mass
                {
                    blanked_track_ids.insert(*tid);
                    continue;
                }
            }

            if let Some(track) = tracks.iter_mut().find(|t| t.get_id() == *tid) {
                track.add_region(region.clone());
                matched_track_ids.insert(*tid);
                used_region_idxs.insert(*ridx);
                if let Some(pos) = unmatched_region_idxs.iter().position(|&i| i == *ridx) {
                    unmatched_region_idxs.remove(pos);
                }
            }
        }

        // --- create new tracks for unmatched regions ---
        let mut new_track_ids = std::collections::HashSet::new();
        for ridx in &unmatched_region_idxs {
            let region = &regions[*ridx];
            // Skip if it overlaps significantly with an existing track.
            let max_overlap: f32 = tracks
                .iter()
                .filter(|t| clip.active_track_ids.contains(&t.get_id()))
                .filter_map(|t| t.last_bound())
                .map(|lb| lb.overlap_area(region) / region.area().max(1.0))
                .fold(0.0f32, f32::max);
            if max_overlap > 0.25 { continue; }

            let track = Track::from_region(
                clip.get_id(),
                clip.frames_per_second,
                clip.crop_rectangle.clone(),
                Some(self.version.clone()),
                &self.config,
                region.clone(),
            );
            let id = track.get_id();
            new_track_ids.insert(id);
            clip.add_active_track_id(id);
            tracks.push(track);
        }

        // --- add blank frame to unactive tracks that are still tracking ---
        let unactive: Vec<u32> = active_ids
            .iter()
            .copied()
            .filter(|id| !matched_track_ids.contains(id) && !new_track_ids.contains(id))
            .collect();

        clip.active_track_ids = matched_track_ids.union(&new_track_ids).copied().collect();

        for tid in unactive {
            if let Some(track) = tracks.iter_mut().find(|t| t.get_id() == tid) {
                track.add_blank_frame();
                if track.tracking() {
                    clip.active_track_ids.insert(tid);
                    log::debug!(
                        "frame {} adding blank frame to {}",
                        clip.current_frame,
                        tid
                    );
                }
            }
        }

        clip.region_history.push(regions);
        Ok(())
    }
}

fn mat_variance(mat: &Mat) -> anyhow::Result<f32> {
    let mean_s = core::mean(mat, &no_array())?;
    let mean   = mean_s[0] as f32;

    let rows = mat.rows() as usize;
    let cols = mat.cols() as usize;
    let n    = (rows * cols) as f32;
    if n == 0.0 { return Ok(0.0); }

    let mut sum_sq = 0.0f32;
    for r in 0..rows {
        for c in 0..cols {
            let v = *mat.at_2d::<f32>(r as i32, c as i32)
                .unwrap_or(&0.0);
            sum_sq += (v - mean).powi(2);
        }
    }
    Ok(sum_sq / n)
}
