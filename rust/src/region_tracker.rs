use crate::kalman::Kalman;
use crate::region::Region;
use crate::rectangle::Rectangle;

/// Per-track Kalman-based region tracker.
/// Mirrors the Python `RegionTracker` class in track.py.
pub struct RegionTracker {
    pub track_id: u32,
    pub clear_run: u32,
    pub kalman_tracker: Kalman,
    pub frames_since_target_seen: usize,
    pub frames: usize,
    pub blank_frames: usize,
    pub last_bound: Option<Region>,
    pub crop_rectangle: Option<Rectangle>,
    pub tracking: bool,
    pub predicted_mid: (f32, f32),

    // Config params (read from TrackingConfig in init).
    pub tracker_type: String,
    pub min_mass_change: f32,
    pub max_distance: f32,
    pub base_distance_change: f32,
    pub restrict_mass_after: f32,
    pub mass_change_percent: Option<f32>,
    pub velocity_multiplier: f32,
    pub base_velocity: f32,
    pub max_blanks: usize,
}

// Default tracker constants (thermal).
const MIN_KALMAN_FRAMES: usize = 18;
const DEFAULT_MAX_DISTANCE: f32 = 2000.0;
const DEFAULT_BASE_DISTANCE_CHANGE: f32 = 450.0;
const DEFAULT_MIN_MASS_CHANGE: f32 = 20.0;
const DEFAULT_RESTRICT_MASS_AFTER: f32 = 1.5;
const DEFAULT_MASS_CHANGE_PERCENT: f32 = 0.55;
const DEFAULT_VELOCITY_MULTIPLIER: f32 = 2.0;
const DEFAULT_BASE_VELOCITY: f32 = 2.0;
const DEFAULT_MAX_BLANKS: usize = 18;

impl RegionTracker {
    pub fn new(
        id: u32,
        config: &crate::clip_track_extractor::TrackingConfig,
        crop_rectangle: Option<Rectangle>,
    ) -> Self {
        Self {
            track_id: id,
            clear_run: 0,
            kalman_tracker: Kalman::new(),
            frames_since_target_seen: 0,
            frames: 0,
            blank_frames: 0,
            last_bound: None,
            crop_rectangle,
            tracking: false,
            predicted_mid: (0.0, 0.0),
            tracker_type: config.tracker_type.clone(),
            min_mass_change: config.params.min_mass_change
                .unwrap_or(DEFAULT_MIN_MASS_CHANGE),
            max_distance: config.params.max_distance
                .unwrap_or(DEFAULT_MAX_DISTANCE),
            base_distance_change: config.params.base_distance_change
                .unwrap_or(DEFAULT_BASE_DISTANCE_CHANGE),
            restrict_mass_after: config.params.restrict_mass_after
                .unwrap_or(DEFAULT_RESTRICT_MASS_AFTER),
            mass_change_percent: config.params.mass_change_percent,
            velocity_multiplier: config.params.velocity_multiplier
                .unwrap_or(DEFAULT_VELOCITY_MULTIPLIER),
            base_velocity: config.params.base_velocity
                .unwrap_or(DEFAULT_BASE_VELOCITY),
            max_blanks: config.params.max_blanks
                .unwrap_or(DEFAULT_MAX_BLANKS),
        }
    }

    /// Update tracker state with a new (non-blank) or blank region.
    pub fn add_region(&mut self, region: &Region) {
        self.frames += 1;
        if region.blank {
            self.blank_frames += 1;
            self.frames_since_target_seen += 1;
            let stop = ((2 * (self.frames - self.frames_since_target_seen)) as usize)
                .min(self.max_blanks);
            self.tracking = self.frames_since_target_seen < stop;
        } else {
            if self.frames_since_target_seen != 0 { self.clear_run = 0; }
            self.clear_run += 1;
            self.tracking = true;
            if let Some(c) = region.centroid {
                self.kalman_tracker.correct(c);
            }
            self.frames_since_target_seen = 0;
        }
        let (px, py) = self.kalman_tracker.predict();
        self.predicted_mid = (px, py);
        self.last_bound = Some(region.clone());
    }

    /// Create a blank region using Kalman prediction, add it, and return it.
    pub fn add_blank_frame(&mut self) -> Region {
        let kalman_amount = self.frames as i64
            - MIN_KALMAN_FRAMES as i64
            - (self.frames_since_target_seen * 2) as i64;

        let region = if kalman_amount > 0 {
            if let Some(lb) = &self.last_bound {
                let mut r = Region::new(
                    (self.predicted_mid.0 - lb.width / 2.0) as f32,
                    (self.predicted_mid.1 - lb.height / 2.0) as f32,
                    lb.width,
                    lb.height,
                    Some([self.predicted_mid.0, self.predicted_mid.1]),
                );
                r.frame_number = lb.frame_number.map(|n| n + 1);
                if let Some(crop) = &self.crop_rectangle {
                    r.crop(crop);
                }
                r.blank = true;
                r.mass = 0.0;
                r.pixel_variance = 0.0;
                r
            } else {
                self.make_copy_blank()
            }
        } else {
            self.make_copy_blank()
        };

        self.add_region(&region.clone());
        region
    }

    fn make_copy_blank(&self) -> Region {
        let mut r = self
            .last_bound
            .as_ref()
            .map(|lb| {
                let mut copy = lb.copy();
                copy.blank = true;
                copy.mass = 0.0;
                copy.pixel_variance = 0.0;
                copy.frame_number = lb.frame_number.map(|n| n + 1);
                copy
            })
            .unwrap_or_else(|| Region::new(0.0, 0.0, 1.0, 1.0, None));
        r.blank = true;
        r
    }

    /// Returns the predicted velocity based on Kalman vs last centroid.
    pub fn predicted_velocity(&self, track: &crate::track::Track) -> (f32, f32) {
        if self.last_bound.is_none()
            || (self.frames - self.blank_frames) <= MIN_KALMAN_FRAMES
        {
            return (0.0, 0.0);
        }
        let centroid = self.last_bound.as_ref().and_then(|lb| lb.centroid).unwrap_or([0.0, 0.0]);
        (
            self.predicted_mid.0 - centroid[0],
            self.predicted_mid.1 - centroid[1],
        )
    }

    // ── region matching ──────────────────────────────────────────────────

    /// Score each candidate region against this tracker, returning
    /// `(distance_score, region_index)` pairs that pass all filters.
    pub fn match_regions(
        &self,
        regions: &[Region],
        track: &crate::track::Track,
    ) -> Vec<(f32, usize)> {
        let avg_mass = track.average_mass();
        let avg_area = track.average_area();
        let max_distances = self.get_max_distance_change(track);
        let mut scores = Vec::new();

        let lb = match &self.last_bound {
            Some(lb) => lb,
            None => return scores,
        };

        for (idx, region) in regions.iter().enumerate() {
            // Mass check.
            let max_mass_change = self.get_max_mass_change_percent(track, avg_mass);
            if let Some(max_mc) = max_mass_change {
                if (avg_mass - region.mass).abs() > max_mc { continue; }
            }

            // Distance check.
            let raw_dists = lb.average_distance(region);
            // Use only the combined top-left + bottom-right average (non-thermal path).
            let distance = (raw_dists[0] + raw_dists[2]) / 2.0;
            let max_dist = max_distances[0];

            if let Some(md) = max_dist {
                if distance > md { continue; }
            }

            // Size change check.
            let size_change = get_size_change(avg_area, region);
            let max_size_change = get_max_size_change(track, region);
            if size_change > max_size_change { continue; }

            scores.push((distance, idx));
        }

        scores
    }

    fn get_max_distance_change(&self, track: &crate::track::Track) -> Vec<Option<f32>> {
        let (mut vx, mut vy) = track.velocity();
        if track.frames() == 1 {
            vx = self.base_velocity;
            vy = self.base_velocity;
        }
        let vx = vx * self.velocity_multiplier;
        let vy = vy * self.velocity_multiplier;
        let vel_dist = vx * vx + vy * vy;

        let (pvx, pvy) = track.predicted_velocity();
        let pred_dist = pvx * pvx + pvy * pvy;
        let max_dist = self.base_distance_change + vel_dist.max(pred_dist);

        vec![Some(max_dist), None, Some(max_dist)]
    }

    fn get_max_mass_change_percent(&self, track: &crate::track::Track, avg_mass: f32) -> Option<f32> {
        let pct = self.mass_change_percent?;
        let fps = track.fps;
        if track.frames() as f32 > self.restrict_mass_after * fps {
            let (vx, vy) = track.velocity();
            let mp = if (vx.abs() + vy.abs()) > 5.0 { pct + 0.1 } else { pct };
            Some(self.min_mass_change.max(avg_mass * mp))
        } else {
            None
        }
    }
}

fn get_size_change(current_area: f32, region: &Region) -> f32 {
    (region.area() - current_area).abs() / (current_area + 50.0)
}

fn get_max_size_change(track: &crate::track::Track, region: &Region) -> f32 {
    let lb = match track.last_bound() { Some(lb) => lb, None => return 1.5 };
    let exiting = region.is_along_border && !lb.is_along_border;
    let entering = !exiting && lb.is_along_border;
    let (vx, vy) = track.velocity();
    let vel = vx.abs() + vy.abs();

    let mut pct = if track.frames() < 5 { 2.0 } else { 1.5 };
    if entering || exiting {
        pct = 2.0;
        if vel > 10.0 { pct *= 3.0; }
    } else if vel > 10.0 {
        pct *= 2.0;
    }
    pct
}
