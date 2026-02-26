use crate::region::Region;
use crate::rectangle::Rectangle;
use crate::region_tracker::RegionTracker;

/// Statistics computed over a completed track.
#[derive(Debug, Clone, Default)]
pub struct TrackMovementStatistics {
    pub movement:         f32,
    pub max_offset:       f32,
    pub score:            f32,
    pub average_mass:     f32,
    pub median_mass:      f32,
    pub delta_std:        f32,
    pub region_jitter:    i32,
    pub jitter_bigger:    i32,
    pub jitter_smaller:   i32,
    pub blank_percent:    i32,
    pub frames_moved:     i32,
    pub mass_std:         f32,
    pub average_velocity: f32,
}

static TRACK_ID_COUNTER: std::sync::atomic::AtomicU32 =
    std::sync::atomic::AtomicU32::new(1);

/// Bounds of a tracked object over time.
/// Mirrors the Python `Track` class in track.py.
pub struct Track {
    id: u32,
    pub clip_id: u32,
    pub start_frame: Option<u32>,
    pub start_s: Option<f32>,
    pub end_s: Option<f32>,
    pub fps: f32,
    pub bounds_history: Vec<Region>,
    pub vel_x: Vec<f32>,
    pub vel_y: Vec<f32>,
    pub tracker: RegionTracker,
    pub tag: String,
    pub prev_frame_num: Option<u32>,
    pub stats: Option<TrackMovementStatistics>,
    pub crop_rectangle: Option<Rectangle>,
    pub tracker_version: Option<String>,
}

impl Track {
    /// Create a new track seeded from the first region detection.
    pub fn from_region(
        clip_id: u32,
        fps: f32,
        crop_rectangle: Option<Rectangle>,
        tracker_version: Option<String>,
        tracking_config: &crate::clip_track_extractor::TrackingConfig,
        region: Region,
    ) -> Self {
        let id = TRACK_ID_COUNTER.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let tracker = RegionTracker::new(id, tracking_config, crop_rectangle.clone());
        let start_frame = region.frame_number;
        let start_s = start_frame.map(|n| n as f32 / fps);
        let mut t = Self {
            id,
            clip_id,
            start_frame,
            start_s,
            end_s: None,
            fps,
            bounds_history: Vec::new(),
            vel_x: Vec::new(),
            vel_y: Vec::new(),
            tracker,
            tag: "unknown".to_string(),
            prev_frame_num: None,
            stats: None,
            crop_rectangle,
            tracker_version,
        };
        t.add_region(region);
        t
    }

    pub fn get_id(&self) -> u32 { self.id }

    // ── properties matching Python ──────────────────────────────────────

    pub fn end_frame(&self) -> Option<u32> {
        self.bounds_history.last().and_then(|r| r.frame_number)
    }

    pub fn frames(&self) -> usize { self.bounds_history.len() }

    pub fn last_bound(&self) -> Option<&Region> { self.bounds_history.last() }

    pub fn last_mass(&self) -> f32 {
        self.bounds_history.last().map(|r| r.mass).unwrap_or(0.0)
    }

    pub fn velocity(&self) -> (f32, f32) {
        let vx = self.vel_x.last().copied().unwrap_or(0.0);
        let vy = self.vel_y.last().copied().unwrap_or(0.0);
        (vx, vy)
    }

    pub fn blank_frames(&self) -> usize { self.tracker.blank_frames }

    pub fn tracking(&self) -> bool { self.tracker.tracking }

    pub fn frames_since_target_seen(&self) -> usize { self.tracker.frames_since_target_seen }

    pub fn predicted_velocity(&self) -> (f32, f32) { self.tracker.predicted_velocity(self) }

    // ── mutation ────────────────────────────────────────────────────────

    pub fn add_region(&mut self, region: Region) {
        // Insert blank frames for any skipped frame numbers.
        if let (Some(prev), Some(cur)) = (self.prev_frame_num, region.frame_number) {
            let diff = cur.saturating_sub(prev).saturating_sub(1);
            for _ in 0..diff {
                self.add_blank_frame();
            }
        }
        self.tracker.add_region(&region);
        self.bounds_history.push(region);
        self.prev_frame_num = self.bounds_history.last().and_then(|r| r.frame_number);
        self.update_velocity();
    }

    pub fn add_blank_frame(&mut self) {
        let region = self.tracker.add_blank_frame();
        self.bounds_history.push(region);
        self.prev_frame_num = self.bounds_history.last().and_then(|r| r.frame_number);
        self.update_velocity();
    }

    fn update_velocity(&mut self) {
        if self.bounds_history.len() >= 2 {
            let n = self.bounds_history.len();
            let cx1 = self.bounds_history[n - 1].centroid.map(|c| c[0]).unwrap_or(0.0);
            let cy1 = self.bounds_history[n - 1].centroid.map(|c| c[1]).unwrap_or(0.0);
            let cx0 = self.bounds_history[n - 2].centroid.map(|c| c[0]).unwrap_or(0.0);
            let cy0 = self.bounds_history[n - 2].centroid.map(|c| c[1]).unwrap_or(0.0);
            self.vel_x.push(cx1 - cx0);
            self.vel_y.push(cy1 - cy0);
        } else {
            self.vel_x.push(0.0);
            self.vel_y.push(0.0);
        }
    }

    /// Match a list of candidate regions against this track.
    /// Returns `(score, region_index)` pairs for candidate matches.
    pub fn match_regions<'a>(&self, regions: &'a [Region]) -> Vec<(f32, usize)> {
        self.tracker.match_regions(regions, self)
    }

    // ── averaging helpers ───────────────────────────────────────────────

    pub fn average_mass(&self) -> f32 {
        let mut sum = 0.0f32;
        let mut count = 0usize;
        for bound in self.bounds_history.iter().rev() {
            if !bound.blank {
                sum += bound.mass;
                count += 1;
                if count == 5 { break; }
            }
        }
        if count == 0 { 0.0 } else { sum / count as f32 }
    }

    pub fn average_area(&self) -> f32 {
        let mut sum = 0.0f32;
        let mut count = 0usize;
        for bound in self.bounds_history.iter().rev() {
            if !bound.blank {
                sum += bound.area();
                count += 1;
                if count == 5 { break; }
            }
        }
        if count == 0 { 0.0 } else { sum / count as f32 }
    }

    // ── post-processing ─────────────────────────────────────────────────

    /// Remove leading/trailing low-mass frames.
    pub fn trim(&mut self) {
        if self.bounds_history.is_empty() { return; }
        let mass_history: Vec<f32> = self.bounds_history.iter().map(|b| b.mass).collect();
        let median = {
            let mut sorted = mass_history.clone();
            sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
            let mid = sorted.len() / 2;
            if sorted.len() % 2 == 0 {
                (sorted[mid - 1] + sorted[mid]) / 2.0
            } else {
                sorted[mid]
            }
        };
        let filter_mass = (0.005 * median).max(2.0);

        let mut start = 0;
        while start < self.bounds_history.len() && mass_history[start] <= filter_mass {
            start += 1;
        }
        let mut end = self.bounds_history.len().saturating_sub(1);
        while end > 0 && mass_history[end] <= filter_mass {
            if self.tracker.frames_since_target_seen > 0 {
                self.tracker.frames_since_target_seen -= 1;
                self.tracker.blank_frames = self.tracker.blank_frames.saturating_sub(1);
            }
            end -= 1;
        }

        if end < start {
            self.bounds_history.clear();
            self.vel_x.clear();
            self.vel_y.clear();
            self.tracker.blank_frames = 0;
        } else {
            self.start_frame = self.start_frame.map(|s| s + start as u32);
            self.bounds_history = self.bounds_history[start..=end].to_vec();
            self.vel_x = self.vel_x[start..=end].to_vec();
            self.vel_y = self.vel_y[start..=end].to_vec();
        }
        self.start_s = self.start_frame.map(|s| s as f32 / self.fps);
    }

    pub fn calculate_stats(&mut self) {
        use crate::tools::eucl_distance_sq;

        if self.bounds_history.len() <= 1 {
            self.stats = Some(TrackMovementStatistics::default());
            return;
        }

        let non_blank: Vec<&Region> =
            self.bounds_history.iter().filter(|b| !b.blank).collect();
        let mass_history: Vec<f32> = non_blank.iter().map(|b| b.mass).collect();
        let var_history: Vec<f32> = non_blank
            .iter()
            .filter(|b| b.pixel_variance > 0.0)
            .map(|b| b.pixel_variance)
            .collect();

        let mut movement = 0.0f32;
        let mut max_offset = 0.0f32;
        let mut frames_moved = 0i32;
        let mut avg_vel = 0.0f32;

        let first_mid = self.bounds_history[0].mid();

        for (i, (vx, vy)) in self.vel_x.iter().zip(self.vel_y.iter()).enumerate() {
            let region = &self.bounds_history[i];
            if !region.blank { avg_vel += vx.abs() + vy.abs(); }
            if i == 0 { continue; }
            let prev = &self.bounds_history[i - 1];
            if region.blank || prev.blank { continue; }
            if region.has_moved(prev) || region.is_along_border {
                let dist = (vx * vx + vy * vy).sqrt();
                movement += dist;
                let offset = eucl_distance_sq(first_mid, region.mid());
                if offset > max_offset { max_offset = offset; }
                frames_moved += 1;
            }
        }

        avg_vel = if mass_history.is_empty() { 0.0 } else { avg_vel / mass_history.len() as f32 };
        max_offset = max_offset.sqrt();
        let delta_std = if var_history.is_empty() { 0.0 } else {
            let mean = var_history.iter().sum::<f32>() / var_history.len() as f32;
            mean.sqrt()
        };

        // Jitter count.
        const JITTER_THRESHOLD: f32 = 0.3;
        const MIN_JITTER: f32 = 5.0;
        let mut jitter_bigger = 0i32;
        let mut jitter_smaller = 0i32;
        for i in 1..self.bounds_history.len() {
            let prev = &self.bounds_history[i - 1];
            let cur  = &self.bounds_history[i];
            if prev.is_along_border || cur.is_along_border { continue; }
            let h_diff = cur.height - prev.height;
            let w_diff = prev.width  - cur.width;
            let th = (MIN_JITTER).max(prev.height * JITTER_THRESHOLD);
            let tv = (MIN_JITTER).max(prev.width  * JITTER_THRESHOLD);
            if h_diff.abs() > th {
                if h_diff > 0.0 { jitter_bigger += 1; } else { jitter_smaller += 1; }
            } else if w_diff.abs() > tv {
                if w_diff > 0.0 { jitter_bigger += 1; } else { jitter_smaller += 1; }
            }
        }

        let total_frames = self.bounds_history.len() as f32;
        let jitter_pct = ((100.0 * (jitter_bigger + jitter_smaller) as f32) / total_frames) as i32;
        let blank_pct = ((100.0 * self.blank_frames() as f32) / total_frames) as i32;

        let movement_pts = movement.sqrt() + max_offset;
        let delta_pts    = delta_std * 25.0;
        let score = movement_pts.min(100.0)
            + delta_pts.min(100.0)
            + (100 - jitter_pct) as f32
            + (100 - blank_pct) as f32;

        let mean_mass = if mass_history.is_empty() { 0.0 } else {
            mass_history.iter().sum::<f32>() / mass_history.len() as f32
        };
        let median_mass = {
            let mut s = mass_history.clone();
            s.sort_by(|a, b| a.partial_cmp(b).unwrap());
            let mid = s.len() / 2;
            if s.len() % 2 == 0 && s.len() > 1 { (s[mid - 1] + s[mid]) / 2.0 }
            else if s.is_empty() { 0.0 }
            else { s[mid] }
        };
        let mass_std = if mass_history.len() < 2 { 0.0 } else {
            let var = mass_history.iter().map(|m| (m - mean_mass).powi(2)).sum::<f32>()
                / mass_history.len() as f32;
            var.sqrt()
        };

        self.stats = Some(TrackMovementStatistics {
            movement,
            max_offset,
            score,
            average_mass: mean_mass,
            median_mass,
            delta_std,
            region_jitter: jitter_pct,
            jitter_bigger,
            jitter_smaller,
            blank_percent: blank_pct,
            frames_moved,
            mass_std,
            average_velocity: avg_vel,
        });
    }
}

impl std::fmt::Display for Track {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Track: {} frames# {}", self.id, self.bounds_history.len())
    }
}
