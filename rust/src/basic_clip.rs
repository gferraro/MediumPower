use std::collections::HashSet;
use opencv::core::Mat;
use crate::frame::ProcessedFrame;
use crate::frame_buffer::FrameBuffer;
use crate::rectangle::Rectangle;

/// Lightweight clip state used during live tracking.
/// Mirrors the Python `BasicClip` class in mediumpower.py.
pub struct BasicClip {
    pub res_x: u32,
    pub res_y: u32,
    /// IDs of tracks currently being followed.
    pub active_track_ids: HashSet<u32>,
    /// All track IDs ever seen in this clip.
    pub track_ids: Vec<u32>,
    /// Frame counter (incremented by `add_frame`).
    pub current_frame: i32,
    pub ffc_affected: bool,
    /// Frame numbers where FFC was active.
    pub ffc_frames: Vec<i32>,
    pub frame_buffer: FrameBuffer,
    pub region_history: Vec<Vec<crate::region::Region>>,
    pub crop_rectangle: Option<Rectangle>,
    pub frames_per_second: f32,
}

impl BasicClip {
    pub fn new() -> Self {
        Self {
            res_x: 160,
            res_y: 120,
            active_track_ids: HashSet::new(),
            track_ids: Vec::new(),
            current_frame: -1,
            ffc_affected: false,
            ffc_frames: Vec::new(),
            frame_buffer: FrameBuffer::new(true, 50),
            region_history: Vec::new(),
            crop_rectangle: None,
            frames_per_second: 9.0,
        }
    }

    /// Always returns 1 — this is a single-clip wrapper.
    pub fn get_id(&self) -> u32 { 1 }

    pub fn add_frame(
        &mut self,
        thermal: Mat,
        filtered: Mat,
        _mask: Option<Mat>,
        ffc_affected: bool,
    ) -> ProcessedFrame {
        self.current_frame += 1;
        if ffc_affected {
            self.ffc_frames.push(self.current_frame);
        }
        self.frame_buffer.add_frame(
            thermal,
            filtered,
            self.current_frame as u32,
            ffc_affected,
        )
    }

    pub fn get_frame(&self, frame_number: u32) -> Option<&ProcessedFrame> {
        self.frame_buffer.get_frame(frame_number)
    }

    pub fn add_active_track_id(&mut self, id: u32) {
        self.active_track_ids.insert(id);
        self.track_ids.push(id);
    }
}
