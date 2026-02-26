use std::collections::HashMap;
use std::collections::VecDeque;
use crate::frame::ProcessedFrame;
use opencv::core::Mat;

/// Stores the most recent `max_frames` processed frames.
/// Mirrors the Python `FrameBuffer` class in mediumpower.py.
pub struct FrameBuffer {
    /// Frames in order of insertion.
    frames: VecDeque<ProcessedFrame>,
    /// Maps frame_number → index into `frames` for O(1) lookup.
    frames_by_frame_number: HashMap<u32, usize>,
    /// Most recently added frame (kept even if evicted from buffer).
    pub current_frame: Option<ProcessedFrame>,
    /// Frame before `current_frame`.
    pub prev_frame: Option<ProcessedFrame>,
    pub max_frames: usize,
    pub keep_frames: bool,
}

impl FrameBuffer {
    pub fn new(keep_frames: bool, max_frames: usize) -> Self {
        let keep = if max_frames > 0 { true } else { keep_frames };
        Self {
            frames: VecDeque::new(),
            frames_by_frame_number: HashMap::new(),
            current_frame: None,
            prev_frame: None,
            max_frames,
            keep_frames: keep,
        }
    }

    /// Add a new processed frame; returns a clone for the caller to keep.
    pub fn add_frame(
        &mut self,
        thermal: Mat,
        filtered: Mat,
        frame_number: u32,
        ffc_affected: bool,
    ) -> ProcessedFrame {
        self.prev_frame = self.current_frame.take();
        let frame = ProcessedFrame::new(thermal, filtered, frame_number, ffc_affected);
        self.current_frame = Some(frame.clone());

        // Evict oldest if at capacity.
        if self.frames.len() == self.max_frames {
            if let Some(evicted) = self.frames.pop_front() {
                self.frames_by_frame_number.remove(&evicted.frame_number);
            }
        }

        let idx = self.frames.len();
        self.frames_by_frame_number.insert(frame_number, idx);
        self.frames.push_back(frame.clone());

        frame
    }

    pub fn get_frame(&self, frame_number: u32) -> Option<&ProcessedFrame> {
        if let Some(&idx) = self.frames_by_frame_number.get(&frame_number) {
            // VecDeque indices shift when we pop_front; search linearly after eviction.
            // For small buffers (max 50 frames) this is acceptable.
            self.frames.iter().find(|f| f.frame_number == frame_number)
        } else if self
            .prev_frame
            .as_ref()
            .map(|f| f.frame_number == frame_number)
            .unwrap_or(false)
        {
            self.prev_frame.as_ref()
        } else if self
            .current_frame
            .as_ref()
            .map(|f| f.frame_number == frame_number)
            .unwrap_or(false)
        {
            self.current_frame.as_ref()
        } else {
            None
        }
    }

    pub fn reset(&mut self) {
        self.frames.clear();
        self.frames_by_frame_number.clear();
    }
}
