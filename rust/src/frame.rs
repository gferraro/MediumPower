use opencv::core::{Mat, CV_32F};
use opencv::prelude::*;
use crate::rectangle::Rectangle;
use crate::region::Region;


/// A processed frame containing the raw thermal channel and the
/// background-subtracted filtered channel.
/// Mirrors the Python `Frame` class defined in mediumpower.py.
#[derive(Debug, Clone)]
pub struct ProcessedFrame {
    /// Raw thermal data (CV_32F), shape (res_y, res_x).
    pub thermal: Mat,
    /// Background-subtracted filtered data (CV_32F), shape (res_y, res_x).
    pub filtered: Mat,
    pub frame_number: u32,
    pub ffc_affected: bool,
    /// Region associated with this frame (set for cropped frames).
    pub region: Option<Region>,
}

impl ProcessedFrame {
    pub fn new(
        thermal: Mat,
        filtered: Mat,
        frame_number: u32,
        ffc_affected: bool,
    ) -> Self {
        Self { thermal, filtered, frame_number, ffc_affected, region: None }
    }

    /// Return a new frame cropped to `region`, with float32 channels.
    pub fn crop_by_region(&self, region: &Region) -> anyhow::Result<Self> {
        let thermal_roi = region.subimage(&self.thermal)?;
        let filtered_roi = region.subimage(&self.filtered)?;

        let mut thermal_f32 = Mat::default();
        let mut filtered_f32 = Mat::default();
        thermal_roi.convert_to(&mut thermal_f32, CV_32F, 1.0, 0.0)?;
        filtered_roi.convert_to(&mut filtered_f32, CV_32F, 1.0, 0.0)?;

        Ok(Self {
            thermal: thermal_f32,
            filtered: filtered_f32,
            frame_number: self.frame_number,
            ffc_affected: self.ffc_affected,
            region: Some(region.clone()),
        })
    }

    /// Resize `thermal` and `filtered` channels to `dim` (height, width),
    /// preserving aspect ratio and padding with the channel minimum value.
    pub fn resize_with_aspect(
        &mut self,
        dim: (i32, i32),
        crop_rectangle: Option<&Rectangle>,
        keep_edge: bool,
        edge_offset: (i32, i32, i32, i32),
        original_region: Option<&Region>,
    ) -> anyhow::Result<()> {
        use crate::tools::resize_and_pad;
        let region_ref = self.region.as_ref();
        let orig_ref   = original_region.or(region_ref);

        self.thermal = resize_and_pad(
            &self.thermal,
            dim,
            region_ref,
            crop_rectangle,
            keep_edge,
            None,
            edge_offset,
            orig_ref,
        )?;

        self.filtered = resize_and_pad(
            &self.filtered,
            dim,
            region_ref,
            crop_rectangle,
            keep_edge,
            Some(0.0),
            edge_offset,
            orig_ref,
        )?;

        Ok(())
    }

    pub fn get_channel(&self, channel: usize) -> &Mat {
        if channel == 0 { &self.thermal } else { &self.filtered }
    }
}
