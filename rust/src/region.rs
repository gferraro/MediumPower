use std::ops::{Deref, DerefMut};
use opencv::core::Mat;
use opencv::prelude::*;
use crate::rectangle::Rectangle;
use crate::tools::eucl_distance_sq;

/// Region extends Rectangle with per-region tracking metadata such as mass,
/// frame number, and border flags.  Mirrors the Python `Region` class in region.py.
#[derive(Debug, Clone)]
pub struct Region {
    pub rect: Rectangle,
    pub centroid: Option<[f32; 2]>,
    pub mass: f32,
    pub frame_number: Option<u32>,
    pub pixel_variance: f32,
    pub id: u32,
    pub was_cropped: bool,
    pub blank: bool,
    pub is_along_border: bool,
    pub in_trap: bool,
}

// Transparent delegation to the inner Rectangle.
impl Deref for Region {
    type Target = Rectangle;
    fn deref(&self) -> &Rectangle { &self.rect }
}
impl DerefMut for Region {
    fn deref_mut(&mut self) -> &mut Rectangle { &mut self.rect }
}

impl Region {
    pub fn new(x: f32, y: f32, width: f32, height: f32, centroid: Option<[f32; 2]>) -> Self {
        Self {
            rect: Rectangle::new(x, y, width, height),
            centroid,
            mass: 0.0,
            frame_number: None,
            pixel_variance: 0.0,
            id: 0,
            was_cropped: false,
            blank: false,
            is_along_border: false,
            in_trap: false,
        }
    }

    pub fn from_ltwh(left: f32, top: f32, width: f32, height: f32) -> Self {
        Self::new(left, top, width, height, None)
    }

    pub fn from_ltrb(left: f32, top: f32, right: f32, bottom: f32) -> Self {
        Self::new(left, top, right - left, bottom - top, None)
    }

    pub fn copy(&self) -> Self { self.clone() }

    /// Scale position, size, and mass by `factor`.
    pub fn rescale(&mut self, factor: f32) {
        self.rect.x      = (self.rect.x * factor) as f32;
        self.rect.y      = (self.rect.y * factor) as f32;
        self.rect.width  = (self.rect.width * factor) as f32;
        self.rect.height = (self.rect.height * factor) as f32;
        self.mass        = self.mass * (factor * factor);
    }

    /// Returns `true` if the region has shifted (not just grown/shrunk) vs `other`.
    pub fn has_moved(&self, other: &Region) -> bool {
        (self.x != other.x && self.right() != other.right())
            || (self.y != other.y && self.bottom() != other.bottom())
    }

    /// Set `is_along_border` based on proximity to `bounds`.
    pub fn set_is_along_border(&mut self, bounds: &Rectangle, edge: f32) {
        self.is_along_border = self.was_cropped
            || self.x       <= bounds.x + edge
            || self.y       <= bounds.y + edge
            || self.right()  >= bounds.right() - edge
            || self.bottom() >= bounds.bottom() - edge;
    }

    /// Compute and store `mass` from a filtered sub-image.
    pub fn calculate_mass(&mut self, filtered: &Mat, threshold: f32) -> anyhow::Result<()> {
        self.mass = calculate_mass(filtered, threshold)?;
        Ok(())
    }

    pub fn on_height_edge(&self, crop: &Rectangle) -> bool {
        self.top() == crop.top() || self.bottom() == crop.bottom()
    }

    pub fn on_width_edge(&self, crop: &Rectangle) -> bool {
        self.left() == crop.left() || self.right() == crop.right()
    }

    /// Distances between corner / midpoint / corner of this region and `other`.
    /// Returns [top-left dist², midpoint dist², bottom-right dist²].
    pub fn average_distance(&self, other: &Region) -> [f32; 3] {
        [
            eucl_distance_sq((other.x, other.y), (self.x, self.y)),
            eucl_distance_sq(
                (other.mid_x(), other.mid_y()),
                (self.mid_x(), self.mid_y()),
            ),
            eucl_distance_sq(
                (other.right(), other.bottom()),
                (self.right(), self.bottom()),
            ),
        ]
    }
}

/// Compute the mass of a filtered image region by blurring and thresholding.
pub fn calculate_mass(filtered: &Mat, threshold: f32) -> anyhow::Result<f32> {
    use crate::tools::blur_and_return_as_mask;
    if filtered.empty() {
        return Ok(0.0);
    }
    let (_, mass) = blur_and_return_as_mask(filtered, threshold)?;
    Ok(mass as f32)
}

impl std::fmt::Display for Region {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.rect)
    }
}
