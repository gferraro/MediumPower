use opencv::core::{Mat, Rect};
use opencv::prelude::*;

/// A rectangle defined by a top-left corner, width, and height.
/// Mirrors the Python `Rectangle` class in rectangle.py.
#[derive(Debug, Clone, PartialEq)]
pub struct Rectangle {
    pub x: f32,
    pub y: f32,
    pub width: f32,
    pub height: f32,
}

impl Rectangle {
    pub fn new(x: f32, y: f32, width: f32, height: f32) -> Self {
        Self { x, y, width, height }
    }

    /// Construct from left, top, right, bottom co-ordinates.
    pub fn from_ltrb(left: f32, top: f32, right: f32, bottom: f32) -> Self {
        Self {
            x: left,
            y: top,
            width: right - left,
            height: bottom - top,
        }
    }

    pub fn to_ltrb(&self) -> [f32; 4] {
        [self.left(), self.top(), self.right(), self.bottom()]
    }

    pub fn to_ltwh(&self) -> [f32; 4] {
        [self.left(), self.top(), self.width, self.height]
    }

    pub fn copy(&self) -> Self {
        self.clone()
    }

    // ── computed edge properties ───────────────────────────────────────────

    #[inline]
    pub fn left(&self) -> f32 { self.x }
    #[inline]
    pub fn top(&self) -> f32 { self.y }
    #[inline]
    pub fn right(&self) -> f32 { self.x + self.width }
    #[inline]
    pub fn bottom(&self) -> f32 { self.y + self.height }
    #[inline]
    pub fn mid_x(&self) -> f32 { self.x + self.width / 2.0 }
    #[inline]
    pub fn mid_y(&self) -> f32 { self.y + self.height / 2.0 }
    #[inline]
    pub fn mid(&self) -> (f32, f32) { (self.mid_x(), self.mid_y()) }
    #[inline]
    pub fn area(&self) -> f32 { self.width * self.height }

    pub fn elongation(&self) -> f32 {
        self.width.max(self.height) / self.width.min(self.height)
    }

    // ── edge setters (adjust width/height to keep the opposite edge fixed) ─

    pub fn set_left(&mut self, value: f32) {
        let old_right = self.right();
        self.x = value;
        self.width = old_right - value;
    }

    pub fn set_top(&mut self, value: f32) {
        let old_bottom = self.bottom();
        self.y = value;
        self.height = old_bottom - value;
    }

    pub fn set_right(&mut self, value: f32) {
        self.width = value - self.x;
    }

    pub fn set_bottom(&mut self, value: f32) {
        self.height = value - self.y;
    }

    // ── geometry operations ───────────────────────────────────────────────

    /// Area of intersection with `other`.
    pub fn overlap_area(&self, other: &Rectangle) -> f32 {
        let x_overlap =
            (self.right().min(other.right()) - self.left().max(other.left())).max(0.0);
        let y_overlap =
            (self.bottom().min(other.bottom()) - self.top().max(other.top())).max(0.0);
        x_overlap * y_overlap
    }

    /// Clamp this rectangle to lie within `bounds`.
    pub fn crop(&mut self, bounds: &Rectangle) {
        let new_left  = self.left().max(bounds.left()).min(bounds.right());
        let new_top   = self.top().max(bounds.top()).min(bounds.bottom());
        let new_right = self.right().min(bounds.right()).max(bounds.left());
        let new_bot   = self.bottom().min(bounds.bottom()).max(bounds.top());
        self.x      = new_left;
        self.y      = new_top;
        self.width  = new_right - new_left;
        self.height = new_bot   - new_top;
    }

    /// Expand all edges by `border`; optionally clamp to `max`.
    pub fn enlarge(&mut self, border: f32, max: Option<&Rectangle>) {
        self.set_left(self.left() - border);
        self.set_right(self.right() + border);
        self.set_top(self.top() - border);
        self.set_bottom(self.bottom() + border);
        if let Some(bounds) = max {
            self.crop(bounds);
        }
    }

    /// Expand width and height evenly while keeping the result within `crop`.
    pub fn enlarge_even(&mut self, width_enlarge: f32, height_enlarge: f32, crop: &Rectangle) {
        self.set_left(self.left() - width_enlarge);
        self.set_right(self.right() + width_enlarge);
        self.set_top(self.top() - height_enlarge);
        self.set_bottom(self.bottom() + height_enlarge);

        let left_adj  = (crop.left() - self.left()).max(0.0).min(crop.width);
        let right_adj = (self.right() - crop.right()).max(0.0).min(crop.width);
        let w_adj     = left_adj.max(right_adj);
        self.set_left(self.left() + w_adj);
        self.set_right(self.right() - w_adj);

        let bot_adj = (self.bottom() - crop.bottom()).max(0.0).min(crop.height);
        let top_adj = (crop.top() - self.top()).max(0.0).min(crop.height);
        let h_adj   = bot_adj.max(top_adj);
        self.set_top(self.top() + h_adj);
        self.set_bottom(self.bottom() - h_adj);
    }

    pub fn contains(&self, x: f32, y: f32) -> bool {
        self.left() <= x && self.right() >= x && self.top() >= y && self.bottom() <= y
    }

    // ── OpenCV Mat slicing ────────────────────────────────────────────────

    /// Returns an owned `Mat` copy of the ROI of `image` bounded by this rectangle.
    pub fn subimage(&self, image: &Mat) -> anyhow::Result<Mat> {
        let rect = Rect::new(
            self.left() as i32,
            self.top() as i32,
            self.width as i32,
            self.height as i32,
        );
        let roi = Mat::roi(image, rect)?;
        let mut owned = Mat::default();
        roi.copy_to(&mut owned)?;
        Ok(owned)
    }

    /// Convert to an OpenCV `Rect`.
    pub fn to_cv_rect(&self) -> Rect {
        Rect::new(
            self.left() as i32,
            self.top() as i32,
            self.width as i32,
            self.height as i32,
        )
    }
}

impl std::fmt::Display for Rectangle {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "(x{},y{},x2{},y2{})",
            self.left(),
            self.top(),
            self.right(),
            self.bottom()
        )
    }
}
