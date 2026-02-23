# cython: language_level=3
# cython: boundscheck=False, wraparound=False


cdef class Rectangle:
    """Defines a rectangle by the topleft point and width / height."""


    def __init__(self, x, y, width=0, height=0):
        self.x = int(x)
        self.y = int(y)
        self.width = int(width)
        self.height = int(height)

    def __reduce__(self):
        return (self.__class__, (self.x, self.y, self.width, self.height))

    @staticmethod
    def from_ltrb(left, top, right, bottom):
        """Construct a rectangle from left, top, right, bottom co-ords."""
        return Rectangle(left, top, width=right - left, height=bottom - top)

    def to_ltrb(self):
        """Return rectangle as left, top, right, bottom co-ords."""
        return [self.left, self.top, self.right, self.bottom]

    def to_ltwh(self):
        """Return rectangle as left, top, width, height."""
        return [self.left, self.top, self.width, self.height]

    def copy(self):
        return Rectangle(self.x, self.y, self.width, self.height)

    @property
    def elongation(self):
        return max(self.width, self.height) / min(self.width, self.height)

    @property
    def mid(self):
        return (self.mid_x, self.mid_y)

    @property
    def mid_x(self):
        return self.x + self.width / 2.0

    @property
    def mid_y(self):
        return self.y + self.height / 2.0

    @property
    def left(self):
        return self.x

    @property
    def top(self):
        return self.y

    @property
    def right(self):
        return self.x + self.width

    @property
    def bottom(self):
        return self.y + self.height

    @left.setter
    def left(self, value):
        cdef int old_right = self.x + self.width
        self.x = int(value)
        self.width = old_right - self.x

    @top.setter
    def top(self, value):
        cdef int old_bottom = self.y + self.height
        self.y = int(value)
        self.height = old_bottom - self.y

    @right.setter
    def right(self, value):
        self.width = int(value) - self.x

    @bottom.setter
    def bottom(self, value):
        self.height = int(value) - self.y

    def overlap_area(self, Rectangle other):
        """Compute the area overlap between this rectangle and another."""
        cdef int x_overlap = max(0, min(self.x + self.width, other.x + other.width) - max(self.x, other.x))
        cdef int y_overlap = max(0, min(self.y + self.height, other.y + other.height) - max(self.y, other.y))
        return x_overlap * y_overlap

    def crop(self, bounds):
        """Crops this rectangle so that it fits within given bounds."""
        self.left = min(bounds.right, max(self.left, bounds.left))
        self.top = min(bounds.bottom, max(self.top, bounds.top))
        self.right = max(bounds.left, min(self.right, bounds.right))
        self.bottom = max(bounds.top, min(self.bottom, bounds.bottom))

    def subimage(self, image):
        """Returns a subsection of the original image bounded by this rectangle."""
        cdef int t = self.y
        cdef int h = self.height
        cdef int l = self.x
        cdef int w = self.width
        return image[t:t + h, l:l + w]

    def enlarge_even(self, int width_enlarge, int height_enlarge, crop):
        cdef int left_adjust, right_adjust, width_adjust
        cdef int top_adjust, bottom_adjust, height_adjust

        self.left -= width_enlarge
        self.right += width_enlarge
        self.top -= height_enlarge
        self.bottom += height_enlarge

        left_adjust = crop.left - self.left
        left_adjust = max(0, left_adjust)
        left_adjust = min(left_adjust, crop.width)

        right_adjust = self.right - crop.right
        right_adjust = max(0, right_adjust)
        right_adjust = min(right_adjust, crop.width)
        width_adjust = max(left_adjust, right_adjust)

        self.left += width_adjust
        self.right -= width_adjust

        bottom_adjust = self.bottom - crop.bottom
        bottom_adjust = max(0, bottom_adjust)
        bottom_adjust = min(bottom_adjust, crop.height)

        top_adjust = crop.top - self.top
        top_adjust = max(0, top_adjust)
        top_adjust = min(top_adjust, crop.height)

        height_adjust = max(bottom_adjust, top_adjust)
        self.top += height_adjust
        self.bottom -= height_adjust

    def enlarge(self, border, max=None):
        """Enlarges this by border amount in each dimension, optionally clamped to max."""
        self.left -= border
        self.right += border
        self.top -= border
        self.bottom += border
        if max:
            self.crop(max)

    def contains(self, x, y):
        """Is this point contained in the rectangle."""
        return self.left <= x and self.right >= x and self.top >= y and self.bottom <= y

    @property
    def area(self):
        return self.width * self.height

    def __repr__(self):
        return "(x{0},y{1},x2{2},y2{3})".format(
            self.left, self.top, self.right, self.bottom
        )

    def __str__(self):
        return "<(x{0},y{1})-h{2}xw{3}>".format(self.x, self.y, self.height, self.width)

    def enlarge_for_rotation(self, crop_rectangle, final_dim=32, extra_needed=13):
        import numpy as np
        import math

        scale_percent = (final_dim / np.array([self.width, self.height])).min()

        extra_pixels = extra_needed / scale_percent
        height_enlarge = math.ceil(extra_pixels / 2)
        width_enlarge = math.ceil(extra_pixels / 2)

        adjusted_height = self.height + extra_pixels
        adjusted_width = self.width + extra_pixels
        if self.width > self.height:
            diff = adjusted_width - adjusted_height
            height_enlarge = math.ceil((extra_pixels + diff) / 2)
        else:
            diff = adjusted_height - adjusted_width
            width_enlarge = math.ceil((extra_pixels + diff) / 2)

        self.enlarge_even(width_enlarge, height_enlarge, crop=crop_rectangle)
