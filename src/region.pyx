# cython: language_level=3
# cython: boundscheck=False, wraparound=False
"""
classifier-pipeline - this is a server side component that manipulates cptv
files and to create a classification model of animals present
Copyright (C) 2018, The Cacophony Project

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program. If not, see <http://www.gnu.org/licenses/>.
"""

from rectangle cimport Rectangle


cdef class Region(Rectangle):
    """Region is a rectangle extended to support mass."""


    def __init__(self, x, y, width=0, height=0, centroid=None, mass=0,
                 frame_number=0, pixel_variance=0, id=0,
                 was_cropped=False, blank=False, is_along_border=False,
                 in_trap=False):
        Rectangle.__init__(self, x, y, width, height)
        self.centroid = centroid
        self.mass = mass
        self.frame_number = frame_number
        self.pixel_variance = pixel_variance
        self.id = id
        self.was_cropped = was_cropped
        self.blank = blank
        self.is_along_border = is_along_border
        self.in_trap = in_trap

    def __reduce__(self):
        return (
            self.__class__,
            (self.x, self.y, self.width, self.height, self.centroid, self.mass,
             self.frame_number, self.pixel_variance, self.id, self.was_cropped,
             self.blank, self.is_along_border, self.in_trap),
        )

    def rescale(self, factor):
        cdef double f = factor
        self.x = int(self.x * f)
        self.y = int(self.y * f)
        self.width = int(self.width * f)
        self.height = int(self.height * f)
        self.mass = self.mass * (f * f)

    @staticmethod
    def from_ltwh(left, top, width, height):
        """Construct a Region from left, top, width, height."""
        return Region(left, top, width=width, height=height, centroid=None)

    @staticmethod
    def from_ltrb(left, top, right, bottom):
        """Construct a Region from left, top, right, bottom co-ords."""
        return Region(left, top, width=right - left, height=bottom - top, centroid=None)

    def to_array(self):
        import numpy as np
        return np.uint16(
            [
                self.left,
                self.top,
                self.right,
                self.bottom,
                self.frame_number if self.frame_number is not None else 0,
                self.mass,
                1 if self.blank else 0,
            ]
        )

    @classmethod
    def region_from_array(cls, region_bounds):
        import numpy as np

        cdef int width = int(region_bounds[2]) - int(region_bounds[0])
        cdef int height = int(region_bounds[3]) - int(region_bounds[1])
        height = max(height, 0)
        width = max(width, 0)
        frame_number = None
        if len(region_bounds) > 4:
            frame_number = region_bounds[4]
        mass = 0
        if len(region_bounds) > 5:
            mass = region_bounds[5]
        blank = False
        if len(region_bounds) > 6:
            blank = region_bounds[6] == 1
        centroid = [
            int(region_bounds[0] + width / 2),
            int(region_bounds[1] + height / 2),
        ]
        return cls(
            region_bounds[0],
            region_bounds[1],
            width,
            height,
            centroid=centroid,
            frame_number=int(frame_number) if frame_number is not None else None,
            mass=mass,
            blank=blank,
        )

    @classmethod
    def region_from_json(cls, region_json):
        frame = region_json.get("frame_number")
        if frame is None:
            frame = region_json.get("frameNumber")
        if frame is None:
            frame = region_json.get("order")
        if "centroid" in region_json:
            centroid = region_json["centroid"]
        else:
            centroid = [
                int(region_json["x"] + region_json["width"] / 2),
                int(region_json["y"] + region_json["height"] / 2),
            ]
        mass = region_json.get("mass", 0)
        if mass is None:
            mass = 0
        return cls(
            region_json["x"],
            region_json["y"],
            region_json["width"],
            region_json["height"],
            centroid=centroid,
            frame_number=frame,
            mass=mass,
            blank=region_json.get("blank", False),
            pixel_variance=region_json.get("pixel_variance", 0),
        )

    def has_moved(self, other):
        """Determines if the region has shifted horizontally or vertically."""
        return (self.x != other.x and self.right != other.right) or (
            self.y != other.y and self.bottom != other.bottom
        )

    def calculate_variance(self, filtered, prev_filtered):
        """Calculates variance on this frame for this region."""
        from ml_tools.tools import calculate_variance

        height, width = filtered.shape
        assert (
            width == self.width and height == self.height
        ), "calculating variance on incorrectly sized filtered"
        self.pixel_variance = calculate_variance(filtered, prev_filtered)

    def set_is_along_border(self, bounds, edge=0):
        self.is_along_border = (
            self.was_cropped
            or self.x <= bounds.x + edge
            or self.y <= bounds.y + edge
            or self.right >= bounds.width - edge
            or self.bottom >= bounds.height - edge
        )

    def copy(self):
        return Region(
            self.x,
            self.y,
            self.width,
            self.height,
            self.centroid,
            self.mass,
            self.frame_number,
            self.pixel_variance,
            self.id,
            self.was_cropped,
            self.blank,
            self.is_along_border,
            self.in_trap,
        )

    def average_distance(self, other):
        """Calculates the distance between 2 regions using top-left, mid, and bottom-right."""
        from tools import eucl_distance_sq

        cdef int ex, ey
        distances = []

        ex = int(other.x)
        ey = int(other.y)
        distances.append(eucl_distance_sq((ex, ey), (self.x, self.y)))

        ex = int(other.mid_x)
        ey = int(other.mid_y)
        distances.append(eucl_distance_sq((ex, ey), (self.mid_x, self.mid_y)))

        distances.append(
            eucl_distance_sq(
                (other.right, other.bottom),
                (self.right, self.bottom),
            )
        )

        return distances

    def on_height_edge(self, crop_region):
        return self.top == crop_region.top or self.bottom == crop_region.bottom

    def on_width_edge(self, crop_region):
        return self.left == crop_region.left or self.right == crop_region.right

    def calculate_mass(self, filtered, threshold):
        """Calculates mass on this frame for this region."""
        height, width = filtered.shape
        assert (
            width == self.width and height == self.height
        ), "calculating mass on incorrectly sized filtered"
        self.mass = calculate_mass(filtered, threshold)

    def meta_dictionary(self):
        """Return region metadata as a dict, excluding internal tracking fields."""
        cdef double pv = self.pixel_variance
        return {
            "x": self.x,
            "y": self.y,
            "width": self.width,
            "height": self.height,
            "mass": self.mass,
            "frame_number": self.frame_number,
            "pixel_variance": round(pv, 2) if pv else 0,
            "blank": self.blank,
            "in_trap": self.in_trap,
        }


cpdef unsigned int calculate_mass(filtered, int threshold):
    """Calculates mass of filtered frame with threshold applied."""
    import numpy as np

    if filtered.size == 0:
        return 0
    _, mass = _blur_and_mask(filtered, threshold)
    return int(mass)


cdef _blur_and_mask(frame, int threshold):
    """Creates a binary mask by applying Gaussian blur and threshold."""
    import cv2

    thresh = cv2.GaussianBlur(frame, (5, 5), 0)
    thresh[thresh - threshold < 0] = 0
    mass = len(thresh[thresh > 0])
    return thresh, mass
