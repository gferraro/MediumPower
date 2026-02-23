# Cython declaration file for region.pyx

from rectangle cimport Rectangle

cdef class Region(Rectangle):
    cdef public object centroid
    cdef public double mass
    cdef public object frame_number
    cdef public double pixel_variance
    cdef public int id
    cdef public bint was_cropped
    cdef public bint blank
    cdef public bint is_along_border
    cdef public bint in_trap

cpdef unsigned int calculate_mass(filtered, int threshold)
