# Cython declaration file for rectangle.pyx
# Required so other .pyx files can cimport Rectangle as a cdef class

cdef class Rectangle:
    cdef public int x, y, width, height
