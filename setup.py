from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy as np

src_dir = "src"

extensions = [
    Extension(
        "rectangle",
        [f"{src_dir}/rectangle.pyx"],
    ),
    Extension(
        "region",
        [f"{src_dir}/region.pyx"],
    ),
    Extension(
        "tools",
        [f"{src_dir}/tools.pyx"],
        include_dirs=[np.get_include()],
    ),
    Extension(
        "kalman",
        [f"{src_dir}/kalman.pyx"],
    ),
    Extension(
        "tracker",
        [f"{src_dir}/tracker.pyx"],
    ),
    Extension(
        "track",
        [f"{src_dir}/track.pyx"],
        include_dirs=[np.get_include()],
    ),
    Extension(
        "motiondetector",
        [f"{src_dir}/motiondetector.pyx"],
        include_dirs=[np.get_include()],
    ),
    Extension(
        "cliptrackextractor",
        [f"{src_dir}/cliptrackextractor.pyx"],
        include_dirs=[np.get_include()],
    ),
    Extension(
        "mediumpower",
        [f"{src_dir}/mediumpower.pyx"],
        include_dirs=[np.get_include()],
    ),
]

setup(
    name="medium-power-tracking",
    ext_modules=cythonize(
        extensions,
        compiler_directives={
            "language_level": "3",
            "boundscheck": False,
            "wraparound": False,
        },
        build_dir="build",
    ),
)
