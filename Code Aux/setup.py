from distutils.core import setup
from Cython.Build import cythonize

setup(
  name = 'MyProject',
  ext_modules = cythonize("cumsum.pyx",["cumsum.pyx"]),
)