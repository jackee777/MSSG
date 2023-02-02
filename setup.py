from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy as np

ext_modules = [
    Extension('mssg_inner',
              sources=[
                  'mssg/mssg_inner.pyx'],
              include_dirs=[np.get_include()],
              language='c++',
#               compiler_directives={'language_level': "3"}
              ),
    Extension('mssg_corpusfile',
              sources=[
                  'mssg/mssg_corpusfile.pyx'],
              include_dirs=[np.get_include()],
              language='c++'
              )
]


setup(
    name="mssg",
    version="0.2.0",
    ext_modules=cythonize(ext_modules),#, language_level = 3),
    include_dirs=[np.get_include()],
)
