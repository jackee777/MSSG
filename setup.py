from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy as np

ext_modules = [
    Extension('ms2vec_inner',
              sources=[
                  'ms2vec/ms2vec_inner.pyx'],
              include_dirs=[np.get_include()],
              language='c++',
#               compiler_directives={'language_level': "3"}
              ),
    Extension('ms2vec_corpusfile',
              sources=[
                  'ms2vec/ms2vec_corpusfile.pyx'],
              include_dirs=[np.get_include()],
              language='c++'
              )
]


setup(
    name="ms2vec",
    version="0.2.0",
    ext_modules=cythonize(ext_modules),#, language_level = 3),
    include_dirs=[np.get_include()],
)
