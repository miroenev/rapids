#!/usr/bin/env python

from distutils.core import setup

setup(name='rapids-dask-hpo',
      version='0.0.0',
      description='Use RAPIDS and Dask for HPO',
      py_modules=['swarm', 'data_utils'] 
     )
