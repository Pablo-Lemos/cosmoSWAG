#!/usr/bin/env python

from setuptools import setup, find_packages

setup(name='cosmoswag',
      version='v0.1.2',
      description='Stochastic Weight Averaging for cosmology',
      author='Pablo Lemos, Miles Cranmer, Shai Slav, Muntazir Abidi, Shirley Ho',
      url='https://github.com/Pablo-Lemos/cosmoSWAG',
      packages=find_packages(),
      install_requires=[
          "numpy"
          "torch",
      ])
