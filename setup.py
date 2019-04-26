#!/usr/bin/env python

from distutils.core import setup
from setuptools import find_packages

setup(
    name='object_pose_utils',
    version='0.1dev',
    author='Brian Okorn',
    packages=find_packages('src'),
    package_dir={'': 'src'},
    description='Utility functions and classes for object pose estimation',
    long_description=open('README.md').read(),
    package_data = {'': ['*.mat', '*.npy']},
)

