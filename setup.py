# -*- coding: utf-8 -*-

# Copyright 2019, Grid.
#
# This source code is licensed under the ReNom Subscription Agreement, version 1.0.
# ReNom Subscription Agreement Ver. 1.0 (https://www.renom.jp/info/license/index.html)


import os
import sys
import re
import shutil
import pathlib
import numpy
from setuptools import setup, find_packages, Extension
import distutils.command.build
from Cython.Build import cythonize
from distutils.extension import Extension


if sys.version_info < (3, 4):
    raise RuntimeError('renom_q requires Python3')

DIR = str(pathlib.Path(__file__).resolve().parent)

setup(
    name="renom_q",
    version="0.2b1",
    packages=['renom_q'],
    include_package_data=True,
    zip_safe=True,
    install_requires=["qiskit"],
)
