#!/usr/bin/env python
# -*- coding: utf-8 -*-

import codecs
import os

from setuptools import setup, find_packages

def read(fname):
    file_path = os.path.join(os.path.dirname(__file__), fname)
    return codecs.open(file_path, encoding="utf-8").read()

setup(
    name='Jay_pdbtools',
    version='0.0.1',
    author='Jay Unruh',
    description='Notebooks and scripts to analyze molecular structure pdb and mmcif files.',
    url='https://github.com/jayunruh/Jay_pdbtools',
    license='GNU GPLv2',
    packages=find_packages(),
    python_requires='>=3.6',
    install_requires=["matplotlib","numpy","pandas","scipy","plotly","py3Dmol","biopython"],
    py_modules=['jpdbtools2','jpdbtools','calc_mlp_jru','pore_utils']
)
