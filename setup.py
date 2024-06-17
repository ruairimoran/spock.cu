#!/usr/bin/env python

from setuptools import setup, find_packages
import io
import os

here = os.path.abspath(os.path.dirname(__file__))

NAME = 'raocp'

# Import version from file
version_file = open(os.path.join(here, 'VERSION'))
VERSION = version_file.read().strip()

DESCRIPTION = 'Solver for multistage risk-averse optimal control problems'


# Import the README and use it as the long-description.
# Note: this will only work if 'README.md' is present in your MANIFEST.in file!
try:
    with io.open(os.path.join(here, 'README.md'), encoding='utf-8') as f:
        long_description = '\n' + f.read()
except FileNotFoundError:
    long_description = DESCRIPTION

setup(name=NAME,
      version=VERSION,
      description=DESCRIPTION,
      long_description=long_description,
      long_description_content_type='text/markdown',
      author=['Ruairi Moran', 'Pantelis Sopasakis'],
      author_email=['rmoran05@qub.ac.uk', 'p.sopasakis@qub.ac.uk'],
      license='Apache 2.0',
      packages=find_packages(
          exclude=["tests"]),
      include_package_data=True,
      install_requires=[
          'numpy==1.26.4', 'scipy', 'matplotlib', 'jinja2', 'PythonTurtle', 'cvxpy'
      ],

      classifiers=[
          'Programming Languages :: Python, C++, CUDA'
      ],
      keywords=['risk averse, optimal control, parallel'],
      url=(
          'https://github.com/ruairimoran/raocp-parallel'
      ),
      zip_safe=False)
