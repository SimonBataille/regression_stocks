#!/usr/bin/env python3

from setuptools import setup

setup(name='regression_stocks',
      version='1.0',
      # list folders, not files
      packages=['regression_stocks'],
      scripts=['regression_stocks/bin/regression_script.py'],
      package_data={'regression_stocks': ['data/regression_data.txt']},
      )
