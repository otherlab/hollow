#!/usr/bin/env python

from setuptools import setup,find_packages

setup(
  # Basics
  name='hollow',
  version='0.0-dev',
  description='Solid simulation based on petsc and geode',
  author='Otherlab et al.',
  author_email='geode-dev@googlegroups.com',
  url='http://github.com/otherlab/hollow',

  # Installation
  packages=find_packages(),
  package_data={'hollow':['*.py','*.so']},
  scripts=['bin/bend-tube'],
)
