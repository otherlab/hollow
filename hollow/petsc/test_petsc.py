#!/usr/bin/env python

from __future__ import division
from hollow import *

def test_init():
  # Can't call petsc_initialize since other tests might have already
  petsc_reinitialize()  
  petsc_set_options(['test','-foo'])
  petsc_finalize()

if __name__ == '__main__':
  test_init()
