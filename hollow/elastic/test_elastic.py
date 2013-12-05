#!/usr/bin/env python

from __future__ import division,print_function,unicode_literals,absolute_import
from hollow import *
from geode import *

def test_neo_hookean():
  props = 1e6,.4 # E,nu
  random = Random(1311)
  neo_hookean_test(props,random,100)

if __name__=='__main__':
  test_neo_hookean()
