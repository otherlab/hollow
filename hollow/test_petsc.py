from __future__ import division

from hollow import *

def test_init():
  'Solve an unstructured 2D Poisson problem'
  petsc_initialize('a test',['test','-blah'])
  petsc_reinitialize()  
  petsc_set_options(['test','-foo'])
  petsc_finalize()
