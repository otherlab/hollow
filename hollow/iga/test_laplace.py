#!/usr/bin/env python

from __future__ import division,print_function,unicode_literals,absolute_import
from hollow import *
from geode import *
from geode.value import parser
import sys

def make_props(**kwargs):
  props = PropManager()
  props.add('petsc','')
  props.add('plot',False)
  for k,v in kwargs.items():
    props.get(k).set(v)
  return props

def laplace_test(props):
  petsc_reinitialize()
  petsc_set_options([sys.argv[0]]+props.petsc().split())
  comm = petsc_comm_world()
  iga = LaplaceTestIGA(comm)
  iga.dim = 2
  iga.dof = 1
  iga.set_from_options()
  iga.set_up()

  # Information
  Log.write('degrees = %s, order = %d, rational = %d'%(iga.degrees,iga.order,iga.rational))
  Log.write('bases = %s, spans = %s'%(iga.bases,iga.spans))
  Log.write('types = %s'%iga.basis_types)

  # Boundary conditions
  shift = 7
  for axis in xrange(iga.dim):
    for side in 0,1:
      iga.set_boundary_value(axis,side,0,shift)

  # Solve
  A = iga.create_mat()
  u = iga.create_vec()
  b = iga.create_vec()
  iga.compute_system(A,b)
  ksp = iga.create_ksp()
  ksp.set_operators(A,A,SAME_NONZERO_PATTERN)
  ksp.set_from_options()
  ksp.solve(b,u)
  Log.write(ksp.report())

  # Measure error
  error = iga.L2_error(u,shift)
  Log.write('error = %g'%error)
  return error

def test_laplace():
  props = make_props(petsc=' -iga_elements 1')
  assert allclose(laplace_test(props),0.03723796978507308)
  props = make_props(petsc=' -iga_elements 10')
  assert allclose(laplace_test(props),0.000109646)

def laplace_plot(props):
  petsc = props.petsc()
  elements = 2**arange(2,7)
  errors = []
  for e in elements:
    props.petsc.set(petsc+' -iga_elements %d'%e)
    errors.append(laplace_test(props))
  a,b = polyfit(log(elements),log(errors),deg=1)
  Log.write('slope = %g'%a)
  import pylab
  pylab.loglog(elements,errors,'bo-',label='errors')
  pylab.loglog(elements,exp(a*log(elements)+b),'g-',label='fit')
  pylab.legend()
  pylab.show()

if __name__=='__main__':
  props = make_props()
  parser.parse(props,'Laplace test')
  print('command = %s'%parser.command(props))
  if props.plot():
    laplace_plot(props)
  else:
    laplace_test(props)
