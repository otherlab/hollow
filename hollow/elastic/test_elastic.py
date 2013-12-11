#!/usr/bin/env python

from __future__ import division,print_function,unicode_literals,absolute_import
from hollow import *
from geode import *
import sys

def test_neo_hookean():
  props = 1e6,.4 # E,nu
  random = Random(1311)
  neo_hookean_test(props,random,100)

def test_warp():
  warp_all_test(100)

def make_props(**kwargs):
  props = PropManager()
  props.add('petsc','')
  props.add('dim',2)
  props.add('youngs_modulus',1e4)
  props.add('poissons_ratio',.45)
  props.add('density',1000.)
  props.add('gravity',9.8)
  props.add('check',False)
  props.add('view',False)
  props.add('mode','tao').set_allowed('tao snes'.split())
  for k,v in kwargs.items():
    props.get(k).set(v)
  return props

def elastic_test(props):
  petsc_reinitialize()
  petsc_set_options([sys.argv[0]]+props.petsc().split())
  comm = petsc_comm_world()
  d = props.dim()
  material = props.youngs_modulus(),props.poissons_ratio()
  rho_g = props.density()*props.gravity()*-axis_vector(d-1,d=d)
  iga = NeoHookeanElastic[d](comm,material,(),rho_g)
  iga.set_from_options()
  iga.set_up()

  # Information
  Log.write('dim = %d, degrees = %s, order = %d, rational = %d'%(d,iga.degrees,iga.order,iga.rational))
  Log.write('bases = %s, spans = %s'%(iga.bases,iga.spans))
  Log.write('types = %s'%iga.basis_types)

  # Boundary conditions
  for axis in xrange(d):
    for side in 0,1:
      for i in xrange(d):
        if axis==0 and side==0:
          iga.set_boundary_value(axis,side,i,0)
        else:
          iga.set_boundary_load (axis,side,i,0)

  # Solve
  u = iga.create_vec()
  u.set(0)
  if props.mode()=='snes':
    snes = iga.create_snes()
    snes.set_from_options()
    if props.check():
      def check(iter,rnorm):
        snes.consistency_test(u,1e-6,1e-3,2e-10,10)
      snes.add_monitor(check)
    snes.solve(None,u)
  elif props.mode()=='tao':
    tao = iga.create_tao()
    tao.set_from_options()
    tao.set_initial_vector(u)
    tao.solve()

  # View
  if props.view():
    geom_file = named_tmpfile(prefix='geom',suffix='.dat')
    u_file = named_tmpfile(prefix='u',suffix='.dat')
    iga.write(geom_file.name)
    iga.write_vec(u_file.name,u)

    from igakit.io import PetIGA
    geom = PetIGA().read(geom_file.name)
    u = PetIGA().read_vec(u_file.name)
    #print(u)
    x = geom.copy()
    x.control[...,:d] += u.reshape(*(x.control.shape[:2]+(d,)))
    if 1:
      from igakit.plot import plt
      plt.figure()
      plt.cwire(x)
      plt.kwire(x)
      plt.surface(x)
      plt.show()
    else:
      import pylab
      n = 20

def test_elastic_snes():
  elastic_test(make_props(mode='snes',petsc='-snes_monitor -iga_elements 2',check=1))

def test_elastic_tao():
  elastic_test(make_props(mode='tao',petsc='-tao_monitor -iga_elements 2',check=1))

if __name__=='__main__':
  test_neo_hookean()
  props = make_props()
  parser.parse(props,'Elastic test')
  print('command = %s'%parser.command(props))
  petsc_initialize('Elastic test',[sys.argv[0]]+props.petsc().split())
  elastic_test(props)
