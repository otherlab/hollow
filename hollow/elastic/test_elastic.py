#!/usr/bin/env python

from __future__ import division,print_function,unicode_literals,absolute_import
from hollow import *
from geode import *
import sys

def test_neo_hookean():
  props = 1e6,.4 # E,nu
  random = Random(1311)
  neo_hookean_test(props,random,100)

def make_props(**kwargs):
  props = PropManager()
  props.add('petsc','')
  props.add('dim',2)
  props.add('degree',2)
  props.add_existing(Prop('resolution',(8,),shape=(-1,)))
  props.add('youngs_modulus',3e4)
  props.add('poissons_ratio',.45)
  props.add('density',1000.)
  props.add('gravity',9.8)
  props.add('check',False)
  props.add('view',False)
  props.add('mode','tao').set_allowed('tao snes'.split())
  for k,v in kwargs.items():
    props.get(k).set(v)
  return props

def geometry(props,iga):
  import igakit.cad
  import igakit.nurbs
  import igakit.io
  d = props.dim()
  n = props.resolution()
  if len(n)==1:
    n = n*ones(d,dtype=int)
  assert len(n)==d
  box = igakit.cad.grid(shape=n,degree=props.degree())
  box = igakit.nurbs.NURBS(box.knots,box.control,fields=box.control[...,:d])
  geom = named_tmpfile(prefix='box',suffix='.nurbs')
  boundary = named_tmpfile(prefix='boundary',suffix='.vec')
  igakit.io.PetIGA().write(geom.name,box)
  igakit.io.PetIGA().write_vec(boundary.name,box.control[...,:d],nurbs=box)
  iga.read(geom.name)
  return geom,boundary

def elastic_test(props):
  petsc_reinitialize()
  petsc_set_options([sys.argv[0]]+props.petsc().split())
  comm = petsc_comm_world()
  d = props.dim()
  material = props.youngs_modulus(),props.poissons_ratio()
  rho_g = props.density()*props.gravity()*-axis_vector(d-1,d=d)
  iga = NeoHookeanElastic[d](comm,material,rho_g)
  geom,boundary = geometry(props,iga)
  iga.set_from_options()
  iga.set_up()

  # Boundary conditions
  x = iga.read_vec(boundary.name)
  dummy = 17 # Overwritten by set_fix_table
  for axis in xrange(d):
    for side in 0,1:
      for i in xrange(d):
        if axis==0 and side==0:
          iga.set_boundary_value(axis,side,i,0)
  iga.set_fix_table(x)

  # Solve
  if props.mode()=='snes':
    snes = iga.create_snes()
    snes.set_from_options()
    if props.check():
      def check(iter,rnorm):
        snes.consistency_test(x,1e-6,1e-3,2e-10,10)
      snes.add_monitor(check)
    snes.solve(None,x)
  elif props.mode()=='tao':
    tao = iga.create_tao()
    tao.set_from_options()
    tao.set_initial_vector(x)
    tao.solve()

  # View
  if props.view():
    geom_file = named_tmpfile(prefix='geom',suffix='.nurbs')
    x_file = named_tmpfile(prefix='x',suffix='.vec')
    iga.write(geom_file.name)
    iga.write_vec(x_file.name,x)

    from igakit.io import PetIGA
    geom = PetIGA().read(geom_file.name)
    x = PetIGA().read_vec(x_file.name,nurbs=geom)
    geom.control[...,:d] = x
    from igakit.plot import plt
    plt.figure()
    plt.cwire(geom)
    plt.kwire(geom)
    plt.surface(geom)
    plt.show()

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
