#!/usr/bin/env python

from __future__ import division,print_function,unicode_literals,absolute_import
from hollow import *
from geode import *
from geode.geometry.platonic import *
import sys

def test_neo_hookean():
  props = 1e6,.4 # E,nu
  random = Random(1311)
  neo_hookean_test(props,random,100)

def test_laplace_cm():
  random = Random(1311)
  laplace_cm_test(random,100)

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
  props.add('model','neo-hookean').set_allowed('neo-hookean laplace'.split())
  props.add('mode','tao').set_allowed('tao snes'.split())
  props.add('type','iga').set_allowed('iga fe'.split())
  for k,v in kwargs.items():
    props.get(k).set(v)
  return props

def check_n(props):
  d = props.dim()
  n = props.resolution()
  if len(n)==1:
    n = n*ones(d,dtype=int)
  assert len(n)==d
  print('resolution = %s'%n)
  return n

def iga_geometry(props,iga):
  import igakit.cad
  import igakit.nurbs
  import igakit.io
  d = props.dim()
  n = check_n(props)
  box = igakit.cad.grid(shape=n,degree=props.degree())
  box = igakit.nurbs.NURBS(box.knots,box.control,fields=box.control[...,:d])
  geom = named_tmpfile(prefix='box',suffix='.nurbs')
  boundary = named_tmpfile(prefix='boundary',suffix='.vec')
  igakit.io.PetIGA().write(geom.name,box)
  igakit.io.PetIGA().write_vec(boundary.name,box.control[...,:d],nurbs=box)
  iga.read(geom.name)
  return geom,boundary

def dm_geometry(props,comm):
  d = props.dim()
  assert d==2
  n = check_n(props)
  mesh = TriangleTopology(grid_topology(*n))
  X = zeros((n[0]+1,n[1]+1,2))
  X[:,:,0] = linspace(0,1,num=n[0]+1)[:,None]
  X[:,:,1] = linspace(0,1,num=n[1]+1)[None,:]
  X = X.reshape(-1,2)

  # Convert to petsc
  dm,edges = dmplex_mesh(comm,mesh,X)

  # Mark one side as a boundary
  mark = []
  for i,e in enumerate(edges):
    if (X[mesh.src(e)]+X[mesh.dst(e)])[0]<1e-10:
      mark.append(i)
  dm.mark('wall',mesh.n_vertices+mesh.n_faces+asarray(mark,dtype=int32))

  # All done
  dm.distribute('chaco')
  return dm

def elastic_test(props):
  petsc_reinitialize()
  petsc_set_options([sys.argv[0]]+props.petsc().split())
  comm = petsc_comm_world()
  d = props.dim()
  material = props.youngs_modulus(),props.poissons_ratio()
  rho_g = props.density()*props.gravity()*-axis_vector(d-1,d=d)
  type = props.type()
  mode = props.mode()

  if type=='iga':
    assert props.model()=='neo-hookean'
    iga = NeoHookeanElasticIGA[d](comm,material,rho_g)
    geom,boundary = iga_geometry(props,iga)
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

    if mode=='snes':
      snes = iga.create_snes()
      snes.set_from_options()
    elif mode=='tao':
      tao = iga.create_tao()
      tao.set_from_options()

  elif type=='fe':
    dm = dm_geometry(props,comm)
    degree = props.degree()
    degree = 1
    petsc_add_options(['ignored','-petscspace_order',str(degree)])
    if 0: # Would be needed for neumann
      petsc_add_options(['ignored','-bd_petscspace_order',str(degree)])
    snes = SNES(comm)
    snes.set_dm(dm)
    fe = FE(comm,d,d),
    print('fe dofs = %s'%fe[0].dofs)
    fe_bd = fe_aux = ()
    dm.create_default_section(('x',),fe,'wall',(0,))
    start = identity_analytic(d)
    model = {'neo-hookean':NeoHookeanElasticModel,'laplace':LaplaceElasticModel}[props.model()]
    model = model[d](fe,fe_aux,fe_bd,start,material,rho_g)
    dm.set_model(model,mode=='tao') # Use a real objective for tao but not for snes
    A = dm.create_matrix()
    snes.set_jacobian(A,A)
    snes.set_from_options()
    if mode=='tao':
      tao = TaoSolver(comm)
      tao.set_snes(snes)
      tao.set_from_options()
    x = dm.create_global_vector()
    dm.project((start,),INSERT_VALUES,x)
  else:
    raise RuntimeError("unknown discretization type '%s'"%type)

  def check(*args):
    if props.check() and (mode=='snes' or type=='fe'):
      snes.consistency_test(x,1e-6,1e-3,2e-10,10)
  check()

  # Solve
  if mode=='snes':
    snes.add_monitor(check)
    snes.solve(None,x)
  elif mode=='tao':
    tao.set_initial_vector(x)
    tao.add_monitor(check)
    tao.solve()

  # View
  if props.view():
    if type=='iga':
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
    elif type=='fe':
      f = named_tmpfile(prefix='elastic',suffix='.vtk')
      dm.write_vtk(f.name,x)
      subprocess.check_call(['vtk',f.name])
    else:
      raise RuntimeError("weird type '%s'"%type)

def test_elastic_iga_snes():
  elastic_test(make_props(type='iga',mode='snes',petsc='-snes_monitor -iga_elements 2',check=1))

def test_elastic_iga_tao():
  elastic_test(make_props(type='iga',mode='tao',petsc='-tao_monitor -iga_elements 2',check=1))

def test_elastic_fe_snes():
  elastic_test(make_props(type='fe',mode='snes',resolution=(8,),petsc='-snes_monitor',check=1))

def test_elastic_fe_tao():
  elastic_test(make_props(type='fe',mode='tao',resolution=(8,),petsc='-tao_monitor',check=1))

def test_laplace_fe_tao():
  elastic_test(make_props(type='fe',mode='tao',model='laplace',density=1,resolution=(8,),petsc='-tao_monitor',check=1))

if __name__=='__main__':
  test_neo_hookean()
  test_laplace_cm()
  props = make_props()
  parser.parse(props,'Elastic test')
  print('command = %s'%parser.command(props))
  petsc_initialize('Elastic test',[sys.argv[0]]+props.petsc().split())
  elastic_test(props)
