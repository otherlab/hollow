#!/usr/bin/env python
# A 2D Poisson test

from __future__ import division,print_function,unicode_literals,absolute_import
from hollow import *
from geode import *
from geode.geometry.platonic import *
from geode.value import parser
import sys

def make_props(**kwargs):
  props = PropManager()
  props.add('petsc','')
  props.add('resolution',30)
  props.add('refine_volume',0.)
  props.add('refine_levels',0)
  props.add('order',1)
  props.add('bc','dirichlet').set_allowed('dirichlet neumann'.split()) \
    .set_help('boundary condition type')
  props.add('dump_mesh',False)
  for k,v in kwargs.items():
    props.get(k).set(v)
  return props

def create_mesh(props,comm):
  # Create mesh data
  n = props.resolution()
  m = 3*n//2
  mesh = TriangleTopology(grid_topology(m,n))
  X = zeros((m+1,n+1,2))
  X[:,:,0] = linspace(0,1,num=m+1)[:,None]
  X[:,:,1] = linspace(0,1,num=n+1)[None,:]
  X = X.reshape(-1,2)

  # Convert to petsc
  dm,edges = dmplex_mesh(comm,mesh,X)
  if props.refine_volume():
    dm.volume_refine(props.refine_volume())
  dm.distribute('chaco')
  if props.refine_levels():
    dm.uniform_refine(props.refine_levels())

  # Dump mesh information
  if props.dump_mesh():
    print('mesh dump:')
    for v in mesh.vertices():
      print('  v %d = %s'%(v,X[v]))
    for i,e in enumerate(edges):
      s = mesh.src(e)
      d = mesh.dst(e)
      print('  e %d = %s (%d %d)'%(i,(X[s]+X[d])/2,s,d))

  # Print some information
  counts = asarray([mesh.n_vertices,mesh.n_edges,mesh.n_faces])
  assert all(counts==dm.counts)
  return dm

def laplace_test(props):
  petsc_reinitialize()
  order = props.get('order')()
  options = [sys.argv[0],'-petscspace_order',order]
  neumann = props.bc()=='neumann'
  if neumann:
    options.extend(['-bd_petscspace_order',order])
  petsc_set_options(map(str,options)+props.petsc().split())
  comm = petsc_comm_world()
  snes = SNES(comm)
  dm = create_mesh(props,comm)
  snes.set_dm(dm)
  dim = dm.dim

  # Finite elements and models
  fe = FE(comm,dim,1),
  print('fe dofs = %s'%fe[0].dofs)
  fe_bd = fe_aux = ()
  if neumann:
    fe_bd = FE(comm,dim-1,1,"bd_"),
    print('fe bd dofs = %s'%fe_bd[0].dofs)
  if 0:
    # Material quadrature must agree with solution quadrature
    fe_aux = FE(comm,dim,1,"mat_",fe[0].qorder()),
    print('fe aux dofs = %s'%fe_aux[0].dofs)
  model = LaplaceTest2d(fe,fe_aux,fe_bd,neumann)
  dm.set_model(model)

  # Sections and fields
  counts = dm.mark_boundary('boundary')
  print('boundary: faces = %d, all = %d'%(counts[0],counts[1]))
  dm.create_default_section(('potential',),model.fe,'boundary',() if neumann else (0,))
  u = dm.create_global_vector()
  r = u.clone()
  if model.fe_aux:
    dm_aux = dm.clone()
    dm_aux.create_default_section(('nu',),model.fe_aux,'',())
    assert 0 # Need clean way to set fields from python (equivalent of SetupMaterial in snes ex12)

  # Check residual for u = 0
  if 0:
    u.set(0)
    f = u.clone()
    snes.residual(u,f)
    print('RHS = %s (sum %g)'%(f.local_copy(),f.sum()))

  # Matrix
  A = dm.create_matrix()
  if neumann:
    A.set_constant_nullspace()
  snes.set_jacobian(A,A)

  # Configure snes
  snes.set_from_options()

  # Solve
  u.set(0)
  snes.solve(None,u)
  print('snes iterations = %d'%snes.iterations)

  if neumann:
    # Fit quadratic to shift and pick minimum
    model.shift = 0
    a = dm.L2_error_vs_exact(u)**2
    model.shift = 1
    c_plus_b = dm.L2_error_vs_exact(u)**2-a
    model.shift = -1
    c_minus_b = dm.L2_error_vs_exact(u)**2-a
    b = (c_plus_b-c_minus_b)/2
    c = (c_plus_b+c_minus_b)/2
    model.shift = -b/(2*c)

  # Measure error
  error = dm.L2_error_vs_exact(u)
  print('L2 error = %g'%error)
  if 0 and u.local_size<100:
    print('u = %s'%repr(list(u.local_copy())))
  return u,error

def test_dirichlet_linear():
  props = make_props(bc='dirichlet',order=1,resolution=2)
  u,e = laplace_test(props)
  assert allclose(e,13/162)
  assert allclose(u.local_copy(),[13/36,25/36])

def test_dirichlet_quadratic():
  props = make_props(bc='dirichlet',order=2,resolution=1)
  u,e = laplace_test(props)
  assert allclose(e,0)
  assert allclose(u.local_copy(),[1/2])

def test_neumann_linear():
  props = make_props(bc='neumann',order=1,resolution=1)
  u,e = laplace_test(props)
  assert allclose(e,0) # Zero only because of low order quadrature
  assert allclose(u.local_copy(),asarray([-5,-1,-1,7])/6)

def test_neumann_quadratic():
  props = make_props(bc='neumann',order=2,resolution=1)
  u,e = laplace_test(props)
  assert allclose(e,0)
  assert allclose(u.local_copy(),asarray([-10,2,2,14,-7,-7,-4,5,5])/12)

if __name__=='__main__':
  props = make_props()
  parser.parse(props,'Laplace test')
  print('command = %s'%parser.command(props))
  laplace_test(props)
