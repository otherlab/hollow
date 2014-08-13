#!/usr/bin/env python

from __future__ import division,print_function,unicode_literals

from hollow import *
from geode import *
import scipy.optimize
import __builtin__

def test_integral():
  "Polynomial functions should be exact up to degree 2*3-1 = 5"
  "Cubics should be exact even with one cell"
  petsc_reinitialize()
  random.seed(23131)
  t = sort(random.randn(4))
  c = random.randn(6)
  def cdot(*a):
    assert len(a)==len(c)
    return __builtin__.sum(ci*ai for ci,ai in zip(c,a))
  def f(t,u,du):
    return cdot(1,t,u,du,t*u*du**2,(t*u*du)**2)
  def g(t,u,du):
    return t*u*du**2
  def h(t,u,du):
    a,b = rollaxis(u,-1)
    return cdot(1,a,b,a*a,a*b,b*b)
  small = 1e-6
  hsmall = small,small
  fixed = zeros((0,0),dtype=int32)
  F = Integral(f,t,(),fixed,small,small)
  G = Integral(g,t,(),fixed,small,small)
  H = Integral(h,t,(2,),fixed,hsmall,hsmall)
  def d(*a):
    return [aa[2]-aa[1] for aa in a]
  assert allclose(F(t),cdot(*d(t,t**2/2,t**2/2,t,t**3/3,t**5/5)))
  assert allclose(G(t*t),d(2/3*t**6))
  assert allclose(G.derivative(t*t),2*t[1:-1])
  hx = array([t,t*t]).T.copy()
  assert allclose(H(hx),cdot(*d(t,t**2/2,t**3/3,t**3/3,t**4/4,t**5/5)))
  assert allclose(H.derivative(hx),array([1+0*t,2*t]).T.copy()[1:-1])
  x = random.randn(4)
  hx = random.randn(4,2)
  F.consistency_test(x,small,1e-5,1e-15,20)
  G.consistency_test(x,small,1e-5,1e-15,20)
  H.consistency_test(hx,small,1e-5,1e-15,20)

def test_laplace():
  """A simple 1D Laplace solve"""
  debug = 0
  petsc_reinitialize()
  petsc_set_options('''laplace
    -tao_type lmvm -tao_lmm_vectors 20 -tao_max_it 1000
    -tao_fatol 1e-10 -tao_frtol 1e-10
    '''.split())
  if debug:
    petsc_add_options('''laplace -tao_view -tao_monitor -tao_converged_reason'''.split())
  def U(t,u,du):
    return du**2/2-sin(t)*u
  ns = asarray([2,4,8,16,32,64,128])
  errors = []
  for n in ns:
    t = pi/n*arange(-1,n+2)
    u0 = zeros(n+3)
    fixed = [(1,),(-2,)]
    u = Integral(U,t,(),fixed,1e-7,1e-7).minimize(u0)
    ti = t[1:-1]
    ui = u[1:-1]
    error = sqrt(sqr(ui-sin(ti)).sum()/n)
    errors.append(error)
    print('laplace error %d = %g'%(n,error))
    if 0 and n==32:
      import pylab
      pylab.plot(ti,ui)
      pylab.show()
  print('ns = %s'%ns)
  print('errors = %s'%-log2(errors))
  assert all(-log2(errors)>[8,9,13,16,18,17,18])
  if debug:
    import pylab
    pylab.plot(log2(ns),-log2(errors))
    pylab.show()

def test_caternary():
  """Consider a curve of length L hanging between two points
  separated by distance D.  We solve for x(t),y(t) with t in [0,L],
  using the penalized pointwise energy
    U(t) = k/2 (sqrt(dx^2+dy^2)-1)^2 + y
  Our initial condition is a circle around (0,C) of radius r through (+-D/2,0):
    r = sqrt(C^2+D^2/4)
    2*r*atan(D/2,C) = L
  """
  debug = 0
  petsc_reinitialize()
  petsc_set_options('''laplace
    -tao_type lmvm -tao_lmm_vectors 20 -tao_max_it 1000
    -tao_fatol 1e-10 -tao_frtol 1e-10
    '''.split())
  if debug:
    petsc_add_options('''laplace -tao_view -tao_monitor -tao_converged_reason'''.split())

  # Compute initial condition
  D = 1
  L = 1.5
  C = scipy.optimize.brentq(lambda C:2*sqrt(C*C+D*D/4)*atan2(D/2,C)-L,-10*D,10*D)
  R = sqrt(C*C+D*D/4)
  def initial(t):
    a = atan(D/2/C)/(L/2)
    x = (0,C)+R*polar(a*t-pi/2)
    assert allclose(t[1],-L/2)
    assert allclose(t[-2],L/2)
    assert allclose(x[1],(-D/2,0))
    assert allclose(x[-2],(D/2,0))
    return x

  # Compute analytic solution
  S = scipy.optimize.brentq(lambda S:S*asinh(L/2/S)-D/2,D/100,10*D)
  def analytic(t):
    x = array([S*asinh(t/S),sqrt(S*S+t*t)-sqrt(S*S+L*L/4)]).T.copy()
    print(x[1],(-D/2,0))
    assert allclose(x[1],(-D/2,0))
    assert allclose(x[-2],(D/2,0))
    return x

  # Define problem
  k = 1000
  def Ut(t,x,dx):
    return k/2*sqr(magnitudes(dx)-1)+x[...,1] 
  n = 50
  t = linspace(-L/2,L/2,num=n+1)
  t = concatenate([[2*t[0]-t[1]],t,[2*t[-1]-t[-2]]]) # Add sentinels
  assert len(t)==n+3
  fixed = [(1,0),(1,1),(-2,0),(-2,1)]
  smallv = 1e-6*ones(2)
  smallx = (t[1]-t[0])*smallv
  U = Integral(Ut,t,(2,),fixed,smallx,smallv)

  # Test
  x0 = initial(t)
  xa = analytic(t)
  U.consistency_test(x0,1e-6,1e-6,1e-15,20)

  # Solve
  petsc_set_options('''caternary -tao_view -tao_monitor -tao_converged_reason
    -tao_type lmvm -tao_lmm_vectors 20 -tao_max_it 1200
    '''.split())
    # -tao_fatol 1e-10 -tao_frtol 1e-10
  import pylab
  x = U.minimize(x0)
  error = maxabs((x-xa)[1:-1])
  print('caternary error = %g'%error)
  if 0:
    pylab.plot(x0[1:-1,0],x0[1:-1,1],'g')
    pylab.plot(x [1:-1,0],x [1:-1,1],'r.-')
    pylab.plot(xa[1:-1,0],xa[1:-1,1],'b')
    pylab.axes().set_aspect('equal')
    pylab.show()

  # Check error
  assert error<.0012

if __name__=='__main__':
  petsc_initialize('ODE tests','test_ode -info ode.log'.split())
  test_caternary()
  test_integral()
  test_laplace()
