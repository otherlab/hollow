// Optimize a second order integral of a curve, using cubic Catmull-Rom splines.
// The focus is on simplicity (all numeric differentiation) and efficient usability from python.

#include <hollow/tao/solver.h>
#include <geode/array/alloca.h>
#include <geode/array/Array3d.h>
#include <geode/array/Array4d.h>
#include <geode/array/NdArray.h>
#include <geode/array/sort.h>
#include <geode/python/Class.h>
#include <geode/utility/Log.h>
#include <geode/structure/Tuple.h>
namespace hollow {
namespace {

typedef PetscReal T;
using Log::cout;
using std::endl;

// Given knots t_i, 0<=i<=n, and values x_i, -1<=i<=n+1 (with one extra value on each end), our
// solution will be a piecewise cubic Catmull-Rom spline defined by
//
//   v_i = quadratic approximate to derivative at x_i
//   dt_i = t_{i+1}-t_i
//   s = (t-t_i)/dt_i
//
//   h00(s) = (1+2s)(1-s)^2
//   h10(s) = s(1-s)^2
//   h01(s) = (3-2s)s^2
//   h11(s) = (s-1)s^2
//
//   x(t) = x_i h00(s) + x_{i+1} h01(s) + dt_i (v_i h10(s) + v_{i+1} h11(s))
//
// For details, see http://en.wikipedia.org/wiki/Cubic_hermite_spline.

// Cubic Gauss-Legendre quadrature on [0,1].  See http://en.wikipedia.org/wiki/Gaussian_quadrature.
static const int quads = 3;
static const T weights[quads] = {5./18,8./18,5./18};
static const T samples[quads] = {(1-sqrt(.6))/2,.5,(1+sqrt(.6))/2};

// Evaluate energy at a bunch of quadrature points.  len(t)==len(x)==len(v) is arbitrary.
typedef boost::function<Array<T>(Array<const T> t, NdArray<const T> x, NdArray<const T> v)> PointEnergy;

struct Integral : public Object {
  GEODE_DECLARE_TYPE(GEODE_NO_EXPORT)
  typedef Integral Self;

  const int n; // Number of cells
  const Array<const int> shape; // Shape of parameter space
  const int d; // prod(shape)
  const Array<const int> xshape; // (n+3,)+shape
  const Array<const int> qshape; // (n*quads,)+shape
  const Array<const int> q2shape; // (n*quads*d*4,)+shape
  const PointEnergy U;
  const Array<const T> t;
  const NdArray<const T> smallx, smallv; // Tiny x and v values used for numerical differentiation
  const Array<Tuple<int,T>> bcs; // Boundary conditions

protected:
  mutable Array<T,2> x_expanded, grad_expanded; // Scratch arrays for expanding and reducing

  Integral(const PointEnergy& U, Array<const T> t, const Array<const int> shape, Array<const int,2> fixed,
           NdArray<const T> smallx, NdArray<const T> smallv)
    : n(t.size()-3)
    , shape(shape)
    , d(shape.product())
    , xshape(concatenate(asarray(vec(n+3)),shape))
    , qshape(concatenate(asarray(vec(n*quads)),shape))
    , q2shape(concatenate(asarray(vec(n*quads*d*4)),shape))
    , U(U)
    , t(t)
    , smallx(smallx)
    , smallv(smallv) {
    GEODE_ASSERT(t.size()>=2+2,"Need at least one cell and one extra t value on either side for tangents");
    GEODE_ASSERT(smallx.shape==shape);
    GEODE_ASSERT(smallv.shape==shape);
    GEODE_ASSERT(smallx.flat.min()>0);
    GEODE_ASSERT(smallv.flat.min()>0);

    // Collect boundary conditions
    GEODE_ASSERT(!fixed.m || fixed.n==xshape.size());
    Array<Tuple<int,T>> bcs(fixed.m,uninit);
    for (int i=0;i<bcs.size();i++) {
      int I = 0;
      for (int j=0;j<fixed.n;j++) {
        int f = fixed(i,j);
        if (f < 0)
          f += xshape[j];
        GEODE_ASSERT(0<=f && f<xshape[j]);
        I = I*xshape[j]+f;
      }
      bcs[i] = tuple(I,T(0));
    }
    sort(bcs);
    for (int i=0;i<bcs.size()-1;i++)
      GEODE_ASSERT(bcs[i].x<bcs[i+1].x);
    const_cast_(this->bcs) = bcs;
  }
public:
  ~Integral() {}

  #define T_INFO_LEFT(i) \
    const T t0 = t[i], \
            t1 = t[i+1], \
            t2 = t[i+2], \
            dt = t2-t1, \
            inv_dt = 1/dt, \
            dt0 = t1-t0, \
            s0 = 1/(dt0*(dt+dt0)), \
            c00 = -s0*sqr(dt), \
            c01 = s0*(dt0+dt)*(dt-dt0), \
            c02 = s0*sqr(dt0);
  #define T_INFO(i) \
    T_INFO_LEFT(i) \
    const T t3 = t[i+3], \
            dt1 = t3-t2, \
            s1 = 1/(dt1*(dt+dt1)), \
            c10 = -s1*sqr(dt1), \
            c11 = s1*(dt+dt1)*(dt1-dt), \
            c12 = s1*sqr(dt);
  #define SPLINE_INFO(s) \
    const T h00 = (1+2*s)*sqr(1-s), g00 = inv_dt*(6*s*(s-1)), \
            h10 = s*sqr(1-s),       g10 = inv_dt*(1+s*(3*s-4)), \
            h01 = (3-2*s)*sqr(s),   g01 = -g00, \
            h11 = (s-1)*sqr(s),     g11 = inv_dt*(s*(3*s-2)), \
            a0 = h10*c00, \
            b0 = g10*c00, \
            a1 = h00+h10*c01+h11*c10, \
            b1 = g00+g10*c01+g11*c10, \
            a2 = h01+h10*c02+h11*c11, \
            b2 = g01+g10*c02+g11*c11, \
            a3 = h11*c12, \
            b3 = g11*c12;
  #define X_INFO(i,a) \
    const T x0 = x(i  ,a), \
            x1 = x(i+1,a), \
            x2 = x(i+2,a), \
            x3 = x(i+3,a);

  NdArray<T> derivative(NdArray<const T> x) const {
    GEODE_ASSERT(x.rank());
    GEODE_ASSERT(x.shape[0]==n+3);
    const auto shape = x.shape.copy();
    shape[0] = n+1;
    const NdArray<T> dx(shape,uninit);
    const int k = x.flat.size()/(n+3);
    for (int i=0;i<=n;i++) {
      T_INFO_LEFT(i)
      for (int a=0;a<k;a++)
        dx.flat[i*k+a] = inv_dt*c00*x.flat[(i  )*k+a]
                       + inv_dt*c01*x.flat[(i+1)*k+a]
                       + inv_dt*c02*x.flat[(i+2)*k+a];
    }
    return dx;
  }

  T operator()(RawArray<const T,2> x) const {
    // Temporary arrays and views
    GEODE_ASSERT(x.sizes()==vec(n+3,d));

    // Collect quadrature points
    Array<T,2> tq(n,quads,uninit);
    Array<T,3> xq(n,quads,d,uninit);
    Array<T,3> vq(n,quads,d,uninit);
    for (int i=0;i<n;i++) {
      T_INFO(i)
      for (int q=0;q<quads;q++) {
        const T s = samples[q];
        tq(i,q) = t1+dt*s;
        SPLINE_INFO(s)
        for (int a=0;a<d;a++) {
          X_INFO(i,a)
          xq(i,q,a) = a0*x0+a1*x1+a2*x2+a3*x3;
          vq(i,q,a) = b0*x0+b1*x1+b2*x2+b3*x3;
        }
      }
    }

    // Compute energy
    const auto Uq_ = U(tq.reshape_own(n*quads),NdArray<const T>(qshape,xq.flat),NdArray<const T>(qshape,vq.flat));
    GEODE_ASSERT(Uq_.size()==n*quads);
    const auto Uq = Uq_.reshape(n,quads);

    // Accumulate
    T sum = 0;
    for (int i=0;i<n;i++) {
      const T dt = t[i+2]-t[i+1];
      for (int q=0;q<quads;q++)
        sum += weights[q]*dt*Uq(i,q);
    }
    return sum;
  }

  void gradient(RawArray<const T,2> x, RawArray<T,2> grad) const {
    // Temporary arrays and views
    GEODE_ASSERT(x.sizes()==vec(n+3,d) && grad.sizes()==x.sizes());
    const auto sx = smallx.flat.raw(),
               sv = smallv.flat.raw();

    // Collect quadrature points
    const int e = 4*d;
    Array<T,3> tq(    n,quads,e,uninit);
    Array<T,4> xq(vec(n,quads,e,d),uninit);
    Array<T,4> vq(vec(n,quads,e,d),uninit);
    for (int i=0;i<n;i++) {
      T_INFO(i)
      for (int q=0;q<quads;q++) {
        const T s = samples[q],
                t = t1+dt*s;
        for (int j=0;j<e;j++)
          tq(i,q,j) = t;
        SPLINE_INFO(s)
        for (int a=0;a<d;a++) {
          X_INFO(i,a)
          const T x = a0*x0+a1*x1+a2*x2+a3*x3,
                  v = b0*x0+b1*x1+b2*x2+b3*x3;
          for (int j=0;j<e;j++) {
            xq(i,q,j,a) = x;
            vq(i,q,j,a) = v;
          }
        }
        for (int a=0;a<d;a++) {
          xq(i,q,4*a  ,a) -= sx[a];
          xq(i,q,4*a+1,a) += sx[a];
          vq(i,q,4*a+2,a) -= sv[a];
          vq(i,q,4*a+3,a) += sv[a];
        }
      }
    }

    // Compute energies
    const auto Uq_ = U(tq.reshape_own(n*quads*e),NdArray<const T>(q2shape,xq.flat),NdArray<const T>(q2shape,vq.flat));
    GEODE_ASSERT(Uq_.size()==n*quads*e);
    const auto Uq = Uq_.reshape(n,quads,e);

    // Accumulate
    grad.fill(0);
    const auto inv_2s = GEODE_RAW_ALLOCA(d,Vector<T,2>);
    for (int a=0;a<d;a++)
      inv_2s[a] = vec(.5/sx[a],.5/sv[a]);
    for (int i=0;i<n;i++) {
      T_INFO(i)
      for (int q=0;q<quads;q++) {
        const T s = samples[q],
                w = dt*weights[q];
        SPLINE_INFO(s)
        for (int a=0;a<d;a++) {
          const T wx = w*inv_2s[a].x*(Uq(i,q,4*a+1)-Uq(i,q,4*a  )),
                  wv = w*inv_2s[a].y*(Uq(i,q,4*a+3)-Uq(i,q,4*a+2));
          grad(i  ,a) += a0*wx+b0*wv;
          grad(i+1,a) += a1*wx+b1*wv;
          grad(i+2,a) += a2*wx+b2*wv;
          grad(i+3,a) += a3*wx+b3*wv;
        }
      }
    }
  }

#if 0
  void hessian(RawArray<const T,2> x, RawArray<T,4> hess) const {
    // Temporary arrays and views
    GEODE_ASSERT(x.sizes()==vec(n+3,d) && hess.sizes()==vec(n+3,4,d,d));
    const auto sx = smallx.flat.raw(),
               sv = smallv.flat.raw();

    // Collect quadrature points
    const int e = 1+8*d+8*d*(d-1);
    Array<T,3> tq(    n,quads,e,uninit);
    Array<T,4> xq(vec(n,quads,e,d),uninit);
    Array<T,4> vq(vec(n,quads,e,d),uninit);
    for (int i=0;i<n;i++) {
      T_INFO(i)
      for (int q=0;q<quads;q++) {
        const T s = samples[q],
                t = t1+dt*s;
        for (int j=0;j<e;j++)
          tq(i,q,j) = t;
        SPLINE_INFO(s)
        for (int a=0;a<d;a++) {
          X_INFO(i,a)
          const T x = a0*x0+a1*x1+a2*x2+a3*x3,
                  v = b0*x0+b1*x1+b2*x2+b3*x3;
          for (int j=0;j<e;j++) {
            xq(i,q,j,a) = x;
            vq(i,q,j,a) = v;
          }
          int j = 1;
          for (int b=0;b<d;b++) {
            const T xb = sx[b],
                    vb = sv[b];
            xq(i,q,j++,a) -= xb;
            xq(i,q,j++,a) += xb;
            vq(i,q,j++,a) -= vb;
            vq(i,q,j++,a) += vb;
            xq(i,q,j  ,a) -= xb;
            vq(i,q,j++,a) -= vb;
            xq(i,q,j  ,a) -= xb;
            vq(i,q,j++,a) += vb;
            xq(i,q,j  ,a) += xb;
            vq(i,q,j++,a) -= vb;
            xq(i,q,j  ,a) += xb;
            vq(i,q,j++,a) += vb;
            for (int c=b+1;c<d;c++) {
              const T xc = sx[c],
                      vc = sv[c];
              xq(i,q,j++,a) -= xb+xc;
              xq(i,q,j++,a) -= xb-xc;
              xq(i,q,j++,a) += xb-xc;
              xq(i,q,j++,a) += xb+xc;
              vq(i,q,j++,a) -= vb+vc;
              vq(i,q,j++,a) -= vb-vc;
              vq(i,q,j++,a) += vb-vc;
              vq(i,q,j++,a) += vb+vc;

              vq(i,q,j++,a) -= sv[b];
              xq(i,q,j  ,a) -= sx[b];
              vq(i,q,j++,a) += sv[b];
              xq(i,q,j  ,a) += sx[b];
              vq(i,q,j++,a) -= sv[b];
              xq(i,q,j  ,a) += sx[b];
              vq(i,q,j++,a) += sv[b];
            }
          }
        }
      }
    }

    // Compute energies
    const auto Uq_ = U(tq.reshape_own(n*quads*d4),NdArray<const T>(q2shape,xq.flat),NdArray<const T>(q2shape,vq.flat));
    GEODE_ASSERT(Uq_.size()==n*quads*d4);
    const auto Uq = Uq_.reshape(n,quads,d4);

    // Accumulate
    grad.fill(0);
    const auto inv_2s = GEODE_RAW_ALLOCA(d,Vector<T,2>);
    for (int a=0;a<d;a++)
      inv_2s[a] = vec(.5/sx[a],.5/sv[a]);
    for (int i=0;i<n;i++) {
      T_INFO(i)
      for (int q=0;q<quads;q++) {
        const T s = samples[q],
                w = dt*weights[q];
        SPLINE_INFO(s)
        for (int b=0;b<d;b++) {
          const T wx = w*inv_2s[b].x*(Uq(i,q,4*b+1)-Uq(i,q,4*b  )),
                  wv = w*inv_2s[b].y*(Uq(i,q,4*b+3)-Uq(i,q,4*b+2));
          grad(i  ,b) += a0*wx+b0*wv;
          grad(i+1,b) += a1*wx+b1*wv;
          grad(i+2,b) += a2*wx+b2*wv;
          grad(i+3,b) += a3*wx+b3*wv;
        }
      }
    }
  }
#endif // hessian

  T operator()(NdArray<const T> x) const {
    GEODE_ASSERT(x.shape==xshape);
    return (*this)(x.flat.reshape(n+3,d));
  }

  NdArray<T> gradient(NdArray<const T> x) const {
    GEODE_ASSERT(x.shape==xshape);
    NdArray<T> grad(xshape,uninit);
    gradient(x.flat.reshape(n+3,d),grad.flat.reshape(n+3,d));
    return grad;
  }

  RawArray<const T,2> expand(RawArray<const T> xr) const {
    GEODE_ASSERT(xr.size()==(n+3)*d-bcs.size());
    x_expanded.resize(n+3,d,false,false);
    const auto xe = x_expanded.flat.raw();
    for (int k=0;k<=bcs.size();k++) {
      const int lo = k ? bcs[k-1].x+1 : 0,
                hi = k<bcs.size() ? bcs[k].x : xe.size();
      if (k < bcs.size())
        xe[hi] = bcs[k].y;
      for (int i=lo;i<hi;i++)
        xe[i] = xr[i-k];
    }
    return xe.reshape(n+3,d);
  }

  void reduce(RawArray<const T> xe, RawArray<T> xr) const {
    for (int k=0;k<=bcs.size();k++) {
      const int lo = k ? bcs[k-1].x+1 : 0,
                hi = k<bcs.size() ? bcs[k].x : xe.size();
      for (int i=lo;i<hi;i++)
        xr[i-k] = xe[i];
    }
  }

  template<class A> static PetscErrorCode value(A, ::Vec x, T* value, void* ctx) {
    const auto& s = *(const Self*)ctx;
    *value = s(s.expand(RawVec<const T>(x)));
    return 0;
  }

  template<class A> static PetscErrorCode gradient(A, ::Vec x, ::Vec grad, void* ctx) {
    const auto& s = *(const Self*)ctx;
    s.grad_expanded.resize(s.n+3,s.d,false);
    s.gradient(s.expand(RawVec<const T>(x)),s.grad_expanded);
    s.reduce(s.grad_expanded.flat,RawVec<T>(grad));
    return 0;
  }

  Ref<Vec> vector() const {
    ::Vec x;
    CHECK(VecCreateSeq(PETSC_COMM_SELF,(n+3)*d-bcs.size(),&x));
    return new_<Vec>(x);
  }

  void consistency_test(NdArray<const T> x, const T small, const T rtol, const T atol, const int steps) const {
    // Build vector
    GEODE_ASSERT(x.shape==xshape);
    const auto v = vector();
    reduce(x.flat,RawVec<T>(v->v));

    // Build an SNES object
    const auto snes = new_<SNES>(PETSC_COMM_SELF);
    CHECK(SNESSetObjective(snes->snes,value<::SNES>,(void*)this));
    CHECK(SNESSetFunction(snes->snes,v->v,gradient<::SNES>,(void*)this));

    // Test
    snes->consistency_test(v,small,rtol,atol,steps);
  }

  NdArray<T> minimize(NdArray<const T> x0) {
    GEODE_ASSERT(x0.shape==xshape);

    // Record boundary conditions
    for (auto& bc : bcs)
      bc.y = x0.flat[bc.x];

    // Construct solution vector.
    const auto x = vector();
    reduce(x0.flat,RawVec<T>(x->v));

    // Solve using TAO
    const auto tao = new_<TaoSolver>(PETSC_COMM_SELF);
    CHECK(TaoSetObjectiveRoutine(tao->tao,value<::TaoSolver>,(void*)this));
    CHECK(TaoSetGradientRoutine(tao->tao,gradient<::TaoSolver>,(void*)this));
    tao->set_from_options();
    tao->set_initial_vector(x);
    tao->solve();

    // Done
    return NdArray<T>(xshape,expand(RawVec<const T>(x->v)).flat.copy());
  }
};

GEODE_DEFINE_TYPE(Integral)

}
}
using namespace hollow;

void wrap_integral() {
  typedef Integral Self;
  Class<Self>("Integral")
    .GEODE_INIT(const PointEnergy&,Array<const T>,Array<const int>,Array<const int,2>,
                NdArray<const T>,NdArray<const T>)
    .GEODE_CALL(NdArray<const T>)
    .GEODE_OVERLOADED_METHOD(NdArray<T>(Self::*)(NdArray<const T>)const,gradient)
    .GEODE_METHOD(minimize)
    .GEODE_METHOD(consistency_test)
    .GEODE_METHOD(derivative)
    ;
}
