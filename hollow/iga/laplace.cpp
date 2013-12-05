// IGA Laplace solver

#include <hollow/iga/iga.h>
#include <hollow/petsc/mpi.h>
#include <geode/math/constants.h>
#include <geode/python/Class.h>
namespace hollow {
namespace {

// We solve the Laplace equation
//
//   -Delta u = f
//
// where
//
//   u = sin(pi x) sin(pi y)
//   u_xx = -pi^2 u
//   f = -Delta u = 2pi^2 u 

template<int d> struct LaplaceTestIGA : public IGA {
  GEODE_DECLARE_TYPE(GEODE_NO_EXPORT)
  typedef IGA Base;
  typedef PetscReal T;

protected:
  LaplaceTestIGA(const MPI_Comm comm)
    : IGA(comm) {
    set_dim(d);
    set_order(1); // Only need first derivatives
  }
public:

  void set_up() override {
    GEODE_ASSERT(dim()==d);
    CHECK(IGASetFormSystem(iga,system,(void*)this));
    Base::set_up();
  }

  static T exact(const T x, const T y) {
    return sin(pi*x)*sin(pi*y);
  }

  static PetscErrorCode system(IGAPoint p, T* K, T* F, void* ctx) {
    const int nen = p->nen;
    const T* N0 = (decltype(N0))p->shape[0];
    const T (*N1)[d] = (decltype(N1))p->shape[1];
    T x[d];
    IGAPointFormPoint(p,x);
    const T f = 2*sqr(pi)*exact(x[0],x[1]);

    for (int a=0;a<nen;a++) {
      F[a] = f*N0[a];
      for (int b=0;b<nen;b++)
        K[a*nen+b] = dot(asarray(N1[a]),asarray(N1[b]));
    }
    return 0;
  }

  T L2_error(const Vec& u, const T shift) const {
    const IGAFormScalar error = [](IGAPoint p, const T* U, const int n, T* S, void* ctx) {
      const T shift = *(const T*)ctx;
      T x[d];
      IGAPointFormPoint(p,x);
      T u;
      IGAPointFormValue(p,U,&u);
      S[0] = sqr(exact(x[0],x[1])+shift-u);
      return PetscErrorCode(0);
    };
    T sqr_error;
    CHECK(IGAComputeScalar(iga,u.v,1,&sqr_error,error,(void*)&shift));
    return sqrt(sqr_error);
  }
};

template<> GEODE_DEFINE_TYPE(LaplaceTestIGA<2>)

}
}
using namespace hollow;

void wrap_laplace_iga() {
  typedef LaplaceTestIGA<2> Self;
  Class<Self>("LaplaceTestIGA")
    .GEODE_INIT(MPI_Comm)
    .GEODE_METHOD(L2_error)
    ;
}
