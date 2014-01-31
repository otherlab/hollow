// Pointwise analytically known functions

#include <hollow/petsc/analytic.h>
#include <geode/python/Class.h>
#include <geode/python/wrap.h>
#include <geode/utility/format.h>
namespace hollow {

GEODE_DEFINE_TYPE(Analytic)

Analytic::Analytic(const int dim, const int count, const F f, void* const ctx)
  : dim(dim)
  , count(count)
  , f(f)
  , ctx(ctx) {}

Analytic::~Analytic() {}

namespace {

template<int d> class IdentityAnalytic : public Analytic {
  GEODE_NEW_FRIEND

  IdentityAnalytic()
    : Analytic(d,d,f,0) {}

  static void f(const T* x, S* u, void*) {
    for (int i=0;i<d;i++)
      u[i] = x[i];
  }
};

Ref<const Analytic> identity_analytic(const int d) {
  switch (d) {
    case 1: return new_<IdentityAnalytic<1>>();
    case 2: return new_<IdentityAnalytic<2>>();
    case 3: return new_<IdentityAnalytic<3>>();
    default: GEODE_NOT_IMPLEMENTED(format("dimension must be 1, 2, or 3, got %d",d));
  }
}

}
}
using namespace hollow;

void wrap_analytic() {
  typedef Analytic Self;
  Class<Self>("Analytic")
    .GEODE_FIELD(dim)
    .GEODE_FIELD(count)
    ;
  GEODE_FUNCTION(identity_analytic)
}
