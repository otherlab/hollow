// Unit tests for constitutive models

#include <hollow/elastic/neo_hookean.h>
#include <hollow/elastic/warp.h>
#include <geode/python/wrap.h>
#include <geode/random/Random.h>
#include <geode/utility/Log.h>
#include <geode/vector/DiagonalMatrix.h>
#include <geode/vector/relative_error.h>
namespace hollow {

using Log::cout;
using std::endl;
typedef double T;

template<class Model> static inline void constitutive_model_test(const Model model, Random& random, const int steps) {
  typedef typename Model::T T;
  static const int d = Model::d;
  typedef Matrix<T,d> TM;
  const T small = 1e-7;

  // No deformation should be zero energy
  const T zero = model.energy(TM::identity_matrix());
  GEODE_ASSERT(zero==0);

  for (int step=0;step<steps;step++) {
    // Pick a random noninverted deformation gradient and a small random displacement
    Matrix<T,d> F, dF;
    random.fill_uniform(F,-1,1);
    if (F.determinant() < 0)
      for (int i=0;i<d;i++)
        F(0,i) *= -1;
    random.fill_uniform(dF,-small,small);

    {
      // Check energy vs. stress
      const T Up = model.energy(F+dF),
              Um = model.energy(F-dF);
      const auto P = model.stress(F);
      const T error = relative_error(Up-Um,2*inner_product(P,dF));
      //cout << "stress error = "<<error<<endl;
      GEODE_ASSERT(error<1e-6);
    } {
      // Check stress vs. differential
      const auto Pp = model.stress(F+dF),
                 Pm = model.stress(F-dF),
                 dP = model.differential(F,dF);
      const T error = relative_error(Pp-Pm,T(2)*dP);
      //cout << "differential error = "<<error<<endl;
      GEODE_ASSERT(error<1e-6);
    }
  }
}

template<class Warp> void warp_test(const Warp warp, const int steps) {
  typedef typename Warp::TV TV;
  const auto random = new_<Random>(18311);
  const T small = 1e-7,
          tolerance = 1e-7;
  for (int s=0;s<steps;s++) {
    const TV x = random->uniform<TV>(-1,1);
    const TV dx = small*random->uniform<TV>(-1,1);
    GEODE_ASSERT(relative_error(warp.map(x+dx)-warp.map(x-dx),T(2)*(warp.F(x)*dx))<tolerance);
  }
}

static void neo_hookean_test(RawArray<const T> props, Random& random, const int steps) {
  constitutive_model_test(NeoHookean<2>(props),random,steps);
  constitutive_model_test(NeoHookean<3>(props),random,steps);
}

static void warp_all_test(const int steps) {
  warp_test(NoWarp<2>(RawArray<const T>()),steps);
  warp_test(NoWarp<3>(RawArray<const T>()),steps);
  warp_test(BendWarp(asarray(vec(7,pi/3))),steps);
}

}
using namespace hollow;

void wrap_elastic_test() {
  GEODE_FUNCTION(neo_hookean_test)
  GEODE_FUNCTION(warp_all_test)
}
