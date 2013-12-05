// Neo Hookean constitutive model
#pragma once

#include <hollow/elastic/lame.h>
#include <geode/array/RawArray.h>
#include <geode/math/constants.h>
#include <geode/vector/Matrix.h>
namespace hollow {

using namespace geode;

// Required facts:
//
//   J = det F
//   cof F = J F^(-T)
//   d(log J) = cof F : dF

template<class T> static inline Matrix<T,2> cofactor_differential(const Matrix<T,2>& f, const Matrix<T,2>& d) {
  return Matrix<T,2>(d(1,1),-d(0,1),-d(1,0),d(0,0));
}

template<class T> static inline Matrix<T,3> cofactor_differential(const Matrix<T,3>& f, const Matrix<T,3>& d) {
#define ENTRY_(i,i1,i2,j,j1,j2) const T r##i##j = f(i1,j1)*d(i2,j2)+f(i2,j2)*d(i1,j1)-f(i1,j2)*d(i2,j1)-f(i2,j1)*d(i1,j2);
#define ENTRY(i,j) ENTRY_(i,((i+1)%3),((i+2)%3),j,((j+1)%3),((j+2)%3))
  ENTRY(0,0) ENTRY(0,1) ENTRY(0,2)
  ENTRY(1,0) ENTRY(1,1) ENTRY(1,2)
  ENTRY(2,0) ENTRY(2,1) ENTRY(2,2)
  return Matrix<T,3>(r00,r10,r20,r01,r11,r21,r02,r12,r22);
#undef ENTRY
#undef ENTRY_
}

template<int d_> struct NeoHookean {
  static const int d = d_;
  typedef double T;

  const T mu, lambda;

  // props = youngs_modulus, poissons_ratio
  NeoHookean(RawArray<const T> props)
    : mu((GEODE_ASSERT(props.size()==2),
          lame_mu(props[0],props[1])))
    , lambda(lame_lambda(props[0],props[1])) {}

  T energy(const Matrix<T,d>& F) const {
    // Bonet and Wood, Nonlinear Continuum Mechanics for Finite Element Analysis, Second Edition, pp. 162
    const T J = F.determinant();
    if (J <= 0)
      return inf;
    const T log_J = log(J);
    return T(.5)*mu*F.sqr_frobenius_norm()+(T(.5)*lambda*log_J-mu)*log_J;
  }

  // First Piola-Kirchhoff stress
  Matrix<T,d> stress(const Matrix<T,d>& F) const {
    const T J = F.determinant();
    GEODE_ASSERT(J > 0);
    return mu*F+(lambda*log(J)-mu)/J*F.cofactor_matrix();
  }

  // Stress differential
  Matrix<T,d> differential(const Matrix<T,d>& F, const Matrix<T,d>& dF) const {
    const T J = F.determinant();
    GEODE_ASSERT(J > 0);
    const T log_J = log(J);
    const auto cof = F.cofactor_matrix();
    const T dJ = inner_product(cof,dF);
    return mu*dF + (lambda*(1-log_J)+mu)*dJ/sqr(J)*cof
                 + (lambda*log_J-mu)/J*cofactor_differential(F,dF);
  }
};

}
