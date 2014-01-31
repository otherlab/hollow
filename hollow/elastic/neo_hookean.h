// Neo Hookean constitutive model
#pragma once

#include <hollow/elastic/lame.h>
#include <geode/array/RawArray.h>
#include <geode/geometry/Box.h>
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

// r = a + b outer(cof(f)) + c dcof(f)
template<class T,int d> static inline void standard_tensor(T r[d*d*d*d], const Matrix<T,d>& f, const T a, const T b, const T c);

template<> inline void standard_tensor<double,2>(double r[16], const Matrix<double,2>& f, const double a, const double b, const double c) {
  const auto g = f.cofactor_matrix();
  #define E0(i,j,k,l) ({ \
    auto t = b*g(i,j)*g(k,l); \
    if (i==k && j==l) t += a; \
    if (i!=k && j!=l) { \
      if (i==j) t += c; \
      else      t -= c; \
    } \
    r[((i*2+k)*2+j)*2+l] = t; });
  #define E1(i,j,k) E0(i,j,k,0) E0(i,j,k,1)
  #define E2(i,j)   E1(i,j,0)   E1(i,j,1)
  #define E3(i)     E2(i,0)     E2(i,1)
  E3(0) E3(1)
  #undef E3
  #undef E2
  #undef E1
  #undef E0
}

template<> inline void standard_tensor<double,3>(double r[81], const Matrix<double,3>& f, const double a, const double b, const double c) {
  const auto g = f.cofactor_matrix();
  #define E0(i,j,k,l) ({ \
    auto t = b*g(i,j)*g(k,l); \
    if (i==k && j==l) t += a; \
    if (i!=k && j!=l) { \
      const auto u = c*f(3-i-k,3-j-l); \
      if ((k-i-l+j+6)%3) t -= u; \
      else               t += u; \
    } \
    r[((i*3+k)*3+j)*3+l] = t; });
  #define E1(i,j,k) E0(i,j,k,0) E0(i,j,k,1) E0(i,j,k,2)
  #define E2(i,j)   E1(i,j,0)   E1(i,j,1)   E1(i,j,2)
  #define E3(i)     E2(i,0)     E2(i,1)     E2(i,2)
  E3(0) E3(1) E3(2)
  #undef E3
  #undef E2
  #undef E1
  #undef E0
}

// Flip to enable Jacobian determinant monitoring
#define HOLLOW_MONITOR_J(...)
//#define HOLLOW_MONITOR_J(...) __VA_ARGS__

template<int d_> struct NeoHookean {
  static const int d = d_;
  typedef double T;

  T mu, lambda;
  HOLLOW_MONITOR_J(mutable Box<T> J_range;)

  NeoHookean(Uninit) {}

  // props = youngs_modulus, poissons_ratio
  NeoHookean(RawArray<const T> props)
    : mu((GEODE_ASSERT(props.size()==2),
          lame_mu(props[0],props[1])))
    , lambda(lame_lambda(props[0],props[1])) {}

  T energy(const Matrix<T,d>& F) const {
    // Bonet and Wood, Nonlinear Continuum Mechanics for Finite Element Analysis, Second Edition, pp. 162
    const T J = F.determinant();
    HOLLOW_MONITOR_J(J_range.enlarge(J));
    if (J <= 0)
      return inf;
    const T log_J = log(J);
    return T(.5)*mu*(F.sqr_frobenius_norm()-d)+(T(.5)*lambda*log_J-mu)*log_J;
  }

  // First Piola-Kirchhoff stress
  Matrix<T,d> stress(const Matrix<T,d>& F) const {
    const T J = F.determinant();
    if (J <= 0)
      return Matrix<T,d>(); // Tao calls this routine for infinite energy sometimes, so we have to return something
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

  // Stress derivative: dP_ijkl = d(Pik)/d(F(jl)
  void derivative(T dP[d*d*d*d], const Matrix<T,d>& F) const {
    const T mu     = this->mu,
            lambda = this->lambda;
    const T J = F.determinant();
    GEODE_ASSERT(J > 0);
    const T log_J = log(J),
            s = (lambda*(1-log_J)+mu)/sqr(J),
            r = (lambda*log_J-mu)/J;
    standard_tensor(dP,F,mu,s,r);
  }
};

}
