// Laplace (identity) constitutive model
#pragma once

#include <hollow/elastic/lame.h>
#include <geode/array/RawArray.h>
#include <geode/vector/Matrix.h>
namespace hollow {

using namespace geode;

template<int d_> struct LaplaceCM {
  static const int d = d_;
  typedef double T;

  LaplaceCM() {}
  LaplaceCM(Uninit) {}
  LaplaceCM(RawArray<const T> props) {}

  T energy(const Matrix<T,d>& F) const {
    return T(.5)*(F-1).sqr_frobenius_norm();
  }

  Matrix<T,d> stress(const Matrix<T,d>& F) const {
    return F-1;
  }

  Matrix<T,d> differential(const Matrix<T,d>& F, const Matrix<T,d>& dF) const {
    return dF;
  }

  void derivative(T dP[d*d*d*d], const Matrix<T,d>& F) const {
    memset(dP,0,d*d*d*d*sizeof(T));
    for (int i=0;i<d;i++)
      for (int j=0;j<d;j++)
        dP[((i*d+i)*d+j)*d+j] = 1;
  }
};

}
