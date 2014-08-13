// Wrapper around a petsc KSP
#pragma once

#include <hollow/petsc/mat.h>
#include <petscksp.h>
namespace hollow {

struct KSP : public Object {
  GEODE_DECLARE_TYPE(HOLLOW_EXPORT)
  typedef Object Base;
  typedef PetscReal T;

  const ::KSP ksp;

protected:
  KSP(const ::KSP ksp); // Steals ownership
public:
  ~KSP();

  MPI_Comm comm() const;

  void set_operators(Mat& A, Mat& P);

  void set_from_options();

  void solve(const Vec& b, Vec& x);

  string report() const;
};

}
