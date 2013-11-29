// Wrapper around petsc SNES
#pragma once

#include <hollow/petsc/vec.h>
#include <hollow/petsc/mat.h>
#include <hollow/petsc/dm.h>
#include <petscsnes.h>
namespace hollow {

struct SNES : public Object {
  GEODE_DECLARE_TYPE(HOLLOW_EXPORT)
  typedef Object Base;

  const ::SNES snes;

protected:
  SNES(const MPI_Comm comm);
public:
  ~SNES();

  void set_from_options();

  void set_dm(const DM& dm);

  // Set Jacobian matrices:
  //   A: Approximate Jacobian matrix
  //   B: Matrix used to construct the preconditioner (usually the same)
  void set_jacobian(const Mat& A, const Mat& P);

  // Solve the nonlinear system F(x) = b
  //   b: RHS, or null for b = 0
  //   x: Initial guess and solution
  void solve(Ptr<const Vec> b, Vec& x);

  int iterations() const;

  // Compute function
  void residual(const Vec& x, Vec& f) const;
};

}
