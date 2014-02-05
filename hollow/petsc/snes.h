// Wrapper around petsc SNES
#pragma once

#include <hollow/petsc/vec.h>
#include <hollow/petsc/mat.h>
#include <hollow/petsc/dm.h>
#include <petscsnes.h>
#include <boost/function.hpp>
namespace hollow {

struct SNES : public Object {
  GEODE_DECLARE_TYPE(HOLLOW_EXPORT)
  typedef Object Base;
  typedef PetscReal T;

  const ::SNES snes;

protected:
  SNES(const MPI_Comm comm);
  SNES(const ::SNES snes); // Steals ownership
public:
  ~SNES();

  MPI_Comm comm() const;

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

  // Compute objective (which must exist)
  T objective(const Vec& x) const;

  // Compute function
  void residual(const Vec& x, Vec& f) const;

  // Do we have various functions?
  bool has_objective() const;
  bool has_jacobian() const;

  // Compare residuals and jacobians to finite differences
  void consistency_test(const Vec& x, const T small, const T rtol, const T atol, const int steps) const;

  // Add an additional monitoring routine
  void add_monitor(const boost::function<void(int,T)>& monitor);
};

}
