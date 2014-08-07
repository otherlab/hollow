// Wrapper around TaoSolver
#pragma once

#include <hollow/petsc/vec.h>
#include <hollow/petsc/snes.h>
#include <taosolver.h>
namespace hollow {

struct TaoSolver : public Object {
  GEODE_DECLARE_TYPE(HOLLOW_EXPORT)
  typedef Object Base;
  typedef PetscReal T;

  const ::TaoSolver tao;

  // Optional SNES from which to derive evaluation functions
  Ptr<const SNES> snes;
  Ptr<Mat> mat;

protected:
  TaoSolver(const MPI_Comm comm);
public:
  ~TaoSolver();

  MPI_Comm comm() const;

  void set_from_options();
  void set_initial_vector(const Vec& x);
  void solve();

  // Derive evaluation functions from the given snes
  void set_snes(const SNES& snes);

  // Add an additional monitoring routine
  void add_monitor(const function<void()>& monitor);
};

}
