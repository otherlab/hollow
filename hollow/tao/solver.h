// Wrapper around TaoSolver
#pragma once

#include <hollow/petsc/vec.h>
#include <boost/function.hpp>
#include <taosolver.h>
namespace hollow {

struct TaoSolver : public Object {
  GEODE_DECLARE_TYPE(HOLLOW_EXPORT)
  typedef Object Base;
  typedef PetscReal T;

  const ::TaoSolver tao; 

protected:
  TaoSolver(const MPI_Comm comm);
public:
  ~TaoSolver();

  void set_from_options();
  void set_initial_vector(const Vec& x);
  void solve();

  // Add an additional monitoring routine
  void add_monitor(const boost::function<void()>& monitor);
};

}
