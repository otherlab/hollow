// Wrapper around a petsc Mat
#pragma once

#include <hollow/config.h>
#include <hollow/vec.h>
#include <petscmat.h>
namespace hollow {

struct Mat : public Object {
  GEODE_DECLARE_TYPE(HOLLOW_EXPORT)
  typedef Object Base;

  const ::Mat m;

protected:
  Mat(const ::Mat m); // Steals ownership
public:
  ~Mat();

  MPI_Comm comm() const;

  void set_constant_nullspace();
};

}
