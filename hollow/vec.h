// Wrapper around a petsc Vec
#pragma once

#include <hollow/config.h>
#include <geode/python/Object.h>
#include <petscvec.h>
namespace hollow {

struct Vec : public Object {
  GEODE_DECLARE_TYPE(HOLLOW_EXPORT)
  typedef Object Base;
  typedef PetscScalar S;

  const ::Vec v;

protected:
  Vec(const ::Vec v); // Steals ownership
public:
  ~Vec();

  // Duplicate the vector
  Ref<Vec> clone() const;

  // Set to a constant
  void set(S alpha);
};

}
