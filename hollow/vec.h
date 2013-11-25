// Wrapper around a petsc Vec
#pragma once

#include <hollow/config.h>
#include <geode/array/Array.h>
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

  // Total size (on all processes)
  int size() const;

  // Size of portion on this process
  int local_size() const;

  // Duplicate the vector
  Ref<Vec> clone() const;

  // Set to a constant
  void set(S alpha);

  // Copy the local part into an array
  Array<S> local_copy() const;

  // Set the local part
  void set_local(RawArray<const S> x);

  S sum() const;
};

}
