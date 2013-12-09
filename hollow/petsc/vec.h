// Wrapper around a petsc Vec
#pragma once

#include <hollow/petsc/config.h>
#include <geode/array/Array.h>
#include <geode/math/uint128.h>
#include <geode/python/Object.h>
#include <petscvec.h>
namespace hollow {

struct Vec : public Object {
  GEODE_DECLARE_TYPE(HOLLOW_EXPORT)
  typedef Object Base;
  typedef PetscReal T;
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
  T norm(NormType type) const;
  T min() const;
  T max() const;

  void axpy(const S a, const Vec& x);
  void waxpy(const S a, const Vec& x, const Vec& y);

  // Set each component to a random value in [lo,hi)
  // This function is deterministic as a function of key, and parallel environment if ownership is contiguous.
  void set_random(const uint128_t key, const T lo, const T hi);
};

static inline PetscScalar dot(const Vec& x, const Vec& y) {
  PetscScalar dot;
  CHECK(VecDot(x.v,y.v,&dot));
  return dot;
}

}
