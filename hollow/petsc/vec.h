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

template<class SV> struct RawVec : public RawArray<SV> {
  typedef RawArray<SV> Base;
  typedef PetscScalar S;
  static_assert(is_same<S,typename remove_const<typename ScalarPolicy<SV>::type>::type>::value,"");
  static const int ratio = sizeof(SV)/sizeof(S);

  const ::Vec v;

  RawVec(::Vec v)
    : Base(size(v),(SV*)data(v,mpl::bool_<is_const<SV>::value>()))
    , v(v) {}

  ~RawVec() {
    typedef typename ScalarPolicy<SV>::type Sc;
    auto data = (Sc*)Base::data();
    restore(v,data);
  }

  template<class TA> const Base& operator=(const TA& source) const {
    return static_cast<const Base&>(*this) = source;
  }

private:

  static int size(const ::Vec v) {
    const int ratio = sizeof(SV)/sizeof(S);
    static_assert(ratio*sizeof(S)==sizeof(SV),"");
    int local;
    CHECK(VecGetLocalSize(v,&local));
    const int size = local/ratio;
    GEODE_ASSERT(size*ratio==local);
    return size;
  }

  static       S* data(const ::Vec v, mpl::false_) {       S* p; CHECK(VecGetArray    (v,&p)); return p; }
  static const S* data(const ::Vec v, mpl::true_ ) { const S* p; CHECK(VecGetArrayRead(v,&p)); return p; }
  static void restore(const ::Vec v,       S*& p) { CHECK(VecRestoreArray    (v,&p)); }
  static void restore(const ::Vec v, const S*& p) { CHECK(VecRestoreArrayRead(v,&p)); }
};

}
