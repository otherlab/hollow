// Wrapper around a petsc Vec

#include <hollow/vec.h>
#include <geode/python/Class.h>
#include <geode/utility/const_cast.h>
namespace hollow {

GEODE_DEFINE_TYPE(Vec)
typedef PetscScalar S;

Vec::Vec(const ::Vec v)
  : v(v) {}

Vec::~Vec() {
  CHECK(VecDestroy(&const_cast_(v)));
}

int Vec::size() const {
  int size;
  CHECK(VecGetSize(v,&size));
  return size;
}

int Vec::local_size() const {
  int local;
  CHECK(VecGetLocalSize(v,&local));
  return local;
}

Ref<Vec> Vec::clone() const {
  ::Vec copy;
  CHECK(VecDuplicate(v,&copy));
  return new_<Vec>(copy);
}

void Vec::set(S alpha) {
  CHECK(VecSet(v,alpha));
}

S Vec::sum() const {
  S s;
  CHECK(VecSum(v,&s));
  return s;
}

Array<S> Vec::local_copy() const {
  const S* x;
  CHECK(VecGetArrayRead(v,&x));
  const auto a = RawArray<const S>(local_size(),x).copy();
  CHECK(VecRestoreArrayRead(v,&x));
  return a;
}

void Vec::set_local(RawArray<const S> x) {
  GEODE_ASSERT(x.size()==local_size());
  S* p;
  CHECK(VecGetArray(v,&p));
  memcpy(p,x.data(),sizeof(S)*x.size());
  CHECK(VecRestoreArray(v,&p));
}

void Vec::axpy(const S a, const Vec& x) {
  GEODE_ASSERT(local_size()==x.local_size());
  CHECK(VecAXPY(v,a,x.v));
}

}
using namespace hollow;

void wrap_vec() {
  typedef hollow::Vec Self;
  Class<Self>("Vec")
    .GEODE_GET(size)
    .GEODE_GET(local_size)
    .GEODE_METHOD(clone)
    .GEODE_METHOD(set)
    .GEODE_METHOD(local_copy)
    .GEODE_METHOD(set_local)
    .GEODE_METHOD(sum)
    .GEODE_METHOD(axpy)
    ;
}
