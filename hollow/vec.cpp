// Wrapper around a petsc Vec

#include <hollow/vec.h>
#include <geode/python/Class.h>
#include <geode/utility/const_cast.h>
namespace hollow {

GEODE_DEFINE_TYPE(Vec)

Vec::Vec(const ::Vec v)
  : v(v) {}

Vec::~Vec() {
  CHECK(VecDestroy(&const_cast_(v)));
}

Ref<Vec> Vec::clone() const {
  ::Vec copy;
  CHECK(VecDuplicate(v,&copy));
  return new_<Vec>(copy);
}

void Vec::set(S alpha) {
  CHECK(VecSet(v,alpha));
}

}
using namespace hollow;

void wrap_vec() {
  typedef hollow::Vec Self;
  Class<Self>("Vec")
    .GEODE_METHOD(clone)
    .GEODE_METHOD(set)
    ;
}
