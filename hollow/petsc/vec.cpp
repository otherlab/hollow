// Wrapper around a petsc Vec

#include <hollow/petsc/vec.h>
#include <geode/python/Class.h>
#include <geode/random/counter.h>
#include <geode/utility/const_cast.h>
namespace hollow {

GEODE_DEFINE_TYPE(Vec)
typedef PetscReal T;
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

void Vec::waxpy(const S a, const Vec& x, const Vec& y) {
  const int n = local_size();
  GEODE_ASSERT(n==x.local_size() && n==y.local_size());
  CHECK(VecWAXPY(v,a,x.v,y.v));
}

T Vec::norm(NormType type) const {
  T norm;
  CHECK(VecNorm(v,type,&norm));
  return norm;
}

T Vec::min() const {
  T m;
  CHECK(VecMin(v,0,&m));
  return m;
}

T Vec::max() const {
  T m;
  CHECK(VecMax(v,0,&m));
  return m;
}

void Vec::set_random(const uint128_t key, const T slo, const T shi) {
  static_assert(boost::is_same<S,double>::value,"PetscScalar must be double for now");
  const T scale = (shi-slo)*pow(2.,-64.);
  int lo,hi;
  CHECK(VecGetOwnershipRange(v,&lo,&hi));
  S* p;
  CHECK(VecGetArray(v,&p));
  for (const int i : range(lo/2,(hi+1)/2)) {
    const auto bits = threefry(key,i);
    const int i0 = 2*i, i1 = i0+1;
    if (lo<=i0) p[i0-lo] = slo+scale*cast_uint128<uint64_t>(bits);
    if (i1< hi) p[i1-lo] = slo+scale*cast_uint128<uint64_t>(bits>>64);
  }
  CHECK(VecRestoreArray(v,&p));
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
    .GEODE_METHOD(set_random)
    ;
}
