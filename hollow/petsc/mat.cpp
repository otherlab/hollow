// Wrapper around a petsc Mat

#include <hollow/petsc/mat.h>
#include <hollow/petsc/mpi.h>
#include <geode/array/Array2d.h>
#include <geode/python/Class.h>
#include <geode/python/enum.h>
#include <geode/utility/const_cast.h>
namespace hollow {

typedef PetscScalar S;
GEODE_DEFINE_TYPE(Mat)

Mat::Mat(const ::Mat m)
  : m(m) {}

Mat::~Mat() {
  CHECK(MatDestroy(&const_cast_(m)));
}

MPI_Comm Mat::comm() const {
  return PetscObjectComm((PetscObject)m);
}

void Mat::set_constant_nullspace() {
  MatNullSpace null;
  CHECK(MatNullSpaceCreate(comm(),PETSC_TRUE,0,0,&null));
  CHECK(MatSetNullSpace(m,null));
  CHECK(MatNullSpaceDestroy(&null));
}

HOLLOW_EXPORT Array<S,2> dense_copy(::Mat A) {
  Vector<int,2> ln, gn;
  CHECK(MatGetSize     (A,&gn.x,&gn.y));
  CHECK(MatGetLocalSize(A,&ln.x,&ln.y));
  GEODE_ASSERT(ln==gn);
  Array<S,2> D(gn,uninit);
  CHECK(MatGetValues(A,gn.x,arange(gn.x).copy().data(),
                       gn.y,arange(gn.y).copy().data(),D.data()));
  return D;
}

}
using namespace hollow;

void wrap_mat() {
  typedef hollow::Mat Self;
  Class<Self>("Mat")
    .GEODE_GET(comm)
    .GEODE_METHOD(set_constant_nullspace)
    ;
}
