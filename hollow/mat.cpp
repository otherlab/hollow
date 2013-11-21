// Wrapper around a petsc Mat

#include <hollow/mat.h>
#include <hollow/mpi.h>
#include <geode/python/Class.h>
#include <geode/utility/const_cast.h>
namespace hollow {

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

}
using namespace hollow;

void wrap_mat() {
  typedef hollow::Mat Self;
  Class<Self>("Mat")
    .GEODE_GET(comm)
    .GEODE_METHOD(set_constant_nullspace)
    ;
}
