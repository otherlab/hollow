// Wrapper around a petsc Mat

#include <hollow/petsc/mat.h>
#include <hollow/petsc/mpi.h>
#include <geode/python/Class.h>
#include <geode/python/enum.h>
#include <geode/utility/const_cast.h>
namespace geode {
GEODE_DEFINE_ENUM(MatStructure,HOLLOW_EXPORT)
} namespace hollow {

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

  GEODE_ENUM(MatStructure)
  GEODE_ENUM_VALUE(DIFFERENT_NONZERO_PATTERN)
  GEODE_ENUM_VALUE(SUBSET_NONZERO_PATTERN)
  GEODE_ENUM_VALUE(SAME_NONZERO_PATTERN)
  GEODE_ENUM_VALUE(SAME_PRECONDITIONER)
}
