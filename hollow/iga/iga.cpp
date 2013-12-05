// Wrapper around a PetIGA IGA object

#include <hollow/iga/iga.h>
#include <hollow/petsc/mpi.h>
#include <geode/array/convert.h>
#include <geode/python/Class.h>
#include <geode/python/enum.h>
namespace geode {
GEODE_DEFINE_ENUM(IGABasisType,HOLLOW_EXPORT)
} namespace hollow {

GEODE_DEFINE_TYPE(IGA)

IGA::IGA(const MPI_Comm comm)
  : iga(0) {
  CHECK(IGACreate(comm,&const_cast_(iga)));
}

IGA::~IGA() {
  CHECK(IGADestroy(&const_cast_(iga)));
}

Array<const int> IGA::bases() const {
  Array<int> bases(dim());
  for (const int i : range(dim())) {
    IGAAxis axis;
    CHECK(IGAGetAxis(iga,i,&axis));
    CHECK(IGAAxisGetSizes(axis,0,&bases[i]));
  }
  return bases;
}

Array<const int> IGA::spans() const {
  Array<int> spans(dim());
  for (const int i : range(dim())) {
    IGAAxis axis;
    CHECK(IGAGetAxis(iga,i,&axis));
    CHECK(IGAAxisGetSizes(axis,&spans[i],0));
  }
  return spans;
}

Array<const int> IGA::degrees() const {
  Array<int> degrees(dim());
  for (const int i : range(dim())) {
    IGAAxis axis;
    CHECK(IGAGetAxis(iga,i,&axis));
    CHECK(IGAAxisGetDegree(axis,&degrees[i]));
  }
  return degrees;
}

vector<IGABasisType> IGA::basis_types() const {
  vector<IGABasisType> types(dim());
  for (const int i : range(dim()))
    types[i] = iga->basis[i]->type;
  return types;
}

}
using namespace hollow;

void wrap_iga() {
  typedef hollow::IGA Self;
  Class<Self>("IGA")
    .GEODE_INIT(MPI_Comm)
    .GEODE_GETSET(dim)
    .GEODE_GETSET(dof)
    .GEODE_GETSET(order)
    .GEODE_GETSET(geometry_dim)
    .GEODE_GETSET(property_dim)
    .GEODE_GET(rational)
    .GEODE_GET(bases)
    .GEODE_GET(spans)
    .GEODE_GET(degrees)
    .GEODE_GET(basis_types)
    .GEODE_METHOD(set_from_options)
    .GEODE_METHOD(set_up)
    .GEODE_METHOD(set_boundary_value)
    .GEODE_METHOD(create_vec)
    .GEODE_METHOD(create_mat)
    .GEODE_METHOD(create_ksp)
    .GEODE_METHOD(compute_system)
    ;

  GEODE_ENUM(IGABasisType)
  GEODE_ENUM_VALUE(IGA_BASIS_BSPLINE)
  GEODE_ENUM_VALUE(IGA_BASIS_BERNSTEIN)
  GEODE_ENUM_VALUE(IGA_BASIS_LAGRANGE)
  GEODE_ENUM_VALUE(IGA_BASIS_HIERARCHICAL)
}
