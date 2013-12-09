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

MPI_Comm IGA::comm() const {
  return PetscObjectComm((PetscObject)iga);
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

void IGA::set_from_options() {
  CHECK(IGASetFromOptions(iga));
}

void IGA::set_up() {
  CHECK(IGASetUp(iga));
}

void IGA::set_boundary_value(const int axis, const int side, const int field, const S value) {
  GEODE_ASSERT(0<=axis && axis<dim());
  GEODE_ASSERT(0<=side && side<2);
  GEODE_ASSERT(0<=field && field<dof());
  CHECK(IGASetBoundaryValue(iga,axis,side,field,value));
}

void IGA::set_boundary_load(const int axis, const int side, const int field, const S value) {
  GEODE_ASSERT(0<=axis && axis<dim());
  GEODE_ASSERT(0<=side && side<2);
  GEODE_ASSERT(0<=field && field<dof());
  CHECK(IGASetBoundaryLoad(iga,axis,side,field,value));
}

Ref<Vec> IGA::create_vec() const {
  ::Vec vec;
  CHECK(IGACreateVec(iga,&vec)); 
  return new_<Vec>(vec);
}

Ref<Mat> IGA::create_mat() const {
  ::Mat mat;
  CHECK(IGACreateMat(iga,&mat)); 
  return new_<Mat>(mat);
}

Ref<KSP> IGA::create_ksp() const {
  ::KSP ksp;
  CHECK(IGACreateKSP(iga,&ksp));
  return new_<KSP>(ksp);
}

Ref<SNES> IGA::create_snes() const {
  ::SNES snes;
  CHECK(IGACreateSNES(iga,&snes));
  return new_<SNES>(snes);
}

void IGA::compute_system(Mat& A, Vec& b) {
  CHECK(IGAComputeSystem(iga,A.m,b.v));
}

void IGA::write(const string& filename) const {
  CHECK(IGAWrite(iga,filename.c_str()));
}

void IGA::write_vec(const string& filename, const Vec& x) const {
  CHECK(IGAWriteVec(iga,x.v,filename.c_str()));
}

}
using namespace hollow;

void wrap_iga() {
  typedef hollow::IGA Self;
  Class<Self>("IGA")
    .GEODE_INIT(MPI_Comm)
    .GEODE_GET(comm)
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
    .GEODE_METHOD(set_boundary_load)
    .GEODE_METHOD(create_vec)
    .GEODE_METHOD(create_mat)
    .GEODE_METHOD(create_ksp)
    .GEODE_METHOD(create_snes)
    .GEODE_METHOD(compute_system)
    .GEODE_METHOD(write)
    .GEODE_METHOD(write_vec)
    ;

  GEODE_ENUM(IGABasisType)
  GEODE_ENUM_VALUE(IGA_BASIS_BSPLINE)
  GEODE_ENUM_VALUE(IGA_BASIS_BERNSTEIN)
  GEODE_ENUM_VALUE(IGA_BASIS_LAGRANGE)
  GEODE_ENUM_VALUE(IGA_BASIS_HIERARCHICAL)
}
