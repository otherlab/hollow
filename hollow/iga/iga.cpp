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

IGAAxis IGA::axis(const int i) const {
  GEODE_ASSERT(unsigned(i)<unsigned(dim()));
  IGAAxis axis;
  CHECK(IGAGetAxis(iga,i,&axis));
  return axis;
}

Array<const int> IGA::bases() const {
  Array<int> bases(dim());
  for (const int i : range(dim()))
    CHECK(IGAAxisGetSizes(axis(i),0,&bases[i]));
  return bases;
}

Array<const int> IGA::spans() const {
  Array<int> spans(dim());
  for (const int i : range(dim()))
    CHECK(IGAAxisGetSizes(axis(i),&spans[i],0));
  return spans;
}

Array<const int> IGA::degrees() const {
  Array<int> degrees(dim());
  for (const int i : range(dim()))
    CHECK(IGAAxisGetDegree(axis(i),&degrees[i]));
  return degrees;
}

vector<IGABasisType> IGA::basis_types() const {
  vector<IGABasisType> types(dim());
  for (const int i : range(dim()))
    types[i] = iga->basis[i]->type;
  return types;
}

Array<const bool> IGA::periodic() const {
  Array<bool> periodic(dim());
  for (const int i : range(dim())) {
    PetscBool p;
    CHECK(IGAAxisGetPeriodic(axis(i),&p));
    periodic[p] = bool(p);
  }
  return periodic;
}

void IGA::set_periodic(RawArray<const bool> p) {
  GEODE_ASSERT(p.size()==dim());
  for (const int i : range(dim()))
    CHECK(IGAAxisSetPeriodic(axis(i),p[i]?PETSC_TRUE:PETSC_FALSE));
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

void IGA::set_fix_table(const Vec& b) {
  CHECK(IGASetFixTable(iga,b.v));
}

Ref<Vec> IGA::create_vec() const {
  GEODE_ASSERT(iga->setup);
  ::Vec vec;
  CHECK(IGACreateVec(iga,&vec)); 
  return new_<Vec>(vec);
}

Ref<Mat> IGA::create_mat() const {
  GEODE_ASSERT(iga->setup);
  ::Mat mat;
  CHECK(IGACreateMat(iga,&mat)); 
  return new_<Mat>(mat);
}

Ref<KSP> IGA::create_ksp() const {
  GEODE_ASSERT(iga->setup);
  ::KSP ksp;
  CHECK(IGACreateKSP(iga,&ksp));
  return new_<KSP>(ksp);
}

Ref<SNES> IGA::create_snes() const {
  GEODE_ASSERT(iga->setup);
  ::SNES snes;
  CHECK(IGACreateSNES(iga,&snes));
  return new_<SNES>(snes);
}

void IGA::compute_system(Mat& A, Vec& b) {
  CHECK(IGAComputeSystem(iga,A.m,b.v));
}

Ref<Vec> IGA::create_property_vec(const int lo, const int hi) const {
  const int dof = this->dof();
  GEODE_ASSERT(hi-lo==dof);
  const int pdim = property_dim();
  GEODE_ASSERT(0<=lo && hi<=pdim);
  const auto x = create_vec();
  const int n = x->local_size();

  GEODE_ASSERT(lo==0 && hi==dof); // TODO: Make more general

  S* p;
  CHECK(VecGetArray(x->v,&p));
  memcpy(p,iga->propertyA,sizeof(S)*n);
  CHECK(VecRestoreArray(x->v,&p));
  return x;
}

void IGA::read(const string& filename) {
  CHECK(IGARead(iga,filename.c_str()));
}

void IGA::write(const string& filename) const {
  CHECK(IGAWrite(iga,filename.c_str()));
}

Ref<Vec> IGA::read_vec(const string& filename) const {
  const auto x = create_vec();
  CHECK(IGAReadVec(iga,x->v,filename.c_str()));
  return x;
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
    .GEODE_GETSET(periodic)
    .GEODE_METHOD(set_from_options)
    .GEODE_METHOD(set_up)
    .GEODE_METHOD(set_boundary_value)
    .GEODE_METHOD(set_boundary_load)
    .GEODE_METHOD(set_fix_table)
    .GEODE_METHOD(create_vec)
    .GEODE_METHOD(create_mat)
    .GEODE_METHOD(create_ksp)
    .GEODE_METHOD(create_snes)
    .GEODE_METHOD(compute_system)
    .GEODE_METHOD(create_property_vec)
    .GEODE_METHOD(read)
    .GEODE_METHOD(read_vec)
    .GEODE_METHOD(write)
    .GEODE_METHOD(write_vec)
    ;

  GEODE_ENUM(IGABasisType)
  GEODE_ENUM_VALUE(IGA_BASIS_BSPLINE)
  GEODE_ENUM_VALUE(IGA_BASIS_BERNSTEIN)
  GEODE_ENUM_VALUE(IGA_BASIS_LAGRANGE)
  GEODE_ENUM_VALUE(IGA_BASIS_HIERARCHICAL)
}
