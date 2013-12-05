// Wrapper around a PetIGA IGA object
#pragma once

#include <hollow/petsc/ksp.h>
#include <petiga.h>
#include <vector>
namespace geode {
GEODE_DECLARE_ENUM(IGABasisType,HOLLOW_EXPORT)
} namespace hollow {
using std::vector;

struct IGA : public Object {
  GEODE_DECLARE_TYPE(HOLLOW_EXPORT)
  typedef Object Base;
  typedef PetscScalar S;

  const ::IGA iga;

protected:
  IGA(const MPI_Comm comm);
public:
  ~IGA();

  #define SIMPLE(type,name,Name) \
    type name() const { type name; CHECK(IGAGet##Name(iga,&name)); return name; } \
    void set_##name(type name) { CHECK(IGASet##Name(iga,name)); }
  SIMPLE(int,dim,Dim)
  SIMPLE(int,dof,Dof)
  SIMPLE(int,order,Order)
  SIMPLE(int,geometry_dim,GeometryDim)
  SIMPLE(int,property_dim,PropertyDim)
  #undef SIMPLE

  PetscBool rational() const {
    return iga->rational;
  }

  // Information about each axis
  Array<const int> bases() const;
  Array<const int> spans() const;
  Array<const int> degrees() const;
  vector<IGABasisType> basis_types() const;

  void set_from_options() {
    CHECK(IGASetFromOptions(iga));
  }

  virtual void set_up() {
    CHECK(IGASetUp(iga));
  }

  void set_boundary_value(const int axis, const int side, const int field, const S value) {
    GEODE_ASSERT(0<=axis && axis<dim());
    GEODE_ASSERT(0<=side && side<2);
    GEODE_ASSERT(0<=field && field<dof());
    CHECK(IGASetBoundaryValue(iga,axis,side,field,value));
  }

  Ref<Vec> create_vec() const {
    ::Vec vec;
    CHECK(IGACreateVec(iga,&vec)); 
    return new_<Vec>(vec);
  }

  Ref<Mat> create_mat() const {
    ::Mat mat;
    CHECK(IGACreateMat(iga,&mat)); 
    return new_<Mat>(mat);
  }

  Ref<KSP> create_ksp() const {
    ::KSP ksp;
    CHECK(IGACreateKSP(iga,&ksp));
    return new_<KSP>(ksp);
  }

  void compute_system(Mat& A, Vec& b) {
    CHECK(IGAComputeSystem(iga,A.m,b.v));
  }
};

}
