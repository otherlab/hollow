// Wrapper around a PetIGA IGA object
#pragma once

#include <hollow/petsc/ksp.h>
#include <hollow/petsc/snes.h>
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

  MPI_Comm comm() const;

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
  Array<const bool> periodic() const;
  void set_periodic(RawArray<const bool> p);
  IGAAxis axis(const int i) const;

  void set_from_options();
  virtual void set_up();
  void set_boundary_value(const int axis, const int side, const int field, const S value);
  void set_boundary_load(const int axis, const int side, const int field, const S value);
  void set_fix_table(const Vec& b);
  Ref<Vec> create_vec() const;
  Ref<Mat> create_mat() const;
  Ref<KSP> create_ksp() const;
  virtual Ref<SNES> create_snes() const;
  void compute_system(Mat& A, Vec& b);

  // Fill a vector from the given range of property fields
  Ref<Vec> create_property_vec(const int lo, const int hi) const;

  // Read and write geometry
  void read(const string& filename);
  void write(const string& filename) const;

  // Read and write solution vectors
  Ref<Vec> read_vec(const string& filename) const;
  void write_vec(const string& filename, const Vec& x) const;
};

}
