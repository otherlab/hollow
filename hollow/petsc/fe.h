// Wrapper around PetscFE
#pragma once

#include <hollow/petsc/config.h>
#include <geode/array/Array.h>
#include <geode/python/Object.h>
#include <petscfe.h>
namespace hollow {

struct FE : public Object {
  GEODE_DECLARE_TYPE(GEODE_NO_EXPORT)
  typedef Object Base;

  PetscFE fe;

protected:
  // Create a petsc finite element
  //   dim: Spatial dimension
  //   components: Number of components
  //   prefix: Petsc options prefix
  //   qorder: Quadrature order of accuracy (-1 to set from options)
  FE(const MPI_Comm comm, const int dim, const int components, const string& prefix="", const int qorder=-1);
public:
  ~FE();

  int spatial_dimension() const;
  int basis_dimension() const; // Number of basis functions *per component*
  int components() const;
  Array<const int> dofs() const; // Degrees of freedom per cell in the complex (length dim()+1)
  int qorder() const;
};

}
