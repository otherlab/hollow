// Wrapper around PetscFEM
#pragma once

#include <hollow/petsc/fe.h>
#include <geode/array/Array2d.h>
#include <petscfe.h>
#include <vector>
namespace hollow {

using std::vector;

// Physics definitions for petsc finite elements
// For details, see fem.pdf
struct Model : public Object {
  GEODE_DECLARE_TYPE(GEODE_NO_EXPORT) 
  typedef Object Base;
  typedef PetscReal T;
  typedef vector<Ref<const FE>> FEs;

  // Spatial dimension
  const int dim;

  // Interior, auxiliary, and boundary elements (per field)
  const FEs fe, fe_aux, fe_bd;
  const Array<PetscFE> fep, fep_aux, fep_bd; // Alternate views of fe et al.
public:
  PetscFEM fem;

  #define FE_ARGS const T u[], const T du[], const T a[], const T da[], const T x[]
  typedef void(*Interior)(FE_ARGS, T result[]);
  typedef void(*Boundary)(FE_ARGS, const T n[], T result[]);
  typedef void(*Exact)(const T x[], T* u, void* ctx);

  const Array<Interior> f0, f1; // Interior fields to dot with v and dv
  const Array<Boundary> b0, b1; // Boundary fields to dot with v and dv along the boundary
  const Array<Interior,2> g0, g1, g2, g3; // Unknown
  const Array<Exact> boundary; // Boundary conditions
  const Array<Exact> exact; // Exact result if available
  const Array<void*> boundary_contexts, exact_contexts; // Contexts for exact and boundary

protected:
  // Create a finite element model with special dimension dim,
  // and fields with component counts fields[0], fields[1], etc.
  Model(const FEs& fe, const FEs& fe_aux, const FEs& fe_bd);
public:
  ~Model();
};

}
