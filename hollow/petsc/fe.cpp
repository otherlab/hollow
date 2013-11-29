// Wrapper around PetscFE

#include <hollow/petsc/fe.h>
#include <hollow/petsc/mpi.h>
#include <geode/array/Array2d.h>
#include <geode/array/Subarray.h>
#include <geode/python/Class.h>
#include <geode/utility/Log.h>
namespace hollow {

using Log::cout;
using std::endl;
GEODE_DEFINE_TYPE(FE)
typedef PetscReal T;

FE::FE(const MPI_Comm comm, const int dim, const int components, const string& prefix, const int qorder) {
  GEODE_ASSERT(qorder,"qorder must positive for set or negative for default");

  // Create primal space
  PetscSpace P;
  CHECK(PetscSpaceCreate(comm,&P));
  CHECK(PetscObjectSetOptionsPrefix((PetscObject)P,prefix.c_str()));
  CHECK(PetscSpaceSetFromOptions(P));
  CHECK(PetscSpacePolynomialSetNumVariables(P,dim));
  CHECK(PetscSpaceSetUp(P));
  int order;
  CHECK(PetscSpaceGetOrder(P,&order));

  // Create a dual space of the same order
  PetscDualSpace Q;
  CHECK(PetscDualSpaceCreate(comm,&Q));
  CHECK(PetscObjectSetOptionsPrefix((PetscObject)Q,prefix.c_str()));
  DM K;
  CHECK(PetscDualSpaceCreateReferenceCell(Q,dim,PETSC_TRUE,&K));
  CHECK(PetscDualSpaceSetDM(Q,K));
  CHECK(DMDestroy(&K));
  CHECK(PetscDualSpaceSetOrder(Q,order));
  CHECK(PetscDualSpaceSetFromOptions(Q));
  CHECK(PetscDualSpaceSetUp(Q));

  // Create element
  CHECK(PetscFECreate(comm,&fe));
  CHECK(PetscObjectSetOptionsPrefix((PetscObject)fe,prefix.c_str()));
  CHECK(PetscFESetFromOptions(fe));
  CHECK(PetscFESetBasisSpace(fe,P));
  CHECK(PetscFESetDualSpace(fe,Q));
  CHECK(PetscFESetNumComponents(fe,components));

  // Don't need spaces anymore
  CHECK(PetscSpaceDestroy(&P));
  CHECK(PetscDualSpaceDestroy(&Q));

  /* Create quadrature (with specified order if given) */
  PetscQuadrature q;
  CHECK(PetscDTGaussJacobiQuadrature(dim,qorder>0?qorder:order,-1,1,&q));
  GEODE_ASSERT(q.numPoints,"Empty quadratures cause havoc elsewhere, need order at least 1");
  cout <<(prefix.size()?"fe ":"fe")<<prefix<<": dim "<<dim<<", quad ";
  if (dim==1) cout << RawArray<const T>(q.numPoints,q.points)<<endl;
  else if (dim==2) cout << RawArray<const Vector<T,2>>(q.numPoints,(const Vector<T,2>*)q.points)<<endl;
  else if (dim==3) cout << RawArray<const Vector<T,3>>(q.numPoints,(const Vector<T,3>*)q.points)<<endl;
  CHECK(PetscFESetQuadrature(fe,q));
}

FE::~FE() {
  CHECK(PetscFEDestroy(&fe));
}

int FE::basis_dimension() const {
  int dim;
  CHECK(PetscFEGetDimension(fe,&dim));
  return dim;
}

int FE::spatial_dimension() const {
  int dim;
  CHECK(PetscFEGetSpatialDimension(fe,&dim));
  return dim;
}

int FE::components() const {
  int c;
  CHECK(PetscFEGetNumComponents(fe,&c));
  return c;
}

Array<const int> FE::dofs() const {
  const int* dofs;
  CHECK(PetscFEGetNumDof(fe,&dofs));
  return RawArray<const int>(spatial_dimension()+1,dofs).copy();
}

int FE::qorder() const {
  PetscSpace P;
  CHECK(PetscFEGetBasisSpace(fe,&P));
  int qorder;
  CHECK(PetscSpaceGetOrder(P,&qorder));
  return qorder;
}

}
using namespace hollow;

void wrap_fe() {
  typedef FE Self;
  Class<Self>("FE")
    .GEODE_INIT(MPI_Comm,int,int,const string&,const int)
    .GEODE_GET(spatial_dimension)
    .GEODE_GET(basis_dimension)
    .GEODE_GET(components)
    .GEODE_GET(dofs)
    .GEODE_GET(qorder)
    ;
}
