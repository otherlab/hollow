// A 2D Laplace test

#include <hollow/petsc/model.h>
#include <geode/python/Class.h>
#include <geode/python/stl.h>
#include <geode/utility/Log.h>
namespace hollow {

using Log::cout;
using std::endl;
typedef double T;
typedef Vector<T,2> TV;

/* A 2D Laplace test.  For Dirichlet conditions, we use exact solution:

    u = x^2 + y^2
    f = 4

  so that

    -\Delta u + f = -4 + 4 = 0

  For Neumann conditions, we have

    \nabla u \cdot -\hat y |_{y=0} = -(2y)|_{y=0} = 0 (bottom)
    \nabla u \cdot  \hat y |_{y=1} =  (2y)|_{y=1} = 2 (top)
    \nabla u \cdot -\hat x |_{x=0} = -(2x)|_{x=0} = 0 (left)
    \nabla u \cdot  \hat x |_{x=1} =  (2x)|_{x=1} = 2 (right)

  Which we can express as

    \nabla u \cdot  \hat n|_\Gamma = 2 (x + y)
*/
struct LaplaceTest2d : public Model {
  GEODE_DECLARE_TYPE(GEODE_NO_EXPORT)
  typedef Model Base;
  static const int d = 2;

  const bool neumann;
  const T zero;
  T shift;

protected:
  LaplaceTest2d(const FEs& fe, const FEs& fe_aux, const FEs& fe_bd, const bool neumann)
    : Base(fe,fe_aux,fe_bd)
    , neumann(neumann)
    , zero(0)
    , shift(0) {
    exact[0] = [](const T x[], T* u, void* ctx) {
      const T shift = *(const T*)ctx;
      *u = sqr(x[0])+sqr(x[1])+shift;
    };
    exact_contexts[0] = (void*)&shift;
    boundary[0] = exact[0]; // TODO: Make this the same only along the boundary?
    boundary_contexts[0] = (void*)&zero;
    #define CHECK_X() ({ \
      for (int i=0;i<d;i++) \
        GEODE_ASSERT(-1e-5<x[i] && x[i]<1+1e-5,format("evaluation outside box: %s",str(RawArray<const T>(d,x)))); \
      })
    #define CHECK_N() ({ \
      const Box<TV> box(TV(0,0),TV(1,1)); \
      const auto xv = vec(x[0],x[1]), nv = vec(n[0],n[1]); \
      GEODE_ASSERT(box.phi(xv)<1e-10,format("evaluation not on boundary: x %s",str(xv))); \
      GEODE_ASSERT(sqr_magnitude(box.normal(xv)-nv)<1e-10, \
        format("bad normal: x %s, n %s (expected %s)",str(xv),str(nv),str(box.normal(xv)))); \
      })
    f0[0] = [](FE_ARGS, T f0[]) { CHECK_X(); f0[0] = 4; };
    b0[0] = neumann ? [](FE_ARGS, const T n[], T f0[]) { CHECK_N(); f0[0] = abs(x[0]-1)<1e-5 || abs(x[1]-1)<1e-5 ? -2 : 0; }
                    : [](FE_ARGS, const T n[], T f0[]) { CHECK_N(); f0[0] = 0; };
    b1[0] = [](FE_ARGS, const T n[], T f1[]) {
      CHECK_X();
      for (int i=0;i<d;i++)
        f1[i] = 0;
    };
    f1[0] = [](FE_ARGS, T f1[]) {
      CHECK_X();
      for (int i=0;i<d;i++)
        f1[i] = du[i];
    };
    g3(0,0) = [](FE_ARGS, T g3[]) {
      CHECK_X();
      for (int i=0;i<d;i++)
        g3[i*d+i] = 1;
    };
  }
};

GEODE_DEFINE_TYPE(LaplaceTest2d)

#if 0
/*
  In 2D for Dirichlet conditions with a variable coefficient, we use exact solution:

    u  = x^2 + y^2
    f  = 6 (x + y)
    nu = (x + y)

  so that

    -\div \nu \grad u + f = -6 (x + y) + 6 (x + y) = 0
*/
void nu_2d(const T x[], PetscScalar *u)
{
  *u = x[0] + x[1];
}

void f0_analytic_u(const PetscScalar u[], const PetscScalar gradU[], const PetscScalar a[], const PetscScalar gradA[], const PetscReal x[], PetscScalar f0[])
{
  f0[0] = 6.0*(x[0] + x[1]);
}

/* gradU[comp*dim+d] = {u_x, u_y} or {u_x, u_y, u_z} */
void f1_analytic_u(const PetscScalar u[], const PetscScalar gradU[], const PetscScalar a[], const PetscScalar gradA[], const PetscReal x[], PetscScalar f1[])
{
  PetscInt d;

  for (d = 0; d < spatialDim; ++d) f1[d] = (x[0] + x[1])*gradU[d];
}
void f1_field_u(const PetscScalar u[], const PetscScalar gradU[], const PetscScalar a[], const PetscScalar gradA[], const PetscReal x[], PetscScalar f1[])
{
  PetscInt d;

  for (d = 0; d < spatialDim; ++d) f1[d] = a[0]*gradU[d];
}

/* < \nabla v, \nabla u + {\nabla u}^T >
   This just gives \nabla u, give the perdiagonal for the transpose */
void g3_analytic_uu(const PetscScalar u[], const PetscScalar gradU[], const PetscScalar a[], const PetscScalar gradA[], const PetscReal x[], PetscScalar g3[])
{
  PetscInt d;

  for (d = 0; d < spatialDim; ++d) g3[d*spatialDim+d] = x[0] + x[1];
}
void g3_field_uu(const PetscScalar u[], const PetscScalar gradU[], const PetscScalar a[], const PetscScalar gradA[], const PetscReal x[], PetscScalar g3[])
{
  PetscInt d;

  for (d = 0; d < spatialDim; ++d) g3[d*spatialDim+d] = a[0];
}
#endif

}
using namespace hollow;

void wrap_laplace() {
  typedef LaplaceTest2d Self;
  typedef Self::FEs FEs;
  Class<Self>("LaplaceTest2d")
    .GEODE_INIT(const FEs&,const FEs&,const FEs&,bool)
    .GEODE_FIELD(neumann)
    .GEODE_FIELD(shift)
    ;
}
