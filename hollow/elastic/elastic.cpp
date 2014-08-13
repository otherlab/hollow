// IGA Elasticity solver

#include <hollow/elastic/laplace.h>
#include <hollow/elastic/neo_hookean.h>
#include <hollow/iga/iga.h>
#include <hollow/tao/tao.h>
#include <hollow/petsc/mpi.h>
#include <geode/array/NdArray.h>
#include <geode/math/constants.h>
#include <geode/python/Class.h>
#include <geode/python/stl.h>
#include <geode/utility/Log.h>
namespace hollow {
namespace {

using Log::cout;
using std::endl;

// See hollow/doc/fem.tex for details.

template<class Material> struct ElasticIGA : public IGA {
  GEODE_DECLARE_TYPE(GEODE_NO_EXPORT)
  typedef ElasticIGA Self;
  typedef IGA Base;
  static const int d = Material::d;
  typedef PetscReal T;
  typedef Vector<T,d> TV;

  const Material material;
  const TV rho_g;

protected:
  ElasticIGA(const MPI_Comm comm, RawArray<const T> material, const TV rho_g)
    : IGA(comm)
    , material(material)
    , rho_g(rho_g) {
    set_dim(d);
    set_dof(d);
    set_order(1); // Only need first derivatives
  }
public:

  void set_up() override {
    GEODE_ASSERT(dim()==d);
    CHECK(IGASetFormFunction(iga,residual,(void*)this));
    CHECK(IGASetFormJacobian(iga,jacobian,(void*)this));
    Base::set_up();
  }

  Ref<SNES> create_snes() const override {
    const auto snes = Base::create_snes();
    // For now, don't use energy as the objective.
    // Optimization problems should create_tao below anyways.
    if (0)
      CHECK(SNESSetObjective(snes->snes,objective<::SNES>,(void*)this));
    return snes;
  }

  Ref<Tao> create_tao() const {
    const auto tao = new_<Tao>(comm());
    CHECK(TaoSetType(tao->tao,"nls"));
    const auto gradient = [](::Tao, ::Vec u, ::Vec grad, void* ctx) {
      const Self& self = *(const Self*)ctx;
      CHECK(IGAComputeFunction(self.iga,u,grad));
      return PetscErrorCode(0);
    };
    const auto hessian = [](::Tao, ::Vec u, ::Mat A, ::Mat P, void* ctx) {
      const Self& self = *(const Self*)ctx;
      GEODE_ASSERT(A==P);
      CHECK(IGAComputeJacobian(self.iga,u,A));
      PetscFunctionReturn(0);
    };
    CHECK(TaoSetObjectiveRoutine(tao->tao,objective<::Tao>,(void*)this));
    CHECK(TaoSetGradientRoutine(tao->tao,gradient,(void*)this));
    const auto A = create_mat();
    CHECK(TaoSetHessianRoutine(tao->tao,A->m,A->m,hessian,(void*)this));
    return tao;
  }

  template<class Ignore> static PetscErrorCode objective(Ignore, ::Vec x, T* energy, void* ctx) {
    const Self& self = *(const Self*)ctx;
    HOLLOW_MONITOR_J(self.material.J_range = Box<T>::empty_box());
    CHECK(IGAComputeScalarCustom(self.iga,x,1,energy,Self::energy,ctx,PETSC_TRUE));
    if (!isfinite(*energy)) // Tao doesn't like infinite energies
      *energy = 1e10;
    HOLLOW_MONITOR_J(cout << "J_range = "<<self.material.J_range<<endl);
    if (self.rho_g == TV())
      GEODE_ASSERT(*energy >= -1e-10);
    return PetscErrorCode(0);
  }

  TV X(IGAPoint p, const T* x_) const {
    TV X;
    IGAPointFormValue(p,x_,X.data());
    return X;
  }

  Matrix<T,d> F(IGAPoint p, const T* x_) const {
    Matrix<T,d> F;
    IGAPointFormGrad(p,x_,F.data());
    return F;
  }

  static PetscErrorCode energy(IGAPoint p, const T* x, const int n, T* energy, void* ctx) {
    assert(n==1);
    const Self& self = *(const Self*)ctx;
    *energy = self.material.energy(self.F(p,x));
    if (self.rho_g != TV())
      *energy -= dot(self.rho_g,self.X(p,x));
    return 0;
  }

  static PetscErrorCode residual(IGAPoint p, const T* x, T* b_, void* ctx) {
    const Self& self = *(const Self*)ctx;
    const int nen = p->nen;
    const T*  N0 = (decltype(N0))p->shape[0];
    const TV* N1 = (decltype(N1))p->shape[1];
    const auto F = self.F(p,x);
    const auto rho_g = self.rho_g;
    const auto P = self.material.stress(F);
    TV* b = (TV*)b_;
    for (int a=0;a<nen;a++)
      b[a] = P*N1[a]; // elasticity
    if (self.rho_g != TV())
      for (int a=0;a<nen;a++)
        b[a] -= rho_g*N0[a]; // gravity
    return 0;
  }

  static PetscErrorCode jacobian(IGAPoint p, const T* x, T* A_, void* ctx) {
    const Self& self = *(const Self*)ctx;
    const int nen = p->nen;
    const T*  N0 = (decltype(N0))p->shape[0];
    const TV* N1 = (decltype(N1))p->shape[1];
    const auto F = self.F(p,x);
    TV* A = (decltype(A))A_;
    for (int a=0;a<nen;a++)
      for (int i=0;i<d;i++) {
        Matrix<T,d> dF;
        dF[i] = N1[a];
        const auto dP = self.material.differential(F,dF);
        for (int b=0;b<nen;b++)
          A[(a*d+i)*nen+b] = dP*N1[b];
      }
    return 0;
  }
};

#if 0
template<class Material> struct ElasticModel : public Model {
  GEODE_DECLARE_TYPE(GEODE_NO_EXPORT)
  typedef ElasticModel Self;
  typedef Model Base;
  static const int d = Material::d;
  typedef PetscReal T;
  typedef Vector<T,d> TV;

  // Boundary condition analytic
  const Ref<const Analytic> boundary;

  // For now, we make these static to allow model functions access without this.
  // This should be replaced with material fields reasonably soon.
  static bool active; // Since these are static, only one can be active at any given time
  static Material material;
  static TV minus_rho_g;

protected:
  ElasticModel(const FEs& fe, const FEs& fe_aux, const FEs& fe_bd,
               const Analytic& boundary, RawArray<const T> material, const TV rho_g)
    : Base(fe,fe_aux,fe_bd)
    , boundary(ref(boundary)) {
    // Check consistency
    GEODE_ASSERT(fe.size()==1);
    GEODE_ASSERT(fe_bd.size()<=1);
    GEODE_ASSERT(fe_aux.size()==0);
    GEODE_ASSERT(fe[0]->spatial_dimension()==d);
    GEODE_ASSERT(fe[0]->components()==d);
    if (fe_bd.size()) {
      GEODE_ASSERT(fe_bd[0]->spatial_dimension()==d);
      GEODE_ASSERT(fe_bd[0]->components()==d);
    }

    // Set boundary conditions
    GEODE_ASSERT(boundary.dim==d);
    GEODE_ASSERT(boundary.count==d);
    Base::boundary[0] = boundary.f;
    boundary_contexts[0] = boundary.ctx;

    // Set static variables
    GEODE_ASSERT(!active,"Only one ElasticModel can exist at any given time");
    active = true;
    Self::material = Material(material);
    Self::minus_rho_g = -rho_g;

    // Helper functions for PDE definition
    #define GRAB_X \
      TV X; \
      for (int i=0;i<d;i++) \
        X[i] = u[i];
    #define GRAB_F \
      Matrix<T,d> F; \
      for (int i=0;i<d;i++) \
        for (int j=0;j<d;j++) \
          F(i,j) = du[i*d+j];

    // Define PDE.
    e = [](FE_ARGS, T* e) {
      GRAB_X GRAB_F
      *e = Self::material.energy(F)+dot(minus_rho_g,X);
    };
    f0[0] = [](FE_ARGS, T f0[d]) {
      for (int i=0;i<d;i++)
        f0[i] = minus_rho_g[i];
    };
    if (0) { // These would apply once we have neumann.  For now, leave them empty.
      b0[0] = [](FE_ARGS, const T n[d], T f0[d]) {
        for (int i=0;i<d;i++)
          f0[i] = 0;
      };
      b1[0] = [](FE_ARGS, const T n[d], T f1[d*d]) {
        for (int ij=0;ij<d*d;ij++)
          f1[ij] = 0;
      };
    }
    f1[0] = [](FE_ARGS, T f1[d*d]) {
      GRAB_F
      const auto P = Self::material.stress(F);
      for (int i=0;i<d;i++)
        for (int j=0;j<d;j++)
          f1[i*d+j] = P(i,j);
    };
    g3(0,0) = [](FE_ARGS, T g3[d*d*d*d]) {
      GRAB_F
      Self::material.derivative(g3,F);
      GRAB_X
    };
  }
public:
  ~ElasticModel() {
    active = false;
  }
};

template<class Material> bool ElasticModel<Material>::active = false;
template<class Material> Material ElasticModel<Material>::material(uninit);
template<class Material> typename ElasticModel<Material>::TV ElasticModel<Material>::minus_rho_g;
#endif

template<> GEODE_DEFINE_TYPE(ElasticIGA<LaplaceCM<2>>)
template<> GEODE_DEFINE_TYPE(ElasticIGA<NeoHookean<2>>)
template<> GEODE_DEFINE_TYPE(ElasticIGA<NeoHookean<3>>)
#if 0
template<> GEODE_DEFINE_TYPE(ElasticModel<LaplaceCM<2>>)
template<> GEODE_DEFINE_TYPE(ElasticModel<NeoHookean<2>>)
template<> GEODE_DEFINE_TYPE(ElasticModel<NeoHookean<3>>)
#endif

}
}
using namespace hollow;

template<class Material> static void wrap_helper(const char* material) {
  static const int d = Material::d;
  typedef typename Material::T T;
  typedef Vector<T,d> TV;
  {
    static const auto name = format("%sElasticIGA%dd",material,d);
    typedef ElasticIGA<Material> Self;
    Class<Self>(name.c_str())
      .GEODE_INIT(MPI_Comm,RawArray<const T>,TV)
      .GEODE_METHOD(create_tao)
      ;
  } {
#if 0
    static const auto name = format("%sElasticModel%dd",material,d);
    typedef ElasticModel<Material> Self;
    typedef typename Self::FEs FEs;
    Class<Self>(name.c_str())
      .GEODE_INIT(const FEs&,const FEs&,const FEs&,const Analytic&,RawArray<const T>,const TV)
      ;
#endif
  }
}

void wrap_elastic() {
  wrap_helper<LaplaceCM<2>>("Laplace");
  wrap_helper<NeoHookean<2>>("NeoHookean");
  wrap_helper<NeoHookean<3>>("NeoHookean");
}
