// Wrapper around petsc SNES

#include <hollow/petsc/snes.h>
#include <hollow/petsc/mpi.h>
#include <geode/python/Class.h>
#include <geode/utility/const_cast.h>
#include <geode/utility/Log.h>
#include <geode/utility/interrupts.h>
#include <geode/vector/relative_error.h>
namespace hollow {

GEODE_DEFINE_TYPE(SNES)
typedef PetscReal T;
using Log::cout;
using std::endl;

SNES::SNES(const MPI_Comm comm)
  : snes(0) {
  CHECK(SNESCreate(comm,&const_cast_(snes)));
  add_monitor([](int,T){check_interrupts();});
}

SNES::SNES(const ::SNES snes)
  : snes(snes) {
  add_monitor([](int,T){check_interrupts();});
}

SNES::~SNES() {
  CHECK(SNESDestroy(&const_cast_(snes)));
}

MPI_Comm SNES::comm() const {
  return PetscObjectComm((PetscObject)snes);
}

void SNES::set_from_options() {
  CHECK(SNESSetFromOptions(snes));
}

void SNES::set_dm(const DM& dm) {
  CHECK(SNESSetDM(snes,dm.dm));
}

void SNES::set_jacobian(const Mat& A, const Mat& P) {
  CHECK(SNESSetJacobian(snes,A.m,P.m,0,0));
}

void SNES::solve(Ptr<const Vec> b, Vec& x) {
  CHECK(SNESSolve(snes,b?b->v:0,x.v));
}

int SNES::iterations() const {
  int n;
  CHECK(SNESGetIterationNumber(snes,&n));
  return n;
}

T SNES::objective(const Vec& x) const {
  T ob;
  CHECK(SNESComputeObjective(snes,x.v,&ob));
  return ob;
}

void SNES::residual(const Vec& x, Vec& f) const {
  CHECK(SNESComputeFunction(snes,x.v,f.v));
}

bool SNES::has_objective() const {
  PetscErrorCode (*objective)(::SNES,::Vec,T*,void*);
  CHECK(SNESGetObjective(snes,&objective,0));
  return objective!=0;
}

bool SNES::has_jacobian() const {
  PetscErrorCode (*jacobian)(::SNES,::Vec,::Mat*,::Mat*,MatStructure*,void*);
  CHECK(SNESGetJacobian(snes,0,0,&jacobian,0));
  return jacobian!=0;
}

void SNES::consistency_test(const Vec& x, const T small, const T rtol, const T atol, const int steps) const {
  const T log_small = log(small);
  const auto dx = x.clone(),
             y  = x.clone(),
             fm = x.clone(),
             fp = x.clone(),
             df = x.clone();
  const bool has_objective = this->has_objective(),
             has_jacobian = this->has_jacobian();
  GEODE_ASSERT(has_objective || has_jacobian);

  // Compute information at x
  const auto f = x.clone();
  residual(x,f);
  ::KSP ksp;
  ::Mat A,P;MatStructure flag;
  if (has_jacobian) {
    CHECK(SNESGetKSP(snes,&ksp));
    CHECK(KSPGetOperators(ksp,&A,&P,&flag));
    CHECK(SNESComputeJacobian(snes,x.v,&A,&P,&flag));
  }

  // Try a variety of differential shifts
  for (int step=0;step<steps;step++) {
    // Compute an exponentially scaled random value to reduce the chance of accidental success
    dx->set_random(step,-1,1);
    T* p;
    CHECK(VecGetArray(dx->v,&p));
    for (auto& d : RawArray<T>(dx->local_size(),p))
      d = copysign(exp(log_small*(2-abs(d))),d); // Exponentially distributed between small and small^2
    CHECK(VecRestoreArray(dx->v,&p));

    // Evaluate information at x-dx,x+dx
    T Up,Um;
    y->waxpy(1,dx,x);
    if (has_objective)
      Up = objective(y);
    residual(y,fp);
    y->waxpy(-1,dx,x);
    if (has_objective)
      Um = objective(y);
    residual(y,fm);

    // Check objective
    if (has_objective) {
      const T d = 2*dot(f,dx);
      const T aerror = abs((Up-Um)-d),
              rerror = relative_error(Up-Um,d);
      cout << "snes objective/residual error = "<<rerror<<" "<<aerror<<" ("<<(Up-Um)<<" vs. "<<d<<")"<<endl;
      GEODE_ASSERT(rerror<rtol || aerror<atol);
    }

    if (has_jacobian) {
      // Check residual vs. jacobian
      fp->axpy(-1,fm); // fp = fp-fm
      CHECK(MatMult(A,dx->v,df->v));
      const T fp_norm = fp->norm(NORM_MAX),
              df_norm = 2*df->norm(NORM_MAX),
              scale = max(fp_norm,df_norm,1e-30);
      fp->axpy(-2,df);
      const T aerror = fp->norm(NORM_MAX),
              rerror = aerror/scale;
      cout << "snes residual/jacobian error = "<<rerror<<" "<<aerror
           <<" (fp norm "<<fp_norm<<", df norm "<<df_norm<<")"<<endl;
      GEODE_ASSERT(rerror<rtol,"residual/jacobian error");
    }
  }
}

typedef boost::function<void(int,T)> Monitor;

void SNES::add_monitor(const Monitor& monitor_) {
  const auto monitor = new Monitor(monitor_);
  const auto call = [](::SNES, int iter, T rnorm, void* ctx) {
    (*(Monitor*)ctx)(iter,rnorm);
    return PetscErrorCode(0);
  };
  const auto del = [](void** ctx) {
    delete (Monitor*)*ctx;
    return PetscErrorCode(0);
  };
  CHECK(SNESMonitorSet(snes,call,(void*)monitor,del));
}

}
using namespace hollow;

void wrap_snes() {
  typedef hollow::SNES Self;
  Class<Self>("SNES")
    .GEODE_INIT(MPI_Comm)
    .GEODE_GET(comm)
    .GEODE_METHOD(set_from_options)
    .GEODE_METHOD(set_dm)
    .GEODE_METHOD(set_jacobian)
    .GEODE_METHOD(solve)
    .GEODE_GET(iterations)
    .GEODE_METHOD(residual)
    .GEODE_METHOD(consistency_test)
    .GEODE_METHOD(add_monitor)
    .GEODE_METHOD(has_objective)
    ;
}
