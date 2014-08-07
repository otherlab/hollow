// Wrapper around TaoSolver

#include <hollow/tao/solver.h>
#include <hollow/petsc/mpi.h>
#include <geode/python/Class.h>
#include <geode/utility/interrupts.h>
namespace hollow {

typedef PetscReal T;
GEODE_DEFINE_TYPE(TaoSolver)

TaoSolver::TaoSolver(const MPI_Comm comm)
  : tao(0) {
  CHECK(TaoCreate(comm,&const_cast_(tao)));
  add_monitor(check_interrupts);
}

TaoSolver::~TaoSolver() {
  CHECK(TaoDestroy(&const_cast_(tao)));
}

MPI_Comm TaoSolver::comm() const {
  return PetscObjectComm((PetscObject)tao);
}

void TaoSolver::set_from_options() {
  CHECK(TaoSetFromOptions(tao));
}

void TaoSolver::set_initial_vector(const Vec& x) {
  CHECK(TaoSetInitialVector(tao,x.v));
}

void TaoSolver::solve() {
  CHECK(TaoSolve(tao));
}

static PetscErrorCode snes_objective(::TaoSolver, ::Vec x, T* objective, void* ctx) {
  const TaoSolver& self = *(const TaoSolver*)ctx;
  CHECK(SNESComputeObjective(self.snes->snes,x,objective));
  return 0;
}
static PetscErrorCode snes_gradient(::TaoSolver, ::Vec x, ::Vec grad, void* ctx) {
  const TaoSolver& self = *(const TaoSolver*)ctx;
  CHECK(SNESComputeFunction(self.snes->snes,x,grad));
  return 0;
}
static PetscErrorCode snes_hessian(::TaoSolver, ::Vec x, ::Mat* A, ::Mat* P, MatStructure* flag, void* ctx) {
  const TaoSolver& self = *(const TaoSolver*)ctx;
  CHECK(SNESComputeJacobian(self.snes->snes,x,A,P,flag));
  return 0;
}

void TaoSolver::set_snes(const SNES& snes) {
  GEODE_ASSERT(snes.has_objective());
  this->snes = ref(snes);
  CHECK(TaoSetObjectiveRoutine(tao,snes_objective,(void*)this));
  CHECK(TaoSetGradientRoutine(tao,snes_gradient,(void*)this));
  ::Mat A, P;
  CHECK(SNESGetJacobian(snes.snes,&A,&P,0,0));
  GEODE_ASSERT(A && P);
  CHECK(TaoSetHessianRoutine(tao,A,P,snes_hessian,(void*)this));
}

typedef function<void()> Monitor;

void TaoSolver::add_monitor(const Monitor& monitor_) {
  const auto monitor = new Monitor(monitor_);
  const auto call = [](::TaoSolver, void* ctx) {
    (*(Monitor*)ctx)();
    return PetscErrorCode(0);
  };
  const auto del = [](void** ctx) {
    delete (Monitor*)*ctx;
    return PetscErrorCode(0);
  };
  CHECK(TaoSetMonitor(tao,call,(void*)monitor,del));
}

}
using namespace hollow;

void wrap_tao_solver() {
  typedef hollow::TaoSolver Self;
  Class<Self>("TaoSolver")
    .GEODE_INIT(MPI_Comm)
    .GEODE_GET(comm)
    .GEODE_METHOD(set_from_options)
    .GEODE_METHOD(set_initial_vector)
    .GEODE_METHOD(solve)
    .GEODE_METHOD(set_snes)
    .GEODE_METHOD(add_monitor)
    ;
}
