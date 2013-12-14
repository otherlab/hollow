// Wrapper around TaoSolver

#include <hollow/tao/solver.h>
#include <hollow/petsc/mpi.h>
#include <geode/python/Class.h>
#include <geode/utility/interrupts.h>
namespace hollow {

GEODE_DEFINE_TYPE(TaoSolver)

TaoSolver::TaoSolver(const MPI_Comm comm)
  : tao(0) {
  CHECK(TaoCreate(comm,&const_cast_(tao)));
  add_monitor(check_interrupts);
}

TaoSolver::~TaoSolver() {
  CHECK(TaoDestroy(&const_cast_(tao)));
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

typedef boost::function<void()> Monitor;

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
    .GEODE_METHOD(set_from_options)
    .GEODE_METHOD(set_initial_vector)
    .GEODE_METHOD(solve)
    .GEODE_METHOD(add_monitor)
    ;
}
