// Wrapper around TaoSolver

#include <hollow/tao/solver.h>
#include <hollow/petsc/mpi.h>
#include <geode/python/Class.h>
namespace hollow {

GEODE_DEFINE_TYPE(TaoSolver)

TaoSolver::TaoSolver(const MPI_Comm comm)
  : tao(0) {
  CHECK(TaoCreate(comm,&const_cast_(tao)));
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

}
using namespace hollow;

void wrap_tao_solver() {
  typedef hollow::TaoSolver Self;
  Class<Self>("TaoSolver")
    .GEODE_INIT(MPI_Comm)
    .GEODE_METHOD(set_from_options)
    .GEODE_METHOD(set_initial_vector)
    .GEODE_METHOD(solve)
    ;
}
