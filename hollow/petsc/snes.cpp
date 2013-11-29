// Wrapper around petsc SNES

#include <hollow/petsc/snes.h>
#include <hollow/petsc/mpi.h>
#include <geode/python/Class.h>
#include <geode/utility/const_cast.h>
namespace hollow {

GEODE_DEFINE_TYPE(SNES)

SNES::SNES(const MPI_Comm comm)
  : snes(0) {
  CHECK(SNESCreate(comm,&const_cast_(snes)));
}

SNES::~SNES() {
  CHECK(SNESDestroy(&const_cast_(snes)));
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

void SNES::residual(const Vec& x, Vec& f) const {
  CHECK(SNESComputeFunction(snes,x.v,f.v));
}

}
using namespace hollow;

void wrap_snes() {
  typedef hollow::SNES Self;
  Class<Self>("SNES")
    .GEODE_INIT(MPI_Comm)
    .GEODE_METHOD(set_from_options)
    .GEODE_METHOD(set_dm)
    .GEODE_METHOD(set_jacobian)
    .GEODE_METHOD(solve)
    .GEODE_GET(iterations)
    .GEODE_METHOD(residual)
    ;
}
