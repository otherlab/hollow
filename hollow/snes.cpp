// Wrapper around petsc SNES

#include <hollow/snes.h>
#include <hollow/mpi.h>
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
    ;
}