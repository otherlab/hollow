// Wrapper around a petsc KSP

#include <hollow/petsc/ksp.h>
#include <hollow/petsc/mpi.h>
#include <geode/python/Class.h>
namespace hollow {

GEODE_DEFINE_TYPE(KSP)

KSP::KSP(const ::KSP ksp)
  : ksp(ksp) {}

KSP::~KSP() {
  CHECK(KSPDestroy(&const_cast_(ksp)));
}

MPI_Comm KSP::comm() const {
  return PetscObjectComm((PetscObject)ksp);
}

void KSP::set_operators(Mat& A, Mat& P) {
  CHECK(KSPSetOperators(ksp,A.m,P.m));
}

void KSP::set_from_options() {
  CHECK(KSPSetFromOptions(ksp));
}

void KSP::solve(const Vec& b, Vec& x) {
  CHECK(KSPSolve(ksp,b.v,x.v));
}

string KSP::report() const {
  // Look up ksp information
  KSPConvergedReason reason;CHECK(KSPGetConvergedReason(ksp,&reason));
  const char* ksptype;CHECK(KSPGetType(ksp,&ksptype));
  PC pc;CHECK(KSPGetPC(ksp,&pc));
  const char* pctype;CHECK(PCGetType(pc,&pctype));
  T error;CHECK(KSPGetResidualNorm(ksp,&error));
  T rtol,atol,dtol;int maxits;CHECK(KSPGetTolerances(ksp,&rtol,&atol,&dtol,&maxits));
  int its;CHECK(KSPGetIterationNumber(ksp,&its));

  // Say what happened
  std::ostringstream s;
  s<<ksptype<<' '<<pctype<<(reason<0?" diverged":" converged")<<": iterations = "<<its<<", ";
  if (reason==KSP_DIVERGED_INDEFINITE_PC)
    s<<"indefinite preconditioner";
  else if (reason==KSP_DIVERGED_ITS)
    s<<"exceeded "<<maxits<<" iterations";
  else if (reason==KSP_CONVERGED_RTOL)
    s<<"rtol = "<<rtol;
  else if (reason==KSP_CONVERGED_ATOL)
    s<<"atol = "<<atol;
  else if (reason==KSP_CONVERGED_ITS)
    s<<"its";
  else if (reason==KSP_DIVERGED_INDEFINITE_MAT)
    s<<"indefinite";
  else
    s<<"unknown reason "<<(int)reason;
  s<<", error = "<<error;
  return s.str();
}

}
using namespace hollow;

void wrap_ksp() {
  typedef hollow::KSP Self;
  Class<Self>("KSP")
    .GEODE_GET(comm)
    .GEODE_METHOD(set_operators)
    .GEODE_METHOD(set_from_options)
    .GEODE_METHOD(solve)
    .GEODE_METHOD(report)
    ;
}
