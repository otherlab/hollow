#include <hollow/config.h>
#include <hollow/mpi.h>
#include <geode/utility/debug.h>
#include <geode/utility/format.h>
#include <geode/utility/Log.h>
#include <geode/utility/process.h>
namespace hollow {

using Log::cerr;
using std::endl;

void check_failed(const int line, const char* function, const char* file, const char* call, const int n) {
  const bool is_mpi = !strncmp(call,"MPI_",4);
  if (is_mpi) {
    const auto msg = format("%s:%s:%d: %s failed: %s",file,function,line,call,mpi_error_string(n));
    cerr << "\nrank " << comm_rank(MPI_COMM_WORLD) << ": " << msg << endl;
    process::backtrace();
    if (getenv("GEODE_BREAK_ON_ASSERT"))
      breakpoint();
    MPI_Abort(MPI_COMM_WORLD,n);
  } else {
    // Assume petsc
    #if PETSC_VERSION_MAJOR<3 || (PETSC_VERSION_MAJOR==3 && PETSC_VERSION_MINOR<2)
      PetscError(line,function,file,"",n,0," ");
    #elif defined(__SDIR__) // Obsolete __SDIR__ support
      PetscError(PETSC_COMM_WORLD,line,function,file,"",n,PETSC_ERROR_INITIAL," ");
    #else
      PetscError(PETSC_COMM_WORLD,line,function,file,   n,PETSC_ERROR_INITIAL," ");
    #endif
    if (getenv("GEODE_BREAK_ON_ASSERT"))
      breakpoint();
  }
  GEODE_FATAL_ERROR();
}

}
