#pragma once

// Must be included first
#include <geode/utility/config.h>
#include <geode/python/forward.h>
#include <petscsys.h>
namespace hollow {

using namespace geode;

#define HOLLOW_EXPORT GEODE_EXPORT

static_assert(!MPI_SUCCESS,"MPI_SUCCESS must be false to use the same CHECK routine for both MPI and Petsc");
HOLLOW_EXPORT void GEODE_NORETURN(check_failed(const int line, const char* function, const char* file,
                                               const char* call, const int n)) GEODE_COLD;

// Check if an MPI or PETSc call succeeds, and bail if not
#define CHECK(call) ({ \
  const int _n = call; \
  if (PetscUnlikely(_n)) \
    ::hollow::check_failed(__LINE__,__FUNCTION__,__FILE__,#call,_n); \
  })

} namespace geode {
GEODE_DECLARE_ENUM(InsertMode,HOLLOW_EXPORT)
}
