// Petsc initialization

#include <hollow/petsc/init.h>
#include <hollow/petsc/mpi.h>
#include <geode/array/Array.h>
#include <geode/python/stl.h>
#include <geode/python/wrap.h>
#if defined(OPEN_MPI) && defined(__linux__)
#include <dlfcn.h>
#endif
#ifdef HOLLOW_TAO
#include <taosolver.h>
#endif
namespace hollow {

bool petsc_initialized() {
  PetscBool initialized;
  CHECK(PetscInitialized(&initialized));
  return initialized!=0;
}

void petsc_finalize() {
  if (petsc_initialized()) {
#ifdef HOLLOW_TAO
    TaoFinalize();
#endif
    PetscFinalize();
  }
}

// Work around annoying dlopen issue, following http://petsc.cs.iit.edu/petsc4py/petsc4py-dev/rev/300045797445
static void dlopen_workaround() {
#if defined(OPEN_MPI) && defined(__linux__)
  const int mode = RTLD_NOW | RTLD_GLOBAL | RTLD_NOLOAD;
  dlopen("libmpi.so.1", mode) || dlopen("libmpi.so.0", mode) || dlopen("libmpi.so", mode);
#endif
}

void petsc_initialize(const string& help, const vector<string>& args) {
  GEODE_ASSERT(!petsc_initialized());
  dlopen_workaround();

  int argc = int(args.size());
  Array<char*> pointers(argc,false);
  for(int i=0;i<argc;i++)
    pointers[i] = (char*)args[i].c_str();
  char** argv = pointers.data();
  CHECK(PetscInitialize(&argc,&argv,0,help.c_str()));
#ifdef HOLLOW_TAO
  CHECK(TaoInitialize(0,0,0,0));
#endif
  GEODE_ASSERT(size_t(argc)==args.size());
  GEODE_ASSERT(argv==pointers.data());
  atexit(petsc_finalize);
}

void petsc_reinitialize() {
  if (!petsc_initialized()) {
    dlopen_workaround();
    PetscInitializeNoArguments();
#ifdef HOLLOW_TAO
    CHECK(TaoInitialize(0,0,0,0));
#endif
    atexit(petsc_finalize);
  }
}

void petsc_set_options(const vector<string>& args) {
  // Verify that the first argument isn't an option, since it's ignored
  GEODE_ASSERT(args.size() && args[0].size() && args[0][0]!='-');
  // Remove all existing options
  CHECK(PetscOptionsClear());
  // Replace with new options
  int argc = int(args.size());
  Array<char*> pointers(argc,false);
  for(int i=0;i<argc;i++)
    pointers[i] = (char*)args[i].c_str();
  char** argv = pointers.data();
  CHECK(PetscOptionsInsert(&argc,&argv,0));
  GEODE_ASSERT(size_t(argc)==args.size());
  GEODE_ASSERT(argv==pointers.data());
}

static MPI_Comm petsc_comm_world() {
  return PETSC_COMM_WORLD;
}

}
using namespace hollow;

void wrap_init() {
  GEODE_FUNCTION(petsc_initialized)
  GEODE_FUNCTION(petsc_initialize)
  GEODE_FUNCTION(petsc_reinitialize)
  GEODE_FUNCTION(petsc_set_options)
  GEODE_FUNCTION(petsc_finalize)
  GEODE_FUNCTION(petsc_comm_world)
}
