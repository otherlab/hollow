// MPI wrappers and convenience routines

#include <hollow/petsc/mpi.h>
#include <geode/python/Class.h>
#include <geode/utility/Log.h>
namespace hollow {

using Log::cout;
using std::endl;

string mpi_error_string(int code) {
  int length;
  char error[MPI_MAX_ERROR_STRING];
  MPI_Error_string(code,error,&length);
  return error;
}

int comm_size(MPI_Comm comm) {
  int size;
  CHECK(MPI_Comm_size(comm,&size));
  return size;
} 

int comm_rank(MPI_Comm comm) {
  int rank;
  CHECK(MPI_Comm_rank(comm,&rank));
  return rank;
}

namespace {
struct Comm : public Object {
  GEODE_DECLARE_TYPE(GEODE_NO_EXPORT)
  MPI_Comm comm;
protected:
  Comm(MPI_Comm comm) : comm(comm) {}
};
GEODE_DEFINE_TYPE(Comm)
}

}

PyObject* to_python(MPI_Comm comm) {
  return to_python(geode::new_<hollow::Comm>(comm));
}

namespace geode {
MPI_Comm FromPython<MPI_Comm>::convert(PyObject* object) {
  return from_python<hollow::Comm&>(object).comm;
}

}
using namespace hollow;

void wrap_mpi() {
  typedef Comm Self;
  Class<Comm>("Comm")
    ;
}
