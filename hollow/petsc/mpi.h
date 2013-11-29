// MPI wrappers and convenience routines
#pragma once

#include <hollow/petsc/config.h>
#include <geode/python/forward.h>
#include <string>
namespace hollow {

using std::string;

// Convert an mpi error code into a string
HOLLOW_EXPORT string mpi_error_string(int code);

// Convenience functions
HOLLOW_EXPORT int comm_size(MPI_Comm comm);
HOLLOW_EXPORT int comm_rank(MPI_Comm comm);

}

// Python conversions.  Warning: These do not manage ownership!  We rely on petsc for that.
HOLLOW_EXPORT PyObject* to_python(MPI_Comm comm);
namespace geode {
template<> struct FromPython<MPI_Comm>{HOLLOW_EXPORT static MPI_Comm convert(PyObject* object);};
}
