// Petsc initialization
#pragma once

#include <hollow/petsc/config.h>
#include <geode/utility/debug.h>
#include <vector>
#include <string>
// Work around configuration issue in old PETSc versions
#define PETSC_RESTRICT PETSC_CXX_RESTRICT
#include <petsc.h>
namespace hollow {

using std::string;
using std::vector;

// Initialize petsc, and automatically finalize on application exit
HOLLOW_EXPORT void petsc_initialize(const string& help, const vector<string>& args);

// Check if petsc is initialized
HOLLOW_EXPORT bool petsc_initialized();

// Ensure that petsc is initialized.  Can be called multiple times, but no help or args support.
HOLLOW_EXPORT void petsc_reinitialize();

// Explicitly finalize petsc.  This happens automatically at program exit if petsc_initialize
// is called, but sometime it is useful to do it sooner.
HOLLOW_EXPORT void petsc_finalize();

// Reset petsc options to the given list.  Any petsc objects already allocated will not change.
HOLLOW_EXPORT void petsc_set_options(const vector<string>& args);

// Add new petsc options without removing old options.  Like argv[0], the first option is ignored.
HOLLOW_EXPORT void petsc_add_options(const vector<string>& args);

}
