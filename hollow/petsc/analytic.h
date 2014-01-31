// Pointwise analytically known functions
#pragma once

#include <hollow/petsc/config.h>
#include <geode/python/Object.h>
namespace hollow {

struct Analytic : public Object {
  GEODE_DECLARE_TYPE(GEODE_NO_EXPORT)
  typedef Object Base;
  typedef PetscReal T;
  typedef PetscScalar S;

  typedef void (*F)(const T*, S*, void*);

  const int dim; // Input spatial dimension
  const int count; // Output count
  const F f; // Takes dim T's, produces count S's
  void* const ctx;

protected:
  Analytic(const int dim, const int count, const F f, void* const ctx);
public:
  ~Analytic();
};

template<int dim,int count> struct AnalyticImpl : public Analytic {
  GEODE_DECLARE_TYPE(GEODE_NEW_FRIEND)
};

}
