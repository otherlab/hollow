#include <geode/python/module.h>
#include <geode/array/convert.h>
#include <geode/mesh/ids.h>

namespace geode {
ARRAY_CONVERSIONS(1,HalfedgeId)
}

GEODE_PYTHON_MODULE(hollow_wrap) {
  // petsc
  GEODE_WRAP(petsc_config)
  GEODE_WRAP(init)
  GEODE_WRAP(mpi)
  GEODE_WRAP(vec)
  GEODE_WRAP(mat)
  GEODE_WRAP(ksp)
  GEODE_WRAP(fe)
  GEODE_WRAP(dm)
  GEODE_WRAP(snes)
  GEODE_WRAP(model)
  GEODE_WRAP(analytic)

  // laplace
  GEODE_WRAP(laplace)

  // iga
  GEODE_WRAP(iga)
  GEODE_WRAP(laplace_iga)

  // tao
  GEODE_WRAP(tao_solver)

  // elastic
  GEODE_WRAP(elastic_test)
  GEODE_WRAP(elastic)

  // ode
  GEODE_WRAP(integral)
}
