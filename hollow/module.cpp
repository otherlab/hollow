#include <geode/python/module.h>

GEODE_PYTHON_MODULE(hollow_wrap) {
  // petsc
  GEODE_WRAP(init)
  GEODE_WRAP(mpi)
  GEODE_WRAP(vec)
  GEODE_WRAP(mat)
  GEODE_WRAP(ksp)
  GEODE_WRAP(fe)
  GEODE_WRAP(dm)
  GEODE_WRAP(snes)
  GEODE_WRAP(model)

  // laplace
  GEODE_WRAP(laplace)

  // iga
  GEODE_WRAP(iga)
  GEODE_WRAP(laplace_iga)
}
