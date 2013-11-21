#include <geode/python/module.h>

GEODE_PYTHON_MODULE(hollow_wrap) {
  GEODE_WRAP(init)
  GEODE_WRAP(mpi)
  GEODE_WRAP(vec)
  GEODE_WRAP(mat)
  GEODE_WRAP(fe)
  GEODE_WRAP(dm)
  GEODE_WRAP(snes)
  GEODE_WRAP(model)
}
