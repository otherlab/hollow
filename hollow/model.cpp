// Wrapper around PetscFEM

#include <hollow/model.h>
#include <geode/python/Class.h>
#include <geode/python/stl.h>
namespace hollow {

GEODE_DEFINE_TYPE(Model)

Model::Model(const FEs& fe, const FEs& fe_aux, const FEs& fe_bd)
  : dim(fe.at(0)->spatial_dimension())
  , fe(fe)
  , fe_aux(fe_aux)
  , fe_bd(fe_bd)
  , fep(fe.size())
  , fep_aux(fe_aux.size())
  , fep_bd(fe_bd.size())
  , f0(fe.size())
  , f1(fe.size())
  , b0(fe.size())
  , b1(fe.size())
  , g0(fe.size(),fe.size())
  , g1(fe.size(),fe.size())
  , g2(fe.size(),fe.size())
  , g3(fe.size(),fe.size())
  , boundary(fe.size())
  , exact(fe.size()) {
  GEODE_ASSERT(!fe_bd.size() || fe.size()==fe_bd.size());
  for (const int i : range(fep    .size())) fep    [i] = fe    [i]->fe;
  for (const int i : range(fep_aux.size())) fep_aux[i] = fe_aux[i]->fe;
  for (const int i : range(fep_bd .size())) fep_bd [i] = fe_bd [i]->fe;
  fem.fe = fep.data();
  fem.feBd = fe_bd.size() ? fep_bd.data() : 0;
  fem.feAux = fep_aux.data();

  fem.f0Funcs = f0.data();
  fem.f1Funcs = f1.data();
  fem.g0Funcs = g0.data();
  fem.g1Funcs = g1.data();
  fem.g2Funcs = g2.data();
  fem.g3Funcs = g3.data();
  fem.f0BdFuncs = b0.data();
  fem.f1BdFuncs = b1.data();
  fem.bcFuncs = boundary.data();
}

Model::~Model() {}

}
using namespace hollow;

void wrap_model() {
  typedef Model Self;
  Class<Self>("Model")
    .GEODE_FIELD(fe)
    .GEODE_FIELD(fe_aux)
    .GEODE_FIELD(fe_bd)
    ;
}
