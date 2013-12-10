// Wrapper around PetscFEM

#include <hollow/petsc/model.h>
#include <geode/python/Class.h>
#include <geode/python/stl.h>
namespace hollow {

GEODE_DEFINE_TYPE(Model)

Model::Model(const FEs& fe, const FEs& fe_aux, const FEs& fe_bd)
  : dim(fe.at(0)->spatial_dimension())
  , fe(fe)
  , fe_aux(fe_aux)
  , fe_bd(fe_bd)
  , fep(int(fe.size()))
  , fep_aux(int(fe_aux.size()))
  , fep_bd(int(fe_bd.size()))
  , f0(int(fe.size()))
  , f1(int(fe.size()))
  , b0(int(fe.size()))
  , b1(int(fe.size()))
  , g0(int(fe.size()),int(fe.size()))
  , g1(int(fe.size()),int(fe.size()))
  , g2(int(fe.size()),int(fe.size()))
  , g3(int(fe.size()),int(fe.size()))
  , boundary(int(fe.size()))
  , exact(int(fe.size()))
  , boundary_contexts(int(fe.size()))
  , exact_contexts(int(fe.size())) {
  // Check consistency
  for (const auto& f : fe)
    GEODE_ASSERT(f->spatial_dimension()==dim);
  for (const auto& f : fe_aux)
    GEODE_ASSERT(f->spatial_dimension()==dim);
  if (fe_bd.size()) {
    GEODE_ASSERT(fe.size()==fe_bd.size());
    for (const auto& f : fe_bd)
      GEODE_ASSERT(f->spatial_dimension()==dim-1);
  }

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
  fem.bcCtxs = boundary_contexts.data();
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
