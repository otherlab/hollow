from __future__ import division,print_function,unicode_literals,absolute_import

from .. import hollow_wrap

if 0:
  LaplaceElasticModel = {2:hollow_wrap.LaplaceElasticModel2d}
  NeoHookeanElasticModel = {2:hollow_wrap.NeoHookeanElasticModel2d,
                            3:hollow_wrap.NeoHookeanElasticModel3d}
NeoHookeanElasticIGA = {2:hollow_wrap.NeoHookeanElasticIGA2d,
                        3:hollow_wrap.NeoHookeanElasticIGA3d}
