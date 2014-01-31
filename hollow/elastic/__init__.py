from __future__ import division,print_function,unicode_literals,absolute_import

from .. import hollow_wrap

LaplaceElasticModel = {2:hollow_wrap.LaplaceElasticModel2d}

NeoHookeanElasticIGA = {2:hollow_wrap.NeoHookeanElasticIGA2d,
                        3:hollow_wrap.NeoHookeanElasticIGA3d}
NeoHookeanElasticModel = {2:hollow_wrap.NeoHookeanElasticModel2d,
                          3:hollow_wrap.NeoHookeanElasticModel3d}
