from __future__ import division,print_function,unicode_literals,absolute_import

from .. import hollow_wrap

def FE(comm,dim,components,prefix='',qorder=-1):
  return hollow_wrap.FE(comm,dim,components,prefix,qorder)
