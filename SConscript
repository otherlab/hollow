Import('env external children')

external(env,'geode',default=1,required=1,libs=['geode'],headers=['geode/utility/config.h'])
external(env,'petsc',required=1,requires=['mpi'],libs=['petsc'],headers=['petsc.h'],body=['  PetscFinalize();'])
external(env,'petiga',required=1,requires=['petsc'],libs=['petiga'],headers=['petiga.h'])
external(env,'tao',flags=['HOLLOW_TAO'],required=1,requires=['petsc'],libs=['tao'],headers=['tao.h'])

env = env.Clone(need_petsc=1,need_petiga=1,need_tao=1)
children(env)
