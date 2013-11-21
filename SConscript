Import('env external children')

external(env,'geode',default=1,required=1,libs=['geode'],headers=['geode/utility/config.h'])
external(env,'petsc',required=1,requires=['mpi'],libs=['petsc'],headers=['petsc.h'],body=['  PetscFinalize();'])
env = env.Clone(need_petsc=1)

children(env)
