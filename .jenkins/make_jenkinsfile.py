import jinja2

env = jinja2.Environment(loader=jinja2.FileSystemLoader('.'))
template = env.get_template('Jenkinsfile.jinja')

ci_base = 'ci-2019.06'

unit_tests = []
vldt_tests = []
unit_tests.append(dict(name='gcc7-py36-cuda9',
                  agent='gpu-short',
                  CC = '/usr/bin/gcc',
                  CXX = '/usr/bin/g++',
                  PYVER = '3.6',
                  CMAKE_BIN = '/usr/bin',
                  ENABLE_CUDA = 'ON',
                  ENABLE_MPI = 'OFF',
                  ENABLE_TBB = 'OFF',
                  ALWAYS_USE_MANAGED_MEMORY = 'OFF',
                  CONTAINER = ci_base + '-cuda9.simg',
                  BUILD_JIT = 'OFF',
                  LLVM_VERSION = '',
                  OMP_NUM_THREADS = '1'))

unit_tests.append(dict(name='gcc7-py36-mpi-cuda9',
                  agent='gpu-short',
                  CC = '/usr/bin/gcc-7',
                  CXX = '/usr/bin/g++-7',
                  PYVER = '3.6',
                  CMAKE_BIN = '/usr/bin',
                  ENABLE_CUDA = 'ON',
                  ENABLE_MPI = 'ON',
                  ENABLE_TBB = 'OFF',
                  ALWAYS_USE_MANAGED_MEMORY = 'OFF',
                  CONTAINER = ci_base + '-cuda9.simg',
                  BUILD_JIT = 'OFF',
                  LLVM_VERSION = '',
                  OMP_NUM_THREADS = '1'))

unit_tests.append(dict(name='gcc7-py36-mpi-cuda10-mng',
                  agent='gpu-short',
                  CC = '/usr/bin/gcc-7',
                  CXX = '/usr/bin/g++-7',
                  PYVER = '3.6',
                  CMAKE_BIN = '/usr/bin',
                  ENABLE_CUDA = 'ON',
                  ENABLE_MPI = 'ON',
                  ENABLE_TBB = 'OFF',
                  ALWAYS_USE_MANAGED_MEMORY = 'ON',
                  CONTAINER = ci_base + '-cuda10.simg',
                  BUILD_JIT = 'OFF',
                  LLVM_VERSION = '',
                  OMP_NUM_THREADS = '1'))

vldt_tests.append(dict(name='vld-gcc6-py36-mpi',
                  agent='linux-cpu',
                  CC = '/usr/bin/gcc-6',
                  CXX = '/usr/bin/g++-6',
                  PYVER = '3.6',
                  CMAKE_BIN = '/usr/bin',
                  ENABLE_CUDA = 'OFF',
                  ENABLE_MPI = 'ON',
                  ENABLE_TBB = 'OFF',
                  ALWAYS_USE_MANAGED_MEMORY = 'OFF',
                  CONTAINER = ci_base + '-ubuntu18.04.simg',
                  BUILD_JIT = 'ON',
                  LLVM_VERSION = '6.0',
                  OMP_NUM_THREADS = '1'))

vldt_tests.append(dict(name='vld-gcc7-py36-mpi-tbb1',
                  agent='linux-cpu',
                  CC = '/usr/bin/gcc-7',
                  CXX = '/usr/bin/g++-7',
                  PYVER = '3.6',
                  CMAKE_BIN = '/usr/bin',
                  ENABLE_CUDA = 'OFF',
                  ENABLE_MPI = 'ON',
                  ENABLE_TBB = 'ON',
                  ALWAYS_USE_MANAGED_MEMORY = 'OFF',
                  CONTAINER = ci_base + '-ubuntu18.04.simg',
                  BUILD_JIT = 'ON',
                  LLVM_VERSION = '6.0',
                  OMP_NUM_THREADS = '1'))

vldt_tests.append(dict(name='vld-gcc8-py36-mpi-tbb3',
                  agent='linux-cpu',
                  CC = '/usr/bin/gcc-8',
                  CXX = '/usr/bin/g++-8',
                  PYVER = '3.6',
                  CMAKE_BIN = '/usr/bin',
                  ENABLE_CUDA = 'OFF',
                  ENABLE_MPI = 'ON',
                  ENABLE_TBB = 'ON',
                  ALWAYS_USE_MANAGED_MEMORY = 'OFF',
                  CONTAINER = ci_base + '-ubuntu18.04.simg',
                  BUILD_JIT = 'ON',
                  LLVM_VERSION = '6.0',
                  OMP_NUM_THREADS = '3'))

vldt_tests.append(dict(name='vld-gcc7-py36-mpi-cuda10',
                  agent='gpu-long',
                  CC = '/usr/bin/gcc-7',
                  CXX = '/usr/bin/g++-7',
                  PYVER = '3.6',
                  CMAKE_BIN = '/usr/bin',
                  ENABLE_CUDA = 'ON',
                  ENABLE_MPI = 'ON',
                  ENABLE_TBB = 'OFF',
                  ALWAYS_USE_MANAGED_MEMORY = 'OFF',
                  CONTAINER = ci_base + '-cuda10.simg',
                  BUILD_JIT = 'OFF',
                  LLVM_VERSION = '',
                  OMP_NUM_THREADS = '1'))


print(template.render(unit_tests=unit_tests, vldt_tests=vldt_tests))
