#!/bin/bash

# Remove old build directory
# rm -rf build

# Make build directory
# mkdir -p build

cd build
export CC=$(which gcc)
export CXX=$(which g++)
echo "Using compilers $($CC --version | head -n 1), $($CXX --version | head -n 1)."

# Compile against correct python
CMAKE_FLAGS="${CMAKE_FLAGS} -DPYTHON_EXECUTABLE=$(which python)"
PYTHON_LIBRARY_PATH=$(python -c "import distutils.sysconfig as sysconfig; import os; print(os.path.join(sysconfig.get_config_var('LIBDIR'), sysconfig.get_config_var('LDLIBRARY')))")
CMAKE_FLAGS="${CMAKE_FLAGS} -DPYTHON_LIBRARY=${PYTHON_LIBRARY_PATH}"

# Install to the conda packages path
CMAKE_FLAGS="${CMAKE_FLAGS} -DCMAKE_INSTALL_PREFIX=${CONDA_PREFIX}/lib/python3.6/site-packages"

cmake ../ ${CMAKE_FLAGS}
make -j4
make install
