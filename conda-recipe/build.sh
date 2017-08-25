mkdir -p build-conda
cd build-conda
rm -rf ./*

export GCC_ARCH=core2

if [ "$(uname)" == "Darwin" ]; then

cmake ../ \
      -DCMAKE_OSX_DEPLOYMENT_TARGET=10.8 \
      -DCMAKE_CXX_FLAGS="-mmacosx-version-min=10.8 -stdlib=libc++" \
      -DCMAKE_INSTALL_PREFIX=${SP_DIR} \
      -DPYTHON_EXECUTABLE=${PYTHON} \
       \
      -DENABLE_MPI=off \
       \
      -DENABLE_CUDA=off \
      -DENABLE_EMBED_CUDA=off \
       \
      -DBUILD_TESTING=off \
      -DMKL_LIBRARIES=""

make install -j 4

else
# Linux build
CC=${PREFIX}/bin/gcc
CXX=${PREFIX}/bin/g++

cmake ../ \
      -DCMAKE_INSTALL_PREFIX=${SP_DIR} \
      -DPYTHON_EXECUTABLE=${PYTHON} \
      -DCUDA_HOST_COMPILER=${CC} \
       \
      -DENABLE_MPI=on \
      -DMPI_CXX_COMPILER=${PREFIX}/bin/mpic++ \
      -DMPI_C_COMPILER=${PREFIX}/bin/mpicc \
       \
      -DENABLE_CUDA=on \
      -DENABLE_EMBED_CUDA=off \
       \
      -DBUILD_TESTING=off \
      -DMKL_LIBRARIES=""

make install -j 2
fi
