rm -rf build/* ;  cd build/
cmake ../ -DPYTHON_EXECUTABLE=`which python` -DCMAKE_INSTALL_PREFIX=${HOME}/local -DSINGLE_PRECISION=OFF -DENABLE_CUDA=OFF -DBUILD_DEPRECATED=ON -DBUILD_JIT=ON -DLLVM_DIR=/opt/local/libexec/llvm-3.8/share/llvm/cmake -DBUILD_TESTING=off -DBUILD_MD=off
make -j10; make install -j10
