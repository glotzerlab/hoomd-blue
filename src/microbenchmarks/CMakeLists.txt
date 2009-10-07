# $Id$
# $URL$
# Maintainer: joaander

if (UNIX)
if(ENABLE_CUDA)
cuda_add_executable(gpu_binning gpu_binning.cu)
cuda_add_executable(gpu_nlist gpu_nlist.cu)
endif(ENABLE_CUDA)
endif (UNIX)
