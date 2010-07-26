# $Id$
# $URL$
# Maintainer: joaander

if (UNIX)
if(ENABLE_CUDA)
cuda_add_executable(gpu_binning gpu_binning.cu)
cuda_add_executable(gpu_nlist gpu_nlist.cu)
cuda_add_executable(gpu_nlist2 gpu_nlist2.cu)
endif(ENABLE_CUDA)
endif (UNIX)
