#include <cuda_runtime.h>

cudaError_t gpu_nlist_needs_update_check_new(unsigned int * d_result,
                                             const float4 *d_last_pos,
                                             const float4 *d_pos,
                                             const unsigned int N,
                                             const gpu_boxsize& box,
                                             const float maxshiftsq);

