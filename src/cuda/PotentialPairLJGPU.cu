#include "PotentialPairEvaluatorLJ.h"
#include "PotentialPairLJGPU.cuh"

cudaError_t gpu_compute_ljtemp_forces(const gpu_force_data_arrays& force_data,
                                      const gpu_pdata_arrays &pdata,
                                      const gpu_boxsize &box,
                                      const gpu_nlist_array &nlist,
                                      float2 *d_params,
                                      float *d_rcutsq,
                                      float *d_ronsq,
                                      int ntypes,
                                      const pair_args& args)
    {
    return gpu_compute_pair_forces<PotentialPairEvaluatorLJ>(force_data,
                                                             pdata,
                                                             box,
                                                             nlist,
                                                             d_params,
                                                             d_rcutsq,
                                                             d_ronsq,
                                                             ntypes,
                                                             args);
    }