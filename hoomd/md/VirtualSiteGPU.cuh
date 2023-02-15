//
// Created by girard01 on 2/14/23.
//


#include "hip/hip_runtime.h"
#include "hoomd/HOOMDMath.h"
#include "hoomd/Index1D.h"
#include "hoomd/ParticleData.cuh"
#include "hoomd/TextureTools.h"

#include "hoomd/GPUPartition.cuh"

#ifdef __HIPCC__
#include "hoomd/WarpTools.cuh"
#endif // __HIPCC__

#include <assert.h>
#include <type_traits>

#ifndef HOOMD_VIRTUALSITEGPU_CUH
#define HOOMD_VIRTUALSITEGPU_CUH
namespace hoomd::md::kernel {
    struct VirtualSiteUpdateKernelArgs {
        Scalar4* d_postype;
        unsigned int* d_mol_list;
        Index2D indexer;
    };

    struct VirtualSiteDecomposeKernelArgs{
        Scalar4* forces;
        Scalar4* postype;
        Scalar* virial;
        Scalar* net_virial;
        uint64_t virial_pitch;
        uint64_t net_virial_pitch;
        unsigned int* d_mol_list;
        Index2D indexer;
    };


#ifdef __HIPCC__

    template<class Mapping>
    __global__ void gpu_update_virtual_site_kernel(Scalar4* postype,
                                                   const unsigned int* d_molecule_list,
                                                   Index2D moleculeIndexer,
                                                   const typename Mapping::param_type param,
                                                   const uint64_t N
                                                   ) {
        auto site = threadIdx.x + blockIdx.x * blockDim.x;
        if(site >= moleculeIndexer.getH())
            return;

        auto constexpr numel_base = Mapping::n_sites;
        const auto site_index = moleculeIndexer(numel_base, site);
        if(site_index >= N) // virtual particle is not local
            return;

        std::array<uint64_t, numel_base> indices;
        for(unsigned char s = 0; s < numel_base; s++)
            indices[s] = d_molecule_list[moleculeIndexer(s, site)];
        Mapping virtual_site(param, indices, d_molecule_list[site_index]);
        virtual_site.reconstructSite(postype);
    }

    template<class Mapping, bool compute_virial>
    __global__ void gpu_decompose_virtual_site_kernel(Scalar4* forces,
                                                      Scalar4* net_forces,
                                                      Scalar* virial,
                                                      Scalar* net_virial,
                                                      Scalar4* postype,
                                                      uint64_t virial_pitch,
                                                      uint64_t net_virial_pitch,
                                                      const unsigned int* d_molecule_list,
                                                      Index2D moleculeIndexer,
                                                      const typename Mapping::param_type param,
                                                      const uint64_t N){
        auto site = threadIdx.x + blockIdx.x * blockDim.x;
        if(site >= moleculeIndexer.getH())
            return;

        auto constexpr numel_base = Mapping::n_sites;
        const auto site_index = moleculeIndexer(numel_base, site);
        if(site_index >= N) // virtual particle is not local
            return;

        std::array<uint64_t, numel_base> indices;
        for(unsigned char s = 0; s < numel_base; s++)
            indices[s] = d_molecule_list[moleculeIndexer(s, site)];
        Mapping virtual_site(param, indices, d_molecule_list[site_index]);
        // decomposeForces set the net_force on the virtual particle to 0, so the virial needs to be
        // decomposed first
        if(compute_virial){
            virtual_site.decomposeVirial(virial,
                                         net_virial,
                                         virial_pitch,
                                         net_virial_pitch,
                                         postype,
                                         net_forces);
        }
        virtual_site.decomposeForces(forces, net_forces);
    }

#endif

    template<class Mapping>
    hipError_t gpu_update_virtual_sites(const VirtualSiteUpdateKernelArgs &args,
                                        const typename Mapping::param_type param) {
        return hipSuccess;
    }

    template<class Mapping>
    hipError_t gpu_decompose_virtual_sites(const VirtualSiteDecomposeKernelArgs& args,
                                           const typename Mapping::param_type param){
        // memset forces to 0 first

        // then compute the ctr forces

        return hipSuccess;
    }
}
#endif //HOOMD_VIRTUALSITEGPU_CUH
