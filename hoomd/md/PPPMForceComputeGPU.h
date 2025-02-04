// Copyright (c) 2009-2024 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

#include "PPPMForceCompute.h"

#ifndef __PPPM_FORCE_COMPUTE_GPU_H__
#define __PPPM_FORCE_COMPUTE_GPU_H__

#ifdef ENABLE_HIP

#if __HIP_PLATFORM_HCC__
#include <hipfft.h>
#elif __HIP_PLATFORM_NVCC__
#include <cufft.h>
typedef cufftComplex hipfftComplex;
typedef cufftHandle hipfftHandle;
#endif

#include <sstream>

// #define USE_HOST_DFFT

#include "hoomd/Autotuner.h"

#ifdef ENABLE_MPI
#include "CommunicatorGridGPU.h"

#ifndef USE_HOST_DFFT
#include "hoomd/extern/dfftlib/src/dfft_cuda.h"
#else
#include "hoomd/extern/dfftlib/src/dfft_host.h"
#endif
#endif

#define CHECK_HIPFFT_ERROR(status)                      \
        {                                               \
        handleHIPFFTResult(status, __FILE__, __LINE__); \
        }

namespace hoomd
    {
namespace md
    {
/*! Order parameter evaluated using the particle mesh method
 */
class PYBIND11_EXPORT PPPMForceComputeGPU : public PPPMForceCompute
    {
    public:
    //! Constructor
    PPPMForceComputeGPU(std::shared_ptr<SystemDefinition> sysdef,
                        std::shared_ptr<NeighborList> nlist,
                        std::shared_ptr<ParticleGroup> group);
    virtual ~PPPMForceComputeGPU();

    protected:
    //! Helper function to setup FFT and allocate the mesh arrays
    virtual void initializeFFT();

    //! Helper function to assign particle coordinates to mesh
    virtual void assignParticles();

    //! Helper function to update the mesh arrays
    virtual void updateMeshes();

    //! Helper function to interpolate the forces
    virtual void interpolateForces();

    //! Compute the optimal influence function
    virtual void computeInfluenceFunction();

    //! Helper function to calculate value of collective variable
    virtual Scalar computePE();

    //! Helper function to compute the virial
    virtual void computeVirial();

    //! Helper function to correct forces on excluded particles
    virtual void fixExclusions();

//! Check for HIPFFT errors
#ifdef __HIP_PLATFORM_HCC__
    inline void handleHIPFFTResult(hipfftResult result, const char* file, unsigned int line) const
#else
    inline void handleHIPFFTResult(cufftResult result, const char* file, unsigned int line) const
#endif
        {
#ifdef __HIP_PLATFORM_HCC__
        if (result != HIPFFT_SUCCESS)
#else
        if (result != CUFFT_SUCCESS)
#endif
            {
            std::ostringstream oss;
            oss << "HIPFFT returned error " << result << " in file " << file << " line " << line
                << std::endl;
            throw std::runtime_error(oss.str());
            }
        }

    private:
    std::shared_ptr<Autotuner<1>>
        m_tuner_assign; //!< Autotuner for assigning binned charges to mesh
    std::shared_ptr<Autotuner<1>> m_tuner_reduce_mesh; //!< Autotuner to reduce meshes for multi GPU
    std::shared_ptr<Autotuner<1>> m_tuner_update;      //!< Autotuner for updating mesh values
    std::shared_ptr<Autotuner<1>> m_tuner_force;       //!< Autotuner for populating the force array

    /// Autotuner for computing the influence function
    std::shared_ptr<Autotuner<1>> m_tuner_influence;

    hipfftHandle m_hipfft_plan;   //!< The FFT plan
    bool m_local_fft;             //!< True if we are only doing local FFTs (not distributed)
    bool m_cufft_initialized;     //!< True if CUFFT has been initialized
    bool m_cuda_dfft_initialized; //!< True if dfft has been initialized

#ifdef ENABLE_MPI
    typedef CommunicatorGridGPU<hipfftComplex> CommunicatorGridGPUComplex;
    std::shared_ptr<CommunicatorGridGPUComplex> m_gpu_grid_comm_forward; //!< Communicate mesh
    std::shared_ptr<CommunicatorGridGPUComplex>
        m_gpu_grid_comm_reverse; //!< Communicate fourier mesh

    dfft_plan m_dfft_plan_forward; //!< Forward distributed FFT
    dfft_plan m_dfft_plan_inverse; //!< Forward distributed FFT
#endif

    GlobalArray<hipfftComplex> m_mesh;         //!< The particle density mesh
    GlobalArray<hipfftComplex> m_mesh_scratch; //!< The particle density mesh per GPU, staging array
    GlobalArray<hipfftComplex> m_inv_fourier_mesh_x; //!< The inverse-fourier transformed force mesh
    GlobalArray<hipfftComplex> m_inv_fourier_mesh_y; //!< The inverse-fourier transformed force mesh
    GlobalArray<hipfftComplex> m_inv_fourier_mesh_z; //!< The inverse-fourier transformed force mesh

    GPUFlags<Scalar> m_sum;                   //!< Sum over fourier mesh values
    GlobalArray<Scalar> m_sum_partial;        //!< Partial sums over fourier mesh values
    GlobalArray<Scalar> m_sum_virial_partial; //!< Partial sums over virial mesh values
    GlobalArray<Scalar> m_sum_virial;         //!< Final sum over virial mesh values
    unsigned int m_block_size;                //!< Block size for fourier mesh reduction
    };

    } // end namespace md
    } // end namespace hoomd

#endif // ENABLE_HIP
#endif // __PPPM_FORCE_COMPUTE_GPU_H__
