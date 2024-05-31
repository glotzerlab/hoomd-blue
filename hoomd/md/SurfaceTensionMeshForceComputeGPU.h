#include "SurfaceTensionMeshForceCompute.h"
#include "SurfaceTensionMeshForceComputeGPU.cuh"
#include "hoomd/Autotuner.h"

#include <memory>

/*! \file SurfaceTensionMeshForceComputeGPU.h
    \brief Declares a class for computing area conservation energy forces on the GPU
*/

#ifdef __HIPCC__
#error This header cannot be compiled by nvcc
#endif

#ifndef __SURFACETENSIONMESHFORCECOMPUTE_GPU_H__
#define __SURFACETENSIONMESHFORCECOMPUTE_GPU_H__

namespace hoomd
    {
namespace md
    {

//! Computes surface tension forces on the mesh on the GPU
/*! SurfaceTension energy forces are computed on every particle in a mesh.

    \ingroup computes

*/
class PYBIND11_EXPORT SurfaceTensionMeshForceComputeGPU
    : public SurfaceTensionMeshForceCompute
    {
    public:
    //! Constructs the compute
    SurfaceTensionMeshForceComputeGPU(std::shared_ptr<SystemDefinition> sysdef,
                                                std::shared_ptr<MeshDefinition> meshdef);

    //! Set the parameters
    virtual void setParams(unsigned int type, Scalar sigma);

    virtual Scalar getArea()
        {
        computeArea();
        return m_area;
        }

    protected:
    unsigned int m_block_size; //!< block size for partial sum memory
    unsigned int m_num_blocks; //!< number of memory blocks reserved for partial sum memory

    std::shared_ptr<Autotuner<1>> m_tuner; //!< Autotuner for block size
    GPUArray<unsigned int> m_flags;     //!< Flags set during the kernel execution
    GPUArray<Scalar> m_params;         //!< Parameters stored on the GPU

    GPUArray<Scalar> m_partial_sum; //!< memory space for partial sum over volume
    GPUArray<Scalar> m_sum;         //!< memory space for sum over volume

    //! Actually compute the forces
    virtual void computeForces(uint64_t timestep);

    virtual void computeArea();
    };

namespace detail
    {
//! Exports the SurfaceTensionMeshForceComputeGPU class to python
void export_SurfaceTensionMeshForceComputeGPU(pybind11::module& m);

    } // end namespace detail
    } // end namespace md
    } // end namespace hoomd

#endif
