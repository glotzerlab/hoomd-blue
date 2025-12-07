// Copyright (c) 2009-2025 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

#ifndef __POTENTIALMESHTRIANGLEPAIR_GPU_H__
#define __POTENTIALMESHTRIANGLEPAIR_GPU_H__

#ifdef ENABLE_HIP

#include <memory>

#include "PotentialMeshTriangle.h"
#include "PotentialMeshTriangleGPU.cuh"

#include "hoomd/Autotuner.h"

/*! \file PotentialMeshTriangleGPU.h
    \brief Defines the template class for standard mesh triangle particle potentials on the GPU
    \note This header cannot be compiled by nvcc
*/

#ifdef __HIPCC__
#error This header cannot be compiled by nvcc
#endif

#include <pybind11/pybind11.h>

namespace hoomd
    {
namespace md
    {
template<class evaluator> class PotentialMeshTriangleGPU : public PotentialMeshTriangle<evaluator>
    {
    public:
    //! Constructs the compute
    PotentialMeshTriangleGPU(std::shared_ptr<SystemDefinition> sysdef, 
		  std::shared_ptr<NeighborList> nlist,
                  std::shared_ptr<MeshDefinition> meshdef);

    //! Destructor
    virtual ~PotentialMeshTriangleGPU() { };

    protected:
    std::shared_ptr<Autotuner<2>> m_tuner_triangles;
    std::shared_ptr<Autotuner<1>> m_tuner_particles;

    //! Actually compute the forces
    void computeForces(uint64_t timestep) override;
    void computeForcesTriangle(uint64_t timestep);
    void computeForcesParticle(uint64_t timestep);

    }; // end class PotentialMeshTriangle

template<class evaluator>
PotentialMeshTriangleGPU<evaluator>::PotentialMeshTriangleGPU(std::shared_ptr<SystemDefinition> sysdef,
                                        std::shared_ptr<NeighborList> nlist,
                                        std::shared_ptr<MeshDefinition> meshdef)
    : PotentialMeshTriangle<evaluator>(sysdef, nlist, meshdef)
    {
    if (!this->m_exec_conf->isCUDAEnabled())
        {
        this->m_exec_conf->msg->error()
            << "Creating a PotentialMeshTriangleGPU with no GPU in the execution configuration"
            << std::endl;
        throw std::runtime_error("Error initializing PotentialMeshTriangleGPU");
        }

    this->m_mesh_data->createMeshTriangleList();

    m_tuner_triangles.reset(new Autotuner<2>({AutotunerBase::makeBlockSizeRange(this->m_exec_conf),
                                    AutotunerBase::getTppListPow2(this->m_exec_conf)},
                                   this->m_exec_conf,
                                   "mesh_triangle_triangles_" + evaluator::getName()));

    m_tuner_particles.reset(new Autotuner<1>({AutotunerBase::makeBlockSizeRange(this->m_exec_conf)},
                                         this->m_exec_conf,
                                         "mesh_triangle_particles"));

    this->m_autotuners.insert(this->m_autotuners.end(), {m_tuner_triangles, m_tuner_particles});

#ifdef ENABLE_MPI
    // synchronize autotuner results across ranks
    m_tuner_triangles->setSync(bool(this->m_pdata->getDomainDecomposition()));
#endif
    }


/*! Actually perform the force computation
    \param timestep Current time step
 */
template<class evaluator>
void PotentialMeshTriangleGPU<evaluator>::computeForces(uint64_t timestep)
    {
    // start by updating the neighborlist
    this->m_nlist->compute(timestep);

    // The GPU implementation CANNOT handle a half neighborlist, error out now
    bool third_law = this->m_nlist->getStorageMode() == NeighborList::half;
    if (third_law)
        {
        this->m_exec_conf->msg->error()
    	<< "PotentialMeshTriangleGPU cannot handle a half neighborlist" << std::endl;
        throw std::runtime_error("Error computing forces in PotentialMeshTriangleGPU");
        }
    computeForcesTriangle(timestep);
    computeForcesParticle(timestep);
    }

template<class evaluator>
void PotentialMeshTriangleGPU<evaluator>::computeForcesTriangle(uint64_t timestep)
    {
    // access the neighbor list, particle data, and system box
    ArrayHandle<unsigned int> d_n_neigh(this->m_nlist->getNNeighArray(),
    				    access_location::device,
    				    access_mode::read);
    ArrayHandle<unsigned int> d_nlist(this->m_nlist->getNListArray(),
    				  access_location::device,
    				  access_mode::read);
    //     Index2D nli = this->m_nlist->getNListIndexer();
    ArrayHandle<size_t> d_head_list(this->m_nlist->getHeadList(),
    				access_location::device,
    				access_mode::read);
    
    // access the particle data arrays
    ArrayHandle<Scalar4> d_pos(this->m_pdata->getPositions(), access_location::device, access_mode::read);
    ArrayHandle<unsigned int> d_tag(this->m_pdata->getTags(), access_location::device, access_mode::read);

    ArrayHandle<Scalar4> d_force(this->m_force, access_location::device, access_mode::readwrite);
    ArrayHandle<Scalar> d_virial(this->m_virial, access_location::device, access_mode::readwrite);

    // access the parameters
    ArrayHandle<Scalar> d_rcutsq(this->m_rcutsq, access_location::device, access_mode::read);

    const GPUArray<typename Angle::members_t>& gpu_meshtriangle_list
        = this->m_mesh_data->getMeshTriangleData()->getGPUTable();
    const Index2D& gpu_table_indexer
        = this->m_mesh_data->getMeshTriangleData()->getGPUTableIndexer();

    ArrayHandle<typename Angle::members_t> d_gpu_meshtrianglelist(gpu_meshtriangle_list,
                                                                  access_location::device,
                                                                  access_mode::read);
    ArrayHandle<unsigned int> d_gpu_meshtriangle_pos_list(
        this->m_mesh_data->getMeshTriangleData()->getGPUPosTable(),
        access_location::device,
        access_mode::read);
    ArrayHandle<unsigned int> d_gpu_n_meshtriangle(
        this->m_mesh_data->getMeshTriangleData()->getNGroupsArray(),
        access_location::device,
        access_mode::read);

    BoxDim box = this->m_pdata->getBox();

    // access parameters
    ArrayHandle<typename evaluator::param_type> d_params(this->m_params,
                                                         access_location::device,
                                                         access_mode::read);

    PDataFlags flags = this->m_pdata->getFlags();
    this->m_exec_conf->setDevice();


    //m_tuner_triangles->begin();

    m_tuner_triangles->begin();
    auto param = m_tuner_triangles->getParam();
    unsigned int block_size = param[0];
    unsigned int threads_per_particle = param[1];

    kernel::gpu_compute_mesh_triangle_triangles_force<evaluator>(
        	    kernel::meshtriangle_args_t(d_force.data,
                                              d_virial.data,
                                              this->m_virial.getPitch(),
                                              this->m_pdata->getN(),
                                	      this->m_pdata->getMaxN(),
                                              d_pos.data,
                                              box,
                                              d_n_neigh.data,
                                              d_nlist.data,
                                              d_head_list.data,
                                              d_rcutsq.data,
                                              this->m_nlist->getNListArray().getPitch(),
                                              this->m_pdata->getNTypes(),
                                              block_size,
        				      flags[pdata_flag::pressure_tensor],
        				      this->m_exec_conf->dev_prop),
        				      threads_per_particle,
                                              d_params.data,
                                              d_gpu_meshtrianglelist.data,
                                              d_gpu_meshtriangle_pos_list.data,
                                              gpu_table_indexer,
                                              d_gpu_n_meshtriangle.data);

    if (this->m_exec_conf->isCUDAErrorCheckingEnabled())
        CHECK_CUDA_ERROR();
    m_tuner_triangles->end();
}
    

template<class evaluator>
void PotentialMeshTriangleGPU<evaluator>::computeForcesParticle(uint64_t timestep)
    {
    // access the neighbor list, particle data, and system box
    ArrayHandle<unsigned int> d_n_neigh(this->m_nlist->getNNeighArray(),
    				    access_location::device,
    				    access_mode::read);
    ArrayHandle<unsigned int> d_nlist(this->m_nlist->getNListArray(),
    				  access_location::device,
    				  access_mode::read);
    //     Index2D nli = this->m_nlist->getNListIndexer();
    ArrayHandle<size_t> d_head_list(this->m_nlist->getHeadList(),
    				access_location::device,
    				access_mode::read);
    
    // access the particle data arrays
    ArrayHandle<Scalar4> d_pos(this->m_pdata->getPositions(), access_location::device, access_mode::read);
    ArrayHandle<unsigned int> d_tag(this->m_pdata->getTags(), access_location::device, access_mode::read);

    ArrayHandle<Scalar4> d_force(this->m_force, access_location::device, access_mode::readwrite);
    ArrayHandle<Scalar> d_virial(this->m_virial, access_location::device, access_mode::readwrite);

    // access the parameters
    ArrayHandle<Scalar> d_rcutsq(this->m_rcutsq, access_location::device, access_mode::read);

    ArrayHandle<unsigned int> d_n_triang(this->m_mesh_data->getNNeighArray(),
    				    access_location::device,
    				    access_mode::read);
    ArrayHandle<unsigned int> d_trianglist(this->m_mesh_data->getTriangleList(),
    				  access_location::device,
    				  access_mode::read);
    //     Index2D nli = m_nlist->getNListIndexer();
    ArrayHandle<unsigned int> d_head_triang(this->m_mesh_data->getHeadList(),
    				access_location::device,
    				access_mode::read);

    BoxDim box = this->m_pdata->getBox();

    // access parameters
    ArrayHandle<typename evaluator::param_type> d_params(this->m_params,
                                                         access_location::device,
                                                         access_mode::read);

    PDataFlags flags = this->m_pdata->getFlags();
    this->m_exec_conf->setDevice();


    m_tuner_particles->begin();

    kernel::gpu_compute_mesh_triangle_particles_force<evaluator>(
        	    kernel::meshtriangle_args_t(d_force.data,
                                              d_virial.data,
                                              this->m_virial.getPitch(),
                                              this->m_pdata->getN(),
                                	      this->m_pdata->getMaxN(),
                                              d_pos.data,
                                              box,
                                              d_n_neigh.data,
                                              d_nlist.data,
                                              d_head_list.data,
                                              d_rcutsq.data,
                                              this->m_nlist->getNListArray().getPitch(),
                                              this->m_pdata->getNTypes(),
        				      m_tuner_particles->getParam()[0],
        				      flags[pdata_flag::pressure_tensor],
        				      this->m_exec_conf->dev_prop),
                                              d_params.data,
                                              d_tag.data,
                                              d_n_triang.data,
                                              d_trianglist.data,
                                              d_head_triang.data);

    if (this->m_exec_conf->isCUDAErrorCheckingEnabled())
        CHECK_CUDA_ERROR();
    m_tuner_particles->end();
}

namespace detail
    {
//! Exports the PotentialMeshTriangle class to python
/*! \param name Name of the class in the exported python module
    \tparam T Evaluator type to export.
*/
template<class T> void export_PotentialMeshTriangleGPU(pybind11::module& m, const std::string& name)
    {
    pybind11::class_<PotentialMeshTriangleGPU<T>, PotentialMeshTriangle<T>, std::shared_ptr<PotentialMeshTriangleGPU<T>>>(
        m,
        name.c_str())
        .def(pybind11::init<std::shared_ptr<SystemDefinition>,std::shared_ptr<NeighborList>,std::shared_ptr<MeshDefinition>>());
    }

    } // end namespace detail
    } // end namespace md
    } // end namespace hoomd

#endif // ENABLE_HIP
#endif // __POTENTIALMESHTRIANGLEPAIR_GPU_H__
