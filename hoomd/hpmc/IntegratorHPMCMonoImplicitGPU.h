// Copyright (c) 2009-2019 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.

#ifndef __HPMC_MONO_IMPLICIT_GPU_H__
#define __HPMC_MONO_IMPLICIT_GPU_H__

#ifdef ENABLE_CUDA

#include "IntegratorHPMCMonoImplicit.h"
#include "IntegratorHPMCMonoGPU.cuh"
#include "IntegratorHPMCMonoImplicitGPU.cuh"
#include "hoomd/Autotuner.h"

#include "hoomd/GPUVector.h"

#include <cuda_runtime.h>

/*! \file IntegratorHPMCMonoImplicitGPU.h
    \brief Defines the template class for HPMC with implicit generated depletant solvent on the GPU
    \note This header cannot be compiled by nvcc
*/

#ifdef NVCC
#error This header cannot be compiled by nvcc
#endif

#include <hoomd/extern/pybind/include/pybind11/pybind11.h>

namespace hpmc
{

//! Template class for HPMC update with implicit depletants on the GPU
/*!
    Depletants are generated randomly on the fly according to the semi-grand canonical ensemble.

    The penetrable depletants model is simulated.

    \ingroup hpmc_integrators
*/
template< class Shape >
class IntegratorHPMCMonoImplicitGPU : public IntegratorHPMCMonoImplicit<Shape>
    {
    public:
        //! Construct the integrator
        IntegratorHPMCMonoImplicitGPU(std::shared_ptr<SystemDefinition> sysdef,
                              std::shared_ptr<CellList> cl,
                              unsigned int seed);
        //! Destructor
        virtual ~IntegratorHPMCMonoImplicitGPU();

        //! Set autotuner parameters
        /*! \param enable Enable/disable autotuning
            \param period period (approximate) in time steps when returning occurs
        */
        virtual void setAutotunerParams(bool enable, unsigned int period)
            {
            // call base class method first
            unsigned int ndim = this->m_sysdef->getNDimensions();
            if (ndim == 3)
                {
                m_tuner_update->setPeriod(period*this->m_nselect*8);
                m_tuner_implicit->setPeriod(period*this->m_nselect*8);
                m_tuner_reinsert->setPeriod(period*this->m_nselect*8);
                }
            else
                {
                m_tuner_update->setPeriod(period*this->m_nselect*4);
                m_tuner_implicit->setPeriod(period*this->m_nselect*4);
                m_tuner_reinsert->setPeriod(period*this->m_nselect*4);
                }

            m_tuner_update->setEnabled(enable);
            m_tuner_reinsert->setEnabled(enable);
            m_tuner_implicit->setEnabled(enable);

            m_tuner_excell_block_size->setPeriod(period);
            m_tuner_excell_block_size->setEnabled(enable);

            }

    protected:
        std::shared_ptr<CellList> m_cl;           //!< Cell list
        GPUArray<unsigned int> m_cell_sets;   //!< List of cells active during each subsweep
        Index2D m_cell_set_indexer;           //!< Indexer into the cell set array
        uint3 m_last_dim;                     //!< Dimensions of the cell list on the last call to update
        unsigned int m_last_nmax;             //!< Last cell list NMax value allocated in excell
        detail::UpdateOrder m_cell_set_order; //!< Update order for cell sets

        GPUVector<Scalar4> m_old_postype;                            //!< List of old particle positions
        GPUVector<Scalar4> m_old_orientation;                    //!< List of old particle orientations

        GPUArray<unsigned int> m_excell_idx;  //!< Particle indices in expanded cells
        GPUArray<unsigned int> m_excell_size; //!< Number of particles in each expanded cell
        Index2D m_excell_list_indexer;        //!< Indexer to access elements of the excell_idx list

        std::unique_ptr<Autotuner> m_tuner_update;             //!< Autotuner for the update step group and block sizes
        std::unique_ptr<Autotuner> m_tuner_excell_block_size;  //!< Autotuner for excell block_size
        std::unique_ptr<Autotuner> m_tuner_implicit;           //!< Autotuner for the depletant overlap check
        std::unique_ptr<Autotuner> m_tuner_reinsert;      //!< Autotuner for the acceptance probability calculation
        mgpu::ContextPtr m_mgpu_context;              //!< MGPU context


        GPUArray<curandState_t> m_curand_state_cell;               //!< Array of cuRAND states per active cell
        GPUArray<curandState_t> m_curand_state_cell_new;           //!< Array of cuRAND states per active cell after update
        GPUArray<unsigned int> m_overlap_cell;                   //!< Flag per cell to indicate overlap
        GPUArray<float> m_log_boltzmann;                     //!< Log of successful reinsertion attempts (new position)
        GPUArray<unsigned int> m_n_success_zero;             //!< Whether number of successful reinsertions is zero (new)
        GPUArray<curandDiscreteDistribution_t> m_poisson_dist; //!< Handles for the poisson distribution histogram
        std::vector<bool> m_poisson_dist_created;               //!< Flag to indicate if Poisson distribution has been initialized

        GPUArray<unsigned int> m_active_cell_ptl_idx;  //!< List of update particle indices per active cell
        GPUArray<unsigned int> m_active_cell_accept;   //!< List of accept/reject flags per active cell
        GPUArray<unsigned int> m_active_cell_move_type_translate;   //!< Type of move proposed in active cell

        GPUArray<unsigned int> m_overlap_cell_scan;                 //!< Scan of m_overlap_cell
        GPUVector<unsigned int> m_depletant_active_cell;            //!< Lookup of active cell idx per overlapping depletant
        GPUVector<unsigned int> m_n_success_forward;                //!< Successful forward-insertions
        GPUVector<unsigned int> m_n_overlap_shape_forward;          //!< Forward-insertions
        GPUVector<unsigned int> m_n_success_reverse;                //!< Successful reverse-insertions
        GPUVector<unsigned int> m_n_overlap_shape_reverse;          //!< Forward-insertions
        GPUVector<float> m_depletant_lnb;                           //!< Configurational bias weights

        cudaStream_t m_stream;                                  //! GPU kernel stream

        //! Take one timestep forward
        virtual void update(unsigned int timestep);

        //! Initialize the Poisson distributions
        virtual void initializePoissonDistribution();

        //! Set up cell sets
        virtual void initializeCellSets();

        //! Set up excell_list
        virtual void initializeExcellMem();

        //! Update the cell width
        virtual void updateCellWidth();
    };

/*! \param sysdef System definition
    \param cl Cell list
    \param seed Random number generator seed

    NOTE: only 3d supported at this time
    */

template< class Shape >
IntegratorHPMCMonoImplicitGPU< Shape >::IntegratorHPMCMonoImplicitGPU(std::shared_ptr<SystemDefinition> sysdef,
                                                                   std::shared_ptr<CellList> cl,
                                                                   unsigned int seed)
    : IntegratorHPMCMonoImplicit<Shape>(sysdef, seed, 0), m_cl(cl), m_cell_set_order(seed+this->m_exec_conf->getRank())
    {
    this->m_exec_conf->msg->notice(5) << "Constructing IntegratorHPMCImplicitGPU" << std::endl;

    this->m_cl->setRadius(1);
    this->m_cl->setComputeTDB(false);
    this->m_cl->setFlagType();
    this->m_cl->setComputeIdx(true);

    // require that cell lists have an even number of cells along each direction
    this->m_cl->setMultiple(2);

    // set last dim to a bogus value so that it will re-init on the first call
    m_last_dim = make_uint3(0xffffffff, 0xffffffff, 0xffffffff);
    m_last_nmax = 0xffffffff;

    GPUArray<unsigned int> excell_size(0, this->m_exec_conf);
    m_excell_size.swap(excell_size);

    GPUArray<unsigned int> excell_idx(0, this->m_exec_conf);
    m_excell_idx.swap(excell_idx);

    GPUVector<Scalar4> old_postype(this->m_exec_conf);
    m_old_postype.swap(old_postype);

    GPUVector<Scalar4> old_orientation(this->m_exec_conf);
    m_old_orientation.swap(old_orientation);

    GPUVector<unsigned int> depletant_active_cell(this->m_exec_conf);
    m_depletant_active_cell.swap(depletant_active_cell);

    GPUVector<unsigned int> n_success_forward(this->m_exec_conf);
    m_n_success_forward.swap(n_success_forward);

    GPUVector<unsigned int> n_overlap_shape_forward(this->m_exec_conf);
    m_n_overlap_shape_forward.swap(n_overlap_shape_forward);

    GPUVector<unsigned int> n_success_reverse(this->m_exec_conf);
    m_n_success_reverse.swap(n_success_reverse);

    GPUVector<unsigned int> n_overlap_shape_reverse(this->m_exec_conf);
    m_n_overlap_shape_reverse.swap(n_overlap_shape_reverse);

    GPUVector<float> depletant_lnb(this->m_exec_conf);
    m_depletant_lnb.swap(depletant_lnb);

    // initialize the autotuners
    // the full block size, stride and group size matrix is searched,
    // encoded as block_size*1000000 + stride*100 + group_size.

    // parameters for count_overlaps kernel
    std::vector<unsigned int> valid_params;

    // parameters for HPMC update kernel
    std::vector<unsigned int> valid_params_update;

    cudaDeviceProp dev_prop = this->m_exec_conf->dev_prop;

    unsigned int max_tpp = this->m_exec_conf->dev_prop.warpSize;

    if (Shape::isParallel())
        {
        for (unsigned int block_size =dev_prop.warpSize; block_size <= (unsigned int) dev_prop.maxThreadsPerBlock; block_size +=dev_prop.warpSize)
            {
            unsigned int s=1;
            while (s <= (unsigned int)dev_prop.warpSize)
                {
                unsigned int stride = 1;
                while (stride <= block_size)
                    {
                    // for parallel overlap checks, use 3d-layout where blockDim.z is limited
                    if (block_size % (s*stride) == 0 && block_size/(s*stride) <= (unsigned int) dev_prop.maxThreadsDim[2])
                        valid_params_update.push_back(block_size*1000000 + stride*100 + s);

                    // increment stride in powers of two
                    stride *= 2;
                    }
                s++;
                }
            }
        }
    else
        {
        // for serial overlap checks, force stride=1. And groups no longer need to evenly divide into warps: only into
        // blocks
        unsigned int stride = 1;

        for (unsigned int block_size = dev_prop.warpSize; block_size <= (unsigned int) dev_prop.maxThreadsPerBlock; block_size += dev_prop.warpSize)
            {
            for (unsigned int group_size=1; group_size <= (unsigned int)dev_prop.warpSize; group_size++)
                {
                if ((block_size % group_size) == 0)
                    valid_params_update.push_back(block_size*1000000 + stride*100 + group_size);
                }
            }
        }

    if (Shape::isParallel())
        {
        for (unsigned int block_size = (unsigned int) dev_prop.warpSize; block_size <= (unsigned int) dev_prop.maxThreadsPerBlock; block_size += (unsigned int) dev_prop.warpSize)
            {
            unsigned int s=1;
            while (s <= max_tpp)
                {
                unsigned int stride = 1;
                while (stride <=max_tpp/s)
                    {
                    // for parallel overlap checks, use 3d-layout where blockDim.z is limited
                    if (block_size/s/stride <= (unsigned int) dev_prop.maxThreadsDim[2])
                        {
                        valid_params.push_back(block_size*1000000 + stride*100 + s);
                        }
                    stride*=2;
                    }
                s = s * 2;
                }
            }
        }
    else
        {
        // for serial overlap checks, force stride=1. A group needs to be smaller than a warp and a power of two
        unsigned int stride = 1;

        for (unsigned int block_size = (unsigned int) dev_prop.warpSize; block_size <= (unsigned int) dev_prop.maxThreadsPerBlock; block_size += (unsigned int) dev_prop.warpSize)
            {
            for (unsigned int group_size=1; group_size <= (unsigned int)max_tpp; group_size*=2)
                {
                if ((block_size % group_size) == 0)
                    valid_params.push_back(block_size*1000000 + stride*100 + group_size);
                }
            }
        }

    m_tuner_update.reset(new Autotuner(valid_params_update, 5, 1000000, "hpmc_update", this->m_exec_conf));
    m_tuner_excell_block_size.reset(new Autotuner(dev_prop.warpSize, dev_prop.maxThreadsPerBlock, dev_prop.warpSize, 5, 1000000, "hpmc_excell_block_size", this->m_exec_conf));
    m_tuner_implicit.reset(new Autotuner(valid_params, 5, 1000000, "hpmc_implicit_count_overlaps", this->m_exec_conf));
    m_tuner_reinsert.reset(new Autotuner(valid_params, 5, 1000000, "hpmc_implicit_reinsert", this->m_exec_conf));

    GPUArray<hpmc_implicit_counters_t> implicit_count(1,this->m_exec_conf);
    this->m_implicit_count.swap(implicit_count);

    GPUArray<curandDiscreteDistribution_t> poisson_dist(1,this->m_exec_conf);
    m_poisson_dist.swap(poisson_dist);

    m_poisson_dist_created.resize(this->m_pdata->getNTypes(), false);

    // create a CUDA stream for kernel execution
    cudaStreamCreate(&m_stream);
    CHECK_CUDA_ERROR();

    // create at ModernGPU context
    m_mgpu_context = mgpu::CreateCudaDeviceAttachStream(m_stream);
    }

//! Destructor
template< class Shape >
IntegratorHPMCMonoImplicitGPU< Shape >::~IntegratorHPMCMonoImplicitGPU()
    {
    // destroy the registered poisson RNG's
    ArrayHandle<curandDiscreteDistribution_t> h_poisson_dist(m_poisson_dist, access_location::host, access_mode::read);
    for (unsigned int type = 0; type < this->m_pdata->getNTypes(); ++type)
        {
        if (m_poisson_dist_created[type])
            {
            curandDestroyDistribution(h_poisson_dist.data[type]);
            }
        }

    cudaStreamDestroy(m_stream);
    CHECK_CUDA_ERROR();
    }

template< class Shape >
void IntegratorHPMCMonoImplicitGPU< Shape >::update(unsigned int timestep)
    {
    if (this->m_patch && !this->m_patch_log)
        {
        this->m_exec_conf->msg->error() << "GPU simulations with patches are unsupported." << std::endl;
        throw std::runtime_error("Error during implicit depletant integration\n");
        }

    IntegratorHPMC::update(timestep);

    // update poisson distributions
    if (this->m_need_initialize_poisson)
        {
        this->updatePoissonParameters();
        initializePoissonDistribution();
        this->m_need_initialize_poisson = false;
        }

        {
        ArrayHandle<hpmc_implicit_counters_t> h_implicit_counters(this->m_implicit_count, access_location::host, access_mode::readwrite);
        this->m_implicit_count_step_start = h_implicit_counters.data[0];
        }

    #ifndef ENABLE_HPMC_REINSERT
    if (this->m_n_trial)
        {
        throw std::runtime_error("ntrial > 0 not supported on the GPU. For CUDA architecture <=6.0, recompile with ENABLE_HPMC_REINSERT=ON.");
        }
    #endif

    // check if we are below a minimum image convention box size
    BoxDim box = this->m_pdata->getBox();
    Scalar3 npd = box.getNearestPlaneDistance();

    // Currently only 3D
    if (this->m_sysdef->getNDimensions() != 3)
        {
        this->m_exec_conf->msg->error() << "At this time, implicit depletant simulations are supported in 3D only." << std::endl;
        throw std::runtime_error("Error performing HPMC update");
        }

    if ((box.getPeriodic().x && npd.x <= this->m_nominal_width*2) ||
        (box.getPeriodic().y && npd.y <= this->m_nominal_width*2) ||
        (this->m_sysdef->getNDimensions() == 3 && box.getPeriodic().z && npd.z <= this->m_nominal_width*2))
        {
        this->m_exec_conf->msg->error() << "Simulation box too small for implicit depletant simulations on GPU - increase it so the minimum image convention works" << std::endl;
        throw std::runtime_error("Error performing HPMC update");
        }


    // update the cell list
    this->m_cl->compute(timestep);

    // start the profile
    if (this->m_prof) this->m_prof->push(this->m_exec_conf, "HPMC");

    // rng for shuffle and grid shift
    hoomd::detail::Saru rng(this->m_seed, timestep, 0xf4a3210e);

    // if the cell list is a different size than last time, reinitialize the cell sets list
    uint3 cur_dim = this->m_cl->getDim();
    unsigned int last_n_active = this->m_cell_set_indexer.getW();
    if (this->m_last_dim.x != cur_dim.x || this->m_last_dim.y != cur_dim.y || this->m_last_dim.z != cur_dim.z)
        {
        this->initializeCellSets();
        this->initializeExcellMem();

        this->m_last_dim = cur_dim;
        this->m_last_nmax = this->m_cl->getNmax();

        // initialize the cell set update order
        this->m_cell_set_order.resize(this->m_cell_set_indexer.getH());
        }

    // initialize RNG states
    if (this->m_cell_set_indexer.getW() != last_n_active)
        {
        GPUArray<curandState_t> curand_state_cell(this->m_cell_set_indexer.getW(), this->m_exec_conf);
        m_curand_state_cell.swap(curand_state_cell);

        GPUArray<curandState_t> curand_state_cell_new(this->m_cell_set_indexer.getW(), this->m_exec_conf);
        m_curand_state_cell_new.swap(curand_state_cell_new);

        GPUArray<unsigned int> overlap_cell(this->m_cell_set_indexer.getW(), this->m_exec_conf);
        m_overlap_cell.swap(overlap_cell);

        GPUArray<unsigned int> overlap_cell_scan(this->m_cell_set_indexer.getW(), this->m_exec_conf);
        m_overlap_cell_scan.swap(overlap_cell_scan);

        GPUArray<float> log_boltzmann(this->m_cell_set_indexer.getW(), this->m_exec_conf);
        m_log_boltzmann.swap(log_boltzmann);

        GPUArray<unsigned int> n_success_zero(this->m_cell_set_indexer.getW(), this->m_exec_conf);
        m_n_success_zero.swap(n_success_zero);

        GPUArray<unsigned int> active_cell_ptl_idx(this->m_cell_set_indexer.getW(), this->m_exec_conf);
        m_active_cell_ptl_idx.swap(active_cell_ptl_idx);

        GPUArray<unsigned int> active_cell_accept(this->m_cell_set_indexer.getW(), this->m_exec_conf);
        m_active_cell_accept.swap(active_cell_accept);

        GPUArray<unsigned int> active_cell_move_type_translate(this->m_cell_set_indexer.getW(), this->m_exec_conf);
        m_active_cell_move_type_translate.swap(active_cell_move_type_translate);
        }

    // if only NMax changed, only need to reallocate excell memory
    if (this->m_last_nmax != this->m_cl->getNmax())
        {
        this->initializeExcellMem();
        this->m_last_nmax = this->m_cl->getNmax();
        }

    // test if we are in domain decomposition mode
    bool domain_decomposition = false;
#ifdef ENABLE_MPI
    if (this->m_comm)
        domain_decomposition = true;
#endif

        {
        // access the particle data
        ArrayHandle<Scalar4> d_postype(this->m_pdata->getPositions(), access_location::device, access_mode::readwrite);
        ArrayHandle<Scalar4> d_orientation(this->m_pdata->getOrientationArray(), access_location::device, access_mode::readwrite);
        ArrayHandle<int3> d_image(this->m_pdata->getImages(), access_location::device, access_mode::readwrite);

        ArrayHandle< unsigned int > d_cell_sets(this->m_cell_sets, access_location::device, access_mode::read);

        ArrayHandle< unsigned int > d_excell_idx(this->m_excell_idx, access_location::device, access_mode::readwrite);
        ArrayHandle< unsigned int > d_excell_size(this->m_excell_size, access_location::device, access_mode::readwrite);

        // access the parameters and interaction matrix
        const std::vector<typename Shape::param_type, managed_allocator<typename Shape::param_type> > & params = this->getParams();
        ArrayHandle<unsigned int> d_overlaps(this->m_overlaps, access_location::device, access_mode::read);

        // access the move sizes by type
        ArrayHandle<Scalar> d_d(this->m_d, access_location::device, access_mode::read);
        ArrayHandle<Scalar> d_a(this->m_a, access_location::device, access_mode::read);

        Scalar3 ghost_width = this->m_cl->getGhostWidth();
        Scalar3 ghost_fraction = this->m_nominal_width / npd;

        // compute the ceiling of the average number of particles in each cell, accounting for ghost particles
        int ncells = cur_dim.x * cur_dim.y * cur_dim.z;
        int particles_per_cell = int(ceil(double(this->m_pdata->getN() + this->m_pdata->getNGhosts()) / double(ncells)));
            {
            // access the cell list data
            ArrayHandle<unsigned int> d_cell_size(this->m_cl->getCellSizeArray(), access_location::device, access_mode::read);
            ArrayHandle<Scalar4> d_cell_xyzf(this->m_cl->getXYZFArray(), access_location::device, access_mode::read);
            ArrayHandle<Scalar4> d_cell_orientation(this->m_cl->getOrientationArray(), access_location::device, access_mode::read);
            ArrayHandle<unsigned int> d_cell_idx(this->m_cl->getIndexArray(), access_location::device, access_mode::read);
            ArrayHandle<unsigned int> d_cell_adj(this->m_cl->getCellAdjArray(), access_location::device, access_mode::read);

            // update the expanded cells
            this->m_tuner_excell_block_size->begin();
            detail::gpu_hpmc_excell(d_excell_idx.data,
                                    d_excell_size.data,
                                    this->m_excell_list_indexer,
                                    d_cell_idx.data,
                                    d_cell_size.data,
                                    d_cell_adj.data,
                                    this->m_cl->getCellIndexer(),
                                    this->m_cl->getCellListIndexer(),
                                    this->m_cl->getCellAdjIndexer(),
                                    this->m_tuner_excell_block_size->getParam());
            if (this->m_exec_conf->isCUDAErrorCheckingEnabled())
                CHECK_CUDA_ERROR();

            this->m_tuner_excell_block_size->end();
            }

        // access the cell list data
        ArrayHandle<unsigned int> d_cell_size(this->m_cl->getCellSizeArray(), access_location::device, access_mode::read);
        ArrayHandle<Scalar4> d_cell_xyzf(this->m_cl->getXYZFArray(), access_location::device, access_mode::read);
        ArrayHandle<Scalar4> d_cell_orientation(this->m_cl->getOrientationArray(), access_location::device, access_mode::read);
        ArrayHandle<unsigned int> d_cell_idx(this->m_cl->getIndexArray(), access_location::device, access_mode::read);
        ArrayHandle<unsigned int> d_cell_adj(this->m_cl->getCellAdjArray(), access_location::device, access_mode::read);

        this->m_old_postype.resize(this->m_pdata->getMaxN());
        this->m_old_orientation.resize(this->m_pdata->getMaxN());

        // process as many depletants in parallel as the maximum possible number of depletants
        Scalar lambda_max(0.0);
        for (unsigned int i = 0; i < this->m_pdata->getNTypes(); ++i)
            {
            Scalar lambda = this->m_lambda[i];
            if (lambda > lambda_max)
                {
                lambda_max = lambda;
                }
            }
        unsigned int groups_per_cell = ((unsigned int) lambda_max)+1;

        unsigned int n_reinsert = 0;

        // on first iteration, synchronize GPU execution stream and update shape parameters
        bool first = true;

        for (unsigned int i = 0; i < this->m_nselect*particles_per_cell; i++)
            {
            // loop over cell sets in a shuffled order
            this->m_cell_set_order.shuffle(timestep,i);

            for (unsigned int j = 0; j < this->m_cell_set_indexer.getH(); j++)
                {
                unsigned cur_set = this->m_cell_set_order[j];

                // save old positions
                ArrayHandle<Scalar4> d_old_postype(this->m_old_postype, access_location::device, access_mode::overwrite);
                ArrayHandle<Scalar4> d_old_orientation(this->m_old_orientation, access_location::device, access_mode::overwrite);

                cudaMemcpy(d_old_postype.data, d_postype.data, sizeof(Scalar4)*(this->m_pdata->getN()+this->m_pdata->getNGhosts()), cudaMemcpyDeviceToDevice);
                cudaMemcpy(d_old_orientation.data, d_orientation.data, sizeof(Scalar4)*(this->m_pdata->getN()+this->m_pdata->getNGhosts()), cudaMemcpyDeviceToDevice);

                // flags about updated particles
                ArrayHandle<unsigned int> d_active_cell_ptl_idx(m_active_cell_ptl_idx, access_location::device, access_mode::overwrite);
                ArrayHandle<unsigned int> d_active_cell_accept(m_active_cell_accept, access_location::device, access_mode::overwrite);
                ArrayHandle<unsigned int> d_active_cell_move_type_translate(m_active_cell_move_type_translate, access_location::device, access_mode::overwrite);

                ArrayHandle<hpmc_counters_t> d_counters(this->m_count_total, access_location::device, access_mode::readwrite);

                // move particles
                this->m_tuner_update->begin();

                unsigned int param = this->m_tuner_update->getParam();
                unsigned int block_size = param / 1000000;
                unsigned int stride = (param % 1000000 ) / 100;
                unsigned int group_size = param % 100;
                detail::gpu_hpmc_update<Shape> (
                    detail::hpmc_args_t(d_postype.data,
                        d_orientation.data,
                        d_counters.data,
                        d_cell_idx.data,
                        d_cell_size.data,
                        d_excell_idx.data,
                        d_excell_size.data,
                        this->m_cl->getCellIndexer(),
                        this->m_cl->getCellListIndexer(),
                        this->m_excell_list_indexer,
                        this->m_cl->getDim(),
                        ghost_width,
                        &d_cell_sets.data[this->m_cell_set_indexer(0,cur_set)],
                        this->m_cell_set_indexer.getW(),
                        this->m_pdata->getN(),
                        this->m_pdata->getNTypes(),
                        this->m_seed + this->m_exec_conf->getRank(),
                        d_d.data,
                        d_a.data,
                        d_overlaps.data,
                        this->m_overlap_idx,
                        this->m_move_ratio,
                        timestep,
                        this->m_sysdef->getNDimensions(),
                        box,
                        i+particles_per_cell*this->m_nselect*(3*j),
                        ghost_fraction,
                        domain_decomposition,
                        block_size,
                        stride,
                        group_size,
                        this->m_hasOrientation,
                        this->m_pdata->getMaxN(),
                        this->m_exec_conf->dev_prop,
                        first,
                        m_stream,
                        (lambda_max > 0.0) ? d_active_cell_ptl_idx.data : 0,
                        (lambda_max > 0.0) ? d_active_cell_accept.data : 0,
                        (lambda_max > 0.0) ? d_active_cell_move_type_translate.data : 0),
                    params.data());

                if (this->m_exec_conf->isCUDAErrorCheckingEnabled())
                    CHECK_CUDA_ERROR();

                this->m_tuner_update->end();

                if (lambda_max > Scalar(0.0))
                    {
                    if (this->m_prof) this->m_prof->push(this->m_exec_conf,"Depletants");

                    // RNG state
                    ArrayHandle<curandState_t> d_curand_state_cell(this->m_curand_state_cell, access_location::device, access_mode::readwrite);
                    ArrayHandle<curandState_t> d_curand_state_cell_new(this->m_curand_state_cell_new, access_location::device, access_mode::readwrite);

                    // overlap flags
                    ArrayHandle<unsigned int> d_overlap_cell(this->m_overlap_cell, access_location::device, access_mode::readwrite);
                    ArrayHandle<unsigned int> d_overlap_cell_scan(this->m_overlap_cell_scan, access_location::device, access_mode::readwrite);

                    ArrayHandle<float> d_log_boltzmann(m_log_boltzmann, access_location::device,access_mode::overwrite);
                    ArrayHandle<unsigned int> d_n_success_zero(m_n_success_zero,access_location::device,access_mode::overwrite);

                    // min/max diameter of insertion sphere
                    ArrayHandle<Scalar> d_d_min(this->m_d_min, access_location::device, access_mode::read);
                    ArrayHandle<Scalar> d_d_max(this->m_d_max, access_location::device, access_mode::read);

                    // Poisson distribution
                    ArrayHandle<curandDiscreteDistribution_t> d_poisson_dist(m_poisson_dist, access_location::device, access_mode::read);

                    // total number of inserted depletants that overlap
                    unsigned int n_overlaps = 0;

                        {
                        // counters
                        ArrayHandle<hpmc_implicit_counters_t> d_implicit_count(this->m_implicit_count, access_location::device, access_mode::readwrite);

                        // Kernel driver arguments
                        unsigned int param = m_tuner_implicit->getParam();
                        unsigned int block_size = param / 1000000;
                        unsigned int stride = (param % 1000000 ) / 100;
                        unsigned int group_size = param % 100;

                        m_tuner_implicit->begin();

                        // invoke kernel
                        detail::gpu_hpmc_implicit_count_overlaps<Shape>(
                            detail::hpmc_implicit_args_t(d_postype.data,
                                d_orientation.data,
                                d_old_postype.data,
                                d_old_orientation.data,
                                d_cell_idx.data,
                                d_cell_size.data,
                                d_excell_idx.data,
                                d_excell_size.data,
                                this->m_cl->getCellIndexer(),
                                this->m_cl->getCellListIndexer(),
                                this->m_excell_list_indexer,
                                this->m_cl->getDim(),
                                ghost_width,
                                &d_cell_sets.data[this->m_cell_set_indexer(0,cur_set)],
                                this->m_cell_set_indexer.getW(),
                                this->m_pdata->getN(),
                                this->m_pdata->getNTypes(),
                                this->m_seed + this->m_exec_conf->getRank(),
                                d_overlaps.data,
                                this->m_overlap_idx,
                                timestep,
                                this->m_sysdef->getNDimensions(),
                                box,
                                i+particles_per_cell*this->m_nselect*(3*j+1),
                                block_size,
                                stride,
                                group_size,
                                this->m_hasOrientation,
                                this->m_pdata->getMaxN(),
                                this->m_exec_conf->dev_prop,
                                d_curand_state_cell.data,
                                d_curand_state_cell_new.data,
                                this->m_type,
                                d_counters.data,
                                d_implicit_count.data,
                                d_poisson_dist.data,
                                d_overlap_cell.data,
                                d_overlap_cell_scan.data,
                                groups_per_cell,
                                d_active_cell_ptl_idx.data,
                                d_active_cell_accept.data,
                                d_active_cell_move_type_translate.data,
                                d_log_boltzmann.data,
                                d_n_success_zero.data,
                                this->m_n_trial,
                                0,
                                n_overlaps,
                                0, 0, 0, 0, 0,
                                d_d_min.data,
                                d_d_max.data,
                                first,
                                m_mgpu_context,
                                m_stream),
                            params.data());

                        if (this->m_exec_conf->isCUDAErrorCheckingEnabled())
                            CHECK_CUDA_ERROR();

                        m_tuner_implicit->end();
                        }

                    if (this->m_n_trial)
                        {
                        // resize O(N_depletant) memory
                        m_depletant_active_cell.resize(n_overlaps);
                        m_n_success_forward.resize(n_overlaps);
                        m_n_overlap_shape_forward.resize(n_overlaps);
                        m_n_success_reverse.resize(n_overlaps);
                        m_n_overlap_shape_reverse.resize(n_overlaps);
                        m_depletant_lnb.resize(n_overlaps);
                        }

                        {
                        // counters
                        ArrayHandle<hpmc_implicit_counters_t> d_implicit_count(this->m_implicit_count, access_location::device, access_mode::readwrite);

                        ArrayHandle<unsigned int> d_depletant_active_cell(m_depletant_active_cell, access_location::device, access_mode::overwrite);
                        ArrayHandle<unsigned int> d_n_success_forward(m_n_success_forward, access_location::device, access_mode::overwrite);
                        ArrayHandle<unsigned int> d_n_overlap_shape_forward(m_n_overlap_shape_forward, access_location::device, access_mode::overwrite);
                        ArrayHandle<unsigned int> d_n_success_reverse(m_n_success_reverse, access_location::device, access_mode::overwrite);
                        ArrayHandle<unsigned int> d_n_overlap_shape_reverse(m_n_overlap_shape_reverse, access_location::device, access_mode::overwrite);
                        ArrayHandle<float> d_depletant_lnb(m_depletant_lnb, access_location::device, access_mode::overwrite);

                        // Kernel driver arguments
                        param = m_tuner_reinsert->getParam();
                        block_size = param / 1000000;
                        stride = (param % 1000000 ) / 100;
                        group_size = param % 100;

                        m_tuner_reinsert->begin();

                        // apply acceptance/rejection criterion
                        detail::gpu_hpmc_implicit_accept_reject<Shape>(
                            detail::hpmc_implicit_args_t(d_postype.data,
                                d_orientation.data,
                                d_old_postype.data,
                                d_old_orientation.data,
                                d_cell_idx.data,
                                d_cell_size.data,
                                d_excell_idx.data,
                                d_excell_size.data,
                                this->m_cl->getCellIndexer(),
                                this->m_cl->getCellListIndexer(),
                                this->m_excell_list_indexer,
                                this->m_cl->getDim(),
                                ghost_width,
                                &d_cell_sets.data[this->m_cell_set_indexer(0,cur_set)],
                                this->m_cell_set_indexer.getW(),
                                this->m_pdata->getN(),
                                this->m_pdata->getNTypes(),
                                this->m_seed + this->m_exec_conf->getRank(),
                                d_overlaps.data,
                                this->m_overlap_idx,
                                timestep,
                                this->m_sysdef->getNDimensions(),
                                box,
                                i+particles_per_cell*this->m_nselect*(3*j+2),
                                block_size,
                                stride,
                                group_size,
                                this->m_hasOrientation,
                                this->m_pdata->getMaxN(),
                                this->m_exec_conf->dev_prop,
                                d_curand_state_cell.data,
                                d_curand_state_cell_new.data,
                                this->m_type,
                                d_counters.data,
                                d_implicit_count.data,
                                d_poisson_dist.data,
                                d_overlap_cell.data,
                                d_overlap_cell_scan.data,
                                groups_per_cell,
                                d_active_cell_ptl_idx.data,
                                d_active_cell_accept.data,
                                d_active_cell_move_type_translate.data,
                                d_log_boltzmann.data,
                                d_n_success_zero.data,
                                this->m_n_trial,
                                d_depletant_active_cell.data,
                                n_overlaps,
                                d_n_success_forward.data,
                                d_n_overlap_shape_forward.data,
                                d_n_success_reverse.data,
                                d_n_overlap_shape_reverse.data,
                                d_depletant_lnb.data,
                                d_d_min.data,
                                d_d_max.data,
                                first,
                                m_mgpu_context,
                                m_stream),
                            params.data());

                        if (this->m_exec_conf->isCUDAErrorCheckingEnabled())
                            CHECK_CUDA_ERROR();

                        m_tuner_reinsert->end();
                        }

                    // increment counter
                    n_reinsert += n_overlaps * this->m_n_trial;

                    if (this->m_prof) this->m_prof->pop(this->m_exec_conf);
                    }

                first = false;
                } // end loop over cell sets
            } // end loop nselect*particles_per_cell

            {
            ArrayHandle<hpmc_implicit_counters_t> h_implicit_count(this->m_implicit_count, access_location::host, access_mode::readwrite);
            h_implicit_count.data->reinsert_count += n_reinsert;
            }

        // shift particles
        Scalar3 shift = make_scalar3(0,0,0);
        shift.x = rng.s(-this->m_nominal_width/Scalar(2.0),this->m_nominal_width/Scalar(2.0));
        shift.y = rng.s(-this->m_nominal_width/Scalar(2.0),this->m_nominal_width/Scalar(2.0));
        if (this->m_sysdef->getNDimensions() == 3)
            {
            shift.z = rng.s(-this->m_nominal_width/Scalar(2.0),this->m_nominal_width/Scalar(2.0));
            }

        detail::gpu_hpmc_shift(d_postype.data,
                               d_image.data,
                               this->m_pdata->getN(),
                               box,
                               shift,
                               128);

        // update the particle data origin
        this->m_pdata->translateOrigin(shift);

        if (this->m_exec_conf->isCUDAErrorCheckingEnabled())
            CHECK_CUDA_ERROR();
        } // end of ArrayHandle scope

    if (this->m_prof) this->m_prof->pop(this->m_exec_conf);

    this->communicate(true);

    // all particle have been moved, the aabb tree is now invalid
    this->m_aabb_tree_invalid = true;
    }

template<class Shape>
void IntegratorHPMCMonoImplicitGPU< Shape >::initializePoissonDistribution()
    {
    // resize GPUArray
    m_poisson_dist.resize(this->m_pdata->getNTypes());
    m_poisson_dist_created.resize(this->m_pdata->getNTypes(), false);

    ArrayHandle<curandDiscreteDistribution_t> h_poisson_dist(m_poisson_dist, access_location::host, access_mode::readwrite);
    for (unsigned int i_type = 0; i_type < this->m_pdata->getNTypes(); ++i_type)
        {
        // parameter for Poisson distribution
        Scalar lambda = this->m_lambda[i_type];

        if (lambda <= Scalar(0.0))
            {
            // guard against invalid parameters
            continue;
            }

        if (m_poisson_dist_created[i_type])
            {
            // release memory for old parameter
            this->m_exec_conf->msg->notice(6) << "Destroying Poisson distribution for type id " << i_type << std::endl;
            curandDestroyDistribution(h_poisson_dist.data[i_type]);

            if (this->m_exec_conf->isCUDAErrorCheckingEnabled())
                CHECK_CUDA_ERROR();
            }

        // create a Poisson distribution object
        this->m_exec_conf->msg->notice(6) << "Creating Poisson distribution for type id " << i_type << std::endl;
        curandCreatePoissonDistribution(lambda,&h_poisson_dist.data[i_type]);

        // keep track of state
        m_poisson_dist_created[i_type] = true;

        if (this->m_exec_conf->isCUDAErrorCheckingEnabled())
            CHECK_CUDA_ERROR();
        }
    }

template< class Shape >
void IntegratorHPMCMonoImplicitGPU< Shape >::initializeCellSets()
    {
    this->m_exec_conf->msg->notice(4) << "hpmc recomputing active cells" << std::endl;
    // "ghost cells" might contain active particles. So they must be included in the active cell sets
    // we should not run into a multiple issue since the base multiple is 2 and the ghost cells added are 2 in each
    // direction. Check just to be on the safe side

    // compute the number of cells in each set
    // every other cell is active along each direction, excluding ghost cells
    uint3 dim = this->m_cl->getDim();
    const Index3D& cell_indexer = this->m_cl->getCellIndexer();
    unsigned int n_active = dim.x / 2 * dim.y / 2;
    unsigned int n_sets = 4;

    if (this->m_sysdef->getNDimensions() == 3)
        {
        n_active *= dim.z / 2;
        n_sets = 8;
        }

    GPUArray< unsigned int > cell_sets(n_active, n_sets, this->m_exec_conf);
    m_cell_sets.swap(cell_sets);
    m_cell_set_indexer = Index2D(n_active, n_sets);

    // build a list of active cells
    ArrayHandle< unsigned int > h_cell_sets(m_cell_sets, access_location::host, access_mode::overwrite);

    // offsets for x and y based on the set index
    unsigned int ox[] = {0, 1, 0, 1, 0, 1, 0, 1};
    unsigned int oy[] = {0, 0, 1, 1, 0, 0, 1, 1};
    unsigned int oz[] = {0, 0, 0, 0, 1, 1, 1, 1};

    for (unsigned int cur_set = 0; cur_set < n_sets; cur_set++)
        {
        unsigned int active_idx = 0;
        // loop over all cells in the active region, using information from num_ghost cells to avoid adding ghost cells
        // to the active set
        for (int k = oz[cur_set]; k < int(dim.z); k+=2)
            for (int j = oy[cur_set]; j < int(dim.y); j+=2)
                for (int i = ox[cur_set]; i < int(dim.x); i+=2)
                    {
                    h_cell_sets.data[m_cell_set_indexer(active_idx, cur_set)] = cell_indexer(i,j,k);
                    active_idx++;
                    }
        }
    }

template< class Shape >
void IntegratorHPMCMonoImplicitGPU< Shape >::initializeExcellMem()
    {
    this->m_exec_conf->msg->notice(4) << "hpmc resizing expanded cells" << std::endl;

    // get the current cell dimensions
    unsigned int num_cells = this->m_cl->getCellIndexer().getNumElements();
    unsigned int num_adj = this->m_cl->getCellAdjIndexer().getW();
    unsigned int num_max = this->m_cl->getNmax();

    // make the excell dimensions the same, but with room for Nmax*Nadj in each cell
    m_excell_list_indexer = Index2D(num_max * num_adj, num_cells);

    // reallocate memory
    m_excell_idx.resize(m_excell_list_indexer.getNumElements());
    m_excell_size.resize(num_cells);
    }

template< class Shape >
void IntegratorHPMCMonoImplicitGPU< Shape >::updateCellWidth()
    {
    IntegratorHPMCMonoImplicit<Shape>::updateCellWidth();

    this->m_cl->setNominalWidth(this->m_nominal_width);

    // attach the parameters to the kernel stream so that they are visible
    // when other kernels are called
    cudaStreamAttachMemAsync(m_stream, this->m_params.data(), 0, cudaMemAttachSingle);
    CHECK_CUDA_ERROR();
    #if (CUDART_VERSION >= 8000)
    cudaMemAdvise(this->m_params.data(), this->m_params.size()*sizeof(typename Shape::param_type), cudaMemAdviseSetReadMostly, 0);
    CHECK_CUDA_ERROR();
    #endif

    for (unsigned int i = 0; i < this->m_pdata->getNTypes(); ++i)
        {
        // attach nested memory regions
        this->m_params[i].attach_to_stream(m_stream);
        }
    }


//! Export this hpmc integrator to python
/*! \param name Name of the class in the exported python module
    \tparam Shape An instantiation of IntegratorHPMCMono<Shape> will be exported
*/
template < class Shape > void export_IntegratorHPMCMonoImplicitGPU(pybind11::module& m, const std::string& name)
    {
     pybind11::class_<IntegratorHPMCMonoImplicitGPU<Shape>, std::shared_ptr< IntegratorHPMCMonoImplicitGPU<Shape> > >(m, name.c_str(), pybind11::base< IntegratorHPMCMonoImplicit<Shape> >())
              .def(pybind11::init< std::shared_ptr<SystemDefinition>, std::shared_ptr<CellList>, unsigned int >())
        ;
    }

} // end namespace hpmc

#endif // ENABLE_CUDA

#endif // __HPMC_MONO_IMPLICIT_GPU_H__
