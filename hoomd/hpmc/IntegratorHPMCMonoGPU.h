#pragma once

#ifdef ENABLE_CUDA

#include "IntegratorHPMCMono.h"
#include "IntegratorHPMCMonoGPU.cuh"
#include "IntegratorHPMCMonoGPU.cuh"
#include "hoomd/Autotuner.h"
#include "hoomd/GlobalArray.h"
#include "hoomd/GPUVector.h"
#include "hoomd/RandomNumbers.h"
#include "hoomd/RNGIdentifiers.h"

#include "hoomd/GPUPartition.cuh"


#include <cuda_runtime.h>

/*! \file IntegratorHPMCMonoGPU.h
    \brief Defines the template class for HPMC on the GPU
    \note This header cannot be compiled by nvcc
*/

#ifdef NVCC
#error This header cannot be compiled by nvcc
#endif

#include <hoomd/extern/pybind/include/pybind11/pybind11.h>

namespace hpmc
{

//! Template class for HPMC update on the GPU
/*!
    \ingroup hpmc_integrators
*/
template< class Shape >
class IntegratorHPMCMonoGPU : public IntegratorHPMCMono<Shape>
    {
    public:
        //! Construct the integrator
        IntegratorHPMCMonoGPU(std::shared_ptr<SystemDefinition> sysdef,
                              std::shared_ptr<CellList> cl,
                              unsigned int seed);
        //! Destructor
        virtual ~IntegratorHPMCMonoGPU();

        //! Set autotuner parameters
        /*! \param enable Enable/disable autotuning
            \param period period (approximate) in time steps when returning occurs
        */
        virtual void setAutotunerParams(bool enable, unsigned int period)
            {
            unsigned int ndim = this->m_sysdef->getNDimensions();
            if (ndim == 3)
                m_tuner_accept->setPeriod(period*this->m_nselect*8);
            else
                m_tuner_accept->setPeriod(period*this->m_nselect*4);

            m_tuner_accept->setEnabled(enable);

            m_tuner_moves->setPeriod(period*this->m_nselect);
            m_tuner_moves->setEnabled(enable);

            m_tuner_narrow->setPeriod(period*this->m_nselect);
            m_tuner_narrow->setEnabled(enable);

            m_tuner_depletants->setPeriod(period*this->m_nselect);
            m_tuner_depletants->setEnabled(enable);

            m_tuner_depletant_num->setPeriod(period*this->m_nselect);
            m_tuner_depletant_num->setEnabled(enable);

            m_tuner_excell_block_size->setPeriod(period);
            m_tuner_excell_block_size->setEnabled(enable);
            }

        //! Enable deterministic simulations
        virtual void setDeterministic(bool deterministic)
            {
            this->m_exec_conf->msg->notice(2) << "hpmc: Sorting cell list to enable deterministic simulations." << std::endl;
            m_cl->setSortCellList(deterministic);
            }

        //! Method called when numbe of particle types changes
        virtual void slotNumTypesChange();

        //! Take one timestep forward
        virtual void update(unsigned int timestep);

    protected:
        std::shared_ptr<CellList> m_cl;                      //!< Cell list
        GlobalArray<unsigned int> m_cell_sets;               ///!< List of cells active during each subsweep
        Index2D m_cell_set_indexer;                          //!< Indexer into the cell set array
        uint3 m_last_dim;                                    //!< Dimensions of the cell list on the last call to update
        unsigned int m_last_nmax;                            //!< Last cell list NMax value allocated in excell

        GlobalArray<unsigned int> m_excell_idx;              //!< Particle indices in expanded cells
        GlobalArray<unsigned int> m_excell_size;             //!< Number of particles in each expanded cell
        Index2D m_excell_list_indexer;                       //!< Indexer to access elements of the excell_idx list

        std::unique_ptr<Autotuner> m_tuner_moves;            //!< Autotuner for proposing moves
        std::unique_ptr<Autotuner> m_tuner_narrow;           //!< Autotuner for the narrow phase
        std::unique_ptr<Autotuner> m_tuner_accept;    //!< Autotuner for the update step group and block sizes
        std::unique_ptr<Autotuner> m_tuner_excell_block_size;  //!< Autotuner for excell block_size
        std::unique_ptr<Autotuner> m_tuner_depletants;       //!< Autotuner for inserting depletants
        std::unique_ptr<Autotuner> m_tuner_depletant_num;    //!< Autotuner for inserting depletants

        GlobalArray<Scalar4> m_trial_postype;                 //!< New positions (and type) of particles
        GlobalArray<Scalar4> m_trial_orientation;             //!< New orientations
        GlobalArray<unsigned int> m_trial_move_type;          //!< Flags to indicate which type of move
        GlobalArray<unsigned int> m_accept;                   //!< Flags to accept particle moves, per particle
        GlobalArray<unsigned int> m_cell_select;              //!< Selected particle per cell

        GlobalArray<unsigned int> m_nlist;                    //!< List of overlapping particles
        GlobalArray<unsigned int> m_nneigh;                   //!< Number of neighbors
        detail::UpdateOrder m_cell_set_order;                 //!< Cell set update order
        unsigned int m_maxn;                                  //!< Max number of neighbors
        GlobalArray<unsigned int> m_overflow;                 //!< Overflow condition for neighbor list

        GlobalArray<hpmc_counters_t> m_counters;                    //!< Per-device counters
        GlobalArray<hpmc_implicit_counters_t> m_implicit_counters;  //!< Per-device counters for depletants

        //!< Variables for implicit depletants
        GlobalArray<Scalar> m_lambda;                              //!< Poisson means, per type pair

        //! Set up cell sets
        virtual void initializeCellSets();

        //! Set up excell_list
        virtual void initializeExcellMem();

        //! Reallocate nlist as necessary
        bool checkReallocate();

        //! Set the nominal width appropriate for looped moves
        virtual void updateCellWidth();

        //! Update GPU memory hints
        virtual void updateGPUAdvice();
    };

template< class Shape >
IntegratorHPMCMonoGPU< Shape >::IntegratorHPMCMonoGPU(std::shared_ptr<SystemDefinition> sysdef,
                                                                   std::shared_ptr<CellList> cl,
                                                                   unsigned int seed)
    : IntegratorHPMCMono<Shape>(sysdef, seed), m_cl(cl),
      m_cell_set_order(seed+this->m_exec_conf->getRank()),
      m_maxn(0)
    {
    this->m_cl->setRadius(1);
    this->m_cl->setComputeTDB(false);
    this->m_cl->setFlagType();
    this->m_cl->setComputeIdx(true);

    // with multiple GPUs, request a cell list per device
    //m_cl->setPerDevice(this->m_exec_conf->allConcurrentManagedAccess());

    // require that cell lists have an even number of cells along each direction
    this->m_cl->setMultiple(2);

    // set last dim to a bogus value so that it will re-init on the first call
    m_last_dim = make_uint3(0xffffffff, 0xffffffff, 0xffffffff);
    m_last_nmax = 0xffffffff;

    cudaDeviceProp dev_prop = this->m_exec_conf->dev_prop;
    m_tuner_moves.reset(new Autotuner(dev_prop.warpSize, dev_prop.maxThreadsPerBlock, dev_prop.warpSize, 5, 1000000, "hpmc_moves", this->m_exec_conf));
    m_tuner_accept.reset(new Autotuner(dev_prop.warpSize, dev_prop.maxThreadsPerBlock, dev_prop.warpSize, 5, 1000000, "hpmc_accept", this->m_exec_conf));
    m_tuner_excell_block_size.reset(new Autotuner(dev_prop.warpSize, dev_prop.maxThreadsPerBlock, dev_prop.warpSize, 5, 1000000, "hpmc_excell_block_size", this->m_exec_conf));
    m_tuner_depletant_num.reset(new Autotuner(dev_prop.warpSize, dev_prop.maxThreadsPerBlock, dev_prop.warpSize, 5, 1000000, "hpmc_depletant_num", this->m_exec_conf));

    // tuning parameters for narrow phase
    std::vector<unsigned int> valid_params;
    const unsigned int narrow_phase_max_tpp = dev_prop.maxThreadsPerBlock;
    for (unsigned int block_size = dev_prop.warpSize; block_size <= (unsigned int) dev_prop.maxThreadsPerBlock; block_size += dev_prop.warpSize)
        {
        for (unsigned int group_size=1; group_size <= narrow_phase_max_tpp; group_size*=2)
            {
            if ((block_size % group_size) == 0)
                valid_params.push_back(block_size*10000 + group_size);
            }
        }
    m_tuner_narrow.reset(new Autotuner(valid_params, 5, 100000, "hpmc_narrow", this->m_exec_conf));
    m_tuner_depletants.reset(new Autotuner(valid_params, 5, 100000, "hpmc_depletants", this->m_exec_conf));

    // initialize memory
    GlobalArray<Scalar4>(0,this->m_exec_conf).swap(m_trial_postype);
    TAG_ALLOCATION(m_trial_postype);

    GlobalArray<Scalar4>(0, this->m_exec_conf).swap(m_trial_orientation);
    TAG_ALLOCATION(m_trial_orientation);

    GlobalArray<unsigned int>(0,this->m_exec_conf).swap(m_trial_move_type);
    TAG_ALLOCATION(m_trial_move_type);

    GlobalArray<unsigned int>(0, this->m_exec_conf).swap(m_accept);
    TAG_ALLOCATION(m_accept);

    GlobalArray<unsigned int>(0, this->m_exec_conf).swap(m_nlist);
    TAG_ALLOCATION(m_nlist);

    GlobalArray<unsigned int>(0, this->m_exec_conf).swap(m_nneigh);
    TAG_ALLOCATION(m_nneigh);

    GlobalArray<unsigned int>(1, this->m_exec_conf).swap(m_overflow);
    TAG_ALLOCATION(m_overflow);

    GlobalArray<unsigned int> excell_size(0, this->m_exec_conf);
    m_excell_size.swap(excell_size);
    TAG_ALLOCATION(m_excell_size);

    GlobalArray<unsigned int> excell_idx(0, this->m_exec_conf);
    m_excell_idx.swap(excell_idx);
    TAG_ALLOCATION(m_excell_idx);

    //! One counter per GPU, separated by an entire memory page
    unsigned int pitch = (getpagesize() + sizeof(hpmc_counters_t)-1)/sizeof(hpmc_counters_t);
    GlobalArray<hpmc_counters_t>(pitch, this->m_exec_conf->getNumActiveGPUs(), this->m_exec_conf).swap(m_counters);
    TAG_ALLOCATION(m_counters);

    if (this->m_exec_conf->allConcurrentManagedAccess())
        {
        // set memory hints
        auto gpu_map = this->m_exec_conf->getGPUIds();
        for (unsigned int idev = 0; idev < this->m_exec_conf->getNumActiveGPUs(); ++idev)
            {
            cudaMemAdvise(m_counters.get()+idev*m_counters.getPitch(), sizeof(hpmc_counters_t)*m_counters.getPitch(), cudaMemAdviseSetPreferredLocation, gpu_map[idev]);
            cudaMemPrefetchAsync(m_counters.get()+idev*m_counters.getPitch(), sizeof(hpmc_counters_t)*m_counters.getPitch(), gpu_map[idev]);
            }
        }

    // ntypes counters per GPU, separated by at least a memory page
    pitch = (getpagesize() + sizeof(hpmc_implicit_counters_t)-1)/sizeof(hpmc_implicit_counters_t);
    GlobalArray<hpmc_implicit_counters_t>(std::max(pitch, this->m_implicit_count.getNumElements()),
        this->m_exec_conf->getNumActiveGPUs(), this->m_exec_conf).swap(m_implicit_counters);
    TAG_ALLOCATION(m_implicit_counters);

    if (this->m_exec_conf->allConcurrentManagedAccess())
        {
        // set memory hints
        auto gpu_map = this->m_exec_conf->getGPUIds();
        for (unsigned int idev = 0; idev < this->m_exec_conf->getNumActiveGPUs(); ++idev)
            {
            cudaMemAdvise(m_implicit_counters.get()+idev*m_implicit_counters.getPitch(),
                sizeof(hpmc_implicit_counters_t)*m_implicit_counters.getPitch(), cudaMemAdviseSetPreferredLocation, gpu_map[idev]);
            cudaMemPrefetchAsync(m_implicit_counters.get()+idev*m_implicit_counters.getPitch(),
                sizeof(hpmc_implicit_counters_t)*m_implicit_counters.getPitch(), gpu_map[idev]);
            }
        }

        {
        ArrayHandle<unsigned int> h_overflow(m_overflow, access_location::host, access_mode::overwrite);
        *h_overflow.data = 0;
        }

    // Depletants
    unsigned int ntypes = this->m_pdata->getNTypes();
    GlobalArray<Scalar> lambda(ntypes*ntypes, this->m_exec_conf);
    m_lambda.swap(lambda);
    TAG_ALLOCATION(m_lambda);

    // memory hint for overlap matrix
    if (this->m_exec_conf->allConcurrentManagedAccess())
        {
        cudaMemAdvise(this->m_overlaps.get(), sizeof(unsigned int)*this->m_overlaps.getNumElements(), cudaMemAdviseSetReadMostly, 0);
        CHECK_CUDA_ERROR();
        }
    }

template< class Shape >
IntegratorHPMCMonoGPU< Shape >::~IntegratorHPMCMonoGPU()
    {
    }

template< class Shape >
void IntegratorHPMCMonoGPU< Shape >::updateGPUAdvice()
    {
    // update memory hints
    if (this->m_exec_conf->allConcurrentManagedAccess())
        {
        // set memory hints
        auto gpu_map = this->m_exec_conf->getGPUIds();
        for (unsigned int idev = 0; idev < this->m_exec_conf->getNumActiveGPUs(); ++idev)
            {
            auto range = this->m_pdata->getGPUPartition().getRange(idev);

            unsigned int nelem = range.second-range.first;
            if (nelem == 0)
                continue;

            cudaMemAdvise(m_trial_postype.get()+range.first, sizeof(Scalar4)*nelem, cudaMemAdviseSetPreferredLocation, gpu_map[idev]);
            cudaMemPrefetchAsync(m_trial_postype.get()+range.first, sizeof(Scalar4)*nelem, gpu_map[idev]);

            cudaMemAdvise(m_trial_move_type.get()+range.first, sizeof(unsigned int)*nelem, cudaMemAdviseSetPreferredLocation, gpu_map[idev]);
            cudaMemPrefetchAsync(m_trial_move_type.get()+range.first, sizeof(unsigned int)*nelem, gpu_map[idev]);
            cudaMemAdvise(m_trial_move_type.get()+range.first, sizeof(unsigned int)*nelem, cudaMemAdviseSetAccessedBy, cudaCpuDeviceId);

            cudaMemAdvise(m_accept.get()+range.first, sizeof(unsigned int)*nelem, cudaMemAdviseSetPreferredLocation, gpu_map[idev]);
            cudaMemPrefetchAsync(m_accept.get()+range.first, sizeof(unsigned int)*nelem, gpu_map[idev]);

            cudaMemAdvise(m_trial_orientation.get()+range.first, sizeof(Scalar4)*nelem, cudaMemAdviseSetPreferredLocation, gpu_map[idev]);
            cudaMemPrefetchAsync(m_trial_orientation.get()+range.first, sizeof(Scalar4)*nelem, gpu_map[idev]);

            cudaMemAdvise(m_nneigh.get()+range.first, sizeof(unsigned int)*nelem, cudaMemAdviseSetPreferredLocation, gpu_map[idev]);
            cudaMemPrefetchAsync(m_nneigh.get()+range.first, sizeof(unsigned int)*nelem, gpu_map[idev]);
            cudaMemAdvise(m_nneigh.get(), sizeof(unsigned int)*m_nneigh.getNumElements(), cudaMemAdviseSetAccessedBy, cudaCpuDeviceId);
            CHECK_CUDA_ERROR();
            }
        }
    }

template< class Shape >
void IntegratorHPMCMonoGPU< Shape >::update(unsigned int timestep)
    {
    if (this->m_patch && !this->m_patch_log)
        {
        this->m_exec_conf->msg->error() << "GPU simulations with patches are unsupported." << std::endl;
        throw std::runtime_error("Error during HPMC integration\n");
        }

    IntegratorHPMC::update(timestep);

    // rng for shuffle and grid shift
    hoomd::RandomGenerator rng(hoomd::RNGIdentifier::HPMCMonoShift, this->m_seed, timestep);

    if (this->m_pdata->getN() > 0)
        {
        // compute the width of the active region
        Scalar3 npd = this->m_pdata->getBox().getNearestPlaneDistance();
        Scalar3 ghost_fraction = this->m_nominal_width / npd;

        // check if we are below a minimum image convention box size
        // the minimum image convention comes from the global box, not the local one
        BoxDim global_box = this->m_pdata->getGlobalBox();
        Scalar3 nearest_plane_distance = global_box.getNearestPlaneDistance();

        if ((global_box.getPeriodic().x && nearest_plane_distance.x <= this->m_nominal_width*2) ||
            (global_box.getPeriodic().y && nearest_plane_distance.y <= this->m_nominal_width*2) ||
            (this->m_sysdef->getNDimensions() == 3 && global_box.getPeriodic().z && nearest_plane_distance.z <= this->m_nominal_width*2))
            {
            this->m_exec_conf->msg->error() << "Simulation box too small for GPU accelerated HPMC execution - increase it so the minimum image convention works" << std::endl;
            throw std::runtime_error("Error performing HPMC update");
            }

        // update the cell list
        this->m_cl->compute(timestep);

        // start the profile
        if (this->m_prof) this->m_prof->push(this->m_exec_conf, "HPMC");

        // if the cell list is a different size than last time, reinitialize the cell sets list
        uint3 cur_dim = this->m_cl->getDim();
        if (m_last_dim.x != cur_dim.x || m_last_dim.y != cur_dim.y || m_last_dim.z != cur_dim.z)
            {
            initializeCellSets();
            initializeExcellMem();

            m_last_dim = cur_dim;
            m_last_nmax = this->m_cl->getNmax();

            // initialize the cell set update order
            this->m_cell_set_order.resize(m_cell_set_indexer.getH());

            GlobalArray<unsigned int> cell_select(this->m_cl->getCellIndexer().getNumElements(), this->m_exec_conf);
            m_cell_select.swap(cell_select);
            TAG_ALLOCATION(m_cell_select);
            }

        // if only NMax changed, only need to reallocate excell memory
        if (m_last_nmax != this->m_cl->getNmax())
            {
            initializeExcellMem();
            m_last_nmax = this->m_cl->getNmax();
            }

        // test if we are in domain decomposition mode
        bool domain_decomposition = false;
#ifdef ENABLE_MPI
        if (this->m_comm)
            domain_decomposition = true;
#endif

        // resize some arrays
        bool resized = m_accept.getNumElements() < this->m_pdata->getMaxN();

        if (resized)
            {
            m_accept.resize(this->m_pdata->getMaxN());
            m_nneigh.resize(this->m_pdata->getMaxN());
            m_trial_postype.resize(this->m_pdata->getMaxN());
            m_trial_orientation.resize(this->m_pdata->getMaxN());
            m_trial_move_type.resize(this->m_pdata->getMaxN());

            updateGPUAdvice();
            }

        // access the cell list data
        ArrayHandle<unsigned int> d_cell_size(this->m_cl->getCellSizeArray(), access_location::device, access_mode::read);
        ArrayHandle<unsigned int> d_cell_idx(this->m_cl->getIndexArray(), access_location::device, access_mode::read);
        ArrayHandle<unsigned int> d_cell_adj(this->m_cl->getCellAdjArray(), access_location::device, access_mode::read);

        ArrayHandle< unsigned int > d_cell_sets(m_cell_sets, access_location::device, access_mode::read);

        ArrayHandle<unsigned int> d_cell_select(m_cell_select, access_location::device, access_mode::overwrite);

        // per-device cell list data
        const ArrayHandle<unsigned int>& d_cell_size_per_device = m_cl->getPerDevice() ?
            ArrayHandle<unsigned int>(m_cl->getCellSizeArrayPerDevice(),access_location::device, access_mode::read) :
            ArrayHandle<unsigned int>(GlobalArray<unsigned int>(), access_location::device, access_mode::read);
        const ArrayHandle<unsigned int>& d_cell_idx_per_device = m_cl->getPerDevice() ?
            ArrayHandle<unsigned int>(m_cl->getIndexArrayPerDevice(), access_location::device, access_mode::read) :
            ArrayHandle<unsigned int>(GlobalArray<unsigned int>(), access_location::device, access_mode::read);

        unsigned int ngpu = this->m_exec_conf->getNumActiveGPUs();
        if (ngpu > 1)
            {
            // reset per-device counters
            ArrayHandle<hpmc_counters_t> d_counters_per_device(this->m_counters, access_location::device, access_mode::overwrite);
            cudaMemset(d_counters_per_device.data, 0, sizeof(hpmc_counters_t)*this->m_counters.getNumElements());
            if (this->m_exec_conf->isCUDAErrorCheckingEnabled()) CHECK_CUDA_ERROR();

            ArrayHandle<hpmc_implicit_counters_t> d_implicit_counters_per_device(this->m_implicit_counters, access_location::device, access_mode::overwrite);
            cudaMemset(d_implicit_counters_per_device.data, 0, sizeof(hpmc_implicit_counters_t)*this->m_implicit_counters.getNumElements());
            if (this->m_exec_conf->isCUDAErrorCheckingEnabled()) CHECK_CUDA_ERROR();
            }

        // access the parameters and interaction matrix
        auto & params = this->getParams();

        ArrayHandle<unsigned int> d_overlaps(this->m_overlaps, access_location::device, access_mode::read);

        // access the move sizes by type
        ArrayHandle<Scalar> d_d(this->m_d, access_location::device, access_mode::read);
        ArrayHandle<Scalar> d_a(this->m_a, access_location::device, access_mode::read);

        BoxDim box = this->m_pdata->getBox();

        Scalar3 ghost_width = this->m_cl->getGhostWidth();

        // randomize particle update order
        this->m_cell_set_order.shuffle(timestep);

        // expanded cells & neighbor list
        ArrayHandle< unsigned int > d_excell_idx(m_excell_idx, access_location::device, access_mode::overwrite);
        ArrayHandle< unsigned int > d_excell_size(m_excell_size, access_location::device, access_mode::overwrite);

        // compute the ceiling of the average number of particles in each cell, accounting for ghost particles
        int ncells = cur_dim.x * cur_dim.y * cur_dim.z;
        int particles_per_cell = int(ceil(double(this->m_pdata->getN() + this->m_pdata->getNGhosts()) / double(ncells)));

        // update the expanded cells
        this->m_tuner_excell_block_size->begin();
        gpu::hpmc_excell(d_excell_idx.data,
                            d_excell_size.data,
                            m_excell_list_indexer,
                            m_cl->getPerDevice() ? d_cell_idx_per_device.data : d_cell_idx.data,
                            m_cl->getPerDevice() ? d_cell_size_per_device.data : d_cell_size.data,
                            d_cell_adj.data,
                            this->m_cl->getCellIndexer(),
                            this->m_cl->getCellListIndexer(),
                            this->m_cl->getCellAdjIndexer(),
                            m_cl->getPerDevice() ? this->m_exec_conf->getNumActiveGPUs() : 1,
                            this->m_tuner_excell_block_size->getParam());
        if (this->m_exec_conf->isCUDAErrorCheckingEnabled()) CHECK_CUDA_ERROR();
        this->m_tuner_excell_block_size->end();

        bool reallocate = false;
        // depletants
        ArrayHandle<Scalar> d_lambda(m_lambda, access_location::device, access_mode::read);

        for (unsigned int i = 0; i < this->m_nselect * particles_per_cell; i++)
            {
                { // ArrayHandle scope
                ArrayHandle<unsigned int> d_nlist(m_nlist, access_location::device, access_mode::read);
                ArrayHandle<unsigned int> d_nneigh(m_nneigh, access_location::device, access_mode::read);
                ArrayHandle<unsigned int> d_overflow(m_overflow, access_location::device, access_mode::read);
                ArrayHandle<unsigned int> d_accept(m_accept, access_location::device, access_mode::overwrite);

                // access data for proposed moves
                ArrayHandle<Scalar4> d_trial_postype(m_trial_postype, access_location::device, access_mode::overwrite);
                ArrayHandle<Scalar4> d_trial_orientation(m_trial_orientation, access_location::device, access_mode::overwrite);
                ArrayHandle<unsigned int> d_trial_move_type(m_trial_move_type, access_location::device, access_mode::overwrite);

                // access the particle data
                ArrayHandle<Scalar4> d_postype(this->m_pdata->getPositions(), access_location::device, access_mode::read);
                ArrayHandle<Scalar4> d_orientation(this->m_pdata->getOrientationArray(), access_location::device, access_mode::read);

                // MC counters
                ArrayHandle<hpmc_counters_t> d_counters(this->m_count_total, access_location::device, access_mode::read);
                ArrayHandle<hpmc_counters_t> d_counters_per_device(this->m_counters, access_location::device, access_mode::read);

                // fill the parameter structure for the GPU kernels
                hpmc::gpu::hpmc_args_t args(
                    d_postype.data,
                    d_orientation.data,
                    ngpu > 1 ? d_counters_per_device.data : d_counters.data,
                    this->m_counters.getPitch(),
                    this->m_cl->getCellIndexer(),
                    this->m_cl->getDim(),
                    ghost_width,
                    this->m_pdata->getN(),
                    this->m_pdata->getNGhosts(),
                    this->m_pdata->getNTypes(),
                    this->m_seed,
                    d_d.data,
                    d_a.data,
                    d_overlaps.data,
                    this->m_overlap_idx,
                    this->m_move_ratio,
                    timestep,
                    this->m_sysdef->getNDimensions(),
                    box,
                    this->m_exec_conf->getRank()*this->m_nselect*particles_per_cell + i,
                    ghost_fraction,
                    domain_decomposition,
                    0, // block size
                    0, // tpp
                    d_accept.data,
                    d_trial_postype.data,
                    d_trial_orientation.data,
                    d_trial_move_type.data,
                    d_excell_idx.data,
                    d_excell_size.data,
                    m_excell_list_indexer,
                    d_nlist.data,
                    d_nneigh.data,
                    m_maxn,
                    d_overflow.data,
                    i == 0,
                    this->m_exec_conf->dev_prop,
                    m_cell_set_indexer,
                    d_cell_sets.data,
                    m_cl->getPerDevice() ? d_cell_size_per_device.data : d_cell_size.data,
                    m_cl->getPerDevice() ? d_cell_idx_per_device.data : d_cell_idx.data,
                    m_cl->getPerDevice() ? this->m_exec_conf->getNumActiveGPUs() : 1,
                    d_cell_select.data,
                    this->m_cl->getCellListIndexer());

                // propose trial moves, \sa hpmc::gpu::kernel::hpmc_moves

                // reset acceptance results and move types
                m_tuner_moves->begin();
                args.block_size = m_tuner_moves->getParam();
                gpu::hpmc_gen_moves<Shape>(args, params.data());
                if (this->m_exec_conf->isCUDAErrorCheckingEnabled())
                    CHECK_CUDA_ERROR();
                m_tuner_moves->end();
                }

            do
                {
                    { // ArrayHandle scope
                    ArrayHandle<unsigned int> d_nlist(m_nlist, access_location::device, access_mode::overwrite);
                    ArrayHandle<unsigned int> d_nneigh(m_nneigh, access_location::device, access_mode::overwrite);
                    ArrayHandle<unsigned int> d_overflow(m_overflow, access_location::device, access_mode::readwrite);
                    ArrayHandle<unsigned int> d_accept(m_accept, access_location::device, access_mode::read);

                    // access data for proposed moves
                    ArrayHandle<Scalar4> d_trial_postype(m_trial_postype, access_location::device, access_mode::read);
                    ArrayHandle<Scalar4> d_trial_orientation(m_trial_orientation, access_location::device, access_mode::read);
                    ArrayHandle<unsigned int> d_trial_move_type(m_trial_move_type, access_location::device, access_mode::read);

                    // access the particle data
                    ArrayHandle<Scalar4> d_postype(this->m_pdata->getPositions(), access_location::device, access_mode::readwrite);
                    ArrayHandle<Scalar4> d_orientation(this->m_pdata->getOrientationArray(), access_location::device, access_mode::readwrite);

                    // MC counters
                    ArrayHandle<hpmc_counters_t> d_counters(this->m_count_total, access_location::device, access_mode::readwrite);
                    ArrayHandle<hpmc_counters_t> d_counters_per_device(this->m_counters, access_location::device, access_mode::readwrite);

                    // depletant counters
                    ArrayHandle<hpmc_implicit_counters_t> d_implicit_count(this->m_implicit_count, access_location::device, access_mode::readwrite);
                    ArrayHandle<hpmc_implicit_counters_t> d_implicit_counters_per_device(this->m_implicit_counters, access_location::device, access_mode::readwrite);

                    // fill the parameter structure for the GPU kernels
                    hpmc::gpu::hpmc_args_t args(
                        d_postype.data,
                        d_orientation.data,
                        ngpu > 1 ? d_counters_per_device.data : d_counters.data,
                        this->m_counters.getPitch(),
                        this->m_cl->getCellIndexer(),
                        this->m_cl->getDim(),
                        ghost_width,
                        this->m_pdata->getN(),
                        this->m_pdata->getNGhosts(),
                        this->m_pdata->getNTypes(),
                        this->m_seed,
                        d_d.data,
                        d_a.data,
                        d_overlaps.data,
                        this->m_overlap_idx,
                        this->m_move_ratio,
                        timestep,
                        this->m_sysdef->getNDimensions(),
                        box,
                        this->m_exec_conf->getRank()*this->m_nselect*particles_per_cell + i,
                        ghost_fraction,
                        domain_decomposition,
                        0, // block size
                        0, // tpp
                        d_accept.data,
                        d_trial_postype.data,
                        d_trial_orientation.data,
                        d_trial_move_type.data,
                        d_excell_idx.data,
                        d_excell_size.data,
                        m_excell_list_indexer,
                        d_nlist.data,
                        d_nneigh.data,
                        m_maxn,
                        d_overflow.data,
                        i == 0,
                        this->m_exec_conf->dev_prop,
                        m_cell_set_indexer,
                        d_cell_sets.data,
                        m_cl->getPerDevice() ? d_cell_size_per_device.data : d_cell_size.data,
                        m_cl->getPerDevice() ? d_cell_idx_per_device.data : d_cell_idx.data,
                        m_cl->getPerDevice() ? this->m_exec_conf->getNumActiveGPUs() : 1,
                        d_cell_select.data,
                        this->m_cl->getCellListIndexer());

                    /*
                     *  check overlaps, new configuration simultaneously against the old and the new configuration
                     */

                    this->m_exec_conf->beginMultiGPU();
                    m_tuner_narrow->begin();
                    unsigned int param = m_tuner_narrow->getParam();
                    args.block_size = param/10000;
                    args.tpp = param%10000;
                    gpu::hpmc_narrow_phase<Shape>(args, params.data());
                    if (this->m_exec_conf->isCUDAErrorCheckingEnabled())
                        CHECK_CUDA_ERROR();
                    m_tuner_narrow->end();
                    this->m_exec_conf->endMultiGPU();
                    } // end ArrayHandle scope

                reallocate = checkReallocate();

                if (reallocate)
                    {
                    continue;
                    }

                    { // ArrayHandle scope
                    ArrayHandle<unsigned int> d_nlist(m_nlist, access_location::device, access_mode::readwrite);
                    ArrayHandle<unsigned int> d_nneigh(m_nneigh, access_location::device, access_mode::readwrite);
                    ArrayHandle<unsigned int> d_overflow(m_overflow, access_location::device, access_mode::readwrite);
                    ArrayHandle<unsigned int> d_accept(m_accept, access_location::device, access_mode::read);

                    // access data for proposed moves
                    ArrayHandle<Scalar4> d_trial_postype(m_trial_postype, access_location::device, access_mode::read);
                    ArrayHandle<Scalar4> d_trial_orientation(m_trial_orientation, access_location::device, access_mode::read);
                    ArrayHandle<unsigned int> d_trial_move_type(m_trial_move_type, access_location::device, access_mode::read);

                    // access the particle data
                    ArrayHandle<Scalar4> d_postype(this->m_pdata->getPositions(), access_location::device, access_mode::readwrite);
                    ArrayHandle<Scalar4> d_orientation(this->m_pdata->getOrientationArray(), access_location::device, access_mode::readwrite);

                    // MC counters
                    ArrayHandle<hpmc_counters_t> d_counters(this->m_count_total, access_location::device, access_mode::readwrite);
                    ArrayHandle<hpmc_counters_t> d_counters_per_device(this->m_counters, access_location::device, access_mode::readwrite);

                    // depletant counters
                    ArrayHandle<hpmc_implicit_counters_t> d_implicit_count(this->m_implicit_count, access_location::device, access_mode::readwrite);
                    ArrayHandle<hpmc_implicit_counters_t> d_implicit_counters_per_device(this->m_implicit_counters, access_location::device, access_mode::readwrite);

                    // fill the parameter structure for the GPU kernels
                    hpmc::gpu::hpmc_args_t args(
                        d_postype.data,
                        d_orientation.data,
                        ngpu > 1 ? d_counters_per_device.data : d_counters.data,
                        this->m_counters.getPitch(),
                        this->m_cl->getCellIndexer(),
                        this->m_cl->getDim(),
                        ghost_width,
                        this->m_pdata->getN(),
                        this->m_pdata->getNGhosts(),
                        this->m_pdata->getNTypes(),
                        this->m_seed,
                        d_d.data,
                        d_a.data,
                        d_overlaps.data,
                        this->m_overlap_idx,
                        this->m_move_ratio,
                        timestep,
                        this->m_sysdef->getNDimensions(),
                        box,
                        this->m_exec_conf->getRank()*this->m_nselect*particles_per_cell + i,
                        ghost_fraction,
                        domain_decomposition,
                        0, // block size
                        0, // tpp
                        d_accept.data,
                        d_trial_postype.data,
                        d_trial_orientation.data,
                        d_trial_move_type.data,
                        d_excell_idx.data,
                        d_excell_size.data,
                        m_excell_list_indexer,
                        d_nlist.data,
                        d_nneigh.data,
                        m_maxn,
                        d_overflow.data,
                        i == 0,
                        this->m_exec_conf->dev_prop,
                        m_cell_set_indexer,
                        d_cell_sets.data,
                        m_cl->getPerDevice() ? d_cell_size_per_device.data : d_cell_size.data,
                        m_cl->getPerDevice() ? d_cell_idx_per_device.data : d_cell_idx.data,
                        m_cl->getPerDevice() ? this->m_exec_conf->getNumActiveGPUs() : 1,
                        d_cell_select.data,
                        this->m_cl->getCellListIndexer());

                    /*
                     * Insert depletants
                     */

                    for (unsigned int itype = 0; itype < this->m_pdata->getNTypes(); ++itype)
                        {
                        if (this->m_fugacity[itype] == 0)
                            continue;

                        // insert depletants on-the-fly
                        this->m_exec_conf->beginMultiGPU();
                        m_tuner_depletants->begin();
                        unsigned int param = m_tuner_depletants->getParam();
                        args.block_size = param/1000;
                        args.tpp = param%1000;

                        gpu::hpmc_implicit_args_t implicit_args(itype,
                            ngpu > 1 ? d_implicit_counters_per_device.data : d_implicit_count.data,
                            m_implicit_counters.getPitch(),
                            d_lambda.data,
                            this->m_fugacity[itype] < 0,
                            this->m_quermass,
                            this->m_sweep_radius
                            );
                        gpu::hpmc_insert_depletants<Shape>(args, implicit_args, params.data());
                        if (this->m_exec_conf->isCUDAErrorCheckingEnabled())
                            CHECK_CUDA_ERROR();
                        m_tuner_depletants->end();
                        this->m_exec_conf->endMultiGPU();
                        }
                    } // end ArrayHandle scope

                reallocate = checkReallocate();
                } while (reallocate);

            for (unsigned int j = 0; j < m_cell_set_indexer.getH(); j++)
                {
                unsigned cur_set = this->m_cell_set_order[j];

                // access data for proposed moves
                ArrayHandle<Scalar4> d_trial_postype(m_trial_postype, access_location::device, access_mode::read);
                ArrayHandle<Scalar4> d_trial_orientation(m_trial_orientation, access_location::device, access_mode::read);
                ArrayHandle<unsigned int> d_trial_move_type(m_trial_move_type, access_location::device, access_mode::read);

                // access the particle data
                ArrayHandle<Scalar4> d_postype(this->m_pdata->getPositions(), access_location::device, access_mode::readwrite);
                ArrayHandle<Scalar4> d_orientation(this->m_pdata->getOrientationArray(), access_location::device, access_mode::readwrite);

                // MC counters
                ArrayHandle<hpmc_counters_t> d_counters(this->m_count_total, access_location::device, access_mode::readwrite);
                ArrayHandle<hpmc_counters_t> d_counters_per_device(this->m_counters, access_location::device, access_mode::readwrite);

                // flags
                ArrayHandle<unsigned int> d_accept(m_accept, access_location::device, access_mode::readwrite);

                // neighbor list
                ArrayHandle<unsigned int> d_nneigh(m_nneigh, access_location::device, access_mode::read);
                ArrayHandle<unsigned int> d_nlist(m_nlist, access_location::device, access_mode::read);

                // Update the particle data and statistics
                m_tuner_accept->begin();
                hpmc::gpu::hpmc_update_args_t args(
                    d_postype.data,
                    d_orientation.data,
                    this->m_seed,
                    this->m_exec_conf->getRank()*this->m_nselect*particles_per_cell + i,
                    timestep,
                    m_cell_set_indexer.getW(),
                    &d_cell_sets.data[m_cell_set_indexer(0,cur_set)],
                    m_cl->getPerDevice() ? d_cell_size_per_device.data : d_cell_size.data,
                    m_cl->getPerDevice() ? d_cell_idx_per_device.data : d_cell_idx.data,
                    m_cl->getPerDevice() ? this->m_exec_conf->getNumActiveGPUs() : 1,
                    this->m_cl->getCellListIndexer(),
                    this->m_cl->getCellIndexer(),
                    ngpu > 1 ? d_counters_per_device.data : d_counters.data,
                    this->m_pdata->getN(),
                    this->m_pdata->getBox(),
                    ghost_width,
                    this->m_cl->getDim(),
                    d_trial_postype.data,
                    d_trial_orientation.data,
                    d_trial_move_type.data,
                    d_accept.data,
                    d_nneigh.data,
                    d_nlist.data,
                    m_maxn,
                    this->m_pdata->getN() + this->m_pdata->getNGhosts(),
                    d_cell_select.data,
                    m_tuner_accept->getParam());
                gpu::hpmc_accept<Shape>(args, params.data());
                if (this->m_exec_conf->isCUDAErrorCheckingEnabled())
                    CHECK_CUDA_ERROR();
                m_tuner_accept->end();
                } // end loop over cell sets
            } // end loop over nselect

        if (ngpu > 1)
            {
            // reduce per-device counters
            ArrayHandle<hpmc_counters_t> h_count_total(this->m_count_total, access_location::host, access_mode::readwrite);
            ArrayHandle<hpmc_counters_t> h_counters_per_device(m_counters, access_location::host, access_mode::read);

            for (unsigned int idev = 0; idev < ngpu; ++idev)
                {
                *h_count_total.data = *h_count_total.data + h_counters_per_device.data[idev*m_counters.getPitch()];
                }

            ArrayHandle<hpmc_implicit_counters_t> h_implicit_count_total(this->m_implicit_count, access_location::host, access_mode::readwrite);
            ArrayHandle<hpmc_implicit_counters_t> h_implicit_counters_per_device(m_implicit_counters, access_location::host, access_mode::read);

            for (unsigned int idev = 0; idev < ngpu; ++idev)
                {
                for (unsigned int itype = 0; itype < this->m_pdata->getNTypes(); ++itype)
                    h_implicit_count_total.data[itype] = h_implicit_count_total.data[itype] + h_implicit_counters_per_device.data[itype+idev*m_implicit_counters.getPitch()];
                }
            }
        }

    // shift particles
    Scalar3 shift = make_scalar3(0,0,0);
    hoomd::UniformDistribution<Scalar> uniform(-this->m_nominal_width/Scalar(2.0),this->m_nominal_width/Scalar(2.0));
    shift.x = uniform(rng);
    shift.y = uniform(rng);
    if (this->m_sysdef->getNDimensions() == 3)
        {
        shift.z = uniform(rng);
        }

    if (this->m_pdata->getN() > 0)
        {
        BoxDim box = this->m_pdata->getBox();

        // access the particle data
        ArrayHandle<Scalar4> d_postype(this->m_pdata->getPositions(), access_location::device, access_mode::readwrite);
        ArrayHandle<Scalar4> d_orientation(this->m_pdata->getOrientationArray(), access_location::device, access_mode::readwrite);
        ArrayHandle<int3> d_image(this->m_pdata->getImages(), access_location::device, access_mode::readwrite);

        gpu::hpmc_shift(d_postype.data,
                        d_image.data,
                        this->m_pdata->getN(),
                        box,
                        shift,
                        128);
        }
    if (this->m_exec_conf->isCUDAErrorCheckingEnabled())
        CHECK_CUDA_ERROR();

    // update the particle data origin
    this->m_pdata->translateOrigin(shift);

    if (this->m_prof) this->m_prof->pop(this->m_exec_conf);

    this->communicate(true);

    // all particle have been moved, the aabb tree is now invalid
    this->m_aabb_tree_invalid = true;
    }

template< class Shape >
bool IntegratorHPMCMonoGPU< Shape >::checkReallocate()
    {
    // read back overflow condition and resize as necessary
    ArrayHandle<unsigned int> h_overflow(m_overflow, access_location::host, access_mode::read);
    unsigned int req_maxn = *h_overflow.data;
    if (req_maxn > m_maxn)
        m_maxn = req_maxn;
    unsigned int req_size_nlist = m_maxn*this->m_pdata->getN();

    // resize
    bool reallocate = req_size_nlist > m_nlist.getNumElements();
    if (reallocate)
        {
        this->m_exec_conf->msg->notice(9) << "hpmc resizing neighbor list " << m_nlist.getNumElements() << " -> " << req_size_nlist << std::endl;

        GlobalArray<unsigned int> nlist(req_size_nlist, this->m_exec_conf);
        m_nlist.swap(nlist);
        TAG_ALLOCATION(m_nlist);

        // update memory hints
        if (this->m_exec_conf->allConcurrentManagedAccess())
            {
            // set memory hints
            auto gpu_map = this->m_exec_conf->getGPUIds();
            for (unsigned int idev = 0; idev < this->m_exec_conf->getNumActiveGPUs(); ++idev)
                {
                auto range = this->m_pdata->getGPUPartition().getRange(idev);

                unsigned int nelem = range.second-range.first;
                if (nelem == 0)
                    continue;

                cudaMemAdvise(m_nlist.get()+range.first*m_maxn, sizeof(unsigned int)*nelem*m_maxn, cudaMemAdviseSetPreferredLocation, gpu_map[idev]);
                cudaMemPrefetchAsync(m_nlist.get()+range.first*m_maxn, sizeof(unsigned int)*nelem*m_maxn, gpu_map[idev]);
                cudaMemAdvise(m_nlist.get(), sizeof(unsigned int)*m_nlist.getNumElements(), cudaMemAdviseSetAccessedBy, cudaCpuDeviceId);
                CHECK_CUDA_ERROR();
                }
            }
        }
    return reallocate;
    }

template< class Shape >
void IntegratorHPMCMonoGPU< Shape >::initializeCellSets()
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

    GlobalArray< unsigned int > cell_sets(n_active, n_sets, this->m_exec_conf);
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
void IntegratorHPMCMonoGPU< Shape >::initializeExcellMem()
    {
    this->m_exec_conf->msg->notice(4) << "hpmc resizing expanded cells" << std::endl;

    // get the current cell dimensions
    unsigned int num_cells = this->m_cl->getCellIndexer().getNumElements();
    unsigned int num_adj = this->m_cl->getCellAdjIndexer().getW();
    unsigned int n_cell_list = this->m_cl->getPerDevice() ? this->m_exec_conf->getNumActiveGPUs() : 1;
    unsigned int num_max = this->m_cl->getNmax()*n_cell_list;

    // make the excell dimensions the same, but with room for Nmax*Nadj in each cell
    m_excell_list_indexer = Index2D(num_max * num_adj, num_cells);

    // reallocate memory
    m_excell_idx.resize(m_excell_list_indexer.getNumElements());
    m_excell_size.resize(num_cells);

    if (this->m_exec_conf->allConcurrentManagedAccess())
        {
        // set memory hints
        auto gpu_map = this->m_exec_conf->getGPUIds();
        for (unsigned int idev = 0; idev < this->m_exec_conf->getNumActiveGPUs(); ++idev)
            {
            cudaMemAdvise(m_excell_idx.get(), sizeof(unsigned int)*m_excell_idx.getNumElements(), cudaMemAdviseSetAccessedBy, gpu_map[idev]);
            cudaMemAdvise(m_excell_size.get(), sizeof(unsigned int)*m_excell_size.getNumElements(), cudaMemAdviseSetAccessedBy, gpu_map[idev]);
            CHECK_CUDA_ERROR();
            }
        }
    }

template< class Shape >
void IntegratorHPMCMonoGPU< Shape >::slotNumTypesChange()
    {
    unsigned int old_ntypes = this->m_params.size();

    // skip the reallocation if the number of types does not change
    // this keeps shape parameters when restoring a snapshot
    // it will result in invalid coefficients if the snapshot has a different type id -> name mapping
    if (this->m_pdata->getNTypes() != old_ntypes)
        {
        unsigned int ntypes = this->m_pdata->getNTypes();

        // resize array
        GlobalArray<Scalar> lambda(ntypes*ntypes, this->m_exec_conf);
        m_lambda.swap(lambda);
        TAG_ALLOCATION(m_lambda);

        // ntypes counters per GPU, separated by at least a memory page
        unsigned int pitch = (getpagesize() + sizeof(hpmc_implicit_counters_t)-1)/sizeof(hpmc_implicit_counters_t);
        GlobalArray<hpmc_implicit_counters_t>(std::max(pitch, this->m_implicit_count.getNumElements()),
            this->m_exec_conf->getNumActiveGPUs(), this->m_exec_conf).swap(m_implicit_counters);
        TAG_ALLOCATION(m_implicit_counters);

        if (this->m_exec_conf->allConcurrentManagedAccess())
            {
            // memory hint for overlap matrix
            cudaMemAdvise(this->m_overlaps.get(), sizeof(unsigned int)*this->m_overlaps.getNumElements(), cudaMemAdviseSetReadMostly, 0);
            CHECK_CUDA_ERROR();
            }
        }

    // call base class method
    IntegratorHPMCMono<Shape>::slotNumTypesChange();
    }

template< class Shape >
void IntegratorHPMCMonoGPU< Shape >::updateCellWidth()
    {
    // call base class method
    IntegratorHPMCMono<Shape>::updateCellWidth();

    // update the cell list
    this->m_cl->setNominalWidth(this->m_nominal_width);

    // set memory hints
    cudaMemAdvise(this->m_params.data(), this->m_params.size()*sizeof(typename Shape::param_type), cudaMemAdviseSetReadMostly, 0);
    CHECK_CUDA_ERROR();

    #if 0
    for (int idev = this->m_exec_conf->getNumActiveGPUs()-1; idev >= 0; idev--)
        {
        // switch to that device
        cudaSetDevice(this->m_exec_conf->getGPUIds()[idev]);

        // attach the parameters to the kernel stream so that they are visible
        // when other kernels are called
        cudaStreamAttachMemAsync(m_streams[idev], this->m_params.data(), 0, cudaMemAttachSingle);
        CHECK_CUDA_ERROR();

        // sync up so we can access the parameters
        cudaDeviceSynchronize();

        for (unsigned int i = 0; i < this->m_pdata->getNTypes(); ++i)
            {
            // attach nested memory regions
            this->m_params[i].attach_to_stream(m_streams[idev]);
            CHECK_CUDA_ERROR();
            }
        }
    #endif

    // reinitialize poisson means array
    ArrayHandle<Scalar> h_lambda(m_lambda, access_location::host, access_mode::overwrite);
    Index2D typpair_idx(this->m_pdata->getNTypes());

    // reset to zero
    std::fill(h_lambda.data, h_lambda.data + typpair_idx.getNumElements(), 0.0);

    for (unsigned int i_type = 0; i_type < this->m_pdata->getNTypes(); ++i_type)
        {
        Shape shape_i(quat<Scalar>(), this->m_params[i_type]);
        Scalar d_i(shape_i.getCircumsphereDiameter());
        Scalar range = 2.0*this->m_sweep_radius + d_i;

        if (this->m_fugacity[i_type] == 0.0)
            continue;

        for (unsigned int j_type = 0; j_type < this->m_pdata->getNTypes(); ++j_type)
            {
            // parameter for Poisson distribution
            Shape shape_j(quat<Scalar>(), this->m_params[j_type]);

            // get OBB and extend by depletant radius
            detail::OBB obb = shape_j.getOBB(vec3<Scalar>(0,0,0));
            obb.lengths.x += 0.5*range;
            obb.lengths.y += 0.5*range;
            if (this->m_sysdef->getNDimensions() == 3)
                obb.lengths.z += 0.5*range;
            else
                obb.lengths.z = 0.5; // unit length

            Scalar lambda = std::abs(this->m_fugacity[i_type]*obb.getVolume());
            h_lambda.data[typpair_idx(i_type,j_type)] = lambda;
            }
        }
    }


//! Export this hpmc integrator to python
/*! \param name Name of the class in the exported python module
    \tparam Shape An instantiation of IntegratorHPMCMono<Shape> will be exported
*/
template < class Shape > void export_IntegratorHPMCMonoGPU(pybind11::module& m, const std::string& name)
    {
     pybind11::class_<IntegratorHPMCMonoGPU<Shape>, std::shared_ptr< IntegratorHPMCMonoGPU<Shape> > >(m, name.c_str(), pybind11::base< IntegratorHPMCMono<Shape> >())
              .def(pybind11::init< std::shared_ptr<SystemDefinition>, std::shared_ptr<CellList>, unsigned int >())
              ;
    }

} // end namespace hpmc

#endif // ENABLE_CUDA
