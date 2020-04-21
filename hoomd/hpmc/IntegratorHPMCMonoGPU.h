// Copyright (c) 2009-2019 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.

#pragma once

#ifdef ENABLE_HIP

#include "hoomd/hpmc/IntegratorHPMCMono.h"
#include "hoomd/hpmc/IntegratorHPMCMonoGPU.cuh"
#include "IntegratorHPMCMonoGPU.cuh"
#include "hoomd/Autotuner.h"
#include "hoomd/GlobalArray.h"
#include "hoomd/GPUVector.h"
#include "hoomd/RandomNumbers.h"
#include "hoomd/RNGIdentifiers.h"

#include "hoomd/GPUPartition.cuh"

#include <hip/hip_runtime.h>

/*! \file IntegratorHPMCMonoGPU.h
    \brief Defines the template class for HPMC on the GPU
    \note This header cannot be compiled by nvcc
*/

#ifdef __HIPCC__
#error This header cannot be compiled by nvcc
#endif

#include <pybind11/pybind11.h>

namespace hpmc
{

namespace detail
{

//! Helper class to manage shuffled update orders in a GlobalVector
/*! Stores an update order from 0 to N-1, inclusive, and can be resized. shuffle() shuffles the order of elements
    to a new random permutation. operator [i] gets the index of the item at order i in the current shuffled sequence.

    NOTE: this should supersede UpdateOrder

    \note we use GPUArrays instead of GlobalArrays currently to allow host access to the shuffled order without an
          unnecessary hipDeviceSynchronize()

    \ingroup hpmc_data_structs
*/
class UpdateOrderGPU
    {
    public:
        //! Constructor
        /*! \param seed Random number seed
            \param N number of integers to shuffle
        */
        UpdateOrderGPU(std::shared_ptr<const ExecutionConfiguration> exec_conf, unsigned int seed, unsigned int N=0)
            : m_seed(seed), m_is_reversed(false), m_update_order(exec_conf), m_reverse_update_order(exec_conf)
            {
            resize(N);
            }

        //! Resize the order
        /*! \param N new size
            \post The order is 0, 1, 2, ... N-1
        */
        void resize(unsigned int N)
            {
            if (!N || N == m_update_order.size())
                return;

            // initialize the update order
            m_update_order.resize(N);
            m_reverse_update_order.resize(N);

            ArrayHandle<unsigned int> h_update_order(m_update_order, access_location::host, access_mode::overwrite);
            ArrayHandle<unsigned int> h_reverse_update_order(m_reverse_update_order, access_location::host, access_mode::overwrite);

            for (unsigned int i = 0; i < N; i++)
                {
                h_update_order.data[i] = i;
                h_reverse_update_order.data[i] = N - i - 1;
                }
            m_is_reversed = false;
            }

        //! Shuffle the order
        /*! \param timestep Current timestep of the simulation
            \note \a timestep is used to seed the RNG, thus assuming that the order is shuffled only once per
            timestep.
        */
        void shuffle(unsigned int timestep, unsigned int select = 0)
            {
            hoomd::RandomGenerator rng(hoomd::RNGIdentifier::HPMCMonoShuffle, m_seed, timestep, select);

            // reverse the order with 1/2 probability
            m_is_reversed = hoomd::UniformIntDistribution(1)(rng);
            }

        //! Access element of the shuffled order
        unsigned int operator[](unsigned int i)
            {
            const GlobalVector<unsigned int>& update_order = m_is_reversed ? m_reverse_update_order : m_update_order;
            ArrayHandle<unsigned int> h_update_order(update_order, access_location::host, access_mode::read);
            return h_update_order.data[i];
            }

        //! Access the underlying GlobalVector
        const GlobalVector<unsigned int> & get() const
            {
            if (m_is_reversed)
                return m_reverse_update_order;
            else
                return m_update_order;
            }

    private:
        unsigned int m_seed;                               //!< Random number seed
        bool m_is_reversed;                                //!< True if order is reversed
        GlobalVector<unsigned int> m_update_order;            //!< Update order
        GlobalVector<unsigned int> m_reverse_update_order;    //!< Inverse permutation
    };

} // end namespace detail

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
            m_tuner_update_pdata->setPeriod(period*this->m_nselect);
            m_tuner_update_pdata->setEnabled(enable);

            m_tuner_moves->setPeriod(period*this->m_nselect);
            m_tuner_moves->setEnabled(enable);

            m_tuner_narrow->setPeriod(period*this->m_nselect);
            m_tuner_narrow->setEnabled(enable);

            if (this->m_patch && !this->m_patch_log)
                {
                this->m_patch->setAutotunerParams(enable,period*this->m_nselect);
                }

            m_tuner_depletants->setPeriod(period*this->m_nselect);
            m_tuner_depletants->setEnabled(enable);

            m_tuner_excell_block_size->setPeriod(period);
            m_tuner_excell_block_size->setEnabled(enable);

            m_tuner_accept->setPeriod(period);
            m_tuner_accept->setEnabled(enable);
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
        uint3 m_last_dim;                                    //!< Dimensions of the cell list on the last call to update
        unsigned int m_last_nmax;                            //!< Last cell list NMax value allocated in excell

        GlobalArray<unsigned int> m_excell_idx;              //!< Particle indices in expanded cells
        GlobalArray<unsigned int> m_excell_size;             //!< Number of particles in each expanded cell
        Index2D m_excell_list_indexer;                       //!< Indexer to access elements of the excell_idx list

        std::unique_ptr<Autotuner> m_tuner_moves;            //!< Autotuner for proposing moves
        std::unique_ptr<Autotuner> m_tuner_narrow;           //!< Autotuner for the narrow phase
        std::unique_ptr<Autotuner> m_tuner_update_pdata;    //!< Autotuner for the update step group and block sizes
        std::unique_ptr<Autotuner> m_tuner_excell_block_size;  //!< Autotuner for excell block_size
        std::unique_ptr<Autotuner> m_tuner_accept;           //!< Autotuner for acceptance kernel
        std::unique_ptr<Autotuner> m_tuner_depletants;       //!< Autotuner for inserting depletants

        GlobalArray<Scalar4> m_trial_postype;                 //!< New positions (and type) of particles
        GlobalArray<Scalar4> m_trial_orientation;             //!< New orientations
        GlobalArray<unsigned int> m_trial_move_type;          //!< Flags to indicate which type of move
        GlobalArray<unsigned int> m_reject_out_of_cell;       //!< Flags to reject particle moves if they are out of the cell, per particle
        GlobalArray<unsigned int> m_reject;                   //!< Flags to reject particle moves, per particle
        GlobalArray<unsigned int> m_reject_out;               //!< Flags to reject particle moves, per particle (temporary)

        GlobalArray<unsigned int> m_nlist;                       //!< List of overlapping particles
        GlobalArray<unsigned int> m_nneigh;                     //!< Number of neighbors
        detail::UpdateOrderGPU m_update_order;                   //!< Particle update order
        unsigned int m_maxn;                                     //!< Max number of neighbors
        GlobalArray<unsigned int> m_overflow;                    //!< Overflow condition for neighbor list
        GlobalArray<unsigned int> m_condition;                  //!< Condition of acceptance kernel

        //! For energy evaluation
        unsigned int m_maxn_patch;                            //!< Maximum number of patch neighbors
        GlobalArray<unsigned int> m_overflow_patch;           //!< Overflow condition for neighbor list
        GlobalArray<unsigned int> m_nlist_patch_old;          //!< List of neighbors in old config
        GlobalArray<float> m_energy_old;                      //!< Energy contribution per neighbor in old config
        GlobalArray<unsigned int> m_nneigh_patch_old;         //!< Number of neighbors in old config
        GlobalArray<unsigned int> m_nlist_patch_new;          //!< List of neighbors in new config
        GlobalArray<float> m_energy_new;                      //!< Energy contribution per neighbor in new config
        GlobalArray<unsigned int> m_nneigh_patch_new;         //!< Number of neighbors in new config
        GlobalArray<Scalar> m_additive_cutoff;                //!< Per-type additive cutoffs from patch potential

        GlobalArray<hpmc_counters_t> m_counters;                    //!< Per-device counters
        GlobalArray<hpmc_implicit_counters_t> m_implicit_counters;  //!< Per-device counters for depletants

        //!< Variables for implicit depletants
        GlobalArray<Scalar> m_lambda;                              //!< Poisson means, per type pair

        //! Set up excell_list
        virtual void initializeExcellMem();

        //! Reallocate nlist as necessary
        bool checkReallocate();

        //! Reallocate nlist as necessary for energy evaluation
        bool checkReallocatePatch();

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
      m_update_order(this->m_exec_conf, seed+this->m_exec_conf->getRank()),
      m_maxn(0), m_maxn_patch(0)
    {
    this->m_cl->setRadius(1);
    this->m_cl->setComputeTDB(false);
    this->m_cl->setFlagType();
    this->m_cl->setComputeIdx(true);

    // with multiple GPUs, request a cell list per device
    m_cl->setPerDevice(this->m_exec_conf->allConcurrentManagedAccess());

    // set last dim to a bogus value so that it will re-init on the first call
    m_last_dim = make_uint3(0xffffffff, 0xffffffff, 0xffffffff);
    m_last_nmax = 0xffffffff;

    hipDeviceProp_t dev_prop = this->m_exec_conf->dev_prop;
    m_tuner_moves.reset(new Autotuner(dev_prop.warpSize, dev_prop.maxThreadsPerBlock, dev_prop.warpSize, 5, 1000000, "hpmc_moves", this->m_exec_conf));
    m_tuner_update_pdata.reset(new Autotuner(dev_prop.warpSize, dev_prop.maxThreadsPerBlock, dev_prop.warpSize, 5, 1000000, "hpmc_update_pdata", this->m_exec_conf));
    m_tuner_excell_block_size.reset(new Autotuner(dev_prop.warpSize, dev_prop.maxThreadsPerBlock, dev_prop.warpSize, 5, 1000000, "hpmc_excell_block_size", this->m_exec_conf));

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

    m_tuner_accept.reset(new Autotuner(valid_params, 5, 100000, "hpmc_accept", this->m_exec_conf));
    m_tuner_narrow.reset(new Autotuner(valid_params, 5, 100000, "hpmc_narrow", this->m_exec_conf));
    m_tuner_depletants.reset(new Autotuner(valid_params, 5, 100000, "hpmc_depletants", this->m_exec_conf));

    // initialize memory
    GlobalArray<Scalar4>(1,this->m_exec_conf).swap(m_trial_postype);
    TAG_ALLOCATION(m_trial_postype);

    GlobalArray<Scalar4>(1, this->m_exec_conf).swap(m_trial_orientation);
    TAG_ALLOCATION(m_trial_orientation);

    GlobalArray<unsigned int>(1,this->m_exec_conf).swap(m_trial_move_type);
    TAG_ALLOCATION(m_trial_move_type);

    GlobalArray<unsigned int>(1, this->m_exec_conf).swap(m_reject_out_of_cell);
    TAG_ALLOCATION(m_reject_out_of_cell);

    GlobalArray<unsigned int>(1, this->m_exec_conf).swap(m_reject);
    TAG_ALLOCATION(m_reject);

    GlobalArray<unsigned int>(1, this->m_exec_conf).swap(m_reject_out);
    TAG_ALLOCATION(m_reject_out);

    GlobalArray<unsigned int>(1, this->m_exec_conf).swap(m_nlist);
    TAG_ALLOCATION(m_nlist);

    GlobalArray<unsigned int>(1, this->m_exec_conf).swap(m_nneigh);
    TAG_ALLOCATION(m_nneigh);

    GlobalArray<unsigned int>(1, this->m_exec_conf).swap(m_nlist_patch_old);
    TAG_ALLOCATION(m_nlist_patch_old);

    GlobalArray<unsigned int>(1, this->m_exec_conf).swap(m_nlist_patch_new);
    TAG_ALLOCATION(m_nlist_patch_new);

    GlobalArray<unsigned int>(1, this->m_exec_conf).swap(m_nneigh_patch_old);
    TAG_ALLOCATION(m_nneigh_patch_old);

    GlobalArray<unsigned int>(1, this->m_exec_conf).swap(m_nneigh_patch_new);
    TAG_ALLOCATION(m_nneigh_patch_new);

    GlobalArray<float>(1, this->m_exec_conf).swap(m_energy_old);
    TAG_ALLOCATION(m_energy_old);

    GlobalArray<float>(1, this->m_exec_conf).swap(m_energy_new);
    TAG_ALLOCATION(m_energy_new);

    GlobalArray<unsigned int>(1, this->m_exec_conf).swap(m_overflow);
    TAG_ALLOCATION(m_overflow);

    GlobalArray<unsigned int>(1, this->m_exec_conf).swap(m_overflow_patch);
    TAG_ALLOCATION(m_overflow_patch);

    GlobalArray<unsigned int>(1, this->m_exec_conf).swap(m_condition);
    TAG_ALLOCATION(m_condition);

    #if defined(__HIP_PLATFORM_NVCC__)
    if (this->m_exec_conf->allConcurrentManagedAccess())
        {
        // set memory hints
        auto gpu_map = this->m_exec_conf->getGPUIds();
        cudaMemAdvise(m_condition.get(), sizeof(unsigned int), cudaMemAdviseSetPreferredLocation, cudaCpuDeviceId);
        cudaMemPrefetchAsync(m_condition.get(), sizeof(unsigned int), cudaCpuDeviceId);

        for (unsigned int idev = 0; idev < this->m_exec_conf->getNumActiveGPUs(); ++idev)
            {
            cudaMemAdvise(m_condition.get(), sizeof(unsigned int), cudaMemAdviseSetAccessedBy, gpu_map[idev]);
            }
        CHECK_CUDA_ERROR();
        }
    #endif

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

    #ifdef __HIP_PLATFORM_NVCC__
    if (this->m_exec_conf->allConcurrentManagedAccess())
        {
        // set memory hints
        auto gpu_map = this->m_exec_conf->getGPUIds();
        for (unsigned int idev = 0; idev < this->m_exec_conf->getNumActiveGPUs(); ++idev)
            {
            cudaMemAdvise(m_counters.get()+idev*m_counters.getPitch(), sizeof(hpmc_counters_t)*m_counters.getPitch(), cudaMemAdviseSetPreferredLocation, gpu_map[idev]);
            cudaMemPrefetchAsync(m_counters.get()+idev*m_counters.getPitch(), sizeof(hpmc_counters_t)*m_counters.getPitch(), gpu_map[idev]);
            }
        CHECK_CUDA_ERROR();
        }
    #endif

    // ntypes counters per GPU, separated by at least a memory page
    pitch = (getpagesize() + sizeof(hpmc_implicit_counters_t)-1)/sizeof(hpmc_implicit_counters_t);
    GlobalArray<hpmc_implicit_counters_t>(std::max(pitch, this->m_implicit_count.getNumElements()),
        this->m_exec_conf->getNumActiveGPUs(), this->m_exec_conf).swap(m_implicit_counters);
    TAG_ALLOCATION(m_implicit_counters);

    #ifdef __HIP_PLATFORM_NVCC__
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
    #endif

        {
        ArrayHandle<unsigned int> h_overflow(m_overflow, access_location::host, access_mode::overwrite);
        *h_overflow.data = 0;
        }

        {
        ArrayHandle<unsigned int> h_overflow_patch(m_overflow_patch, access_location::host, access_mode::overwrite);
        *h_overflow_patch.data = 0;
        }

    // Depletants
    unsigned int ntypes = this->m_pdata->getNTypes();
    GlobalArray<Scalar> lambda(ntypes*ntypes, this->m_exec_conf);
    m_lambda.swap(lambda);
    TAG_ALLOCATION(m_lambda);

    #ifdef __HIP_PLATFORM_NVCC__
    // memory hint for overlap matrix
    if (this->m_exec_conf->allConcurrentManagedAccess())
        {
        cudaMemAdvise(this->m_overlaps.get(), sizeof(unsigned int)*this->m_overlaps.getNumElements(), cudaMemAdviseSetReadMostly, 0);
        CHECK_CUDA_ERROR();
        }
    #endif

    // patch
    GlobalArray<Scalar>(this->m_pdata->getNTypes(), this->m_exec_conf).swap(m_additive_cutoff);
    TAG_ALLOCATION(m_additive_cutoff);
    }

template< class Shape >
IntegratorHPMCMonoGPU< Shape >::~IntegratorHPMCMonoGPU()
    {
    }

template< class Shape >
void IntegratorHPMCMonoGPU< Shape >::updateGPUAdvice()
    {
    #ifdef __HIP_PLATFORM_NVCC__
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

            cudaMemAdvise(m_reject.get()+range.first, sizeof(unsigned int)*nelem, cudaMemAdviseSetPreferredLocation, gpu_map[idev]);
            cudaMemPrefetchAsync(m_reject.get()+range.first, sizeof(unsigned int)*nelem, gpu_map[idev]);

            cudaMemAdvise(m_trial_orientation.get()+range.first, sizeof(Scalar4)*nelem, cudaMemAdviseSetPreferredLocation, gpu_map[idev]);
            cudaMemPrefetchAsync(m_trial_orientation.get()+range.first, sizeof(Scalar4)*nelem, gpu_map[idev]);

            cudaMemAdvise(m_nneigh.get()+range.first, sizeof(unsigned int)*nelem, cudaMemAdviseSetPreferredLocation, gpu_map[idev]);
            cudaMemPrefetchAsync(m_nneigh.get()+range.first, sizeof(unsigned int)*nelem, gpu_map[idev]);

            cudaMemAdvise(m_reject_out.get()+range.first, sizeof(unsigned int)*nelem, cudaMemAdviseSetPreferredLocation, gpu_map[idev]);
            cudaMemPrefetchAsync(m_reject_out.get()+range.first, sizeof(unsigned int)*nelem, gpu_map[idev]);

            cudaMemAdvise(m_reject_out_of_cell.get()+range.first, sizeof(unsigned int)*nelem, cudaMemAdviseSetPreferredLocation, gpu_map[idev]);
            cudaMemPrefetchAsync(m_reject_out_of_cell.get()+range.first, sizeof(unsigned int)*nelem, gpu_map[idev]);

            if (this->m_patch && !this->m_patch_log)
                {
                cudaMemAdvise(m_nneigh_patch_old.get()+range.first, sizeof(unsigned int)*nelem, cudaMemAdviseSetPreferredLocation, gpu_map[idev]);
                cudaMemPrefetchAsync(m_nneigh_patch_old.get()+range.first, sizeof(unsigned int)*nelem, gpu_map[idev]);

                cudaMemAdvise(m_nneigh_patch_new.get()+range.first, sizeof(unsigned int)*nelem, cudaMemAdviseSetPreferredLocation, gpu_map[idev]);
                cudaMemPrefetchAsync(m_nneigh_patch_new.get()+range.first, sizeof(unsigned int)*nelem, gpu_map[idev]);
                }
            CHECK_CUDA_ERROR();
            }
        }
    #endif
    }

template< class Shape >
void IntegratorHPMCMonoGPU< Shape >::update(unsigned int timestep)
    {
    IntegratorHPMC::update(timestep);

    if (this->m_patch && !this->m_patch_log)
        {
        ArrayHandle<Scalar> h_additive_cutoff(m_additive_cutoff, access_location::host, access_mode::overwrite);
        for (unsigned int itype = 0; itype < this->m_pdata->getNTypes(); ++itype)
            {
            h_additive_cutoff.data[itype] = this->m_patch->getAdditiveCutoff(itype);
            }
        }

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

        // if the cell list is a different size than last time, reinitialize the expanded cell list
        uint3 cur_dim = this->m_cl->getDim();
        if (m_last_dim.x != cur_dim.x || m_last_dim.y != cur_dim.y || m_last_dim.z != cur_dim.z
            || m_last_nmax != this->m_cl->getNmax())
            {
            initializeExcellMem();

            m_last_dim = cur_dim;
            m_last_nmax = this->m_cl->getNmax();
            }

        // test if we are in domain decomposition mode
        bool domain_decomposition = false;
#ifdef ENABLE_MPI
        if (this->m_comm)
            domain_decomposition = true;
#endif

        // resize some arrays
        bool resized = m_reject.getNumElements() < this->m_pdata->getMaxN();

        bool update_gpu_advice = false;

        if (resized)
            {
            m_reject.resize(this->m_pdata->getMaxN());
            m_reject_out_of_cell.resize(this->m_pdata->getMaxN());
            m_reject_out.resize(this->m_pdata->getMaxN());
            m_nneigh.resize(this->m_pdata->getMaxN());
            m_trial_postype.resize(this->m_pdata->getMaxN());
            m_trial_orientation.resize(this->m_pdata->getMaxN());
            m_trial_move_type.resize(this->m_pdata->getMaxN());

            update_gpu_advice = true;
            }

        if (m_nneigh_patch_old.getNumElements() < this->m_pdata->getMaxN()
            && this->m_patch && !this->m_patch_log)
            {
            m_nneigh_patch_old.resize(this->m_pdata->getMaxN());
            m_nneigh_patch_new.resize(this->m_pdata->getMaxN());

            update_gpu_advice = true;
            }


        if (update_gpu_advice)
            updateGPUAdvice();

        m_update_order.resize(this->m_pdata->getN());

        // access the cell list data
        ArrayHandle<unsigned int> d_cell_size(this->m_cl->getCellSizeArray(), access_location::device, access_mode::read);
        ArrayHandle<unsigned int> d_cell_idx(this->m_cl->getIndexArray(), access_location::device, access_mode::read);
        ArrayHandle<unsigned int> d_cell_adj(this->m_cl->getCellAdjArray(), access_location::device, access_mode::read);

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
            hipMemset(d_counters_per_device.data, 0, sizeof(hpmc_counters_t)*this->m_counters.getNumElements());
            if (this->m_exec_conf->isCUDAErrorCheckingEnabled()) CHECK_CUDA_ERROR();

            ArrayHandle<hpmc_implicit_counters_t> d_implicit_counters_per_device(this->m_implicit_counters, access_location::device, access_mode::overwrite);
            hipMemset(d_implicit_counters_per_device.data, 0, sizeof(hpmc_implicit_counters_t)*this->m_implicit_counters.getNumElements());
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
        this->m_update_order.shuffle(timestep);

        // expanded cells & neighbor list
        ArrayHandle< unsigned int > d_excell_idx(m_excell_idx, access_location::device, access_mode::overwrite);
        ArrayHandle< unsigned int > d_excell_size(m_excell_size, access_location::device, access_mode::overwrite);

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
                            this->m_exec_conf->getNumActiveGPUs(),
                            this->m_tuner_excell_block_size->getParam());
        if (this->m_exec_conf->isCUDAErrorCheckingEnabled()) CHECK_CUDA_ERROR();
        this->m_tuner_excell_block_size->end();

        bool reallocate = false;
        // depletants
        ArrayHandle<Scalar> d_lambda(m_lambda, access_location::device, access_mode::read);

        for (unsigned int i = 0; i < this->m_nselect; i++)
            {
                { // ArrayHandle scope
                ArrayHandle<unsigned int> d_update_order_by_ptl(m_update_order.get(), access_location::device, access_mode::read);
                ArrayHandle<unsigned int> d_nlist(m_nlist, access_location::device, access_mode::read);
                ArrayHandle<unsigned int> d_nneigh(m_nneigh, access_location::device, access_mode::read);
                ArrayHandle<unsigned int> d_overflow(m_overflow, access_location::device, access_mode::read);
                ArrayHandle<unsigned int> d_reject_out_of_cell(m_reject_out_of_cell, access_location::device, access_mode::overwrite);

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
                gpu::hpmc_args_t args(
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
                    this->m_seed + this->m_exec_conf->getRank()*this->m_nselect + i,
                    d_d.data,
                    d_a.data,
                    d_overlaps.data,
                    this->m_overlap_idx,
                    this->m_move_ratio,
                    timestep,
                    this->m_sysdef->getNDimensions(),
                    box,
                    i,
                    ghost_fraction,
                    domain_decomposition,
                    0, // block size
                    0, // tpp
                    d_reject_out_of_cell.data,
                    d_trial_postype.data,
                    d_trial_orientation.data,
                    d_trial_move_type.data,
                    d_update_order_by_ptl.data,
                    d_excell_idx.data,
                    d_excell_size.data,
                    m_excell_list_indexer,
                    d_nlist.data,
                    d_nneigh.data,
                    m_maxn,
                    d_overflow.data,
                    i == 0,
                    this->m_exec_conf->dev_prop,
                    this->m_pdata->getGPUPartition());

                // propose trial moves, \sa gpu::kernel::hpmc_moves

                // reset acceptance results and move types
                m_tuner_moves->begin();
                args.block_size = m_tuner_moves->getParam();
                gpu::hpmc_gen_moves<Shape>(args, params.data());
                if (this->m_exec_conf->isCUDAErrorCheckingEnabled())
                    CHECK_CUDA_ERROR();
                m_tuner_moves->end();
                }

            // make sure neighbor list size is sufficient before running the kernels
            checkReallocate();

            do
                {
                    { // ArrayHandle scope
                    ArrayHandle<unsigned int> d_update_order_by_ptl(m_update_order.get(), access_location::device, access_mode::read);
                    ArrayHandle<unsigned int> d_nlist(m_nlist, access_location::device, access_mode::overwrite);
                    ArrayHandle<unsigned int> d_nneigh(m_nneigh, access_location::device, access_mode::overwrite);
                    ArrayHandle<unsigned int> d_overflow(m_overflow, access_location::device, access_mode::readwrite);
                    ArrayHandle<unsigned int> d_reject_out_of_cell(m_reject_out_of_cell, access_location::device, access_mode::read);

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
                    gpu::hpmc_args_t args(
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
                        this->m_exec_conf->getRank()*this->m_nselect + i,
                        ghost_fraction,
                        domain_decomposition,
                        0, // block size
                        0, // tpp
                        d_reject_out_of_cell.data,
                        d_trial_postype.data,
                        d_trial_orientation.data,
                        d_trial_move_type.data,
                        d_update_order_by_ptl.data,
                        d_excell_idx.data,
                        d_excell_size.data,
                        m_excell_list_indexer,
                        d_nlist.data,
                        d_nneigh.data,
                        m_maxn,
                        d_overflow.data,
                        i == 0,
                        this->m_exec_conf->dev_prop,
                        this->m_pdata->getGPUPartition());

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
                    ArrayHandle<unsigned int> d_update_order_by_ptl(m_update_order.get(), access_location::device, access_mode::read);
                    ArrayHandle<unsigned int> d_nlist(m_nlist, access_location::device, access_mode::readwrite);
                    ArrayHandle<unsigned int> d_nneigh(m_nneigh, access_location::device, access_mode::readwrite);
                    ArrayHandle<unsigned int> d_overflow(m_overflow, access_location::device, access_mode::readwrite);
                    ArrayHandle<unsigned int> d_reject_out_of_cell(m_reject_out_of_cell, access_location::device, access_mode::read);

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
                    gpu::hpmc_args_t args(
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
                        this->m_exec_conf->getRank()*this->m_nselect + i,
                        ghost_fraction,
                        domain_decomposition,
                        0, // block size
                        0, // tpp
                        d_reject_out_of_cell.data,
                        d_trial_postype.data,
                        d_trial_orientation.data,
                        d_trial_move_type.data,
                        d_update_order_by_ptl.data,
                        d_excell_idx.data,
                        d_excell_size.data,
                        m_excell_list_indexer,
                        d_nlist.data,
                        d_nneigh.data,
                        m_maxn,
                        d_overflow.data,
                        i == 0,
                        this->m_exec_conf->dev_prop,
                        this->m_pdata->getGPUPartition());

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
                        args.block_size = param/10000;
                        args.tpp = param%10000;

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

            if (this->m_patch && !this->m_patch_log)
                {
                // make sure neighbor list size is sufficient before running the kernels
                checkReallocatePatch();

                do
                    {
                    // access data for proposed moves
                    ArrayHandle<Scalar4> d_trial_postype(m_trial_postype, access_location::device, access_mode::read);
                    ArrayHandle<Scalar4> d_trial_orientation(m_trial_orientation, access_location::device, access_mode::read);

                    // access the particle data
                    ArrayHandle<Scalar4> d_postype(this->m_pdata->getPositions(), access_location::device, access_mode::readwrite);
                    ArrayHandle<Scalar4> d_orientation(this->m_pdata->getOrientationArray(), access_location::device, access_mode::readwrite);

                    ArrayHandle<Scalar> d_charge(this->m_pdata->getCharges(), access_location::device, access_mode::read);
                    ArrayHandle<Scalar> d_diameter(this->m_pdata->getDiameters(), access_location::device, access_mode::read);
                    ArrayHandle<Scalar> d_additive_cutoff(m_additive_cutoff, access_location::device, access_mode::read);

                        {
                        /*
                         *  evaluate energy of old and new configuration simultaneously against the old and the new configuration
                         */
                        ArrayHandle<unsigned int> d_nlist_patch_old(m_nlist_patch_old, access_location::device, access_mode::overwrite);
                        ArrayHandle<float> d_energy_old(m_energy_old, access_location::device, access_mode::overwrite);
                        ArrayHandle<unsigned int> d_nneigh_patch_old(m_nneigh_patch_old, access_location::device, access_mode::overwrite);

                        ArrayHandle<unsigned int> d_nlist_patch_new(m_nlist_patch_new, access_location::device, access_mode::overwrite);
                        ArrayHandle<float> d_energy_new(m_energy_new, access_location::device, access_mode::overwrite);
                        ArrayHandle<unsigned int> d_nneigh_patch_new(m_nneigh_patch_new, access_location::device, access_mode::overwrite);

                        ArrayHandle<unsigned int> d_overflow_patch(m_overflow_patch, access_location::device, access_mode::readwrite);

                        PatchEnergy::gpu_args_t patch_args(
                            d_postype.data,
                            d_orientation.data,
                            d_trial_postype.data,
                            d_trial_orientation.data,
                            this->m_cl->getCellIndexer(),
                            this->m_cl->getDim(),
                            ghost_width,
                            this->m_pdata->getN(),
                            this->m_pdata->getNGhosts(),
                            this->m_pdata->getNTypes(),
                            box,
                            d_excell_idx.data,
                            d_excell_size.data,
                            m_excell_list_indexer,
                            this->m_patch->getRCut(),
                            d_additive_cutoff.data,
                            d_nlist_patch_old.data,
                            d_nneigh_patch_old.data,
                            d_energy_old.data,
                            d_nlist_patch_new.data,
                            d_nneigh_patch_new.data,
                            d_energy_new.data,
                            m_maxn_patch,
                            d_overflow_patch.data,
                            d_charge.data,
                            d_diameter.data,
                            this->m_pdata->getGPUPartition());

                        // compute patch energy on default stream
                        this->m_patch->computePatchEnergyGPU(patch_args, 0);
                        } // end ArrayHandle scope

                    reallocate = checkReallocatePatch();
                    } while (reallocate);
                } // end patch energy

            /*
             * make accept/reject decisions
             */
             bool done = false;
             while (!done)
                {
                    {
                    ArrayHandle<unsigned int> d_update_order_by_ptl(m_update_order.get(), access_location::device, access_mode::read);
                    ArrayHandle<unsigned int> d_trial_move_type(m_trial_move_type, access_location::device, access_mode::read);
                    ArrayHandle<unsigned int> d_reject_out_of_cell(m_reject_out_of_cell, access_location::device, access_mode::read);
                    ArrayHandle<unsigned int> d_reject(m_reject, access_location::device, access_mode::readwrite);
                    ArrayHandle<unsigned int> d_reject_out(m_reject_out, access_location::device, access_mode::overwrite);
                    ArrayHandle<unsigned int> d_nneigh(m_nneigh, access_location::device, access_mode::read);
                    ArrayHandle<unsigned int> d_nlist(m_nlist, access_location::device, access_mode::read);
                    ArrayHandle<unsigned int> d_condition(m_condition, access_location::device, access_mode::overwrite);

                    // patch energy
                    ArrayHandle<unsigned int> d_nlist_patch_old(m_nlist_patch_old, access_location::device, access_mode::read);
                    ArrayHandle<unsigned int> d_nlist_patch_new(m_nlist_patch_new, access_location::device, access_mode::read);

                    ArrayHandle<unsigned int> d_nneigh_patch_old(m_nneigh_patch_old, access_location::device, access_mode::read);
                    ArrayHandle<unsigned int> d_nneigh_patch_new(m_nneigh_patch_new, access_location::device, access_mode::read);

                    ArrayHandle<float> d_energy_old(m_energy_old, access_location::device, access_mode::read);
                    ArrayHandle<float> d_energy_new(m_energy_new, access_location::device, access_mode::read);

                    // reset condition flag
                    hipMemsetAsync(d_condition.data, 0, sizeof(unsigned int));
                    if (this->m_exec_conf->isCUDAErrorCheckingEnabled())
                        CHECK_CUDA_ERROR();

                    this->m_exec_conf->beginMultiGPU();
                    m_tuner_accept->begin();
                    unsigned int param = m_tuner_accept->getParam();
                    unsigned int block_size = param/10000;
                    unsigned int tpp = param%10000;
                    gpu::hpmc_accept(d_update_order_by_ptl.data,
                        d_trial_move_type.data,
                        d_reject_out_of_cell.data,
                        d_reject.data,
                        d_reject_out.data,
                        d_nneigh.data,
                        d_nlist.data,
                        this->m_pdata->getN() + this->m_pdata->getNGhosts(),
                        this->m_pdata->getN(),
                        this->m_pdata->getGPUPartition(),
                        m_maxn,
                        (this->m_patch != 0) && !this->m_patch_log,
                        d_nlist_patch_old.data,
                        d_nlist_patch_new.data,
                        d_nneigh_patch_old.data,
                        d_nneigh_patch_new.data,
                        d_energy_old.data,
                        d_energy_new.data,
                        m_maxn_patch,
                        d_condition.data,
                        this->m_seed,
                        this->m_exec_conf->getRank()*this->m_nselect + i,
                        timestep,
                        block_size,
                        tpp);

                    if (this->m_exec_conf->isCUDAErrorCheckingEnabled())
                        CHECK_CUDA_ERROR();
                    m_tuner_accept->end();
                    this->m_exec_conf->endMultiGPU();

                    }
                // update reject flags
                std::swap(m_reject,  m_reject_out);

                    {
                    ArrayHandle<unsigned int> h_condition(m_condition, access_location::host, access_mode::read);
                    if (*h_condition.data == 0)
                        done = true;
                    }
                } //end while (!done)

                {
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
                ArrayHandle<unsigned int> d_reject(m_reject, access_location::device, access_mode::read);

                // Update the particle data and statistics
                this->m_exec_conf->beginMultiGPU();
                m_tuner_update_pdata->begin();
                gpu::hpmc_update_args_t args(
                    d_postype.data,
                    d_orientation.data,
                    ngpu > 1 ? d_counters_per_device.data : d_counters.data,
                    this->m_pdata->getGPUPartition(),
                    d_trial_postype.data,
                    d_trial_orientation.data,
                    d_trial_move_type.data,
                    d_reject.data,
                    m_maxn,
                    m_tuner_update_pdata->getParam()
                    );
                gpu::hpmc_update_pdata<Shape>(args, params.data());
                if (this->m_exec_conf->isCUDAErrorCheckingEnabled())
                    CHECK_CUDA_ERROR();
                m_tuner_update_pdata->end();
                this->m_exec_conf->endMultiGPU();
                }
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

    bool maxn_changed = false;
    if (req_maxn > m_maxn)
        {
        m_maxn = req_maxn;
        maxn_changed = true;
        }

    unsigned int req_size_nlist = m_maxn*this->m_pdata->getN();

    // resize
    bool reallocate = req_size_nlist > m_nlist.getNumElements();
    if (reallocate)
        {
        this->m_exec_conf->msg->notice(9) << "hpmc resizing neighbor list " << m_nlist.getNumElements() << " -> " << req_size_nlist << std::endl;

        GlobalArray<unsigned int> nlist(req_size_nlist, this->m_exec_conf);
        m_nlist.swap(nlist);
        TAG_ALLOCATION(m_nlist);

        #ifdef __HIP_PLATFORM_NVCC__
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
                CHECK_CUDA_ERROR();
                }
            }
        #endif
        }
    return reallocate || maxn_changed;
    }

template< class Shape >
bool IntegratorHPMCMonoGPU< Shape >::checkReallocatePatch()
    {
    // read back overflow condition and resize as necessary
    ArrayHandle<unsigned int> h_overflow_patch(m_overflow_patch, access_location::host, access_mode::read);
    unsigned int req_maxn = *h_overflow_patch.data;

    bool maxn_changed = false;
    if (req_maxn > m_maxn_patch)
        {
        m_maxn_patch = req_maxn;
        maxn_changed = true;
        }

    unsigned int req_size_nlist = m_maxn_patch*this->m_pdata->getN();

    // resize
    bool reallocate = req_size_nlist > m_nlist_patch_old.getNumElements();
    if (reallocate)
        {
        this->m_exec_conf->msg->notice(9) << "hpmc resizing patch neighbor list " << m_nlist_patch_old.getNumElements() << " -> " << req_size_nlist << std::endl;

        GlobalArray<unsigned int> nlist_patch_old(req_size_nlist, this->m_exec_conf);
        m_nlist_patch_old.swap(nlist_patch_old);
        TAG_ALLOCATION(m_nlist_patch_old);

        GlobalArray<unsigned int> nlist_patch_new(req_size_nlist, this->m_exec_conf);
        m_nlist_patch_new.swap(nlist_patch_new);
        TAG_ALLOCATION(m_nlist_patch_new);

        GlobalArray<float> energy_old(req_size_nlist, this->m_exec_conf);
        m_energy_old.swap(energy_old);
        TAG_ALLOCATION(m_energy_old);

        GlobalArray<float> energy_new(req_size_nlist, this->m_exec_conf);
        m_energy_new.swap(energy_new);
        TAG_ALLOCATION(m_energy_new);

        #ifdef __HIP_PLATFORM_NVCC__
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

                cudaMemAdvise(m_nlist_patch_old.get()+range.first*m_maxn_patch, sizeof(unsigned int)*nelem*m_maxn_patch, cudaMemAdviseSetPreferredLocation, gpu_map[idev]);
                cudaMemPrefetchAsync(m_nlist_patch_old.get()+range.first*m_maxn_patch, sizeof(unsigned int)*nelem*m_maxn_patch, gpu_map[idev]);
                CHECK_CUDA_ERROR();

                cudaMemAdvise(m_nlist_patch_new.get()+range.first*m_maxn_patch, sizeof(unsigned int)*nelem*m_maxn_patch, cudaMemAdviseSetPreferredLocation, gpu_map[idev]);
                cudaMemPrefetchAsync(m_nlist_patch_new.get()+range.first*m_maxn_patch, sizeof(unsigned int)*nelem*m_maxn_patch, gpu_map[idev]);
                CHECK_CUDA_ERROR();

                cudaMemAdvise(m_energy_old.get()+range.first*m_maxn_patch, sizeof(float)*nelem*m_maxn_patch, cudaMemAdviseSetPreferredLocation, gpu_map[idev]);
                cudaMemPrefetchAsync(m_energy_old.get()+range.first*m_maxn_patch, sizeof(float)*nelem*m_maxn_patch, gpu_map[idev]);
                CHECK_CUDA_ERROR();

                cudaMemAdvise(m_energy_new.get()+range.first*m_maxn_patch, sizeof(float)*nelem*m_maxn_patch, cudaMemAdviseSetPreferredLocation, gpu_map[idev]);
                cudaMemPrefetchAsync(m_energy_new.get()+range.first*m_maxn_patch, sizeof(float)*nelem*m_maxn_patch, gpu_map[idev]);
                CHECK_CUDA_ERROR();
                }
            }
        #endif
        }
    return reallocate || maxn_changed;
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

    #if defined(__HIP_PLATFORM_NVCC__) && 0 // excell is currently not multi-GPU optimized, let the CUDA driver figure this out
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
    #endif
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

        // resize arrays
        GlobalArray<Scalar> lambda(ntypes*ntypes, this->m_exec_conf);
        m_lambda.swap(lambda);
        TAG_ALLOCATION(m_lambda);

        GlobalArray<Scalar> additive_cutoff(ntypes*ntypes, this->m_exec_conf);
        m_additive_cutoff.swap(additive_cutoff);
        TAG_ALLOCATION(m_additive_cutoff);

        // ntypes counters per GPU, separated by at least a memory page
        unsigned int pitch = (getpagesize() + sizeof(hpmc_implicit_counters_t)-1)/sizeof(hpmc_implicit_counters_t);
        GlobalArray<hpmc_implicit_counters_t>(std::max(pitch, this->m_implicit_count.getNumElements()),
            this->m_exec_conf->getNumActiveGPUs(), this->m_exec_conf).swap(m_implicit_counters);
        TAG_ALLOCATION(m_implicit_counters);

        #ifdef __HIP_PLATFORM_NVCC__
        if (this->m_exec_conf->allConcurrentManagedAccess())
            {
            // memory hint for overlap matrix
            cudaMemAdvise(this->m_overlaps.get(), sizeof(unsigned int)*this->m_overlaps.getNumElements(), cudaMemAdviseSetReadMostly, 0);
            CHECK_CUDA_ERROR();
            }
        #endif
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

    #ifdef __HIP_PLATFORM_NVCC__
    // set memory hints
    cudaMemAdvise(this->m_params.data(), this->m_params.size()*sizeof(typename Shape::param_type), cudaMemAdviseSetReadMostly, 0);
    CHECK_CUDA_ERROR();
    #endif

    // sync up so we can access the parameters
    hipDeviceSynchronize();

    for (unsigned int i = 0; i < this->m_pdata->getNTypes(); ++i)
        {
        // attach nested memory regions
        this->m_params[i].set_memory_hint();
        CHECK_CUDA_ERROR();
        }

    // reinitialize poisson means array
    ArrayHandle<Scalar> h_lambda(m_lambda, access_location::host, access_mode::overwrite);
    Index2D typpair_idx(this->m_pdata->getNTypes());

    // reset to zero
    std::fill(h_lambda.data, h_lambda.data + typpair_idx.getNumElements(), 0.0);

    for (unsigned int i_type = 0; i_type < this->m_pdata->getNTypes(); ++i_type)
        {
        Shape shape_i(quat<Scalar>(), this->m_params[i_type]);
        Scalar d_i(shape_i.getCircumsphereDiameter());
        Scalar range = this->m_quermass ? 2.0*this->m_sweep_radius : d_i;

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
     pybind11::class_<IntegratorHPMCMonoGPU<Shape>, IntegratorHPMCMono<Shape>, std::shared_ptr< IntegratorHPMCMonoGPU<Shape> > >(m, name.c_str())
              .def(pybind11::init< std::shared_ptr<SystemDefinition>, std::shared_ptr<CellList>, unsigned int >())
              ;
    }

} // end namespace hpmc

#endif // ENABLE_HIP
