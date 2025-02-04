// Copyright (c) 2009-2024 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

/*! \file LoadBalancerGPU.cc
    \brief Defines the LoadBalancerGPU class
*/

#ifdef ENABLE_HIP
#include "LoadBalancerGPU.h"
#include "LoadBalancerGPU.cuh"
#include <hip/hip_runtime.h>

#include "CachedAllocator.h"

using namespace std;

namespace hoomd
    {
/*!
 * \param sysdef System definition
 * \param decomposition Domain decomposition
 */
LoadBalancerGPU::LoadBalancerGPU(std::shared_ptr<SystemDefinition> sysdef,
                                 std::shared_ptr<Trigger> trigger)
    : LoadBalancer(sysdef, trigger)
    {
    // allocate data connected to the maximum number of particles
    m_pdata->getMaxParticleNumberChangeSignal()
        .connect<LoadBalancerGPU, &LoadBalancerGPU::slotMaxNumChanged>(this);

    GPUArray<unsigned int> off_ranks(m_pdata->getMaxN(), m_exec_conf);
    m_off_ranks.swap(off_ranks);

    m_tuner.reset(new Autotuner<1>({AutotunerBase::makeBlockSizeRange(m_exec_conf)},
                                   this->m_exec_conf,
                                   "load_balance"));
    m_autotuners.push_back(m_tuner);
    }

LoadBalancerGPU::~LoadBalancerGPU()
    {
    // disconnect from the signal
    m_pdata->getMaxParticleNumberChangeSignal()
        .disconnect<LoadBalancerGPU, &LoadBalancerGPU::slotMaxNumChanged>(this);
    }

#ifdef ENABLE_MPI
void LoadBalancerGPU::countParticlesOffRank(std::map<unsigned int, unsigned int>& cnts)
    {
    // do nothing if rank doesn't own any particles
    if (m_pdata->getN() == 0)
        {
        return;
        }

        // mark the current ranks of each particle (hijack the comm flags array)
        {
        ArrayHandle<unsigned int> d_comm_flag(m_pdata->getCommFlags(),
                                              access_location::device,
                                              access_mode::overwrite);
        ArrayHandle<Scalar4> d_pos(m_pdata->getPositions(),
                                   access_location::device,
                                   access_mode::read);
        ArrayHandle<unsigned int> d_cart_ranks(m_decomposition->getCartRanks(),
                                               access_location::device,
                                               access_mode::read);

        m_tuner->begin();
        kernel::gpu_load_balance_mark_rank(d_comm_flag.data,
                                           d_pos.data,
                                           d_cart_ranks.data,
                                           m_decomposition->getGridPos(),
                                           m_pdata->getBox(),
                                           m_decomposition->getDomainIndexer(),
                                           m_pdata->getN(),
                                           m_tuner->getParam()[0]);
        if (m_exec_conf->isCUDAErrorCheckingEnabled())
            CHECK_CUDA_ERROR();
        m_tuner->end();
        }

    // select the particles that should be sent to other ranks
    vector<unsigned int> off_rank;
        {
        ArrayHandle<unsigned int> d_comm_flag(m_pdata->getCommFlags(),
                                              access_location::device,
                                              access_mode::read);
        ArrayHandle<unsigned int> d_off_ranks(m_off_ranks,
                                              access_location::device,
                                              access_mode::overwrite);

        // size the temporary storage
        const unsigned int n_off_rank
            = kernel::gpu_load_balance_select_off_rank(d_off_ranks.data,
                                                       d_comm_flag.data,
                                                       m_pdata->getN(),
                                                       m_exec_conf->getRank());

        // copy just the subset of particles that are off rank on the device into host memory
        // this can save substantially on the memcpy if there are many particles on a rank
        off_rank.resize(n_off_rank);
        hipMemcpy(&off_rank[0],
                  d_off_ranks.data,
                  sizeof(unsigned int) * n_off_rank,
                  hipMemcpyDeviceToHost);
        }

    // perform the counting on the host
    for (unsigned int cur_p = 0; cur_p < off_rank.size(); ++cur_p)
        {
        cnts[off_rank[cur_p]]++;
        }
    }
#endif // ENABLE_MPI

namespace detail
    {
void export_LoadBalancerGPU(pybind11::module& m)
    {
    pybind11::class_<LoadBalancerGPU, LoadBalancer, std::shared_ptr<LoadBalancerGPU>>(
        m,
        "LoadBalancerGPU")
        .def(pybind11::init<std::shared_ptr<SystemDefinition>, std::shared_ptr<Trigger>>());
    }

    } // end namespace detail

    } // end namespace hoomd

#endif // ENABLE_HIP
