/*
Highly Optimized Object-oriented Many-particle Dynamics -- Blue Edition
(HOOMD-blue) Open Source Software License Copyright 2009-2015 The Regents of
the University of Michigan All rights reserved.

HOOMD-blue may contain modifications ("Contributions") provided, and to which
copyright is held, by various Contributors who have granted The Regents of the
University of Michigan the right to modify and/or distribute such Contributions.

You may redistribute, use, and create derivate works of HOOMD-blue, in source
and binary forms, provided you abide by the following conditions:

* Redistributions of source code must retain the above copyright notice, this
list of conditions, and the following disclaimer both in the code and
prominently in any materials provided with the distribution.

* Redistributions in binary form must reproduce the above copyright notice, this
list of conditions, and the following disclaimer in the documentation and/or
other materials provided with the distribution.

* All publications and presentations based on HOOMD-blue, including any reports
or published results obtained, in whole or in part, with HOOMD-blue, will
acknowledge its use according to the terms posted at the time of submission on:
http://codeblue.umich.edu/hoomd-blue/citations.html

* Any electronic documents citing HOOMD-Blue will link to the HOOMD-Blue website:
http://codeblue.umich.edu/hoomd-blue/

* Apart from the above required attributions, neither the name of the copyright
holder nor the names of HOOMD-blue's contributors may be used to endorse or
promote products derived from this software without specific prior written
permission.

Disclaimer

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDER AND CONTRIBUTORS ``AS IS'' AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, AND/OR ANY
WARRANTIES THAT THIS SOFTWARE IS FREE OF INFRINGEMENT ARE DISCLAIMED.

IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT,
INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE
OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF
ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/
// Maintainer: mphoward

/*! \file LoadBalancerGPU.cc
    \brief Defines the LoadBalancerGPU class
*/

#ifdef ENABLE_MPI
#ifdef ENABLE_CUDA

#include "LoadBalancerGPU.h"
#include "LoadBalancerGPU.cuh"

#include "CachedAllocator.h"

using namespace std;

#include <boost/bind.hpp>
using namespace boost;

#include <boost/python.hpp>
using namespace boost::python;

/*!
 * \param sysdef System definition
 * \param decomposition Domain decomposition
 */
LoadBalancerGPU::LoadBalancerGPU(boost::shared_ptr<SystemDefinition> sysdef,
                                 boost::shared_ptr<DomainDecomposition> decomposition)
    : LoadBalancer(sysdef, decomposition)
    {
    // allocate data connected to the maximum number of particles
    m_max_numchange_conn = m_pdata->connectMaxParticleNumberChange(bind(&LoadBalancerGPU::slotMaxNumChanged, this));

    GPUArray<unsigned int> off_ranks(m_pdata->getMaxN(), m_exec_conf);
    m_off_ranks.swap(off_ranks);

    GPUFlags<unsigned int> n_off_rank(m_exec_conf);
    m_n_off_rank.swap(n_off_rank);

    m_tuner.reset(new Autotuner(32, 1024, 32, 5, 100000, "load_balance", this->m_exec_conf));
    }

LoadBalancerGPU::~LoadBalancerGPU()
    {
    // disconnect from the signal
    m_max_numchange_conn.disconnect();
    }

void LoadBalancerGPU::countParticlesOffRank(std::map<unsigned int, unsigned int>& cnts)
    {
    // do nothing if rank doesn't own any particles
    if (m_pdata->getN() == 0) return;

    // mark the current ranks of each particle (hijack the comm flags array)
        {
        ArrayHandle<unsigned int> d_comm_flag(m_pdata->getCommFlags(), access_location::device, access_mode::overwrite);
        ArrayHandle<Scalar4> d_pos(m_pdata->getPositions(), access_location::device, access_mode::read);
        ArrayHandle<unsigned int> d_cart_ranks(m_decomposition->getCartRanks(), access_location::device, access_mode::read);

        m_tuner->begin();
        gpu_load_balance_mark_rank(d_comm_flag.data,
                                   d_pos.data,
                                   d_cart_ranks.data,
                                   m_decomposition->getGridPos(),
                                   m_pdata->getBox(),
                                   m_decomposition->getDomainIndexer(),
                                   m_pdata->getN(),
                                   m_tuner->getParam());
        if (m_exec_conf->isCUDAErrorCheckingEnabled()) CHECK_CUDA_ERROR();
        m_tuner->end();

        }

    // select the particles that should be sent to other ranks
    vector<unsigned int> off_rank;
        {
        ArrayHandle<unsigned int> d_comm_flag(m_pdata->getCommFlags(), access_location::device, access_mode::read);
        ArrayHandle<unsigned int> d_off_ranks(m_off_ranks, access_location::device, access_mode::overwrite);
        m_n_off_rank.resetFlags(0);

        // size the temporary storage
        void *d_tmp_storage = NULL;
        size_t tmp_storage_bytes = 0;
        gpu_load_balance_select_off_rank(d_off_ranks.data,
                                         m_n_off_rank.getDeviceFlags(),
                                         d_comm_flag.data,
                                         d_tmp_storage,
                                         tmp_storage_bytes,
                                         m_pdata->getN(),
                                         m_exec_conf->getRank());

        // always allocate a minimum of 4 bytes so that d_tmp_storage is never NULL
        size_t n_alloc = (tmp_storage_bytes > 0) ? tmp_storage_bytes : 4;
        ScopedAllocation<unsigned char> d_alloc(m_exec_conf->getCachedAllocator(), n_alloc);
        d_tmp_storage = (void*)d_alloc();

        // perform the selection
        gpu_load_balance_select_off_rank(d_off_ranks.data,
                                         m_n_off_rank.getDeviceFlags(),
                                         d_comm_flag.data,
                                         d_tmp_storage,
                                         tmp_storage_bytes,
                                         m_pdata->getN(),
                                         m_exec_conf->getRank());

        // copy just the subset of particles that are off rank on the device into host memory
        // this can save substantially on the memcpy if there are many particles on a rank
        const unsigned int n_off_rank = m_n_off_rank.readFlags();
        off_rank.resize(n_off_rank);
        cudaMemcpy(&off_rank[0], d_off_ranks.data, sizeof(unsigned int)*n_off_rank, cudaMemcpyDeviceToHost);
        }

    // perform the counting on the host
    for (unsigned int cur_p=0; cur_p < off_rank.size(); ++cur_p)
        {
        cnts[off_rank[cur_p]]++;
        }
    }

void export_LoadBalancerGPU()
    {
    class_<LoadBalancerGPU, boost::shared_ptr<LoadBalancerGPU>, bases<LoadBalancer>, boost::noncopyable>
    ("LoadBalancerGPU", init< boost::shared_ptr<SystemDefinition>, boost::shared_ptr<DomainDecomposition> >())
    ;
    }

#endif // ENABLE_CUDA
#endif // ENABLE_MPI
