/*
Highly Optimized Object-oriented Many-particle Dynamics -- Blue Edition
(HOOMD-blue) Open Source Software License Copyright 2009-2016 The Regents of
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

/*! \file LoadBalancerGPU.h
    \brief Declares an updater that changes the MPI domain decomposition to balance the load using the GPU
*/

#ifdef NVCC
#error This header cannot be compiled by nvcc
#endif

#ifdef ENABLE_MPI
#ifdef ENABLE_CUDA

#ifndef __LOADBALANCERGPU_H__
#define __LOADBALANCERGPU_H__

#include "GPUFlags.h"
#include "LoadBalancer.h"
#include "Autotuner.h"
#include <boost/signals2.hpp>

//! GPU implementation of dynamic load balancing
class LoadBalancerGPU : public LoadBalancer
    {
    public:
        //! Constructor
        LoadBalancerGPU(boost::shared_ptr<SystemDefinition> sysdef,
                        boost::shared_ptr<DomainDecomposition> decomposition);

        //! Destructor
        virtual ~LoadBalancerGPU();
    
        //! Set autotuner parameters
        /*!
         * \param enable Enable/disable autotuning
         * \param period period (approximate) in time steps when returning occurs
         */
        virtual void setAutotunerParams(bool enable, unsigned int period)
            {
            LoadBalancer::setAutotunerParams(enable, period);
            m_tuner->setPeriod(period);
            m_tuner->setEnabled(enable);
            }

        //! Resize the per particle data when there is a max number of particle change
        void slotMaxNumChanged()
            {
            GPUArray<unsigned int> off_ranks(m_pdata->getMaxN(), m_exec_conf);
            m_off_ranks.swap(off_ranks);
            }
    protected:
        //! Count the number of particles that have gone off either edge of the rank along a dimension on the GPU
        virtual void countParticlesOffRank(std::map<unsigned int, unsigned int>& cnts);

    private:
        boost::signals2::connection m_max_numchange_conn;   //!< Connection to max particle number change signal

        boost::scoped_ptr<Autotuner> m_tuner;   //!< Autotuner for block size counting particles
        GPUArray<unsigned int> m_off_ranks;     //!< Array to hold the ranks of particles that have moved
        GPUFlags<unsigned int> m_n_off_rank;    //!< Device flag to count the total number of particles off rank
    };

//! Export the LoadBalancerGPU to python
void export_LoadBalancerGPU();

#endif // __LOADBALANCERGPU_H__

#endif // ENABLE_CUDA
#endif // ENABLE_MPI