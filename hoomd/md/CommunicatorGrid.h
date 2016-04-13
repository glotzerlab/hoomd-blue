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

// Maintainer: jglaser

#ifndef __COMMUNICATOR_GRID_H__
#define __COMMUNICATOR_GRID_H__

#include "hoomd/GPUArray.h"
#include "hoomd/SystemDefinition.h"
#include <boost/shared_ptr.hpp>

#ifdef ENABLE_MPI

/*! Class to communicate the boundary layer of a regular grid
 */
template<typename T>
class CommunicatorGrid
    {
    public:
        //! Constructor
        CommunicatorGrid(boost::shared_ptr<SystemDefinition> sysdef, uint3 dim,
            uint3 embed, uint3 offset, bool add_outer_layer_to_inner);

        //! Communicate grid
        virtual void communicate(const GPUArray<T>& grid);

    protected:
        boost::shared_ptr<SystemDefinition> m_sysdef;        //!< System definition
        boost::shared_ptr<ParticleData> m_pdata;             //!< Particle data
        boost::shared_ptr<const ExecutionConfiguration> m_exec_conf; //!< Execution configuration
        boost::shared_ptr<Profiler> m_prof;                  //!< Profiler

        uint3 m_dim;                                         //!< Dimensions of grid
        uint3 m_embed;                                       //!< Embedding dimensions
        uint3 m_offset;                                      //!< Offset of inner grid in array
        bool m_add_outer;                                    //!< True if outer ghost layer is added to inner cells

        std::set<unsigned int> m_neighbors;                  //!< List of unique neighbor ranks
        GPUArray<T> m_send_buf;                              //!< Send buffer
        GPUArray<T> m_recv_buf;
        GPUArray<unsigned int> m_send_idx;                   //!< Indices of grid cells in send buf
        GPUArray<unsigned int> m_recv_idx;                   //!< Indices of grid cells in recv buf
        std::map<unsigned int,unsigned int> m_begin;         //!< Begin offset of every rank in send/recv buf
        std::map<unsigned int,unsigned int> m_end;           //!< End offset of every rank in send/recv buf

        //! Initialize grid communication
        virtual void initGridComm();
    };

#endif // ENABLE_MPI
#endif // __COMMUNICATOR_GRID_H__
