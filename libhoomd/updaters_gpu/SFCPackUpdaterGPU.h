/*
Highly Optimized Object-oriented Many-particle Dynamics -- Blue Edition
(HOOMD-blue) Open Source Software License Copyright 2008-2011 Ames Laboratory
Iowa State University and The Regents of the University of Michigan All rights
reserved.

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

/*! \file SFCPackUpdaterGPU.h
    \brief Declares the SFCPackUpdaterGPU class
*/

#ifdef NVCC
#error This header cannot be compiled by nvcc
#endif

#ifdef ENABLE_CUDA

#ifdef WIN32
#pragma warning( push )
#pragma warning( disable : 4103 )
#endif

#include <boost/shared_ptr.hpp>
#include <boost/signals2.hpp>
#include <vector>
#include <utility>

#include "Updater.h"

#include "SFCPackUpdater.h"
#include "SFCPackUpdaterGPU.cuh"
#include "GPUArray.h"

#ifndef __SFCPACK_UPDATER_GPU_H__
#define __SFCPACK_UPDATER_GPU_H__

//! Sort the particles
/*! GPU implementation of SFCPackUpdater

    \ingroup updaters
*/
class SFCPackUpdaterGPU : public SFCPackUpdater
    {
    public:
        //! Constructor
        SFCPackUpdaterGPU(boost::shared_ptr<SystemDefinition> sysdef);

        //! Destructor
        virtual ~SFCPackUpdaterGPU();

    protected:
        // reallocate internal data structure
        virtual void reallocate();

    private:
        GPUArray<unsigned int> m_gpu_particle_bins;    //!< Particle bins
        GPUArray<unsigned int> m_gpu_sort_order;       //!< Generated sort order of the particles

        boost::signals2::connection m_max_particle_num_change_connection; //!< Connection to the maximum particle number change signal of particle data

        //! Helper function that actually performs the sort
        virtual void getSortedOrder2D();

        //! Helper function that actually performs the sort
        virtual void getSortedOrder3D();

        //! Apply the sorted order to the particle data
        virtual void applySortOrder();

        mgpu::ContextPtr m_mgpu_context;                    //!< MGPU context (for sorting)
    };

//! Export the SFCPackUpdaterGPU class to python
void export_SFCPackUpdaterGPU();

#endif // __SFC_PACK_UPDATER_GPU_H_

#ifdef WIN32
#pragma warning( pop )
#endif

#endif // ENABLE_CUDA
