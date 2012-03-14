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

// Maintainer: ndtrung

#include <boost/shared_ptr.hpp>

#ifndef __FIRE_ENERGY_MINIMIZER_RIGID_GPU_H__
#define __FIRE_ENERGY_MINIMIZER_RIGID_GPU_H__

#include "FIREEnergyMinimizerRigid.h"

/*! \file FIREEnergyMinimizerRigidGPU.h
    \brief Declares a class for rigid body energy minimization on GPU
*/

#ifdef NVCC
#error This header cannot be compiled by nvcc
#endif

//! Finds the nearest basin in the potential energy landscape
/*! \b Overview
    
    \ingroup updaters
*/
class FIREEnergyMinimizerRigidGPU : public FIREEnergyMinimizerRigid
    {
    public:
        //! Constructs the minimizer and associates it with the system
        FIREEnergyMinimizerRigidGPU(boost::shared_ptr<SystemDefinition>, boost::shared_ptr<ParticleGroup>, Scalar, bool=true);

        //! Destroys the minimizer
        virtual ~FIREEnergyMinimizerRigidGPU() {}
        
        //! Resets the minimizer
        virtual void reset();

        //! Iterates forward one step
        virtual void update(unsigned int);
        
    protected:
        GPUArray<float> m_sum_pe;                  //!< memory space for the sum over potential energy
        GPUArray<float> m_sum_Pt;                 //!< memory space for the sum over P, vsq, fsq
        GPUArray<float> m_sum_Pr;                 //!< memory space for the sum over P, wsq, tsq 
        
        unsigned int m_block_size;                //!< block size for partial sum memory
        unsigned int m_num_blocks;                //!< number of memory blocks reserved for partial sum memory
        GPUArray<float> m_partial_sum_pe;         //!< memory space for partial sum over P
    private:

    };

//! Exports the FIREEnergyMinimizerRigidGPU class to python
void export_FIREEnergyMinimizerRigidGPU();

#endif // #ifndef __FIRE_ENERGY_MINIMIZER_RIGID_GPU_H__

