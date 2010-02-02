/*
Highly Optimized Object-oriented Many-particle Dynamics -- Blue Edition
(HOOMD-blue) Open Source Software License Copyright 2008, 2009 Ames Laboratory
Iowa State University and The Regents of the University of Michigan All rights
reserved.

HOOMD-blue may contain modifications ("Contributions") provided, and to which
copyright is held, by various Contributors who have granted The Regents of the
University of Michigan the right to modify and/or distribute such Contributions.

Redistribution and use of HOOMD-blue, in source and binary forms, with or
without modification, are permitted, provided that the following conditions are
met:

* Redistributions of source code must retain the above copyright notice, this
list of conditions, and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright notice, this
list of conditions, and the following disclaimer in the documentation and/or
other materials provided with the distribution.

* Neither the name of the copyright holder nor the names of HOOMD-blue's
contributors may be used to endorse or promote products derived from this
software without specific prior written permission.

Disclaimer

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDER AND CONTRIBUTORS ``AS IS''
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, AND/OR
ANY WARRANTIES THAT THIS SOFTWARE IS FREE OF INFRINGEMENT ARE DISCLAIMED.

IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT,
INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE
OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF
ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

// $Id: FIREEnergyMinimizer.h 2587 2010-01-08 17:02:54Z joaander $
// $URL: https://codeblue.umich.edu/hoomd-blue/svn/trunk/libhoomd/updaters/FIREEnergyMinimizer.h $
// Maintainer: askeys

#include <boost/shared_ptr.hpp>

#ifndef __FIRE_ENERGY_MINIMIZER_GPU_H__
#define __FIRE_ENERGY_MINIMIZER_GPU_H__

#include "FIREEnergyMinimizer.h"

/*! \file FIREEnergyMinimizer.h
    \brief Declares a base class for all energy minimization methods
*/

//! Finds the nearest basin in the potential energy landscape
/*! \b Overview
    
    \ingroup updaters
*/
class FIREEnergyMinimizerGPU : public FIREEnergyMinimizer
    {
    public:
        //! Constructs the minimizer and associates it with the system
        FIREEnergyMinimizerGPU(boost::shared_ptr<SystemDefinition>, Scalar);

        //! Destroys the minimizer
        virtual ~FIREEnergyMinimizerGPU() {}
        
        //! Resets the minimizer
        virtual void reset();

        //! Iterates forward one step
        virtual void update(unsigned int);
        
    protected:
        //! Creates the underlying NVE integrator
        virtual void createIntegrator();  
        unsigned int m_nparticles;              //!< number of particles in the system
        unsigned int m_block_size;              //!< block size for partial sum memory
        unsigned int m_num_blocks;              //!< number of memory blocks reserved for partial sum memory
        GPUArray<float> m_partial_sum1;         //!< memory space for partial sum over P
        GPUArray<float> m_partial_sum2;         //!< memory space for partial sum over vsq
        GPUArray<float> m_partial_sum3;         //!< memory space for partial sum over asq
        GPUArray<float> m_sum;                  //!< memory space for sum over vsq
        GPUArray<float> m_sum3;                 //!< memory space for the sum over P, vsq, asq 
        
    private:

    };

//! Exports the FIREEnergyMinimizerGPU class to python
void export_FIREEnergyMinimizerGPU();

#endif // #ifndef __FIRE_ENERGY_MINIMIZER_GPU_H__

