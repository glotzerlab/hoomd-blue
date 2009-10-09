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

// $Id$
// $URL$
// Maintainer: joaander

#include <boost/shared_ptr.hpp>

#include "ForceCompute.h"
#include "NeighborList.h"

/*! \file GaussianForceCompute.h
    \brief Declares the GaussianForceCompute class
*/

#ifndef __GAUSSIANFORCECOMPUTE_H__
#define __GAUSSIANFORCECOMPUTE_H__

//! Computes Gaussian forces on each particle
/*! The total pair force is summed for each particle when compute() is called. Forces are only summed between
    neighboring particles with a separation distance less than \c r_cut. A NeighborList must be provided
    to identify these neighbors. Calling compute() in this class will in turn result in a call to the
    NeighborList's compute() to make sure that the neighbor list is up to date.

    \f[ V(r) = \varepsilon \exp \left[ -\frac{1}{2} \left(\frac{r}{\sigma} \right)^2 \right] \f]

    \ingroup computes
*/
class GaussianForceCompute : public ForceCompute
    {
    public:
        //! Constructs the compute
        GaussianForceCompute(boost::shared_ptr<SystemDefinition> sysdef,
                             boost::shared_ptr<NeighborList> nlist,
                             Scalar r_cut);
        
        //! Destructor
        virtual ~GaussianForceCompute();
        
        //! Set the parameters for a single type pair
        virtual void setParams(unsigned int typ1, unsigned int typ2, Scalar epsilon, Scalar sigma);
        
        //! Returns a list of log quantities this compute calculates
        virtual std::vector< std::string > getProvidedLogQuantities();
        
        //! Calculates the requested log value and returns it
        virtual Scalar getLogValue(const std::string& quantity, unsigned int timestep);
        
        //! Shifting modes that can be applied to the energy
        enum energyShiftMode
            {
            no_shift = 0,
            shift
            };
            
        //! Set the mode to use for shifting the energy
        void setShiftMode(energyShiftMode mode)
            {
            m_shift_mode = mode;
            }
            
    protected:
        boost::shared_ptr<NeighborList> m_nlist;    //!< The neighborlist to use for the computation
        Scalar m_r_cut;                             //!< Cuttoff radius beyond which the force is set to 0
        unsigned int m_ntypes;                      //!< Store the width and height of lj1 and lj2 here
        energyShiftMode m_shift_mode;               //!< Store the mode with which to handle the energy shift at r_cut
        
        // This is a low level force summing class, it ONLY sums forces, and doesn't do high
        // level concepts like mixing. That is for the caller to handle. So, I only store
        // epsilon and sigma here
        Scalar * __restrict__ m_epsilon;    //!< Parameter for computing forces (m_ntypes by m_ntypes array)
        Scalar * __restrict__ m_sigma;      //!< Parameter for computing forces (m_ntypes by m_ntypes array)
        
        //! Actually compute the forces
        virtual void computeForces(unsigned int timestep);
    };

//! Exports the GaussianForceCompute class to python
void export_GaussianForceCompute();

#endif

