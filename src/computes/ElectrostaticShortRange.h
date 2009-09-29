/*
Highly Optimized Object-Oriented Molecular Dynamics (HOOMD) Open
Source Software License
Copyright (c) 2008 Ames Laboratory Iowa State University
All rights reserved.

Redistribution and use of HOOMD, in source and binary forms, with or
without modification, are permitted, provided that the following
conditions are met:

* Redistributions of source code must retain the above copyright notice,
this list of conditions and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright
notice, this list of conditions and the following disclaimer in the
documentation and/or other materials provided with the distribution.

* Neither the name of the copyright holder nor the names HOOMD's
contributors may be used to endorse or promote products derived from this
software without specific prior written permission.

Disclaimer

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDER AND
CONTRIBUTORS ``AS IS''  AND ANY EXPRESS OR IMPLIED WARRANTIES,
INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY
AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED.

IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS  BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF
THE POSSIBILITY OF SUCH DAMAGE.
*/

// conditionally compile in only if boost is 1.35 or later
#include <boost/version.hpp>
#if (BOOST_VERSION >= 103500)

#include <boost/shared_ptr.hpp>

#include "ForceCompute.h"
#include "NeighborList.h"

/*! \file ElectrostaticShortRange.h
    \brief Declares a ElectroStaticShortRange
*/

#ifndef __ELECTROSTATICSHORTRANGE_H__
#define __ELECTROSTATICSHORTRANGE_H__

//! Computes the short-range electrostatic part of the force on each particle
/*! In order for this class to be useful it needs to be complemented with a suitable class to compute the long range 
	electrostatic part The total pair force is summed for each particle when compute() is called.
    Forces are only summed between neighboring particles with a separation distance less than \c r_cut. 
	A NeighborList must be provide  to identify these neighbors. Calling compute() in this class will in turn result in 
	a call to the NeighborList's compute() to make sure that the neighbor list is up to date.

    Usage: Construct a ElectrostaticShortRange class, providing it an already constructed ParticleData and NeighborList.
    The parameter alpha splits the short range and the long range electrostatic part.

    Details on how the parameter alpha is defined are found in
    "How to mesh up Ewald sums.I. A theoretical and numerical comparison of various particle mesh routines"
    M. Deserno and C. Holm
    J. Chem. Phys. 109, 7678 (1998).

    NOTE: This class does not compute the parameter alpha, it uses the alpha as specified. If alpha is chosen too small,
    then the cut-off needs to be increased. It is therefore advisable not to use this class as is, but rather wihin a
    wrapper class that takes care of this issue.

    Forces can be computed directly by calling compute() and then retrieved with a call to acquire(), but
    a more typical usage will be to add the force compute to NVEUpdator or NVTUpdator.

    This base class defines the interface for performing the short-range part of the electrostatic force
    computations. It does provide a functional, single threaded method for computing the forces.

*/
class ElectrostaticShortRange : public ForceCompute
    {
    public:
        //! Constructs the compute
        ElectrostaticShortRange(boost::shared_ptr<SystemDefinition> sysdef,
								boost::shared_ptr<NeighborList> nlist,
								Scalar r_cut,
								Scalar alpha,
								Scalar delta,
								Scalar min_value);
        
        //! Destructor
        virtual ~ElectrostaticShortRange();
        
    protected:
        boost::shared_ptr<NeighborList> m_nlist;    //!< The neighborlist to use for the computation
        Scalar m_r_cut; //!< Cuttoff radius beyond which the force is set to 0
        Scalar m_alpha;  //!< split parameter of the short-range vs long-range forces.
        
        //! spacing of the table lookup to compute forces,
        /*!
        the look up table is build at the discrete points
        defined between 0 and r_cut+2*m_delta in intervals of m_delta, a typical
        value with modest memory use that gives very high accuraccy is m_delta=sigma/10
        but sigma/5 gives accurate enough results */
        Scalar m_delta;
        
        //! Minimum value in lookup table
        /*! In a MD the value of the force is never computed at dd=0, that is, where
        particles are in close contact. m_min_value is the minimum separation expected
        for two particles, this value is user-supplied, yet important as if for whatever
        reason this number is too small, the tables may contain errors at
        short-distance that may go unnoticed. The program does not check whether the
        values calculated satisfy this constraint. A consevative value could be sigma/4 */
        Scalar m_min_value;
        
        Scalar *f_table; //!< look up table for force
        Scalar *e_table; //!< look up table for energy
        
        //! Actually compute the forces
        virtual void computeForces(unsigned int timestep);
    };

// Exports the ElectrostaticShortRange class to python
// void export_ElectrostaticShortRange();

#endif
#endif


