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

// Maintainer: joaander

#include "ForceCompute.h"
#include "ParticleGroup.h"
#include <boost/shared_ptr.hpp>
#include "saruprng.h"
#include "HOOMDMath.h"
#include "VectorMath.h"

#include "EvaluatorConstraintEllipsoid.h"


/*! \file ActiveForceCompute.h
    \brief Declares a class for computing active forces
*/

#ifdef NVCC
#error This header cannot be compiled by nvcc
#endif

#ifndef __ACTIVEFORCECOMPUTE_H__
#define __ACTIVEFORCECOMPUTE_H__

//! Adds an active force to a number of particles
/*! \ingroup computes
*/
class ActiveForceCompute : public ForceCompute
{
    
    public:
        //! Constructs the compute
        ActiveForceCompute(boost::shared_ptr<SystemDefinition> sysdef,
                             boost::shared_ptr<ParticleGroup> group,
                             int seed, boost::python::list f_lst,
                             bool orientation_link, Scalar rotation_diff,
                             Scalar3 P,
                             Scalar rx,
                             Scalar ry,
                             Scalar rz);

        //! Destructor
        ~ActiveForceCompute();

    protected:
        //! Actually compute the forces
        virtual void computeForces(unsigned int timestep);
        
        //! Set forces for particles
        void setForces(unsigned int i);

        //! Orientational diffusion for spherical particles
        void rotationalDiffusion(unsigned int timestep, unsigned int i);

        //! Set constraints if particles confined to a surface
        void setConstraint(unsigned int i);
        
        boost::shared_ptr<ParticleGroup> m_group;   //!< Group of particles on which this force is applied
        bool m_orientationLink;
        Scalar m_rotationDiff;
        Scalar m_rotationConst;
        Scalar3 m_P;          //!< Position of the Ellipsoid
        Scalar m_rx;          //!< Radius in X direction of the Ellipsoid
        Scalar m_ry;          //!< Radius in Y direction of the Ellipsoid
        Scalar m_rz;          //!< Radius in Z direction of the Ellipsoid
        int m_seed;
        GPUArray<Scalar3> m_activeVec; //! active force unit vectors for each particle
        GPUArray<Scalar> m_activeMag; //! active force magnitude for each particle
        unsigned int last_computed;
};

//! Exports the ActiveForceComputeClass to python
void export_ActiveForceCompute();
// debug flag
#endif
