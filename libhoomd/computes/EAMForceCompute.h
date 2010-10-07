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

// $Id$
// $URL$
// Maintainer: morozov

#include <boost/shared_ptr.hpp>

#include "ForceCompute.h"
#include "NeighborList.h"

/*! \file EAMForceCompute.h
    \brief Declares the EAMForceCompute class
*/

#ifndef __EAMFORCECOMPUTE_H__
#define __EAMFORCECOMPUTE_H__

//! Computes Lennard-Jones forces on each particle
/*! The total pair force is summed for each particle when compute() is called. Forces are only summed between
    neighboring particles with a separation distance less than \c r_cut. A NeighborList must be provided
    to identify these neighbors. Calling compute() in this class will in turn result in a call to the
    NeighborList's compute() to make sure that the neighbor list is up to date.

    Usage: Construct a EAMForceCompute, providing it an already constructed ParticleData and NeighborList.
    Then set parameters for all possible pairs of types by calling setParams.

    Forces can be computed directly by calling compute() and then retrieved with a call to acquire(), but
    a more typical usage will be to add the force compute to NVEUpdater or NVTUpdater.

    \ingroup computes
*/
class EAMForceCompute : public ForceCompute
    {
    public:
        //! Constructs the compute
        EAMForceCompute(boost::shared_ptr<SystemDefinition> sysdef,  char *filename, int type_of_file);
        virtual void set_neighbor_list(boost::shared_ptr<NeighborList> nlist);
        virtual Scalar get_r_cut();
        //! Destructor
        virtual ~EAMForceCompute();

        //! Returns a list of log quantities this compute calculates
        virtual std::vector< std::string > getProvidedLogQuantities();

        //! Calculates the requested log value and returns it
        virtual Scalar getLogValue(const std::string& quantity, unsigned int timestep);

        //! Shifting modes that can be applied to the energy
        virtual void loadFile(char *filename, int type_of_file);


    protected:
        boost::shared_ptr<NeighborList> m_nlist;    //!< The neighborlist to use for the computation
        Scalar m_r_cut;                                //!< Cuttoff radius beyond which the force is set to 0
        unsigned int m_ntypes;                        //!< Store the width and height of lj1 and lj2 here

        Scalar drho;
        Scalar dr;
        Scalar rdrho;
        Scalar rdr;
        vector<Scalar> mass;
        vector<int> types;
        vector<string> names;
        unsigned int nr;
        unsigned int nrho;


        vector<Scalar> electronDensity; //array rho(r)
        vector<float2> pairPotential; //array Z(r)
        vector<Scalar> embeddingFunction; //array F(rho)

        vector<Scalar> derivativeElectronDensity; //array rho'(r)
        vector<Scalar> derivativePairPotential; //array Z'(r)
        vector<Scalar> derivativeEmbeddingFunction; //array F'(rho)
        //! Actually compute the forces
        virtual void computeForces(unsigned int timestep);
    };

//! Exports the EAMForceCompute class to python
void export_EAMForceCompute();

#endif

