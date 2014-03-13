/*
Highly Optimized Object-oriented Many-particle Dynamics -- Blue Edition
(HOOMD-blue) Open Source Software License Copyright 2009-2014 The Regents of
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


// Maintainer: morozov

#include <boost/shared_ptr.hpp>

#include "ForceCompute.h"
#include "NeighborList.h"

/*! \file EAMForceCompute.h
    \brief Declares the EAMForceCompute class
*/

#ifdef NVCC
#error This header cannot be compiled by nvcc
#endif

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

        //! Destructor
        virtual ~EAMForceCompute();

        //! Sets the neighbor list to be used for the EAM force
        virtual void set_neighbor_list(boost::shared_ptr<NeighborList> nlist);

        //! Get the r cut value read from the EAM potential file
        virtual Scalar get_r_cut();

        //! Returns a list of log quantities this compute calculates
        virtual std::vector< std::string > getProvidedLogQuantities();

        //! Calculates the requested log value and returns it
        virtual Scalar getLogValue(const std::string& quantity, unsigned int timestep);

        //! Shifting modes that can be applied to the energy
        virtual void loadFile(char *filename, int type_of_file);


    protected:
        boost::shared_ptr<NeighborList> m_nlist;       //!< The neighborlist to use for the computation
        Scalar m_r_cut;                                //!< Cuttoff radius beyond which the force is set to 0
        unsigned int m_ntypes;                         //!< Store the width and height of lj1 and lj2 here

        Scalar drho;                                   //!< Undocumented parameter
        Scalar dr;                                     //!< Undocumented parameter
        Scalar rdrho;                                  //!< Undocumented parameter
        Scalar rdr;                                    //!< Undocumented parameter
        vector<Scalar> mass;                           //!< Undocumented parameter
        vector<int> types;                             //!< Undocumented parameter
        vector<string> names;                          //!< Undocumented parameter
        unsigned int nr;                               //!< Undocumented parameter
        unsigned int nrho;                             //!< Undocumented parameter


        vector<Scalar> electronDensity;                //!< array rho(r)
        vector<Scalar2> pairPotential;                  //!< array Z(r)
        vector<Scalar> embeddingFunction;              //!< array F(rho)

        vector<Scalar> derivativeElectronDensity;      //!< array rho'(r)
        vector<Scalar> derivativePairPotential;        //!< array Z'(r)
        vector<Scalar> derivativeEmbeddingFunction;    //!< array F'(rho)

        //! Actually compute the forces
        virtual void computeForces(unsigned int timestep);
    };

//! Exports the EAMForceCompute class to python
void export_EAMForceCompute();

#endif
