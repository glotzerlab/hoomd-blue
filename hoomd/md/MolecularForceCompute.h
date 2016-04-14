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

// Maintainer: jglaser

#include "hoomd/ForceConstraint.h"
#include "NeighborList.h"

/*! \file MolecularForceCompute.h
    \brief Base class for ForceConstraints that depend on a molecular topology

    Implements the data structures that define a molecule topology.
    MolecularForceCompute maintains a list of local molecules and their constituent particles, and
    the particles are sorted according to global particle tag.

    The data structures are initialized by calling initMolecules(). This is done in the derived class
    whenever particles are reordered.

    Every molecule has a unique contiguous tag, 0 <=tag <m_n_molecules_global.

    Derived classes take care of resizing the ghost layer accordingly so that
    spanning molecules are communicated correctly. They connect to the Communciator
    signal using addGhostLayerWidthRequest() .

    In MPI simulations, MolecularForceCompute ensures that local molecules are complete by requesting communication of all
    members of a molecule even if only a single particle member falls into the ghost layer.
*/

#ifdef NVCC
#error This header cannot be compiled by nvcc
#endif

#ifndef __MolecularForceCompute_H__
#define __MolecularForceCompute_H__

const unsigned int NO_MOLECULE = (unsigned int)0xffffffff;

class MolecularForceCompute : public ForceConstraint
    {
    public:
        //! Constructs the compute
        MolecularForceCompute(boost::shared_ptr<SystemDefinition> sysdef);

        //! Destructor
        virtual ~MolecularForceCompute();

        //! Return the number of DOF removed by this constraint
        virtual unsigned int getNDOFRemoved() { return 0; }

        #ifdef ENABLE_MPI
        //! Get ghost particle fields requested by this pair potential
        virtual CommFlags getRequestedCommFlags(unsigned int timestep)
            {
            CommFlags flags = CommFlags(0);

            // request communication of tags
            flags[comm_flag::tag] = 1;

            flags |= ForceConstraint::getRequestedCommFlags(timestep);

            return flags;
            }
        #endif

        //! Return molecule index
        const Index2D& getMoleculeIndexer()
            {
            checkParticlesSorted();

            return m_molecule_indexer;
            }

        //! Return molecule list
        const GPUVector<unsigned int>& getMoleculeList()
            {
            checkParticlesSorted();

            return m_molecule_list;
            }

        //! Return molecule lengths
        const GPUVector<unsigned int>& getMoleculeLengths()
            {
            checkParticlesSorted();

            return m_molecule_length;
            }

        //! Return molecule order
        const GPUVector<unsigned int>& getMoleculeOrder()
            {
            checkParticlesSorted();

            return m_molecule_order;
            }

    protected:
        GPUVector<unsigned int> m_molecule_tag;     //!< Molecule tag per particle tag
        unsigned int m_n_molecules_global;          //!< Global number of molecules

    private:
        GPUVector<unsigned int> m_molecule_list;    //!< 2D Array of molecule members
        GPUVector<unsigned int> m_molecule_length;  //!< List of molecule lengths
        GPUVector<unsigned int> m_molecule_order;   //!< Order in molecule by local ptl idx

        Index2D m_molecule_indexer;                 //!< Index of the molecule table

        //! construct a list of local molecules
        virtual void initMolecules();

        //! Helper function to check if particles have been sorted and rebuild indices if necessary
        void checkParticlesSorted()
            {
            if (m_particles_sorted)
                {
                // rebuild molecule list
                initMolecules();
                m_particles_sorted = false;
                }
            }


    };

//! Exports the MolecularForceCompute to python
void export_MolecularForceCompute();

#endif
