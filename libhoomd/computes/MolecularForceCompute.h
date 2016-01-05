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

#include "ForceConstraint.h"
#include "NeighborList.h"

/*! \file MolecularForceCompute.h
    \brief Base class for ForceConstraints that depend on a molecular topology

    Holds the data structures defining molecule topologies which are
    required for communication
*/

#ifdef NVCC
#error This header cannot be compiled by nvcc
#endif

#ifndef __MolecularForceCompute_H__
#define __MolecularForceCompute_H__


class MolecularForceCompute : public ForceConstraint
    {
    public:
        //! Constructs the compute
        MolecularForceCompute(boost::shared_ptr<SystemDefinition> sysdef,
            boost::shared_ptr<NeighborList> nlist);

        //! Destructor
        virtual ~MolecularForceCompute();

        //! Return the number of DOF removed by this constraint
        virtual unsigned int getNDOFRemoved() { return 0; }

        #ifdef ENABLE_MPI
        //! Get ghost particle fields requested by this pair potential
        virtual CommFlags getRequestedCommFlags(unsigned int timestep)
            {
            CommFlags flags = CommFlags(0);

            // request communication of particle forces
            flags[comm_flag::net_force] = 1;

            // request communication of tags
            flags[comm_flag::tag] = 1;

            flags |= ForceConstraint::getRequestedCommFlags(timestep);

            return flags;
            }

        //! Set the communicator object
        virtual void setCommunicator(boost::shared_ptr<Communicator> comm)
            {
            if (!m_comm && comm)
                {
                // register this class with the communciator
                m_comm_migrate_connection = comm->addMigrateRequest(
                    boost::bind(&MolecularForceCompute::askMigrateRequest, this, _1));
                // register this class with the communciator
                m_comm_ghost_layer_connection = comm->addGhostLayerWidthRequest(
                    boost::bind(&MolecularForceCompute::askGhostLayerWidth, this, _1));
                }

            // call base class method to set m_comm
            ForceConstraint::setCommunicator(comm);
            }

        //! Returns true if we need to migrate in this timestep
        virtual bool askMigrateRequest(unsigned int timestep);

        //! Returns the requested ghost layer width for all types
        /*! \param type the type for which we are requesting info
         */
        virtual Scalar askGhostLayerWidth(unsigned int type)
            {
            // save the last returned value
            m_last_d_max = m_d_max;

            return m_d_max + m_nlist->getRBuff();
            }

        #endif

    protected:
        boost::shared_ptr<NeighborList> m_nlist;    //!< Pointer to neighbor list

        GPUVector<unsigned int> m_molecule_list;    //!< 2D Array of molecule members
        GPUVector<unsigned int> m_molecule_length;  //!< List of lengths molecule lengths
        GPUVector<int> m_molecule_ridx;             //!< Per particle local molecule idx

        Index2D m_molecule_indexer;                 //!< Index of the molecule table
        boost::signals2::connection m_comm_migrate_connection; //!< Connection to be asked for migrate requests
        boost::signals2::connection m_comm_ghost_layer_connection; //!< Connection to be asked for ghost layer width requests

        Scalar m_d_max;                             //!< Current maximum molecule diameter
        Scalar m_last_d_max;                        //!< Maximum molecule diameter in last time step

        //! Fill the molecule list
        virtual void initMolecules() {};

        //! Get the maximum molecule diameter
        virtual Scalar getMaxDiameter();
    };

//! Exports the MolecularForceCompute to python
void export_MolecularForceCompute();

#endif
