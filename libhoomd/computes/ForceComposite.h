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

#include "MolecularForceCompute.h"
#include "NeighborList.h"

/*! \file ForceComposite.h
    \brief Implementation of a rigid body force compute

    Rigid body data is stored per type. Every rigid body is defined by a unique central particle of
    the rigid body type. A rigid body can only have one particle of that type.

    Nested rigid bodies are not supported, i.e. when a rigid body contains a rigid body ptl of another type.

    The particle data body tag is equal to the tag of central particle, and therefore not-contiguous.
    The molecule/body id can therefore be used to look up the central particle easily.
*/

#ifdef NVCC
#error This header cannot be compiled by nvcc
#endif

#ifndef __ForceComposite_H__
#define __ForceComposite_H__

class ForceComposite : public MolecularForceCompute
    {
    public:
        //! Constructs the compute
        ForceComposite(boost::shared_ptr<SystemDefinition> sysdef);

        //! Destructor
        virtual ~ForceComposite();

        //! Set the coordinates for the template for a rigid body of type typeid
        /*! \param body_type The type of rigid body
         * \param type Types of the constituent particles
         * \param pos Relative positions of the constituent particles
         * \param orientation Orientations of the constituent particles
         */
        virtual void setParam(unsigned int body_typeid,
            std::vector<unsigned int>& type,
            std::vector<Scalar3>& pos,
            std::vector<Scalar4>& orientation);

        //! Return the number of DOF removed by this constraint
        //virtual unsigned int getNDOFRemoved() { return m_ndof_removed; }

        //! Returns true because we compute the torque on the central particle
        virtual bool isAnisotropic()
            {
            return true;
            }

        #ifdef ENABLE_MPI
        //! Get ghost particle fields requested by this pair potential
        virtual CommFlags getRequestedCommFlags(unsigned int timestep);
        #endif

        //! Update the composite body degrees of freedom
        /*  Using the position, velocity and orientation of the central particle
         * \param update_rq If true, the positions and orientations of the central particle
         *       have been updated (otherwise velocity and angular momentum only)
         */
        virtual void updateCompositeDOFs(unsigned int timestep, bool update_rq);

        //! Create copies of rigid body constituent particles
        virtual void createRigidBodies();

    protected:
        bool m_bodies_changed;          //!< True if constituent particles have changed
        bool m_ptls_added_removed;      //!< True if particles have been added or removed

        unsigned int m_ndof_removed;    //!< Number of degrees of freedom removed

        GPUArray<unsigned int> m_body_types;    //!< Constituent ptl types per type id (2D)
        GPUArray<Scalar3> m_body_pos;           //!< Constituent ptl offsets per type id (2D)
        GPUArray<Scalar4> m_body_orientation;   //!< Constituent ptl orientations per type id (2D)
        GPUArray<unsigned int> m_body_len;      //!< Length of body per type id

        Index2D m_body_idx;                     //!< Indexer for body parameters

        std::vector<Scalar> m_d_max;                              //!< Maximum body diameter per type
        std::vector<bool> m_d_max_changed;                        //!< True if maximum body diameter changed (per type)

        //! Helper function to be called when the number of types changes
        void slotNumTypesChange();

        //! Method to be called when particles are added or removed
        void slotPtlsAddedRemoved()
            {
            m_ptls_added_removed = true;
            }

        //! Return the requested minimum ghost layer width
        virtual Scalar requestGhostLayerWidth(unsigned int type);

       //! Return the requested ghost layer width to be added to the existing ghost layer
        virtual Scalar requestGhostLayerExtraWidth(unsigned int type, Scalar r_ghost_max);

        #ifdef ENABLE_MPI
        //! Set the communicator object
        virtual void setCommunicator(boost::shared_ptr<Communicator> comm)
            {
            // call base class method to set m_comm
            MolecularForceCompute::setCommunicator(comm);

            if (!m_comm_ghost_layer_connection.connected())
                {
                // register this class with the communciator
                m_comm_ghost_layer_connection = m_comm->addGhostLayerWidthRequest(
                    boost::bind(&ForceComposite::requestGhostLayerWidth, this, _1));

                m_comm_extra_ghost_layer_connection = m_comm->addGhostLayerExtraWidthRequest(
                    boost::bind(&ForceComposite::requestGhostLayerExtraWidth, this, _1, _2));
                }
           }
        #endif

        //! Compute the forces and torques on the central particle
        virtual void computeForces(unsigned int timestep);

    private:
        //! Connection o the signal notifying when number of particle types changes
        boost::signals2::connection m_num_type_change_connection;

        //! Connection to particle data signal when particle number changes
        boost::signals2::connection m_global_ptl_num_change_connection;

        boost::signals2::connection m_comm_ghost_layer_connection; //!< Connection to be asked for ghost layer width requests
        boost::signals2::connection m_comm_extra_ghost_layer_connection; //!< Connection to be asked for extra ghost layer width requests
    };

//! Exports the ForceComposite to python
void export_ForceComposite();

#endif
