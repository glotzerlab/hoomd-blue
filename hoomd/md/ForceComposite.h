// Copyright (c) 2009-2016 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.


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
            std::vector<Scalar4>& orientation,
            std::vector<Scalar>& charge,
            std::vector<Scalar>& diameter);

        //! Returns true because we compute the torque on the central particle
        virtual bool isAnisotropic()
            {
            return true;
            }

        #ifdef ENABLE_MPI
        //! Get ghost particle fields requested by this pair potential
        virtual CommFlags getRequestedCommFlags(unsigned int timestep);
        #endif

        //! Update the constituent particles of a composite particle
        /*  Using the position, velocity and orientation of the central particle
         */
        virtual void updateCompositeParticles(unsigned int timestep);

        //! Validate or create copies of rigid body constituent particles
        /*! \param create If true, expand central particle types into rigid bodies, modifying the number of particles
         */
        virtual void validateRigidBodies(bool create=false);

    protected:
        bool m_bodies_changed;          //!< True if constituent particles have changed
        bool m_ptls_added_removed;      //!< True if particles have been added or removed

        GPUArray<unsigned int> m_body_types;    //!< Constituent ptl types per type id (2D)
        GPUArray<Scalar3> m_body_pos;           //!< Constituent ptl offsets per type id (2D)
        GPUArray<Scalar4> m_body_orientation;   //!< Constituent ptl orientations per type id (2D)
        GPUArray<unsigned int> m_body_len;      //!< Length of body per type id

        std::vector<std::vector<Scalar> > m_body_charge;      //!< Constituent ptl charges
        std::vector<std::vector<Scalar> > m_body_diameter;    //!< Constituent ptl diameters0
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
    };

//! Exports the ForceComposite to python
void export_ForceComposite();

#endif
