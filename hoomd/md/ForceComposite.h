// Copyright (c) 2009-2019 The Regents of the University of Michigan
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

#ifdef ENABLE_CUDA
#include "hoomd/GPUPartition.cuh"
#endif

#include <hoomd/extern/pybind/include/pybind11/pybind11.h>

#ifndef __ForceComposite_H__
#define __ForceComposite_H__

class PYBIND11_EXPORT ForceComposite : public MolecularForceCompute
    {
    public:
        //! Constructs the compute
        ForceComposite(std::shared_ptr<SystemDefinition> sysdef);

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

        //! Update the constituent particles of a composite particle using the position, velocity and orientation of the central particle
        virtual void updateCompositeParticles(unsigned int timestep);

        //! Validate or create copies of rigid body constituent particles
        /*! \param create If true, expand central particle types into rigid bodies, modifying the number of particles
         */
        virtual void validateRigidBodies(bool create=false);

    protected:
        bool m_bodies_changed;          //!< True if constituent particles have changed
        bool m_ptls_added_removed;      //!< True if particles have been added or removed

        GlobalArray<unsigned int> m_body_types;    //!< Constituent ptl types per type id (2D)
        GlobalArray<Scalar3> m_body_pos;           //!< Constituent ptl offsets per type id (2D)
        GlobalArray<Scalar4> m_body_orientation;   //!< Constituent ptl orientations per type id (2D)
        GlobalArray<unsigned int> m_body_len;      //!< Length of body per type id

        std::vector<std::vector<Scalar> > m_body_charge;      //!< Constituent ptl charges
        std::vector<std::vector<Scalar> > m_body_diameter;    //!< Constituent ptl diameters
        Index2D m_body_idx;                     //!< Indexer for body parameters

        std::vector<Scalar> m_d_max;                              //!< Maximum body diameter per constituent particle type
        std::vector<bool> m_d_max_changed;                        //!< True if maximum body diameter changed (per type)
        std::vector<Scalar> m_body_max_diameter;                  //!< List of diameters for all body types
        Scalar m_global_max_d;                                    //!< Maximum over all body diameters

        bool m_memory_initialized;                  //!< True if arrays are allocated

        //! Helper function to be called when the number of types changes
        void slotNumTypesChange();

        //! Method to be called when particles are added or removed
        void slotPtlsAddedRemoved()
            {
            m_ptls_added_removed = true;
            }

        //! Returns the maximum diameter over all rigid bodies
        Scalar getMaxBodyDiameter()
            {
            lazyInitMem();

            if (m_global_max_d_changed)
                {
                // find maximum diameter over all bodies
                Scalar d_max(0.0);
                ArrayHandle<unsigned int> h_body_len(m_body_len, access_location::host, access_mode::read);
                for (unsigned int i = 0; i < m_pdata->getNTypes(); ++i)
                    {
                    if (h_body_len.data[i] != 0 && m_body_max_diameter[i] > d_max)
                        d_max = m_body_max_diameter[i];
                    }

                // cache value
                m_global_max_d = d_max;
                m_global_max_d_changed = false;

                m_exec_conf->msg->notice(7) << "ForceComposite: Maximum body diameter is " << m_global_max_d << std::endl;
                }

            return m_global_max_d;
            }

        //! Return the requested minimum ghost layer width
        virtual Scalar requestExtraGhostLayerWidth(unsigned int type);

        #ifdef ENABLE_MPI
        //! Set the communicator object
        virtual void setCommunicator(std::shared_ptr<Communicator> comm)
            {
            // call base class method to set m_comm
            MolecularForceCompute::setCommunicator(comm);

            if (!m_comm_ghost_layer_connected)
                {
                // register this class with the communicator
                m_comm->getExtraGhostLayerWidthRequestSignal().connect<ForceComposite, &ForceComposite::requestExtraGhostLayerWidth>(this);
                m_comm_ghost_layer_connected = true;
                }
           }
        #endif

        //! Compute the forces and torques on the central particle
        virtual void computeForces(unsigned int timestep);

        //! Helper method to calculate the body diameter
        Scalar getBodyDiameter(unsigned int body_type);

        //! Initialize memory
        virtual void lazyInitMem();

    private:
        #ifdef ENABLE_MPI
        bool m_comm_ghost_layer_connected; //!< Track if we have already connected ghost layer width requests
        #endif
        bool m_global_max_d_changed;       //!< True if we updated any rigid body
    };

//! Exports the ForceComposite to python
void export_ForceComposite(pybind11::module& m);

#endif
