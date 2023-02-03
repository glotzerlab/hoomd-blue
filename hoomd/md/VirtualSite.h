//
// Created by girard01 on 2/3/23.
//

#ifndef HOOMD_VIRTUALSITE_H
#define HOOMD_VIRTUALSITE_H

#include "MolecularForceCompute.h"

namespace hoomd::md{
    template<class Mapping>
    class VirtualSite : public MolecularForceCompute{

        VirtualSite(std::shared_ptr<SystemDefinition> sysdef);

#ifdef ENABLE_MPI
        //! Get ghost particle fields requested by this pair potential
        virtual CommFlags getRequestedCommFlags(uint64_t timestep);
#endif

        /// Update the position of the virtual site
        virtual void updateVirtualParticles(uint64_t timestep);

        //! Distribute the force & virial from the virtual site
        virtual void computeForces(uint64_t timestep){
            if (m_n_molecules_global == 0)
            {
                return;
            }

            Index2D molecule_indexer = getMoleculeIndexer();
            unsigned int nmol = molecule_indexer.getH();

            ArrayHandle<unsigned int> h_molecule_length(getMoleculeLengths(),
                                                        access_location::host,
                                                        access_mode::read);
            ArrayHandle<unsigned int> h_molecule_list(getMoleculeList(),
                                                      access_location::host,
                                                      access_mode::read);

            // access particle data
            ArrayHandle<unsigned int> h_rtag(m_pdata->getRTags(), access_location::host, access_mode::read);
            ArrayHandle<unsigned int> h_tag(m_pdata->getTags(), access_location::host, access_mode::read);
            ArrayHandle<Scalar4> h_postype(m_pdata->getPositions(),
                                           access_location::host,
                                           access_mode::read);

            ArrayHandle<Scalar4> h_net_force(m_pdata->getNetForce(),
                                             access_location::host,
                                             access_mode::readwrite);
            ArrayHandle<Scalar4> h_net_torque(m_pdata->getNetTorqueArray(),
                                              access_location::host,
                                              access_mode::readwrite);
            ArrayHandle<Scalar> h_net_virial(m_pdata->getNetVirial(),
                                             access_location::host,
                                             access_mode::readwrite);

            ArrayHandle<Scalar4> h_force(m_force, access_location::host, access_mode::overwrite);
            ArrayHandle<Scalar4> h_torque(m_torque, access_location::host, access_mode::overwrite);
            ArrayHandle<Scalar> h_virial(m_virial, access_location::host, access_mode::overwrite);

            // reset constraint forces and torques
            memset(h_force.data, 0, sizeof(Scalar4) * m_pdata->getN());
            memset(h_torque.data, 0, sizeof(Scalar4) * m_pdata->getN());
            memset(h_virial.data, 0, sizeof(Scalar) * m_virial.getNumElements());

            unsigned int n_particles_local = m_pdata->getN() + m_pdata->getNGhosts();
            size_t net_virial_pitch = m_pdata->getNetVirial().getPitch();

            PDataFlags flags = m_pdata->getFlags();
            bool compute_virial = false;
            if (flags[pdata_flag::pressure_tensor])
            {
                compute_virial = true;
            }



        }

    protected:
#ifdef ENABLE_MPI
        /// The system's communicator.
        std::shared_ptr<Communicator> m_comm;
#endif

    };

}

#endif //HOOMD_VIRTUALSITE_H
