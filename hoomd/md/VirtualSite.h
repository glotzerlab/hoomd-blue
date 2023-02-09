//
// Created by girard01 on 2/3/23.
//

#ifndef HOOMD_VIRTUALSITE_H
#define HOOMD_VIRTUALSITE_H

#include <array>
#include "MolecularForceCompute.h"
#include "VirtualSiteMappings.h"
#include "hoomd/managed_allocator.h"

namespace hoomd::md{

    class VirtualSite : public MolecularForceCompute{
    public:
        VirtualSite(std::shared_ptr<SystemDefinition> sysdef) : MolecularForceCompute(sysdef){}

        virtual void updateVirtualParticles(uint64_t){};

#ifdef ENABLE_MPI
        //! Get ghost particle fields requested by this pair potential
        virtual CommFlags getRequestedCommFlags(uint64_t timestep){
            CommFlags flags = CommFlags(0);
            PDataFlags pdata_flags = this->m_pdata->getFlags();
            if (pdata_flags[pdata_flag::pressure_tensor])
            {
                flags[comm_flag::net_virial] = 1;
            }
            flags |= MolecularForceCompute::getRequestedCommFlags(timestep);

            return flags;
        }
#endif


    protected:
#ifdef ENABLE_MPI
        /// The system's communicator.
        std::shared_ptr<Communicator> m_comm;
#endif
    };

    template<class Mapping = virtualsites::Type2>
    class _VirtualSite : public VirtualSite{

        _VirtualSite(std::shared_ptr<SystemDefinition> sysdef) : VirtualSite(sysdef){}


        /// Update the position of the virtual site
        virtual void updateVirtualParticles(uint64_t timestep){
            if (m_n_molecules_global == 0)
            {
                return;
            }


            // access molecule order (this needs to be on top because of ArrayHandle scope) and its
            // pervasive use across this function.
            ArrayHandle<unsigned int> h_molecule_order(getMoleculeOrder(),
                                                       access_location::host,
                                                       access_mode::read);
            ArrayHandle<unsigned int> h_molecule_len(getMoleculeLengths(),
                                                     access_location::host,
                                                     access_mode::read);
            ArrayHandle<unsigned int> h_molecule_idx(getMoleculeIndex(),
                                                     access_location::host,
                                                     access_mode::read);

            // access the particle data arrays
            ArrayHandle<Scalar4> h_postype(m_pdata->getPositions(),
                                           access_location::host,
                                           access_mode::readwrite);
            ArrayHandle<int3> h_image(m_pdata->getImages(), access_location::host, access_mode::readwrite);

            Index2D molecule_indexer = getMoleculeIndexer();
            auto nmol = molecule_indexer.getH();
            ArrayHandle<unsigned int> h_molecule_length(getMoleculeLengths(),
                                                        access_location::host,
                                                        access_mode::read);
            ArrayHandle<unsigned int> h_molecule_list(getMoleculeList(),
                                                      access_location::host,
                                                      access_mode::read);
            for(auto isite = 0; isite < nmol; isite++){
                // last particle of the molecule is the virtual site
                std::array<uint64_t, Mapping::n_sites> indices;
                auto siteLength = Mapping::n_sites;
                assert(siteLength == h_molecule_length.data[isite]);
                auto moleculeStart = &(h_molecule_list.data[isite]);
                std::copy(moleculeStart, moleculeStart + siteLength - 1, indices.data());
                uint64_t site = *(moleculeStart + siteLength);

                // only distribute the force from the vsite if its local
                if(site >= m_pdata->getN())
                    continue;
                Mapping map(m_site_parameters, indices, site);
                map.reconstructSite(h_postype.data);
            }

        }

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

            for (unsigned int isite = 0; isite < nmol; isite++) {

                // last particle of the molecule is the virtual site
                std::array<uint64_t, Mapping::n_sites> indices;
                auto siteLength = Mapping::n_sites;
                assert(siteLength == h_molecule_length.data[isite]);
                auto moleculeStart = &(h_molecule_list.data[isite]);
                std::copy(moleculeStart, moleculeStart + siteLength - 1, indices.data());
                uint64_t site = *(moleculeStart + siteLength);

                // only distribute the force from the vsite if its local
                if(site >= m_pdata->getN())
                    continue;
                Mapping map(m_site_parameters, indices, site);
                map.decomposeForce(h_net_force.data);

                if(compute_virial)
                    map.decomposeVirial(h_virial.data, m_virial_pitch); // TODO: check definitions of forces / virial / net_virial required here. Net virial on virtual site should be zero.
            }
        }


    protected:
        typename Mapping::param_type m_site_parameters;
    };

    namespace detail{

        void export_virtual_site_base(pybind11::module& m);

        template<class Mapping>
        void export_virtual_site(pybind11::module& m, const std::string& name){
            pybind11::class_<_VirtualSite<Mapping>, VirtualSite, std::shared_ptr<_VirtualSite<Mapping>>>(m, name)
                    .def(pybind11::init<std::shared_ptr<SystemDefinition>>());
        }
    }
}

#endif //HOOMD_VIRTUALSITE_H
