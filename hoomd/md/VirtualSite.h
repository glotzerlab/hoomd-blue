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

    class VirtualSiteBase : public MolecularForceCompute{
    public:
        VirtualSiteBase(std::shared_ptr<SystemDefinition> sysdef) : MolecularForceCompute(sysdef){}

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

    template<class Mapping>
    class VirtualSite : public VirtualSiteBase{
    public:
        VirtualSite(std::shared_ptr<SystemDefinition> sysdef) : VirtualSiteBase(sysdef){}


        /// Update the position of the virtual site
        virtual void updateVirtualParticles(uint64_t timestep){
            if (m_n_molecules_global == 0)
            {
                return;
            }

            // access the particle data arrays
            ArrayHandle<Scalar4> h_postype(m_pdata->getPositions(),
                                           access_location::host,
                                           access_mode::readwrite);
            /*ArrayHandle<int3> h_image(m_pdata->getImages(),
                                      access_location::host,
                                      access_mode::readwrite);*/

            Index2D molecule_indexer = getMoleculeIndexer();
            auto nmol = molecule_indexer.getH();
            ArrayHandle<unsigned int> h_molecule_list(getMoleculeList(),
                                                      access_location::host,
                                                      access_mode::read);
            for(auto isite = 0; isite < nmol; isite++){
                // last particle of the molecule is the virtual site
                std::array<uint64_t, Mapping::n_sites> indices;
                constexpr auto siteLength = Mapping::n_sites;
                for(unsigned char s = 0; s < siteLength; s++){
                    indices[s] = h_molecule_list.data[molecule_indexer(s, isite)];
                }
                uint64_t site = h_molecule_list.data[molecule_indexer(siteLength, isite)];

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

            ArrayHandle<unsigned int> h_molecule_list(getMoleculeList(),
                                                      access_location::host,
                                                      access_mode::read);

            ArrayHandle<Scalar4> h_postype(m_pdata->getPositions(),
                                           access_location::host,
                                           access_mode::read);

            ArrayHandle<Scalar4> h_net_force(m_pdata->getNetForce(),
                                             access_location::host,
                                             access_mode::readwrite);
            ArrayHandle<Scalar> h_net_virial(m_pdata->getNetVirial(),
                                             access_location::host,
                                             access_mode::readwrite);

            ArrayHandle<Scalar4> h_force(m_force, access_location::host, access_mode::overwrite);
            ArrayHandle<Scalar> h_virial(m_virial, access_location::host, access_mode::overwrite);

            // reset constraint forces
            memset(h_force.data, 0, sizeof(Scalar4) * m_pdata->getN());
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
                constexpr auto siteLength = Mapping::n_sites;

                for(unsigned char s = 0; s < siteLength; s++){
                    indices[s] = h_molecule_list.data[molecule_indexer(s, isite)];
                }
                uint64_t site = h_molecule_list.data[molecule_indexer(siteLength, isite)];

                // only distribute the force from the vsite if its local
                if(site >= m_pdata->getN())
                    continue;
                Mapping map(m_site_parameters, indices, site);

                if(compute_virial)
                    map.decomposeVirial(h_virial.data, h_net_virial.data, m_virial_pitch, net_virial_pitch, h_postype.data, h_net_force.data);

                map.decomposeForce(h_force.data, h_net_force.data);
            }
        }


    protected:
        typename Mapping::param_type m_site_parameters;
    };

    namespace detail{
        template<class Mapping>
        void export_virtual_site(pybind11::module& m, const std::string& name){
            pybind11::class_<VirtualSite<Mapping>, VirtualSiteBase, std::shared_ptr<VirtualSite<Mapping>>>(m, name.c_str())
                    .def(pybind11::init<std::shared_ptr<SystemDefinition>>());
        }
    }
}

#endif //HOOMD_VIRTUALSITE_H
