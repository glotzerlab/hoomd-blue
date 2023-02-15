//
// Created by girard01 on 2/14/23.
//

#ifndef HOOMD_VIRTUALSITEGPU_H
#define HOOMD_VIRTUALSITEGPU_H

#include "VirtualSite.h"
#include "VirtualSiteGPU.cuh"

#include "hoomd/Autotuner.h"

namespace hoomd::md {
    template<class Mapping>
    class VirtualSiteGPU : public VirtualSite<Mapping> {
    public:
        VirtualSiteGPU(std::shared_ptr<SystemDefinition> sysdef) : VirtualSite<Mapping>(sysdef){}

        void updateVirtualParticles(uint64_t timestep) override;

        void computeForces(uint64_t timestep) override;
    protected:

    };

    template<class Mapping>
    void VirtualSiteGPU<Mapping>::updateVirtualParticles(uint64_t timestep){

    };

    template<class Mapping>
    void VirtualSiteGPU<Mapping>::computeForces(uint64_t timestep) {

    }

    namespace detail{
        template<class Mapping>
        void export_virtual_siteGPU(pybind11::module& m, std::string name){
            pybind11::class_<VirtualSiteGPU<Mapping>, VirtualSite<Mapping>, std::shared_ptr<VirtualSiteGPU<Mapping>>>(m, name.c_str())
                    .def(pybind11::init<std::shared_ptr<SystemDefinition>>());
        }
    }
}

#endif //HOOMD_VIRTUALSITEGPU_H
