//
// Created by girard01 on 2/9/23.
//
#include "VirtualSite.h"

namespace hoomd::md::detail{
    void export_virtual_site_base(pybind11::module& m){
        pybind11::class_<VirtualSite, MolecularForceCompute, std::shared_ptr<VirtualSite>>(m, "VirtualSiteBase")
                .def(pybind11::init<std::shared_ptr<SystemDefinition>>());
    }
}