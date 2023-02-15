//
// Created by girard01 on 2/9/23.
//
#include "VirtualSite.h"

namespace hoomd::md::detail{
    void export_virtual_site_base(pybind11::module& m){
        pybind11::class_<VirtualSiteBase, MolecularForceCompute, std::shared_ptr<VirtualSiteBase>>(m, "VirtualSiteBase")
                .def(pybind11::init<std::shared_ptr<SystemDefinition>>());
    }
}