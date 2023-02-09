//
// Created by girard01 on 2/9/23.
//
#include "VirtualSite.h"
#include "VirtualSiteMappings.h"

#define VS_CLASS @_vstype@
#define EXPORT_FUNCTION export_virtual_site@_vstype@
// clang-format on

namespace hoomd::md::detail{
    void EXPORT_FUNCTION(pybind11::module& m){
        export_virtual_site<VS_CLASS>(m, "VirtualSite@_vstype@");
    }
}