//
// Created by girard01 on 2/9/23.
//
#include "hoomd/md/VirtualSiteGPU.h"
#include "hoomd/md/VirtualSiteMappings.h"

#define VS_CLASS hoomd::md::virtualsites::@_vstype@
#define EXPORT_FUNCTION export_virtual_site@_vstype@GPU
// clang-format on

namespace hoomd::md {

    extern template class VirtualSite<VS_CLASS>;
    namespace  detail {
    void EXPORT_FUNCTION( pybind11::module &m ) {
    export_virtual_siteGPU<VS_CLASS>(m, "VirtualSiteGPU@_vstype@");
}
}
}