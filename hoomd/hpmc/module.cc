// Copyright (c) 2009-2016 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.

// Include the defined classes that are to be exported to python
#include "IntegratorHPMC.h"
#include "IntegratorHPMCMono.h"
#include "IntegratorHPMCMonoImplicit.h"

#include "ShapeSphere.h"
#include "ShapeConvexPolygon.h"
#include "ShapePolyhedron.h"
#include "ShapeConvexPolyhedron.h"
#include "ShapeSpheropolyhedron.h"
#include "ShapeSpheropolygon.h"
#include "ShapeSimplePolygon.h"
#include "ShapeEllipsoid.h"
#include "ShapeFacetedSphere.h"
#include "ShapeSphinx.h"
#include "ShapeUnion.h"
#include "AnalyzerSDF.h"
#include "UpdaterBoxMC.h"

#include "ShapeProxy.h"

#include "GPUTree.h"

#ifdef ENABLE_CUDA
#include "IntegratorHPMCMonoGPU.h"
#endif

#include "modules.h"

/*! \file module.cc
    \brief Export classes to python
*/

// Include boost.python to do the exporting
#include <boost/python.hpp>

using namespace boost::python;
using namespace hpmc;
using namespace std;

namespace hpmc
{

//! HPMC implementation details
/*! The detail namespace contains classes and functions that are not part of the HPMC public interface. These are
    subject to change without notice and are designed solely for internal use within HPMC.
*/
namespace detail
{

// could move the make_param functions back??

}; // end namespace detail

}; // end namespace hpmc

using namespace hpmc::detail;

//! Define the _hpmc python module exports
BOOST_PYTHON_MODULE(_hpmc)
    {
    export_IntegratorHPMC();

    export_hpmc();
    export_UpdaterBoxMC();
    export_hpmc_gpu();
    export_sdf();
    export_external_fields();
    export_free_volume();
    export_shape_params();
    export_muvt();

    export_sphere();
    export_convex_polygon();
    export_simple_polygon();
    export_spheropolygon();
    export_polyhedron();
    export_ellipsoid();
    export_faceted_sphere();
    export_sphinx();

    class_<sph_params, boost::shared_ptr<sph_params> >("sph_params");
    class_<ell_params, boost::shared_ptr<ell_params> >("ell_params");
    class_<poly2d_verts, boost::shared_ptr<poly2d_verts> >("poly2d_verts");
    class_<poly3d_data, boost::shared_ptr<poly3d_data> >("poly3d_data");
    class_< poly3d_verts<8>, boost::shared_ptr< poly3d_verts<8> > >("poly3d_verts8");
    class_< poly3d_verts<16>, boost::shared_ptr< poly3d_verts<16> > >("poly3d_verts16");
    class_< poly3d_verts<32>, boost::shared_ptr< poly3d_verts<32> > >("poly3d_verts32");
    class_< poly3d_verts<64>, boost::shared_ptr< poly3d_verts<64> > >("poly3d_verts64");
    class_< poly3d_verts<128>, boost::shared_ptr< poly3d_verts<128> > >("poly3d_verts128");
    class_<ShapePolyhedron::param_type, boost::shared_ptr<ShapePolyhedron::param_type> >("poly3d_params");
    class_<faceted_sphere_params, boost::shared_ptr<faceted_sphere_params> >("faceted_sphere_params");
    class_<sphinx3d_params, boost::shared_ptr<sphinx3d_params> >("sphinx3d_params")
        .def_readwrite("circumsphereDiameter",&sphinx3d_params::circumsphereDiameter);
    class_< union_params<ShapeSphere>, boost::shared_ptr< union_params<ShapeSphere> > >("msph_params");

    def("make_poly2d_verts", &make_poly2d_verts);
    def("make_poly3d_data", &make_poly3d_data);
    def("make_poly3d_verts8", &make_poly3d_verts<8>);
    def("make_poly3d_verts16", &make_poly3d_verts<16>);
    def("make_poly3d_verts32", &make_poly3d_verts<32>);
    def("make_poly3d_verts64", &make_poly3d_verts<64>);
    def("make_poly3d_verts128", &make_poly3d_verts<128>);
    def("make_ell_params", &make_ell_params);
    def("make_sph_params", &make_sph_params);
    def("make_faceted_sphere", &make_faceted_sphere);
    def("make_sphinx3d_params", &make_sphinx3d_params);
    def("make_sphere_union_params", &make_union_params<ShapeSphere>);
    def("make_overlapreal3", &make_overlapreal3);
    def("make_overlapreal4", &make_overlapreal4);

    // export counters
    export_hpmc_implicit_counters();
    }

/*! \defgroup hpmc_integrators HPMC integrators
*/

/*! \defgroup hpmc_analyzers HPMC analyzers
*/

/*! \defgroup shape Shapes
    Shape classes define the geometry of shapes and associated overlap checks
*/

/*! \defgroup vecmath Vector Math
    Vector, matrix, and quaternion math routines
*/

/*! \defgroup hpmc_detail Details
    HPMC implementation details
    @{
*/

/*! \defgroup hpmc_data_structs Data structures
    HPMC internal data structures
*/

/*! \defgroup hpmc_kernels HPMC kernels
    HPMC GPU kernels
*/

/*! \defgroup minkowski Minkowski methods
    Classes and functions related to Minkowski overlap detection methods
*/

/*! \defgroup overlap Other overlap methods
    Classes and functions related to other (brute force) overlap methods
*/

/*! @} */
