// Copyright (c) 2009-2024 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

// Include the defined classes that are to be exported to python
#include "IntegratorHPMC.h"
#include "IntegratorHPMCMono.h"
#include "IntegratorHPMCMonoNEC.h"

#include "ComputeSDF.h"
#include "ExternalFieldWall.h"
#include "ShapeConvexPolygon.h"
#include "ShapeConvexPolyhedron.h"
#include "ShapeEllipsoid.h"
#include "ShapeFacetedEllipsoid.h"
#include "ShapePolyhedron.h"
#include "ShapeSimplePolygon.h"
#include "ShapeSphere.h"
#include "ShapeSpheropolygon.h"
#include "ShapeSpheropolyhedron.h"
#include "ShapeSphinx.h"
#include "ShapeUnion.h"
#include "ShapeUtils.h"
#include "UpdaterBoxMC.h"
#include "UpdaterClusters.h"
#include "UpdaterMuVT.h"
#include "UpdaterQuickCompress.h"

#include "GPUTree.h"

#ifdef ENABLE_HIP
#include "IntegratorHPMCMonoGPU.h"
#endif

#include "modules.h"

namespace hoomd
    {
namespace hpmc
    {
namespace detail
    {
// Declare export methods in this file instead of in header files to avoid unecessary recompilations
// of this file.

void exportPairPotential(pybind11::module& m);

void exportPairPotentialLennardJones(pybind11::module& m);

void exportPairPotentialAngularStep(pybind11::module& m);

void exportPairPotentialStep(pybind11::module& m);

void exportPairPotentialUnion(pybind11::module& m);
    } // namespace detail
    } // namespace hpmc
    } // namespace hoomd

using namespace hoomd::hpmc;
using namespace hoomd::hpmc::detail;
using namespace std;

//! Define the _hpmc python module exports
PYBIND11_MODULE(_hpmc, m)
    {
    export_IntegratorHPMC(m);

    export_UpdaterBoxMC(m);
    export_UpdaterQuickCompress(m);
    export_wall_classes(m);
    export_wall_list(m);
    export_MassPropertiesBase(m);

    export_sphere(m);
    export_convex_polygon(m);

    export_simple_polygon(m);

    export_spheropolygon(m);

    export_polyhedron(m);

    export_ellipsoid(m);

    export_faceted_ellipsoid(m);

    export_sphinx(m);

    export_union_convex_polyhedron(m);

    export_union_faceted_ellipsoid(m);

    export_union_sphere(m);

    export_convex_polyhedron(m);

    export_convex_spheropolyhedron(m);

    pybind11::class_<SphereParams, std::shared_ptr<SphereParams>>(m, "SphereParams")
        .def(pybind11::init<pybind11::dict>())
        .def("asDict", &SphereParams::asDict);
    pybind11::class_<EllipsoidParams, std::shared_ptr<EllipsoidParams>>(m, "EllipsoidParams")
        .def(pybind11::init<pybind11::dict>())
        .def("asDict", &EllipsoidParams::asDict);
    pybind11::class_<PolygonVertices, std::shared_ptr<PolygonVertices>>(m, "PolygonVertices")
        .def(pybind11::init<pybind11::dict>())
        .def("asDict", &PolygonVertices::asDict);
    pybind11::class_<TriangleMesh, std::shared_ptr<TriangleMesh>>(m, "TriangleMesh")
        .def(pybind11::init<pybind11::dict>())
        .def("asDict", &TriangleMesh::asDict);
    pybind11::class_<PolyhedronVertices, std::shared_ptr<PolyhedronVertices>>(m,
                                                                              "PolyhedronVertices")
        .def(pybind11::init<pybind11::dict>())
        .def("asDict", &PolyhedronVertices::asDict);
    pybind11::class_<FacetedEllipsoidParams, std::shared_ptr<FacetedEllipsoidParams>>(
        m,
        "FacetedEllipsoidParams")
        .def(pybind11::init<pybind11::dict>())
        .def("asDict", &FacetedEllipsoidParams::asDict);
    pybind11::class_<SphinxParams, std::shared_ptr<SphinxParams>>(m, "SphinxParams")
        .def_readwrite("circumsphereDiameter", &SphinxParams::circumsphereDiameter)
        .def(pybind11::init<pybind11::dict>())
        .def("asDict", &SphinxParams::asDict);
    pybind11::class_<ShapeUnion<ShapeSphere>::param_type,
                     std::shared_ptr<ShapeUnion<ShapeSphere>::param_type>>(m, "SphereUnionParams")
        .def(pybind11::init<pybind11::dict>())
        .def("asDict", &ShapeUnion<ShapeSphere>::param_type::asDict);

    pybind11::class_<ShapeUnion<ShapeSpheropolyhedron>::param_type,
                     std::shared_ptr<ShapeUnion<ShapeSpheropolyhedron>::param_type>>(
        m,
        "mpoly3d_params")
        .def(pybind11::init<pybind11::dict>())
        .def("asDict", &ShapeUnion<ShapeSpheropolyhedron>::param_type::asDict);
    pybind11::class_<ShapeUnion<ShapeFacetedEllipsoid>::param_type,
                     std::shared_ptr<ShapeUnion<ShapeFacetedEllipsoid>::param_type>>(
        m,
        "mfellipsoid_params")
        .def(pybind11::init<pybind11::dict>())
        .def("asDict", &ShapeUnion<ShapeFacetedEllipsoid>::param_type::asDict);

    // export counters
    export_hpmc_implicit_counters(m);

    export_hpmc_muvt_counters(m);
    export_hpmc_clusters_counters(m);

    export_hpmc_nec_counters(m);

    exportPairPotential(m);
    exportPairPotentialLennardJones(m);
    exportPairPotentialAngularStep(m);
    exportPairPotentialStep(m);
    exportPairPotentialUnion(m);
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
