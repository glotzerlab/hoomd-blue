// Include the defined classes that are to be exported to python
#include "UpdaterClusters.h"

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
#include "AnalyzerSDF.h"
#include "UpdaterBoxNPT.h"
#include "ShapeUnion.h"

/*! \file module_clusters.cc
    \brief Export classes to python
*/

// Include boost.python to do the exporting
#include <boost/python.hpp>

using namespace boost::python;

namespace hpmc
{

//! Export the base HPMCMono integrators
void export_clusters()
    {
    // implicit depletant cluster moves
    export_UpdaterClusters< ShapeSphere >("UpdaterClustersSphere");
    export_UpdaterClusters< ShapeConvexPolygon >("UpdaterClustersConvexPolygon");
    export_UpdaterClusters< ShapeSimplePolygon >("UpdaterClustersSimplePolygon");
    export_UpdaterClusters< ShapeConvexPolyhedron<8> >("UpdaterClustersConvexPolyhedron8");
    export_UpdaterClusters< ShapeConvexPolyhedron<16> >("UpdaterClustersConvexPolyhedron8");
    export_UpdaterClusters< ShapeConvexPolyhedron<32> >("UpdaterClustersConvexPolyhedron16");
    export_UpdaterClusters< ShapeConvexPolyhedron<64> >("UpdaterClustersConvexPolyhedron32");
    export_UpdaterClusters< ShapeConvexPolyhedron<128> >("UpdaterClustersConvexPolyhedron64");
    export_UpdaterClusters< ShapePolyhedron >("UpdaterClustersPolyhedron");
    export_UpdaterClusters< ShapeSpheropolyhedron<8> >("UpdaterClustersSpheropolyhedron8");
    export_UpdaterClusters< ShapeSpheropolyhedron<16> >("UpdaterClustersSpheropolyhedron16");
    export_UpdaterClusters< ShapeSpheropolyhedron<32> >("UpdaterClustersSpheropolyhedron32");
    export_UpdaterClusters< ShapeSpheropolyhedron<64> >("UpdaterClustersSpheropolyhedron64");
    export_UpdaterClusters< ShapeSpheropolyhedron<128> >("UpdaterClustersSpheropolyhedron128");
    export_UpdaterClusters< ShapeEllipsoid >("UpdaterClustersEllipsoid");
    export_UpdaterClusters< ShapeSpheropolygon >("UpdaterClustersSpheropolygon");
    export_UpdaterClusters< ShapeFacetedSphere >("UpdaterClustersFacetedSphere");
    export_UpdaterClusters< ShapeSphinx >("UpdaterClustersSphinx");
    export_UpdaterClusters< ShapeUnion<ShapeSphere> >("UpdaterClustersSphereUnion");
    }
}
