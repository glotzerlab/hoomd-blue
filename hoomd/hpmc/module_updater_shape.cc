// Copyright (c) 2009-2016 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.

// Include the defined classes that are to be exported to python

#include <hoomd/extern/pybind/include/pybind11/pybind11.h>
#include <hoomd/extern/pybind/include/pybind11/stl.h>

#include "ShapeSphere.h"
#include "ShapeConvexPolygon.h"
#include "ShapeSpheropolygon.h"
#include "ShapePolyhedron.h"
#include "ShapeConvexPolyhedron.h"
#include "ShapeSpheropolyhedron.h"
#include "ShapeSimplePolygon.h"
#include "ShapeEllipsoid.h"
#include "ShapeFacetedSphere.h"
#include "ShapeSphinx.h"
#include "ShapeUnion.h"

#include "UpdaterShape.h"
#include "ShapeMoves.h"

namespace hpmc
{

template<class Shape>
void export_ShapeMoveInterface(pybind11::module& m, const std::string& name)
    {
    pybind11::class_<   shape_move_function<Shape, Saru>,
                        std::shared_ptr< shape_move_function<Shape, Saru> >,
                        shape_move_function_wrap<Shape, Saru> >
    (m, (name + "Interface").c_str())
    .def(pybind11::init< unsigned int >())
    ;
    }

template<class Shape>
void export_ScaleShearShapeMove(pybind11::module& m, const std::string& name)
    {
    pybind11::class_< elastic_shape_move_function<Shape, Saru>, std::shared_ptr< elastic_shape_move_function<Shape, Saru> > >
    (m, name.c_str(), pybind11::base< shape_move_function<Shape, Saru> >())
    .def(pybind11::init< unsigned int, const Scalar&, Scalar>())
    ;
    }

template< typename Shape >
void export_ShapeLogBoltzmann(pybind11::module& m, const std::string& name)
    {
    pybind11::class_< ShapeLogBoltzmannFunction<Shape>, std::shared_ptr< ShapeLogBoltzmannFunction<Shape> > >
    (m, (name + "Interface").c_str() )
    .def(pybind11::init< >())
    ;
    }

template<class Shape>
void export_ShapeSpringLogBoltzmannFunction(pybind11::module& m, const std::string& name)
    {
    pybind11::class_< ShapeSpring<Shape>, std::shared_ptr< ShapeSpring<Shape> > >
    (m, name.c_str(), pybind11::base< ShapeLogBoltzmannFunction<Shape> >())
    .def(pybind11::init< Scalar, typename Shape::param_type >())
    ;
    }

template<class Shape>
void export_AlchemyLogBoltzmannFunction(pybind11::module& m, const std::string& name)
    {
    pybind11::class_< AlchemyLogBoltzmannFunction<Shape>, std::shared_ptr< AlchemyLogBoltzmannFunction<Shape> > >
    (m, name.c_str(), pybind11::base< ShapeLogBoltzmannFunction<Shape> >())
    .def(pybind11::init< >())
    ;
    }

template<class Shape>
void export_ConvexPolyhedronGeneralizedShapeMove(pybind11::module& m, const std::string& name)
    {
    pybind11::class_< convex_polyhedron_generalized_shape_move<Shape, Saru>, std::shared_ptr< convex_polyhedron_generalized_shape_move<Shape, Saru> > >
    (m, name.c_str(), pybind11::base< shape_move_function<Shape, Saru> >())
    .def(pybind11::init< unsigned int, Scalar, Scalar, Scalar >())
    ;
    }

template<class Shape>
void export_PythonShapeMove(pybind11::module& m, const std::string& name)
    {
    pybind11::class_< python_callback_parameter_shape_move<Shape, Saru>, std::shared_ptr< python_callback_parameter_shape_move<Shape, Saru> > >
    (m, name.c_str(), pybind11::base< shape_move_function<Shape, Saru> >())
    .def(pybind11::init<unsigned int,
                        pybind11::object,
                        std::vector< std::vector<Scalar> >,
                        std::vector<Scalar>,
                        Scalar >())
    ;
    }
template<class Shape>
void export_ConstantShapeMove(pybind11::module& m, const std::string& name)
    {
    pybind11::class_< constant_shape_move<Shape, Saru>, std::shared_ptr< constant_shape_move<Shape, Saru> > >
    (m, name.c_str(), pybind11::base< shape_move_function<Shape, Saru> >())
    .def(pybind11::init<unsigned int,
                        const std::vector< typename Shape::param_type >& >())
    ;
    }


template void export_UpdaterShape< ShapeSphere >(pybind11::module& m, const std::string& name);
template void export_ShapeMoveInterface< ShapeSphere >(pybind11::module& m, const std::string& name);
template void export_ShapeLogBoltzmann< ShapeSphere >(pybind11::module& m, const std::string& name);
template void export_AlchemyLogBoltzmannFunction< ShapeSphere >(pybind11::module& m, const std::string& name);
template void export_PythonShapeMove< ShapeSphere >(pybind11::module& m, const std::string& name);
template void export_ConstantShapeMove< ShapeSphere >(pybind11::module& m, const std::string& name);

template void export_ShapeMoveInterface< ShapeEllipsoid >(pybind11::module& m, const std::string& name);
template void export_ShapeLogBoltzmann< ShapeEllipsoid >(pybind11::module& m, const std::string& name);
template void export_ScaleShearShapeMove< ShapeEllipsoid >(pybind11::module& m, const std::string& name);
template void export_ShapeSpringLogBoltzmannFunction< ShapeEllipsoid >(pybind11::module& m, const std::string& name);
template void export_AlchemyLogBoltzmannFunction< ShapeEllipsoid >(pybind11::module& m, const std::string& name);
template void export_UpdaterShape< ShapeEllipsoid >(pybind11::module& m, const std::string& name);
template void export_PythonShapeMove< ShapeEllipsoid >(pybind11::module& m, const std::string& name);
template void export_ConstantShapeMove< ShapeEllipsoid >(pybind11::module& m, const std::string& name);

template void export_ShapeMoveInterface< ShapeConvexPolygon >(pybind11::module& m, const std::string& name);
template void export_ShapeLogBoltzmann< ShapeConvexPolygon >(pybind11::module& m, const std::string& name);
template void export_AlchemyLogBoltzmannFunction< ShapeConvexPolygon >(pybind11::module& m, const std::string& name);
template void export_UpdaterShape< ShapeConvexPolygon >(pybind11::module& m, const std::string& name);
template void export_PythonShapeMove< ShapeConvexPolygon >(pybind11::module& m, const std::string& name);
template void export_ConstantShapeMove< ShapeConvexPolygon >(pybind11::module& m, const std::string& name);

template void export_ShapeMoveInterface< ShapeSimplePolygon >(pybind11::module& m, const std::string& name);
template void export_ShapeLogBoltzmann< ShapeSimplePolygon >(pybind11::module& m, const std::string& name);
template void export_AlchemyLogBoltzmannFunction< ShapeSimplePolygon >(pybind11::module& m, const std::string& name);
template void export_UpdaterShape< ShapeSimplePolygon >(pybind11::module& m, const std::string& name);
template void export_PythonShapeMove< ShapeSimplePolygon >(pybind11::module& m, const std::string& name);
template void export_ConstantShapeMove< ShapeSimplePolygon >(pybind11::module& m, const std::string& name);

template void export_ShapeMoveInterface< ShapeSpheropolygon >(pybind11::module& m, const std::string& name);
template void export_ShapeLogBoltzmann< ShapeSpheropolygon >(pybind11::module& m, const std::string& name);
template void export_AlchemyLogBoltzmannFunction< ShapeSpheropolygon >(pybind11::module& m, const std::string& name);
template void export_UpdaterShape< ShapeSpheropolygon >(pybind11::module& m, const std::string& name);
template void export_PythonShapeMove< ShapeSpheropolygon >(pybind11::module& m, const std::string& name);
template void export_ConstantShapeMove< ShapeSpheropolygon >(pybind11::module& m, const std::string& name);

template void export_ShapeMoveInterface< ShapePolyhedron >(pybind11::module& m, const std::string& name);
template void export_ShapeLogBoltzmann< ShapePolyhedron >(pybind11::module& m, const std::string& name);
template void export_AlchemyLogBoltzmannFunction< ShapePolyhedron >(pybind11::module& m, const std::string& name);
template void export_UpdaterShape< ShapePolyhedron >(pybind11::module& m, const std::string& name);
template void export_PythonShapeMove< ShapePolyhedron >(pybind11::module& m, const std::string& name);
template void export_ConstantShapeMove< ShapePolyhedron >(pybind11::module& m, const std::string& name);

template void export_ShapeMoveInterface< ShapeConvexPolyhedron<8> >(pybind11::module& m, const std::string& name);
template void export_ShapeLogBoltzmann< ShapeConvexPolyhedron<8> >(pybind11::module& m, const std::string& name);
template void export_ScaleShearShapeMove< ShapeConvexPolyhedron<8> >(pybind11::module& m, const std::string& name);
template void export_ShapeSpringLogBoltzmannFunction< ShapeConvexPolyhedron<8> >(pybind11::module& m, const std::string& name);
template void export_AlchemyLogBoltzmannFunction< ShapeConvexPolyhedron<8> >(pybind11::module& m, const std::string& name);
template void export_ConvexPolyhedronGeneralizedShapeMove< ShapeConvexPolyhedron<8> >(pybind11::module& m, const std::string& name);
template void export_UpdaterShape< ShapeConvexPolyhedron<8> >(pybind11::module& m, const std::string& name);
template void export_PythonShapeMove< ShapeConvexPolyhedron<8> >(pybind11::module& m, const std::string& name);
template void export_ConstantShapeMove< ShapeConvexPolyhedron<8> >(pybind11::module& m, const std::string& name);

template void export_ShapeMoveInterface< ShapeConvexPolyhedron<16> >(pybind11::module& m, const std::string& name);
template void export_ShapeLogBoltzmann< ShapeConvexPolyhedron<16> >(pybind11::module& m, const std::string& name);
template void export_ScaleShearShapeMove< ShapeConvexPolyhedron<16> >(pybind11::module& m, const std::string& name);
template void export_ShapeSpringLogBoltzmannFunction< ShapeConvexPolyhedron<16> >(pybind11::module& m, const std::string& name);
template void export_AlchemyLogBoltzmannFunction< ShapeConvexPolyhedron<16> >(pybind11::module& m, const std::string& name);
template void export_ConvexPolyhedronGeneralizedShapeMove< ShapeConvexPolyhedron<16> >(pybind11::module& m, const std::string& name);
template void export_UpdaterShape< ShapeConvexPolyhedron<16> >(pybind11::module& m, const std::string& name);
template void export_PythonShapeMove< ShapeConvexPolyhedron<16> >(pybind11::module& m, const std::string& name);
template void export_ConstantShapeMove< ShapeConvexPolyhedron<16> >(pybind11::module& m, const std::string& name);

template void export_ShapeMoveInterface< ShapeConvexPolyhedron<32> >(pybind11::module& m, const std::string& name);
template void export_ShapeLogBoltzmann< ShapeConvexPolyhedron<32> >(pybind11::module& m, const std::string& name);
template void export_ScaleShearShapeMove< ShapeConvexPolyhedron<32> >(pybind11::module& m, const std::string& name);
template void export_ShapeSpringLogBoltzmannFunction< ShapeConvexPolyhedron<32> >(pybind11::module& m, const std::string& name);
template void export_AlchemyLogBoltzmannFunction< ShapeConvexPolyhedron<32> >(pybind11::module& m, const std::string& name);
template void export_ConvexPolyhedronGeneralizedShapeMove< ShapeConvexPolyhedron<32> >(pybind11::module& m, const std::string& name);
template void export_UpdaterShape< ShapeConvexPolyhedron<32> >(pybind11::module& m, const std::string& name);
template void export_PythonShapeMove< ShapeConvexPolyhedron<32> >(pybind11::module& m, const std::string& name);
template void export_ConstantShapeMove< ShapeConvexPolyhedron<32> >(pybind11::module& m, const std::string& name);

template void export_ShapeMoveInterface< ShapeConvexPolyhedron<64> >(pybind11::module& m, const std::string& name);
template void export_ShapeLogBoltzmann< ShapeConvexPolyhedron<64> >(pybind11::module& m, const std::string& name);
template void export_ScaleShearShapeMove< ShapeConvexPolyhedron<64> >(pybind11::module& m, const std::string& name);
template void export_ShapeSpringLogBoltzmannFunction< ShapeConvexPolyhedron<64> >(pybind11::module& m, const std::string& name);
template void export_AlchemyLogBoltzmannFunction< ShapeConvexPolyhedron<64> >(pybind11::module& m, const std::string& name);
template void export_ConvexPolyhedronGeneralizedShapeMove< ShapeConvexPolyhedron<64> >(pybind11::module& m, const std::string& name);
template void export_UpdaterShape< ShapeConvexPolyhedron<64> >(pybind11::module& m, const std::string& name);
template void export_PythonShapeMove< ShapeConvexPolyhedron<64> >(pybind11::module& m, const std::string& name);
template void export_ConstantShapeMove< ShapeConvexPolyhedron<64> >(pybind11::module& m, const std::string& name);

template void export_ShapeMoveInterface< ShapeConvexPolyhedron<128> >(pybind11::module& m, const std::string& name);
template void export_ShapeLogBoltzmann< ShapeConvexPolyhedron<128> >(pybind11::module& m, const std::string& name);
template void export_ScaleShearShapeMove< ShapeConvexPolyhedron<128> >(pybind11::module& m, const std::string& name);
template void export_ShapeSpringLogBoltzmannFunction< ShapeConvexPolyhedron<128> >(pybind11::module& m, const std::string& name);
template void export_AlchemyLogBoltzmannFunction< ShapeConvexPolyhedron<128> >(pybind11::module& m, const std::string& name);
template void export_ConvexPolyhedronGeneralizedShapeMove< ShapeConvexPolyhedron<128> >(pybind11::module& m, const std::string& name);
template void export_UpdaterShape< ShapeConvexPolyhedron<128> >(pybind11::module& m, const std::string& name);
template void export_PythonShapeMove< ShapeConvexPolyhedron<128> >(pybind11::module& m, const std::string& name);
template void export_ConstantShapeMove< ShapeConvexPolyhedron<128> >(pybind11::module& m, const std::string& name);

template void export_ShapeMoveInterface< ShapeSpheropolyhedron<8> >(pybind11::module& m, const std::string& name);
template void export_ShapeLogBoltzmann< ShapeSpheropolyhedron<8> >(pybind11::module& m, const std::string& name);
template void export_AlchemyLogBoltzmannFunction< ShapeSpheropolyhedron<8> >(pybind11::module& m, const std::string& name);
template void export_UpdaterShape< ShapeSpheropolyhedron<8> >(pybind11::module& m, const std::string& name);
template void export_PythonShapeMove< ShapeSpheropolyhedron<8> >(pybind11::module& m, const std::string& name);
template void export_ConstantShapeMove< ShapeSpheropolyhedron<8> >(pybind11::module& m, const std::string& name);

template void export_ShapeMoveInterface< ShapeSpheropolyhedron<16> >(pybind11::module& m, const std::string& name);
template void export_ShapeLogBoltzmann< ShapeSpheropolyhedron<16> >(pybind11::module& m, const std::string& name);
template void export_AlchemyLogBoltzmannFunction< ShapeSpheropolyhedron<16> >(pybind11::module& m, const std::string& name);
template void export_UpdaterShape< ShapeSpheropolyhedron<16> >(pybind11::module& m, const std::string& name);
template void export_PythonShapeMove< ShapeSpheropolyhedron<16> >(pybind11::module& m, const std::string& name);
template void export_ConstantShapeMove< ShapeSpheropolyhedron<16> >(pybind11::module& m, const std::string& name);

template void export_ShapeMoveInterface< ShapeSpheropolyhedron<32> >(pybind11::module& m, const std::string& name);
template void export_ShapeLogBoltzmann< ShapeSpheropolyhedron<32> >(pybind11::module& m, const std::string& name);
template void export_AlchemyLogBoltzmannFunction< ShapeSpheropolyhedron<32> >(pybind11::module& m, const std::string& name);
template void export_UpdaterShape< ShapeSpheropolyhedron<32> >(pybind11::module& m, const std::string& name);
template void export_PythonShapeMove< ShapeSpheropolyhedron<32> >(pybind11::module& m, const std::string& name);
template void export_ConstantShapeMove< ShapeSpheropolyhedron<32> >(pybind11::module& m, const std::string& name);

template void export_ShapeMoveInterface< ShapeSpheropolyhedron<64> >(pybind11::module& m, const std::string& name);
template void export_ShapeLogBoltzmann< ShapeSpheropolyhedron<64> >(pybind11::module& m, const std::string& name);
template void export_AlchemyLogBoltzmannFunction< ShapeSpheropolyhedron<64> >(pybind11::module& m, const std::string& name);
template void export_UpdaterShape< ShapeSpheropolyhedron<64> >(pybind11::module& m, const std::string& name);
template void export_PythonShapeMove< ShapeSpheropolyhedron<64> >(pybind11::module& m, const std::string& name);
template void export_ConstantShapeMove< ShapeSpheropolyhedron<64> >(pybind11::module& m, const std::string& name);

template void export_ShapeMoveInterface< ShapeSpheropolyhedron<128> >(pybind11::module& m, const std::string& name);
template void export_ShapeLogBoltzmann< ShapeSpheropolyhedron<128> >(pybind11::module& m, const std::string& name);
template void export_AlchemyLogBoltzmannFunction< ShapeSpheropolyhedron<128> >(pybind11::module& m, const std::string& name);
template void export_UpdaterShape< ShapeSpheropolyhedron<128> >(pybind11::module& m, const std::string& name);
template void export_PythonShapeMove< ShapeSpheropolyhedron<128> >(pybind11::module& m, const std::string& name);
template void export_ConstantShapeMove< ShapeSpheropolyhedron<128> >(pybind11::module& m, const std::string& name);

template void export_ShapeMoveInterface< ShapeFacetedSphere >(pybind11::module& m, const std::string& name);
template void export_ShapeLogBoltzmann< ShapeFacetedSphere >(pybind11::module& m, const std::string& name);
template void export_AlchemyLogBoltzmannFunction< ShapeFacetedSphere >(pybind11::module& m, const std::string& name);
template void export_UpdaterShape< ShapeFacetedSphere >(pybind11::module& m, const std::string& name);
template void export_PythonShapeMove< ShapeFacetedSphere >(pybind11::module& m, const std::string& name);
template void export_ConstantShapeMove< ShapeFacetedSphere >(pybind11::module& m, const std::string& name);

template void export_ShapeMoveInterface< ShapeSphinx >(pybind11::module& m, const std::string& name);
template void export_ShapeLogBoltzmann< ShapeSphinx >(pybind11::module& m, const std::string& name);
template void export_AlchemyLogBoltzmannFunction< ShapeSphinx >(pybind11::module& m, const std::string& name);
template void export_UpdaterShape< ShapeSphinx >(pybind11::module& m, const std::string& name);
template void export_PythonShapeMove< ShapeSphinx >(pybind11::module& m, const std::string& name);
template void export_ConstantShapeMove< ShapeSphinx >(pybind11::module& m, const std::string& name);

template void export_ShapeMoveInterface< ShapeUnion<ShapeSphere> >(pybind11::module& m, const std::string& name);
template void export_ShapeLogBoltzmann< ShapeUnion<ShapeSphere> >(pybind11::module& m, const std::string& name);
template void export_AlchemyLogBoltzmannFunction< ShapeUnion<ShapeSphere> >(pybind11::module& m, const std::string& name);
template void export_UpdaterShape< ShapeUnion<ShapeSphere> >(pybind11::module& m, const std::string& name);
template void export_PythonShapeMove< ShapeUnion<ShapeSphere> >(pybind11::module& m, const std::string& name);
template void export_ConstantShapeMove< ShapeUnion<ShapeSphere> >(pybind11::module& m, const std::string& name);

}
