// Copyright (c) 2009-2016 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.

#ifndef __MODULES__
#define __MODULES__

#ifndef NVCC
#include <hoomd/extern/pybind/include/pybind11/pybind11.h>
#endif

namespace hpmc
{

void export_sphere(pybind11::module& m);
void export_convex_polygon(pybind11::module& m);
void export_simple_polygon(pybind11::module& m);
void export_spheropolygon(pybind11::module& m);
void export_polyhedron(pybind11::module& m);
void export_ellipsoid(pybind11::module& m);
void export_faceted_sphere(pybind11::module& m);
void export_sphinx(pybind11::module& m);
void export_union_sphere8(pybind11::module& m);
void export_union_sphere16(pybind11::module& m);
void export_union_sphere32(pybind11::module& m);
void export_union_sphere64(pybind11::module& m);
void export_union_sphere128(pybind11::module& m);
void export_union_sphere256(pybind11::module& m);
void export_union_sphere512(pybind11::module& m);
void export_convex_polyhedron8(pybind11::module& m);
void export_convex_polyhedron16(pybind11::module& m);
void export_convex_polyhedron32(pybind11::module& m);
void export_convex_polyhedron64(pybind11::module& m);
void export_convex_polyhedron128(pybind11::module& m);

void export_convex_spheropolyhedron8(pybind11::module& m);
void export_convex_spheropolyhedron16(pybind11::module& m);
void export_convex_spheropolyhedron32(pybind11::module& m);
void export_convex_spheropolyhedron64(pybind11::module& m);
void export_convex_spheropolyhedron128(pybind11::module& m);

void export_external_fields(pybind11::module& m);
}

#endif // __MODULES__
