// Copyright (c) 2009-2024 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

#ifndef __MODULES__
#define __MODULES__

#ifndef __HIPCC__
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#endif

namespace hoomd
    {
namespace hpmc
    {
namespace detail
    {
void export_sphere(pybind11::module& m);
void export_convex_polygon(pybind11::module& m);
void export_simple_polygon(pybind11::module& m);
void export_spheropolygon(pybind11::module& m);
void export_polyhedron(pybind11::module& m);
void export_ellipsoid(pybind11::module& m);
void export_faceted_ellipsoid(pybind11::module& m);
void export_sphinx(pybind11::module& m);
void export_union_convex_polyhedron(pybind11::module& m);
void export_union_faceted_ellipsoid(pybind11::module& m);
void export_union_sphere(pybind11::module& m);
void export_convex_polyhedron(pybind11::module& m);
void export_convex_spheropolyhedron(pybind11::module& m);

    } // namespace detail
    } // namespace hpmc
    } // namespace hoomd

#endif // __MODULES__
