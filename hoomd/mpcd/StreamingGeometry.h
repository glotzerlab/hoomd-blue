// Copyright (c) 2009-2024 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

/*!
 * \file mpcd/StreamingGeometry.h
 * \brief Definition of valid MPCD streaming geometries.
 */

#ifndef MPCD_STREAMING_GEOMETRY_H_
#define MPCD_STREAMING_GEOMETRY_H_

#include "BoundaryCondition.h"
#include "BulkGeometry.h"
#include "SlitGeometry.h"
#include "SlitPoreGeometry.h"

#ifndef __HIPCC__
#include <pybind11/pybind11.h>

namespace hoomd
    {
namespace mpcd
    {
namespace detail
    {
//! Export boundary enum to python
void export_boundary(pybind11::module& m);

//! Export BulkGeometry to python
void export_BulkGeometry(pybind11::module& m);

//! Export SlitGeometry to python
void export_SlitGeometry(pybind11::module& m);

//! Export SlitPoreGeometry to python
void export_SlitPoreGeometry(pybind11::module& m);

    } // end namespace detail
    } // end namespace mpcd
    } // end namespace hoomd
#endif // __HIPCC__
#endif // MPCD_STREAMING_GEOMETRY_H_
