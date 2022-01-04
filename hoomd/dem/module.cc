// Copyright (c) 2009-2022 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

// Include the defined classes that are to be exported to python
#include <hoomd/HOOMDMath.h>

#include "NoFriction.h"

#include "DEM2DForceCompute.h"
#include "DEM2DForceComputeGPU.h"
#include "DEM3DForceCompute.h"
#include "DEM3DForceComputeGPU.h"
#include "DEMEvaluator.h"
#include "SWCAPotential.h"
#include "WCAPotential.h"

#include <iterator>
#include <pybind11/pybind11.h>

namespace hoomd
    {
namespace dem
    {
namespace detail
    {
void export_NF_WCA_2D(pybind11::module& m);
void export_NF_WCA_3D(pybind11::module& m);
void export_NF_SWCA_3D(pybind11::module& m);
void export_NF_SWCA_2D(pybind11::module& m);

// Export all of the parameter wrapper objects to the python interface
void export_params(pybind11::module& m)
    {
    pybind11::class_<NoFriction<Scalar>>(m, "NoFriction").def(pybind11::init());

    typedef WCAPotential<Scalar, Scalar4, NoFriction<Scalar>> WCA;
    typedef SWCAPotential<Scalar, Scalar4, NoFriction<Scalar>> SWCA;

    pybind11::class_<WCA>(m, "WCAPotential").def(pybind11::init<Scalar, NoFriction<Scalar>>());
    pybind11::class_<SWCA>(m, "SWCAPotential").def(pybind11::init<Scalar, NoFriction<Scalar>>());
    }

    } // end namespace detail
    } // end namespace dem
    } // end namespace hoomd

PYBIND11_MODULE(_dem, m)
    {
    hoomd::dem::detail::export_params(m);
    hoomd::dem::detail::export_NF_WCA_2D(m);
    hoomd::dem::detail::export_NF_WCA_3D(m);
    hoomd::dem::detail::export_NF_SWCA_2D(m);
    hoomd::dem::detail::export_NF_SWCA_3D(m);
    }
