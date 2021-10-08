// Copyright (c) 2009-2021 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.

#include <hoomd/HOOMDMath.h>

#include "../DEM2DForceCompute.h"
#include "../DEM2DForceComputeGPU.h"
#include "../DEM3DForceCompute.h"
#include "../DEM3DForceComputeGPU.h"
#include "../DEMEvaluator.h"
#include "../NoFriction.h"
#include "../SWCAPotential.h"
#include "../WCAPotential.h"

#include <pybind11/pybind11.h>

namespace hoomd
    {
namespace dem
    {
namespace detail
    {
void export_NF_WCA_3D(pybind11::module& m)
    {
    typedef WCAPotential<Scalar, Scalar4, NoFriction<Scalar>> WCA;
    typedef DEM3DForceCompute<Scalar, Scalar4, WCA> WCA_DEM_3D;

    pybind11::class_<WCA_DEM_3D, ForceCompute, std::shared_ptr<WCA_DEM_3D>>(m, "WCADEM3D")
        .def(pybind11::init<std::shared_ptr<SystemDefinition>,
                            std::shared_ptr<md::NeighborList>,
                            Scalar,
                            WCA>())
        .def("setParams", &WCA_DEM_3D::setParams)
        .def("setRcut", &WCA_DEM_3D::setRcut)
        .def("connectDEMGSDShapeSpec", &WCA_DEM_3D::connectDEMGSDShapeSpec)
        .def("slotWriteDEMGSDShapeSpec", &WCA_DEM_3D::slotWriteDEMGSDShapeSpec)
        .def("getTypeShapesPy", &WCA_DEM_3D::getTypeShapesPy);

#ifdef ENABLE_HIP
    typedef DEM3DForceComputeGPU<Scalar, Scalar4, WCA> WCA_DEM_3D_GPU;

    pybind11::class_<WCA_DEM_3D_GPU, WCA_DEM_3D, std::shared_ptr<WCA_DEM_3D_GPU>>(m, "WCADEM3DGPU")
        .def(pybind11::init<std::shared_ptr<SystemDefinition>,
                            std::shared_ptr<md::NeighborList>,
                            Scalar,
                            WCA>())
        .def("setParams", &WCA_DEM_3D_GPU::setParams)
        .def("setRcut", &WCA_DEM_3D_GPU::setRcut)
        .def("setAutotunerParams", &WCA_DEM_3D_GPU::setAutotunerParams);
#endif
    }

    } // end namespace detail
    } // end namespace dem
    } // end namespace hoomd
