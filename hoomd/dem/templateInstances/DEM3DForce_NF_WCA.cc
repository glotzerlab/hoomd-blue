// Copyright (c) 2009-2019 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.

#include <hoomd/HOOMDMath.h>

#include "../DEMEvaluator.h"
#include "../NoFriction.h"
#include "../WCAPotential.h"
#include "../SWCAPotential.h"
#include "../DEM2DForceCompute.h"
#include "../DEM2DForceComputeGPU.h"
#include "../DEM3DForceCompute.h"
#include "../DEM3DForceComputeGPU.h"

#include <hoomd/extern/pybind/include/pybind11/pybind11.h>
namespace py = pybind11;

void export_NF_WCA_3D(py::module& m)
    {
    typedef WCAPotential<Scalar, Scalar4, NoFriction<Scalar> > WCA;
    typedef DEM3DForceCompute<Scalar, Scalar4, WCA> WCA_DEM_3D;

    py::class_<WCA_DEM_3D, std::shared_ptr<WCA_DEM_3D> >(m, "WCADEM3D", py::base<ForceCompute>())
        .def(py::init< std::shared_ptr<SystemDefinition>, std::shared_ptr<NeighborList>, Scalar, WCA>())
        .def("setParams", &WCA_DEM_3D::setParams)
        .def("setRcut", &WCA_DEM_3D::setRcut)
        .def("connectDEMGSDShapeSpec", &WCA_DEM_3D::connectDEMGSDShapeSpec)
        .def("slotWriteDEMGSDShapeSpec", &WCA_DEM_3D::slotWriteDEMGSDShapeSpec)
        .def("getTypeShapesPy", &WCA_DEM_3D::getTypeShapesPy)
        ;

#ifdef ENABLE_CUDA
    typedef DEM3DForceComputeGPU<Scalar, Scalar4, WCA> WCA_DEM_3D_GPU;

    py::class_<WCA_DEM_3D_GPU, std::shared_ptr<WCA_DEM_3D_GPU> >(m, "WCADEM3DGPU", py::base<WCA_DEM_3D>())
        .def(py::init< std::shared_ptr<SystemDefinition>, std::shared_ptr<NeighborList>, Scalar, WCA>())
        .def("setParams", &WCA_DEM_3D_GPU::setParams)
        .def("setRcut", &WCA_DEM_3D_GPU::setRcut)
        .def("setAutotunerParams", &WCA_DEM_3D_GPU::setAutotunerParams)
        ;
#endif
    }
