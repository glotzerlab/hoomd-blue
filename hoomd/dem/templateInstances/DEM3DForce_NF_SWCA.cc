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

void export_NF_SWCA_3D(py::module& m)
    {
    typedef SWCAPotential<Scalar, Scalar4, NoFriction<Scalar> > SWCA;
    typedef DEM3DForceCompute<Scalar, Scalar4, SWCA> SWCA_DEM_3D;

    py::class_<SWCA_DEM_3D, std::shared_ptr<SWCA_DEM_3D> >(m, "SWCADEM3D", py::base<ForceCompute>())
        .def(py::init< std::shared_ptr<SystemDefinition>, std::shared_ptr<NeighborList>, Scalar, SWCA>())
        .def("setParams", &SWCA_DEM_3D::setParams)
        .def("setRcut", &SWCA_DEM_3D::setRcut)
        .def("connectDEMGSDShapeSpec", &SWCA_DEM_3D::connectDEMGSDShapeSpec)
        .def("slotWriteDEMGSDShapeSpec", &SWCA_DEM_3D::slotWriteDEMGSDShapeSpec)
        .def("getTypeShapesPy", &SWCA_DEM_3D::getTypeShapesPy)
        ;

#ifdef ENABLE_CUDA
    typedef DEM3DForceComputeGPU<Scalar, Scalar4, SWCA> SWCA_DEM_3D_GPU;

    py::class_<SWCA_DEM_3D_GPU, std::shared_ptr<SWCA_DEM_3D_GPU> >(m, "SWCADEM3DGPU", py::base<SWCA_DEM_3D>())
        .def(py::init< std::shared_ptr<SystemDefinition>, std::shared_ptr<NeighborList>, Scalar, SWCA>())
        .def("setParams", &SWCA_DEM_3D_GPU::setParams)
        .def("setRcut", &SWCA_DEM_3D_GPU::setRcut)
        .def("setAutotunerParams", &SWCA_DEM_3D_GPU::setAutotunerParams)
        ;
#endif
    }
