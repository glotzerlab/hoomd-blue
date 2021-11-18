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
void export_NF_SWCA_2D(pybind11::module& m)
    {
    typedef SWCAPotential<Scalar, Scalar4, NoFriction<Scalar>> SWCA;
    typedef DEM2DForceCompute<Scalar, Scalar4, SWCA> SWCA_DEM_2D;

    pybind11::class_<SWCA_DEM_2D, ForceCompute, std::shared_ptr<SWCA_DEM_2D>>(m, "SWCADEM2D")
        .def(pybind11::init<std::shared_ptr<SystemDefinition>,
                            std::shared_ptr<md::NeighborList>,
                            Scalar,
                            SWCA>())
        .def("setParams", &SWCA_DEM_2D::setParams)
        .def("setRcut", &SWCA_DEM_2D::setRcut)
        .def("connectDEMGSDShapeSpec", &SWCA_DEM_2D::connectDEMGSDShapeSpec)
        .def("slotWriteDEMGSDShapeSpec", &SWCA_DEM_2D::slotWriteDEMGSDShapeSpec)
        .def("getTypeShapesPy", &SWCA_DEM_2D::getTypeShapesPy);

#ifdef ENABLE_HIP
    typedef DEM2DForceComputeGPU<Scalar, Scalar2, Scalar4, SWCA> SWCA_DEM_2D_GPU;

    pybind11::class_<SWCA_DEM_2D_GPU, SWCA_DEM_2D, std::shared_ptr<SWCA_DEM_2D_GPU>>(m,
                                                                                     "SWCADEM2DGPU")
        .def(pybind11::init<std::shared_ptr<SystemDefinition>,
                            std::shared_ptr<md::NeighborList>,
                            Scalar,
                            SWCA>())
        .def("setParams", &SWCA_DEM_2D_GPU::setParams)
        .def("setRcut", &SWCA_DEM_2D_GPU::setRcut)
        .def("setAutotunerParams", &SWCA_DEM_2D_GPU::setAutotunerParams);
#endif
    }

    } // end namespace detail
    } // end namespace dem
    } // end namespace hoomd
