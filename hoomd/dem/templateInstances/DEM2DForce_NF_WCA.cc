// Copyright (c) 2009-2022 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

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
void export_NF_WCA_2D(pybind11::module& m)
    {
    typedef WCAPotential<Scalar, Scalar4, NoFriction<Scalar>> WCA;
    typedef DEM2DForceCompute<Scalar, Scalar4, WCA> WCA_DEM_2D;

    pybind11::class_<WCA_DEM_2D, ForceCompute, std::shared_ptr<WCA_DEM_2D>>(m, "WCADEM2D")
        .def(pybind11::init<std::shared_ptr<SystemDefinition>,
                            std::shared_ptr<md::NeighborList>,
                            Scalar,
                            WCA>())
        .def("setParams", &WCA_DEM_2D::setParams)
        .def("setRcut", &WCA_DEM_2D::setRcut)
        .def("connectDEMGSDShapeSpec", &WCA_DEM_2D::connectDEMGSDShapeSpec)
        .def("slotWriteDEMGSDShapeSpec", &WCA_DEM_2D::slotWriteDEMGSDShapeSpec)
        .def("getTypeShapesPy", &WCA_DEM_2D::getTypeShapesPy);

#ifdef ENABLE_HIP
    typedef DEM2DForceComputeGPU<Scalar, Scalar2, Scalar4, WCA> WCA_DEM_2D_GPU;

    pybind11::class_<WCA_DEM_2D_GPU, WCA_DEM_2D, std::shared_ptr<WCA_DEM_2D_GPU>>(m, "WCADEM2DGPU")
        .def(pybind11::init<std::shared_ptr<SystemDefinition>,
                            std::shared_ptr<md::NeighborList>,
                            Scalar,
                            WCA>())
        .def("setParams", &WCA_DEM_2D_GPU::setParams)
        .def("setRcut", &WCA_DEM_2D_GPU::setRcut)
        .def("setAutotunerParams", &WCA_DEM_2D_GPU::setAutotunerParams);
#endif
    }

    } // end namespace detail
    } // end namespace dem
    } // end namespace hoomd
