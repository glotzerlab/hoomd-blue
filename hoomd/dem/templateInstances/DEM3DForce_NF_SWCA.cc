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
void export_NF_SWCA_3D(pybind11::module& m)
    {
    typedef SWCAPotential<Scalar, Scalar4, NoFriction<Scalar>> SWCA;
    typedef DEM3DForceCompute<Scalar, Scalar4, SWCA> SWCA_DEM_3D;

    pybind11::class_<SWCA_DEM_3D, ForceCompute, std::shared_ptr<SWCA_DEM_3D>>(m, "SWCADEM3D")
        .def(pybind11::init<std::shared_ptr<SystemDefinition>,
                            std::shared_ptr<md::NeighborList>,
                            Scalar,
                            SWCA>())
        .def("setParams", &SWCA_DEM_3D::setParams)
        .def("setRcut", &SWCA_DEM_3D::setRcut)
        .def("connectDEMGSDShapeSpec", &SWCA_DEM_3D::connectDEMGSDShapeSpec)
        .def("slotWriteDEMGSDShapeSpec", &SWCA_DEM_3D::slotWriteDEMGSDShapeSpec)
        .def("getTypeShapesPy", &SWCA_DEM_3D::getTypeShapesPy);

#ifdef ENABLE_HIP
    typedef DEM3DForceComputeGPU<Scalar, Scalar4, SWCA> SWCA_DEM_3D_GPU;

    pybind11::class_<SWCA_DEM_3D_GPU, SWCA_DEM_3D, std::shared_ptr<SWCA_DEM_3D_GPU>>(m,
                                                                                     "SWCADEM3DGPU")
        .def(pybind11::init<std::shared_ptr<SystemDefinition>,
                            std::shared_ptr<md::NeighborList>,
                            Scalar,
                            SWCA>())
        .def("setParams", &SWCA_DEM_3D_GPU::setParams)
        .def("setRcut", &SWCA_DEM_3D_GPU::setRcut)
        .def("setAutotunerParams", &SWCA_DEM_3D_GPU::setAutotunerParams);
#endif
    }

    } // end namespace detail
    } // end namespace dem
    } // end namespace hoomd
