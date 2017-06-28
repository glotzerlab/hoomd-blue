// Copyright (c) 2009-2017 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.

// Maintainer: mphoward

/*!
 * \file mpcd/ATCollisionMethodGPU.h
 * \brief Definition of mpcd::ATCollisionMethodGPU
 */

#include "ATCollisionMethodGPU.h"
#include "ATCollisionMethodGPU.cuh"

mpcd::ATCollisionMethodGPU::ATCollisionMethodGPU(std::shared_ptr<mpcd::SystemData> sysdata,
                                                 unsigned int cur_timestep,
                                                 unsigned int period,
                                                 int phase,
                                                 unsigned int seed,
                                                 std::shared_ptr<mpcd::CellThermoCompute> thermo,
                                                 std::shared_ptr<mpcd::CellThermoCompute> rand_thermo,
                                                 std::shared_ptr<::Variant> T)
    : mpcd::ATCollisionMethod(sysdata,cur_timestep,period,phase,seed,thermo,rand_thermo,T)
    {
    m_tuner_draw.reset(new Autotuner(32, 1024, 32, 5, 100000, "mpcd_at_draw", m_exec_conf));
    m_tuner_apply.reset(new Autotuner(32, 1024, 32, 5, 100000, "mpcd_at_apply", m_exec_conf));
    }

void mpcd::ATCollisionMethodGPU::drawVelocities(unsigned int timestep)
    {
    mpcd::ATCollisionMethod::drawVelocities(timestep);
    }

void mpcd::ATCollisionMethodGPU::applyVelocities()
    {
    mpcd::ATCollisionMethod::applyVelocities();
    }

/*!
 * \param m Python module to export to
 */
void mpcd::detail::export_ATCollisionMethodGPU(pybind11::module& m)
    {
    namespace py = pybind11;
    py::class_<mpcd::ATCollisionMethodGPU, std::shared_ptr<mpcd::ATCollisionMethodGPU> >
        (m, "ATCollisionMethodGPU", py::base<mpcd::ATCollisionMethod>())
        .def(py::init<std::shared_ptr<mpcd::SystemData>,
                      unsigned int,
                      unsigned int,
                      int,
                      unsigned int,
                      std::shared_ptr<mpcd::CellThermoCompute>,
                      std::shared_ptr<mpcd::CellThermoCompute>,
                      std::shared_ptr<::Variant>>())
    ;
    }
