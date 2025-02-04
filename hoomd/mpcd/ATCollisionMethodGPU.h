// Copyright (c) 2009-2024 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

/*!
 * \file mpcd/ATCollisionMethodGPU.h
 * \brief Declaration of mpcd::ATCollisionMethodGPU
 */

#ifndef MPCD_AT_COLLISION_METHOD_GPU_H_
#define MPCD_AT_COLLISION_METHOD_GPU_H_

#ifdef __HIPCC__
#error This header cannot be compiled by nvcc
#endif

#include "ATCollisionMethod.h"
#include "hoomd/Autotuner.h"

namespace hoomd
    {
namespace mpcd
    {
class PYBIND11_EXPORT ATCollisionMethodGPU : public mpcd::ATCollisionMethod
    {
    public:
    //! Constructor
    ATCollisionMethodGPU(std::shared_ptr<SystemDefinition> sysdef,
                         uint64_t cur_timestep,
                         uint64_t period,
                         int phase,
                         std::shared_ptr<Variant> T);

    void setCellList(std::shared_ptr<mpcd::CellList> cl);

    protected:
    //! Draw velocities for particles in each cell on the GPU
    virtual void drawVelocities(uint64_t timestep);

    //! Apply the random velocities to particles in each cell on the GPU
    virtual void applyVelocities();

    private:
    std::shared_ptr<Autotuner<1>> m_tuner_draw;  //!< Tuner for drawing random velocities
    std::shared_ptr<Autotuner<1>> m_tuner_apply; //!< Tuner for applying random velocities
    };
    } // end namespace mpcd
    } // end namespace hoomd
#endif // MPCD_AT_COLLISION_METHOD_GPU_H_
