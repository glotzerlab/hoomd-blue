// Copyright (c) 2009-2019 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.

// Maintainer: mphoward

/*!
 * \file mpcd/ATCollisionMethod.h
 * \brief Declaration of mpcd::ATCollisionMethod
 */

#ifndef MPCD_AT_COLLISION_METHOD_H_
#define MPCD_AT_COLLISION_METHOD_H_

#ifdef NVCC
#error This header cannot be compiled by nvcc
#endif

#include "CollisionMethod.h"
#include "CellThermoCompute.h"

#include "hoomd/Variant.h"

namespace mpcd
{

//! Implements the Anderson thermostat collision rule for MPCD.
class PYBIND11_EXPORT ATCollisionMethod : public mpcd::CollisionMethod
    {
    public:
        //! Constructor
        ATCollisionMethod(std::shared_ptr<mpcd::SystemData> sysdata,
                          unsigned int cur_timestep,
                          unsigned int period,
                          int phase,
                          unsigned int seed,
                          std::shared_ptr<mpcd::CellThermoCompute> thermo,
                          std::shared_ptr<mpcd::CellThermoCompute> rand_thermo,
                          std::shared_ptr<::Variant> T);

        //! Destructor
        virtual ~ATCollisionMethod();

        //! Set the temperature and enable the thermostat
        void setTemperature(std::shared_ptr<::Variant> T)
            {
            m_T = T;
            }

    protected:
        std::shared_ptr<mpcd::CellThermoCompute> m_thermo;      //!< Cell thermo
        std::shared_ptr<mpcd::CellThermoCompute> m_rand_thermo; //!< Cell thermo for random velocities
        std::shared_ptr<::Variant> m_T; //!< Temperature for thermostat

        //! Implementation of the collision rule
        virtual void rule(unsigned int timestep);

        //! Draw velocities for particles in each cell
        virtual void drawVelocities(unsigned int timestep);

        //! Apply the random velocities to particles in each cell
        virtual void applyVelocities();
    };

namespace detail
{
//! Export ATCollisionMethod to python
void export_ATCollisionMethod(pybind11::module& m);
} // end namespace detail

} // end namespace mpcd

#endif // MPCD_AT_COLLISION_METHOD_H_
