// Copyright (c) 2009-2019 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.


// Maintainer: joaander

#ifndef _COMPUTE_THERMO_TYPES_H_
#define _COMPUTE_THERMO_TYPES_H_

#include "HOOMDMath.h"
/*! \file ComputeThermoTypes.h
    \brief Data structures common to both CPU and GPU implementations of ComputeThermo
    */

//! Enum for indexing the GPUArray of computed values
struct thermo_index
    {
    //! The enum
    enum Enum
        {
        translational_kinetic_energy=0,      //!< Index for the kinetic energy in the GPUArray
        rotational_kinetic_energy,       //!< Rotational kinetic energy
        potential_energy,    //!< Index for the potential energy in the GPUArray
        pressure,            //!< Total pressure
        pressure_xx,         //!< Index for the xx component of the pressure tensor in the GPUArray
        pressure_xy,         //!< Index for the xy component of the pressure tensor in the GPUArray
        pressure_xz,         //!< Index for the xz component of the pressure tensor in the GPUArray
        pressure_yy,         //!< Index for the yy component of the pressure tensor in the GPUArray
        pressure_yz,         //!< Index for the yz component of the pressure tensor in the GPUArray
        pressure_zz,         //!< Index for the zz component of the pressure tensor in the GPUArray
        num_quantities       // final element to count number of quantities
        };
    };

//! structure for storing the components of the pressure tensor
struct PressureTensor
    {
    //! The six components of the upper triangular pressure tensor
    Scalar xx; //!< xx component
    Scalar xy; //!< xy component
    Scalar xz; //!< xz component
    Scalar yy; //!< yy component
    Scalar yz; //!< yz component
    Scalar zz; //!< zz component
    };
#endif
