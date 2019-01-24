// Copyright (c) 2009-2019 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.

// Maintainer: joaander

/*! \file Variant.h
    \brief Declares the Variant and related classes
*/

#ifdef NVCC
#error This header cannot be compiled by nvcc
#endif

#ifndef __VARIANT_H__
#define __VARIANT_H__

// ensure that HOOMDMath.h is the first thing included
#include "HOOMDMath.h"

#include <map>
#include <hoomd/extern/pybind/include/pybind11/pybind11.h>

//! Base type for time varying quantities
/*! Virtual base class for variables specified to vary over time. The base class provides
    a common interface for all subclass Variants.
     - A double value can be gotten from the Variant at any given time
     - An offset can be specified. A value set at time "t" in a variant will be returned
       at the actual time t+offset. The intended use for the offset is so that times
       can be specified as 0, 1000, 1e6 (or whatever) and be conveniently referenced
       with 0 being the current step.
    \ingroup utils
*/
class PYBIND11_EXPORT Variant
    {
    public:
        //! Constructor
        Variant() : m_offset(0) { }
        //! Virtual destructor
        virtual ~Variant() { }
        //! Gets the value at a given time step (just returns 0)
        virtual double getValue(unsigned int timestep)
            {
            return 0.0;
            }
        //! Sets the offset
        virtual void setOffset(unsigned int offset)
            {
            m_offset = offset;
            }
    protected:
        unsigned int m_offset;  //!< Offset time
    };

//! Constant variant
/*! Specifies a value that is constant over all time */
class PYBIND11_EXPORT VariantConst : public Variant
    {
    public:
        //! Constructor
        VariantConst(double val) : m_val(val)
            {
            }
        //! Gets the value at a given time step
        virtual double getValue(unsigned int timestep)
            {
            return m_val;
            }

    private:
        double m_val;       //!< The value
    };

//! Linearly interpolated variant
/*! This variant is given a series of timestep,value pairs. The value at each time step is
    calculated via linear interpolation between the two nearest points. When the timestep
    requested is before or after the end of the specified points, the value at the beginning
    (or end) is returned.
*/
class PYBIND11_EXPORT VariantLinear : public Variant
    {
    public:
        //! Constructs an empty variant
        VariantLinear();
        //! Gets the value at a given time step
        virtual double getValue(unsigned int timestep);
        //! Sets a point in the interpolation
        void setPoint(unsigned int timestep, double val);

    private:
        std::map<unsigned int, double> m_values;    //!< Values to interpolate
        std::map<unsigned int, double>::iterator    m_a,    //!< First point in the pair to interpolate
        m_b;    //!< Second point in the pair to interpolate
    };

//! Exports Variant* classes to python
void export_Variant(pybind11::module& m);

#endif
