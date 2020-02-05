// Copyright (c) 2009-2019 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.

#pragma once

#include <cstdint>
#include <pybind11/pybind11.h>

#include "HOOMDMath.h"

/** Defines quantities that vary with time steps

    Variant provides an interface to define quanties (such as kT) that vary over time. The base
    class provides a callable interface. Derived classes implement specific kinds of varying
    quantities.
*/
class PYBIND11_EXPORT Variant
    {
    public:
        /// Construct a Variant
        Variant() { }

        virtual ~Variant() { }

        /** Return the value of the Variant at the given time step

            @param timestep Time step to query
            @returns The value of the variant
        */
        virtual Scalar operator()(uint64_t timestep)
            {
            return 0;
            }
    };

/** Constant value

    Variant that provides a constant value.
*/
class PYBIND11_EXPORT VariantConstant : public Variant
    {
    public:

        /** Construct a VariantConstant

            @param value The value.
        */
        VariantConstant(Scalar value)
            : m_value(value)
            {
            }

        Scalar operator()(uint64_t timestep)
            {
            return m_value;
            }

        /// Set the value
        void setValue(Scalar value)
            {
            m_value = value;
            }

        /// Get the value
        Scalar getValue()
            {
            return m_value;
            }

    protected:
        /// The value
        Scalar m_value;
    };

/// Export Variant classes to Python
void export_Variant(pybind11::module& m);
