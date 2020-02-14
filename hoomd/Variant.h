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

/** Ramp variant

    Variant that ramps linearly from A to B over a given number of steps.
*/
class PYBIND11_EXPORT VariantRamp : public Variant
    {
    public:

        /** Construct a VariantRamp

            @param A The starting value.
            @param B The ending value.
            @param t_start The starting time step.
            @param t_ramp The length of the ramp.
        */
        VariantRamp(Scalar A, Scalar B, uint64_t t_start, uint64_t t_ramp)
            {
            setA(A);
            setB(B);
            setTStart(t_start);
            setTRamp(t_ramp);
            }

        Scalar operator()(uint64_t timestep)
            {
            if (timestep < m_t_start)
                {
                return m_A;
                }
            else if (timestep < m_t_start + m_t_ramp)
                {
                uint64_t v = timestep - m_t_start;
                double s = double(v) / double(m_t_ramp);
                return m_B * s + m_A * (1.0 - s);
                }
            else
                {
                return m_B;
                }
            }

        /// Set the starting value.
        void setA(Scalar A)
            {
            m_A = A;
            }

        /// Get the starting value.
        Scalar getA()
            {
            return m_A;
            }

        /// Set the ending value.
        void setB(Scalar B)
            {
            m_B = B;
            }

        /// Get the ending value.
        Scalar getB()
            {
            return m_B;
            }

        /// Set the starting time step.
        void setTStart(uint64_t t_start)
            {
            m_t_start = t_start;
            }

        /// Get the starting time step
        uint64_t getTStart()
            {
            return m_t_start;
            }

        /// Set the length of the ramp.
        void setTRamp(uint64_t t_ramp)
            {
            // doubles can only represent integers accuracy up to 2**53.
            if (t_ramp >= 9007199254740992ull)
                {
                throw std::invalid_argument("t_ramp must be less than 2**53");
                }
            m_t_ramp = t_ramp;
            }

        /// Get the length of the ramp
        uint64_t getTRamp()
            {
            return m_t_ramp;
            }

    protected:
        /// The starting value.
        Scalar m_A;

        /// The ending value.
        Scalar m_B;

        /// The starting time step.
        uint64_t m_t_start;

        /// The length of the ramp.
        uint64_t m_t_ramp;
    };

/// Export Variant classes to Python
void export_Variant(pybind11::module& m);
