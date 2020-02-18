// Copyright (c) 2009-2019 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.

#pragma once

#include <cstdint>
#include <pybind11/pybind11.h>

#include "HOOMDMath.h"

/** Defines quantities that vary with time steps.

    Variant provides an interface to define quanties (such as kT) that vary over time. The base
    class provides a callable interface. Derived classes implement specific kinds of varying
    quantities.
*/
class PYBIND11_EXPORT Variant
    {
    public:
        /// Construct a Variant,
        Variant() { }

        virtual ~Variant() { }

        /** Return the value of the Variant at the given time step.

            @param timestep Time step to query.
            @returns The value of the variant.
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

        /** Construct a VariantConstant.

            @param value The value.
        */
        VariantConstant(Scalar value)
            : m_value(value)
            {
            }

        /// Return the value.
        Scalar operator()(uint64_t timestep)
            {
            return m_value;
            }

        /// Set the value.
        void setValue(Scalar value)
            {
            m_value = value;
            }

        /// Get the value.
        Scalar getValue()
            {
            return m_value;
            }

    protected:
        /// The value.
        Scalar m_value;
    };

/** Ramp variant.

    Variant that ramps linearly from A to B over a given number of steps.
*/
class PYBIND11_EXPORT VariantRamp : public Variant
    {
    public:

        /** Construct a VariantRamp.

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

        /// Evaluate the ramp.
        Scalar operator()(uint64_t timestep)
            {
            if (timestep < m_t_start)
                {
                return m_A;
                }
            else if (timestep < m_t_start + m_t_ramp)
                {
                // interpolate between A and B
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

        /// Get the starting time step.
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

        /// Get the length of the ramp.
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


/** Ramp variant

    Variant that cycles linearly from A to B and back again over a given number of steps.
*/
class PYBIND11_EXPORT VariantCycle : public Variant
    {
    public:

        /** Construct a VariantCycle.

            @param A The first value.
            @param B The second value.
            @param t_start The starting time step.
            @param t_A The hold time at the first value.
            @param t_AB The time spent ramping from A to B.
            @param t_B The hold time at the second value.
            @param t_BA The time spent ramping from B to A.
        */
        VariantCycle(Scalar A,
                     Scalar B,
                     uint64_t t_start,
                     uint64_t t_A,
                     uint64_t t_AB,
                     uint64_t t_B,
                     uint64_t t_BA)
            {
            setA(A);
            setB(B);
            setTStart(t_start);
            setTA(t_A);
            setTAB(t_AB);
            setTB(t_B);
            setTBA(t_BA);
            }

        /// Evaluate the periodic cycle
        Scalar operator()(uint64_t timestep)
            {
            if (timestep < m_t_start)
                {
                return m_A;
                }
            else
                {
                // find the t value within the period
                uint64_t delta = timestep - m_t_start;
                uint64_t period = m_t_A + m_t_AB + m_t_B + m_t_BA;
                delta -= (delta / period) * period;

                // select value based on the position in the cycle
                if (delta < m_t_A)
                    {
                    return m_A;
                    }
                else if (delta < m_t_A + m_t_AB)
                    {
                    uint64_t v = delta - m_t_A;
                    double s = double(v) / double(m_t_AB);
                    return m_B * s + m_A * (1.0 - s);
                    }
                else if (delta < m_t_A + m_t_AB + m_t_B)
                    {
                    return m_B;
                    }
                else
                    {
                    uint64_t v = delta - (m_t_A + m_t_AB + m_t_B);
                    double s = double(v) / double(m_t_BA);
                    return m_A * s + m_B * (1.0 - s);
                    }
                }
            }

        /// Set the first value.
        void setA(Scalar A)
            {
            m_A = A;
            }

        /// Get the first value.
        Scalar getA()
            {
            return m_A;
            }

        /// Set the second value.
        void setB(Scalar B)
            {
            m_B = B;
            }

        /// Get the second value.
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

        /// Set the length of time at A.
        void setTA(uint64_t t_A)
            {
            m_t_A = t_A;
            }

        /// Get the length of time at A.
        uint64_t getTA()
            {
            return m_t_A;
            }

        /// Set the length of the AB ramp.
        void setTAB(uint64_t t_AB)
            {
            // doubles can only represent integers accuracy up to 2**53.
            if (t_AB >= 9007199254740992ull)
                {
                throw std::invalid_argument("t_AB must be less than 2**53");
                }
            m_t_AB = t_AB;
            }

        /// Get the length of the AB ramp.
        uint64_t getTAB()
            {
            return m_t_AB;
            }

        /// Set the length of time at B.
        void setTB(uint64_t t_B)
            {
            m_t_B = t_B;
            }

        /// Get the length of time at B.
        uint64_t getTB()
            {
            return m_t_B;
            }

        /// Set the length of the BA ramp.
        void setTBA(uint64_t t_BA)
            {
            // doubles can only represent integers accuracy up to 2**53.
            if (t_BA >= 9007199254740992ull)
                {
                throw std::invalid_argument("t_BA must be less than 2**53");
                }
            m_t_BA = t_BA;
            }

        /// Get the length of the BA ramp.
        uint64_t getTBA()
            {
            return m_t_BA;
            }

    protected:
        /// The starting value.
        Scalar m_A;

        /// The ending value.
        Scalar m_B;

        /// The starting time step.
        uint64_t m_t_start;

        /// The hold time at the first value.
        uint64_t m_t_A;

        /// The hold time at the second value.
        uint64_t m_t_B;

        /// The length of the AB ramp.
        uint64_t m_t_AB;

        /// The length of the BA ramp.
        uint64_t m_t_BA;
    };


/// Export Variant classes to Python
void export_Variant(pybind11::module& m);
