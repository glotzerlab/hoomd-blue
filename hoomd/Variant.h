// Copyright (c) 2009-2024 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

#pragma once

#include <cstdint>
#include <pybind11/pybind11.h>
#include <utility>

#include "HOOMDMath.h"

namespace hoomd
    {
/** Defines scalar quantities that vary with time steps.

    Variant provides an interface to define scalar quanties (such as kT) that vary over time. The
   base class provides a callable interface. Derived classes implement specific kinds of varying
    quantities.
*/
class PYBIND11_EXPORT Variant
    {
    public:
    /// Construct a Variant.
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

    /// Returns the minimum of the variant
    virtual Scalar min() = 0;

    /// Returns the maximum of the variant
    virtual Scalar max() = 0;

    /// Returns the range [min, max] of the variant
    virtual std::pair<Scalar, Scalar> range()
        {
        return std::pair<Scalar, Scalar>(min(), max());
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
    VariantConstant(Scalar value) : m_value(value) { }

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
    Scalar getValue() const
        {
        return m_value;
        }

    /// Returns the given constant, c
    virtual Scalar min()
        {
        return m_value;
        }

    /// Returns the given constant, c
    virtual Scalar max()
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
            double s = double(timestep - m_t_start) / double(m_t_ramp);
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
    Scalar getA() const
        {
        return m_A;
        }

    /// Set the ending value.
    void setB(Scalar B)
        {
        m_B = B;
        }

    /// Get the ending value.
    Scalar getB() const
        {
        return m_B;
        }

    /// Set the starting time step.
    void setTStart(uint64_t t_start)
        {
        m_t_start = t_start;
        }

    /// Get the starting time step.
    uint64_t getTStart() const
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
    uint64_t getTRamp() const
        {
        return m_t_ramp;
        }

    /// Return min
    Scalar min()
        {
        return m_A > m_B ? m_B : m_A;
        }

    /// Return max
    Scalar max()
        {
        return m_A > m_B ? m_A : m_B;
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

/** Cycle variant

    Variant that cycles linearly from A to B and back again over a given number of steps.
*/
class PYBIND11_EXPORT VariantCycle : public Variant
    {
    public:
    /** Construct a VariantCycle.

        @param A The first value.
        @param B The second value.
        @param t_start The starting time step.
        @param t_A The holding time at A.
        @param t_AB The time spent ramping from A to B.
        @param t_B The holding time at B.
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

    /// Set A.
    void setA(Scalar A)
        {
        m_A = A;
        }

    /// Get A.
    Scalar getA() const
        {
        return m_A;
        }

    /// Set B.
    void setB(Scalar B)
        {
        m_B = B;
        }

    /// Get B.
    Scalar getB() const
        {
        return m_B;
        }

    /// Set the starting time step.
    void setTStart(uint64_t t_start)
        {
        m_t_start = t_start;
        }

    /// Get the starting time step.
    uint64_t getTStart() const
        {
        return m_t_start;
        }

    /// Set the holding time at A.
    void setTA(uint64_t t_A)
        {
        m_t_A = t_A;
        }

    /// Get the holding time at A.
    uint64_t getTA() const
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
    uint64_t getTAB() const
        {
        return m_t_AB;
        }

    /// Set the holding time at B.
    void setTB(uint64_t t_B)
        {
        m_t_B = t_B;
        }

    /// Get the holding time at B.
    uint64_t getTB() const
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
    uint64_t getTBA() const
        {
        return m_t_BA;
        }

    /// Return min
    Scalar min()
        {
        return m_A > m_B ? m_B : m_A;
        }

    /// Return max
    Scalar max()
        {
        return m_A > m_B ? m_A : m_B;
        }

    protected:
    /// The starting value.
    Scalar m_A;

    /// The ending value.
    Scalar m_B;

    /// The starting time step.
    uint64_t m_t_start;

    /// The holding time at A.
    uint64_t m_t_A;

    /// The holding time at B.
    uint64_t m_t_B;

    /// The length of the AB ramp.
    uint64_t m_t_AB;

    /// The length of the BA ramp.
    uint64_t m_t_BA;
    };

/** Power variant

    Variant that goes from m_A -> m_B as timestep ^ (power)
*/
class PYBIND11_EXPORT VariantPower : public Variant
    {
    public:
    /** Construct a VariantPower.

        @param A the initial value
        @param B the final value
        @param power the power to approach as
        @param t_start the first timestep
        @param t_ramp the length of the approach
    */
    VariantPower(Scalar A, Scalar B, double power, uint64_t t_start, uint64_t t_ramp)
        : m_A(A), m_B(B), m_power(power), m_t_start(t_start), m_t_ramp(t_ramp)
        {
        m_offset = computeOffset(m_A, m_B);
        setStartEnd();
        }

    /// Return the value.
    Scalar operator()(uint64_t timestep)
        {
        if (timestep <= m_t_start)
            {
            return m_A;
            }
        else if (timestep < m_t_start + m_t_ramp)
            {
            double s = double(timestep - m_t_start) / double(m_t_ramp);
            double inv_result = m_inv_end * s + m_inv_start * (1.0 - s);
            return pow(inv_result, m_power) - m_offset;
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
        setInternals();
        }

    /// Get the starting value.
    Scalar getA() const
        {
        return m_A;
        }

    /// Set the ending value.
    void setB(Scalar B)
        {
        m_B = B;
        setInternals();
        }

    /// Get the ending value.
    Scalar getB() const
        {
        return m_B;
        }

    /// Set the approaching power.
    void setPower(double power)
        {
        m_power = power;
        setStartEnd();
        }

    /// Get the ending value.
    Scalar getPower() const
        {
        return m_power;
        }

    /// Set the starting time step.
    void setTStart(uint64_t t_start)
        {
        m_t_start = t_start;
        }

    /// Get the starting time step.
    uint64_t getTStart() const
        {
        return m_t_start;
        }

    /// Set the length of the ramp.
    void setTRamp(uint64_t t_ramp)
        {
        // Doubles can only represent integers accurately up to 2**53.
        if (t_ramp >= 9007199254740992ull)
            {
            throw std::invalid_argument("t_ramp must be less than 2**53");
            }
        m_t_ramp = t_ramp;
        }

    /// Get the length of the ramp.
    uint64_t getTRamp() const
        {
        return m_t_ramp;
        }

    /// Return min
    Scalar min()
        {
        return m_A > m_B ? m_B : m_A;
        }

    /// Return max
    Scalar max()
        {
        return m_A > m_B ? m_A : m_B;
        }

    protected:
    /// Get the new offset
    double computeOffset(Scalar a, Scalar b)
        {
        if (a > 0 && b > 0)
            {
            return 0;
            }
        else
            {
            return a > b ? -b : -a;
            }
        }

    void setStartEnd()
        {
        m_inv_start = pow(m_A + m_offset, 1.0 / m_power);
        m_inv_end = pow(m_B + m_offset, 1.0 / m_power);
        }

    void setInternals()
        {
        auto new_offset = computeOffset(m_A, m_B);
        if (new_offset != m_offset)
            {
            m_offset = new_offset;
            setStartEnd();
            }
        }

    /// initial value
    Scalar m_A;

    /// final value
    Scalar m_B;

    /// power of the approach to m_B
    double m_power;

    /// starting timestep
    uint64_t m_t_start;

    /// length of approach to m_B
    uint64_t m_t_ramp;

    /// offset from given positions allows for negative values
    double m_offset;

    /// internal start to work with negative values
    double m_inv_start;

    /// internal end to work with negative values
    double m_inv_end;
    };

namespace detail
    {
/// Export Variant classes to Python
void export_Variant(pybind11::module& m);

    } // end namespace detail

    } // end namespace hoomd
