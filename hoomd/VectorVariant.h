// Copyright (c) 2009-2023 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

// Copyright (c) 2009-2023 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

#pragma once

#include <cstdint>
#include <pybind11/pybind11.h>
#include <utility>

#include "BoxDim.h"
#include "HOOMDMath.h"

namespace hoomd
    {
/** Defines vector quantities that vary with time steps.

    VectorVariant provides an interface to define vector quanties (such as box dimensions) that vary
   over time. The base class provides a callable interface. Derived classes implement specific kinds
   of varying quantities.
*/
template<unsigned int ndim> class PYBIND11_EXPORT VectorVariant
    {
    public:
    virtual ~VectorVariant() { }
    typedef std::array<Scalar, ndim> array_type;

    /** Return the value of the Variant at the given time step.

        @param timestep Time step to query.
        @returns The value of the variant.
    */
    virtual array_type operator()(uint64_t timestep)
        {
        std::array<Scalar, ndim> ret;
        ret.fill(0);
        return ret;
        }
    };

class PYBIND11_EXPORT VectorVariantBox : public VectorVariant<6>
    {
    protected:
    static std::array<Scalar, 6> box_to_array(std::shared_ptr<BoxDim> box)
        {
        return std::array<Scalar, 6> {box->getL().x,
                                      box->getL().y,
                                      box->getL().z,
                                      box->getTiltFactorXY(),
                                      box->getTiltFactorXZ(),
                                      box->getTiltFactorYZ()};
        }
    };

class PYBIND11_EXPORT VectorVariantBoxConstant : public VectorVariantBox
    {
    public:
    /** Construct a VectorVariantBoxConstant.

        @param box The box.
    */
    VectorVariantBoxConstant(std::shared_ptr<BoxDim> box)
        : m_box(box)
        {
        }

    virtual ~VectorVariantBoxConstant() { }

    /// Return the value.
    virtual array_type operator()(uint64_t timestep)
        {
        return box_to_array(m_box);
        }

    std::shared_ptr<BoxDim> getBox()
        {
        return m_box;
        }

    void setBox(std::shared_ptr<BoxDim> box)
        {
        m_box = box;
        }

    protected:
    std::shared_ptr<BoxDim> m_box;
    };

class PYBIND11_EXPORT VectorVariantBoxLinear : public VectorVariantBox
    {
    public:
    /** Construct a VectorVariantBoxLinear to interpolate between two boxes linearly in time.

        @param box1 The initial box
        @param box2 The final box
    */
    VectorVariantBoxLinear(std::shared_ptr<BoxDim> box1,
                           std::shared_ptr<BoxDim> box2,
                           uint64_t t_start,
                           uint64_t t_ramp)
        : m_box1(box1), m_box2(box2), m_t_start(t_start), m_t_ramp(t_ramp)
        {
        }

    /// Return the value.
    virtual array_type operator()(uint64_t timestep)
        {
        if (timestep < m_t_start)
            {
            return std::array<Scalar, 6> {m_box1->getL().x,
                                          m_box1->getL().y,
                                          m_box1->getL().z,
                                          m_box1->getTiltFactorXY(),
                                          m_box1->getTiltFactorXZ(),
                                          m_box1->getTiltFactorYZ()};
            }
        else if (timestep >= m_t_start + m_t_ramp)
            {
            return std::array<Scalar, 6> {m_box2->getL().x,
                                          m_box2->getL().y,
                                          m_box2->getL().z,
                                          m_box2->getTiltFactorXY(),
                                          m_box2->getTiltFactorXZ(),
                                          m_box2->getTiltFactorYZ()};
            }
        else
            {
            double s = double(timestep - m_t_start) / double(m_t_ramp);
            std::array<Scalar, 6> value;
            Scalar3 L1 = m_box1->getL();
            Scalar3 L2 = m_box2->getL();
            Scalar xy1 = m_box1->getTiltFactorXY();
            Scalar xy2 = m_box2->getTiltFactorXY();
            Scalar xz1 = m_box1->getTiltFactorXZ();
            Scalar xz2 = m_box2->getTiltFactorXZ();
            Scalar yz1 = m_box1->getTiltFactorYZ();
            Scalar yz2 = m_box2->getTiltFactorYZ();
            value[0] = L2.x * s + L1.x * (1.0 - s);
            value[1] = L2.y * s + L1.y * (1.0 - s);
            value[2] = L2.z * s + L1.z * (1.0 - s);
            value[3] = xy2 * s + xy1 * (1.0 - s);
            value[4] = xz2 * s + xz1 * (1.0 - s);
            value[5] = yz2 * s + yz1 * (1.0 - s);
            return value;
            }
        }

    std::shared_ptr<BoxDim> getBox1()
        {
        return m_box1;
        }

    void setBox1(std::shared_ptr<BoxDim> box)
        {
        m_box1 = box;
        }

    std::shared_ptr<BoxDim> getBox2()
        {
        return m_box2;
        }

    void setBox2(std::shared_ptr<BoxDim> box)
        {
        m_box2 = box;
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

    protected:
    /// The starting box.
    std::shared_ptr<BoxDim> m_box1;

    /// The final box.
    std::shared_ptr<BoxDim> m_box2;

    /// The starting time step.
    uint64_t m_t_start;

    /// The length of the ramp.
    uint64_t m_t_ramp;
    };

class PYBIND11_EXPORT VectorVariantBoxInverseVolumeRamp : public VectorVariantBox
    {
    public:
    VectorVariantBoxInverseVolumeRamp(std::shared_ptr<BoxDim> box1,
                                      Scalar final_volume,
                                      uint64_t t_start,
                                      uint64_t t_ramp)
        : m_box1(box1), m_final_volume(final_volume), m_t_start(t_start), m_t_ramp(t_ramp)
        {
        m_is2D = m_box1->getL().z == 0;
        m_vol1 = m_box1->getVolume(m_is2D);
        }

    virtual array_type operator()(uint64_t timestep)
        {
        if (timestep < m_t_start)
            {
            return box_to_array(m_box1);
            }
        else if (timestep >= m_t_start + m_t_ramp)
            {
            Scalar scale;
            if (m_is2D)
                {
                scale = pow(m_final_volume / m_vol1, Scalar(1.0 / 2.0));
                }
            else
                {
                scale = pow(m_final_volume / m_vol1, Scalar(1.0 / 3.0));
                }
            std::array<Scalar, 6> value;
            Scalar3 L1 = m_box1->getL();
            value[0] = L1.x * scale;
            value[1] = L1.y * scale;
            value[2] = L1.z * scale;
            value[3] = m_box1->getTiltFactorXY();
            value[4] = m_box1->getTiltFactorXZ();
            value[5] = m_box1->getTiltFactorYZ();
            return value;
            }
        else
            {
            double s = double(timestep - m_t_start) / double(m_t_ramp);
            Scalar current_volume = s * m_final_volume + (1.0 - s) * m_vol1;
            Scalar scale;
            if (m_is2D)
                {
                scale = pow(current_volume / m_vol1, Scalar(1.0 / 2.0));
                }
            else
                {
                scale = pow(current_volume / m_vol1, Scalar(1.0 / 3.0));
                }

            std::array<Scalar, 6> value;
            Scalar3 L1 = m_box1->getL();
            value[0] = L1.x * scale;
            value[1] = L1.y * scale;
            value[2] = L1.z * scale;
            value[3] = m_box1->getTiltFactorXY();
            value[4] = m_box1->getTiltFactorXZ();
            value[5] = m_box1->getTiltFactorYZ();
            return value;
            }
        }

    std::shared_ptr<BoxDim> getBox1()
        {
        return m_box1;
        }

    void setBox1(std::shared_ptr<BoxDim> box)
        {
        m_box1 = box;
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

    protected:
    /// The starting box.
    std::shared_ptr<BoxDim> m_box1;

    /// The volume of box1.
    Scalar m_vol1;

    /// The volume of the box at the end of the ramp.
    Scalar m_final_volume;

    /// The starting time step.
    uint64_t m_t_start;

    /// The length of the ramp.
    uint64_t m_t_ramp;

    /// Whether box1 is 2-dimensional or not
    bool m_is2D;

    Scalar m_current_volume;
    };

namespace detail
    {
/// Export Variant classes to Python
void export_VectorVariantBox(pybind11::module& m);
void export_VectorVariantBoxConstant(pybind11::module& m);

    } // end namespace detail

    } // end namespace hoomd
