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
#include "Variant.h"

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
    VectorVariantBoxConstant(std::shared_ptr<BoxDim> box) : m_box(box) { }

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

class PYBIND11_EXPORT VectorVariantBoxInterpolate : public VectorVariantBox
    {
    public:
    /** Construct a VectorVariantBoxInterpolate to interpolate between two boxes linearly in time.

        @param box1 The initial box
        @param box2 The final box
    */
    VectorVariantBoxInterpolate(std::shared_ptr<BoxDim> box1,
                                std::shared_ptr<BoxDim> box2,
                                std::shared_ptr<Variant> variant)
        : m_box1(box1), m_box2(box2), m_variant(variant)
        {
        }

    /// Return the value.
    virtual array_type operator()(uint64_t timestep)
        {
        Scalar min = m_variant->min();
        Scalar max = m_variant->max();
        Scalar cur_value = (*m_variant)(timestep);
        Scalar scale = 0;
        if (cur_value == max)
            {
            scale = 1;
            }
        else if (cur_value > min)
            {
            scale = (cur_value - min) / (max - min);
            }

        const auto& box1 = *m_box1;
        const auto& box2 = *m_box2;
        Scalar3 new_L = box2.getL() * scale + box1.getL() * (1.0 - scale);
        Scalar xy = box2.getTiltFactorXY() * scale + (1.0 - scale) * box1.getTiltFactorXY();
        Scalar xz = box2.getTiltFactorXZ() * scale + (1.0 - scale) * box1.getTiltFactorXZ();
        Scalar yz = box2.getTiltFactorYZ() * scale + (1.0 - scale) * box1.getTiltFactorYZ();
        array_type value = {new_L.x, new_L.y, new_L.z, xy, xz, yz};
        return value;
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

    /// Set the variant for interpolation
    void setVariant(std::shared_ptr<Variant> variant)
        {
        m_variant = variant;
        }

    /// Get the variant for interpolation
    std::shared_ptr<Variant> getVariant()
        {
        return m_variant;
        }

    protected:
    /// The starting box, associated with the minimum of the variant.
    std::shared_ptr<BoxDim> m_box1;

    /// The final box, associated with the maximum of the variant.
    std::shared_ptr<BoxDim> m_box2;

    /// Variant that interpolates between boxes.
    std::shared_ptr<Variant> m_variant;
    };

class PYBIND11_EXPORT VectorVariantBoxInverseVolumeRamp : public VectorVariantBox
    {
    public:
    VectorVariantBoxInverseVolumeRamp(std::shared_ptr<BoxDim> box1,
                                      Scalar final_volume,
                                      uint64_t t_start,
                                      uint64_t t_ramp)
        : m_initial_box(box1), m_final_volume(final_volume), m_variant(0, 1, t_start, t_ramp)
        {
        m_is2D = m_initial_box->getL().z == 0;
        m_initial_volume = m_initial_box->getVolume(m_is2D);
        }

    virtual array_type operator()(uint64_t timestep)
        {
        Scalar s = m_variant(timestep);
        // current inverse volume = s * (1 / m_final_volume) + (1-s) * (1/m_vol1)
        // current volume = 1 / (current inverse volume)
        Scalar current_volume = 1 / (s / m_final_volume + (1.0 - s) / m_initial_volume);
        Scalar L_scale;
        if (m_is2D)
            {
            L_scale = pow(current_volume / m_initial_volume, Scalar(1.0 / 2.0));
            }
        else
            {
            L_scale = pow(current_volume / m_initial_volume, Scalar(1.0 / 3.0));
            }

        std::array<Scalar, 6> value;
        Scalar3 L1 = m_initial_box->getL();
        value[0] = L1.x * L_scale;
        value[1] = L1.y * L_scale;
        value[2] = L1.z * L_scale;
        value[3] = m_initial_box->getTiltFactorXY();
        value[4] = m_initial_box->getTiltFactorXZ();
        value[5] = m_initial_box->getTiltFactorYZ();
        return value;
        }

    std::shared_ptr<BoxDim> getInitialBox()
        {
        return m_initial_box;
        }

    void setInitialBox(std::shared_ptr<BoxDim> box)
        {
        m_initial_box = box;
        m_is2D = box->getL().z == 0;
        m_initial_volume = box->getVolume(m_is2D);
        }

    /// Set the starting time step.
    void setTStart(uint64_t t_start)
        {
        m_variant.setTStart(t_start);
        }

    /// Get the starting time step.
    uint64_t getTStart() const
        {
        return m_variant.getTStart();
        }

    /// Set the length of the ramp.
    void setTRamp(uint64_t t_ramp)
        {
        m_variant.setTRamp(t_ramp);
        }

    /// Get the length of the ramp.
    uint64_t getTRamp() const
        {
        return m_variant.getTRamp();
        }

    /// Set the final volume
    void setFinalVolume(Scalar volume)
        {
        m_final_volume = volume;
        }

    /// Get the final volume
    Scalar getFinalVolume() const
        {
        return m_final_volume;
        }

    protected:
    /// The starting box.
    std::shared_ptr<BoxDim> m_initial_box;

    /// The volume of box1.
    Scalar m_initial_volume;

    /// The volume of the box at the end of the ramp.
    Scalar m_final_volume;

    /// Whether box1 is 2-dimensional or not
    bool m_is2D;

    /// The current value of the volume
    Scalar m_current_volume;

    /// Variant for computing scale value
    VariantRamp m_variant; //!< Variant that interpolates between boxes
    };

namespace detail
    {
/// Export Variant classes to Python
void export_VectorVariantBox(pybind11::module& m);
void export_VectorVariantBoxConstant(pybind11::module& m);

    } // end namespace detail

    } // end namespace hoomd
