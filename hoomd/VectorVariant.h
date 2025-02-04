// Copyright (c) 2009-2024 The Regents of the University of Michigan.
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

/** Box vector variant.

    VectorVariant class for representing box parameters. The operator() returns an array with 6
   elements that represent Lx, Ly, Lz, xy, xz, and yz.
*/
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

/** Constant box vector variant

    Returns a constant vector.
    */
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

/** Interpolate box vector variant

    Vector variant that interpolates between two boxes based on a given scalar variant.
    Returns the vector corresponding to initial_box when the scalar variant evaluates to its minimum
   value. Returns the vector correspolding to final_box when the scalar variant evaluates to its
   maximum value. Returns the array corresponding to the interpolated box when the scalar variant
   evaluates to values between its maximum and minimum values. The i-th component of the
   interpolated box vector corresponds to the weighted average of the i-th components of initial_box
   and final_box, where the weight f given to final_box is equal to the difference in the value of
   the scalar variant and the minimum value of the scalar variant, normalized by the difference in
   the maximum and minimum values of the scalar variant. I.e., f = (variant(timestep) -
   variant.minimum) / (variant.maximum - variant.minimum).

*/
class PYBIND11_EXPORT VectorVariantBoxInterpolate : public VectorVariantBox
    {
    public:
    /** Construct a VectorVariantBoxInterpolate to interpolate between two boxes.

        @param initial_box The initial box
        @param final_box The final box
    */
    VectorVariantBoxInterpolate(std::shared_ptr<BoxDim> initial_box,
                                std::shared_ptr<BoxDim> final_box,
                                std::shared_ptr<Variant> variant)
        : m_initial_box(initial_box), m_final_box(final_box), m_variant(variant)
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

        const auto& initial_box = *m_initial_box;
        const auto& final_box = *m_final_box;
        Scalar3 new_L = final_box.getL() * scale + initial_box.getL() * (1.0 - scale);
        Scalar xy
            = final_box.getTiltFactorXY() * scale + (1.0 - scale) * initial_box.getTiltFactorXY();
        Scalar xz
            = final_box.getTiltFactorXZ() * scale + (1.0 - scale) * initial_box.getTiltFactorXZ();
        Scalar yz
            = final_box.getTiltFactorYZ() * scale + (1.0 - scale) * initial_box.getTiltFactorYZ();
        array_type value = {new_L.x, new_L.y, new_L.z, xy, xz, yz};
        return value;
        }

    std::shared_ptr<BoxDim> getInitialBox()
        {
        return m_initial_box;
        }

    void setInitialBox(std::shared_ptr<BoxDim> box)
        {
        m_initial_box = box;
        }

    std::shared_ptr<BoxDim> getFinalBox()
        {
        return m_final_box;
        }

    void setFinalBox(std::shared_ptr<BoxDim> box)
        {
        m_final_box = box;
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
    std::shared_ptr<BoxDim> m_initial_box;

    /// The final box, associated with the maximum of the variant.
    std::shared_ptr<BoxDim> m_final_box;

    /// Variant that interpolates between boxes.
    std::shared_ptr<Variant> m_variant;
    };

/** Inverse volume interpolation box vector variant.

    Returns the array corresponding to the box whose inverse volume (i.e., density) ramps from
   initial_box.volume to final_volume over t_ramp steps while keeping the box shape constant.
*/
class PYBIND11_EXPORT VectorVariantBoxInverseVolumeRamp : public VectorVariantBox
    {
    public:
    VectorVariantBoxInverseVolumeRamp(std::shared_ptr<BoxDim> initial_box,
                                      Scalar final_volume,
                                      uint64_t t_start,
                                      uint64_t t_ramp)
        : m_initial_box(initial_box), m_final_volume(final_volume), m_variant(0, 1, t_start, t_ramp)
        {
        m_is_2d = m_initial_box->getL().z == 0;
        m_initial_volume = m_initial_box->getVolume(m_is_2d);
        }

    virtual array_type operator()(uint64_t timestep)
        {
        Scalar s = m_variant(timestep);
        // current inverse volume = s * (1 / m_final_volume) + (1-s) * (1/m_vol1)
        // current volume = 1 / (current inverse volume)
        Scalar current_volume = 1 / (s / m_final_volume + (1.0 - s) / m_initial_volume);
        Scalar L_scale;
        if (m_is_2d)
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
        m_is_2d = box->getL().z == 0;
        m_initial_volume = box->getVolume(m_is_2d);
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

    /// The volume of the initial box.
    Scalar m_initial_volume;

    /// The volume of the box at the end of the ramp.
    Scalar m_final_volume;

    /// Whether initial_box is 2-dimensional or not
    bool m_is_2d;

    /// Variant for computing scale value
    VariantRamp m_variant;
    };

namespace detail
    {
/// Export Variant classes to Python
void export_VectorVariantBoxClasses(pybind11::module& m);

    } // end namespace detail

    } // end namespace hoomd
