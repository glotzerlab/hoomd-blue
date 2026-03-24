// Copyright (c) 2009-2025 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

/*! \file md/LeesEdwardsBoxDeformer.h
    \brief Declaration of Lees–Edwards box deformer for triclinic boxes
*/

#ifdef __HIPCC__
#error This header cannot be compiled by nvcc
#endif

#ifndef __LEES_EDWARDS_BOX_DEFORMER_H__
#define __LEES_EDWARDS_BOX_DEFORMER_H__

#include "BoxDeformer.h"
#include "hoomd/Autotuner.h"

namespace hoomd
    {
namespace md
    {
/*
    Lees-Edwards deformation. Box flipping and particle remapping performed
    when tilt exceeds a threshold.
*/
class PYBIND11_EXPORT LeesEdwardsBoxDeformer : public BoxDeformer
    {
    public:
    LeesEdwardsBoxDeformer(std::shared_ptr<SystemDefinition> sysdef,
                           Scalar xy_rate,
                           Scalar max_xy_tilt);

    virtual ~LeesEdwardsBoxDeformer();

    /// Get the shear rate (in xy)
    Scalar getShearRate()
        {
        return m_xy_rate;
        }

    /// Set the shear rate (in xy)
    void setShearRate(const Scalar xy_rate)
        {
        m_xy_rate = xy_rate;
        }

    /// Get the maximum tilt in xy before remapping
    Scalar getMaxXYTilt()
        {
        return m_max_xy_tilt;
        }

    /// Set the maximum tilt in xy before remapping
    void setMaxXYTilt(const Scalar max_xy_tilt)
        {
        if (max_xy_tilt < 0.0)
            {
            throw std::invalid_argument("max_xy_tilt must be non-negative");
            }
        m_max_xy_tilt = max_xy_tilt;
        }

    protected:
    Scalar m_xy_rate;     //!< shear rate, d(xy)/dt
    Scalar m_max_xy_tilt; //!< maximum tilt in xy before remapping

    /// Compute the new box based on the shear rate
    BoxDim computeNewBox(uint64_t timestep, const BoxDim& old_box) override;

    /// Box flip and particle remapping (called after default PBC wrapping)
    void processAfterDeformation(const BoxDim& old_box, const BoxDim& new_box) override;

#ifdef ENABLE_HIP
    private:
    std::shared_ptr<Autotuner<1>> m_tuner_remap; //!< Autotuner for block size
#endif
    };

    } // end namespace md
    } // end namespace hoomd

#endif
