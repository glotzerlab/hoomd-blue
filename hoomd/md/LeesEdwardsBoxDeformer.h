// Copyright (c) 2009-2025 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

/*! \file LeesEdwardsBoxDeformer.h
    \brief Lees–Edwards box deformer for triclinic boxes.
*/

#ifdef __HIPCC__
#error This header cannot be compiled by nvcc
#endif

#ifndef __LEES_EDWARDS_BOX_DEFORMER_H__
#define __LEES_EDWARDS_BOX_DEFORMER_H__

#include "BoxDeformer.h"

namespace hoomd
    {
namespace md
    {
/*
    Lees-Edwards deformation. Box flipping and particle remapping performed
    when tilt exceeds 0.5.
*/
class PYBIND11_EXPORT LeesEdwardsBoxDeformer : public BoxDeformer
    {
    public:
    LeesEdwardsBoxDeformer(std::shared_ptr<SystemDefinition> sysdef, Scalar deltaT);

    virtual ~LeesEdwardsBoxDeformer();

    /// Set the shear rate (in xy)
    virtual void setShearRate(const Scalar xy_rate)
        {
        m_xy_rate = xy_rate;
        }

    /// Get the shear rate (in xy)
    Scalar getShearRate()
        {
        return m_xy_rate;
        }

    protected:
    BoxDim m_new_box;  //!< box object
    Scalar m_xy;       //!< xy tilt
    const Scalar m_xz; //!< xz tilt
    const Scalar m_yz; //!< yz tilt
    Scalar m_xy_rate;  //!< shear rate, d(xy)/dt

    /// Compute the new box based on the shear rate
    BoxDim computeNewBox(uint64_t timestep) override;

    /// Box flip and particle remapping (called after default PBC wrapping)
    void postDeformationProcessing(const BoxDim& old_box, BoxDim& new_box) override;
    };

namespace detail
    {
/// Export LeesEdwardsBoxDeformer to python
void export_LeesEdwardsBoxDeformer(pybind11::module& m);
    } // end namespace detail
    } // end namespace md
    } // end namespace hoomd

#endif
