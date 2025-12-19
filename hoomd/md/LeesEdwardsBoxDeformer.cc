// Copyright (c) 2009-2025 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

/*! \file LeesEdwardsBoxDeformer.cc
    \brief Lees–Edwards box deformer for triclinic boxes.
*/

#include "LeesEdwardsBoxDeformer.h"

namespace hoomd
    {
namespace md
    {
/** @param sysdef System definition containing the particle data this method acts on
    @param deltaT Time step
*/
LeesEdwardsBoxDeformer::LeesEdwardsBoxDeformer(std::shared_ptr<SystemDefinition> sysdef,
                                               Scalar deltaT)
    : BoxDeformer(sysdef, deltaT), m_new_box(m_pdata->getGlobalBox()),
      m_xy(m_new_box.getTiltFactorXY()), m_xz(m_new_box.getTiltFactorXZ()),
      m_yz(m_new_box.getTiltFactorYZ()), m_xy_rate(m_new_box.getTiltDeformationRateXY())
    {
    }

LeesEdwardsBoxDeformer::~LeesEdwardsBoxDeformer() { }

BoxDim LeesEdwardsBoxDeformer::computeNewBox(uint64_t timestep)
    {
    // Update tilt factor using stored rate and deltaT
    m_xy += m_xy_rate * m_deltaT;

    // Return updated BoxDim
    m_new_box.setTiltFactors(m_xy, m_xz, m_yz);
    m_new_box.setTiltDeformationRates(m_xy_rate, 0.0, 0.0);

    return m_new_box;
    }

void LeesEdwardsBoxDeformer::postDeformationProcessing(const BoxDim& old_box, BoxDim& new_box)
    {
    // Call base class to perform default PBC wrapping
    BoxDeformer::postDeformationProcessing(old_box, new_box);

    // Extra processing: box flipping and particle remapping
    int flip = static_cast<int>(std::floor(m_xy + Scalar(0.5)));

    if (flip != 0)
        {
        // Update stored tilt
        m_xy -= Scalar(flip);
        new_box.setTiltFactors(m_xy, m_xz, m_yz);

        // Remap particle positions
        ArrayHandle<Scalar4> h_pos(m_pdata->getPositions(),
                                   access_location::host,
                                   access_mode::readwrite);

        Scalar Ly = new_box.getL().y;
        for (unsigned int i = 0; i < m_pdata->getN(); i++)
            {
            h_pos.data[i].x -= Scalar(flip) * Ly;
            }

        m_new_box = new_box;
        }
    }

namespace detail
    {
void export_LeesEdwardsBoxDeformer(pybind11::module& m)
    {
    pybind11::class_<LeesEdwardsBoxDeformer, BoxDeformer, std::shared_ptr<LeesEdwardsBoxDeformer>>(
        m,
        "LeesEdwardsBoxDeformer")
        .def(pybind11::init<std::shared_ptr<SystemDefinition>, Scalar>())
        .def_property("shear_rate",
                      &LeesEdwardsBoxDeformer::getShearRate,
                      &LeesEdwardsBoxDeformer::setShearRate);
    }

    } // end namespace detail
    } // end namespace md
    } // end namespace hoomd
