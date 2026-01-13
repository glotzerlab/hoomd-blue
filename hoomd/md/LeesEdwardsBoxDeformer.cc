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
 */
LeesEdwardsBoxDeformer::LeesEdwardsBoxDeformer(std::shared_ptr<SystemDefinition> sysdef,
                                               Scalar xy_rate,
                                               Scalar max_xy_tilt)
    : BoxDeformer(sysdef), m_xy_rate(xy_rate), m_max_xy_tilt(max_xy_tilt)
    {
    m_exec_conf->msg->notice(5) << "Constructing LeesEdwardsBoxDeformer" << std::endl;
    }

LeesEdwardsBoxDeformer::~LeesEdwardsBoxDeformer()
    {
    m_exec_conf->msg->notice(5) << "Destroying LeesEdwardsBoxDeformer" << std::endl;
    }

BoxDim LeesEdwardsBoxDeformer::computeNewBox(uint64_t timestep, const BoxDim& old_box)
    {
    // Get the tilt factors
    Scalar xy = old_box.getTiltFactorXY();
    const Scalar xz = old_box.getTiltFactorXZ();
    const Scalar yz = old_box.getTiltFactorYZ();

    // Update xy tilt factor using stored rate and deltaT
    xy += m_xy_rate * m_deltaT;

    // Return updated box
    BoxDim new_box = old_box;
    new_box.setTiltFactors(xy, xz, yz);
    new_box.setTiltDeformationRates(m_xy_rate, 0.0, 0.0);

    return new_box;
    }

void LeesEdwardsBoxDeformer::processAfterDeformation(const BoxDim& old_box, BoxDim& new_box)
    {
    // Call base class to perform default PBC wrapping
    BoxDeformer::processAfterDeformation(old_box, new_box);

    // Extra processing: box flipping and particle remapping
    int flip = static_cast<int>(std::floor(old_box.getTiltFactorXY() + m_max_xy_tilt));

    if (flip != 0)
        {
        // Get and update tilt
        Scalar xy = old_box.getTiltFactorXY();
        const Scalar xz = old_box.getTiltFactorXZ();
        const Scalar yz = old_box.getTiltFactorYZ();

        xy -= Scalar(flip);
        new_box.setTiltFactors(xy, xz, yz);

        // Remap particle positions
        ArrayHandle<Scalar4> h_pos(m_pdata->getPositions(),
                                   access_location::host,
                                   access_mode::readwrite);

        Scalar Ly = new_box.getL().y;
        for (unsigned int i = 0; i < m_pdata->getN(); i++)
            {
            h_pos.data[i].x -= Scalar(flip) * Ly;
            }
        }
    }

namespace detail
    {
void export_LeesEdwardsBoxDeformer(pybind11::module& m)
    {
    pybind11::class_<LeesEdwardsBoxDeformer, BoxDeformer, std::shared_ptr<LeesEdwardsBoxDeformer>>(
        m,
        "LeesEdwardsBoxDeformer")
        .def(pybind11::init<std::shared_ptr<SystemDefinition>, Scalar, Scalar>())
        .def_property("shear_rate",
                      &LeesEdwardsBoxDeformer::getShearRate,
                      &LeesEdwardsBoxDeformer::setShearRate)
        .def_property("max_xy_tilt",
                      &LeesEdwardsBoxDeformer::getMaxXYTilt,
                      &LeesEdwardsBoxDeformer::setMaxXYTilt);
    }

    } // end namespace detail
    } // end namespace md
    } // end namespace hoomd
