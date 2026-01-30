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
    // Get the tilt factors (xz, yz are unchanged but needed to reset the box)
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
    // Box flipping and particle remapping
    Scalar xy = new_box.getTiltFactorXY();
    int flip = static_cast<int>(std::floor((xy + m_max_xy_tilt) / (2.0 * m_max_xy_tilt)));

    if (flip != 0)
        {
        // Update xy tilt and get L
        xy -= 2.0 * m_max_xy_tilt * Scalar(flip);
        Scalar Ly = new_box.getL().y;

        // Remap particle coordinates
#ifdef ENABLE_MPI
        SnapshotParticleData<Scalar> snap;

        // Take a snapshot on rank 0
        m_pdata->takeSnapshot(snap);

        if (m_exec_conf->getRank() == 0)
            {
            for (unsigned int i = 0; i < snap.size; i++)
                {
                snap.pos[i].x -= Scalar(flip) * Ly;
                //  Immediately wrap into box to avoid temporal out-of-bounds errors on MPI
                new_box.wrap(snap.pos[i], snap.vel[i], snap.image[i]);
                }
            }
        // Broadcast updates to all ranks
        bcast(snap.pos, 0, m_exec_conf->getMPICommunicator());

        m_pdata->initializeFromSnapshot(snap);
#else
        // Serial execution
        ArrayHandle<Scalar4> h_pos(m_pdata->getPositions(),
                                   access_location::host,
                                   access_mode::readwrite);
        ArrayHandle<int3> h_image(m_pdata->getImages(),
                                  access_location::host,
                                  access_mode::readwrite);

        for (unsigned int i = 0; i < m_pdata->getN(); i++)
            {
            h_pos.data[i].x -= Scalar(flip) * Ly;
            }
#endif
        // Reset the box with updated xy tilt (and the unchanged xz, yz)
        const Scalar xz = new_box.getTiltFactorXZ();
        const Scalar yz = new_box.getTiltFactorYZ();
        new_box.setTiltFactors(xy, xz, yz);

        m_pdata->setGlobalBox(new_box);
        }

    // Call base class to perform default PBC wrapping
    BoxDeformer::processAfterDeformation(old_box, new_box);
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
