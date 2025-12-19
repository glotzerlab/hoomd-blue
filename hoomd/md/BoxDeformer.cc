// Copyright (c) 2009-2025 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

/*! \file BoxDeformer.cc
    \brief Declares the BoxDeformer class
*/

#include "BoxDeformer.h"

using namespace std;

namespace hoomd
    {
namespace md
    {
/** @param sysdef System definition containing the particle data this method acts on
    @param deltaT Time step
*/
BoxDeformer::BoxDeformer(std::shared_ptr<SystemDefinition> sysdef, Scalar deltaT)
    : m_sysdef(sysdef), m_pdata(sysdef->getParticleData()), m_deltaT(deltaT)
    {
    assert(m_pdata);
    }

BoxDeformer::~BoxDeformer() { }

void BoxDeformer::setDeltaT(Scalar deltaT)
    {
    if (deltaT < 0.0)
        throw std::domain_error("delta_t must be positive");
    m_deltaT = deltaT;
    }

void BoxDeformer::update(uint64_t timestep)
    {
    // Get the current box
    BoxDim old_box = m_pdata->getGlobalBox();

    // Compute new box (child class will determine the new box geometry)
    BoxDim new_box = computeNewBox(timestep);

    if (new_box != old_box)
        {
        // Set the new global box
        m_pdata->setGlobalBox(new_box);

        // Apply any post-deformation processing
        postDeformationProcessing(old_box, new_box);
        }
    }

// Post deformation particle processing: PBC wrapping by default but child classes can add up
void BoxDeformer::postDeformationProcessing(const BoxDim& old_box, BoxDim& new_box)
    {
    ArrayHandle<Scalar4> h_pos(m_pdata->getPositions(),
                               access_location::host,
                               access_mode::readwrite);
    ArrayHandle<Scalar4> h_vel(m_pdata->getVelocities(),
                               access_location::host,
                               access_mode::readwrite);
    ArrayHandle<int3> h_image(m_pdata->getImages(), access_location::host, access_mode::readwrite);

    for (unsigned int i = 0; i < m_pdata->getN(); i++)
        {
        new_box.wrap(h_pos.data[i], h_vel.data[i], h_image.data[i]);
        }
    }

namespace detail
    {
void export_BoxDeformer(pybind11::module& m)
    {
    pybind11::class_<BoxDeformer, std::shared_ptr<BoxDeformer>>(m, "BoxDeformer")
        .def(pybind11::init<std::shared_ptr<SystemDefinition>, Scalar>());
    }

    } // end namespace detail
    } // end namespace md
    } // end namespace hoomd
