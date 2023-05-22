// Copyright (c) 2009-2023 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

/*! \file SphereResizeUpdater.cc
    \brief Defines the SphereResizeUpdater class
*/

#include "SphereResizeUpdater.h"

#include <iostream>
#include <math.h>
#include <stdexcept>

using namespace std;

namespace hoomd
    {
/*! \param sysdef System definition containing the particle data to set the sphere size on
    \param R length of R over time

    The default setting is to scale particle positions along with the sphere. (Gabby) this should be 
    changed in the sphere version, because it's a closed system and they'll never be on the edges
*/

SphereResizeUpdater::SphereResizeUpdater(std::shared_ptr<SystemDefinition> sysdef,
                                   std::shared_ptr<Trigger> trigger,
                                   std::shared_ptr<Sphere> sphere1,
                                   std::shared_ptr<Sphere> sphere2,
                                   std::shared_ptr<Variant> variant,
                                   std::shared_ptr<ParticleGroup> group)
    : Updater(sysdef, trigger), m_sphere1(sphere1), m_sphere2(sphere2), m_variant(variant), m_group(group)
    {
    assert(m_pdata);
    assert(m_variant);
    m_exec_conf->msg->notice(5) << "Constructing SphereResizeUpdater" << endl;
    }

SphereResizeUpdater::~SphereResizeUpdater()
    {
    m_exec_conf->msg->notice(5) << "Destroying SphereResizeUpdater" << endl;
    }

/// Get sphere1
std::shared_ptr<Sphere> SphereResizeUpdater::getSphere1()
    {
    return m_sphere1;
    }

/// Set a new sphere1
void SphereResizeUpdater::setSphere1(std::shared_ptr<Sphere> sphere1)
    {
    m_sphere1 = sphere1;
    }

/// Get sphere2
std::shared_ptr<Sphere> SphereResizeUpdater::getSphere2()
    {
    return m_sphere2;
    }

void SphereResizeUpdater::setSphere2(std::shared_ptr<Sphere> sphere2)
    {
    m_sphere2 = sphere2;
    }

/// Get the current sphere based on the timestep
Sphere SphereResizeUpdater::getCurrentSphere(uint64_t timestep)
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

    const auto& sphere1 = *m_sphere1;
    const auto& sphere2 = *m_sphere2;
    Scalar new_R = sphere2.getR() * scale + sphere1.getR() * (1.0 - scale);

    Sphere new_sphere = Sphere(new_R);
    return new_sphere;
    }

/** Perform the needed calculations to scale the sphere size
    \param timestep Current time step of the simulation
*/
void SphereResizeUpdater::update(uint64_t timestep)
    {
    Updater::update(timestep);
    m_exec_conf->msg->notice(10) << "Sphere resize update" << endl;

    // first, compute the new sphere
    Sphere new_sphere = getCurrentSphere(timestep);

    // check if the current sphere size is the same
    Sphere cur_sphere = m_pdata->getSphere();

    // only change the sphere if there is a change in the sphere dimensions
    if (new_sphere != cur_sphere)
        {
        // set the new sphere
        m_pdata->setSphere(new_sphere);

        // scale the particle positions (if we have been asked to)
        // move the particles to be inside the new sphere
        //(Gabby) The particles aren't going to be outside of the new sphere, it's closed
        //scaleAndWrapParticles(cur_sphere, new_sphere);
        }
    }

/// Scale particles to the new sphere and wrap any others back into the sphere
/*
void SphereResizeUpdater::scaleAndWrapParticles(const Sphere& cur_sphere, const Sphere& new_sphere)
    {
    ArrayHandle<Scalar4> h_pos(m_pdata->getPositions(),
                               access_location::host,
                               access_mode::readwrite);

    for (unsigned int group_idx = 0; group_idx < m_group->getNumMembers(); group_idx++)
        {
        unsigned int j = m_group->getMemberIndex(group_idx);
        // obtain scaled coordinates in the old global sphere
        Scalar3 fractional_pos
            = cur_sphere.makeFraction(make_scalar3(h_pos.data[j].x, h_pos.data[j].y, h_pos.data[j].z));

        // intentionally scale both rigid body and free particles, this
        // may waste a few cycles but it enables the debug inSphere checks
        // to be left as is (otherwise, setRV cannot fixup rigid body
        // positions without failing the check)
        Scalar3 scaled_pos = new_sphere.makeCoordinates(fractional_pos);
        h_pos.data[j].x = scaled_pos.x;
        h_pos.data[j].y = scaled_pos.y;
        h_pos.data[j].z = scaled_pos.z;
        }

    // ensure that the particles are still in their
    // local spheres by wrapping them if they are not
    ArrayHandle<int3> h_image(m_pdata->getImages(), access_location::host, access_mode::readwrite);

    const Sphere& local_sphere = m_pdata->getSphere();

    for (unsigned int i = 0; i < m_pdata->getN(); i++)
        {
        // need to update the image if we move particles from one side
        // of the sphere to the other
        local_sphere.wrap(h_pos.data[i], h_image.data[i]);
        }
    }
*/
namespace detail
    {
void export_SphereResizeUpdater(pybind11::module& m)
    {
    pybind11::class_<SphereResizeUpdater, Updater, std::shared_ptr<SphereResizeUpdater>>(
        m,
        "SphereResizeUpdater")
        .def(pybind11::init<std::shared_ptr<SystemDefinition>,
                            std::shared_ptr<Trigger>,
                            std::shared_ptr<Sphere>,
                            std::shared_ptr<Sphere>,
                            std::shared_ptr<Variant>,
                            std::shared_ptr<ParticleGroup>>())
        .def_property("sphere1", &SphereResizeUpdater::getSphere1, &SphereResizeUpdater::setSphere1)
        .def_property("sphere2", &SphereResizeUpdater::getSphere2, &SphereResizeUpdater::setSphere2)
        .def_property("variant", &SphereResizeUpdater::getVariant, &SphereResizeUpdater::setVariant)
        .def_property_readonly("filter",
                               [](const std::shared_ptr<SphereResizeUpdater> method)
                               { return method->getGroup()->getFilter(); })
        .def("get_current_sphere", &SphereResizeUpdater::getCurrentSphere);
    }

    } // end namespace detail

    } // end namespace hoomd
