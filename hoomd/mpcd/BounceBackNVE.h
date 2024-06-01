// Copyright (c) 2009-2024 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

/*!
 * \file BounceBackNVE.h
 * \brief Declares the BounceBackNVE class for doing NVE integration with bounce-back
 *        boundary conditions imposed by a geometry.
 */

#ifndef MPCD_BOUNCE_BACK_NVE_H_
#define MPCD_BOUNCE_BACK_NVE_H_

#ifdef __HIPCC__
#error This header cannot be compiled by nvcc
#endif

#include "hoomd/md/IntegrationMethodTwoStep.h"
#include <pybind11/pybind11.h>

namespace hoomd
    {
namespace mpcd
    {
//! Integrator that applies bounce-back boundary conditions in NVE.
/*!
 * This integrator applies "bounce-back" boundary conditions according to a template \a Geometry
 * class. Particles away from the boundary evolve according to the standard velocity Verlet
 * equations. When a particle moves to cross a boundary during the first Verlet step, its position
 * is restored to the boundary. The particle's tangential velocity is then reflected according to
 * the slip or no-slip condition, while the normal velocity is always reflected to maintain the
 * no-penetration condition. The particle velocity during this collision is the halfway point (after
 * the current acceleration has been applied). The final velocity step proceeds as usual for the
 * Verlet algorithm after the reflections are completed. This reflection procedure may induce a
 * small amount of slip near the surface from the acceleration.
 */
template<class Geometry>
class PYBIND11_EXPORT BounceBackNVE : public hoomd::md::IntegrationMethodTwoStep
    {
    public:
    //! Constructor
    BounceBackNVE(std::shared_ptr<SystemDefinition> sysdef,
                  std::shared_ptr<ParticleGroup> group,
                  std::shared_ptr<Geometry> geom);

    //! Destructor
    virtual ~BounceBackNVE();

    //! Performs the first step of the integration
    virtual void integrateStepOne(uint64_t timestep);

    //! Performs the second step of the integration
    virtual void integrateStepTwo(uint64_t timestep);

    //! Get the streaming geometry
    std::shared_ptr<Geometry> getGeometry()
        {
        return m_geom;
        }

    //! Set the streaming geometry
    void setGeometry(std::shared_ptr<Geometry> geom)
        {
        m_geom = geom;
        }

    //! Check that particles lie inside the geometry
    bool checkParticles();

    protected:
    std::shared_ptr<Geometry> m_geom; //!< Bounce-back geometry
    };

template<class Geometry>
BounceBackNVE<Geometry>::BounceBackNVE(std::shared_ptr<SystemDefinition> sysdef,
                                       std::shared_ptr<ParticleGroup> group,
                                       std::shared_ptr<Geometry> geom)
    : IntegrationMethodTwoStep(sysdef, group), m_geom(geom)
    {
    m_exec_conf->msg->notice(5) << "Constructing BounceBackNVE + " << Geometry::getName()
                                << std::endl;
    }

template<class Geometry> BounceBackNVE<Geometry>::~BounceBackNVE()
    {
    m_exec_conf->msg->notice(5) << "Destroying BounceBackNVE + " << Geometry::getName()
                                << std::endl;
    }

template<class Geometry> void BounceBackNVE<Geometry>::integrateStepOne(uint64_t timestep)
    {
    // particle data
    ArrayHandle<Scalar4> h_pos(m_pdata->getPositions(),
                               access_location::host,
                               access_mode::readwrite);
    ArrayHandle<int3> h_image(m_pdata->getImages(), access_location::host, access_mode::readwrite);
    ArrayHandle<Scalar4> h_vel(m_pdata->getVelocities(),
                               access_location::host,
                               access_mode::readwrite);
    ArrayHandle<Scalar3> h_accel(m_pdata->getAccelerations(),
                                 access_location::host,
                                 access_mode::read);
    const BoxDim box = m_pdata->getBox();

    // group members
    const unsigned int group_size = m_group->getNumMembers();
    ArrayHandle<unsigned int> h_group(m_group->getIndexArray(),
                                      access_location::host,
                                      access_mode::read);

    for (unsigned int idx = 0; idx < group_size; ++idx)
        {
        const unsigned int pid = h_group.data[idx];

        // load velocity + mass
        const Scalar4 velmass = h_vel.data[pid];
        Scalar3 vel = make_scalar3(velmass.x, velmass.y, velmass.z);
        const Scalar mass = velmass.w;

        // update velocity first according to verlet step
        const Scalar3 accel = h_accel.data[pid];
        vel += Scalar(0.5) * m_deltaT * accel;

        // load position and type
        const Scalar4 postype = h_pos.data[pid];
        Scalar3 pos = make_scalar3(postype.x, postype.y, postype.z);
        const Scalar type = postype.w;

        // update position while bouncing-back velocity
        Scalar dt_remain = m_deltaT;
        bool collide = false;
        do
            {
            pos += dt_remain * vel;
            collide = m_geom->detectCollision(pos, vel, dt_remain);
            } while (dt_remain > 0 && collide);

        // wrap final position
        box.wrap(pos, h_image.data[pid]);

        // write position and velocity back out
        h_pos.data[pid] = make_scalar4(pos.x, pos.y, pos.z, type);
        h_vel.data[pid] = make_scalar4(vel.x, vel.y, vel.z, mass);
        }
    }

template<class Geometry> void BounceBackNVE<Geometry>::integrateStepTwo(uint64_t timestep)
    {
    ArrayHandle<Scalar4> h_vel(m_pdata->getVelocities(),
                               access_location::host,
                               access_mode::readwrite);
    ArrayHandle<Scalar3> h_accel(m_pdata->getAccelerations(),
                                 access_location::host,
                                 access_mode::readwrite);
    ArrayHandle<Scalar4> h_net_force(m_pdata->getNetForce(),
                                     access_location::host,
                                     access_mode::read);

    const unsigned int group_size = m_group->getNumMembers();
    ArrayHandle<unsigned int> h_group(m_group->getIndexArray(),
                                      access_location::host,
                                      access_mode::read);

    for (unsigned int idx = 0; idx < group_size; ++idx)
        {
        const unsigned int pid = h_group.data[idx];

        // load net force and velocity, compute a = F / m
        const Scalar4 net_force = h_net_force.data[pid];
        Scalar3 accel = make_scalar3(net_force.x, net_force.y, net_force.z);
        Scalar4 vel = h_vel.data[pid];
        accel.x /= vel.w;
        accel.y /= vel.w;
        accel.z /= vel.w;

        // then, update the velocity
        vel.x += Scalar(0.5) * accel.x * m_deltaT;
        vel.y += Scalar(0.5) * accel.y * m_deltaT;
        vel.z += Scalar(0.5) * accel.z * m_deltaT;

        h_vel.data[pid] = vel;
        h_accel.data[pid] = accel;
        }
    }

/*!
 * Checks each particle position to determine if it lies within the geometry. If any particle is
 * out of bounds, an error is raised.
 */
template<class Geometry> bool BounceBackNVE<Geometry>::checkParticles()
    {
    ArrayHandle<Scalar4> h_pos(m_pdata->getPositions(), access_location::host, access_mode::read);
    ArrayHandle<unsigned int> h_tag(m_pdata->getTags(), access_location::host, access_mode::read);

    ArrayHandle<unsigned int> h_group(m_group->getIndexArray(),
                                      access_location::host,
                                      access_mode::read);
    const unsigned int group_size = m_group->getNumMembers();

    bool out_of_bounds = false;
    for (unsigned int idx = 0; idx < group_size; ++idx)
        {
        const unsigned int pid = h_group.data[idx];

        const Scalar4 postype = h_pos.data[pid];
        const Scalar3 pos = make_scalar3(postype.x, postype.y, postype.z);
        if (m_geom->isOutside(pos))
            {
            out_of_bounds = true;
            break;
            }
        }

#ifdef ENABLE_MPI
    if (m_exec_conf->getNRanks() > 1)
        {
        MPI_Allreduce(MPI_IN_PLACE,
                      &out_of_bounds,
                      1,
                      MPI_CXX_BOOL,
                      MPI_LOR,
                      m_exec_conf->getMPICommunicator());
        }
#endif // ENABLE_MPI

    return !out_of_bounds;
    }

namespace detail
    {
//! Exports the BounceBackNVE class to python
template<class Geometry> void export_BounceBackNVE(pybind11::module& m)
    {
    const std::string name = "BounceBackNVE" + Geometry::getName();

    pybind11::class_<BounceBackNVE<Geometry>,
                     hoomd::md::IntegrationMethodTwoStep,
                     std::shared_ptr<BounceBackNVE<Geometry>>>(m, name.c_str())
        .def(pybind11::init<std::shared_ptr<SystemDefinition>,
                            std::shared_ptr<ParticleGroup>,
                            std::shared_ptr<Geometry>>())
        .def_property("geometry",
                      &BounceBackNVE<Geometry>::getGeometry,
                      &BounceBackNVE<Geometry>::setGeometry)
        .def("check_particles", &BounceBackNVE<Geometry>::checkParticles);
    }
    } // end namespace detail
    } // end namespace mpcd
    } // end namespace hoomd
#endif // #ifndef MPCD_BOUNCE_BACK_NVE_H_
