// Copyright (c) 2009-2024 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

/*!
 * \file mpcd/BounceBackStreamingMethod.h
 * \brief Declaration of mpcd::BounceBackStreamingMethod
 */

#ifndef MPCD_CONFINED_STREAMING_METHOD_H_
#define MPCD_CONFINED_STREAMING_METHOD_H_

#ifdef __HIPCC__
#error This header cannot be compiled by nvcc
#endif

#include "StreamingMethod.h"
#include <pybind11/pybind11.h>

namespace hoomd
    {
namespace mpcd
    {
//! MPCD confined streaming method
/*!
 * This method implements the base version of ballistic propagation of MPCD
 * particles in confined geometries.
 *
 * \tparam Geometry The confining geometry.
 * \tparam Force The solvent force.
 *
 * The integration scheme is essentially Verlet with specular reflections. The particle is streamed
 * forward over the time interval. If it moves outside the Geometry, it is placed back on the
 * boundary and its velocity is updated according to the boundary conditions. Streaming then
 * continues until the timestep is completed.
 *
 * To facilitate this, every Geometry must supply two methods:
 *  1. detectCollision(): Determines when and where a collision occurs. If one does, this method
 * moves the particle back, reflects its velocity, and gives the time still remaining to integrate.
 *  2. isOutside(): Determines whether a particles lies outside the Geometry.
 *
 */
template<class Geometry, class Force>
class PYBIND11_EXPORT BounceBackStreamingMethod : public mpcd::StreamingMethod
    {
    public:
    //! Constructor
    /*!
     * \param sysdef System definition
     * \param cur_timestep Current system timestep
     * \param period Number of timesteps between collisions
     * \param phase Phase shift for periodic updates
     * \param geom Streaming geometry
     * \param force Solvent force
     */
    BounceBackStreamingMethod(std::shared_ptr<SystemDefinition> sysdef,
                              unsigned int cur_timestep,
                              unsigned int period,
                              int phase,
                              std::shared_ptr<Geometry> geom,
                              std::shared_ptr<Force> force)
        : mpcd::StreamingMethod(sysdef, cur_timestep, period, phase), m_geom(geom), m_force(force)
        {
        }

    //! Implementation of the streaming rule
    virtual void stream(uint64_t timestep);

    //! Get the streaming geometry
    std::shared_ptr<Geometry> getGeometry() const
        {
        return m_geom;
        }

    //! Set the streaming geometry
    void setGeometry(std::shared_ptr<Geometry> geom)
        {
        m_geom = geom;
        }

    //! Set the solvent force
    std::shared_ptr<Force> getForce() const
        {
        return m_force;
        }

    //! Get the solvent force
    void setForce(std::shared_ptr<Force> force)
        {
        m_force = force;
        }

    //! Check that particles lie inside the geometry
    virtual bool checkParticles();

    protected:
    std::shared_ptr<Geometry> m_geom; //!< Streaming geometry
    std::shared_ptr<Force> m_force;   //!< Solvent force
    };

/*!
 * \param timestep Current time to stream
 */
template<class Geometry, class Force>
void BounceBackStreamingMethod<Geometry, Force>::stream(uint64_t timestep)
    {
    if (!shouldStream(timestep))
        return;

    if (!m_cl)
        {
        throw std::runtime_error("Cell list has not been set");
        }

    const BoxDim box = m_cl->getCoverageBox();

    ArrayHandle<Scalar4> h_pos(m_mpcd_pdata->getPositions(),
                               access_location::host,
                               access_mode::readwrite);
    ArrayHandle<Scalar4> h_vel(m_mpcd_pdata->getVelocities(),
                               access_location::host,
                               access_mode::readwrite);
    const Scalar mass = m_mpcd_pdata->getMass();

    // default construct a force if one is not set
    const Force force = (m_force) ? *m_force : Force();

    for (unsigned int cur_p = 0; cur_p < m_mpcd_pdata->getN(); ++cur_p)
        {
        const Scalar4 postype = h_pos.data[cur_p];
        Scalar3 pos = make_scalar3(postype.x, postype.y, postype.z);
        const unsigned int type = __scalar_as_int(postype.w);

        const Scalar4 vel_cell = h_vel.data[cur_p];
        Scalar3 vel = make_scalar3(vel_cell.x, vel_cell.y, vel_cell.z);
        // estimate next velocity based on current acceleration
        vel += Scalar(0.5) * m_mpcd_dt * force.evaluate(pos) / mass;

        // propagate the particle to its new position ballistically
        Scalar dt_remain = m_mpcd_dt;
        bool collide = true;
        do
            {
            pos += dt_remain * vel;
            collide = m_geom->detectCollision(pos, vel, dt_remain);
            } while (dt_remain > 0 && collide);
        // finalize velocity update
        vel += Scalar(0.5) * m_mpcd_dt * force.evaluate(pos) / mass;

        // wrap and update the position
        int3 image = make_int3(0, 0, 0);
        box.wrap(pos, image);

        h_pos.data[cur_p] = make_scalar4(pos.x, pos.y, pos.z, __int_as_scalar(type));
        h_vel.data[cur_p]
            = make_scalar4(vel.x, vel.y, vel.z, __int_as_scalar(mpcd::detail::NO_CELL));
        }

    // particles have moved, so the cell cache is no longer valid
    m_mpcd_pdata->invalidateCellCache();
    }

/*!
 * Checks each MPCD particle position to determine if it lies within the geometry. If any particle
 * is out of bounds, an error is raised.
 */
template<class Geometry, class Force>
bool BounceBackStreamingMethod<Geometry, Force>::checkParticles()
    {
    ArrayHandle<Scalar4> h_pos(m_mpcd_pdata->getPositions(),
                               access_location::host,
                               access_mode::read);
    ArrayHandle<unsigned int> h_tag(m_mpcd_pdata->getTags(),
                                    access_location::host,
                                    access_mode::read);

    bool out_of_bounds = false;
    for (unsigned int idx = 0; idx < m_mpcd_pdata->getN(); ++idx)
        {
        const Scalar4 postype = h_pos.data[idx];
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
//! Export mpcd::StreamingMethod to python
/*!
 * \param m Python module to export to
 */
template<class Geometry, class Force> void export_BounceBackStreamingMethod(pybind11::module& m)
    {
    const std::string name = "BounceBackStreamingMethod" + Geometry::getName() + Force::getName();
    pybind11::class_<mpcd::BounceBackStreamingMethod<Geometry, Force>,
                     mpcd::StreamingMethod,
                     std::shared_ptr<mpcd::BounceBackStreamingMethod<Geometry, Force>>>(
        m,
        name.c_str())
        .def(pybind11::init<std::shared_ptr<SystemDefinition>,
                            unsigned int,
                            unsigned int,
                            int,
                            std::shared_ptr<Geometry>,
                            std::shared_ptr<Force>>())
        .def_property_readonly("geometry",
                               &mpcd::BounceBackStreamingMethod<Geometry, Force>::getGeometry)
        .def_property_readonly("mpcd_particle_force",
                               &mpcd::BounceBackStreamingMethod<Geometry, Force>::getForce)
        .def("check_mpcd_particles", &BounceBackStreamingMethod<Geometry, Force>::checkParticles);
    }
    } // end namespace detail
    } // end namespace mpcd
    } // end namespace hoomd
#endif // MPCD_CONFINED_STREAMING_METHOD_H_
