// Copyright (c) 2009-2019 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.

// Maintainer: mphoward

/*!
 * \file mpcd/ConfinedStreamingMethod.h
 * \brief Declaration of mpcd::ConfinedStreamingMethod
 */

#ifndef MPCD_CONFINED_STREAMING_METHOD_H_
#define MPCD_CONFINED_STREAMING_METHOD_H_

#ifdef NVCC
#error This header cannot be compiled by nvcc
#endif

#include "StreamingMethod.h"
#include "hoomd/extern/pybind/include/pybind11/pybind11.h"

namespace mpcd
{

//! MPCD confined streaming method
/*!
 * This method implements the base version of ballistic propagation of MPCD
 * particles in confined geometries.
 *
 * \tparam Geometry The confining geometry (e.g., BulkGeometry, SlitGeometry).
 *
 * The integration scheme is essentially Verlet with specular reflections. The particle is streamed forward over
 * the time interval. If it moves outside the Geometry, it is placed back on the boundary and its velocity is
 * updated according to the boundary conditions. Streaming then continues until the timestep is completed.
 *
 * To facilitate this, every Geometry must supply three methods:
 *  1. detectCollision(): Determines when and where a collision occurs. If one does, this method moves the
 *                        particle back, reflects its velocity, and gives the time still remaining to integrate.
 *  2. isOutside(): Determines whether a particles lies outside the Geometry.
 *  3. validateBox(): Checks whether the global simulation box is consistent with the streaming geometry.
 *
 */
template<class Geometry>
class PYBIND11_EXPORT ConfinedStreamingMethod : public mpcd::StreamingMethod
    {
    public:
        //! Constructor
        /*!
         * \param sysdata MPCD system data
         * \param cur_timestep Current system timestep
         * \param period Number of timesteps between collisions
         * \param phase Phase shift for periodic updates
         * \param geom Streaming geometry
         */
        ConfinedStreamingMethod(std::shared_ptr<mpcd::SystemData> sysdata,
                                unsigned int cur_timestep,
                                unsigned int period,
                                int phase,
                                std::shared_ptr<const Geometry> geom)
        : mpcd::StreamingMethod(sysdata, cur_timestep, period, phase),
          m_geom(geom), m_validate_geom(true)
          {}

        //! Implementation of the streaming rule
        virtual void stream(unsigned int timestep);

        //! Get the streaming geometry
        std::shared_ptr<const Geometry> getGeometry() const
            {
            return m_geom;
            }

        //! Set the streaming geometry
        void setGeometry(std::shared_ptr<const Geometry> geom)
            {
            m_validate_geom = true;
            m_geom = geom;
            }

    protected:
        std::shared_ptr<const Geometry> m_geom; //!< Streaming geometry
        bool m_validate_geom;   //!< If true, run a validation check on the geometry

        //! Validate the system with the streaming geometry
        void validate();

        //! Check that particles lie inside the geometry
        virtual bool validateParticles();
    };

/*!
 * \param timestep Current time to stream
 */
template<class Geometry>
void ConfinedStreamingMethod<Geometry>::stream(unsigned int timestep)
    {
    if (!shouldStream(timestep)) return;

    if (m_validate_geom)
        {
        validate();
        m_validate_geom = false;
        }

    if (m_prof) m_prof->push("MPCD stream");

    const BoxDim& box = m_mpcd_sys->getCellList()->getCoverageBox();

    ArrayHandle<Scalar4> h_pos(m_mpcd_pdata->getPositions(), access_location::host, access_mode::readwrite);
    ArrayHandle<Scalar4> h_vel(m_mpcd_pdata->getVelocities(), access_location::host, access_mode::readwrite);
    const Scalar mass = m_mpcd_pdata->getMass();

    // acquire polymorphic pointer to the external field
    const mpcd::ExternalField* field = (m_field) ? m_field->get(access_location::host) : nullptr;

    for (unsigned int cur_p = 0; cur_p < m_mpcd_pdata->getN(); ++cur_p)
        {
        const Scalar4 postype = h_pos.data[cur_p];
        Scalar3 pos = make_scalar3(postype.x, postype.y, postype.z);
        const unsigned int type = __scalar_as_int(postype.w);

        const Scalar4 vel_cell = h_vel.data[cur_p];
        Scalar3 vel = make_scalar3(vel_cell.x, vel_cell.y, vel_cell.z);
        // estimate next velocity based on current acceleration
        if (field)
            {
            vel += Scalar(0.5) * m_mpcd_dt * field->evaluate(pos) / mass;
            }

        // propagate the particle to its new position ballistically
        Scalar dt_remain = m_mpcd_dt;
        bool collide = true;
        do
            {
            pos += dt_remain * vel;
            collide = m_geom->detectCollision(pos, vel, dt_remain);
            }
        while (dt_remain > 0 && collide);
        // finalize velocity update
        if (field)
            {
            vel += Scalar(0.5) * m_mpcd_dt * field->evaluate(pos) / mass;
            }

        // wrap and update the position
        int3 image = make_int3(0,0,0);
        box.wrap(pos, image);

        h_pos.data[cur_p] = make_scalar4(pos.x, pos.y, pos.z, __int_as_scalar(type));
        h_vel.data[cur_p] = make_scalar4(vel.x, vel.y, vel.z, __int_as_scalar(mpcd::detail::NO_CELL));
        }

    // particles have moved, so the cell cache is no longer valid
    m_mpcd_pdata->invalidateCellCache();
    if (m_prof) m_prof->pop();
    }

template<class Geometry>
void ConfinedStreamingMethod<Geometry>::validate()
    {
    // ensure that the global box is padded enough for periodic boundaries
    const BoxDim& box = m_pdata->getGlobalBox();
    const Scalar cell_width = m_mpcd_sys->getCellList()->getCellSize();
    if (!m_geom->validateBox(box, cell_width))
        {
        m_exec_conf->msg->error() << "ConfinedStreamingMethod: box too small for " << Geometry::getName() << " geometry. Increase box size." << std::endl;
        throw std::runtime_error("Simulation box too small for confined streaming method");
        }

    // check that no particles are out of bounds
    unsigned char error = !validateParticles();
    #ifdef ENABLE_MPI
    if (m_exec_conf->getNRanks() > 1)
        MPI_Allreduce(MPI_IN_PLACE, &error, 1, MPI_UNSIGNED_CHAR, MPI_LOR, m_exec_conf->getMPICommunicator());
    #endif // ENABLE_MPI
    if (error)
        throw std::runtime_error("Invalid MPCD particle configuration for confined geometry");
    }

/*!
 * Checks each MPCD particle position to determine if it lies within the geometry. If any particle is
 * out of bounds, an error is raised.
 */
template<class Geometry>
bool ConfinedStreamingMethod<Geometry>::validateParticles()
    {
    ArrayHandle<Scalar4> h_pos(m_mpcd_pdata->getPositions(), access_location::host, access_mode::read);
    ArrayHandle<unsigned int> h_tag(m_mpcd_pdata->getTags(), access_location::host, access_mode::read);

    for (unsigned int idx = 0; idx < m_mpcd_pdata->getN(); ++idx)
        {
        const Scalar4 postype = h_pos.data[idx];
        const Scalar3 pos = make_scalar3(postype.x, postype.y, postype.z);
        if (m_geom->isOutside(pos))
            {
            m_exec_conf->msg->errorAllRanks() << "MPCD particle with tag " << h_tag.data[idx] << " at (" << pos.x << "," << pos.y << "," << pos.z
                          << ") lies outside the " << Geometry::getName() << " geometry. Fix configuration." << std::endl;
            return false;
            }
        }

    return true;
    }

namespace detail
{
//! Export mpcd::StreamingMethod to python
/*!
 * \param m Python module to export to
 */
template<class Geometry>
void export_ConfinedStreamingMethod(pybind11::module& m)
    {
    namespace py = pybind11;
    const std::string name = "ConfinedStreamingMethod" + Geometry::getName();
    py::class_<mpcd::ConfinedStreamingMethod<Geometry>, std::shared_ptr<mpcd::ConfinedStreamingMethod<Geometry>>>
        (m, name.c_str(), py::base<mpcd::StreamingMethod>())
        .def(py::init<std::shared_ptr<mpcd::SystemData>, unsigned int, unsigned int, int, std::shared_ptr<const Geometry>>())
        .def_property("geometry", &mpcd::ConfinedStreamingMethod<Geometry>::getGeometry,&mpcd::ConfinedStreamingMethod<Geometry>::setGeometry);
    }
} // end namespace detail
} // end namespace mpcd
#endif // MPCD_CONFINED_STREAMING_METHOD_H_
