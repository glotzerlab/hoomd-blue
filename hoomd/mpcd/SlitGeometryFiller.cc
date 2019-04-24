// Copyright (c) 2009-2019 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.

// Maintainer: mphoward

/*!
 * \file mpcd/SlitGeometryFiller.cc
 * \brief Definition of mpcd::SlitGeometryFiller
 */

#include "SlitGeometryFiller.h"
#include "hoomd/RandomNumbers.h"
#include "hoomd/RNGIdentifiers.h"

mpcd::SlitGeometryFiller::SlitGeometryFiller(std::shared_ptr<mpcd::SystemData> sysdata,
                                             Scalar density,
                                             unsigned int type,
                                             std::shared_ptr<::Variant> T,
                                             unsigned int seed,
                                             std::shared_ptr<const mpcd::detail::SlitGeometry> geom)
    : mpcd::VirtualParticleFiller(sysdata, density, type, T, seed), m_geom(geom)
    {
    m_exec_conf->msg->notice(5) << "Constructing MPCD SlitGeometryFiller" << std::endl;
    }

mpcd::SlitGeometryFiller::~SlitGeometryFiller()
    {
    m_exec_conf->msg->notice(5) << "Destroying MPCD SlitGeometryFiller" << std::endl;
    }

void mpcd::SlitGeometryFiller::computeNumFill()
    {
    // as a precaution, validate the global box with the current cell list
    const BoxDim& global_box = m_pdata->getGlobalBox();
    const Scalar cell_size = m_cl->getCellSize();
    if (!m_geom->validateBox(global_box, cell_size))
        {
        m_exec_conf->msg->error() << "Invalid slit geometry for global box, cannot fill virtual particles." << std::endl;
        throw std::runtime_error("Invalid slit geometry for global box");
        }

    // box and slit geometry
    const BoxDim& box = m_pdata->getBox();
    const Scalar3 L = box.getL();
    const Scalar A = L.x * L.y;
    const Scalar H = m_geom->getH();

    // default is not to fill anything
    m_z_min = -H; m_z_max = H;
    m_N_hi = m_N_lo = 0;

    /*
     * Determine the lowest / highest extent of a cell containing a particle within the channel.
     * This is done by round the walls onto the cell grid away from zero, and then including the
     * max shift of this cell edge.
     */
    const Scalar max_shift = m_cl->getMaxGridShift();
    const Scalar global_lo = global_box.getLo().z;
    if (box.getHi().z >= H)
        {
        m_z_max = cell_size * std::ceil((H-global_lo)/cell_size) + global_lo + max_shift;
        m_N_hi = std::round((m_z_max - H) * A * m_density);
        }

    if (box.getLo().z <= -H)
        {
        m_z_min = cell_size * std::floor((-H-global_lo)/cell_size) + global_lo - max_shift;
        m_N_lo = std::round((-H-m_z_min) * A * m_density);
        }

    // total number of fill particles
    m_N_fill = m_N_hi + m_N_lo;
    }

/*!
 * \param timestep Current timestep to draw particles
 */
void mpcd::SlitGeometryFiller::drawParticles(unsigned int timestep)
    {
    ArrayHandle<Scalar4> h_pos(m_mpcd_pdata->getPositions(), access_location::host, access_mode::readwrite);
    ArrayHandle<Scalar4> h_vel(m_mpcd_pdata->getVelocities(), access_location::host, access_mode::readwrite);
    ArrayHandle<unsigned int> h_tag(m_mpcd_pdata->getTags(), access_location::host, access_mode::readwrite);

    const BoxDim& box = m_pdata->getBox();
    Scalar3 lo = box.getLo();
    Scalar3 hi = box.getHi();

    const Scalar vel_factor = fast::sqrt(m_T->getValue(timestep) / m_mpcd_pdata->getMass());

    // index to start filling from
    const unsigned int first_idx = m_mpcd_pdata->getN() + m_mpcd_pdata->getNVirtual() - m_N_fill;
    for (unsigned int i=0; i < m_N_fill; ++i)
        {
        const unsigned int tag = m_first_tag + i;
        hoomd::RandomGenerator rng(hoomd::RNGIdentifier::SlitGeometryFiller, m_seed, tag, timestep);
        signed char sign = (i >= m_N_lo) - (i < m_N_lo);
        if (sign == -1) // bottom
            {
            lo.z = m_z_min; hi.z = -m_geom->getH();
            }
        else // top
            {
            lo.z = m_geom->getH(); hi.z = m_z_max;
            }

        const unsigned int pidx = first_idx + i;
        h_pos.data[pidx] = make_scalar4(hoomd::UniformDistribution<Scalar>(lo.x, hi.x)(rng),
                                        hoomd::UniformDistribution<Scalar>(lo.y, hi.y)(rng),
                                        hoomd::UniformDistribution<Scalar>(lo.z, hi.z)(rng),
                                        __int_as_scalar(m_type));

        hoomd::NormalDistribution<Scalar> gen(vel_factor, 0.0);
        Scalar3 vel;
        gen(vel.x, vel.y, rng);
        vel.z = gen(rng);
        // TODO: should these be given zero net-momentum contribution (relative to the frame of reference?)
        h_vel.data[pidx] = make_scalar4(vel.x + sign * m_geom->getVelocity(),
                                        vel.y,
                                        vel.z,
                                        __int_as_scalar(mpcd::detail::NO_CELL));
        h_tag.data[pidx] = tag;
        }
    }

/*!
 * \param m Python module to export to
 */
void mpcd::detail::export_SlitGeometryFiller(pybind11::module& m)
    {
    namespace py = pybind11;
    py::class_<mpcd::SlitGeometryFiller, std::shared_ptr<mpcd::SlitGeometryFiller>>
        (m, "SlitGeometryFiller", py::base<mpcd::VirtualParticleFiller>())
        .def(py::init<std::shared_ptr<mpcd::SystemData>,
                      Scalar,
                      unsigned int,
                      std::shared_ptr<::Variant>,
                      unsigned int,
                      std::shared_ptr<const mpcd::detail::SlitGeometry>>())
        .def("setGeometry", &mpcd::SlitGeometryFiller::setGeometry)
        ;
    }
