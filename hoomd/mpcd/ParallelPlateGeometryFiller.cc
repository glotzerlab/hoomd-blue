// Copyright (c) 2009-2024 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

/*!
 * \file mpcd/ParallelPlateGeometryFiller.cc
 * \brief Definition of mpcd::ParallelPlateGeometryFiller
 */

#include "ParallelPlateGeometryFiller.h"
#include "hoomd/RNGIdentifiers.h"
#include "hoomd/RandomNumbers.h"

namespace hoomd
    {
mpcd::ParallelPlateGeometryFiller::ParallelPlateGeometryFiller(
    std::shared_ptr<SystemDefinition> sysdef,
    const std::string& type,
    Scalar density,
    std::shared_ptr<Variant> T,
    std::shared_ptr<const mpcd::ParallelPlateGeometry> geom)
    : mpcd::ManualVirtualParticleFiller(sysdef, type, density, T), m_geom(geom)
    {
    m_exec_conf->msg->notice(5) << "Constructing MPCD ParallelPlateGeometryFiller" << std::endl;
    }

mpcd::ParallelPlateGeometryFiller::~ParallelPlateGeometryFiller()
    {
    m_exec_conf->msg->notice(5) << "Destroying MPCD ParallelPlateGeometryFiller" << std::endl;
    }

void mpcd::ParallelPlateGeometryFiller::fill(uint64_t timestep)
    {
    const BoxDim& box = m_pdata->getBox();
    if (box.getTiltFactorXY() != Scalar(0.0) || box.getTiltFactorXZ() != Scalar(0.0)
        || box.getTiltFactorYZ() != Scalar(0.0))
        {
        throw std::runtime_error("ParallelPlateGeometryFiller does not work with skewed boxes");
        }
    mpcd::ManualVirtualParticleFiller::fill(timestep);
    }

void mpcd::ParallelPlateGeometryFiller::computeNumFill()
    {
    const BoxDim& global_box = m_pdata->getGlobalBox();
    const Scalar3 cell_size_vector = m_cl->getCellSize();
    if (fabs(cell_size_vector.x - cell_size_vector.y) > Scalar(1e-6)
        || fabs(cell_size_vector.x - cell_size_vector.z) > Scalar(1e-6))
        {
        throw std::runtime_error("Cell size must be constant");
        }
    const Scalar cell_size = cell_size_vector.y;

    // box and slit geometry
    const BoxDim& box = m_pdata->getBox();
    const Scalar3 L = box.getL();
    const Scalar A = L.x * L.y;
    const Scalar H = m_geom->getH();

    // default is not to fill anything
    m_y_min = -H;
    m_y_max = H;
    m_N_hi = m_N_lo = 0;

    /*
     * Determine the lowest / highest extent of a cell containing a particle within the channel.
     * This is done by round the walls onto the cell grid away from zero, and then including the
     * max shift of this cell edge.
     */
    const Scalar max_shift = m_cl->getMaxGridShift().y;
    const Scalar global_lo = global_box.getLo().y;
    if (box.getHi().y >= H)
        {
        m_y_max = cell_size * std::ceil((H - global_lo) / cell_size) + global_lo + max_shift;
        m_N_hi = (unsigned int)std::round((m_y_max - H) * A * m_density);
        }

    if (box.getLo().y <= -H)
        {
        m_y_min = cell_size * std::floor((-H - global_lo) / cell_size) + global_lo - max_shift;
        m_N_lo = (unsigned int)std::round((-H - m_y_min) * A * m_density);
        }

    // total number of fill particles
    m_N_fill = m_N_hi + m_N_lo;
    }

/*!
 * \param timestep Current timestep to draw particles
 */
void mpcd::ParallelPlateGeometryFiller::drawParticles(uint64_t timestep)
    {
    ArrayHandle<Scalar4> h_pos(m_mpcd_pdata->getPositions(),
                               access_location::host,
                               access_mode::readwrite);
    ArrayHandle<Scalar4> h_vel(m_mpcd_pdata->getVelocities(),
                               access_location::host,
                               access_mode::readwrite);
    ArrayHandle<unsigned int> h_tag(m_mpcd_pdata->getTags(),
                                    access_location::host,
                                    access_mode::readwrite);

    const BoxDim& box = m_pdata->getBox();
    Scalar3 lo = box.getLo();
    Scalar3 hi = box.getHi();

    const Scalar vel_factor = fast::sqrt((*m_T)(timestep) / m_mpcd_pdata->getMass());

    uint16_t seed = m_sysdef->getSeed();

    // index to start filling from
    for (unsigned int i = 0; i < m_N_fill; ++i)
        {
        const unsigned int tag = m_first_tag + i;
        hoomd::RandomGenerator rng(
            hoomd::Seed(hoomd::RNGIdentifier::ParallelPlateGeometryFiller, timestep, seed),
            hoomd::Counter(tag, m_filler_id));
        signed char sign = (char)((i >= m_N_lo) - (i < m_N_lo));
        if (sign == -1) // bottom
            {
            lo.y = m_y_min;
            hi.y = -m_geom->getH();
            }
        else // top
            {
            lo.y = m_geom->getH();
            hi.y = m_y_max;
            }

        const unsigned int pidx = m_first_idx + i;
        h_pos.data[pidx] = make_scalar4(hoomd::UniformDistribution<Scalar>(lo.x, hi.x)(rng),
                                        hoomd::UniformDistribution<Scalar>(lo.y, hi.y)(rng),
                                        hoomd::UniformDistribution<Scalar>(lo.z, hi.z)(rng),
                                        __int_as_scalar(m_type));

        hoomd::NormalDistribution<Scalar> gen(vel_factor, 0.0);
        Scalar3 vel;
        gen(vel.x, vel.y, rng);
        vel.z = gen(rng);
        // TODO: should these be given zero net-momentum contribution (relative to the frame of
        // reference?)
        h_vel.data[pidx] = make_scalar4(vel.x + sign * m_geom->getSpeed(),
                                        vel.y,
                                        vel.z,
                                        __int_as_scalar(mpcd::detail::NO_CELL));
        h_tag.data[pidx] = tag;
        }
    }

/*!
 * \param m Python module to export to
 */
void mpcd::detail::export_ParallelPlateGeometryFiller(pybind11::module& m)
    {
    pybind11::class_<mpcd::ParallelPlateGeometryFiller,
                     mpcd::VirtualParticleFiller,
                     std::shared_ptr<mpcd::ParallelPlateGeometryFiller>>(
        m,
        "ParallelPlateGeometryFiller")
        .def(pybind11::init<std::shared_ptr<SystemDefinition>,
                            const std::string&,
                            Scalar,
                            std::shared_ptr<Variant>,
                            std::shared_ptr<const mpcd::ParallelPlateGeometry>>())
        .def_property_readonly("geometry", &mpcd::ParallelPlateGeometryFiller::getGeometry);
    }

    } // end namespace hoomd
