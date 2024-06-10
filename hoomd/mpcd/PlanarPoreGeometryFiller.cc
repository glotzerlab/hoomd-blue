// Copyright (c) 2009-2024 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

/*!
 * \file mpcd/PlanarPoreGeometryFiller.cc
 * \brief Definition of mpcd::PlanarPoreGeometryFiller
 */

#include "PlanarPoreGeometryFiller.h"
#include "hoomd/RNGIdentifiers.h"
#include "hoomd/RandomNumbers.h"

#include <array>

namespace hoomd
    {
mpcd::PlanarPoreGeometryFiller::PlanarPoreGeometryFiller(
    std::shared_ptr<SystemDefinition> sysdef,
    const std::string& type,
    Scalar density,
    std::shared_ptr<Variant> T,
    std::shared_ptr<const mpcd::PlanarPoreGeometry> geom)
    : mpcd::ManualVirtualParticleFiller(sysdef, type, density, T), m_num_boxes(0),
      m_boxes(MAX_BOXES, m_exec_conf), m_ranges(MAX_BOXES, m_exec_conf)
    {
    m_exec_conf->msg->notice(5) << "Constructing MPCD PlanarPoreGeometryFiller" << std::endl;

    setGeometry(geom);

    // unphysical values in cache to always force recompute
    m_needs_recompute = true;
    m_recompute_cache = make_scalar4(-1, -1, -1, -1);
    m_pdata->getBoxChangeSignal()
        .connect<mpcd::PlanarPoreGeometryFiller, &mpcd::PlanarPoreGeometryFiller::notifyRecompute>(
            this);
    }

mpcd::PlanarPoreGeometryFiller::~PlanarPoreGeometryFiller()
    {
    m_exec_conf->msg->notice(5) << "Destroying MPCD PlanarPoreGeometryFiller" << std::endl;
    m_pdata->getBoxChangeSignal()
        .disconnect<mpcd::PlanarPoreGeometryFiller,
                    &mpcd::PlanarPoreGeometryFiller::notifyRecompute>(this);
    }

void mpcd::PlanarPoreGeometryFiller::fill(uint64_t timestep)
    {
    const BoxDim& box = m_pdata->getBox();
    if (box.getTiltFactorXY() != Scalar(0.0) || box.getTiltFactorXZ() != Scalar(0.0)
        || box.getTiltFactorYZ() != Scalar(0.0))
        {
        throw std::runtime_error("PlanarPoreGeometryFiller does not work with skewed boxes");
        }
    mpcd::ManualVirtualParticleFiller::fill(timestep);
    }

void mpcd::PlanarPoreGeometryFiller::computeNumFill()
    {
    const Scalar3 cell_size_vector = m_cl->getCellSize();
    if (fabs(cell_size_vector.x - cell_size_vector.y) > Scalar(1e-6)
        || fabs(cell_size_vector.x - cell_size_vector.z) > Scalar(1e-6))
        {
        throw std::runtime_error("Cell size must be constant");
        }
    const Scalar cell_size = cell_size_vector.y;
    const Scalar3 max_shift = m_cl->getMaxGridShift();

    // check if fill-relevant variables have changed (can't use signal because cell list build may
    // not have triggered yet)
    m_needs_recompute
        |= (m_recompute_cache.x != cell_size || m_recompute_cache.y != max_shift.x
            || m_recompute_cache.z != max_shift.y || m_recompute_cache.w != m_density);

    // only recompute if needed
    if (!m_needs_recompute)
        return;

    // as a precaution, validate the global box with the current cell list
    const BoxDim& global_box = m_pdata->getGlobalBox();

    // box and slit geometry
    const BoxDim& box = m_pdata->getBox();
    const Scalar3 lo = box.getLo();
    const Scalar3 hi = box.getHi();
    const Scalar H = m_geom->getH();
    const Scalar L = m_geom->getL();
    const Scalar Lz = box.getL().z;

    /*
     * Determine the lowest / highest extent of a cells overlapping the boundaries.
     * This is done by round the walls onto the cell grid toward/away from zero, and then including
     * the max shift of this cell edge.
     */
    const Scalar3 global_lo = global_box.getLo();
    const Scalar3 global_hi = global_box.getHi();
    Scalar2 x_bounds, y_bounds;
    // upper bound on lower wall in x
    x_bounds.x = cell_size * std::ceil((-L - global_lo.x) / cell_size) + global_lo.x + max_shift.x;
    // lower bound on upper wall in x
    x_bounds.y = cell_size * std::floor((L - global_lo.x) / cell_size) + global_lo.x - max_shift.x;
    // lower bound on lower wall in y
    y_bounds.x = cell_size * std::floor((-H - global_lo.y) / cell_size) + global_lo.y - max_shift.y;
    // upper bound on upper wall in y
    y_bounds.y = cell_size * std::ceil((H - global_lo.y) / cell_size) + global_lo.y + max_shift.y;

    // define the 6 2D bounding boxes (lo.x,hi.x,lo.y,hi.y) for filling in this geometry (z is
    // infinite) (this is essentially a U shape inside the pore).
    std::array<Scalar4, MAX_BOXES> allboxes;
    allboxes[0] = make_scalar4(-L, x_bounds.x, y_bounds.y, global_hi.y);
    allboxes[1] = make_scalar4(x_bounds.y, L, y_bounds.y, global_hi.y);
    allboxes[2] = make_scalar4(-L, x_bounds.x, global_lo.y, y_bounds.x);
    allboxes[3] = make_scalar4(x_bounds.y, L, global_lo.y, y_bounds.x);
    allboxes[4] = make_scalar4(-L, L, H, y_bounds.y);
    allboxes[5] = make_scalar4(-L, L, y_bounds.x, -H);

    // find all boxes that overlap the domain
    ArrayHandle<Scalar4> h_boxes(m_boxes, access_location::host, access_mode::overwrite);
    ArrayHandle<uint2> h_ranges(m_ranges, access_location::host, access_mode::overwrite);
    m_num_boxes = 0;
    m_N_fill = 0;
    for (unsigned int i = 0; i < MAX_BOXES; ++i)
        {
        const Scalar4 fillbox = allboxes[i];
        // test for bounding box overlap between the domain and the fill box
        if (!(hi.x < fillbox.x || lo.x > fillbox.y || hi.y < fillbox.z || lo.y > fillbox.w))
            {
            // some overlap, so clamp the box to the local domain
            const Scalar4 clampbox = make_scalar4(std::max(fillbox.x, lo.x),
                                                  std::min(fillbox.y, hi.x),
                                                  std::max(fillbox.z, lo.y),
                                                  std::min(fillbox.w, hi.y));

            // determine volume (# of particles) for filling
            const Scalar volume = (clampbox.y - clampbox.x) * (clampbox.w - clampbox.z) * Lz;
            const unsigned int N_box = (unsigned int)std::round(volume * m_density);

            // only add box if it isn't empty
            if (N_box != 0)
                {
                h_boxes.data[m_num_boxes] = clampbox;
                h_ranges.data[m_num_boxes] = make_uint2(m_N_fill, m_N_fill + N_box);
                ++m_num_boxes;

                m_N_fill += N_box;
                }
            }
        }

    // size is now updated, cache the cell dimensions used
    m_needs_recompute = false;
    m_recompute_cache = make_scalar4(cell_size, max_shift.x, max_shift.y, m_density);
    }

/*!
 * \param timestep Current timestep to draw particles
 */
void mpcd::PlanarPoreGeometryFiller::drawParticles(uint64_t timestep)
    {
    // quit early if not filling to ensure we don't access any memory that hasn't been set
    if (m_N_fill == 0)
        return;

    ArrayHandle<Scalar4> h_pos(m_mpcd_pdata->getPositions(),
                               access_location::host,
                               access_mode::readwrite);
    ArrayHandle<Scalar4> h_vel(m_mpcd_pdata->getVelocities(),
                               access_location::host,
                               access_mode::readwrite);
    ArrayHandle<unsigned int> h_tag(m_mpcd_pdata->getTags(),
                                    access_location::host,
                                    access_mode::readwrite);
    const Scalar vel_factor = fast::sqrt((*m_T)(timestep) / m_mpcd_pdata->getMass());

    const BoxDim& box = m_pdata->getBox();
    Scalar3 lo = box.getLo();
    Scalar3 hi = box.getHi();

    // boxes for filling
    ArrayHandle<Scalar4> h_boxes(m_boxes, access_location::host, access_mode::read);
    ArrayHandle<uint2> h_ranges(m_ranges, access_location::host, access_mode::read);
    // set these counters so that they get filled on the first pass
    int boxid = -1;
    unsigned int boxlast = 0;

    uint16_t seed = m_sysdef->getSeed();

    // index to start filling from
    for (unsigned int i = 0; i < m_N_fill; ++i)
        {
        const unsigned int tag = m_first_tag + i;
        hoomd::RandomGenerator rng(
            hoomd::Seed(hoomd::RNGIdentifier::PlanarPoreGeometryFiller, timestep, seed),
            hoomd::Counter(tag, m_filler_id));

        // advanced past end of this box range, take the next
        if (i >= boxlast)
            {
            ++boxid;
            boxlast = h_ranges.data[boxid].y;
            const Scalar4 fillbox = h_boxes.data[boxid];
            lo.x = fillbox.x;
            hi.x = fillbox.y;
            lo.y = fillbox.z;
            hi.y = fillbox.w;
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
        h_vel.data[pidx]
            = make_scalar4(vel.x, vel.y, vel.z, __int_as_scalar(mpcd::detail::NO_CELL));
        h_tag.data[pidx] = tag;
        }
    }

/*!
 * \param m Python module to export to
 */
void mpcd::detail::export_PlanarPoreGeometryFiller(pybind11::module& m)
    {
    pybind11::class_<mpcd::PlanarPoreGeometryFiller,
                     mpcd::VirtualParticleFiller,
                     std::shared_ptr<mpcd::PlanarPoreGeometryFiller>>(m, "PlanarPoreGeometryFiller")
        .def(pybind11::init<std::shared_ptr<SystemDefinition>,
                            const std::string&,
                            Scalar,
                            std::shared_ptr<Variant>,
                            std::shared_ptr<const mpcd::PlanarPoreGeometry>>())
        .def_property_readonly("geometry", &mpcd::PlanarPoreGeometryFiller::getGeometry);
    }

    } // end namespace hoomd
