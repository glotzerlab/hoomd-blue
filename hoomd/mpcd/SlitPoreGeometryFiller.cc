// Copyright (c) 2009-2019 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.

// Maintainer: mphoward

/*!
 * \file mpcd/SlitPoreGeometryFiller.cc
 * \brief Definition of mpcd::SlitPoreGeometryFiller
 */

#include "SlitPoreGeometryFiller.h"
#include "hoomd/RandomNumbers.h"
#include "hoomd/RNGIdentifiers.h"

#include <array>

mpcd::SlitPoreGeometryFiller::SlitPoreGeometryFiller(std::shared_ptr<mpcd::SystemData> sysdata,
                                             Scalar density,
                                             unsigned int type,
                                             std::shared_ptr<::Variant> T,
                                             unsigned int seed,
                                             std::shared_ptr<const mpcd::detail::SlitPoreGeometry> geom)
    : mpcd::VirtualParticleFiller(sysdata, density, type, T, seed),
      m_num_boxes(0), m_boxes(MAX_BOXES, m_exec_conf), m_ranges(MAX_BOXES, m_exec_conf)
    {
    m_exec_conf->msg->notice(5) << "Constructing MPCD SlitPoreGeometryFiller" << std::endl;

    setGeometry(geom);

    // unphysical values in cache to always force recompute
    m_needs_recompute = true;
    m_recompute_cache = make_scalar3(-1,-1,-1);
    m_pdata->getBoxChangeSignal().connect<mpcd::SlitPoreGeometryFiller, &mpcd::SlitPoreGeometryFiller::notifyRecompute>(this);
    }

mpcd::SlitPoreGeometryFiller::~SlitPoreGeometryFiller()
    {
    m_exec_conf->msg->notice(5) << "Destroying MPCD SlitPoreGeometryFiller" << std::endl;
    m_pdata->getBoxChangeSignal().disconnect<mpcd::SlitPoreGeometryFiller, &mpcd::SlitPoreGeometryFiller::notifyRecompute>(this);
    }

void mpcd::SlitPoreGeometryFiller::computeNumFill()
    {
    const Scalar cell_size = m_cl->getCellSize();
    const Scalar max_shift = m_cl->getMaxGridShift();

    // check if fill-relevant variables have changed (can't use signal because cell list build may not have triggered yet)
    m_needs_recompute |= (m_recompute_cache.x != cell_size ||
                          m_recompute_cache.y != max_shift ||
                          m_recompute_cache.z != m_density);

    // only recompute if needed
    if (!m_needs_recompute) return;

    // as a precaution, validate the global box with the current cell list
    const BoxDim& global_box = m_pdata->getGlobalBox();
    if (!m_geom->validateBox(global_box, cell_size))
        {
        m_exec_conf->msg->error() << "Invalid slit pore geometry for global box, cannot fill virtual particles." << std::endl;
        throw std::runtime_error("Invalid slit pore geometry for global box");
        }

    // box and slit geometry
    const BoxDim& box = m_pdata->getBox();
    const Scalar3 lo = box.getLo();
    const Scalar3 hi = box.getHi();
    const Scalar H = m_geom->getH();
    const Scalar L = m_geom->getL();
    const Scalar Ly = box.getL().y;

    /*
     * Determine the lowest / highest extent of a cells overlapping the boundaries.
     * This is done by round the walls onto the cell grid toward/away from zero, and then including the
     * max shift of this cell edge.
     */
    const Scalar3 global_lo = global_box.getLo();
    const Scalar3 global_hi = global_box.getHi();
    Scalar2 x_bounds, z_bounds;
    // upper bound on lower wall in x
    x_bounds.x = cell_size * std::ceil((-L-global_lo.x)/cell_size) + global_lo.x + max_shift;
    // lower bound on upper wall in x
    x_bounds.y = cell_size * std::floor((L-global_lo.x)/cell_size) + global_lo.x - max_shift;
    // lower bound on lower wall in z
    z_bounds.x = cell_size * std::floor((-H-global_lo.z)/cell_size) + global_lo.z - max_shift;
    // upper bound on upper wall in z
    z_bounds.y = cell_size * std::ceil((H-global_lo.z)/cell_size) + global_lo.z + max_shift;

    // define the 6 2D bounding boxes (lo.x,hi.x,lo.z,hi.z) for filling in this geometry (y is infinite)
    // (this is essentially a U shape inside the pore).
    std::array<Scalar4,MAX_BOXES> allboxes;
    allboxes[0] = make_scalar4(-L,x_bounds.x,z_bounds.y,global_hi.z);
    allboxes[1] = make_scalar4(x_bounds.y,L,z_bounds.y,global_hi.z);
    allboxes[2] = make_scalar4(-L,x_bounds.x,global_lo.z,z_bounds.x);
    allboxes[3] = make_scalar4(x_bounds.y,L,global_lo.z,z_bounds.x);
    allboxes[4] = make_scalar4(-L,L,H,z_bounds.y);
    allboxes[5] = make_scalar4(-L,L,z_bounds.x,-H);

    // find all boxes that overlap the domain
    ArrayHandle<Scalar4> h_boxes(m_boxes, access_location::host, access_mode::overwrite);
    ArrayHandle<uint2> h_ranges(m_ranges, access_location::host, access_mode::overwrite);
    m_num_boxes = 0;
    m_N_fill = 0;
    for (unsigned int i=0; i < MAX_BOXES; ++i)
        {
        const Scalar4 fillbox = allboxes[i];
        // test for bounding box overlap between the domain and the fill box
        if (!(hi.x < fillbox.x || lo.x > fillbox.y ||
              hi.z < fillbox.z || lo.z > fillbox.w))
            {
            // some overlap, so clamp the box to the local domain
            const Scalar4 clampbox = make_scalar4(std::max(fillbox.x,lo.x),
                                                  std::min(fillbox.y,hi.x),
                                                  std::max(fillbox.z,lo.z),
                                                  std::min(fillbox.w,hi.z));

            // determine volume (# of particles) for filling
            const Scalar volume = (clampbox.y-clampbox.x)*Ly*(clampbox.w-clampbox.z);
            const unsigned int N_box = std::round(volume * m_density);

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
    m_recompute_cache = make_scalar3(cell_size, max_shift, m_density);
    }

/*!
 * \param timestep Current timestep to draw particles
 */
void mpcd::SlitPoreGeometryFiller::drawParticles(unsigned int timestep)
    {
    // quit early if not filling to ensure we don't access any memory that hasn't been set
    if (m_N_fill == 0) return;

    ArrayHandle<Scalar4> h_pos(m_mpcd_pdata->getPositions(), access_location::host, access_mode::readwrite);
    ArrayHandle<Scalar4> h_vel(m_mpcd_pdata->getVelocities(), access_location::host, access_mode::readwrite);
    ArrayHandle<unsigned int> h_tag(m_mpcd_pdata->getTags(), access_location::host, access_mode::readwrite);
    const Scalar vel_factor = fast::sqrt(m_T->getValue(timestep) / m_mpcd_pdata->getMass());

    const BoxDim& box = m_pdata->getBox();
    Scalar3 lo = box.getLo();
    Scalar3 hi = box.getHi();

    // boxes for filling
    ArrayHandle<Scalar4> h_boxes(m_boxes, access_location::host, access_mode::read);
    ArrayHandle<uint2> h_ranges(m_ranges, access_location::host, access_mode::read);
    // set these counters so that they get filled on the first pass
    int boxid = -1;
    unsigned int boxlast = 0;

    // index to start filling from
    const unsigned int first_idx = m_mpcd_pdata->getN() + m_mpcd_pdata->getNVirtual() - m_N_fill;
    for (unsigned int i=0; i < m_N_fill; ++i)
        {
        const unsigned int tag = m_first_tag + i;
        hoomd::RandomGenerator rng(hoomd::RNGIdentifier::SlitPoreGeometryFiller, m_seed, tag, timestep);

        // advanced past end of this box range, take the next
        if (i >= boxlast)
            {
            ++boxid;
            boxlast = h_ranges.data[boxid].y;
            const Scalar4 fillbox = h_boxes.data[boxid];
            lo.x = fillbox.x;
            hi.x = fillbox.y;
            lo.z = fillbox.z;
            hi.z = fillbox.w;
            }

        const unsigned int pidx = first_idx + i;
        h_pos.data[pidx] = make_scalar4(hoomd::UniformDistribution<Scalar>(lo.x,hi.x)(rng),
                                        hoomd::UniformDistribution<Scalar>(lo.y,hi.y)(rng),
                                        hoomd::UniformDistribution<Scalar>(lo.z,hi.z)(rng),
                                        __int_as_scalar(m_type));

        hoomd::NormalDistribution<Scalar> gen(vel_factor, 0.0);
        Scalar3 vel;
        gen(vel.x, vel.y, rng);
        vel.z = gen(rng);
        // TODO: should these be given zero net-momentum contribution (relative to the frame of reference?)
        h_vel.data[pidx] = make_scalar4(vel.x,
                                        vel.y,
                                        vel.z,
                                        __int_as_scalar(mpcd::detail::NO_CELL));
        h_tag.data[pidx] = tag;
        }
    }

/*!
 * \param m Python module to export to
 */
void mpcd::detail::export_SlitPoreGeometryFiller(pybind11::module& m)
    {
    namespace py = pybind11;
    py::class_<mpcd::SlitPoreGeometryFiller, std::shared_ptr<mpcd::SlitPoreGeometryFiller>>
        (m, "SlitPoreGeometryFiller", py::base<mpcd::VirtualParticleFiller>())
        .def(py::init<std::shared_ptr<mpcd::SystemData>,
                      Scalar,
                      unsigned int,
                      std::shared_ptr<::Variant>,
                      unsigned int,
                      std::shared_ptr<const mpcd::detail::SlitPoreGeometry>>())
        .def("setGeometry", &mpcd::SlitPoreGeometryFiller::setGeometry)
        ;
    }
