// Copyright (c) 2009-2025 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

/*!
 * \file ReverseNonequilibriumShearFlow.cc
 * \brief Definition of reverse nonequilibrium shear flow updater
 */

#include <algorithm>

#include "ReverseNonequilibriumShearFlow.h"
#include "ReverseNonequilibriumShearFlowUtilities.h"

namespace hoomd
    {
mpcd::ReverseNonequilibriumShearFlow::ReverseNonequilibriumShearFlow(
    std::shared_ptr<SystemDefinition> sysdef,
    std::shared_ptr<Trigger> trigger,
    unsigned int num_swap,
    Scalar slab_width,
    Scalar target_momentum)
    : Updater(sysdef, trigger), m_mpcd_pdata(sysdef->getMPCDParticleData()), m_num_swap(num_swap),
      m_slab_width(slab_width), m_target_momentum(target_momentum), m_summed_momentum_exchange(0),
      m_num_lo(0), m_particles_lo(m_exec_conf), m_num_hi(0), m_particles_hi(m_exec_conf),
      m_update_slabs(true)
    {
    m_exec_conf->msg->notice(5) << "Constructing ReverseNonequilibriumShearFlow" << std::endl;

    m_pdata->getBoxChangeSignal()
        .connect<mpcd::ReverseNonequilibriumShearFlow,
                 &mpcd::ReverseNonequilibriumShearFlow::requestUpdateSlabs>(this);

    GPUArray<Scalar2> particles_staged(2 * m_num_swap, m_exec_conf);
    m_particles_staged.swap(particles_staged);
    }

mpcd::ReverseNonequilibriumShearFlow::~ReverseNonequilibriumShearFlow()
    {
    m_exec_conf->msg->notice(5) << "Destroying ReverseNonequilibriumShearFlow" << std::endl;
    m_pdata->getBoxChangeSignal()
        .disconnect<mpcd::ReverseNonequilibriumShearFlow,
                    &mpcd::ReverseNonequilibriumShearFlow::requestUpdateSlabs>(this);
    }

/*!
 * \param num_swap Max number of swaps
 */
void mpcd::ReverseNonequilibriumShearFlow::setNumSwap(unsigned int num_swap)
    {
    if (num_swap > m_num_swap)
        {
        GPUArray<Scalar2> particles_staged(2 * num_swap, m_exec_conf);
        m_particles_staged.swap(particles_staged);
        }
    m_num_swap = num_swap;
    }

//! Set the target momentum
void mpcd::ReverseNonequilibriumShearFlow::setTargetMomentum(Scalar target_momentum)
    {
    m_target_momentum = target_momentum;
    }

/*!
 * \param slab_width Slab width
 */
void mpcd::ReverseNonequilibriumShearFlow::setSlabWidth(Scalar slab_width)
    {
    m_slab_width = slab_width;
    requestUpdateSlabs();
    }

void mpcd::ReverseNonequilibriumShearFlow::setSlabs()
    {
    const BoxDim& global_box = m_pdata->getGlobalBox();
    if (m_slab_width > Scalar(0.5) * global_box.getNearestPlaneDistance().y)
        {
        throw std::runtime_error("Slab width cannot be larger than Ly/2");
        }

    const Scalar3 global_lo = global_box.getLo();
    m_pos_lo = make_scalar2(global_lo.y, global_lo.y + m_slab_width);
    m_pos_hi = make_scalar2(Scalar(0.0), m_slab_width);
    }

/*!
 * \param timestep Current time step of the simulation
 */
void mpcd::ReverseNonequilibriumShearFlow::update(uint64_t timestep)
    {
    // if slabs have changed, update them
    if (m_update_slabs)
        {
        setSlabs();
        m_update_slabs = false;
        }

    findSwapParticles();
    sortOutSwapParticles();
    stageSwapParticles();
    swapParticleMomentum();
    }

/*!
 * Finds all particles in the "lower" and "upper" slab in y direction and
 * puts them into two GPUArrays, sorted by their momentum closest to -/+ target_momentum in x
 * direction.
 */
void mpcd::ReverseNonequilibriumShearFlow::findSwapParticles()
    {
    // fill the layers with (unsorted) particles. this uses do loop for reallocation
    bool filled = false;
    do
        {
        const size_t num_lo_alloc = m_particles_lo.getNumElements();
        const size_t num_hi_alloc = m_particles_hi.getNumElements();
            {
            ArrayHandle<Scalar4> h_pos(m_mpcd_pdata->getPositions(),
                                       access_location::host,
                                       access_mode::read);
            ArrayHandle<Scalar4> h_vel(m_mpcd_pdata->getVelocities(),
                                       access_location::host,
                                       access_mode::read);

            ArrayHandle<Scalar2> h_particles_lo(m_particles_lo,
                                                access_location::host,
                                                access_mode::overwrite);
            ArrayHandle<Scalar2> h_particles_hi(m_particles_hi,
                                                access_location::host,
                                                access_mode::overwrite);

            // filter particles into their slab in y-direction and record momentum in x-direction
            m_num_lo = 0;
            m_num_hi = 0;
            for (unsigned int idx = 0; idx < m_mpcd_pdata->getN(); ++idx)
                {
                const Scalar4 vel = h_vel.data[idx];
                const Scalar momentum = vel.x * m_mpcd_pdata->getMass();
                const Scalar y = h_pos.data[idx].y;
                if (m_pos_lo.x <= y && y < m_pos_lo.y
                    && momentum > Scalar(0.0)) // lower slab, search for positive momentum
                    {
                    if (m_num_lo < num_lo_alloc)
                        {
                        h_particles_lo.data[m_num_lo]
                            = make_scalar2(__int_as_scalar(idx), momentum);
                        }
                    ++m_num_lo;
                    }
                else if (m_pos_hi.x <= y && y < m_pos_hi.y
                         && momentum < Scalar(0.0)) // higher slab, search for negative momentum
                    {
                    if (m_num_hi < num_hi_alloc)
                        {
                        h_particles_hi.data[m_num_hi]
                            = make_scalar2(__int_as_scalar(idx), momentum);
                        }
                    ++m_num_hi;
                    }
                }
            }
        filled = true;

        // reallocate if required and go again, this won't happen too much
        if (m_num_lo > num_lo_alloc)
            {
            GPUArray<Scalar2> particles_lo(m_num_lo, m_exec_conf);
            m_particles_lo.swap(particles_lo);
            filled = false;
            }
        if (m_num_hi > num_hi_alloc)
            {
            GPUArray<Scalar2> particles_hi(m_num_hi, m_exec_conf);
            m_particles_hi.swap(particles_hi);
            filled = false;
            }

        } while (!filled);
    }

void mpcd::ReverseNonequilibriumShearFlow::sortOutSwapParticles()
    {
    ArrayHandle<Scalar2> h_particles_lo(m_particles_lo,
                                        access_location::host,
                                        access_mode::readwrite);
    ArrayHandle<Scalar2> h_particles_hi(m_particles_hi,
                                        access_location::host,
                                        access_mode::readwrite);
    ArrayHandle<unsigned int> h_tag(m_mpcd_pdata->getTags(),
                                    access_location::host,
                                    access_mode::read);

    const unsigned int num_top_lo = std::min(m_num_swap, m_num_lo);
    const unsigned int num_top_hi = std::min(m_num_swap, m_num_hi);
    if (std::isinf(m_target_momentum))
        {
        std::partial_sort(h_particles_lo.data,
                          h_particles_lo.data + num_top_lo,
                          h_particles_lo.data + m_num_lo,
                          detail::MaximumMomentum(h_tag.data));
        std::partial_sort(h_particles_hi.data,
                          h_particles_hi.data + num_top_hi,
                          h_particles_hi.data + m_num_hi,
                          detail::MinimumMomentum(h_tag.data));
        }
    else
        {
        std::partial_sort(h_particles_lo.data,
                          h_particles_lo.data + num_top_lo,
                          h_particles_lo.data + m_num_lo,
                          detail::CompareMomentumToTarget(m_target_momentum, h_tag.data));
        std::partial_sort(h_particles_hi.data,
                          h_particles_hi.data + num_top_hi,
                          h_particles_hi.data + m_num_hi,
                          detail::CompareMomentumToTarget(-m_target_momentum, h_tag.data));
        }

    m_top_particles_lo.resize(num_top_lo);
    std::copy(h_particles_lo.data, h_particles_lo.data + num_top_lo, m_top_particles_lo.begin());

    m_top_particles_hi.resize(num_top_hi);
    std::copy(h_particles_hi.data, h_particles_hi.data + num_top_hi, m_top_particles_hi.begin());
    }

/*!
 * Stage particles momenta from both slabs into a queue for swapping
 */
void mpcd::ReverseNonequilibriumShearFlow::stageSwapParticles()
    {
    // determine number of pairs to swap
    const unsigned int num_top_lo = static_cast<unsigned int>(m_top_particles_lo.size());
    const unsigned int num_top_hi = static_cast<unsigned int>(m_top_particles_hi.size());
    unsigned int num_top_lo_global = num_top_lo;
    unsigned int num_top_hi_global = num_top_hi;
#ifdef ENABLE_MPI
    if (m_sysdef->isDomainDecomposed())
        {
        auto mpi_comm = m_exec_conf->getMPICommunicator();
        MPI_Allreduce(MPI_IN_PLACE, &num_top_lo_global, 1, MPI_UNSIGNED, MPI_SUM, mpi_comm);
        MPI_Allreduce(MPI_IN_PLACE, &num_top_hi_global, 1, MPI_UNSIGNED, MPI_SUM, mpi_comm);
        }
#endif // ENABLE_MPI
    const unsigned int num_pairs
        = std::min(m_num_swap, std::min(num_top_lo_global, num_top_hi_global));

    // stage swaps into queue
    ArrayHandle<Scalar2> h_particles_staged(m_particles_staged,
                                            access_location::host,
                                            access_mode::overwrite);
    m_num_staged = 0;
    unsigned int lo_idx = 0;
    unsigned int hi_idx = 0;
    for (unsigned int i = 0; i < num_pairs; ++i)
        {
#ifdef ENABLE_MPI
        if (m_sysdef->isDomainDecomposed())
            {
            const auto mpi_comm = m_exec_conf->getMPICommunicator();
            const int rank = m_exec_conf->getRank();

            // find particle in lo slab from all ranks (p > 0)
            MPI_Op lo_op;
            Scalar_Int lo_swap;
            lo_swap.i = rank;
            if (std::isinf(m_target_momentum))
                {
                lo_op = MPI_MAXLOC;
                lo_swap.s = (lo_idx < num_top_lo) ? m_top_particles_lo[lo_idx].y : Scalar(0.0);
                }
            else
                {
                lo_op = MPI_MINLOC;
                lo_swap.s = (lo_idx < num_top_lo)
                                ? std::fabs(m_top_particles_lo[lo_idx].y - m_target_momentum)
                                : std::numeric_limits<Scalar>::max();
                }
            MPI_Allreduce(MPI_IN_PLACE, &lo_swap, 1, MPI_HOOMD_SCALAR_INT, lo_op, mpi_comm);

            // find particle in hi slab to swap (p < 0)
            Scalar_Int hi_swap;
            hi_swap.i = rank;
            if (hi_idx < num_top_hi)
                {
                hi_swap.s = (std::isinf(m_target_momentum))
                                ? m_top_particles_hi[hi_idx].y
                                : std::fabs(m_top_particles_hi[hi_idx].y + m_target_momentum);
                }
            else
                {
                hi_swap.s = std::numeric_limits<Scalar>::max();
                }
            MPI_Allreduce(MPI_IN_PLACE, &hi_swap, 1, MPI_HOOMD_SCALAR_INT, MPI_MINLOC, mpi_comm);

            if (lo_swap.i == rank || hi_swap.i == rank) // at most, 2 ranks participate in the swap
                {
                if (lo_swap.i != hi_swap.i) // particles to swap are on different ranks, needs MPI
                    {
                    Scalar2 particle;
                    unsigned int dest;
                    if (lo_swap.i == rank)
                        {
                        particle = m_top_particles_lo[lo_idx++];
                        dest = hi_swap.i;
                        }
                    else
                        {
                        particle = m_top_particles_hi[hi_idx++];
                        dest = lo_swap.i;
                        }

                    MPI_Sendrecv_replace(&particle.y,
                                         1,
                                         MPI_HOOMD_SCALAR,
                                         dest,
                                         0,
                                         dest,
                                         0,
                                         mpi_comm,
                                         MPI_STATUS_IGNORE);

                    h_particles_staged.data[m_num_staged++] = particle;
                    }
                else // particles are on the same rank
                    {
                    Scalar2 lo_particle = m_top_particles_lo[lo_idx++];
                    Scalar2 hi_particle = m_top_particles_hi[hi_idx++];
                    std::swap(lo_particle.y, hi_particle.y);
                    h_particles_staged.data[m_num_staged++] = lo_particle;
                    h_particles_staged.data[m_num_staged++] = hi_particle;
                    }
                }
            }
        else
#endif // ENABLE_MPI
            {
            Scalar2 lo_particle = m_top_particles_lo[lo_idx++];
            Scalar2 hi_particle = m_top_particles_hi[hi_idx++];
            std::swap(lo_particle.y, hi_particle.y);
            h_particles_staged.data[m_num_staged++] = lo_particle;
            h_particles_staged.data[m_num_staged++] = hi_particle;
            }
        }
    }

/*!
 * Apply new momenta to particles from the queue.
 */
void mpcd::ReverseNonequilibriumShearFlow::swapParticleMomentum()
    {
    ArrayHandle<Scalar4> h_vel(m_mpcd_pdata->getVelocities(),
                               access_location::host,
                               access_mode::readwrite);
    ArrayHandle<Scalar2> h_particles_staged(m_particles_staged,
                                            access_location::host,
                                            access_mode::read);

    // perform swap and sum momentum exchange
    Scalar momentum_sum(0);
    const Scalar mass = m_mpcd_pdata->getMass();
    for (unsigned int i = 0; i < m_num_staged; ++i)
        {
        const Scalar2 pidx_mom = h_particles_staged.data[i];
        const unsigned int pidx = __scalar_as_int(pidx_mom.x);
        const Scalar new_momentum = pidx_mom.y;

        const Scalar current_momentum = h_vel.data[pidx].x * mass;

        h_vel.data[pidx].x = new_momentum / mass;
        momentum_sum += std::fabs(new_momentum - current_momentum);
        }

#ifdef ENABLE_MPI
    if (m_sysdef->isDomainDecomposed())
        {
        auto comm = m_exec_conf->getMPICommunicator();
        MPI_Allreduce(MPI_IN_PLACE, &momentum_sum, 1, MPI_HOOMD_SCALAR, MPI_SUM, comm);
        }
#endif

    // absolute value gives extra factor of 2, divide out when accumulating
    m_summed_momentum_exchange += Scalar(0.5) * momentum_sum;
    }

namespace mpcd
    {
namespace detail
    {
/*!
 * \param m Python module to export to
 */
void export_ReverseNonequilibriumShearFlow(pybind11::module& m)
    {
    namespace py = pybind11;
    py::class_<mpcd::ReverseNonequilibriumShearFlow,
               Updater,
               std::shared_ptr<mpcd::ReverseNonequilibriumShearFlow>>(
        m,
        "ReverseNonequilibriumShearFlow")
        .def(py::init<std::shared_ptr<SystemDefinition>,
                      std::shared_ptr<Trigger>,
                      unsigned int,
                      Scalar,
                      Scalar>())
        .def_property("num_swaps",
                      &mpcd::ReverseNonequilibriumShearFlow::getNumSwap,
                      &mpcd::ReverseNonequilibriumShearFlow::setNumSwap)
        .def_property("slab_width",
                      &mpcd::ReverseNonequilibriumShearFlow::getSlabWidth,
                      &mpcd::ReverseNonequilibriumShearFlow::setSlabWidth)
        .def_property("target_momentum",
                      &mpcd::ReverseNonequilibriumShearFlow::getTargetMomentum,
                      &mpcd::ReverseNonequilibriumShearFlow::setTargetMomentum)
        .def_property_readonly("summed_exchanged_momentum",
                               &mpcd::ReverseNonequilibriumShearFlow::getSummedExchangedMomentum);
    }
    } // namespace detail
    } // namespace mpcd
    } // end namespace hoomd
