// Copyright (c) 2009-2024 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

/*!
 * \file mpcd/CellThermoCompute.cc
 * \brief Definition of mpcd::CellThermoCompute
 */

#include "CellThermoCompute.h"
#include "ReductionOperators.h"

namespace hoomd
    {
/*!
 * \param sysdef System definition
 */
mpcd::CellThermoCompute::CellThermoCompute(std::shared_ptr<SystemDefinition> sysdef,
                                           std::shared_ptr<mpcd::CellList> cl)
    : Compute(sysdef), m_mpcd_pdata(m_sysdef->getMPCDParticleData()), m_cl(cl),
      m_needs_net_reduce(true), m_cell_vel(m_exec_conf), m_cell_energy(m_exec_conf),
      m_ncells_alloc(0)
    {
    m_exec_conf->msg->notice(5) << "Constructing MPCD CellThermoCompute" << std::endl;

    GPUArray<double> net_properties(mpcd::detail::thermo_index::num_quantities, m_exec_conf);
    m_net_properties.swap(net_properties);

#ifdef ENABLE_MPI
    if (m_exec_conf->getNRanks() > 1)
        {
        m_use_mpi = true;
        m_vel_comm = std::make_shared<mpcd::CellCommunicator>(m_sysdef, m_cl);
        m_energy_comm = std::make_shared<mpcd::CellCommunicator>(m_sysdef, m_cl);
        }
    else
        {
        m_use_mpi = false;
        }
#endif // ENABLE_MPI

    // the thermo properties need to be recomputed if the virtual particles change
    m_mpcd_pdata->getNumVirtualSignal()
        .connect<mpcd::CellThermoCompute, &mpcd::CellThermoCompute::slotNumVirtual>(this);
    }

mpcd::CellThermoCompute::~CellThermoCompute()
    {
    m_exec_conf->msg->notice(5) << "Destroying MPCD CellThermoCompute" << std::endl;
    m_mpcd_pdata->getNumVirtualSignal()
        .disconnect<mpcd::CellThermoCompute, &mpcd::CellThermoCompute::slotNumVirtual>(this);
    }

void mpcd::CellThermoCompute::compute(uint64_t timestep)
    {
    Compute::compute(timestep);
    // check if computation should proceed, and always mark the calculation as occurring at this
    // timestep, even if forced
    if (!shouldCompute(timestep))
        return;
    m_last_computed = timestep;

    // cell list needs to be up to date first
    m_cl->compute(timestep);

    // ensure optional flags are up to date
    updateFlags();

    const unsigned int ncells = m_cl->getNCells();
    if (ncells != m_ncells_alloc)
        {
        reallocate(ncells);
        }

    computeCellProperties(timestep);
    m_needs_net_reduce = true;
    }

void mpcd::CellThermoCompute::computeCellProperties(uint64_t timestep)
    {
/*
 * In MPI simulations, begin by calculating the velocities and energies of
 * cells that lie along the boundaries. These values will then be communicated
 * while calculations proceed on the inner cells.
 */
#ifdef ENABLE_MPI
    if (m_use_mpi)
        {
        beginOuterCellProperties();

        m_vel_comm->begin(m_cell_vel, mpcd::detail::CellVelocityPackOp());
        if (m_flags[mpcd::detail::thermo_options::energy])
            m_energy_comm->begin(m_cell_energy, mpcd::detail::CellEnergyPackOp());
        }
#endif // ENABLE_MPI

    /*
     * While communication is occurring on the outer cells, do the full calculation
     * on the inner cells. In non-MPI simulations, only this part happens.
     */
    calcInnerCellProperties();

    /*
     * Execute any additional callbacks that can be overlapped with outer communication.
     */
    if (!m_callbacks.empty())
        m_callbacks.emit(timestep);

/*
 * In MPI simulations, we need to finalize the communication on the outer cells
 * and normalize the communicated data.
 */
#ifdef ENABLE_MPI
    if (m_use_mpi)
        {
        if (m_flags[mpcd::detail::thermo_options::energy])
            m_energy_comm->finalize(m_cell_energy, mpcd::detail::CellEnergyPackOp());
        m_vel_comm->finalize(m_cell_vel, mpcd::detail::CellVelocityPackOp());

        finishOuterCellProperties();
        }
#endif // ENABLE_MPI
    }

namespace mpcd
    {
namespace detail
    {
//! Sums properties of an MPCD cell on the CPU
/*!
 * This lightweight class is used in both beginOuterCellProperties() and
 * calcInnerCellProperties(). The code has been consolidated into one place
 * here to avoid some duplication.
 */
struct CellPropertySum
    {
    //! Constructor
    /*!
     * \param cell_list_ Cell list
     * \param cell_np_ Number of particles per cell
     * \param cli_ Cell list indexer
     * \param vel_ MPCD particle velocities
     * \param mass_ MPCD mass
     * \param embed_vel_ Embedded particle velocities
     * \param embed_idx_ Embedded particle indexes
     * \param N_mpcd_ Number of MPCD particles
     */
    CellPropertySum(const unsigned int* cell_list_,
                    const unsigned int* cell_np_,
                    const Index2D& cli_,
                    const Scalar4* vel_,
                    const Scalar mass_,
                    const Scalar4* embed_vel_,
                    const unsigned int* embed_idx_,
                    const unsigned int N_mpcd_)
        : cell_list(cell_list_), cell_np(cell_np_), cli(cli_), vel(vel_), mass(mass_),
          embed_vel(embed_vel_), embed_idx(embed_idx_), N_mpcd(N_mpcd_)
        {
        }

    //! Computes the total momentum, kinetic energy, and number of particles in a cell
    /*!
     * \param momentum Cell momentum (output)
     * \param ke Cell kinetic energy (output)
     * \param np Number of particles in cell (output)
     * \param cell Index of cell to evaluate
     * \param energy If true, then the kinetic energy is evaluated into \a ke
     */
    inline void compute(double4& momentum,
                        double& ke,
                        unsigned int& np,
                        const unsigned int cell,
                        const bool energy)
        {
        momentum = make_double4(0.0, 0.0, 0.0, 0.0);
        ke = 0.0;
        np = cell_np[cell];

        for (unsigned int offset = 0; offset < np; ++offset)
            {
            // Load particle data
            const unsigned int cur_p = cell_list[cli(offset, cell)];
            double3 vel_i;
            double mass_i;
            if (cur_p < N_mpcd)
                {
                Scalar4 vel_cell = vel[cur_p];
                vel_i = make_double3(vel_cell.x, vel_cell.y, vel_cell.z);
                mass_i = mass;
                }
            else
                {
                Scalar4 vel_m = embed_vel[embed_idx[cur_p - N_mpcd]];
                vel_i = make_double3(vel_m.x, vel_m.y, vel_m.z);
                mass_i = vel_m.w;
                }

            // add momentum
            momentum.x += mass_i * vel_i.x;
            momentum.y += mass_i * vel_i.y;
            momentum.z += mass_i * vel_i.z;
            momentum.w += mass_i;

            // also compute ke of the particle
            if (energy)
                ke += 0.5 * mass_i * (vel_i.x * vel_i.x + vel_i.y * vel_i.y + vel_i.z * vel_i.z);
            }
        }

    const unsigned int* cell_list; //!< Cell list
    const unsigned int* cell_np;   //!< Number of particles per cell
    const Index2D cli;             //!< Cell list indexer

    const Scalar4* vel;            //!< MPCD particle velocities
    const Scalar mass;             //!< MPCD particle mass
    const Scalar4* embed_vel;      //!< Embedded particle velocities
    const unsigned int* embed_idx; //!< Embedded particle indexes
    const unsigned int N_mpcd;     //!< Number of MPCD particles
    };
    } // end namespace detail
    } // end namespace mpcd

#ifdef ENABLE_MPI
void mpcd::CellThermoCompute::beginOuterCellProperties()
    {
    // Cell list
    ArrayHandle<unsigned int> h_cell_list(m_cl->getCellList(),
                                          access_location::host,
                                          access_mode::read);
    ArrayHandle<unsigned int> h_cell_np(m_cl->getCellSizeArray(),
                                        access_location::host,
                                        access_mode::read);
    const Index2D& cli = m_cl->getCellListIndexer();

    // MPCD particle data
    ArrayHandle<Scalar4> h_vel(m_mpcd_pdata->getVelocities(),
                               access_location::host,
                               access_mode::read);
    const Scalar mpcd_mass = m_mpcd_pdata->getMass();
    const unsigned int N_mpcd = m_mpcd_pdata->getN() + m_mpcd_pdata->getNVirtual();

    // Embedded particle data
    std::unique_ptr<ArrayHandle<Scalar4>> h_embed_vel;
    std::unique_ptr<ArrayHandle<unsigned int>> h_embed_member_idx;
    if (m_cl->getEmbeddedGroup())
        {
        h_embed_vel.reset(new ArrayHandle<Scalar4>(m_pdata->getVelocities(),
                                                   access_location::host,
                                                   access_mode::read));
        h_embed_member_idx.reset(
            new ArrayHandle<unsigned int>(m_cl->getEmbeddedGroup()->getIndexArray(),
                                          access_location::host,
                                          access_mode::read));
        }

    // Cell properties
    ArrayHandle<double4> h_cell_vel(m_cell_vel, access_location::host, access_mode::overwrite);
    ArrayHandle<double3> h_cell_energy(m_cell_energy,
                                       access_location::host,
                                       access_mode::overwrite);
    ArrayHandle<unsigned int> h_cells(m_vel_comm->getCells(),
                                      access_location::host,
                                      access_mode::read);
    mpcd::detail::CellPropertySum summer(h_cell_list.data,
                                         h_cell_np.data,
                                         cli,
                                         h_vel.data,
                                         mpcd_mass,
                                         (m_cl->getEmbeddedGroup()) ? h_embed_vel->data : NULL,
                                         (m_cl->getEmbeddedGroup()) ? h_embed_member_idx->data
                                                                    : NULL,
                                         N_mpcd);

    // Loop over all outer cells and compute total momentum, mass, energy
    const bool need_energy = m_flags[mpcd::detail::thermo_options::energy];
    for (unsigned int idx = 0; idx < m_vel_comm->getNCells(); ++idx)
        {
        const unsigned int cur_cell = h_cells.data[idx];

        // compute the cell properties
        double4 momentum;
        double ke(0.0);
        unsigned int np(0);
        summer.compute(momentum, ke, np, cur_cell, need_energy);

        h_cell_vel.data[cur_cell] = make_double4(momentum.x, momentum.y, momentum.z, momentum.w);
        if (need_energy)
            h_cell_energy.data[cur_cell] = make_double3(ke, 0.0, __int_as_double(np));
        }
    }

void mpcd::CellThermoCompute::finishOuterCellProperties()
    {
    ArrayHandle<double4> h_cell_vel(m_cell_vel, access_location::host, access_mode::readwrite);
    ArrayHandle<double3> h_cell_energy(m_cell_energy,
                                       access_location::host,
                                       access_mode::readwrite);
    ArrayHandle<unsigned int> h_cells(m_vel_comm->getCells(),
                                      access_location::host,
                                      access_mode::read);

    // Loop over all outer cells and normalize the summed quantities
    const bool need_energy = m_flags[mpcd::detail::thermo_options::energy];
    for (unsigned int idx = 0; idx < m_vel_comm->getNCells(); ++idx)
        {
        const unsigned int cur_cell = h_cells.data[idx];

        // average cell properties if the cell has mass
        const double4 cell_vel = h_cell_vel.data[cur_cell];
        double3 vel_cm = make_double3(cell_vel.x, cell_vel.y, cell_vel.z);
        const double mass = cell_vel.w;

        if (mass > 0.)
            {
            // average velocity is only defined when there is some mass in the cell
            vel_cm.x /= mass;
            vel_cm.y /= mass;
            vel_cm.z /= mass;
            }
        h_cell_vel.data[cur_cell] = make_double4(vel_cm.x, vel_cm.y, vel_cm.z, mass);

        if (need_energy)
            {
            const double3 cell_energy = h_cell_energy.data[cur_cell];
            const double ke = cell_energy.x;
            double temp(0.0);
            const unsigned int np = __double_as_int(cell_energy.z);
            // temperature is only defined for 2 or more particles
            if (np > 1)
                {
                const double ke_cm
                    = 0.5 * mass
                      * (vel_cm.x * vel_cm.x + vel_cm.y * vel_cm.y + vel_cm.z * vel_cm.z);
                temp = 2. * (ke - ke_cm) / (m_sysdef->getNDimensions() * (np - 1));
                }
            h_cell_energy.data[cur_cell] = make_double3(ke, temp, __int_as_double(np));
            }
        }
    }
#endif // ENABLE_MPI

void mpcd::CellThermoCompute::calcInnerCellProperties()
    {
    // Cell list
    const Index2D& cli = m_cl->getCellListIndexer();
    ArrayHandle<unsigned int> h_cell_list(m_cl->getCellList(),
                                          access_location::host,
                                          access_mode::read);
    ArrayHandle<unsigned int> h_cell_np(m_cl->getCellSizeArray(),
                                        access_location::host,
                                        access_mode::read);

    // MPCD particle data
    const unsigned int N_mpcd = m_mpcd_pdata->getN() + m_mpcd_pdata->getNVirtual();
    const Scalar mpcd_mass = m_mpcd_pdata->getMass();
    ArrayHandle<Scalar4> h_vel(m_mpcd_pdata->getVelocities(),
                               access_location::host,
                               access_mode::read);

    // Embedded particle data
    std::unique_ptr<ArrayHandle<Scalar4>> h_embed_vel;
    std::unique_ptr<ArrayHandle<unsigned int>> h_embed_member_idx;
    if (m_cl->getEmbeddedGroup())
        {
        h_embed_vel.reset(new ArrayHandle<Scalar4>(m_pdata->getVelocities(),
                                                   access_location::host,
                                                   access_mode::read));
        h_embed_member_idx.reset(
            new ArrayHandle<unsigned int>(m_cl->getEmbeddedGroup()->getIndexArray(),
                                          access_location::host,
                                          access_mode::read));
        }

    // Cell properties
    ArrayHandle<double4> h_cell_vel(m_cell_vel, access_location::host, access_mode::readwrite);
    ArrayHandle<double3> h_cell_energy(m_cell_energy,
                                       access_location::host,
                                       access_mode::readwrite);
    mpcd::detail::CellPropertySum summer(h_cell_list.data,
                                         h_cell_np.data,
                                         cli,
                                         h_vel.data,
                                         mpcd_mass,
                                         (m_cl->getEmbeddedGroup()) ? h_embed_vel->data : NULL,
                                         (m_cl->getEmbeddedGroup()) ? h_embed_member_idx->data
                                                                    : NULL,
                                         N_mpcd);

    // determine which cells are inner
    uint3 lo, hi;
    const Index3D& ci = m_cl->getCellIndexer();
#ifdef ENABLE_MPI
    if (m_use_mpi)
        {
        auto num_comm_cells = m_cl->getNComm();
        lo = make_uint3(num_comm_cells[static_cast<unsigned int>(mpcd::detail::face::west)],
                        num_comm_cells[static_cast<unsigned int>(mpcd::detail::face::south)],
                        num_comm_cells[static_cast<unsigned int>(mpcd::detail::face::down)]);
        hi = make_uint3(
            ci.getW() - num_comm_cells[static_cast<unsigned int>(mpcd::detail::face::east)],
            ci.getH() - num_comm_cells[static_cast<unsigned int>(mpcd::detail::face::north)],
            ci.getD() - num_comm_cells[static_cast<unsigned int>(mpcd::detail::face::up)]);
        }
    else
#endif // ENABLE_MPI
        {
        lo = make_uint3(0, 0, 0);
        hi = m_cl->getDim();
        }

    // iterate over all of the inner cells and compute average velocity, energy, temperature
    const bool need_energy = m_flags[mpcd::detail::thermo_options::energy];
    for (unsigned int k = lo.z; k < hi.z; ++k)
        {
        for (unsigned int j = lo.y; j < hi.y; ++j)
            {
            for (unsigned int i = lo.x; i < hi.x; ++i)
                {
                const unsigned int cur_cell = ci(i, j, k);

                // compute the cell properties
                double4 momentum;
                double ke(0.0);
                unsigned int np(0);
                summer.compute(momentum, ke, np, cur_cell, need_energy);

                const double mass = momentum.w;
                double3 vel_cm = make_double3(0.0, 0.0, 0.0);
                if (mass > 0.)
                    {
                    vel_cm.x = momentum.x / mass;
                    vel_cm.y = momentum.y / mass;
                    vel_cm.z = momentum.z / mass;
                    }

                h_cell_vel.data[cur_cell] = make_double4(vel_cm.x, vel_cm.y, vel_cm.z, mass);
                if (need_energy)
                    {
                    double temp(0.0);
                    if (np > 1)
                        {
                        const double ke_cm
                            = 0.5 * mass
                              * (vel_cm.x * vel_cm.x + vel_cm.y * vel_cm.y + vel_cm.z * vel_cm.z);
                        temp = 2. * (ke - ke_cm) / (m_sysdef->getNDimensions() * (np - 1));
                        }
                    h_cell_energy.data[cur_cell] = make_double3(ke, temp, __int_as_double(np));
                    }
                } // i
            } // j
        } // k
    }

void mpcd::CellThermoCompute::computeNetProperties()
    {
    // first reduce the properties on the rank
    unsigned int n_temp_cells = 0;
        {
        const Index3D& ci = m_cl->getCellIndexer();
        uint3 upper = make_uint3(ci.getW(), ci.getH(), ci.getD());
#ifdef ENABLE_MPI
        // in MPI, remove duplicate cells along direction of communication
        if (m_use_mpi)
            {
            auto num_comm = m_cl->getNComm();
            upper.x -= num_comm[static_cast<unsigned int>(mpcd::detail::face::east)];
            upper.y -= num_comm[static_cast<unsigned int>(mpcd::detail::face::north)];
            upper.z -= num_comm[static_cast<unsigned int>(mpcd::detail::face::up)];
            }
#endif // ENABLE_MPI

        ArrayHandle<double4> h_cell_vel(m_cell_vel, access_location::host, access_mode::read);
        ArrayHandle<double3> h_cell_energy(m_cell_energy, access_location::host, access_mode::read);

        const bool need_energy = m_flags[mpcd::detail::thermo_options::energy];

        double3 net_momentum = make_double3(0, 0, 0);
        double energy(0.0), temp(0.0);
        for (unsigned int k = 0; k < upper.z; ++k)
            {
            for (unsigned int j = 0; j < upper.y; ++j)
                {
                for (unsigned int i = 0; i < upper.x; ++i)
                    {
                    const unsigned int idx = ci(i, j, k);

                    const double4 cell_vel_mass = h_cell_vel.data[idx];
                    const double3 cell_vel
                        = make_double3(cell_vel_mass.x, cell_vel_mass.y, cell_vel_mass.z);
                    const double cell_mass = cell_vel_mass.w;

                    net_momentum.x += cell_mass * cell_vel.x;
                    net_momentum.y += cell_mass * cell_vel.y;
                    net_momentum.z += cell_mass * cell_vel.z;

                    if (need_energy)
                        {
                        const double3 cell_energy = h_cell_energy.data[idx];
                        energy += cell_energy.x;

                        if (__double_as_int(cell_energy.z) > 1)
                            {
                            temp += cell_energy.y;
                            ++n_temp_cells;
                            }
                        }
                    }
                }
            }

        ArrayHandle<double> h_net_properties(m_net_properties,
                                             access_location::host,
                                             access_mode::overwrite);
        h_net_properties.data[mpcd::detail::thermo_index::momentum_x] = net_momentum.x;
        h_net_properties.data[mpcd::detail::thermo_index::momentum_y] = net_momentum.y;
        h_net_properties.data[mpcd::detail::thermo_index::momentum_z] = net_momentum.z;

        h_net_properties.data[mpcd::detail::thermo_index::energy] = energy;
        h_net_properties.data[mpcd::detail::thermo_index::temperature] = temp;
        }

#ifdef ENABLE_MPI
    if (m_use_mpi)
        {
        ArrayHandle<double> h_net_properties(m_net_properties,
                                             access_location::host,
                                             access_mode::readwrite);
        MPI_Allreduce(MPI_IN_PLACE,
                      h_net_properties.data,
                      mpcd::detail::thermo_index::num_quantities,
                      MPI_DOUBLE,
                      MPI_SUM,
                      m_exec_conf->getMPICommunicator());

        MPI_Allreduce(MPI_IN_PLACE,
                      &n_temp_cells,
                      1,
                      MPI_UNSIGNED,
                      MPI_SUM,
                      m_exec_conf->getMPICommunicator());
        }
#endif // ENABLE_MPI

    if (n_temp_cells > 0)
        {
        ArrayHandle<double> h_net_properties(m_net_properties,
                                             access_location::host,
                                             access_mode::readwrite);
        h_net_properties.data[mpcd::detail::thermo_index::temperature] /= (double)n_temp_cells;
        }

    m_needs_net_reduce = false;
    }

void mpcd::CellThermoCompute::updateFlags()
    {
    mpcd::detail::ThermoFlags flags;

    if (!m_flag_signal.empty())
        {
        m_flag_signal.emit_accumulate([&](mpcd::detail::ThermoFlags f) { flags |= f; });
        }

    m_flags = flags;
    }

/*!
 * \param ncells Number of cells
 */
void mpcd::CellThermoCompute::reallocate(unsigned int ncells)
    {
    // Grow arrays to match the size if necessary
    m_cell_vel.resize(ncells);
    m_cell_energy.resize(ncells);

    m_ncells_alloc = ncells;
    }

/*!
 * \param m Python module
 */
void mpcd::detail::export_CellThermoCompute(pybind11::module& m)
    {
    pybind11::class_<mpcd::CellThermoCompute, Compute, std::shared_ptr<mpcd::CellThermoCompute>>(
        m,
        "CellThermoCompute")
        .def(pybind11::init<std::shared_ptr<SystemDefinition>, std::shared_ptr<mpcd::CellList>>());
    }

    } // end namespace hoomd
