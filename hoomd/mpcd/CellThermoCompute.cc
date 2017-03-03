// Copyright (c) 2009-2017 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.

// Maintainer: mphoward

/*!
 * \file mpcd/CellThermoCompute.cc
 * \brief Definition of mpcd::CellThermoCompute
 */

#include "CellThermoCompute.h"
#include "ReductionOperators.h"

mpcd::CellThermoCompute::CellThermoCompute(std::shared_ptr<mpcd::SystemData> sysdata,
                                           const std::string& suffix)
        : Compute(sysdata->getSystemDefinition()),
          m_mpcd_pdata(sysdata->getParticleData()),
          m_cl(sysdata->getCellList()),
          m_needs_net_reduce(true), m_cell_vel(m_exec_conf), m_cell_energy(m_exec_conf),
          m_ncells_alloc(0)
    {
    assert(m_mpcd_pdata);
    assert(m_cl);
    m_exec_conf->msg->notice(5) << "Constructing MPCD CellThermoCompute" << std::endl;

    GPUArray<double> net_properties(mpcd::detail::thermo_index::num_quantities, m_exec_conf);
    m_net_properties.swap(net_properties);

    #ifdef ENABLE_MPI
    if (m_exec_conf->getNRanks() > 1)
        {
        m_cell_comm = std::make_shared<mpcd::CellCommunicator>(m_sysdef, m_cl);
        }
    #endif // ENABLE_MPI

    // quantities supplied to the logger
    m_logname_list.push_back(std::string("mpcd_momentum_x") + suffix);
    m_logname_list.push_back(std::string("mpcd_momentum_y") + suffix);
    m_logname_list.push_back(std::string("mpcd_momentum_z") + suffix);
    m_logname_list.push_back(std::string("mpcd_energy") + suffix);
    m_logname_list.push_back(std::string("mpcd_temperature") + suffix);
    }

mpcd::CellThermoCompute::~CellThermoCompute()
    {
    m_exec_conf->msg->notice(5) << "Destroying MPCD CellThermoCompute" << std::endl;
    }

void mpcd::CellThermoCompute::compute(unsigned int timestep)
    {
    if (!shouldCompute(timestep)) return;

    // cell list needs to be up to date first
    m_cl->compute(timestep);

    const unsigned int ncells = m_cl->getNCells();
    if (ncells != m_ncells_alloc)
        {
        reallocate(ncells);
        }

    computeCellProperties();
    m_needs_net_reduce = true;
    }

std::vector<std::string> mpcd::CellThermoCompute::getProvidedLogQuantities()
    {
    return m_logname_list;
    }

Scalar mpcd::CellThermoCompute::getLogValue(const std::string& quantity, unsigned int timestep)
    {
    compute(timestep);
    if (quantity == m_logname_list[0])
        {
        return getNetMomentum().x;
        }
    else if (quantity == m_logname_list[1])
        {
        return getNetMomentum().y;
        }
    else if (quantity == m_logname_list[2])
        {
        return getNetMomentum().z;
        }
    else if (quantity == m_logname_list[3])
        {
        return getNetEnergy();
        }
    else if (quantity == m_logname_list[4])
        {
        return getTemperature();
        }
    else
        {
        m_exec_conf->msg->error() << "mpcd: " << quantity << " is not a valid log quantity" << std::endl;
        throw std::runtime_error("Error getting MPCD log value");
        }
    }

void mpcd::CellThermoCompute::computeCellProperties()
    {
    // Cell list
    const unsigned int ncells = m_cl->getNCells();
    const Index2D& cli = m_cl->getCellListIndexer();
    ArrayHandle<unsigned int> h_cell_list(m_cl->getCellList(), access_location::host, access_mode::read);
    ArrayHandle<unsigned int> h_cell_np(m_cl->getCellSizeArray(), access_location::host, access_mode::read);

    // MPCD particle data
    const unsigned int N_mpcd = m_mpcd_pdata->getN();
    const Scalar mpcd_mass = m_mpcd_pdata->getMass();
    ArrayHandle<Scalar4> h_vel(m_mpcd_pdata->getVelocities(), access_location::host, access_mode::read);

    // Embedded particle data
    std::unique_ptr< ArrayHandle<Scalar4> > h_embed_vel;
    std::unique_ptr< ArrayHandle<unsigned int> > h_embed_member_idx;
    if (m_cl->getEmbeddedGroup())
        {
        h_embed_vel.reset(new ArrayHandle<Scalar4>(m_pdata->getVelocities(), access_location::host, access_mode::read));
        h_embed_member_idx.reset(new ArrayHandle<unsigned int>(m_cl->getEmbeddedGroup()->getIndexArray(), access_location::host, access_mode::read));
        }

    // Sum the momentum, mass, and kinetic energy per-cell
        {
        ArrayHandle<Scalar4> h_cell_vel(m_cell_vel, access_location::host, access_mode::overwrite);
        ArrayHandle<Scalar3> h_cell_energy(m_cell_energy, access_location::host, access_mode::overwrite);
        for (unsigned int cur_cell = 0; cur_cell < ncells; ++cur_cell)
            {
            const unsigned int np = h_cell_np.data[cur_cell];
            double4 momentum = make_double4(0.0, 0.0, 0.0, 0.0);
            double ke(0.0);

            for (unsigned int cur_offset = 0; cur_offset < np; ++cur_offset)
                {
                // Load particle data
                const unsigned int cur_p = h_cell_list.data[cli(cur_offset, cur_cell)];
                double3 vel_i;
                double mass_i;
                if (cur_p < N_mpcd)
                    {
                    Scalar4 vel_cell = h_vel.data[cur_p];
                    vel_i = make_double3(vel_cell.x, vel_cell.y, vel_cell.z);
                    mass_i = mpcd_mass;
                    }
                else
                    {
                    Scalar4 vel_m = h_embed_vel->data[h_embed_member_idx->data[cur_p - N_mpcd]];
                    vel_i = make_double3(vel_m.x, vel_m.y, vel_m.z);
                    mass_i = vel_m.w;
                    }

                // add momentum
                momentum.x += mass_i * vel_i.x;
                momentum.y += mass_i * vel_i.y;
                momentum.z += mass_i * vel_i.z;
                momentum.w += mass_i;

                // also compute ke of the particle
                ke += (double)(0.5) * mass_i * (vel_i.x * vel_i.x + vel_i.y * vel_i.y + vel_i.z * vel_i.z);
                }

            h_cell_vel.data[cur_cell] = momentum;
            h_cell_energy.data[cur_cell] = make_scalar3(ke, 0.0, __int_as_scalar(np));
            }
        }

    #ifdef ENABLE_MPI
    // Reduce cell properties across ranks
    if (m_cell_comm)
        {
        m_cell_comm->reduce(m_cell_vel, mpcd::ops::Sum());
        m_cell_comm->reduce(m_cell_energy, mpcd::detail::SumScalar2Int());
        }
    #endif // ENABLE_MPI

        {
        ArrayHandle<Scalar4> h_cell_vel(m_cell_vel, access_location::host, access_mode::readwrite);
        ArrayHandle<Scalar3> h_cell_energy(m_cell_energy, access_location::host, access_mode::readwrite);
        for (unsigned int cur_cell = 0; cur_cell < ncells; ++cur_cell)
            {
            // average cell properties if the cell has mass
            const Scalar4 cell_vel = h_cell_vel.data[cur_cell];
            Scalar3 vel_cm = make_scalar3(cell_vel.x, cell_vel.y, cell_vel.z);
            const Scalar mass = cell_vel.w;

            const Scalar3 cell_energy = h_cell_energy.data[cur_cell];
            const Scalar ke = cell_energy.x;
            Scalar temp(0.0);
            const unsigned int np = __scalar_as_int(cell_energy.z);
            if (mass > 0.)
                {
                // average velocity is only defined when there is some mass in the cell
                vel_cm /= mass;

                // temperature is only defined for 2 or more particles
                if (np > 1)
                    {
                    const Scalar ke_cm = Scalar(0.5) * mass * dot(vel_cm, vel_cm);
                    temp = Scalar(2.) * (ke - ke_cm) / Scalar(m_sysdef->getNDimensions() * (np-1));
                    }
                }
            h_cell_vel.data[cur_cell] = make_scalar4(vel_cm.x, vel_cm.y, vel_cm.z, mass);
            h_cell_energy.data[cur_cell] = make_scalar3(ke, temp, __int_as_scalar(np));
            }
        }
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
        if (m_cell_comm)
            {
            auto num_comm = m_cl->getNComm();
            upper.x -= num_comm[static_cast<unsigned int>(mpcd::detail::face::east)];
            upper.y -= num_comm[static_cast<unsigned int>(mpcd::detail::face::north)];
            upper.z -= num_comm[static_cast<unsigned int>(mpcd::detail::face::up)];
            }
        #endif // ENABLE_MPI

        ArrayHandle<Scalar4> h_cell_vel(m_cell_vel, access_location::host, access_mode::read);
        ArrayHandle<Scalar3> h_cell_energy(m_cell_energy, access_location::host, access_mode::read);

        double3 net_momentum = make_double3(0,0,0);
        double energy(0.0), temp(0.0);
        for (unsigned int k=0; k < upper.z; ++k)
            {
            for (unsigned int j=0; j < upper.y; ++j)
                {
                for (unsigned int i=0; i < upper.x; ++i)
                    {
                    const unsigned int idx = ci(i,j,k);

                    const Scalar4 cell_vel_mass = h_cell_vel.data[idx];
                    const double3 cell_vel = make_double3(cell_vel_mass.x, cell_vel_mass.y, cell_vel_mass.z);
                    const double cell_mass = cell_vel_mass.w;

                    net_momentum.x += cell_mass * cell_vel.x;
                    net_momentum.y += cell_mass * cell_vel.y;
                    net_momentum.z += cell_mass * cell_vel.z;

                    const Scalar3 cell_energy = h_cell_energy.data[idx];
                    energy += cell_energy.x;

                    if (__scalar_as_int(cell_energy.z) > 1)
                        {
                        temp += cell_energy.y;
                        ++n_temp_cells;
                        }
                    }
                }
            }

        ArrayHandle<double> h_net_properties(m_net_properties, access_location::host, access_mode::overwrite);
        h_net_properties.data[mpcd::detail::thermo_index::momentum_x] = net_momentum.x;
        h_net_properties.data[mpcd::detail::thermo_index::momentum_y] = net_momentum.y;
        h_net_properties.data[mpcd::detail::thermo_index::momentum_z] = net_momentum.z;

        h_net_properties.data[mpcd::detail::thermo_index::energy] = energy;
        h_net_properties.data[mpcd::detail::thermo_index::temperature] = temp;
        }

    #ifdef ENABLE_MPI
    if (m_cell_comm)
        {
        ArrayHandle<double> h_net_properties(m_net_properties, access_location::host, access_mode::readwrite);
        MPI_Allreduce(MPI_IN_PLACE,
                      h_net_properties.data,
                      mpcd::detail::thermo_index::num_quantities,
                      MPI_DOUBLE,
                      MPI_SUM,
                      m_exec_conf->getMPICommunicator());

        MPI_Allreduce(MPI_IN_PLACE, &n_temp_cells, 1, MPI_UNSIGNED, MPI_SUM, m_exec_conf->getMPICommunicator());
        }
    #endif // ENABLE_MPI

    if (n_temp_cells > 0)
        {
        ArrayHandle<double> h_net_properties(m_net_properties, access_location::host, access_mode::readwrite);
        h_net_properties.data[mpcd::detail::thermo_index::temperature] /= (double)n_temp_cells;
        }

    m_needs_net_reduce = false;
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
    namespace py = pybind11;

    py::class_<mpcd::CellThermoCompute, std::shared_ptr<mpcd::CellThermoCompute> >
        (m, "CellThermoCompute", py::base<Compute>())
        .def(py::init< std::shared_ptr<mpcd::SystemData> >())
        .def(py::init< std::shared_ptr<mpcd::SystemData>, const std::string& >());
    }
