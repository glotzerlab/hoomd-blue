// Copyright (c) 2009-2024 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

/*!
 * \file mpcd/SRDCollisionMethod.h
 * \brief Definition of mpcd::SRDCollisionMethod
 */

#include "SRDCollisionMethod.h"
#include "hoomd/RNGIdentifiers.h"
#include "hoomd/RandomNumbers.h"

namespace hoomd
    {
mpcd::SRDCollisionMethod::SRDCollisionMethod(std::shared_ptr<SystemDefinition> sysdef,
                                             unsigned int cur_timestep,
                                             unsigned int period,
                                             int phase,
                                             Scalar angle)
    : mpcd::CollisionMethod(sysdef, cur_timestep, period, phase), m_rotvec(m_exec_conf),
      m_angle(angle), m_factors(m_exec_conf)
    {
    m_exec_conf->msg->notice(5) << "Constructing MPCD SRD collision method" << std::endl;
    }

mpcd::SRDCollisionMethod::~SRDCollisionMethod()
    {
    m_exec_conf->msg->notice(5) << "Destroying MPCD SRD collision method" << std::endl;
    detachCallbacks();
    }

void mpcd::SRDCollisionMethod::rule(uint64_t timestep)
    {
    m_thermo->compute(timestep);

    // resize the rotation vectors and rescale factors
    m_rotvec.resize(m_cl->getNCells());
    if (m_T)
        {
        m_factors.resize(m_cl->getNCells());
        }

    // draw rotation vectors for each cell
    drawRotationVectors(timestep);

    // apply collision rule
    rotate(timestep);
    }

void mpcd::SRDCollisionMethod::drawRotationVectors(uint64_t timestep)
    {
    // cell indexers and rotation vectors
    const Index3D& ci = m_cl->getCellIndexer();
    const Index3D& global_ci = m_cl->getGlobalCellIndexer();
    ArrayHandle<double3> h_rotvec(m_rotvec, access_location::host, access_mode::overwrite);

    // optional scale factors
    std::unique_ptr<ArrayHandle<double>> h_factors;
    std::unique_ptr<ArrayHandle<double3>> h_cell_energy;
    Scalar T_set(1.0);
    const bool use_thermostat = (m_T) ? true : false;
    if (use_thermostat)
        {
        h_factors.reset(
            new ArrayHandle<double>(m_factors, access_location::host, access_mode::overwrite));
        h_cell_energy.reset(new ArrayHandle<double3>(m_thermo->getCellEnergies(),
                                                     access_location::host,
                                                     access_mode::read));
        T_set = (*m_T)(timestep);
        }

    uint16_t seed = m_sysdef->getSeed();

    for (unsigned int k = 0; k < ci.getD(); ++k)
        {
        for (unsigned int j = 0; j < ci.getH(); ++j)
            {
            for (unsigned int i = 0; i < ci.getW(); ++i)
                {
                const int3 global_cell = m_cl->getGlobalCell(make_int3(i, j, k));
                const unsigned int global_idx
                    = global_ci(global_cell.x, global_cell.y, global_cell.z);
                const unsigned int idx = ci(i, j, k);

                // Initialize the PRNG using the current cell index, timestep, and seed for the hash
                hoomd::RandomGenerator rng(
                    hoomd::Seed(hoomd::RNGIdentifier::SRDCollisionMethod, timestep, seed),
                    hoomd::Counter(global_idx));

                // draw rotation vector off the surface of the sphere
                double3 rotvec;
                hoomd::SpherePointGenerator<double> sphgen;
                sphgen(rng, rotvec);
                h_rotvec.data[idx] = rotvec;

                if (use_thermostat)
                    {
                    const double3 cell_energy = h_cell_energy->data[idx];
                    const unsigned int np = __double_as_int(cell_energy.z);
                    double factor = 1.0;
                    if (np > 1)
                        {
                        // the total number of degrees of freedom in the cell divided by 2
                        const double alpha = m_sysdef->getNDimensions() * (np - 1) / (double)2.;

                        // draw a random kinetic energy for the cell at the set temperature
                        hoomd::GammaDistribution<double> gamma_gen(alpha, T_set);
                        const double rand_ke = gamma_gen(rng);

                        // generate the scale factor from the current temperature
                        // (don't use the kinetic energy of this cell, since this
                        // is total not relative to COM)
                        const double cur_ke = alpha * cell_energy.y;
                        factor = (cur_ke > 0.) ? fast::sqrt(rand_ke / cur_ke) : 1.;
                        }
                    h_factors->data[idx] = factor;
                    }
                }
            }
        }
    }

void mpcd::SRDCollisionMethod::rotate(uint64_t timestep)
    {
    // acquire MPCD particle data
    ArrayHandle<Scalar4> h_vel(m_mpcd_pdata->getVelocities(),
                               access_location::host,
                               access_mode::readwrite);
    const unsigned int N_mpcd = m_mpcd_pdata->getN() + m_mpcd_pdata->getNVirtual();
    unsigned int N_tot = N_mpcd;
    // acquire additionally embedded particle data
    std::unique_ptr<ArrayHandle<unsigned int>> h_embed_cell_ids;
    std::unique_ptr<ArrayHandle<Scalar4>> h_vel_embed;
    std::unique_ptr<ArrayHandle<unsigned int>> h_embed_group;
    if (m_embed_group)
        {
        h_embed_group.reset(new ArrayHandle<unsigned int>(m_embed_group->getIndexArray(),
                                                          access_location::host,
                                                          access_mode::read));
        h_vel_embed.reset(new ArrayHandle<Scalar4>(m_pdata->getVelocities(),
                                                   access_location::host,
                                                   access_mode::readwrite));
        h_embed_cell_ids.reset(new ArrayHandle<unsigned int>(m_cl->getEmbeddedGroupCellIds(),
                                                             access_location::host,
                                                             access_mode::read));
        N_tot += m_embed_group->getNumMembers();
        }

    // acquire cell velocities
    ArrayHandle<double4> h_cell_vel(m_thermo->getCellVelocities(),
                                    access_location::host,
                                    access_mode::read);

    // load rotation vector and precompute functions for rotation matrix
    ArrayHandle<double3> h_rotvec(m_rotvec, access_location::host, access_mode::read);
    const double angle_rad = m_angle * M_PI / 180.0;
    const double cos_a = slow::cos(angle_rad);
    const double one_minus_cos_a = 1.0 - cos_a;
    const double sin_a = slow::sin(angle_rad);

    // load scale factors if required
    const bool use_thermostat = (m_T) ? true : false;
    std::unique_ptr<ArrayHandle<double>> h_factors;
    if (use_thermostat)
        {
        h_factors.reset(
            new ArrayHandle<double>(m_factors, access_location::host, access_mode::read));
        }

    for (unsigned int cur_p = 0; cur_p < N_tot; ++cur_p)
        {
        double3 vel;
        unsigned int cell;
        // these properties are needed for the embedded particles only
        unsigned int idx(0);
        double mass(0);
        if (cur_p < N_mpcd)
            {
            const Scalar4 vel_cell = h_vel.data[cur_p];
            vel = make_double3(vel_cell.x, vel_cell.y, vel_cell.z);
            cell = __scalar_as_int(vel_cell.w);
            }
        else
            {
            idx = h_embed_group->data[cur_p - N_mpcd];

            const Scalar4 vel_mass = h_vel_embed->data[idx];
            vel = make_double3(vel_mass.x, vel_mass.y, vel_mass.z);
            mass = vel_mass.w;
            cell = h_embed_cell_ids->data[cur_p - N_mpcd];
            }

        // subtract average velocity
        const double4 avg_vel = h_cell_vel.data[cell];
        vel.x -= avg_vel.x;
        vel.y -= avg_vel.y;
        vel.z -= avg_vel.z;

        // get rotation vector
        double3 rot_vec = h_rotvec.data[cell];

        // perform the rotation in double precision
        // TODO: should we optimize out the matrix construction for the CPU?
        //       Or, consider using vectorization and/or Eigen?
        double3 new_vel;
        new_vel.x = (cos_a + rot_vec.x * rot_vec.x * one_minus_cos_a) * vel.x;
        new_vel.x += (rot_vec.x * rot_vec.y * one_minus_cos_a - sin_a * rot_vec.z) * vel.y;
        new_vel.x += (rot_vec.x * rot_vec.z * one_minus_cos_a + sin_a * rot_vec.y) * vel.z;

        new_vel.y = (cos_a + rot_vec.y * rot_vec.y * one_minus_cos_a) * vel.y;
        new_vel.y += (rot_vec.x * rot_vec.y * one_minus_cos_a + sin_a * rot_vec.z) * vel.x;
        new_vel.y += (rot_vec.y * rot_vec.z * one_minus_cos_a - sin_a * rot_vec.x) * vel.z;

        new_vel.z = (cos_a + rot_vec.z * rot_vec.z * one_minus_cos_a) * vel.z;
        new_vel.z += (rot_vec.x * rot_vec.z * one_minus_cos_a - sin_a * rot_vec.y) * vel.x;
        new_vel.z += (rot_vec.y * rot_vec.z * one_minus_cos_a + sin_a * rot_vec.x) * vel.y;

        // rescale the temperature if thermostatting is enabled
        if (use_thermostat)
            {
            double factor = h_factors->data[cell];
            new_vel.x *= factor;
            new_vel.y *= factor;
            new_vel.z *= factor;
            }

        new_vel.x += avg_vel.x;
        new_vel.y += avg_vel.y;
        new_vel.z += avg_vel.z;

        // set the new velocity
        if (cur_p < N_mpcd)
            {
            h_vel.data[cur_p]
                = make_scalar4(new_vel.x, new_vel.y, new_vel.z, __int_as_scalar(cell));
            }
        else
            {
            h_vel_embed->data[idx] = make_scalar4(new_vel.x, new_vel.y, new_vel.z, mass);
            }
        }
    }

void mpcd::SRDCollisionMethod::setCellList(std::shared_ptr<mpcd::CellList> cl)
    {
    if (cl != m_cl)
        {
        CollisionMethod::setCellList(cl);
        detachCallbacks();
        if (m_cl)
            {
            m_thermo = std::make_shared<mpcd::CellThermoCompute>(m_sysdef, m_cl);
            attachCallbacks();
            }
        else
            {
            m_thermo = std::shared_ptr<mpcd::CellThermoCompute>();
            }
        }
    }

void mpcd::SRDCollisionMethod::attachCallbacks()
    {
    assert(m_thermo);
    m_thermo->getFlagsSignal()
        .connect<mpcd::SRDCollisionMethod, &mpcd::SRDCollisionMethod::getRequestedThermoFlags>(
            this);
    }

void mpcd::SRDCollisionMethod::detachCallbacks()
    {
    if (m_thermo)
        {
        m_thermo->getFlagsSignal()
            .disconnect<mpcd::SRDCollisionMethod,
                        &mpcd::SRDCollisionMethod::getRequestedThermoFlags>(this);
        }
    }

namespace mpcd
    {
namespace detail
    {
/*!
 * \param m Python module to export to
 */
void export_SRDCollisionMethod(pybind11::module& m)
    {
    pybind11::class_<mpcd::SRDCollisionMethod,
                     mpcd::CollisionMethod,
                     std::shared_ptr<mpcd::SRDCollisionMethod>>(m, "SRDCollisionMethod")
        .def(pybind11::
                 init<std::shared_ptr<SystemDefinition>, unsigned int, unsigned int, int, Scalar>())
        .def_property("angle",
                      &mpcd::SRDCollisionMethod::getRotationAngle,
                      &mpcd::SRDCollisionMethod::setRotationAngle)
        .def_property("kT",
                      &mpcd::SRDCollisionMethod::getTemperature,
                      &mpcd::SRDCollisionMethod::setTemperature);
    }
    } // namespace detail
    } // namespace mpcd
    } // end namespace hoomd
