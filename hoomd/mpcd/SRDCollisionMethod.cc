// Copyright (c) 2009-2017 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.

// Maintainer: mphoward

/*!
 * \file mpcd/SRDCollisionMethod.h
 * \brief Definition of mpcd::SRDCollisionMethod
 */

#include "SRDCollisionMethod.h"
#include "hoomd/extern/saruprng.h"

#define MPCD_2PI 6.283185307179586

mpcd::SRDCollisionMethod::SRDCollisionMethod(std::shared_ptr<mpcd::SystemData> sysdata,
                                             unsigned int cur_timestep,
                                             unsigned int period,
                                             int phase,
                                             unsigned int seed,
                                             std::shared_ptr<mpcd::CellThermoCompute> thermo)
    : mpcd::CollisionMethod(sysdata,cur_timestep,period,phase,seed),
      m_thermo(thermo), m_rotvec(m_exec_conf), m_angle(0.0)
    {
    m_exec_conf->msg->notice(5) << "Constructing MPCD SRD collision method" << std::endl;
    }


mpcd::SRDCollisionMethod::~SRDCollisionMethod()
    {
    m_exec_conf->msg->notice(5) << "Destroying MPCD SRD collision method" << std::endl;
    }

void mpcd::SRDCollisionMethod::collide(unsigned int timestep)
    {
    if (!shouldCollide(timestep)) return;

    if (m_prof) m_prof->push("MPCD collide");
    // set random grid shift
    drawGridShift(timestep);
    if (m_prof) m_prof->pop();

    // update cell list and thermo
    m_cl->compute(timestep);
    m_thermo->compute(timestep);

    if (m_prof) m_prof->push(m_exec_conf, "MPCD collide");
    // draw rotation vectors for each cell
    drawRotationVectors(timestep);

    // apply collision rule
    rotate(timestep);
    if (m_prof) m_prof->pop(m_exec_conf);
    }

void mpcd::SRDCollisionMethod::drawRotationVectors(unsigned int timestep)
    {
    // resize the rotation vectors
    m_rotvec.resize(m_cl->getNCells());

    const Index3D& ci = m_cl->getCellIndexer();
    const Index3D& global_ci = m_cl->getGlobalCellIndexer();
    ArrayHandle<double3> h_rotvec(m_rotvec, access_location::host, access_mode::overwrite);

    for (unsigned int k=0; k < ci.getD(); ++k)
        {
        for (unsigned int j=0; j < ci.getH(); ++j)
            {
            for (unsigned int i=0; i < ci.getW(); ++i)
                {
                const int3 global_cell = m_cl->getGlobalCell(make_int3(i,j,k));
                const unsigned int global_idx = global_ci(global_cell.x, global_cell.y, global_cell.z);

                // Initialize the PRNG using the current cell index, timestep, and seed for the hash
                Saru saru(global_idx, timestep, m_seed);

                // calculate the random rotation vector for the cell
                const double theta = saru.d(0, MPCD_2PI);
                const double u = saru.d(-1.0, 1.0);

                /*
                 * Sometimes numbers get drawn really close to -1 or +1, and the machine precision difference is a really
                 * small (negative) number. This causes sqrt() to fail with nan error, so we need to handle those cases by
                 * forcing the sqrt() to 0.0.
                 */
                double sqrtu = 0.0;
                const double one_minus_u2 = 1.0-u*u;
                if (one_minus_u2 > 0.0)
                    {
                    sqrtu = slow::sqrt(one_minus_u2);
                    }

                h_rotvec.data[ci(i,j,k)] = make_double3(sqrtu * slow::cos(theta), sqrtu*slow::sin(theta), u);
                }
            }
        }
    }

void mpcd::SRDCollisionMethod::rotate(unsigned int timestep)
    {
    // acquire MPCD particle data
    ArrayHandle<Scalar4> h_vel(m_mpcd_pdata->getVelocities(), access_location::host, access_mode::readwrite);
    const unsigned int N_mpcd = m_mpcd_pdata->getN();
    unsigned int N_tot = N_mpcd;
    // acquire additionally embedded particle data
    std::unique_ptr< ArrayHandle<unsigned int> > h_embed_cell_ids;
    std::unique_ptr< ArrayHandle<Scalar4> > h_vel_embed;
    std::unique_ptr< ArrayHandle<unsigned int> > h_embed_group;
    if (m_embed_group)
        {
        h_embed_group.reset(new ArrayHandle<unsigned int>(m_embed_group->getIndexArray(), access_location::host, access_mode::read));
        h_vel_embed.reset(new ArrayHandle<Scalar4>(m_pdata->getVelocities(), access_location::host, access_mode::readwrite));
        h_embed_cell_ids.reset(new ArrayHandle<unsigned int>(m_cl->getEmbeddedGroupCellIds(), access_location::host, access_mode::read));
        N_tot += m_embed_group->getNumMembers();
        }

    // acquire cell velocities
    ArrayHandle<Scalar4> h_cell_vel(m_thermo->getCellVelocities(), access_location::host, access_mode::read);

    // load rotation vector and precompute functions for rotation matrix
    ArrayHandle<double3> h_rotvec(m_rotvec, access_location::host, access_mode::read);
    const double cos_a = slow::cos(m_angle);
    const double one_minus_cos_a = 1.0 - cos_a;
    const double sin_a = slow::sin(m_angle);

    for (unsigned int cur_p = 0; cur_p < N_tot; ++cur_p)
        {
        double3 vel;
        unsigned int cell;
        // these properties are needed for the embedded particles only
        unsigned int idx(0); double mass(0);
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
        const Scalar4 avg_vel = h_cell_vel.data[cell];
        vel.x -= avg_vel.x;
        vel.y -= avg_vel.y;
        vel.z -= avg_vel.z;

        // get rotation vector
        double3 rot_vec = h_rotvec.data[cell];

        // perform the rotation in double precision
        // TODO: should we optimize out the matrix construction for the CPU?
        //       Or, consider using vectorization and/or Eigen?
        double3 new_vel;
        new_vel.x = (cos_a + rot_vec.x*rot_vec.x*one_minus_cos_a) * vel.x;
        new_vel.x += (rot_vec.x*rot_vec.y*one_minus_cos_a - sin_a*rot_vec.z) * vel.y;
        new_vel.x += (rot_vec.x*rot_vec.z*one_minus_cos_a + sin_a*rot_vec.y) * vel.z;

        new_vel.y = (cos_a + rot_vec.y*rot_vec.y*one_minus_cos_a) * vel.y;
        new_vel.y += (rot_vec.x*rot_vec.y*one_minus_cos_a + sin_a*rot_vec.z) * vel.x;
        new_vel.y += (rot_vec.y*rot_vec.z*one_minus_cos_a - sin_a*rot_vec.x) * vel.z;

        new_vel.z = (cos_a + rot_vec.z*rot_vec.z*one_minus_cos_a) * vel.z;
        new_vel.z += (rot_vec.x*rot_vec.z*one_minus_cos_a - sin_a*rot_vec.y) * vel.x;
        new_vel.z += (rot_vec.y*rot_vec.z*one_minus_cos_a + sin_a*rot_vec.x) * vel.y;

        new_vel.x += avg_vel.x;
        new_vel.y += avg_vel.y;
        new_vel.z += avg_vel.z;

        // set the new velocity
        if (cur_p < N_mpcd)
            {
            h_vel.data[cur_p] = make_scalar4(new_vel.x, new_vel.y, new_vel.z, __int_as_scalar(cell));
            }
        else
            {
            h_vel_embed->data[idx] = make_scalar4(new_vel.x, new_vel.y, new_vel.z, mass);
            }
        }
    }

/*!
 * \param m Python module to export to
 */
void mpcd::detail::export_SRDCollisionMethod(pybind11::module& m)
    {
    namespace py = pybind11;
    py::class_<mpcd::SRDCollisionMethod, std::shared_ptr<mpcd::SRDCollisionMethod> >
        (m, "SRDCollisionMethod", py::base<mpcd::CollisionMethod>())
        .def(py::init<std::shared_ptr<mpcd::SystemData>,
                      unsigned int,
                      unsigned int,
                      int,
                      unsigned int,
                      std::shared_ptr<mpcd::CellThermoCompute>>())
        .def("setRotationAngle", &mpcd::SRDCollisionMethod::setRotationAngle)
    ;
    }
