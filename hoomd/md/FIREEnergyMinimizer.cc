// Copyright (c) 2009-2019 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.


// Maintainer: jglaser



#include "FIREEnergyMinimizer.h"

using namespace std;
namespace py = pybind11;

/*! \file FIREEnergyMinimizer.h
    \brief Contains code for the FIREEnergyMinimizer class
*/

/*! \param sysdef SystemDefinition this method will act on. Must not be NULL.
    \param dt maximum step size

    \post The method is constructed with the given particle data and a NULL profiler.
*/
FIREEnergyMinimizer::FIREEnergyMinimizer(std::shared_ptr<SystemDefinition> sysdef, Scalar dt)
    :   IntegratorTwoStep(sysdef, dt),
        m_nmin(5),
        m_finc(Scalar(1.1)),
        m_fdec(Scalar(0.5)),
        m_alpha_start(Scalar(0.1)),
        m_falpha(Scalar(0.99)),
        m_ftol(Scalar(1e-1)),
        m_wtol(Scalar(1e-1)),
        m_etol(Scalar(1e-3)),
        m_energy_total(Scalar(0.0)),
        m_old_energy(Scalar(0.0)),
        m_deltaT_max(dt),
        m_deltaT_set(dt/Scalar(10.0)),
        m_run_minsteps(10)
    {
    m_exec_conf->msg->notice(5) << "Constructing FIREEnergyMinimizer" << endl;

    // sanity check
    assert(m_sysdef);
    assert(m_pdata);
    reset();
    }

FIREEnergyMinimizer::~FIREEnergyMinimizer()
    {
    m_exec_conf->msg->notice(5) << "Destroying FIREEnergyMinimizer" << endl;
    }

/*! \param finc is the new fractional increase to set
*/
void FIREEnergyMinimizer::setFinc(Scalar finc)
    {
    if (!(finc > 1.0))
        {
        m_exec_conf->msg->error() << "integrate.mode_minimize_fire: fractional increase in timestep should be > 1" << endl;
        throw runtime_error("Error setting parameters for FIREEnergyMinimizer");
        }
        m_finc = finc;
    }

/*! \param fdec is the new fractional decrease to set
*/
void FIREEnergyMinimizer::setFdec(Scalar fdec)
    {
    if (!(fdec < 1.0 && fdec >= 0.0))
        {
        m_exec_conf->msg->error() << "integrate.mode_minimize_fire: fractional decrease in timestep should be between 0 and 1" << endl;
        throw runtime_error("Error setting parameters for FIREEnergyMinimizer");
        }
        m_fdec = fdec;
    }

/*! \param alpha_start is the new initial coupling parameter to set

The coupling parameter "alpha" enters into the equations of motion as
v = v*(1-alpha) + alpha*(f_unit*|v|).  Thus, the stronger the coupling, the
more important the "f dot v" term.  When the search direction is successful
for > Nmin steps alpha is decreased by falpha.
*/
void FIREEnergyMinimizer::setAlphaStart(Scalar alpha_start)
    {
    if (!(alpha_start < 1.0 && alpha_start > 0.0))
        {
        m_exec_conf->msg->error() << "integrate.mode_minimize_fire: alpha_start should be between 0 and 1" << endl;
        throw runtime_error("Error setting parameters for FIREEnergyMinimizer");
        }
        m_alpha_start = alpha_start;
    }

/*! \param falpha is the fractional decrease in alpha upon finding a valid search direction

The coupling parameter "alpha" enters into the equations of motion as
v = v*(1-alpha) + alpha*(f_unit*|v|).  Thus, the stronger the coupling, the
more important the "f dot v" term.  When the search direction is successful
for > Nmin steps alpha is decreased by falpha.
*/
void FIREEnergyMinimizer::setFalpha(Scalar falpha)
    {
    if (!(falpha < 1.0 && falpha > 0.0))
        {
        m_exec_conf->msg->error() << "integrate.mode_minimize_fire: falpha should be between 0 and 1" << endl;
        throw runtime_error("Error setting parameters for FIREEnergyMinimizer");
        }
        m_falpha = falpha;
    }

void FIREEnergyMinimizer::reset()
    {
    m_converged = false;
    m_n_since_negative = m_nmin+1;
    m_n_since_start = 0;
    m_alpha = m_alpha_start;
    m_was_reset = true;
    m_energy_total = 0.0;

    ArrayHandle<Scalar4> h_vel(m_pdata->getVelocities(), access_location::host, access_mode::readwrite);
    ArrayHandle<Scalar4> h_angmom(m_pdata->getAngularMomentumArray(), access_location::host, access_mode::readwrite);

    unsigned int n = m_pdata->getN();
    for (unsigned int i=0; i<n; i++)
        {
        h_vel.data[i].x = Scalar(0.0);
        h_vel.data[i].y = Scalar(0.0);
        h_vel.data[i].z = Scalar(0.0);
        h_angmom.data[i] = make_scalar4(0,0,0,0);
        }

    setDeltaT(m_deltaT_set);
    }

/*! \param timesteps is the current timestep
*/
void FIREEnergyMinimizer::update(unsigned int timestep)
    {
    if (m_converged)
        return;

    IntegratorTwoStep::update(timestep);

    Scalar Pt(0.0); //translational power
    Scalar Pr(0.0); //rotational power
    Scalar vnorm(0.0);
    Scalar fnorm(0.0);
    Scalar tnorm(0.0);
    Scalar wnorm(0.0);

    // Calculate the per-particle potential energy over particles in the group
    Scalar energy(0.0);

    unsigned int total_group_size = 0;

    {
    const GlobalArray< Scalar4 >& net_force = m_pdata->getNetForce();
    ArrayHandle<Scalar4> h_net_force(net_force, access_location::host, access_mode::read);

    // total potential energy
    double pe_total = 0.0;

    for (auto method = m_methods.begin(); method != m_methods.end(); ++method)
        {
        std::shared_ptr<ParticleGroup> current_group = (*method)->getGroup();
        unsigned int group_size = current_group->getIndexArray().getNumElements();
        total_group_size += group_size;

        for (unsigned int group_idx = 0; group_idx < group_size; group_idx++)
            {
            unsigned int j = current_group->getMemberIndex(group_idx);
            pe_total += (double)h_net_force.data[j].w;
            }
        }

    m_energy_total = pe_total;

    #ifdef ENABLE_MPI
    if (m_pdata->getDomainDecomposition())
        {
        MPI_Allreduce(MPI_IN_PLACE, &pe_total, 1, MPI_HOOMD_SCALAR, MPI_SUM, m_exec_conf->getMPICommunicator());
        MPI_Allreduce(MPI_IN_PLACE, &total_group_size, 1, MPI_INT, MPI_SUM, m_exec_conf->getMPICommunicator());
        }
    #endif

    energy = pe_total/Scalar(total_group_size);
    }


    if (m_was_reset)
        {
        m_was_reset = false;
        m_old_energy = energy + Scalar(100000)*m_etol;
        }

    ArrayHandle<Scalar4> h_vel(m_pdata->getVelocities(), access_location::host, access_mode::readwrite);
    ArrayHandle<Scalar3> h_accel(m_pdata->getAccelerations(), access_location::host, access_mode::readwrite);

    #ifdef ENABLE_MPI
    bool aniso = false;
    #endif

    for (auto method = m_methods.begin(); method != m_methods.end(); ++method)
        {
        std::shared_ptr<ParticleGroup> current_group = (*method)->getGroup();
        unsigned int group_size = current_group->getIndexArray().getNumElements();
        for (unsigned int group_idx = 0; group_idx < group_size; group_idx++)
            {
            unsigned int j = current_group->getMemberIndex(group_idx);
            Pt += h_accel.data[j].x*h_vel.data[j].x + h_accel.data[j].y*h_vel.data[j].y + h_accel.data[j].z*h_vel.data[j].z;
            fnorm += h_accel.data[j].x*h_accel.data[j].x+h_accel.data[j].y*h_accel.data[j].y+h_accel.data[j].z*h_accel.data[j].z;
            vnorm += h_vel.data[j].x*h_vel.data[j].x+ h_vel.data[j].y*h_vel.data[j].y + h_vel.data[j].z*h_vel.data[j].z;
            }

        if ((*method)->getAnisotropic())
            {
            #ifdef ENABLE_MPI
            aniso = true;
            #endif

            ArrayHandle<Scalar4> h_net_torque(m_pdata->getNetTorqueArray(), access_location::host, access_mode::read);
            ArrayHandle<Scalar4> h_orientation(m_pdata->getOrientationArray(), access_location::host, access_mode::read);
            ArrayHandle<Scalar4> h_angmom(m_pdata->getAngularMomentumArray(), access_location::host, access_mode::read);
            ArrayHandle<Scalar3> h_inertia(m_pdata->getMomentsOfInertiaArray(), access_location::host, access_mode::read);

            for (unsigned int group_idx = 0; group_idx < group_size; group_idx++)
                {
                unsigned int j = current_group->getMemberIndex(group_idx);

                vec3<Scalar> t(h_net_torque.data[j]);
                quat<Scalar> p(h_angmom.data[j]);
                quat<Scalar> q(h_orientation.data[j]);
                vec3<Scalar> I(h_inertia.data[j]);

                // rotate torque into principal frame
                t = rotate(conj(q),t);

                // check for zero moment of inertia
                bool x_zero, y_zero, z_zero;
                x_zero = (I.x < EPSILON); y_zero = (I.y < EPSILON); z_zero = (I.z < EPSILON);

                // ignore torque component along an axis for which the moment of inertia zero
                if (x_zero) t.x = 0;
                if (y_zero) t.y = 0;
                if (z_zero) t.z = 0;

                // s is the pure imaginary quaternion with im. part equal to true angular velocity
                vec3<Scalar> s = (Scalar(1./2.) * conj(q) * p).v;

                // rotational power = torque * angvel
                Pr += dot(t,s);
                tnorm += dot(t,t);
                wnorm += dot(s,s);
                }
            }
        }

    #ifdef ENABLE_MPI
    if (m_pdata->getDomainDecomposition())
        {
        MPI_Allreduce(MPI_IN_PLACE, &fnorm, 1, MPI_HOOMD_SCALAR, MPI_SUM, m_exec_conf->getMPICommunicator());
        MPI_Allreduce(MPI_IN_PLACE, &vnorm, 1, MPI_HOOMD_SCALAR, MPI_SUM, m_exec_conf->getMPICommunicator());
        MPI_Allreduce(MPI_IN_PLACE, &Pt, 1, MPI_HOOMD_SCALAR, MPI_SUM, m_exec_conf->getMPICommunicator());

        if (aniso)
            {
            MPI_Allreduce(MPI_IN_PLACE, &tnorm, 1, MPI_HOOMD_SCALAR, MPI_SUM, m_exec_conf->getMPICommunicator());
            MPI_Allreduce(MPI_IN_PLACE, &wnorm, 1, MPI_HOOMD_SCALAR, MPI_SUM, m_exec_conf->getMPICommunicator());
            MPI_Allreduce(MPI_IN_PLACE, &Pr, 1, MPI_HOOMD_SCALAR, MPI_SUM, m_exec_conf->getMPICommunicator());
            }
        }
    #endif

    fnorm = sqrt(fnorm);
    vnorm = sqrt(vnorm);

    tnorm = sqrt(tnorm);
    wnorm = sqrt(wnorm);

    unsigned int ndof = m_sysdef->getNDimensions()*total_group_size;
    m_exec_conf->msg->notice(10) << "FIRE fnorm " << fnorm << " tnorm " << tnorm << " delta_E " << energy-m_old_energy << std::endl;
    m_exec_conf->msg->notice(10) << "FIRE vnorm " << vnorm << " tnorm " << wnorm << std::endl;
    m_exec_conf->msg->notice(10) << "FIRE Pt " << Pt << " Pr " << Pr << std::endl;

    if ((fnorm/sqrt(Scalar(ndof)) < m_ftol && wnorm/sqrt(Scalar(ndof)) < m_wtol  && fabs(energy-m_old_energy) < m_etol) && m_n_since_start >= m_run_minsteps)
        {
        m_exec_conf->msg->notice(4) << "FIRE converged in timestep " << timestep << std::endl;
        m_converged = true;
        return;
        }

    Scalar factor_t;
    if (fabs(fnorm) > EPSILON)
        factor_t = m_alpha*vnorm/fnorm;
    else
        factor_t = 1.0;

    Scalar factor_r = 0.0;

    if (fabs(tnorm) > EPSILON)
        factor_r = m_alpha * wnorm / tnorm;
    else
        factor_r = 1.0;

    for (auto method = m_methods.begin(); method != m_methods.end(); ++method)
        {
        std::shared_ptr<ParticleGroup> current_group = (*method)->getGroup();
        unsigned int group_size = current_group->getIndexArray().getNumElements();
        for (unsigned int group_idx = 0; group_idx < group_size; group_idx++)
            {
            unsigned int j = current_group->getMemberIndex(group_idx);
            h_vel.data[j].x = h_vel.data[j].x*(1.0-m_alpha) + h_accel.data[j].x*factor_t;
            h_vel.data[j].y = h_vel.data[j].y*(1.0-m_alpha) + h_accel.data[j].y*factor_t;
            h_vel.data[j].z = h_vel.data[j].z*(1.0-m_alpha) + h_accel.data[j].z*factor_t;
            }

        if ((*method)->getAnisotropic())
            {
            ArrayHandle<Scalar4> h_angmom(m_pdata->getAngularMomentumArray(), access_location::host, access_mode::readwrite);
            ArrayHandle<Scalar4> h_orientation(m_pdata->getOrientationArray(), access_location::host, access_mode::read);
            ArrayHandle<Scalar4> h_net_torque(m_pdata->getNetTorqueArray(), access_location::host, access_mode::read);
            ArrayHandle<Scalar3> h_inertia(m_pdata->getMomentsOfInertiaArray(), access_location::host, access_mode::read);

            for (unsigned int group_idx = 0; group_idx < group_size; group_idx++)
                {
                unsigned int j = current_group->getMemberIndex(group_idx);
                vec3<Scalar> t(h_net_torque.data[j]);
                quat<Scalar> p(h_angmom.data[j]);
                quat<Scalar> q(h_orientation.data[j]);
                vec3<Scalar> I(h_inertia.data[j]);

                // rotate torque into principal frame
                t = rotate(conj(q),t);

                // check for zero moment of inertia
                bool x_zero, y_zero, z_zero;
                x_zero = (I.x < EPSILON); y_zero = (I.y < EPSILON); z_zero = (I.z < EPSILON);

                // ignore torque component along an axis for which the moment of inertia zero
                if (x_zero) t.x = 0;
                if (y_zero) t.y = 0;
                if (z_zero) t.z = 0;

                // update angular momentum
                p = p*Scalar(1.0-m_alpha) + Scalar(2.0)*q*t*factor_r;
                h_angmom.data[j] = quat_to_scalar4(p);
                }
            }
        }

    // A simply naive measure is to sum up the power coming from translational and rotational motions,
    // more sophisticated measure can be devised later
    Scalar P = Pt + Pr;

    if (P > Scalar(0.0))
        {
        m_n_since_negative++;
        if (m_n_since_negative > m_nmin)
            {
            IntegratorTwoStep::setDeltaT(std::min(m_deltaT*m_finc, m_deltaT_max));
            m_alpha *= m_falpha;
            }
        }
    else if (P <= Scalar(0.0))
        {
        IntegratorTwoStep::setDeltaT(m_deltaT*m_fdec);
        m_alpha = m_alpha_start;
        m_n_since_negative = 0;

        m_exec_conf->msg->notice(6) << "FIRE zero velocities" << std::endl;

        for (auto method = m_methods.begin(); method != m_methods.end(); ++method)
            {
            std::shared_ptr<ParticleGroup> current_group = (*method)->getGroup();
            unsigned int group_size = current_group->getIndexArray().getNumElements();
            for (unsigned int group_idx = 0; group_idx < group_size; group_idx++)
                {
                unsigned int j = current_group->getMemberIndex(group_idx);
                h_vel.data[j].x = Scalar(0.0);
                h_vel.data[j].y = Scalar(0.0);
                h_vel.data[j].z = Scalar(0.0);
                }

            if ((*method)->getAnisotropic())
                {
                ArrayHandle<Scalar4> h_angmom(m_pdata->getAngularMomentumArray(), access_location::host, access_mode::readwrite);
                for (unsigned int group_idx = 0; group_idx < group_size; group_idx++)
                    {
                    unsigned int j = current_group->getMemberIndex(group_idx);
                    h_angmom.data[j] = make_scalar4(0,0,0,0);
                    }
                }

            }
        }
    m_n_since_start++;
    m_old_energy = energy;

    }

void export_FIREEnergyMinimizer(py::module& m)
    {
    py::class_<FIREEnergyMinimizer, std::shared_ptr<FIREEnergyMinimizer> >(m, "FIREEnergyMinimizer", py::base<IntegratorTwoStep>())
        .def(py::init< std::shared_ptr<SystemDefinition>, Scalar>())
        .def("reset", &FIREEnergyMinimizer::reset)
        .def("setDeltaT", &FIREEnergyMinimizer::setDeltaT)
        .def("hasConverged", &FIREEnergyMinimizer::hasConverged)
        .def("getEnergy", &FIREEnergyMinimizer::getEnergy)
        .def("setNmin", &FIREEnergyMinimizer::setNmin)
        .def("setFinc", &FIREEnergyMinimizer::setFinc)
        .def("setFdec", &FIREEnergyMinimizer::setFdec)
        .def("setAlphaStart", &FIREEnergyMinimizer::setAlphaStart)
        .def("setFalpha", &FIREEnergyMinimizer::setFalpha)
        .def("setFtol", &FIREEnergyMinimizer::setFtol)
        .def("setWtol", &FIREEnergyMinimizer::setWtol)
        .def("setEtol", &FIREEnergyMinimizer::setEtol)
        .def("setMinSteps", &FIREEnergyMinimizer::setMinSteps)
        ;
    }
