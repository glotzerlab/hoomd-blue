// Copyright (c) 2009-2019 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.

#include "TwoStepNVTMTK.h"

#include "hoomd/VectorMath.h"
#include "hoomd/RandomNumbers.h"
#include "hoomd/RNGIdentifiers.h"

#ifdef ENABLE_MPI
#include "hoomd/Communicator.h"
#include "hoomd/HOOMDMPI.h"
#endif


using namespace std;
namespace py = pybind11;

/*! \file TwoStepNVTMTK.h
    \brief Contains code for the TwoStepNVTMTK class
*/

/*! \param sysdef SystemDefinition this method will act on. Must not be NULL.
    \param group The group of particles this integration method is to work on
    \param thermo compute for thermodynamic quantities
    \param tau NVT period
    \param T Temperature set point
    \param suffix Suffix to attach to the end of log quantity names
*/
TwoStepNVTMTK::TwoStepNVTMTK(std::shared_ptr<SystemDefinition> sysdef,
                       std::shared_ptr<ParticleGroup> group,
                       std::shared_ptr<ComputeThermo> thermo,
                       Scalar tau,
                       std::shared_ptr<Variant> T,
                       const std::string& suffix)
    : IntegrationMethodTwoStep(sysdef, group), m_thermo(thermo), m_tau(tau), m_T(T), m_exp_thermo_fac(1.0)
    {
    m_exec_conf->msg->notice(5) << "Constructing TwoStepNVTMTK" << endl;

    if (m_tau <= 0.0)
        m_exec_conf->msg->warning() << "integrate.nvt: tau set less than 0.0 in NVTUpdater" << endl;

    // set initial state
    if (!restartInfoTestValid(getIntegratorVariables(), "nvt_mtk", 4))
        {
        initializeIntegratorVariables();
        setValidRestart(false);
        }
    else
        {
        setValidRestart(true);
        }

    m_log_name = string("nvt_mtk_reservoir_energy") + suffix;
    }

TwoStepNVTMTK::~TwoStepNVTMTK()
    {
    m_exec_conf->msg->notice(5) << "Destroying TwoStepNVTMTK" << endl;
    }

/*! Returns a list of log quantities this compute calculates
*/
std::vector< std::string > TwoStepNVTMTK::getProvidedLogQuantities()
    {
    vector<string> result;
    result.push_back(m_log_name);
    return result;
    }

/*! \param quantity Name of the log quantity to get
    \param timestep Current time step of the simulation
    \param my_quantity_flag passed as false, changed to true if quantity logged here
*/

Scalar TwoStepNVTMTK::getLogValue(const std::string& quantity, unsigned int timestep, bool &my_quantity_flag)
    {
    if (quantity == m_log_name)
        {
        my_quantity_flag = true;
        Scalar g = m_thermo->getNDOF();
        IntegratorVariables v = getIntegratorVariables();
        Scalar& xi = v.variable[0];
        Scalar& eta = v.variable[1];
        Scalar thermostat_energy = (Scalar) g * m_T->getValue(timestep) * (xi*xi*m_tau*m_tau / Scalar(2.0) + eta);

        if (m_aniso)
            {
            Scalar& xi_rot = v.variable[2];
            Scalar& eta_rot = v.variable[3];
            thermostat_energy += (Scalar)m_thermo->getRotationalNDOF()*m_T->getValue(timestep)
                                   *(eta_rot + m_tau*m_tau*xi_rot*xi_rot/Scalar(2.0));
            }

        return thermostat_energy;
        }
    else
        return Scalar(0);
    }

/*! \param timestep Current time step
    \post Particle positions are moved forward to timestep+1 and velocities to timestep+1/2 per the velocity verlet
          method.
*/
void TwoStepNVTMTK::integrateStepOne(unsigned int timestep)
    {
    if (m_group->getNumMembersGlobal() == 0)
        {
        m_exec_conf->msg->error() << "integrate.nvt(): Integration group empty." << std::endl;
        throw std::runtime_error("Error during NVT integration.");
        }

    unsigned int group_size = m_group->getNumMembers();

    // profile this step
    if (m_prof)
        m_prof->push("NVT step 1");

    // scope array handles for proper releasing before calling the thermo compute
    {
    ArrayHandle<Scalar4> h_vel(m_pdata->getVelocities(), access_location::host, access_mode::readwrite);
    ArrayHandle<Scalar3> h_accel(m_pdata->getAccelerations(), access_location::host, access_mode::readwrite);
    ArrayHandle<Scalar4> h_pos(m_pdata->getPositions(), access_location::host, access_mode::readwrite);

    for (unsigned int group_idx = 0; group_idx < group_size; group_idx++)
        {
        unsigned int j = m_group->getMemberIndex(group_idx);

        // load variables
        Scalar3 v = make_scalar3(h_vel.data[j].x, h_vel.data[j].y, h_vel.data[j].z);
        Scalar3 pos = make_scalar3(h_pos.data[j].x, h_pos.data[j].y, h_pos.data[j].z);
        Scalar3 accel = h_accel.data[j];

        // update velocity and position
        v = v + Scalar(1.0/2.0)*accel*m_deltaT;

        // rescale velocity
        v *= m_exp_thermo_fac;

        pos += m_deltaT * v;

        // store updated variables
        h_vel.data[j].x = v.x;
        h_vel.data[j].y = v.y;
        h_vel.data[j].z = v.z;

        h_pos.data[j].x = pos.x;
        h_pos.data[j].y = pos.y;
        h_pos.data[j].z = pos.z;
        }

    // particles may have been moved slightly outside the box by the above steps, wrap them back into place
    const BoxDim& box = m_pdata->getBox();

    ArrayHandle<int3> h_image(m_pdata->getImages(), access_location::host, access_mode::readwrite);

    for (unsigned int group_idx = 0; group_idx < group_size; group_idx++)
        {
        unsigned int j = m_group->getMemberIndex(group_idx);
        // wrap the particles around the box
        box.wrap(h_pos.data[j], h_image.data[j]);
        }
    }

    // Integration of angular degrees of freedom using symplectic and
    // time-reversal symmetric integration scheme of Miller et al., extended by thermostat
    if (m_aniso)
        {
        // thermostat factor
        IntegratorVariables v = getIntegratorVariables();
        Scalar xi_rot = v.variable[2];
        Scalar exp_fac = exp(-m_deltaT/Scalar(2.0)*xi_rot);

        ArrayHandle<Scalar4> h_orientation(m_pdata->getOrientationArray(), access_location::host, access_mode::readwrite);
        ArrayHandle<Scalar4> h_angmom(m_pdata->getAngularMomentumArray(), access_location::host, access_mode::readwrite);
        ArrayHandle<Scalar4> h_net_torque(m_pdata->getNetTorqueArray(), access_location::host, access_mode::read);
        ArrayHandle<Scalar3> h_inertia(m_pdata->getMomentsOfInertiaArray(), access_location::host, access_mode::read);

        for (unsigned int group_idx = 0; group_idx < group_size; group_idx++)
            {
            unsigned int j = m_group->getMemberIndex(group_idx);

            quat<Scalar> q(h_orientation.data[j]);
            quat<Scalar> p(h_angmom.data[j]);
            vec3<Scalar> t(h_net_torque.data[j]);
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

            // advance p(t)->p(t+deltaT/2), q(t)->q(t+deltaT)
            // using Trotter factorization of rotation Liouvillian
            p += m_deltaT*q*t;

            // apply thermostat
            p = p*exp_fac;

            quat<Scalar> p1, p2, p3; // permutated quaternions
            quat<Scalar> q1, q2, q3;
            Scalar phi1, cphi1, sphi1;
            Scalar phi2, cphi2, sphi2;
            Scalar phi3, cphi3, sphi3;

            if (!z_zero)
                {
                p3 = quat<Scalar>(-p.v.z,vec3<Scalar>(p.v.y,-p.v.x,p.s));
                q3 = quat<Scalar>(-q.v.z,vec3<Scalar>(q.v.y,-q.v.x,q.s));
                phi3 = Scalar(1./4.)/I.z*dot(p,q3);
                cphi3 = slow::cos(Scalar(1./2.)*m_deltaT*phi3);
                sphi3 = slow::sin(Scalar(1./2.)*m_deltaT*phi3);

                p=cphi3*p+sphi3*p3;
                q=cphi3*q+sphi3*q3;
                }

            if (!y_zero)
                {
                p2 = quat<Scalar>(-p.v.y,vec3<Scalar>(-p.v.z,p.s,p.v.x));
                q2 = quat<Scalar>(-q.v.y,vec3<Scalar>(-q.v.z,q.s,q.v.x));
                phi2 = Scalar(1./4.)/I.y*dot(p,q2);
                cphi2 = slow::cos(Scalar(1./2.)*m_deltaT*phi2);
                sphi2 = slow::sin(Scalar(1./2.)*m_deltaT*phi2);

                p=cphi2*p+sphi2*p2;
                q=cphi2*q+sphi2*q2;
                }

           if (!x_zero)
                {
                p1 = quat<Scalar>(-p.v.x,vec3<Scalar>(p.s,p.v.z,-p.v.y));
                q1 = quat<Scalar>(-q.v.x,vec3<Scalar>(q.s,q.v.z,-q.v.y));
                phi1 = Scalar(1./4.)/I.x*dot(p,q1);
                cphi1 = slow::cos(m_deltaT*phi1);
                sphi1 = slow::sin(m_deltaT*phi1);

                p=cphi1*p+sphi1*p1;
                q=cphi1*q+sphi1*q1;
                }

            if (! y_zero)
                {
                p2 = quat<Scalar>(-p.v.y,vec3<Scalar>(-p.v.z,p.s,p.v.x));
                q2 = quat<Scalar>(-q.v.y,vec3<Scalar>(-q.v.z,q.s,q.v.x));
                phi2 = Scalar(1./4.)/I.y*dot(p,q2);
                cphi2 = slow::cos(Scalar(1./2.)*m_deltaT*phi2);
                sphi2 = slow::sin(Scalar(1./2.)*m_deltaT*phi2);

                p=cphi2*p+sphi2*p2;
                q=cphi2*q+sphi2*q2;
                }

            if (! z_zero)
                {
                p3 = quat<Scalar>(-p.v.z,vec3<Scalar>(p.v.y,-p.v.x,p.s));
                q3 = quat<Scalar>(-q.v.z,vec3<Scalar>(q.v.y,-q.v.x,q.s));
                phi3 = Scalar(1./4.)/I.z*dot(p,q3);
                cphi3 = slow::cos(Scalar(1./2.)*m_deltaT*phi3);
                sphi3 = slow::sin(Scalar(1./2.)*m_deltaT*phi3);

                p=cphi3*p+sphi3*p3;
                q=cphi3*q+sphi3*q3;
                }

            // renormalize (improves stability)
            q = q*(Scalar(1.0)/slow::sqrt(norm2(q)));

            h_orientation.data[j] = quat_to_scalar4(q);
            h_angmom.data[j] = quat_to_scalar4(p);
            }
        }

    // get temperature and advance thermostat
    advanceThermostat(timestep);

    // done profiling
    if (m_prof)
        m_prof->pop();
    }

/*! \param timestep Current time step
    \post particle velocities are moved forward to timestep+1
*/
void TwoStepNVTMTK::integrateStepTwo(unsigned int timestep)
    {
    unsigned int group_size = m_group->getNumMembers();

    const GlobalArray< Scalar4 >& net_force = m_pdata->getNetForce();

    // profile this step
    if (m_prof)
        m_prof->push("NVT step 2");

    ArrayHandle<Scalar4> h_vel(m_pdata->getVelocities(), access_location::host, access_mode::readwrite);
    ArrayHandle<Scalar3> h_accel(m_pdata->getAccelerations(), access_location::host, access_mode::readwrite);

    ArrayHandle<Scalar4> h_net_force(net_force, access_location::host, access_mode::read);

    // perform second half step of Nose-Hoover integration

    for (unsigned int group_idx = 0; group_idx < group_size; group_idx++)
        {
        unsigned int j = m_group->getMemberIndex(group_idx);

        // load velocity
        Scalar3 v = make_scalar3(h_vel.data[j].x, h_vel.data[j].y, h_vel.data[j].z);
        Scalar3 accel = h_accel.data[j];
        Scalar3 net_force = make_scalar3(h_net_force.data[j].x,h_net_force.data[j].y,h_net_force.data[j].z);

        // first, calculate acceleration from the net force
        Scalar m = h_vel.data[j].w;
        Scalar minv = Scalar(1.0) / m;
        accel = net_force*minv;

        // rescale velocity
        v *= m_exp_thermo_fac;

        // update velocity
        v += Scalar(1.0/2.0) * m_deltaT * accel;

        // store velocity
        h_vel.data[j].x = v.x;
        h_vel.data[j].y = v.y;
        h_vel.data[j].z = v.z;

        // store acceleration
        h_accel.data[j] = accel;
        }

    if (m_aniso)
        {
        IntegratorVariables v = getIntegratorVariables();
        Scalar xi_rot = v.variable[2];
        Scalar exp_fac = exp(-m_deltaT/Scalar(2.0)*xi_rot);

        // angular degrees of freedom
        ArrayHandle<Scalar4> h_orientation(m_pdata->getOrientationArray(), access_location::host, access_mode::read);
        ArrayHandle<Scalar4> h_angmom(m_pdata->getAngularMomentumArray(), access_location::host, access_mode::readwrite);
        ArrayHandle<Scalar4> h_net_torque(m_pdata->getNetTorqueArray(), access_location::host, access_mode::read);
        ArrayHandle<Scalar3> h_inertia(m_pdata->getMomentsOfInertiaArray(), access_location::host, access_mode::read);

        for (unsigned int group_idx = 0; group_idx < group_size; group_idx++)
            {
            unsigned int j = m_group->getMemberIndex(group_idx);

            quat<Scalar> q(h_orientation.data[j]);
            quat<Scalar> p(h_angmom.data[j]);
            vec3<Scalar> t(h_net_torque.data[j]);
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

            // apply thermostat
            p = p*exp_fac;

            // advance p(t+deltaT/2)->p(t+deltaT)
            p += m_deltaT*q*t;

            h_angmom.data[j] = quat_to_scalar4(p);
            }
        }

    // done profiling
    if (m_prof)
        m_prof->pop();
    }

void TwoStepNVTMTK::advanceThermostat(unsigned int timestep, bool broadcast)
    {
    IntegratorVariables v = getIntegratorVariables();
    Scalar& xi = v.variable[0];
    Scalar& eta = v.variable[1];

    // compute the current thermodynamic properties
    m_thermo->compute(timestep+1);

    Scalar curr_T_trans = m_thermo->getTranslationalTemperature();

    // update the state variables Xi and eta
    Scalar xi_prime = xi + Scalar(1.0/2.0)*m_deltaT/m_tau/m_tau*(curr_T_trans/m_T->getValue(timestep) - Scalar(1.0));
    xi = xi_prime + Scalar(1.0/2.0)*m_deltaT/m_tau/m_tau*(curr_T_trans/m_T->getValue(timestep) - Scalar(1.0));
    eta += xi_prime*m_deltaT;

    // update loop-invariant quantity
    m_exp_thermo_fac = exp(-Scalar(1.0/2.0)*xi*m_deltaT);

    #ifdef ENABLE_MPI
    if (m_comm && broadcast)
        {
        // broadcast integrator variables from rank 0 to other processors
        MPI_Bcast(&xi, 1, MPI_HOOMD_SCALAR, 0, m_exec_conf->getMPICommunicator());
        MPI_Bcast(&eta, 1, MPI_HOOMD_SCALAR, 0, m_exec_conf->getMPICommunicator());
        }
    #endif

    if (m_aniso)
        {
        // update thermostat for rotational DOF
        Scalar &xi_rot = v.variable[2];
        Scalar &eta_rot = v.variable[3];

        Scalar curr_ke_rot = m_thermo->getRotationalKineticEnergy();
        unsigned int ndof_rot = m_thermo->getRotationalNDOF();

        Scalar xi_prime_rot = xi_rot + Scalar(1.0/2.0)*m_deltaT/m_tau/m_tau*
            (Scalar(2.0)*curr_ke_rot/ndof_rot/m_T->getValue(timestep) - Scalar(1.0));
        xi_rot = xi_prime_rot + Scalar(1.0/2.0)*m_deltaT/m_tau/m_tau*
            (Scalar(2.0)*curr_ke_rot/ndof_rot/m_T->getValue(timestep) - Scalar(1.0));

        eta_rot += xi_prime_rot*m_deltaT;

        #ifdef ENABLE_MPI
        if (m_comm)
            {
            // broadcast integrator variables from rank 0 to other processors
            MPI_Bcast(&xi_rot, 1, MPI_HOOMD_SCALAR, 0, m_exec_conf->getMPICommunicator());
            MPI_Bcast(&eta_rot, 1, MPI_HOOMD_SCALAR, 0, m_exec_conf->getMPICommunicator());
            }
        #endif
        }

    setIntegratorVariables(v);
    }

void TwoStepNVTMTK::randomizeVelocities(unsigned int timestep)
    {
    if (m_shouldRandomize == false)
        {
        return;
        }

    m_exec_conf->msg->notice(6) << "TwoStepNVTMTK randomizing velocities" << std::endl;

    IntegratorVariables v = getIntegratorVariables();
    Scalar& xi = v.variable[0];

    unsigned int g = m_thermo->getNDOF();
    Scalar sigmasq_t = Scalar(1.0)/((Scalar) g*m_tau*m_tau);

    bool master = m_exec_conf->getRank() == 0;
    hoomd::RandomGenerator rng(hoomd::RNGIdentifier::TwoStepNVTMTK, m_seed_randomize, timestep);

    if (master)
        {
        // draw a random Gaussian thermostat variable on rank 0
        xi = hoomd::NormalDistribution<Scalar>(sqrt(sigmasq_t))(rng);
        }

    #ifdef ENABLE_MPI
    if (m_comm)
        {
        // broadcast integrator variables from rank 0 to other processors
        MPI_Bcast(&xi, 1, MPI_HOOMD_SCALAR, 0, m_exec_conf->getMPICommunicator());
        }
    #endif

    if (m_aniso)
        {
        // update thermostat for rotational DOF
        Scalar &xi_rot = v.variable[2];
        Scalar sigmasq_r = Scalar(1.0)/((Scalar)m_thermo->getRotationalNDOF()*m_tau*m_tau);

        if (master)
            {
            xi_rot = hoomd::NormalDistribution<Scalar>(sqrt(sigmasq_r))(rng);
            }

        #ifdef ENABLE_MPI
        if (m_comm)
            {
            // broadcast integrator variables from rank 0 to other processors
            MPI_Bcast(&xi_rot, 1, MPI_HOOMD_SCALAR, 0, m_exec_conf->getMPICommunicator());
            }
        #endif
        }

    setIntegratorVariables(v);

    // call base class method
    IntegrationMethodTwoStep::randomizeVelocities(timestep);
    }

void export_TwoStepNVTMTK(py::module& m)
    {
    py::class_<TwoStepNVTMTK, std::shared_ptr<TwoStepNVTMTK> >(m, "TwoStepNVTMTK", py::base<IntegrationMethodTwoStep>())
        .def(py::init< std::shared_ptr<SystemDefinition>,
                       std::shared_ptr<ParticleGroup>,
                       std::shared_ptr<ComputeThermo>,
                       Scalar,
                       std::shared_ptr<Variant>,
                       const std::string&
                       >())
        .def("setT", &TwoStepNVTMTK::setT)
        .def("setTau", &TwoStepNVTMTK::setTau)
        ;
    }
