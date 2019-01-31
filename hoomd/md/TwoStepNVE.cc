// Copyright (c) 2009-2019 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.


// Maintainer: joaander



#include "TwoStepNVE.h"
#include "hoomd/VectorMath.h"


using namespace std;
namespace py = pybind11;

/*! \file TwoStepNVE.h
    \brief Contains code for the TwoStepNVE class
*/

/*! \param sysdef SystemDefinition this method will act on. Must not be NULL.
    \param group The group of particles this integration method is to work on
    \param skip_restart Skip initialization of the restart information
*/
TwoStepNVE::TwoStepNVE(std::shared_ptr<SystemDefinition> sysdef,
                       std::shared_ptr<ParticleGroup> group,
                       bool skip_restart)
    : IntegrationMethodTwoStep(sysdef, group), m_limit(false), m_limit_val(1.0), m_zero_force(false)
    {
    m_exec_conf->msg->notice(5) << "Constructing TwoStepNVE" << endl;

    if (!skip_restart)
        {
        // set a named, but otherwise blank set of integrator variables
        IntegratorVariables v = getIntegratorVariables();

        if (!restartInfoTestValid(v, "nve", 0))
            {
            v.type = "nve";
            v.variable.resize(0);
            setValidRestart(false);
            }
        else
            setValidRestart(true);

        setIntegratorVariables(v);
        }
    }

TwoStepNVE::~TwoStepNVE()
    {
    m_exec_conf->msg->notice(5) << "Destroying TwoStepNVE" << endl;
    }

/*! \param limit Distance to limit particle movement each time step

    Once the limit is set, future calls to update() will never move a particle
    a distance larger than the limit in a single time step
*/
void TwoStepNVE::setLimit(Scalar limit)
    {
    m_limit = true;
    m_limit_val = limit;
    }

/*! Disables the limit, allowing particles to move normally
*/
void TwoStepNVE::removeLimit()
    {
    m_limit = false;
    }

/*! \param timestep Current time step
    \post Particle positions are moved forward to timestep+1 and velocities to timestep+1/2 per the velocity verlet
          method.
*/
void TwoStepNVE::integrateStepOne(unsigned int timestep)
    {
    unsigned int group_size = m_group->getNumMembers();

    // profile this step
    if (m_prof)
        m_prof->push("NVE step 1");

    ArrayHandle<Scalar4> h_vel(m_pdata->getVelocities(), access_location::host, access_mode::readwrite);
    ArrayHandle<Scalar3> h_accel(m_pdata->getAccelerations(), access_location::host, access_mode::readwrite);
    ArrayHandle<Scalar4> h_pos(m_pdata->getPositions(), access_location::host, access_mode::readwrite);

    // perform the first half step of velocity verlet
    // r(t+deltaT) = r(t) + v(t)*deltaT + (1/2)a(t)*deltaT^2
    // v(t+deltaT/2) = v(t) + (1/2)a*deltaT
    for (unsigned int group_idx = 0; group_idx < group_size; group_idx++)
        {
        unsigned int j = m_group->getMemberIndex(group_idx);
        if (m_zero_force)
            h_accel.data[j].x = h_accel.data[j].y = h_accel.data[j].z = 0.0;

        Scalar dx = h_vel.data[j].x*m_deltaT + Scalar(1.0/2.0)*h_accel.data[j].x*m_deltaT*m_deltaT;
        Scalar dy = h_vel.data[j].y*m_deltaT + Scalar(1.0/2.0)*h_accel.data[j].y*m_deltaT*m_deltaT;
        Scalar dz = h_vel.data[j].z*m_deltaT + Scalar(1.0/2.0)*h_accel.data[j].z*m_deltaT*m_deltaT;

        // limit the movement of the particles
        if (m_limit)
            {
            Scalar len = sqrt(dx*dx + dy*dy + dz*dz);
            if (len > m_limit_val)
                {
                dx = dx / len * m_limit_val;
                dy = dy / len * m_limit_val;
                dz = dz / len * m_limit_val;
                }
            }

        h_pos.data[j].x += dx;
        h_pos.data[j].y += dy;
        h_pos.data[j].z += dz;

        h_vel.data[j].x += Scalar(1.0/2.0)*h_accel.data[j].x*m_deltaT;
        h_vel.data[j].y += Scalar(1.0/2.0)*h_accel.data[j].y*m_deltaT;
        h_vel.data[j].z += Scalar(1.0/2.0)*h_accel.data[j].z*m_deltaT;
        }

    // particles may have been moved slightly outside the box by the above steps, wrap them back into place
    const BoxDim& box = m_pdata->getBox();

    ArrayHandle<int3> h_image(m_pdata->getImages(), access_location::host, access_mode::readwrite);

    for (unsigned int group_idx = 0; group_idx < group_size; group_idx++)
        {
        unsigned int j = m_group->getMemberIndex(group_idx);
        box.wrap(h_pos.data[j], h_image.data[j]);
        }

    // Integration of angular degrees of freedom using symplectic and
    // time-reversal symmetric integration scheme of Miller et al.
    if (m_aniso)
        {
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

    // done profiling
    if (m_prof)
        m_prof->pop();
    }

/*! \param timestep Current time step
    \post particle velocities are moved forward to timestep+1
*/
void TwoStepNVE::integrateStepTwo(unsigned int timestep)
    {
    unsigned int group_size = m_group->getNumMembers();

    const GlobalArray< Scalar4 >& net_force = m_pdata->getNetForce();

    // profile this step
    if (m_prof)
        m_prof->push("NVE step 2");

    ArrayHandle<Scalar4> h_vel(m_pdata->getVelocities(), access_location::host, access_mode::readwrite);
    ArrayHandle<Scalar3> h_accel(m_pdata->getAccelerations(), access_location::host, access_mode::readwrite);

    ArrayHandle<Scalar4> h_net_force(net_force, access_location::host, access_mode::read);

    // v(t+deltaT) = v(t+deltaT/2) + 1/2 * a(t+deltaT)*deltaT
    for (unsigned int group_idx = 0; group_idx < group_size; group_idx++)
        {
        unsigned int j = m_group->getMemberIndex(group_idx);

        if (m_zero_force)
            {
            h_accel.data[j].x = h_accel.data[j].y = h_accel.data[j].z = 0.0;
            }
        else
            {
            // first, calculate acceleration from the net force
            Scalar minv = Scalar(1.0) / h_vel.data[j].w;
            h_accel.data[j].x = h_net_force.data[j].x*minv;
            h_accel.data[j].y = h_net_force.data[j].y*minv;
            h_accel.data[j].z = h_net_force.data[j].z*minv;
            }

        // then, update the velocity
        h_vel.data[j].x += Scalar(1.0/2.0)*h_accel.data[j].x*m_deltaT;
        h_vel.data[j].y += Scalar(1.0/2.0)*h_accel.data[j].y*m_deltaT;
        h_vel.data[j].z += Scalar(1.0/2.0)*h_accel.data[j].z*m_deltaT;

        // limit the movement of the particles
        if (m_limit)
            {
            Scalar vel = sqrt(h_vel.data[j].x*h_vel.data[j].x+h_vel.data[j].y*h_vel.data[j].y+h_vel.data[j].z*h_vel.data[j].z);
            if ( (vel*m_deltaT) > m_limit_val)
                {
                h_vel.data[j].x = h_vel.data[j].x / vel * m_limit_val / m_deltaT;
                h_vel.data[j].y = h_vel.data[j].y / vel * m_limit_val / m_deltaT;
                h_vel.data[j].z = h_vel.data[j].z / vel * m_limit_val / m_deltaT;
                }
            }
        }

    if (m_aniso)
        {
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

            // advance p(t+deltaT/2)->p(t+deltaT)
            p += m_deltaT*q*t;

            h_angmom.data[j] = quat_to_scalar4(p);
            }
        }

    // done profiling
    if (m_prof)
        m_prof->pop();
    }

void export_TwoStepNVE(py::module& m)
    {
    py::class_<TwoStepNVE, std::shared_ptr<TwoStepNVE> >(m, "TwoStepNVE", py::base<IntegrationMethodTwoStep>())
        .def(py::init< std::shared_ptr<SystemDefinition>, std::shared_ptr<ParticleGroup>, bool >())
        .def("setLimit", &TwoStepNVE::setLimit)
        .def("removeLimit", &TwoStepNVE::removeLimit)
        .def("setZeroForce", &TwoStepNVE::setZeroForce)
        ;
    }
