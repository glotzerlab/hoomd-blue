// Copyright (c) 2009-2024 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

#include "TwoStepBD.h"
#include "hoomd/HOOMDMath.h"
#include "hoomd/VectorMath.h"

#include "hoomd/RNGIdentifiers.h"
#include "hoomd/RandomNumbers.h"
using namespace hoomd;

#ifdef ENABLE_MPI
#include "hoomd/HOOMDMPI.h"
#endif

using namespace std;

namespace hoomd
    {
namespace md
    {
/** @param sysdef SystemDefinition this method will act on. Must not be NULL.
    @param group The group of particles this integration method is to work on
    @param T Temperature set point as a function of time
*/
TwoStepBD::TwoStepBD(std::shared_ptr<SystemDefinition> sysdef,
                     std::shared_ptr<ParticleGroup> group,
                     std::shared_ptr<Variant> T,
                     bool noiseless_t,
                     bool noiseless_r)
    : TwoStepLangevinBase(sysdef, group, T), m_noiseless_t(noiseless_t), m_noiseless_r(noiseless_r)
    {
    m_exec_conf->msg->notice(5) << "Constructing TwoStepBD" << endl;
    }

TwoStepBD::~TwoStepBD()
    {
    m_exec_conf->msg->notice(5) << "Destroying TwoStepBD" << endl;
    }

/*! @param timestep Current time step
    @post Particle positions are moved forward to timestep+1

    The integration method here is from the book "The Langevin and Generalised Langevin Approach to
   the Dynamics of Atomic, Polymeric and Colloidal Systems", chapter 6.
*/
void TwoStepBD::integrateStepOne(uint64_t timestep)
    {
    unsigned int group_size = m_group->getNumMembers();

    // grab some initial variables
    const Scalar currentTemp = m_T->operator()(timestep);
    const unsigned int D = m_sysdef->getNDimensions();

    const GlobalArray<Scalar4>& net_force = m_pdata->getNetForce();
    ArrayHandle<Scalar4> h_vel(m_pdata->getVelocities(),
                               access_location::host,
                               access_mode::readwrite);
    ArrayHandle<Scalar4> h_pos(m_pdata->getPositions(),
                               access_location::host,
                               access_mode::readwrite);
    ArrayHandle<int3> h_image(m_pdata->getImages(), access_location::host, access_mode::readwrite);
    ArrayHandle<unsigned int> h_tag(m_pdata->getTags(), access_location::host, access_mode::read);

    ArrayHandle<Scalar4> h_net_force(net_force, access_location::host, access_mode::read);
    ArrayHandle<Scalar> h_gamma(m_gamma, access_location::host, access_mode::read);

    ArrayHandle<Scalar3> h_gamma_r(m_gamma_r, access_location::host, access_mode::read);
    ArrayHandle<Scalar4> h_orientation(m_pdata->getOrientationArray(),
                                       access_location::host,
                                       access_mode::readwrite);
    ArrayHandle<Scalar4> h_torque(m_pdata->getNetTorqueArray(),
                                  access_location::host,
                                  access_mode::readwrite);

    ArrayHandle<Scalar4> h_angmom(m_pdata->getAngularMomentumArray(),
                                  access_location::host,
                                  access_mode::readwrite);
    ArrayHandle<Scalar3> h_inertia(m_pdata->getMomentsOfInertiaArray(),
                                   access_location::host,
                                   access_mode::read);

    const BoxDim& box = m_pdata->getBox();

    uint16_t seed = m_sysdef->getSeed();

    // perform the first half step
    // r(t+deltaT) = r(t) + (Fc(t) + Fr)*deltaT/gamma
    // v(t+deltaT) = random distribution consistent with T
    for (unsigned int group_idx = 0; group_idx < group_size; group_idx++)
        {
        unsigned int j = m_group->getMemberIndex(group_idx);
        unsigned int ptag = h_tag.data[j];

        // Initialize the RNG
        RandomGenerator rng(hoomd::Seed(RNGIdentifier::TwoStepBD, timestep, seed),
                            hoomd::Counter(ptag));

        // compute the random force
        UniformDistribution<Scalar> uniform(Scalar(-1), Scalar(1));
        Scalar rx = uniform(rng);
        Scalar ry = uniform(rng);
        Scalar rz = uniform(rng);

        Scalar gamma;
        unsigned int type = __scalar_as_int(h_pos.data[j].w);
        gamma = h_gamma.data[type];

        // compute the bd force (the extra factor of 3 is because <rx^2> is 1/3 in the uniform -1,1
        // distribution it is not the dimensionality of the system
        Scalar coeff = fast::sqrt(Scalar(3.0) * Scalar(2.0) * gamma * currentTemp / m_deltaT);
        if (m_noiseless_t)
            coeff = Scalar(0.0);
        Scalar Fr_x = rx * coeff;
        Scalar Fr_y = ry * coeff;
        Scalar Fr_z = rz * coeff;

        if (D < 3)
            Fr_z = Scalar(0.0);

        // update position
        h_pos.data[j].x += (h_net_force.data[j].x + Fr_x) * m_deltaT / gamma;
        h_pos.data[j].y += (h_net_force.data[j].y + Fr_y) * m_deltaT / gamma;
        h_pos.data[j].z += (h_net_force.data[j].z + Fr_z) * m_deltaT / gamma;

        // particles may have been moved slightly outside the box by the above steps, wrap them back
        // into place
        box.wrap(h_pos.data[j], h_image.data[j]);

        if (m_noiseless_t)
            {
            h_vel.data[j].x = h_net_force.data[j].x / gamma;
            h_vel.data[j].y = h_net_force.data[j].y / gamma;
            if (D > 2)
                h_vel.data[j].z = h_net_force.data[j].z / gamma;
            else
                h_vel.data[j].z = 0;
            }
        else
            {
            // draw a new random velocity for particle j
            Scalar mass = h_vel.data[j].w;
            Scalar sigma = fast::sqrt(currentTemp / mass);
            NormalDistribution<Scalar> normal(sigma);
            h_vel.data[j].x = normal(rng);
            h_vel.data[j].y = normal(rng);
            if (D > 2)
                h_vel.data[j].z = normal(rng);
            else
                h_vel.data[j].z = 0;
            }

        // rotational random force and orientation quaternion updates
        if (m_aniso)
            {
            unsigned int type_r = __scalar_as_int(h_pos.data[j].w);
            Scalar3 gamma_r = h_gamma_r.data[type_r];
            if (gamma_r.x > 0 || gamma_r.y > 0 || gamma_r.z > 0)
                {
                vec3<Scalar> p_vec;
                quat<Scalar> q(h_orientation.data[j]);
                vec3<Scalar> t(h_torque.data[j]);
                vec3<Scalar> I(h_inertia.data[j]);

                bool x_zero, y_zero, z_zero;
                x_zero = (I.x == 0);
                y_zero = (I.y == 0);
                z_zero = (I.z == 0);

                Scalar3 sigma_r
                    = make_scalar3(fast::sqrt(Scalar(2.0) * gamma_r.x * currentTemp / m_deltaT),
                                   fast::sqrt(Scalar(2.0) * gamma_r.y * currentTemp / m_deltaT),
                                   fast::sqrt(Scalar(2.0) * gamma_r.z * currentTemp / m_deltaT));
                if (m_noiseless_r)
                    sigma_r = make_scalar3(0, 0, 0);

                // original Gaussian random torque
                // Gaussian random distribution is preferred in terms of preserving the exact math
                vec3<Scalar> bf_torque;
                bf_torque.x = NormalDistribution<Scalar>(sigma_r.x)(rng);
                bf_torque.y = NormalDistribution<Scalar>(sigma_r.y)(rng);
                bf_torque.z = NormalDistribution<Scalar>(sigma_r.z)(rng);

                if (x_zero)
                    {
                    bf_torque.x = 0;
                    t.x = 0;
                    }
                if (y_zero)
                    {
                    bf_torque.y = 0;
                    t.y = 0;
                    }
                if (z_zero)
                    {
                    bf_torque.z = 0;
                    t.z = 0;
                    }

                // use the damping by gamma_r and rotate back to lab frame
                // Notes For the Future: take special care when have anisotropic gamma_r
                // if aniso gamma_r, first rotate the torque into particle frame and divide the
                // different gamma_r and then rotate the "angular velocity" back to lab frame and
                // integrate
                bf_torque = rotate(q, bf_torque);
                if (D < 3)
                    {
                    bf_torque.x = 0;
                    bf_torque.y = 0;
                    t.x = 0;
                    t.y = 0;
                    }

                // do the integration for quaternion
                q += Scalar(0.5) * m_deltaT * ((t + bf_torque) / vec3<Scalar>(gamma_r)) * q;
                q = q * (Scalar(1.0) / slow::sqrt(norm2(q)));
                h_orientation.data[j] = quat_to_scalar4(q);

                if (m_noiseless_r)
                    {
                    p_vec.x = t.x / gamma_r.x;
                    p_vec.y = t.y / gamma_r.y;
                    p_vec.z = t.z / gamma_r.z;
                    }
                else
                    {
                    // draw a new random ang_mom for particle j in body frame
                    p_vec.x = NormalDistribution<Scalar>(fast::sqrt(currentTemp * I.x))(rng);
                    p_vec.y = NormalDistribution<Scalar>(fast::sqrt(currentTemp * I.y))(rng);
                    p_vec.z = NormalDistribution<Scalar>(fast::sqrt(currentTemp * I.z))(rng);
                    }

                if (x_zero)
                    p_vec.x = 0;
                if (y_zero)
                    p_vec.y = 0;
                if (z_zero)
                    p_vec.z = 0;

                // !! Note this isn't well-behaving in 2D,
                // !! because may have effective non-zero ang_mom in x,y

                // store ang_mom quaternion
                quat<Scalar> p = Scalar(2.0) * q * p_vec;
                h_angmom.data[j] = quat_to_scalar4(p);
                }
            }
        }
    }

/*! @param timestep Current time step
 */
void TwoStepBD::integrateStepTwo(uint64_t timestep)
    {
    // there is no step 2 in Brownian dynamics.
    }

namespace detail
    {
void export_TwoStepBD(pybind11::module& m)
    {
    pybind11::class_<TwoStepBD, TwoStepLangevinBase, std::shared_ptr<TwoStepBD>>(m, "TwoStepBD")
        .def(pybind11::init<std::shared_ptr<SystemDefinition>,
                            std::shared_ptr<ParticleGroup>,
                            std::shared_ptr<Variant>,
                            bool,
                            bool>());
    }

    } // end namespace detail
    } // end namespace md
    } // end namespace hoomd
