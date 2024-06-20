// Copyright (c) 2009-2024 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

#include "TwoStepLangevinBase.h"
#include "TwoStepRATTLENVE.h"
#include "hoomd/HOOMDMath.h"
#include "hoomd/VectorMath.h"

#include "hoomd/RNGIdentifiers.h"
#include "hoomd/RandomNumbers.h"

#ifndef __TWO_STEP_RATTLE_BD_H__
#define __TWO_STEP_RATTLE_BD_H__

/*! \file TwoStepRATTLEBD.h
    \brief Declares the TwoStepRATTLEBD class
*/

#ifdef NVCC
#error This header cannot be compiled by nvcc
#endif

#include <pybind11/pybind11.h>

namespace hoomd
    {
namespace md
    {
//! Integrates part of the system forward in two steps with Brownian dynamics
/*! Implements RATTLE applied on Brownian dynamics.

    Brownian dynamics modifies the Langevin equation by setting the acceleration term to 0 and
   assuming terminal velocity.

    \ingroup updaters
*/
template<class Manifold> class PYBIND11_EXPORT TwoStepRATTLEBD : public TwoStepLangevinBase
    {
    public:
    //! Constructs the integration method and associates it with the system
    TwoStepRATTLEBD(std::shared_ptr<SystemDefinition> sysdef,
                    std::shared_ptr<ParticleGroup> group,
                    Manifold manifold,
                    std::shared_ptr<Variant> T,
                    bool noiseless_t,
                    bool noiseless_r,
                    Scalar tolerance = 0.000001);

    virtual ~TwoStepRATTLEBD();

    //! Performs the first step of the integration
    virtual void integrateStepOne(uint64_t timestep);

    //! Performs the second step of the integration
    virtual void integrateStepTwo(uint64_t timestep);

    //! Includes the RATTLE forces to the virial/net force
    virtual void includeRATTLEForce(uint64_t timestep);

    //! Get the number of degrees of freedom granted to a given group
    virtual Scalar getTranslationalDOF(std::shared_ptr<ParticleGroup> group)
        {
        // get the size of the intersection between query_group and m_group
        unsigned int intersect_size
            = ParticleGroup::groupIntersection(group, m_group)->getNumMembersGlobal();

        return Manifold::dimension() * intersect_size;
        }

    /// Sets tolerance
    void setTolerance(Scalar tolerance)
        {
        m_tolerance = tolerance;
        };

    /// Gets tolerance
    Scalar getTolerance()
        {
        return m_tolerance;
        };

    protected:
    //! Helper function to be called when box changes
    void setBoxChange()
        {
        m_box_changed = true;
        }

    Manifold m_manifold; //!< The manifold used for the RATTLE constraint
    bool m_noiseless_t;
    bool m_noiseless_r;
    Scalar m_tolerance; //!< The tolerance value of the RATTLE algorithm, setting the tolerance to
                        //!< the manifold
    bool m_box_changed;
    };

/*! \file TwoStepRATTLEBD.h
    \brief Contains code for the TwoStepRATTLEBD class
*/

/*! \param sysdef SystemDefinition this method will act on. Must not be NULL.
    \param group The group of particles this integration method is to work on
    \param manifold The manifold describing the constraint during the RATTLE integration method
    \param T Temperature set point as a function of time
    \param noiseless_t If set true, there will be no translational noise (random force)
    \param noiseless_r If set true, there will be no rotational noise (random torque)
    \param tolerance Tolerance for the RATTLE iteration algorithm
*/
template<class Manifold>
TwoStepRATTLEBD<Manifold>::TwoStepRATTLEBD(std::shared_ptr<SystemDefinition> sysdef,
                                           std::shared_ptr<ParticleGroup> group,
                                           Manifold manifold,
                                           std::shared_ptr<Variant> T,
                                           bool noiseless_t,
                                           bool noiseless_r,
                                           Scalar tolerance)
    : TwoStepLangevinBase(sysdef, group, T), m_manifold(manifold), m_noiseless_t(noiseless_t),
      m_noiseless_r(noiseless_r), m_tolerance(tolerance), m_box_changed(false)
    {
    m_exec_conf->msg->notice(5) << "Constructing TwoStepRATTLEBD" << std::endl;

    m_pdata->getBoxChangeSignal()
        .template connect<TwoStepRATTLEBD<Manifold>, &TwoStepRATTLEBD<Manifold>::setBoxChange>(
            this);

    if (!m_manifold.fitsInsideBox(m_pdata->getGlobalBox()))
        {
        throw std::runtime_error("Parts of the manifold are outside the box");
        }
    }

template<class Manifold> TwoStepRATTLEBD<Manifold>::~TwoStepRATTLEBD()
    {
    m_pdata->getBoxChangeSignal()
        .template disconnect<TwoStepRATTLEBD<Manifold>, &TwoStepRATTLEBD<Manifold>::setBoxChange>(
            this);
    m_exec_conf->msg->notice(5) << "Destroying TwoStepRATTLEBD" << std::endl;
    }

/*! \param timestep Current time step
    \post Particle positions are moved forward to timestep+1

    The integration method here is from the book "The Langevin and Generalised Langevin Approach to
   the Dynamics of Atomic, Polymeric and Colloidal Systems", chapter 6.
*/

template<class Manifold> void TwoStepRATTLEBD<Manifold>::integrateStepOne(uint64_t timestep)
    {
    unsigned int group_size = m_group->getNumMembers();

    const Scalar currentTemp = m_T->operator()(timestep);

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

    const BoxDim box = m_pdata->getBox();

    if (m_box_changed)
        {
        if (!m_manifold.fitsInsideBox(m_pdata->getGlobalBox()))
            {
            throw std::runtime_error("Parts of the manifold are outside the box");
            }
        m_box_changed = false;
        }

    uint16_t seed = m_sysdef->getSeed();

    // perform the first half step
    // r(t+deltaT) = r(t) + (Fc(t) + Fr)*deltaT/gamma
    // iterative: r(t+deltaT) = r(t+deltaT) - J^(-1)*residual
    // v(t+deltaT) = random distribution consistent with T
    for (unsigned int group_idx = 0; group_idx < group_size; group_idx++)
        {
        unsigned int j = m_group->getMemberIndex(group_idx);
        unsigned int ptag = h_tag.data[j];

        // Initialize the RNG
        RandomGenerator rng(hoomd::Seed(RNGIdentifier::TwoStepBD, timestep, seed),
                            hoomd::Counter(ptag, 1));

        // Initialize the RNG
        RandomGenerator rng_b(
            hoomd::Seed(RNGIdentifier::TwoStepBD, timestep, seed),
            hoomd::Counter(ptag, 2)); // This random number generator generates the same numbers as
                                      // in includeRATTLEForce for each particle such that the
                                      // Brownian force stays consistent

        Scalar gamma;
        unsigned int type = __scalar_as_int(h_pos.data[j].w);
        gamma = h_gamma.data[type];

        Scalar deltaT_gamma = m_deltaT / gamma;

        Scalar3 vec_rand;
        if (m_noiseless_t)
            {
            vec_rand.x = h_net_force.data[j].x / gamma;
            vec_rand.y = h_net_force.data[j].x / gamma;
            vec_rand.z = h_net_force.data[j].x / gamma;
            }
        else
            {
            // draw a new random velocity for particle j
            Scalar mass = h_vel.data[j].w;
            Scalar sigma1 = fast::sqrt(currentTemp / mass);
            NormalDistribution<Scalar> norm(sigma1);

            vec_rand.x = norm(rng);
            vec_rand.y = norm(rng);
            vec_rand.z = norm(rng);
            }

        Scalar3 next_pos;
        next_pos.x = h_pos.data[j].x;
        next_pos.y = h_pos.data[j].y;
        next_pos.z = h_pos.data[j].z;

        Scalar3 normal = m_manifold.derivative(next_pos);
        Scalar norm_normal = fast::rsqrt(dot(normal, normal));

        normal.x *= norm_normal;
        normal.y *= norm_normal;
        normal.z *= norm_normal;

        Scalar rand_norm = dot(vec_rand, normal);
        vec_rand.x -= rand_norm * normal.x;
        vec_rand.y -= rand_norm * normal.y;
        vec_rand.z -= rand_norm * normal.z;

        h_vel.data[j].x = vec_rand.x;
        h_vel.data[j].y = vec_rand.y;
        h_vel.data[j].z = vec_rand.z;

        Scalar rx, ry, rz, coeff;

        if (currentTemp > 0)
            {
            // compute the random force
            UniformDistribution<Scalar> uniform(Scalar(-1), Scalar(1));
            rx = uniform(rng_b);
            ry = uniform(rng_b);
            rz = uniform(rng_b);

            Scalar normal_r = rx * normal.x + ry * normal.y + rz * normal.z;

            rx = rx - normal_r * normal.x;
            ry = ry - normal_r * normal.y;
            rz = rz - normal_r * normal.z;

            // compute the bd force (the extra factor of 3 is because <rx^2> is 1/3 in the uniform
            // -1,1 distribution it is not the dimensionality of the system
            coeff = fast::sqrt(Scalar(6.0) * currentTemp / deltaT_gamma);
            if (m_noiseless_t)
                coeff = Scalar(0.0);
            }
        else
            {
            rx = 0;
            ry = 0;
            rz = 0;
            coeff = 0;
            }

        Scalar dx = (h_net_force.data[j].x + rx * coeff) * deltaT_gamma;
        Scalar dy = (h_net_force.data[j].y + ry * coeff) * deltaT_gamma;
        Scalar dz = (h_net_force.data[j].z + rz * coeff) * deltaT_gamma;

        h_pos.data[j].x += dx;
        h_pos.data[j].y += dy;
        h_pos.data[j].z += dz;

        // particles may have been moved slightly outside the box by the above steps, wrap them back
        // into place
        box.wrap(h_pos.data[j], h_image.data[j]);

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

                // use the d_invamping by gamma_r and rotate back to lab frame
                // Notes For the Future: take special care when have anisotropic gamma_r
                // if aniso gamma_r, first rotate the torque into particle frame and divide the
                // different gamma_r and then rotate the "angular velocity" back to lab frame and
                // integrate
                bf_torque = rotate(q, bf_torque);

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

/*! \param timestep Current time step
 */
template<class Manifold> void TwoStepRATTLEBD<Manifold>::integrateStepTwo(uint64_t timestep)
    {
    // there is no step 2 in Brownian dynamics.
    }

template<class Manifold> void TwoStepRATTLEBD<Manifold>::includeRATTLEForce(uint64_t timestep)
    {
    unsigned int group_size = m_group->getNumMembers();

    const Scalar currentTemp = m_T->operator()(timestep);

    const GlobalArray<Scalar4>& net_force = m_pdata->getNetForce();
    const GlobalArray<Scalar>& net_virial = m_pdata->getNetVirial();
    ArrayHandle<Scalar4> h_pos(m_pdata->getPositions(), access_location::host, access_mode::read);
    ArrayHandle<unsigned int> h_tag(m_pdata->getTags(), access_location::host, access_mode::read);

    ArrayHandle<Scalar4> h_net_force(net_force, access_location::host, access_mode::readwrite);
    ArrayHandle<Scalar> h_net_virial(net_virial, access_location::host, access_mode::readwrite);
    ArrayHandle<Scalar> h_gamma(m_gamma, access_location::host, access_mode::read);

    size_t net_virial_pitch = net_virial.getPitch();

    uint16_t seed = m_sysdef->getSeed();

    // perform the first half step
    // r(t+deltaT) = r(t) + (Fc(t) + Fr)*deltaT/gamma
    // iterative: r(t+deltaT) = r(t+deltaT) - J^(-1)*residual
    // v(t+deltaT) = random distribution consistent with T
    for (unsigned int group_idx = 0; group_idx < group_size; group_idx++)
        {
        unsigned int j = m_group->getMemberIndex(group_idx);
        unsigned int ptag = h_tag.data[j];

        // Initialize the RNG
        RandomGenerator rng_b(hoomd::Seed(RNGIdentifier::TwoStepBD, timestep, seed),
                              hoomd::Counter(ptag, 2));

        Scalar gamma;
        unsigned int type = __scalar_as_int(h_pos.data[j].w);
        gamma = h_gamma.data[type];
        Scalar deltaT_gamma = m_deltaT / gamma;

        Scalar3 next_pos;
        next_pos.x = h_pos.data[j].x;
        next_pos.y = h_pos.data[j].y;
        next_pos.z = h_pos.data[j].z;

        Scalar3 normal = m_manifold.derivative(next_pos);
        Scalar norm_normal = fast::rsqrt(dot(normal, normal));

        normal.x *= norm_normal;
        normal.y *= norm_normal;
        normal.z *= norm_normal;

        Scalar rx, ry, rz, coeff;

        if (currentTemp > 0)
            {
            // compute the random force
            UniformDistribution<Scalar> uniform(Scalar(-1), Scalar(1));
            rx = uniform(rng_b);
            ry = uniform(rng_b);
            rz = uniform(rng_b);

            Scalar normal_r = rx * normal.x + ry * normal.y + rz * normal.z;

            rx = rx - normal_r * normal.x;
            ry = ry - normal_r * normal.y;
            rz = rz - normal_r * normal.z;

            // compute the bd force (the extra factor of 3 is because <rx^2> is 1/3 in the uniform
            // -1,1 distribution it is not the dimensionality of the system
            coeff = fast::sqrt(Scalar(6.0) * currentTemp / deltaT_gamma);
            if (m_noiseless_t)
                coeff = Scalar(0.0);
            }
        else
            {
            rx = 0;
            ry = 0;
            rz = 0;
            coeff = 0;
            }

        Scalar Fr_x = rx * coeff;
        Scalar Fr_y = ry * coeff;
        Scalar Fr_z = rz * coeff;

        // update position
        Scalar mu = 0.0;

        Scalar inv_alpha = -Scalar(1.0) / deltaT_gamma;

        Scalar3 residual;
        Scalar resid;

        unsigned int iteration = 0;
        do
            {
            iteration++;
            residual.x = h_pos.data[j].x - next_pos.x
                         + (h_net_force.data[j].x + Fr_x - mu * normal.x) * deltaT_gamma;
            residual.y = h_pos.data[j].y - next_pos.y
                         + (h_net_force.data[j].y + Fr_y - mu * normal.y) * deltaT_gamma;
            residual.z = h_pos.data[j].z - next_pos.z
                         + (h_net_force.data[j].z + Fr_z - mu * normal.z) * deltaT_gamma;
            resid = m_manifold.implicitFunction(next_pos);

            Scalar3 next_normal = m_manifold.derivative(next_pos);

            Scalar nndotr = dot(next_normal, residual);
            Scalar nndotn = dot(next_normal, normal);
            Scalar beta = (resid + nndotr) / nndotn;

            next_pos.x = next_pos.x - beta * normal.x + residual.x;
            next_pos.y = next_pos.y - beta * normal.y + residual.y;
            next_pos.z = next_pos.z - beta * normal.z + residual.z;
            mu = mu - beta * inv_alpha;

            } while (maxNorm(residual, resid) > m_tolerance && iteration < maxiteration);

        if (iteration == maxiteration)
            {
            m_exec_conf->msg->warning()
                << "The RATTLE integrator needed an unusual high number of iterations!" << std::endl
                << "It is recomended to change the initial configuration or lower the step size."
                << std::endl;
            }

        h_net_force.data[j].x -= mu * normal.x;
        h_net_force.data[j].y -= mu * normal.y;
        h_net_force.data[j].z -= mu * normal.z;

        h_net_virial.data[0 * net_virial_pitch + j] -= mu * normal.x * h_pos.data[j].x;
        h_net_virial.data[1 * net_virial_pitch + j]
            -= 0.5 * mu * (normal.y * h_pos.data[j].x + normal.x * h_pos.data[j].y);
        h_net_virial.data[2 * net_virial_pitch + j]
            -= 0.5 * mu * (normal.z * h_pos.data[j].x + normal.x * h_pos.data[j].z);
        h_net_virial.data[3 * net_virial_pitch + j] -= mu * normal.y * h_pos.data[j].y;
        h_net_virial.data[4 * net_virial_pitch + j]
            -= 0.5 * mu * (normal.y * h_pos.data[j].z + normal.z * h_pos.data[j].y);
        h_net_virial.data[5 * net_virial_pitch + j] -= mu * normal.z * h_pos.data[j].z;
        }
    }

namespace detail
    {
template<class Manifold> void export_TwoStepRATTLEBD(pybind11::module& m, const std::string& name)
    {
    pybind11::class_<TwoStepRATTLEBD<Manifold>,
                     TwoStepLangevinBase,
                     std::shared_ptr<TwoStepRATTLEBD<Manifold>>>(m, name.c_str())
        .def(pybind11::init<std::shared_ptr<SystemDefinition>,
                            std::shared_ptr<ParticleGroup>,
                            Manifold,
                            std::shared_ptr<Variant>,
                            bool,
                            bool,
                            Scalar>())
        .def_property("tolerance",
                      &TwoStepRATTLEBD<Manifold>::getTolerance,
                      &TwoStepRATTLEBD<Manifold>::setTolerance);
    }
    } // end namespace detail
    } // end namespace md
    } // end namespace hoomd

#endif // #ifndef __TWO_STEP_RATTLE_BD_H__
