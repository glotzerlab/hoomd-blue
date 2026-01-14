// Copyright (c) 2009-2025 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

// $Id$
// $URL$

#ifndef __PAIR_EVALUATOR_YLZ_H__
#define __PAIR_EVALUATOR_YLZ_H__

#ifndef __HIPCC__
#include <string>
#endif

#ifdef ENABLE_HIP
#include <hip/hip_runtime.h>
#endif
#include "hoomd/HOOMDMath.h"
#include "hoomd/VectorMath.h"
#include <iostream>
/*! \file EvaluatorYLZ.h
    \brief Defines the dipole potential
*/

// need to declare these class methods with __device__ qualifiers when building
// in nvcc.  HOSTDEVICE is __host__ __device__ when included in nvcc and blank
// when included into the host compiler
#ifdef __HIPCC__
#define HOSTDEVICE __host__ __device__
#define DEVICE __device__
#else
#define HOSTDEVICE
#define DEVICE
#endif

namespace hoomd
    {
namespace md
    {
class EvaluatorPairYLZ
    {
    public:
    struct param_type
        {
        Scalar eps;  //! the energy scale of the potential.
        Scalar phi;  //! Sets the local curvature of the particles: phi = sin(theta_0).
        Scalar beta; //! sets the scale of the orietnational coupling.
        Scalar rmin; //! cutoff of the first minimum where the Lennard-Jones begins.
        int twozeta; //! exponential of the cosine potential.

#ifdef ENABLE_HIP
        //! Set CUDA memory hints
        void set_memory_hint() const
            {
            // default implementation does nothing
            }
#endif

        //! Load dynamic data members into shared memory and increase pointer
        /*! \param ptr Pointer to load data to (will be incremented)
            \param available_bytes Size of remaining shared memory
            allocation
        */
        DEVICE void load_shared(char*& ptr, unsigned int& available_bytes) { }

        HOSTDEVICE void allocate_shared(char*& ptr, unsigned int& available_bytes) const { }

        HOSTDEVICE param_type() : eps(0), phi(0), beta(0), rmin(0), twozeta(0) { }

#ifndef __HIPCC__

        param_type(pybind11::dict v, bool managed)
            {
            eps = v["eps"].cast<Scalar>();
            phi = v["phi"].cast<Scalar>();
            beta = v["beta"].cast<Scalar>();
            rmin = v["rmin"].cast<Scalar>();
            twozeta = v["twozeta"].cast<unsigned int>();
            }

        pybind11::object toPython()
            {
            pybind11::dict v;
            v["eps"] = eps;
            v["phi"] = phi;
            v["beta"] = beta;
            v["rmin"] = rmin;
            v["twozeta"] = twozeta;
            return v;
            }

#endif
        }
#if HOOMD_LONGREAL_SIZE == 32
        __attribute__((aligned(8)));
#else
        __attribute__((aligned(16)));
#endif

    struct shape_type
        {
        vec3<Scalar> mu;

        //! Load dynamic data members into shared memory and increase pointer
        /*! \param ptr Pointer to load data to (will be incremented)
            \param available_bytes Size of remaining shared memory allocation
        */
        DEVICE void load_shared(char*& ptr, unsigned int& available_bytes) { }

        HOSTDEVICE void allocate_shared(char*& ptr, unsigned int& available_bytes) const { }

        HOSTDEVICE shape_type() : mu {0, 0, 0} { }

#ifndef __HIPCC__

        shape_type(vec3<Scalar> mu_, bool managed = false) : mu(mu_) { }

        shape_type(pybind11::object mu_obj, bool managed)
            {
            auto mu_ = (pybind11::tuple)mu_obj;
            mu = vec3<Scalar>(mu_[0].cast<Scalar>(), mu_[1].cast<Scalar>(), mu_[2].cast<Scalar>());
            }

        pybind11::object toPython()
            {
            return pybind11::make_tuple(mu.x, mu.y, mu.z);
            }
#endif // __HIPCC__

#ifdef ENABLE_HIP
        //! Attach managed memory to CUDA stream
        void set_memory_hint() const { }
#endif
        };

    //! Constructs the pair potential evaluator
    /*! \param _dr Displacement vector between particle centers of mass
        \param _rcutsq Squared distance at which the potential goes to 0
        \param _quat_i Quaternion of i^{th} particle
        \param _quat_j Quaternion of j^{th} particle
        \param _eps YLZ energy scale
        \param _phi
        \param _beta
        \param _rmin
        \param _params Per type pair parameters of this potential
    */
    HOSTDEVICE EvaluatorPairYLZ(Scalar3& _dr,
                                Scalar4& _quat_i,
                                Scalar4& _quat_j,
                                Scalar _rcutsq,
                                const param_type& _params)
        : dr(_dr), rcutsq(_rcutsq), quat_i(_quat_i), quat_j(_quat_j), mu_i {0, 0, 0},
          mu_j {0, 0, 0}, eps(_params.eps), phi(_params.phi), beta(_params.beta),
          rmin(_params.rmin), twozeta(_params.twozeta)
        {
        }

    //! Whether the pair potential uses shape.
    HOSTDEVICE static bool needsShape()
        {
        return true;
        }

    //! Whether the pair potential needs particle tags.
    HOSTDEVICE static bool needsTags()
        {
        return false;
        }

    //! Accept the optional tags
    /*! \param tag_i Tag of particle i
        \param tag_j Tag of particle j
    */
    HOSTDEVICE void setTags(unsigned int tagi, unsigned int tagj) { }

    //! whether pair potential requires charges
    HOSTDEVICE static bool needsCharge()
        {
        return false;
        }

    //! Accept the optional charge values.
    /*! \param qi Charge of particle i
        \param qj Charge of particle j
    */
    HOSTDEVICE void setCharge(Scalar qi, Scalar qj) { }

    /// Whether the potential implements the energy_shift parameter
    HOSTDEVICE static bool constexpr implementsEnergyShift()
        {
        return false;
        }

    //! Accept the optional shape values
    /*! \param shape_i Shape of particle i
        \param shape_j Shape of particle j
    */
    HOSTDEVICE void setShape(const shape_type* shapei, const shape_type* shapej)
        {
        mu_i = shapei->mu;
        mu_j = shapej->mu;
        }

    //! Evaluate the force and energy
    /*! \param force Output parameter to write the computed force.
        \param pair_eng Output parameter to write the computed pair energy.
        \param energy_shift If true, the potential must be shifted so that
            V(r) is continuous at the cutoff.
        \param torque_i The torque exerted on the i^th particle.
        \param torque_j The torque exerted on the j^th particle.
        \return True if they are evaluated or false if they are not because
            we are beyond the cutoff.
    */
    HOSTDEVICE bool evaluate(Scalar3& force,
                             Scalar& pair_eng,
                             bool energy_shift,
                             Scalar3& torque_i,
                             Scalar3& torque_j)
        {
        vec3<Scalar> rvec(dr);
        Scalar rsq = dot(rvec, rvec);

        if (rsq > rcutsq)
            return false;

        Scalar rinv = fast::rsqrt(rsq);
        Scalar r2inv = Scalar(1.0) / rsq;
        Scalar r4inv = r2inv * r2inv;
        Scalar rminsq = rmin * rmin;

        // convert symmetry axis vector in the body frame
        vec3<Scalar> p_i = rotate(quat<Scalar>(quat_i), mu_i);
        vec3<Scalar> p_j = rotate(quat<Scalar>(quat_j), mu_j);

        vec3<Scalar> f;
        vec3<Scalar> t_i;
        vec3<Scalar> t_j;
        Scalar e = Scalar(0.0);        // define energy of the system
        Scalar r = Scalar(1.0) / rinv; // define distance between particles

        bool dipole_i_interactions = (mu_i != vec3<Scalar>(0, 0, 0));
        bool dipole_j_interactions = (mu_j != vec3<Scalar>(0, 0, 0));
        bool dipole_interactions = dipole_i_interactions && dipole_j_interactions;

        // dipole-dipole
        if (dipole_interactions)
            {
            vec3<Scalar> rhat = rvec * rinv;
            Scalar a = dot(p_i, p_j) - dot(p_i, rhat) * dot(p_j, rhat)
                       + phi * dot((p_i - p_j), rhat) - phi * phi;
            vec3<Scalar> da_drhat = p_i * (phi - dot(p_j, rhat)) - p_j * (phi + dot(p_i, rhat));
            vec3<Scalar> da_dni = p_j + rhat * (phi - dot(p_j, rhat));
            vec3<Scalar> da_dnj = p_i - rhat * (phi + dot(p_i, rhat));

            if (rsq < rminsq)
                {
                Scalar dUdr = eps * Scalar(4.0) * rinv * (rminsq * r2inv - rminsq * rminsq * r4inv);
                vec3<Scalar> dU_drhat = -beta * da_drhat;

                // forces
                f += -dUdr * rhat - (dU_drhat - dot(dU_drhat, rhat) * rhat) * rinv;

                // torques
                vec3<Scalar> dU_dni = -beta * da_dni;
                vec3<Scalar> dU_dnj = -beta * da_dnj;

                t_i += cross(dU_dni, p_i);
                t_j += cross(dU_dnj, p_j);

                e += eps
                     * ((rminsq * rminsq * r4inv - Scalar(2.0) * rminsq * r2inv)
                        - beta * (a - Scalar(1.0)));
                }
            else
                {
                Scalar rcut = fast::sqrt(rcutsq);
                Scalar zeta = Scalar(twozeta) / Scalar(2.0);
                Scalar gamma = Scalar(1.0) + beta * (a - Scalar(1.0));
                Scalar inv_rmin_diff = Scalar(1.0) / (rcut - rmin);
                Scalar rdiff = r - rmin;
                Scalar pi = Scalar(M_PI);
                Scalar cos_val = fast::cos(pi / Scalar(2.0) * rdiff * inv_rmin_diff);
                Scalar sin_val = fast::sin(pi / Scalar(2.0) * rdiff * inv_rmin_diff);
                Scalar cos_pow_minus = fast::pow(cos_val, twozeta - 1);
                Scalar dUdr = gamma * eps * zeta * pi * inv_rmin_diff * cos_pow_minus * sin_val;
                Scalar U_a = -eps * cos_pow_minus * cos_val;
                vec3<Scalar> dU_drhat = beta * U_a * da_drhat;
                f += -dUdr * rhat - (dU_drhat - dot(dU_drhat, rhat) * rhat) * rinv;
                e += U_a * gamma;

                // torques
                vec3<Scalar> dU_dni = beta * U_a * da_dni;
                vec3<Scalar> dU_dnj = beta * U_a * da_dnj;
                t_i += cross(dU_dni, p_i); // flip cross product to remove negative
                t_j += cross(dU_dnj, p_j); // flip cross product to remove negative
                }
            }
        // dipole i - electrostatic j

        force = vec_to_scalar3(f);
        torque_i = vec_to_scalar3(t_i);
        torque_j = vec_to_scalar3(t_j);
        pair_eng = e;
        return true;
        }

    DEVICE Scalar evalPressureLRCIntegral()
        {
        return 0;
        }

    DEVICE Scalar evalEnergyLRCIntegral()
        {
        return 0;
        }

#ifndef __HIPCC__
    //! Get the name of the potential
    /*! \returns The potential name.
     */
    static std::string getName()
        {
        return "ylz";
        }
    static std::string getShapeParamName()
        {
        return "Mu";
        }
    std::string getShapeSpec() const
        {
        throw std::runtime_error("Shape definition not supported for this pair potential.");
        }
#endif

    protected:
    Scalar3 dr;             //!< Stored vector pointing between particle centers of mass
    Scalar rcutsq;          //!< Stored rcutsq from the constructor
    Scalar4 quat_i, quat_j; //!< Stored quaternion of ith and jth particle from constructor
    vec3<Scalar> mu_i;      /// Symmetry axis for the ith particle
    vec3<Scalar> mu_j;      /// Symmetry axis for jth particle
    Scalar eps;
    Scalar phi;
    Scalar beta;
    Scalar rmin;
    unsigned int twozeta;
    // const param_type &params;   //!< The pair potential parameters
    };

    } // end namespace md
    } // end namespace hoomd

#endif // __PAIR_EVALUATOR_YLZ_H__
