// Copyright (c) 2009-2025 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

#ifndef __PAIR_EVALUATOR_FRICTIONLJBASE_H__
#define __PAIR_EVALUATOR_FRICTIONLJBASE_H__

#ifndef __HIPCC__
#include <string>
#endif

#ifdef ENABLE_HIP
#include <hip/hip_runtime.h>
#endif
#include "hoomd/RNGIdentifiers.h"
#include "hoomd/RandomNumbers.h"
#include "hoomd/VectorMath.h"

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

#define TOLERANCE_DU 1e-6

namespace hoomd
    {
namespace md
    {
template<class Derived> class EvaluatorPairFrictionLJBase
    {
    public:
    // Add the eval_factors() method declaration
    HOSTDEVICE void eval_factors(Scalar& factor_f, Scalar& factor_r, Scalar w, Scalar du)
        {
        static_cast<Derived*>(this)->eval_factors(factor_f, factor_r, w, du);
        }

    struct param_type
        {
        Scalar sigma_6;     // Sigma^6 parameter of the LJ-Potential
        Scalar epsilon_x_4; // 4*Epsilon parameter of the LJ-Potential
        Scalar
            gamma; // Gamma parameter of the frictional contacts (optional depends on friction type)
        Scalar
            kappa; // kappa parameter of the frictional contacts (optional depends on friction type)
        Scalar pair_temp; // Temperature of the pairwise thermostat

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

        HOSTDEVICE param_type() : sigma_6(0), epsilon_x_4(0), gamma(0), kappa(0), pair_temp(0) { }

#ifndef __HIPCC__

        param_type(pybind11::dict v, bool managed)
            {
            auto sigma(v["sigma"].cast<Scalar>());
            auto epsilon(v["epsilon"].cast<Scalar>());
            Scalar gamma_f = v.contains("gamma_f") ? v["gamma_f"].cast<Scalar>() : Scalar(0.0);
            Scalar kappa_f = v.contains("kappa_f") ? v["kappa_f"].cast<Scalar>() : Scalar(0.0);
            auto kT(v["kT"].cast<Scalar>());

            sigma_6 = sigma * sigma * sigma * sigma * sigma * sigma;
            epsilon_x_4 = Scalar(4.0) * epsilon;
            gamma = gamma_f;
            kappa = kappa_f;
            pair_temp = kT;
            }

        pybind11::object toPython()
            {
            pybind11::dict v;
            v["sigma"] = pow(sigma_6, 1. / 6.);
            v["epsilon"] = epsilon_x_4 / 4.0;
            if (gamma != Scalar(0.0))
                v["gamma_f"] = gamma;
            if (kappa != Scalar(0.0))
                v["kappa_f"] = kappa;
            v["kT"] = Scalar(1.0) * pair_temp;
            return v;
            }

#endif
        }
#if HOOMD_LONGREAL_SIZE == 32
        __attribute__((aligned(8)));
#else
        __attribute__((aligned(16)));
#endif

    //! Constructs the pair potential evaluator
    /*! \param _dr Displacement vector between particle centers of mass
        \param _rcutsq Squared distance at which the potential goes to 0
        \param _angvel_i Angular veloctiy ot i^{th} particle
        \param _angvel_j Angular veloctiy ot j^{th} particle
        \param _dia_i Diameter ot i^{th} particle
        \param _dia_j Diameter ot i^{th} particle
        \param _epsilon_x_4 Epsilon of the LJ Potential
        \param _sigma_6 Sigma of the LJ Potential
        \param gamma Gamma parameter of the frictional contact
        \param kappa Kappa parameter of the frictional contact
        \param pair_temp Pair temperature of the frictional contact
        \param _params Per type pair parameters of this potential
    */
    HOSTDEVICE EvaluatorPairFrictionLJBase(Scalar3& _dr,
                                           Scalar3& _angvel_i,
                                           Scalar3& _angvel_j,
                                           Scalar3& _dv,
                                           Scalar _dia_i,
                                           Scalar _dia_j,
                                           Scalar _rcutsq,
                                           const param_type& _params)
        : dr(_dr), rcutsq(_rcutsq), angvel_i(_angvel_i), angvel_j(_angvel_j), dv(_dv),
          dia_i(_dia_i), dia_j(_dia_j),
          lj1(_params.epsilon_x_4 * _params.sigma_6 * _params.sigma_6),
          lj2(_params.epsilon_x_4 * _params.sigma_6), gamma(_params.gamma), kappa(_params.kappa),
          pair_temp(_params.pair_temp)
        {
        }

    //! Whether the pair potential needs particle tags.
    HOSTDEVICE static bool needsTags()
        {
        return false;
        }

    //! whether pair potential requires charges
    HOSTDEVICE static bool needsCharge()
        {
        return true;
        }

    //! whether pair potential requires nu_ito
    HOSTDEVICE static bool needsNu()
        {
        return true;
        }

    // Seed, Timestep, and the particle ids are necessary for the correlation of the pair noise
    // (equation 26 and 27 of manuscript)
    HOSTDEVICE void
    set_seed_ij_timestep(uint16_t seed, unsigned int i, unsigned int j, uint64_t timestep)
        {
        m_seed = seed;
        m_i = i;
        m_j = j;
        m_timestep = timestep;
        }

    //! Set the timestep size
    HOSTDEVICE void setDeltaT(Scalar dt)
        {
        m_deltaT = dt;
        }

    //! Accept the optional tags
    /*! \param tag_i Tag of particle i
        \param tag_j Tag of particle j
    */
    HOSTDEVICE void setTags(unsigned int tagi, unsigned int tagj) { }

    //! Accept the optional charge values
    /*! \param qi Charge of particle i
        \param qj Charge of particle j
    */
    HOSTDEVICE void setCharge(Scalar qi, Scalar qj) { }

    //! Accept the optional nu value
    /*! \param nu_ito nu value for the ito formalism
     */
    HOSTDEVICE void setNu(Scalar nu_ito)
        {
        nu_ij = nu_ito;
        }
    //! Set the third law value
    /*! \param third_law
     */
    HOSTDEVICE void setThirdLaw(bool third_law)
        {
        m_third_law = third_law;
        }

    //! Evaluate the force and energy
    /*! \param force Output parameter to write the computed force.
        \param pair_eng Output parameter to write the computed pair energy.
        \param torque_i The torque exerted on the i^th particle.
        \param torque_j The torque exerted on the j^th particle.
        \return True if they are evaluated or false if they are not because
            we are beyond the cutoff.
    */
    HOSTDEVICE bool evaluate(Scalar3& force, Scalar& pair_eng, Scalar3& torque_i, Scalar3& torque_j)
        {
        vec3<Scalar> rvec(dr);
        vec3<Scalar> w_i(angvel_i);
        vec3<Scalar> w_j(angvel_j);
        vec3<Scalar> v_ij(dv);
        Scalar d_i(dia_i);
        Scalar d_j(dia_j);

        Scalar rsq = dot(rvec, rvec);

        if (rsq > rcutsq)
            return false;

        //! Define the force and torques which get applied to the particles
        vec3<Scalar> f;
        vec3<Scalar> t_j(0.0, 0.0, 0.0);
        vec3<Scalar> t_i(0.0, 0.0, 0.0);

        //! Calculation of the repulsive LJ-Force
        Scalar rinv = fast::rsqrt(rsq);
        Scalar r2inv = Scalar(1.0) / rsq;
        Scalar r6inv = r2inv * r2inv * r2inv;

        //! Calculation of the repulsive LJ-Force
        Scalar force_divr = r2inv * r6inv * (Scalar(12.0) * lj1 * r6inv - Scalar(6.0) * lj2);
        f = force_divr * rvec;

        //! Calculation of the rotational friction Force

        //! e_ij: unit vector between center of masses
        vec3<Scalar> e_ij = Scalar(-1.0) * rvec * rinv;

        //! Project v_ij onto perpendicular to e_ij
        vec3<Scalar> P_e_v = v_ij - (dot(v_ij, e_ij) * e_ij);

        //! Calculate (w_i*R_i+w_j*R_j)
        vec3<Scalar> wiRiwjRj = Scalar(0.5) * (d_i * w_i + d_j * w_j);

        //! Calculate the relative tangential velocity u_ij at the contact point
        vec3<Scalar> u_ij = P_e_v - cross(wiRiwjRj, e_ij);
        Scalar du = fast::sqrt(dot(u_ij, u_ij));

        //! du should not be too small to avoid division by zero
        if (TOLERANCE_DU < du)
            {
            unsigned int m_oi, m_oj;
            int sign_xi = 1;
            // initialize the RNG
            if (m_i > m_j)
                {
                m_oi = m_j;
                m_oj = m_i;
                }
            else
                {
                m_oi = m_i;
                m_oj = m_j;
                if (!m_third_law)
                    sign_xi
                        = -1; // If the neighborlist is only halfe the antisymmetric correlations
                              // have to be handled here! (Maybe there is a better way?)
                }

            //! Init Random number Generator
            hoomd::RandomGenerator rng(
                hoomd::Seed(hoomd::RNGIdentifier::EvaluatorPairFrictionLJBase, m_timestep, m_seed),
                hoomd::Counter(m_oi, m_oj));

            //! Distant dependend factor
            Scalar w = fast::sqrt(dot(f, f));

            Scalar factor_f = Scalar(0.0);
            Scalar factor_r = Scalar(0.0);

            //! How the factors are calculated depends on the friction type (currently implemented
            //! in EvaluatorPairFrictionLJVariants.h is linear, constant and coulombNewton)
            eval_factors(factor_f, factor_r, w, du);

            //! Calculation of the Rotational Friction force and torque
            vec3<Scalar> f_f = factor_f * (P_e_v + cross(e_ij, wiRiwjRj));
            vec3<Scalar> exff = cross(e_ij, f_f);

            //! Noise for rotational friction
            Scalar sigma_f = fast::sqrt(pair_temp / m_deltaT);

            Scalar xi_x = sign_xi * hoomd::NormalDistribution<Scalar>(sigma_f)(rng);
            Scalar xi_y = sign_xi * hoomd::NormalDistribution<Scalar>(sigma_f)(rng);
            Scalar xi_z = sign_xi * hoomd::NormalDistribution<Scalar>(sigma_f)(rng);

            Scalar N_x = hoomd::NormalDistribution<Scalar>(sigma_f)(rng);
            Scalar N_y = hoomd::NormalDistribution<Scalar>(sigma_f)(rng);
            Scalar N_z = hoomd::NormalDistribution<Scalar>(sigma_f)(rng);

            vec3<Scalar> xi = vec3<Scalar>(xi_x, xi_y, xi_z);
            vec3<Scalar> N = vec3<Scalar>(N_x, N_y, N_z);

            vec3<Scalar> f_r = factor_r * (xi - dot(xi, e_ij) * e_ij - cross(e_ij, N));
            vec3<Scalar> t_r = factor_r * (cross(e_ij, xi) + N - dot(N, e_ij) * e_ij);

            //! Calculate the torque on the particles
            t_i = (d_i / Scalar(2.0)) * (exff + t_r);
            t_j = (d_j / Scalar(2.0)) * (exff + t_r);

            //! Add all forces
            f = f + f_f + f_r;
            }

        force = vec_to_scalar3(f);
        torque_i = vec_to_scalar3(t_i);
        torque_j = vec_to_scalar3(t_j);
        // pair_eng = e;

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
        return "FrictionLJBase";
        }
#endif

    protected:
    Scalar3 dr;    //!< Stored vector pointing between particle centers of mass
    Scalar rcutsq; //!< Stored rcutsq from the constructor
    Scalar3 angvel_i,
        angvel_j;        //!< Stored angular momentum of ith and jth particle from the constructor
    Scalar3 dv;          //!< Stored velocity difference vij between the ith and jth particle
    Scalar dia_i, dia_j; //!< Stored diameter of ith and jth particle from the constructor
    Scalar nu_ij;        //!< Factor which depends if Ito or the Stratonovich case.
    bool m_third_law;    //!< Boolean storing if only a half neighborlist is used
    Scalar lj1, lj2;  //!< lj1 and lj2 parameter extracted from the params passed to the constructor
    Scalar gamma;     //!< Optional gamma parameter from the constructor
    Scalar kappa;     //!< Optional kappa parameter from the constructor
    Scalar pair_temp; //!< User set temperature for the DPD like PRNG
    Scalar m_deltaT;  //!< Timestep size stored from constructor
    uint16_t m_seed;  //!< User set seed for thermostat PRNG
    unsigned int m_i; //!< Index of first particle. For use in PRNG
    unsigned int m_j; //!< Index of second particle. For use in PRNG
    uint64_t m_timestep; //!< timestep for use in PRNG
    // const param_type &params;   //!< The pair potential parameters
    };

    } // end namespace md
    } // end namespace hoomd

#endif // __PAIR_EVALUATOR_FRICTIONLJBASE_H__
