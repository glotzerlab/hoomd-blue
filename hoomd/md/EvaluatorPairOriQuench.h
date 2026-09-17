// Copyright (c) 2009-2026 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

// $Id$
// $URL$

#ifndef __PAIR_EVALUATOR_DIPOLE_H__
#define __PAIR_EVALUATOR_DIPOLE_H__

#ifndef __HIPCC__
#include <string>
#endif

#ifdef ENABLE_HIP
#include <hip/hip_runtime.h>
#endif
#include "hoomd/VectorMath.h"
#include <iostream>
/*! \file EvaluatorPairOriQuench.h
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
class EvaluatorPairOriQuench
    {
    public:
    struct param_type
        {
        Scalar A;     //! The electrostatic energy scale.
	Scalar tau;
	Scalar d;

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

        HOSTDEVICE param_type() : A(0), tau(0), d(0) { }

#ifndef __HIPCC__

        param_type(pybind11::dict v, bool managed)
            {
            A = v["A"].cast<Scalar>();
            tau = v["tau"].cast<Scalar>();
            d = v["d"].cast<Scalar>();
            }

        pybind11::object toPython()
            {
            pybind11::dict v;
            v["A"] = A;
            v["tau"] = tau;
            v["d"] = d;
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
        \param _A Electrostatic energy scale
        \param _kappa Inverse screening length
        \param _params Per type pair parameters of this potential
    */
    HOSTDEVICE EvaluatorPairOriQuench(Scalar3& _dr,
                                   Scalar4& _quat_i,
                                   Scalar4& _quat_j,
                                   Scalar _rcutsq,
                                   const param_type& _params)
        : dr(_dr), rcutsq(_rcutsq), quat_i(_quat_i), quat_j(_quat_j),
          mu_i {0, 0, 0}, mu_j {0, 0, 0}, A(_params.A), tau(_params.tau),
	  d(_params.d)
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

    //! whether pair potential requires charges
    HOSTDEVICE static bool needsCharge()
        {
        return false;
        }

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

    //! Accept the optional tags
    /*! \param tag_i Tag of particle i
        \param tag_j Tag of particle j
    */
    HOSTDEVICE void setTags(unsigned int tagi, unsigned int tagj) { }

    //! Accept the optional charge values
    /*! \param qi Charge of particle i
        \param qj Charge of particle j
    */
    HOSTDEVICE void setCharge(Scalar qi, Scalar qj)
        {
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

	rvec.z = 0;

	Scalar r = fast::sqrt(rsq);

        Scalar rinv = 1/r;
        Scalar r2inv = rinv * rinv;	

        // convert dipole vector in the body frame of each particle to space
        // frame
        vec3<Scalar> p_i = rotate(quat<Scalar>(quat_i), mu_i);
        vec3<Scalar> p_j = rotate(quat<Scalar>(quat_j), mu_j);

        vec3<Scalar> f = vec3<Scalar>(0, 0, 0);
        vec3<Scalar> t_i = vec3<Scalar>(0, 0, 0);
        vec3<Scalar> t_j = vec3<Scalar>(0, 0, 0);
        Scalar e = Scalar(0.0);

        bool i_interactions = (mu_i != vec3<Scalar>(0, 0, 0));
        bool j_interactions = (mu_j != vec3<Scalar>(0, 0, 0));

        // dipole-dipole
        if (i_interactions)
            {
	    
	    Scalar rT_dot_mu = rvec.y*p_i.x - rvec.x*p_i.y; 

	    Scalar prefactor = A;

	    if( rT_dot_mu < 0 ) prefactor *= -1;

	    e = -prefactor*rT_dot_mu*rinv;

	    if( r > d && tau > 0)
	    	{
	    	Scalar prefactor2 = fast::exp(-tau*(r-d));
	    	e *= prefactor2;
	    	prefactor *= prefactor2;

	    	f.x = e*tau*rvec.x*rinv;
	    	f.y = e*tau*rvec.y*rinv;
		}

	    f.x += -prefactor*p_i.y*rinv + e*rvec.x*r2inv;
	    f.y += prefactor*p_i.x*rinv + e*rvec.y*r2inv;

	    t_i.z = -prefactor*dot(rvec,p_i);

	    }



        if (j_interactions)
            {
	    Scalar rT_dot_mu = rvec.y*p_j.x - rvec.x*p_j.y; 
	    Scalar prefactor = A;

	    if( rT_dot_mu < 0 ) prefactor *= -1;

	    Scalar e1 = -prefactor*rT_dot_mu*rinv;

	    if( r > d && tau > 0)
	    	{
	    	Scalar prefactor2 = fast::exp(-tau*(r-d));
	    	e1 *= prefactor2;
	    	prefactor *= prefactor2;

	    	f.x += e*tau*rvec.x*rinv;
	    	f.y += e*tau*rvec.y*rinv;
		}

	    f.x -= prefactor*p_j.y*rinv - e1*rvec.x*r2inv;
	    f.y -= -prefactor*p_j.x*rinv - e1*rvec.y*r2inv;

	    e += e1;

	    t_j.z = -prefactor*dot(rvec,p_j);
	    }

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
        return "oriquench";
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
    vec3<Scalar> mu_i;      /// Magnetic moment for ith particle
    vec3<Scalar> mu_j;      /// Magnetic moment for jth particle
    Scalar A;
    Scalar tau;
    Scalar d;
    // const param_type &params;   //!< The pair potential parameters
    };

    } // end namespace md
    } // end namespace hoomd

#endif // __PAIR_EVALUATOR_DIPOLE_H__
