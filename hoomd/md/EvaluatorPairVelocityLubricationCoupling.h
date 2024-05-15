// Copyright (c) 2009-2022 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

#ifndef __PAIR_EVALUATOR_VELOCITY_LUBRICATION_COUPLING_H__
#define __PAIR_EVALUATOR_VELOCITY_LUBRICATION_COUPLING_H__

#ifndef __HIPCC__
#include <string>
#endif

#ifdef ENABLE_HIP
#include <hip/hip_runtime.h>
#endif
#include "hoomd/HOOMDMath.h"
#include "hoomd/VectorMath.h"
#include <iostream>
/*! \file EvaluatorPairVelocityLubricationCoupling.h
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
class EvaluatorPairVelocityLubricationCoupling
    {
    public:
    struct param_type
        {
        Scalar mu;   //! viscocity 
	bool take_momentum;
	bool take_velocity;


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

        HOSTDEVICE param_type() : mu(0), take_momentum(true), take_velocity(true) { }

#ifndef __HIPCC__

        param_type(pybind11::dict v, bool managed)
            {
            mu = v["mu"].cast<Scalar>();
            take_momentum = v["take_momentum"].cast<bool>();
            take_velocity = v["take_velocity"].cast<bool>();
            }

        pybind11::object toPython()
            {
            pybind11::dict v;
            v["mu"] = mu;
            v["take_momentum"] = take_momentum;
            v["take_velocity"] = take_velocity;
            return std::move(v);
            }

#endif
        }
#ifdef SINGLE_PRECISION
        __attribute__((aligned(8)));
#else
        __attribute__((aligned(16)));
#endif

    // Nullary structure required by AnisoPotentialPair.
    struct shape_type
        {
        //! Load dynamic data members into shared memory and increase pointer
        /*! \param ptr Pointer to load data to (will be incremented)
            \param available_bytes Size of remaining shared memory allocation
        */
        DEVICE void load_shared(char*& ptr, unsigned int& available_bytes) { }

        HOSTDEVICE void allocate_shared(char*& ptr, unsigned int& available_bytes) const { }

        HOSTDEVICE shape_type() { }

#ifndef __HIPCC__

        shape_type(pybind11::object shape_params, bool managed) { }

        pybind11::object toPython()
            {
            return pybind11::none();
            }
#endif

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
        \param _mu effective viscocity 
        \param _params Per type pair parameters of this potential
    */
    HOSTDEVICE EvaluatorPairVelocityLubricationCoupling(Scalar3& _dr,
                                                Scalar4& _quat_i,
                                                Scalar4& _quat_j,
                                                Scalar _rcutsq,
                                                const param_type& _params)
        : dr(_dr), rcutsq(_rcutsq), quat_i(_quat_i), quat_j(_quat_j), ang_i {0, 0, 0},ang_j {0, 0, 0},ang_mom {0, 0, 0},velocity {0, 0, 0}, diameter_i(0),diameter_j(0),massi(0),massj(0), mu(_params.mu), take_momentum(_params.take_momentum), take_velocity(_params.take_velocity)

        {
        }

    //! uses diameter
    HOSTDEVICE static bool needsDiameter()
        {
        return true;
        }

    HOSTDEVICE static bool needsMass()
        {
        return true;
        }

    //! Whether the pair potential uses shape.
    HOSTDEVICE static bool needsShape()
        {
        return false;
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

    //! Whether the pair potential needs particle angular momentum
    HOSTDEVICE static bool needsAngularMomentum()
        {
        return true;
        }

    //! Whether the pair potential needs particle velocity 
    HOSTDEVICE static bool needsVelocity()
        {
        return true;
        }

    /// Whether the potential implements the energy_shift parameter
    HOSTDEVICE static bool constexpr implementsEnergyShift()
        {
        return false;
        }

    //! Accept the optional diameter values
    /*! \param di Diameter of particle i
        \param dj Diameter of particle j
    */
    HOSTDEVICE void setDiameter(Scalar di, Scalar dj)
        {
        diameter_i = di;
        diameter_j = dj;
        }

    HOSTDEVICE void setMass(Scalar mass_i, Scalar mass_j)
        {
        massi = mass_i;
        massj = mass_j;
        }

    //! Accept the optional shape values
    /*! \param shape_i Shape of particle i
        \param shape_j Shape of particle j
    */
    HOSTDEVICE void setShape(const shape_type* shapei, const shape_type* shapej) { }

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

    //! Accept the optional charge values
    /*! \param ai Angular momentum of particle i
        \param aj Angular momentum of particle j
    */
    HOSTDEVICE void setAngularMomentum(vec3<Scalar> ai, vec3<Scalar> aj)
        {
	if(take_momentum)
		{
		ang_i = ai;
		ang_j = ai;
		ang_mom = ai+aj;
		}
	else 
		ang_i = vec3<Scalar>(0,0,0);
		ang_j = vec3<Scalar>(0,0,0);
		ang_mom = vec3<Scalar>(0,0,0);
	}

    HOSTDEVICE void setVelocities(vec3<Scalar> vi)
        {
	if(take_velocity)
		{
		velocity = vi;
		}
	else 
		velocity = vec3<Scalar>(0,0,0);
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
            vec3<Scalar> rvec(dr); /// Turn distance between particles into a vector
            Scalar rsq = dot(rvec, rvec); /// get square distance 

            if (rsq > rcutsq)
                return false;

            else
                {
                Scalar diameter = 0.5 * (diameter_i + diameter_j);
	            Scalar beta = diameter_j / diameter_i;
	            Scalar betainv = diameter_i / diameter_j;
	            Scalar r_a = diameter_i/2.0;
	            Scalar r_b = diameter_j/2.0;
	            Scalar pi = 2.0 * acos(0.0);

                Scalar rinv = fast::rsqrt(rsq);
	            vec3<Scalar> n_v = -1.0 * rvec * rinv; // Normal vector connecting the two particles
	            
	            Scalar sphere_inertia_i = 0.4 * massi * r_a*r_a; // Sphere moment of inertia assuming a uniform density
	            Scalar sphere_inertia_j = 0.4 * massj * r_b*r_b;
	            vec3<Scalar> ang_vel_i =  ang_i / sphere_inertia_i; 
	            vec3<Scalar> ang_vel_j =  ang_j / sphere_inertia_j; 
	            vec3<Scalar> sum_avel =  ang_vel_i + ang_vel_j;	    
	            vec3<Scalar> diff_avel =  ang_vel_i - ang_vel_j;	    
	            vec3<Scalar> delta_U =  -1.0 * velocity;    
                Scalar d = 1.0 / rinv - diameter;
	            Scalar laminv = ((r_a+r_b)/2)/d;
                Scalar log_laminv = fast::log(laminv);
	            Scalar dot_vel = dot(n_v,delta_U);
	            vec3<Scalar> cross_vel = cross(n_v,delta_U);
	            Scalar dot_angvel_sum = dot(n_v,sum_avel);
	            Scalar dot_angvel_diff = dot(n_v,diff_avel);
	            vec3<Scalar> cross_angvel_diff = cross(diff_avel,n_v);
	            vec3<Scalar> cross_angvel_sum = cross(sum_avel,n_v);
	            Scalar YA11 = 6.0*pi*r_a*((8.0*beta+4.0*beta*beta+8.0*beta*beta*beta)/
		        	  (15.0*(1.0+beta)*(1.0+beta)*(1.0+beta))*log_laminv); 
	            Scalar YB11 = -4.0 * pi * r_a*r_a * (beta*(4.0+beta)/
		            	(5.0*(1.0+beta)*(1.0+beta))*log_laminv);
	            Scalar YC12 = 8.0*pi*r_a*r_a*r_a * (beta*beta/(10.0+10.0*beta))*log_laminv;
	            Scalar XA11 = 6.0 * pi * r_a * (2.0*beta*beta/((1.0+beta)*(1.0+beta)*(1.0+beta))*laminv+
		            	(beta+7.0*beta*beta+beta*beta*beta)/(5.0*(1.0+beta)*
		        	(1.0+beta)*(1.0+beta))*log_laminv);
	            vec3<Scalar> f_vel = XA11 * dot_vel * n_v + YA11 * (delta_U - n_v*dot_vel);  
                

	            vec3<Scalar> f_rot = -(r_a+r_b)/2*YA11*cross_angvel_sum + 
	        			(1-(r_b*r_a+4*r_b*r_b)/(4*r_a*r_a+r_a*r_b))*YB11/2*cross_angvel_diff;  
                vec3<Scalar> f = mu * (f_vel + f_rot);
                vec3<Scalar> t_i = mu * (YB11*cross_vel+(r_a+r_b)*YB11/2*(sum_avel-dot_angvel_sum* n_v)+
		       	      (1-4*r_a/r_b)/2*YC12*(diff_avel-dot_angvel_diff* n_v));
                vec3<Scalar> t_j = mu * ((r_a*r_b+4*r_b*r_b)/(4*r_a*r_a+r_a*r_b)*YB11*cross_vel+
		        	(r_b*(r_a+r_b)*(r_a+4*r_b))/(2*r_a*r_b+8*r_a*r_a)*YB11*(sum_avel-dot_angvel_sum* n_v)
		    	-(1-4*r_b/r_a)*YC12*(diff_avel-dot_angvel_diff* n_v));
                force = vec_to_scalar3(f);
                torque_i = vec_to_scalar3(t_i);
                torque_j = vec_to_scalar3(t_j);
                }
        pair_eng = 0;
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
        return "rotational coupling";
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
    vec3<Scalar> ang_mom;   /// Sum of angular momentum for ith and jth particle
    vec3<Scalar> ang_i;   /// angular momentum for ith particle
    vec3<Scalar> ang_j;   /// angular momentum for jth particle
    vec3<Scalar> velocity;   /// difference of velocity for ith and jth particle
    Scalar diameter_i; /// diameter of particle i 
    Scalar diameter_j; /// diameter of particle j 
    Scalar mu;
    Scalar massi;
    Scalar massj;
    bool take_momentum;
    bool take_velocity;
    // const param_type &params;   //!< The pair potential parameters
    };

    } // end namespace md
    } // end namespace hoomd

#endif // __PAIR_EVALUATOR_VELOCITY_LUBRICATION_COUPLING_H__
