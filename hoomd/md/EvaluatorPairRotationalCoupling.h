// Copyright (c) 2009-2022 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

#ifndef __PAIR_EVALUATOR_ROTATIONAL_COUPLING_H__
#define __PAIR_EVALUATOR_ROTATIONAL_COUPLING_H__

#ifndef __HIPCC__
#include <string>
#endif

#ifdef ENABLE_HIP
#include <hip/hip_runtime.h>
#endif
#include "hoomd/HOOMDMath.h"
#include "hoomd/VectorMath.h"
#include <iostream>
/*! \file EvaluatorPairDipole.h
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
class EvaluatorPairRotationalCoupling
    {
    public:
    struct param_type
        {
        Scalar kappa; //! force coefficient.
	bool take_momentum;

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

        HOSTDEVICE param_type() : kappa(0), take_momentum(true) { }

#ifndef __HIPCC__

        param_type(pybind11::dict v, bool managed)
            {
            kappa = v["kappa"].cast<Scalar>();
            take_momentum = v["take_momentum"].cast<bool>();
            }

        pybind11::object toPython()
            {
            pybind11::dict v;
            v["kappa"] = kappa;
            v["take_momentum"] = take_momentum;
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
        \param _A Electrostatic energy scale
        \param _kappa Inverse screening length
        \param _params Per type pair parameters of this potential
    */
    HOSTDEVICE EvaluatorPairRotationalCoupling(Scalar3& _dr,
                                               Scalar4& _quat_i,
                                               Scalar4& _quat_j,
                                               Scalar _rcutsq,
                                               const param_type& _params)
        : dr(_dr), rcutsq(_rcutsq), quat_i(_quat_i), quat_j(_quat_j), ang_mom {0, 0, 0}, am {true},
          kappa(_params.kappa), take_momentum(_params.take_momentum)
        {
        }

    //! uses diameter
    HOSTDEVICE static bool needsDiameter()
        {
        return false;
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

    /// Whether the potential implements the energy_shift parameter
    HOSTDEVICE static bool constexpr implementsEnergyShift()
        {
        return false;
        }

    //! Accept the optional diameter values
    /*! \param di Diameter of particle i
        \param dj Diameter of particle j
    */
    HOSTDEVICE void setDiameter(Scalar di, Scalar dj) { }

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
    HOSTDEVICE void setAngularMomentum(vec3<Scalar> ai)
        {
	am = true;
	if(take_momentum)
		{
		ang_mom = ai;

		if (ang_mom.x * ang_mom.x + ang_mom.y * ang_mom.y + ang_mom.z * ang_mom.z < 1e-5)
		    am = false;
		}
	else 
		ang_mom = vec3<Scalar>(0,0,1);
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
        if (am)
            {
            vec3<Scalar> rvec(dr);
            Scalar rsq = dot(rvec, rvec);

            if (rsq > rcutsq)
                return false;

            Scalar rinv = fast::rsqrt(rsq);
            Scalar rcutinv = fast::rsqrt(rcutsq);

            Scalar d = rinv - rcutinv;

            vec3<Scalar> f = kappa * d * d * rinv * cross(ang_mom, rvec);

            force = vec_to_scalar3(f);
            }
        else
            {
            force = make_scalar3(0, 0, 0);
            }
        torque_i = make_scalar3(0, 0, 0);
        torque_j = make_scalar3(0, 0, 0);

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
    bool am;
    Scalar kappa;
    bool take_momentum;
    // const param_type &params;   //!< The pair potential parameters
    };

    } // end namespace md
    } // end namespace hoomd

#endif // __PAIR_EVALUATOR_ROTATIONAL_COUPLING_H__
