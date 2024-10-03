// Copyright (c) 2009-2024 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

#ifndef __PAIR_MODULATOR_H__
#define __PAIR_MODULATOR_H__

#ifndef __HIPCC__
#include <pybind11/pybind11.h>
#include <string>
#endif

#include "hoomd/HOOMDMath.h"
#include "hoomd/ManagedArray.h"

/** need to declare these class methods with __device__ qualifiers when building in nvcc
    HOSTDEVICE is __host__ __device__ when included in nvcc and blank when included into the host
    compiler
*/
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

/** Class to modulate an isotropic pair potential by an envelope to create an anisotropic pair
   potential.

    Applies the DirectionalEnvelope to the PairEvaluator, accounting for multiple
   DirectionalEnvelopes per particle.

    The energy is defined as \f$ U_{ij}(r) = f_i f_j U(r) \f$ where each \f$ f = f(\vec{dr},
   \vec{n}) \f$.

    \tparam PairEvaluator An isotropic pair evaluator.
    \tparam DirectionalEnvelope An envelope like PatchEnvelope.
*/
template<typename PairEvaluator, typename DirectionalEnvelope> class PairModulator
    {
    public:
    struct param_type
        {
        param_type() { }

        // param_type(typename PairEvaluator::param_type _pair_p, typename
        // DirectionalEnvelope::param_type _envel_p)
        //     {
        //         pair_p = _pair_p;
        //         envel_p = _envel_p;
        //     }
#ifndef __HIPCC__
        param_type(pybind11::dict params, bool managed)
            {
            pair_p = typename PairEvaluator::param_type(params["pair_params"], managed);
            envel_p = typename DirectionalEnvelope::param_type(params["envelope_params"]);
            }

        pybind11::dict toPython()
            {
            pybind11::dict v;

            v["pair_params"] = pair_p.asDict();
            v["envelope_params"] = envel_p.toPython();

            return v;
            }
#endif
        DEVICE void load_shared(char*& ptr, unsigned int& available_bytes)
            {
            pair_p.load_shared(ptr, available_bytes);
            }

        HOSTDEVICE void allocate_shared(char*& ptr, unsigned int& available_bytes) const
            {
            pair_p.allocate_shared(ptr, available_bytes);
            }

#ifdef ENABLE_HIP
        //! Attach managed memory to CUDA stream
        void set_memory_hint() const
            {
            pair_p.set_memory_hint();
            }
#endif
        typename PairEvaluator::param_type pair_p;
        typename DirectionalEnvelope::param_type envel_p;
        };

    struct shape_type
        {
        DEVICE void load_shared(char*& ptr, unsigned int& available_bytes)
            {
            envelope.load_shared(ptr, available_bytes);
            }

        HOSTDEVICE void allocate_shared(char*& ptr, unsigned int& available_bytes) const
            {
            envelope.allocate_shared(ptr, available_bytes);
            }

        HOSTDEVICE shape_type() { }

#ifndef __HIPCC__

        shape_type(pybind11::list shape_param, bool managed)
            : envelope(static_cast<unsigned int>(pybind11::len(shape_param)), managed)
            {
            for (size_t i = 0; i < pybind11::len(shape_param); i++)
                {
                envelope[int(i)] = typename DirectionalEnvelope::shape_type(shape_param[i]);
                }
            }

        pybind11::object toPython()
            {
            pybind11::list envelope_py;
            for (size_t i = 0; i < envelope.size(); i++)
                {
                envelope_py.append(envelope[int(i)].toPython());
                }
            return envelope_py;
            }
#endif

#ifdef ENABLE_HIP
        void set_memory_hint() const
            {
            envelope.set_memory_hint();
            }
#endif
        ManagedArray<typename DirectionalEnvelope::shape_type> envelope;
        };

    //! Constructs the pair potential evaluator
    /*!
      \param _dr Displacement vector pointing from particle j to particle i
      \param _rcutsq Squared distance at which the potential is set to 0
      \param _q_i Quaternion of the i^{th} particle
      \param _q_j Quaternion of the j^{th} particle
      \param _params Per type pair parameters of the potential
    */
    DEVICE PairModulator(const Scalar3& _dr,
                         const Scalar4& _q_i,
                         const Scalar4& _q_j,
                         const Scalar _rcutsq,
                         const param_type& _params)
        : dr(_dr), rsq(dot(_dr, _dr)), rcutsq(_rcutsq), q_i(_q_i), q_j(_q_j), params(_params)
        {
        }

    DEVICE static bool needsCharge()
        {
        return (PairEvaluator::needsCharge() || DirectionalEnvelope::needsCharge());
        }

    DEVICE void setCharge(Scalar qi, Scalar qj)
        {
        m_charge_i = qi;
        m_charge_j = qj;
        }

    DEVICE static bool needsShape()
        {
        return true;
        }

    DEVICE void setShape(const shape_type* _shapei, const shape_type* _shapej)
        {
        shape_i = _shapei;
        shape_j = _shapej;
        }

    HOSTDEVICE static bool needsTags()
        {
        return false;
        }

    HOSTDEVICE void setTags(unsigned int tagi, unsigned int tagj) { }

    HOSTDEVICE static bool constexpr implementsEnergyShift()
        {
        return true;
        }

    //! Evaluate the force and energy
    /*!
      \param force Output parameter to write the computed force.
      \param pair_eng Output parameter to write the computed pair energy.
      \param energy_shift If true, the potential must be shifted so that V(r) is continuous at the
      cutoff. \param torque_i The torque exerted on the i^{th particle. \param torque_j The torque
      exerted on the j^{th} particle. \note There is no need to check if rsq < rcutsq in this
      method. Cutoff tests are performed in PotentialPair. \return True if force and energy are
      evaluated, false if not because r>rcut.
    */
    DEVICE bool evaluate(Scalar3& force,
                         Scalar& pair_eng,
                         bool energy_shift,
                         Scalar3& torque_i,
                         Scalar3& torque_j)
        {
        force = make_scalar3(0, 0, 0);
        pair_eng = Scalar(0);
        torque_i = make_scalar3(0, 0, 0);
        torque_j = make_scalar3(0, 0, 0);

        for (unsigned int envelope_i = 0; envelope_i < shape_i->envelope.size(); envelope_i++)
            {
            for (unsigned int envelope_j = 0; envelope_j < shape_j->envelope.size(); envelope_j++)
                {
                Scalar3 this_force = make_scalar3(0, 0, 0);
                Scalar3 grad_envelopes = make_scalar3(0, 0, 0);
                Scalar this_pair_eng = Scalar(0);
                Scalar3 this_torque_i = make_scalar3(0, 0, 0);
                Scalar3 this_torque_j = make_scalar3(0, 0, 0);
                Scalar force_divr(Scalar(0));
                Scalar envelope(Scalar(0));

                PairEvaluator pair_eval(rsq, rcutsq, params.pair_p);
                pair_eval.setCharge(m_charge_i, m_charge_j);

                // compute pair potential
                if (!pair_eval.evalForceAndEnergy(force_divr, this_pair_eng, energy_shift))
                    {
                    return false;
                    }

                DirectionalEnvelope envel_eval(dr,
                                               q_i,
                                               q_j,
                                               rcutsq,
                                               params.envel_p,
                                               shape_i->envelope[envelope_i],
                                               shape_j->envelope[envelope_j]);
                envel_eval.setCharge(m_charge_i, m_charge_j);

                // compute envelope
                // this_torque_i and this_torque_j get populated with the
                //   torque envelopes and are missing the factor of pair energy
                envel_eval.evaluate(grad_envelopes, envelope, this_torque_i, this_torque_j);

                /// modulate forces
                this_force.x = this_pair_eng * grad_envelopes.x + dr.x * force_divr * envelope;
                this_force.y = this_pair_eng * grad_envelopes.y + dr.y * force_divr * envelope;
                this_force.z = this_pair_eng * grad_envelopes.z + dr.z * force_divr * envelope;

                // modulate torques
                // U (pair_eng) is isotropic so it can be taken out of the derivatives that deal
                // with orientation.
                this_torque_i.x *= this_pair_eng;
                this_torque_i.y *= this_pair_eng;
                this_torque_i.z *= this_pair_eng;

                this_torque_j.x *= this_pair_eng;
                this_torque_j.y *= this_pair_eng;
                this_torque_j.z *= this_pair_eng;

                // modulate pair energy
                this_pair_eng *= envelope;

                force += this_force;
                pair_eng += this_pair_eng;
                torque_i += this_torque_i;
                torque_j += this_torque_j;
                }
            }

        return true;
        }

#ifndef __HIPCC__
    static std::string getName()
        {
        return PairEvaluator::getName() + "_" + DirectionalEnvelope::getName();
        }
    static std::string getShapeParamName()
        {
        return "Patches";
        }
    std::string getShapeSpec() const
        {
        throw std::runtime_error("Shape definition not supported for this pair potential.");
        }
#endif

    protected:
    Scalar3 dr;
    Scalar rsq;
    Scalar rcutsq;
    const Scalar4& q_i;
    const Scalar4& q_j;
    const param_type& params;
    const shape_type* shape_i;
    const shape_type* shape_j;

    Scalar m_charge_i, m_charge_j;
    };

    } // end namespace md
    } // end namespace hoomd

#endif // __PAIR_MODULATOR_H__
