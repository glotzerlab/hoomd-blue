// Copyright (c) 2009-2022 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

/*
  This class aplies the DirectionalEnvelope to the PairEvaluator, turning the isotropic pair potential into an anisotropic potential.
*/

#ifndef __PAIR_MODULATOR_H__
#define __PAIR_MODULATOR_H__

#ifndef __HIPCC__
#include <string>
#include <pybind11/pybind11.h>
#endif

#include "hoomd/HOOMDMath.h"
#include "hoomd/ManagedArray.h"

// need to declare these class methods with __device__ qualifiers when building in nvcc
//! HOSTDEVICE is __host__ __device__ when included in nvcc and blank when included into the host
//! compiler
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

//! Class to modulate an isotropic pair potential by an envelope to create an anisotropic pair potential.
template <typename PairEvaluator, typename DirectionalEnvelope>
class PairModulator
{
public:

    struct param_type
    {
        param_type()
            {
            }

        param_type(typename PairEvaluator::param_type _pairP, typename DirectionalEnvelope::param_type _envelP)
            {
                pairP = _pairP;
                envelP = _envelP;
            }
#ifndef __HIPCC__
        param_type(pybind11::dict params, bool managed)
            {
                pairP = typename PairEvaluator::param_type(params["pair_params"], managed);
                envelP = typename DirectionalEnvelope::param_type(params["envelope_params"]);
            }

        pybind11::dict toPython()
            {
                pybind11::dict v;

                v["pair_params"] = pairP.asDict();
                v["envelope_params"] = envelP.toPython();

                return v;
            }
#endif
        DEVICE void load_shared(char*& ptr, unsigned int& available_bytes)
            {
                pairP.load_shared(ptr, available_bytes);
            }

        HOSTDEVICE void allocate_shared(char*& ptr, unsigned int& available_bytes) const
            {
                pairP.allocate_shared(ptr, available_bytes);
            }

#ifdef ENABLE_HIP
        //! Attach managed memory to CUDA stream
        void set_memory_hint() const
            {
                pairP.set_memory_hint();
            }
#endif
        typename PairEvaluator::param_type pairP;
        typename DirectionalEnvelope::param_type envelP;
    };

// Nullary structure required by AnisoPotentialPair.
    struct shape_type
    {
        //! Load dynamic data members into shared memory and increase pointer
        /*! \param ptr Pointer to load data to (will be incremented)
          \param available_bytes Size of remaining shared memory allocation
        */
        DEVICE void load_shared(char*& ptr, unsigned int& available_bytes) {
            envelope.load_shared(ptr, available_bytes);
}

        HOSTDEVICE void allocate_shared(char*& ptr, unsigned int& available_bytes) const {
            envelope.allocate_shared(ptr, available_bytes);
}

        HOSTDEVICE shape_type() { }

#ifndef __HIPCC__

        shape_type(pybind11::object shape_params, bool managed)
            : envelope(pybind11::len(shape_params), managed)
            {
                pybind11::list shape_param = shape_params;
                for (size_t i = 0; i < pybind11::len(shape_param); i++)
                    {

                        envelope[int(i)] = typename DirectionalEnvelope::shape_type(shape_param[i]);
                    }
            }

        pybind11::object toPython()
            {
                pybind11::list envelope_py;
                for (size_t i = 0; i < envelope.size();  i++)
                    {
                        envelope_py.append(envelope[int(i)].toPython());
                    }
                return envelope_py;
            }
#endif

#ifdef ENABLE_HIP
        //! Attach managed memory to CUDA stream
        void set_memory_hint() const {
            envelope.set_memory_hint();
}
#endif
        ManagedArray<typename DirectionalEnvelope::shape_type> envelope;
    };

    //! Constructs the pair potential evaluator
    /*!
      \param _dr Displacement vector pointing from particle rj to ri
      \param _rcutsq Squared distance at which the potential is set to 0
      \param _quat_i Quaternion of the ith particle
      \param _quat_j Quaternion of the jth particle
      \param _params Per type pair parameters of the potential
    */
    DEVICE PairModulator( const Scalar3& _dr,
                          const Scalar4& _quat_i,
                          const Scalar4& _quat_j,
                          const Scalar _rcutsq,
                          const param_type& _params)
        : dr(_dr),
          rsq(dot(_dr, _dr)),
          rcutsq(_rcutsq),
          quat_i(_quat_i),
          quat_j(_quat_j),
          params(_params)
        { }

    //! Whether pair potential requires charges
    DEVICE static bool needsCharge()
        {
            return (PairEvaluator::needsCharge() || DirectionalEnvelope::needsCharge());
        }

    //! Accept the optional charge values
    /*!
      \param qi Charge of particle i
      \param qj Charge of particle j
    */
    DEVICE void setCharge(Scalar qi, Scalar qj)
        {
            // store qi and qj for later
            // if (PairEvaluator::needsCharge())
            //     pairEval.setCharge(qi, qj);
            // if (DirectionalEnvelope::needsCharge())
            //     envelEval.setCharge(qi, qj);
        }

    //! Whether the pair potential uses shape.
    DEVICE static bool needsShape()
        {
            return true;
        }

    //! Accept the optional tags
    /*!
      \param tag_i Tag of particle i
      \param tag_j Tag of particle j
    */
    DEVICE void setShape(const shape_type* _shapei, const shape_type* _shapej) {
        shape_i = _shapei;
        shape_j = _shapej;
    }

    //! Whether the pair potential needs particle tags.
    HOSTDEVICE static bool needsTags()
        {
        return (PairEvaluator::needsTags() || DirectionalEnvelope::needsTags());
        }

    //! No modulated potential needs tags
    HOSTDEVICE void setTags(unsigned int tagi, unsigned int tagj)
        {
        }

    HOSTDEVICE static bool constexpr implementsEnergyShift()
        {
            return true;
        }

    //! Evaluate the force and energy
    /*!
      \param force Output parameter to write the computed force.
      \param pair_eng Output parameter to write the computed pair energy.
      \param energy_shift If true, the potential must be shifted so that V(r) is continuous at the cutoff.
      \param torque_i The torque exerted on the ith particle.
      \param torque_j The torque exerted on the jth particle.
      \note There is no need to check if rsq < rcutsq in this method. Cutoff tests are performed
      in PotentialPair.
      \return True if force and energy are evaluated, false if not because r>rcut.
    */
    DEVICE bool evaluate(Scalar3& force,
                         Scalar& pair_eng,
                         bool energy_shift,
                         Scalar3& torque_i,
                         Scalar3& torque_j)
        {

            force = make_scalar3(0,0,0);
            pair_eng = Scalar(0);
            torque_i = make_scalar3(0,0,0);
            torque_j = make_scalar3(0,0,0);

            for (unsigned int patchi = 0; patchi < shape_i->envelope.size(); patchi++)
                {
                    for (unsigned int patchj = 0; patchj < shape_j->envelope.size(); patchj++)
                        {
                            Scalar3 this_force = make_scalar3(0,0,0);
                            Scalar3 grad_mods = make_scalar3(0,0,0);
                            Scalar this_pair_eng = Scalar(0);
                            Scalar3 this_torque_i = make_scalar3(0,0,0);
                            Scalar3 this_torque_j = make_scalar3(0,0,0);
                            Scalar force_divr(Scalar(0));
                            Scalar envelope(Scalar(0));

                            PairEvaluator pair_eval(rsq, rcutsq, params.pairP);

                            // compute pair potential
                            if (!pair_eval.evalForceAndEnergy(force_divr, this_pair_eng, energy_shift))
                                {
                                    return false;
                                }

                            DirectionalEnvelope envel_eval(dr, quat_i, quat_j, rcutsq, params.envelP, shape_i->envelope[patchi], shape_j->envelope[patchj]);

                            // compute envelope
                            // this_torque_i and this_torque_j get populated with the
                            //   torque envelopes and are missing the factor of pair energy
                            envel_eval.evaluate(grad_mods, envelope, this_torque_i, this_torque_j);

                            // modulate forces

                            // second term has the negative sign for force calculation in force_divr

                            // term1 = self.iso.force(magdr) * normalize(dr) * self.patch.fi(dr, self.ni_world) * self.patch.fj(dr, self.nj_world)

                            //                        [term2         ]   [term1                 ]
                            this_force.x = this_pair_eng*grad_mods.x + dr.x*force_divr*envelope;
                            this_force.y = this_pair_eng*grad_mods.y + dr.y*force_divr*envelope;
                            this_force.z = this_pair_eng*grad_mods.z + dr.z*force_divr*envelope;

                            // modulate torques
                            // U (pair_eng) is isotropic so it can be taken out of the derivatives that deal with orientation.
                            this_torque_i.x *= this_pair_eng; // here, the "anisotropic" part can't have distance dependence
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
    //! Get the name of this potential
    /*!
      \returns The potential name.
    */
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
    const Scalar4& quat_i;
    const Scalar4& quat_j;
    const param_type& params;
    const shape_type* shape_i;
    const shape_type* shape_j;

    // PairEvaluator pairEval;           //!< An isotropic pair evaluator
    // DirectionalEnvelope envelEval;    //!< A directional envelope evaluator

};

    } // end namespace md
    } // end namespace hoomd

#endif // __PAIR_MODULATOR_H__
