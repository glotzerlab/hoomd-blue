// Copyright (c) 2009-2022 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.



// Used to be called EvaluatorPairIsoModulated


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
                // temporary width is 1
                num_patches = 1;
                pairP = ManagedArray<typename PairEvaluator::param_type>(1);
                envelP = ManagedArray<typename DirectionalEnvelope::param_type>(1);
                pairP[0] = _pairP;
                envelP[0] = _envelP;
            }

        param_type(pybind11::dict params, bool managed)
            {
                // temporary width is 1
                num_patches = 1;
                pairP = ManagedArray<typename PairEvaluator::param_type>(1, managed);
                envelP = ManagedArray<typename DirectionalEnvelope::param_type>(1, managed);
                pairP[0] = typename PairEvaluator::param_type(params["pair_params"], managed);
                envelP[0] = typename DirectionalEnvelope::param_type(params["envelope_params"]);
            }

        pybind11::dict asDict()
            {
                pybind11::dict v;

                v["pair_params"] = pairP[0].asDict();
                v["envelope_params"] = envelP[0].asDict();

                return v;
            }
        DEVICE void load_shared(char*& ptr, unsigned int& available_bytes)
            {
                // TODO: might have problem with Table potential?
                pairP.load_shared(ptr, available_bytes);
                envelP.load_shared(ptr, available_bytes);
                for (unsigned int i = 0; i < pairP.size(); i++)
                    {
                        pairP[i].load_shared(ptr, available_bytes);
                        // no envelopes with managed arrays in them
                    }
            }

        HOSTDEVICE void allocate_shared(char*& ptr, unsigned int& available_bytes) const
            {
                // TODO: might have problem with Table potential?
                pairP.load_shared(ptr, available_bytes);
                envelP.load_shared(ptr, available_bytes);
                for (unsigned int i = 0; i < pairP.size(); i++)
                    {
                        pairP[i].load_shared(ptr, available_bytes);
                        // no envelopes with managed arrays in them
                    }
            }

#ifdef ENABLE_HIP
        //! Attach managed memory to CUDA stream
        void set_memory_hint() const
            {
                // TODO: might have problem with Table potential?
                pairP.set_memory_hint();
                envelP.set_memory_hint();
                for (unsigned int i = 0; i < pairP.size(); i++)
                    {
                        pairP[i].set_memory_hint();
                        // no envelopes with managed arrays in them
                    }
            }
#endif
        ManagedArray<typename PairEvaluator::param_type> pairP;
        ManagedArray<typename DirectionalEnvelope::param_type> envelP;
        unsigned int num_patches;
    };

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

        pybind11::dict asDict()
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
    /*!
      \param _dr Displacement vector pointing from particle rj to ri
      \param _rcutsq Squared distance at which the potential is set to 0
      \param _quat_eye Quaternion of the ith particle
      \param _quat_jay Quaternion of the jth particle
      \param _params Per type pair parameters of the potential
    */
    DEVICE PairModulator( const Scalar3& _dr,
                          const Scalar4& _quat_eye,
                          const Scalar4& _quat_jay,
                          const Scalar _rcutsq,
                          const param_type& _params)
        : dr(_dr),
          rsq(_dr.x*_dr.x + _dr.y*_dr.y + _dr.z*_dr.z),
          rcutsq(_rcutsq),
          quat_i(_quat_eye),
          quat_j(_quat_jay),
          params(_params)
        { }

    //! If diameter is used
    DEVICE static bool needsDiameter()
        {
            return (PairEvaluator::needsDiameter() || DirectionalEnvelope::needsDiameter());
        }

    //! Accept the optional diameter values
    /*!
      \param di Diameter of particle i
      \param dj Diameter of particle j
    */
    DEVICE void setDiameter(Scalar di, Scalar dj)
        {
        }

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
            return false;
        }

    //! Accept the optional tags
    /*!
      \param tag_i Tag of particle i
      \param tag_j Tag of particle j
    */
    DEVICE void setShape(const shape_type* shapei, const shape_type* shapej) { }

    //! Whether the pair potential needs particle tags.
    HOSTDEVICE static bool needsTags()
        {
            return false;
        }

    //! No modulated potential needs tags
    HOSTDEVICE void setTags(unsigned int tagi, unsigned int tagj)
        {
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

            //for loop over patchi and patchj
            for (unsigned int patchi = 0; patchi < params.num_patches; patchi++)
                {
                    for (unsigned int patchj = 0; patchj < params.num_patches; patchj++)
                        {
                            Scalar3 this_force = make_scalar3(0,0,0);
                            Scalar this_pair_eng = Scalar(0);
                            Scalar3 this_torque_i = make_scalar3(0,0,0);
                            Scalar3 this_torque_j = make_scalar3(0,0,0);

                            Index2D patch_indexer(params.num_patches, params.num_patches); // num_patches comes from width of square matrix of passed parameters
                            PairEvaluator pair_eval(rsq, rcutsq, params.pairP[patch_indexer(patchi, patchj)]);
                            // compute pair potential
                            Scalar force_divr(Scalar(0));
                            if (!pair_eval.evalForceAndEnergy(force_divr, this_pair_eng, energy_shift))
                                {
                                    return false;
                                }

                            DirectionalEnvelope envel_eval(dr, quat_i, quat_j, rcutsq, params.envelP[patch_indexer(patchi, patchj)]);
                            // compute envelope
                            Scalar envelope(Scalar(0));
                            // here, this_torque_i and this_torque_j get populated with the
                            // torque envelopes and are missing the factor of pair energy
                            envel_eval.evaluate(this_force, envelope, this_torque_i, this_torque_j);

                            // modulate forces
                            // TODO check this math. yes.
            
                            // second term has the negative sign for force calculation in force_divr

                            // term1 = self.iso.force(magdr) * normalize(dr) * self.patch.fi(dr, self.ni_world) * self.patch.fj(dr, self.nj_world)


            
                            //        [term2         ]   [term1                 ]
                            // TODO call this grad of modulators
                            this_force.x = this_pair_eng*this_force.x + dr.x*force_divr*envelope;
                            this_force.y = this_pair_eng*this_force.y + dr.y*force_divr*envelope;
                            this_force.z = this_pair_eng*this_force.z + dr.z*force_divr*envelope;



                            // modulate torques
                            // TODO check this math. Finished checking Jan 4 2023
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

    std::string getShapeSpec() const
        {
            // TODO this is just copied in:
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

    // PairEvaluator pairEval;           //!< An isotropic pair evaluator
    // DirectionalEnvelope envelEval;    //!< A directional envelope evaluator

};

    } // end namespace md
    } // end namespace hoomd

#endif // __PAIR_MODULATOR_H__
