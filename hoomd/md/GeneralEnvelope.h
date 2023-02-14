// Copyright (c) 2009-2022 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

#ifndef __GENERAL_ENVELOPE_H__
#define __GENERAL_ENVELOPE_H__

#ifndef __HIPCC__
#include <string>
#endif
#include <string.h>
#include "hoomd/HOOMDMath.h"
#include "hoomd/VectorMath.h"

// need to declare these class methods with __device__ qualifiers when building in nvcc
// DEVICE is __host__ __device__ when included in nvcc and blank when included into the host
// compiler
#ifdef __HIPCC__
#define DEVICE __device__
#define HOSTDEVICE __host__ __device__
#else
#define DEVICE
#define HOSTDEVICE
#endif

namespace hoomd
    {
namespace md
    {

/*
  The GeneralEnvelope creates the pair potential modulator.
*/

        std::string vecString(vec3<Scalar> a) {
            return std::to_string(a.x) + ", " + std::to_string(a.y) + ", " + std::to_string(a.z) + '\n';
        }


class GeneralEnvelope
{
public:
    struct param_type
    {
        param_type()
            {
            }

        param_type(pybind11::dict params)
            {
                cosalpha = fast::cos(params["alpha"].cast<Scalar>());
                omega = params["omega"].cast<Scalar>();
                auto ni_ = (pybind11::tuple)params["ni"];
                auto nj_ = (pybind11::tuple)params["nj"];

                ni = vec3<Scalar>(ni_[0].cast<Scalar>(), ni_[1].cast<Scalar>(), ni_[2].cast<Scalar>());
                nj = vec3<Scalar>(nj_[0].cast<Scalar>(), nj_[1].cast<Scalar>(), nj_[2].cast<Scalar>());

                // normalize
                ni = ni * fast::rsqrt(dot(ni, ni));
                nj = nj * fast::rsqrt(dot(nj, nj));
            }

        pybind11::dict asDict()
            {
                pybind11::dict v;

                v["alpha"] = fast::acos(cosalpha);
                v["omega"] = omega;

                vec3<Scalar> ex(1,0,0);
                //vec3<Scalar> ni = rotate(qpi, ex);
                //vec3<Scalar> nj = rotate(qpj, ex);

                v["ni"] = pybind11::make_tuple(ni.x, ni.y, ni.z);
                v["nj"] = pybind11::make_tuple(nj.x, nj.y, nj.z);

                return v;
            }

        vec3<Scalar> ni;
        vec3<Scalar> nj;
        Scalar cosalpha;
        Scalar omega;
    }__attribute__((aligned(16)));

    DEVICE GeneralEnvelope( // TODO: Change name to PatchModulator. It is not general. It assumes a single off-center patch
        const Scalar3& _dr,
        const Scalar4& _quat_i, // Note in hoomd, the quaternion is how to get from the particle orientation to align to the world orientation. so World = qi Local qi-1
        const Scalar4& _quat_j,
        const Scalar _rcutsq,
        const param_type& _params)
        : dr(_dr), qi(_quat_i), qj(_quat_j), params(_params)
        {
            // compute current janus direction vectors
            vec3<Scalar> ex(1,0,0); // ex = ni
            vec3<Scalar> ey(0,1,0);
            vec3<Scalar> ez(0,0,1);

            // orientation vectors of particle a
            a1 = rotate(conj(qi), ex);
            a2 = rotate(conj(qi), ey);
            a3 = rotate(conj(qi), ez);

            // orientation vectors of particle b
            b1 = rotate(conj(qj), ex);
            b2 = rotate(conj(qj), ey);
            b3 = rotate(conj(qj), ez);

            // compute distance
            drsq = dot(dr, dr);
            magdr = fast::sqrt(drsq);

            // cos(angle between dr and pointing vector)
            costhetai = -dot(vec3<Scalar>(dr), params.ni) / magdr; // negative because dr = dx = pi - pj
            costhetaj = dot(vec3<Scalar>(dr), params.nj) / magdr;
        }

    //! uses diameter
    DEVICE static bool needsDiameter() { return false; }

    //! Accept the optional diameter values
    /*!
      \param di Diameter of particle i
      \param dj Diameter of particle j
    */
    DEVICE void setDiameter(Scalar di, Scalar dj) { }

    //! whether pair potential requires charges
    DEVICE static bool needsCharge() { return false; }

    //! Accept the optional charge values
    /*!
      \param qi Charge of particle i
      \param qj Charge of particle j
    */
    DEVICE void setCharge(Scalar qi, Scalar qj) { }

    //! Whether the pair potential needs particle tags.
    DEVICE static bool needsTags() { return false; }

    //! Accept the optional tags
    /*! \param tag_i Tag of particle i
        \param tag_j Tag of particle j
    */
    HOSTDEVICE void setTags(unsigned int tagi, unsigned int tagj) { }


    DEVICE inline Scalar Modulatori() // called f(dr, ni) in the derivation
        {
            return Scalar(1.0) / ( Scalar(1.0) + fast::exp(-params.omega*(costhetai-params.cosalpha)) );
        }

    DEVICE inline Scalar Modulatorj() // called f(dr, nj) in the derivation
        {
            return Scalar(1.0) / ( Scalar(1.0) + fast::exp(-params.omega*(costhetaj-params.cosalpha)) );
        }

    DEVICE Scalar ModulatorPrimei()
        {
            Scalar fact = Modulatori();
            // the -1 comes from doing the derivative with respect to ni
            return Scalar(-1) * params.omega * fast::exp(-params.omega*(costhetai-params.cosalpha)) * fact * fact;
        }

    DEVICE Scalar ModulatorPrimej()
        {
            Scalar fact = Modulatorj();
            return params.omega * fast::exp(-params.omega*(costhetaj-params.cosalpha)) * fact * fact;
        }


    //! Evaluate the force and energy
    /*
      // TODO update this
      \Param force Output parameter to write the computed force.
      \param envelope Output parameter to write the amount of modulation of the isotropic part
      \param torque_div_energy_i The torque exterted on the i^th particle, divided by energy of interaction.
      \param torque_div_energy_j The torque exterted on the j^th particle, divided by energy of interaction.
      \note There is no need to check if rsq < rcutsq in this method. Cutoff tests are performed in PotentialPair from the PairModulator.
      \return Always true
    */
    DEVICE bool evaluate(Scalar3& force,
                         Scalar& envelope,
                         Scalar3& torque_div_energy_i, //torque_modulator
                         Scalar3& torque_div_energy_j) //torque_modulator
        {
            // common calculations
            Scalar modi = Modulatori();
            Scalar modj = Modulatorj();
            Scalar modPi = ModulatorPrimei();
            Scalar modPj = ModulatorPrimej();

            // the overall modulation
            envelope = modi*modj;

            // intermediate calculations
            Scalar iPj = modPi*modj/magdr; // TODO: make variable name more descriptive and check if these are correct. Jan 4: They are correct
            Scalar jPi = modPj*modi/magdr;
            // TODO Jan 4 2023: I don't think this division by s.magdr should be here mathematically, but probably for efficiency


            // NEW way with Philipp Feb 9

            torque_div_energy_i =
                vec_to_scalar3( params.ni.x * cross( vec3<Scalar>(a1), dr)) +
                vec_to_scalar3( params.ni.y * cross( vec3<Scalar>(a2), dr)) +
                vec_to_scalar3( params.ni.z * cross( vec3<Scalar>(a3), dr));

            torque_div_energy_i *= Scalar(-1) * Modulatorj() * ModulatorPrimei() / magdr; // this last bit is iPj

            torque_div_energy_j =
                vec_to_scalar3( params.nj.x * cross( vec3<Scalar>(b1), dr)) +
                vec_to_scalar3( params.nj.y * cross( vec3<Scalar>(b2), dr)) +
                vec_to_scalar3( params.nj.z * cross( vec3<Scalar>(b3), dr));

            torque_div_energy_j *= Scalar(-1) * Modulatori() * ModulatorPrimej() / magdr;

            force.x = -(iPj*(-a1.x - costhetai*dr.x/magdr) // iPj includes a factor of 1/magdr
                        + jPi*(b1.x - costhetaj*dr.x/magdr));
            force.y = -(iPj*(-a1.y - costhetai*dr.y/magdr)
                        + jPi*(b1.y - costhetaj*dr.y/magdr));
            force.z = -(iPj*(-a1.z - costhetai*dr.z/magdr)
                        + jPi*(b1.z - costhetaj*dr.z/magdr));

            return true;
        }

#ifndef _HIPCC_
    //! Get the name of the potential
    static std::string getName()
        {
            return std::string("generalenvelope");
        }
#endif

private:
    vec3<Scalar> dr;
    quat<Scalar> qi;
    quat<Scalar> qj;

    const param_type& params;

    vec3<Scalar> a1, a2, a3;
    vec3<Scalar> b1, b2, b3;
    Scalar drsq;
    Scalar magdr;

    vec3<Scalar> ei, ej;
    Scalar costhetai;
    Scalar costhetaj;
};

    } // end namespace md
    } // end namespace hoomd

#endif // __GENERAL_ENVELOPE_H__
