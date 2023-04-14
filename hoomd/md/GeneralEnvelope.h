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

                // std::cout << "nj local " << vecString(nj) << "\n";
                // std::cout << "ni local " << vecString(ni) << "\n";
            }

        pybind11::dict asDict()
            {
                pybind11::dict v;

                v["alpha"] = fast::acos(cosalpha);
                v["omega"] = omega;

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
            
            // rotate from particle to world frame
            vec3<Scalar> ex(1,0,0);
            vec3<Scalar> ey(0,1,0);
            vec3<Scalar> ez(0,0,1);

            // orientation vectors of particle a in world frame
            a1 = rotate(qi, ex);
            a2 = rotate(qi, ey);
            a3 = rotate(qi, ez);
            // patch direction of particle a in world frame
            ni_world = rotate(qi, params.ni);
            // std::cout << "ni world: " << vecString(ni_world);

            // orientation vectors of particle b in world frame
            b1 = rotate(qj, ex);
            b2 = rotate(qj, ey);
            b3 = rotate(qj, ez);

            nj_world = rotate(qj, params.nj);


            // compute distance
            drsq = dot(dr, dr);
            magdr = fast::sqrt(drsq);

            rhat = dr/magdr;

            // cos(angle between dr and pointing vector)            
            costhetai = -dot(vec3<Scalar>(rhat), ni_world); // negative because dr = dx = pi - pj
            costhetaj = dot(vec3<Scalar>(rhat), nj_world);
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


    DEVICE Scalar fi() // called f(dr, ni) in the derivation
    //fi in python
        {
            return Scalar(1.0) / ( Scalar(1.0) + fast::exp(-params.omega*(costhetai-params.cosalpha)) );
        }

    DEVICE inline Scalar Modulatori() { return fi(); }

    DEVICE inline Scalar fj() // called f(dr, nj) in the derivation
    // fj in python
        {
            return Scalar(1.0) / ( Scalar(1.0) + fast::exp(-params.omega*(costhetaj-params.cosalpha)) );
        }

    DEVICE inline Scalar Modulatorj() { return fj(); }


    DEVICE Scalar3 dfi_du()
        {
            //      rhat *    (-self.omega        * exp(-self.omega * (self._costhetai(dr, ni_world) - self.cosalpha)) *  self.fi(dr, ni_world)**2)
            Scalar fact = fi();
            return -params.omega * fast::exp(params.omega * (params.cosalpha - costhetai))  * fact * fact;
        }

    DEVICE Scalar3 dfj_du()
        {
            Scalar fact = fj();
            //     rhat * (self.omega * exp(-self.omega * (self._costhetaj(dr, nj_world) - self.cosalpha)) * self.fj(dr, nj_world)**2)
            return params.omega * fast::exp(params.omega * (params.cosalpha - costhetaj))  * fact * fact;
        }
    
    DEVICE Scalar ModulatorPrimei() // TODO call it derivative with respect to costhetai
        {
            Scalar fact = Modulatori(); // TODO only calculate Modulatori once per instantiation
            // the -1 comes from doing the derivative with respect to ni
            // return Scalar(-1) * params.omega * fast::exp(-params.omega*(costhetai-params.cosalpha)) * fact * fact;
            return params.omega * fast::exp(-params.omega*(costhetai-params.cosalpha)) * fact * fact;
        }

    DEVICE Scalar ModulatorPrimej() // TODO name after derivative
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

            vec3<Scalar> dfi_dni = dfi_du() * rhat; // TODO add -rhat here and take out above
            
            torque_div_energy_i =
                vec_to_scalar3( params.ni.x * cross( vec3<Scalar>(a1), dfi_dni)) +
                vec_to_scalar3( params.ni.y * cross( vec3<Scalar>(a2), dfi_dni)) +
                vec_to_scalar3( params.ni.z * cross( vec3<Scalar>(a3), dfi_dni));

            torque_div_energy_i *= Scalar(-1) * Modulatorj();

            vec3<Scalar> dfj_dnj = dfj_du() * rhat; // still positive
            
            torque_div_energy_j =
                vec_to_scalar3( params.nj.x * cross( vec3<Scalar>(b1), dfj_dnj)) +
                vec_to_scalar3( params.nj.y * cross( vec3<Scalar>(b2), dfj_dnj)) +
                vec_to_scalar3( params.nj.z * cross( vec3<Scalar>(b3), dfj_dnj));
            
            // std::cout << "j term 3 / modulatorPrimej" << vecString(vec_to_scalar3( params.nj.z * cross( vec3<Scalar>(b3), dr)));
            
            torque_div_energy_j *= Scalar(-1) * Modulatori();

            // term2 = self.iso.energy(magdr) * (

            // THIS PART in here:
            //     dfj_duj * duj_dr * self.patch.fi(dr, self.ni_world) + dfi_dui * dui_dr * self.patch.fj(dr, self.nj_world)


            // find df/dr = df/du * du/dr

            // find du/dr using quotient rule, where u = "hi" / "lo" = dot(dr,n) / magdr

            Scalar lo = magdr;
            Scalar3 dlo = vec_to_scalar3(rhat);

            //something wrong: this has to be a scalar
            Scalar3 dfi_dui = dfi_dni();

            Scalar hi = dot(dr, vec3<Scalar>(ni_world));
            Scalar3 dhi = vec_to_scalar3(ni_world);
            // quotient rule
            Scalar3 dui_dr = (lo*dhi - hi*dlo) / (lo*lo);


            Scalar3 dfj_duj = dfj_dnj();
            hi = dot(vec3<Scalar>(dr), vec3<Scalar>(nj_world));
            dhi = vec_to_scalar3(nj_world);
            // lo and dlo are the same
            Scalar3 duj_dr = (lo*dhi - hi*dlo) / (lo*lo);
            
            // force = dfj_duj * duj_dr * fi() + dfi_dui*dui_dr * fj();


            //negative here bc forcedivr has the implicit negative in PairModulator
            force.x = -(iPj*(-ni_world.x - costhetai*dr.x/magdr) // iPj includes a factor of 1/magdr. costhetai includes factor of 1/magdr
                        + jPi*(nj_world.x - costhetaj*dr.x/magdr));
            force.y = -(iPj*(-ni_world.y - costhetai*dr.y/magdr)
                        + jPi*(nj_world.y - costhetaj*dr.y/magdr));
            force.z = -(iPj*(-ni_world.z - costhetai*dr.z/magdr)
                        + jPi*(nj_world.z - costhetaj*dr.z/magdr));

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
    vec3<Scalar> ni_world, nj_world;

    const param_type& params;

    vec3<Scalar> a1, a2, a3;
    vec3<Scalar> b1, b2, b3;
    Scalar drsq;
    Scalar magdr;
    vec3<Scalar> rhat;

    vec3<Scalar> ei, ej;
    Scalar costhetai;
    Scalar costhetaj;
};

    } // end namespace md
    } // end namespace hoomd

#endif // __GENERAL_ENVELOPE_H__
