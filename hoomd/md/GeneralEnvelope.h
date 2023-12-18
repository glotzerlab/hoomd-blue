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

        param_type(pybind11::dict params) // param dict can take any python type
            {
                cosalpha = fast::cos(params["alpha"].cast<Scalar>());
                omega = params["omega"].cast<Scalar>();
            }

        pybind11::dict toPython()
            {
                pybind11::dict v;
                v["alpha"] = fast::acos(cosalpha);
                v["omega"] = omega;
                return v;
            }

        Scalar cosalpha;
        Scalar omega;
    }__attribute__((aligned(16)));

    struct shape_type
    {
        HOSTDEVICE shape_type() { }

#ifndef __HIPCC__

        shape_type(pybind11::object patch_location)
            {
            pybind11::tuple n_py = patch_location;
            if (len(n_py) != 3)
                throw std::runtime_error("Each patch position must have 3 elements");
            vec3<Scalar> n = vec3<Scalar>(pybind11::cast<Scalar>(n_py[0]),
                                            pybind11::cast<Scalar>(n_py[1]),
                                            pybind11::cast<Scalar>(n_py[2]));

            // normalize
            n = n * fast::rsqrt(dot(n, n));
            m_n = vec_to_scalar3(n);
            }

        pybind11::object toPython()
            {
            return pybind11::make_tuple(m_n.x, m_n.y, m_n.z);
            }
#endif

        Scalar3 m_n;
    };

    DEVICE GeneralEnvelope( // TODO: Change name to PatchModulator. It is not general. It assumes a single off-center patch
        const Scalar3& _dr,
        const Scalar4& _quat_i, // Note in hoomd, the quaternion is how to get from the particle orientation to align to the world orientation. so World = qi Local qi-1
        const Scalar4& _quat_j,
        const Scalar _rcutsq,
        const param_type& _params,
        const shape_type& shape_i,
        const shape_type& shape_j)
// #ifdef __HIPCC__
        // : dr(_dr), qi(_quat_i), qj(_quat_j), params(_params), n_i(shape_i.m_n), n_j(shape_j.m_n)
// #else
        : dr(_dr), R_i(rotmat3<ShortReal>((quat<ShortReal>)_quat_i)), R_j(rotmat3<ShortReal>((quat<ShortReal>)_quat_j)), params(_params), n_i(shape_i.m_n), n_j(shape_j.m_n)
// #endif
        {
            // compute current janus direction vectors

            // rotate from particle to world frame
            vec3<ShortReal> ex(1,0,0);
            vec3<ShortReal> ey(0,1,0);
            vec3<ShortReal> ez(0,0,1);

            //TODO Real --> ShortReal could help

            // TODO try converting to rotation matrices

            // a1, a2, a3 are orientation vectors of particle a in world frame
            // b1, b2, b3 are orientation vectors of particle b in world frame
            // ni_world, patch direction of particle a in world frame
// #ifdef __HIPCC__
            a1 = R_i * ex;
            a2 = R_i * ey;
            a3 = R_i * ez;
            ni_world = R_i * (vec3<ShortReal>)n_i;
            b1 = R_j * ex;
            b2 = R_j * ey;
            b3 = R_j * ez;
            nj_world = R_j * (vec3<ShortReal>)n_j; 
// #else
//             a1 = rotate(qi, ex);
//             a2 = rotate(qi, ey);
//             a3 = rotate(qi, ez);
//             ni_world = rotate(qi, n_i);
//             b1 = rotate(qj, ex);
//             b2 = rotate(qj, ey);
//             b3 = rotate(qj, ez);
//             nj_world = rotate(qj, n_j);
// #endif
            
            // std::cout << "ni world: " << vecString(ni_world);

            // compute distance
            drsq = dot(dr, dr);
            magdr = fast::sqrt(drsq);

            rhat = dr/magdr;

            // cos(angle between dr and pointing vector)
            Scalar costhetai = -dot(vec3<Scalar>(rhat), ni_world); // negative because dr = dx = pi - pj
            Scalar costhetaj = dot(vec3<Scalar>(rhat), nj_world);

            exp_negOmega_times_CosThetaI_minus_CosAlpha = fast::exp(-params.omega*(costhetai-params.cosalpha));
            exp_negOmega_times_CosThetaJ_minus_CosAlpha = fast::exp(-params.omega*(costhetaj-params.cosalpha));
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
            return Scalar(1.0) / ( Scalar(1.0) + exp_negOmega_times_CosThetaI_minus_CosAlpha );
        }


    DEVICE inline Scalar fj() // called f(dr, nj) in the derivation
    // fj in python
        {
            return Scalar(1.0) / ( Scalar(1.0) + exp_negOmega_times_CosThetaJ_minus_CosAlpha );
        }


    DEVICE Scalar dfi_du(Scalar fi)
        {
            //      rhat *    (-self.omega        * exp(-self.omega * (self._costhetai(dr, ni_world) - self.cosalpha)) *  self.fi(dr, ni_world)**2)
            // Scalar fact = fi();
            return -params.omega * exp_negOmega_times_CosThetaI_minus_CosAlpha  * fi * fi;
        }

    DEVICE Scalar dfj_du(Scalar fj)
        {
            // Scalar fact = fj();
            //     rhat * (self.omega * exp(-self.omega * (self._costhetaj(dr, nj_world) - self.cosalpha)) * self.fj(dr, nj_world)**2)
            return params.omega * exp_negOmega_times_CosThetaJ_minus_CosAlpha  * fj * fj;
        }

    DEVICE Scalar ModulatorPrimei() // TODO call it derivative with respect to costhetai
        {
            Scalar fact = fi(); // TODO only calculate Modulatori once per instantiation
            // the -1 comes from doing the derivative with respect to ni
            // return Scalar(-1) * params.omega * fast::exp(-params.omega*(costhetai-params.cosalpha)) * fact * fact;
            return params.omega * exp_negOmega_times_CosThetaI_minus_CosAlpha * fact * fact;
        }

    DEVICE Scalar ModulatorPrimej() // TODO name after derivative
        {
            Scalar fact = fj();
            return params.omega * exp_negOmega_times_CosThetaJ_minus_CosAlpha * fact * fact;
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
            Scalar modi = fi();
            Scalar modj = fj();

            // the overall modulation
            envelope = modi*modj;

            vec3<Scalar> dfi_dni = dfi_du(modi) * rhat; // TODO add -rhat here and take out above (afuyeaad)

            torque_div_energy_i =
                vec_to_scalar3( n_i.x * cross( vec3<Scalar>(a1), dfi_dni)) +
                vec_to_scalar3( n_i.y * cross( vec3<Scalar>(a2), dfi_dni)) +
                vec_to_scalar3( n_i.z * cross( vec3<Scalar>(a3), dfi_dni));

            torque_div_energy_i *= Scalar(-1) * modj;

            vec3<Scalar> dfj_dnj = dfj_du(modj) * rhat; // still positive

            torque_div_energy_j =
                vec_to_scalar3( n_j.x * cross( vec3<Scalar>(b1), dfj_dnj)) +
                vec_to_scalar3( n_j.y * cross( vec3<Scalar>(b2), dfj_dnj)) +
                vec_to_scalar3( n_j.z * cross( vec3<Scalar>(b3), dfj_dnj));

            // std::cout << "j term 3 / modulatorPrimej" << vecString(vec_to_scalar3( shape_j.m_n.z * cross( vec3<Scalar>(b3), dr)));

            torque_div_energy_j *= Scalar(-1) * modi;

            // term2 = self.iso.energy(magdr) * (

            // THIS PART in here:
            //     dfj_duj * duj_dr * self.patch.fi(dr, self.ni_world) + dfi_dui * dui_dr * self.patch.fj(dr, self.nj_world)


            // find df/dr = df/du * du/dr

            // find du/dr using quotient rule, where u = "hi" / "lo" = dot(dr,n) / magdr

            Scalar lo = magdr;
            vec3<Scalar> dlo = rhat;

            // //something wrong: this has to be a scalar
            // Scalar3 dfi_dui = dfi_dni();
            Scalar dfi_dui = -dfi_du(modi); // TODO: remove this negative when I take out the one at afuyeaad

            Scalar hi = -dot(dr, vec3<Scalar>(ni_world));
            vec3<Scalar> dhi = -ni_world;
            // // quotient rule
            vec3<Scalar> dui_dr = (lo*dhi - hi*dlo) / (lo*lo);
            //           dui_dr = removedrhat * ( magdr* -ni_world - -dot(dr, vec3<Scalar>(ni_world)) * rhat ) / (magdr*magdr)

            Scalar dfj_duj = dfj_du(modj);
            hi = dot(vec3<Scalar>(dr), vec3<Scalar>(nj_world));
            dhi = nj_world;
            // // lo and dlo are the same
            vec3<Scalar> duj_dr = (lo*dhi - hi*dlo) / (lo*lo);

            force = -vec_to_scalar3(dfj_duj*duj_dr * fi() + dfi_dui*dui_dr * fj());

            // std::cout << " my force " << vecString(vec3<Scalar>(force));

            // std::cout << "Old force " << vecString(-(iPj * (-ni_world - costhetai*rhat) + jPi * (nj_world - costhetaj *rhat)));

            // is     their      [1/magdr * (-ni_world - costhetai*rhat)] the same as my dui_dr?
            // is     their      [-ni_world/magdr - costhetai*rhat/magdr] the same as my dui_dr?
            //   Expand costhetai to dot product definition
            // is     their      [-ni_world/magdr - -dot(vec3<Scalar>(rhat), ni_world) *rhat/magdr] the same as my dui_dr?
            //   Expand rhat to dr/magdr

            // is     their      [-ni_world/magdr - -dot(dr, ni_world)/magdr * rhat/magdr] the same as my dui_dr?

            //         ... yes but I also needed to add a negative sign to dfi_du()

            // Problems
            // 1. Extra factor of rhat on the whole term in mine, taken out April 25

            //          dui_dr = rhat * ( (-ni_world)/ (magdr)  -   -dot(dr, vec3<Scalar>(ni_world)) * rhat/ (magdr*magdr) )
            //                 ^
            //                / \
            //                 |
            // copy from above |
            //          dui_dr = rhat * ( magdr* (-ni_world)  -   -dot(dr, vec3<Scalar>(ni_world)) * rhat ) / (magdr*magdr)


            // Josh is ok with this:
            // force = -(iPj * (-ni_world - costhetai*rhat) + jPi * (nj_world - costhetaj *rhat))

            // rewrite with my notation:
            // iPj = modPi*modj/magdr
            // iPj = ModulatorPrimei() * Modulatorj() / magdr
            // iPj = -dfi_du() * fj() / magdr

            // jPi = modPj*modi/magdr
            // jPi = ModulatorPrimej() * Modulatori() / magdr
            // jPi = dfj_du() * fi() / magdr
            // force_my_variables = -( -dfi_du() * fj() / magdr * (-ni_world - costhetai*rhat)
            //                        + dfj_du() * fi() / magdr * (nj_world - costhetaj *rhat) )



            // //negative here bc forcedivr has the implicit negative in PairModulator
            // iPj includes a factor of 1/magdr. costhetai includes factor of 1/magdr
            // force.x = -(iPj*(-ni_world.x - costhetai*dr.x/magdr) + jPi*(nj_world.x - costhetaj*dr.x/magdr));
            // force.y = -(iPj*(-ni_world.y - costhetai*dr.y/magdr) + jPi*(nj_world.y - costhetaj*dr.y/magdr));
            // force.z = -(iPj*(-ni_world.z - costhetai*dr.z/magdr) + jPi*(nj_world.z - costhetaj*dr.z/magdr));

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
// #ifdef __HIPCC__
    quat<Scalar> qi;
    quat<Scalar> qj;
// #else
    rotmat3<ShortReal> R_i;
    rotmat3<ShortReal> R_j;
// #endif
    vec3<Scalar> dr;
    vec3<Scalar> ni_world, nj_world;

    const param_type& params;
    vec3<Scalar> n_i, n_j;
    vec3<Scalar> a1, a2, a3;
    vec3<Scalar> b1, b2, b3;
    Scalar drsq;
    Scalar magdr;
    vec3<Scalar> rhat;

    vec3<Scalar> ei, ej;
    Scalar exp_negOmega_times_CosThetaI_minus_CosAlpha;
    Scalar exp_negOmega_times_CosThetaJ_minus_CosAlpha;
};

    } // end namespace md
    } // end namespace hoomd

#endif // __GENERAL_ENVELOPE_H__
