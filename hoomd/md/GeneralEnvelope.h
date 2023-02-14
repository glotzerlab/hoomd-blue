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

                // Find quaternions to rotate from (1,0,0) to ni and nj
                vec3<Scalar> ex(1,0,0);

                qpi = quat(Scalar(1) + dot(ex, ni), cross(ex, ni));
                qpi = qpi * fast::rsqrt(norm2(qpi));

                qpj = quat(Scalar(1) + dot(ex, nj), cross(ex, nj));
                qpj = qpj * fast::rsqrt(norm2(qpj));

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

        quat<Scalar> qpi;
        quat<Scalar> qpj;
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

            ei = rotate(conj(qi), ex);
            a1 = rotate(conj(qi), ex);
            a2 = rotate(conj(qi), ey);
            a3 = rotate(conj(qi), ez);
            
            // patch points relative to x (a1) direction of particle
            
            // When user provides ni directly, we don't need to rotate a1 by qpi to get ni anymore
            // ni = rotate(params.qpi, a1);

            

            
            // ni = \sum alpha_m * am
            // alpha_1 = ni.x; = dot(ex, ni)
            // alpha_2 = ni.y;
            // alpha_3 = ni.z;
            
            // TODO combine rotations ni = rotate(params.qpi * conj(qi), ex)

            ej = rotate(conj(qj), ex);
            b1 = rotate(conj(qj), ex);
            b2 = rotate(conj(qj), ey);
            b3 = rotate(conj(qj), ez);

            // patch points relative to x (b1) direction of particle
            // nj = rotate(params.qpj, b1);

            // compute distance
            drsq = dot(dr, dr);
            magdr = fast::sqrt(drsq);

            // Possible idea to rotate into particle frame so (1,0,0) is pointing in the direction of the patch
            // rotate dr
            // rotate nj

            // cos(angle between dr and pointing vector)
            // which as first implemented is the same as the angle between the patch and pointing director
            doti = -dot(vec3<Scalar>(dr), ei) / magdr; // negative because dr = dx = pi - pj
            dotj = dot(vec3<Scalar>(dr), ej) / magdr;

          //  std::cout << "ni when calculating costhetai" + vecString(vec3<Scalar>(ni));
            // std::cout << "params.ni when calculating costhetai" + vecString(vec3<Scalar>(params.ni));
            costhetai = -dot(vec3<Scalar>(dr), params.ni) / magdr;
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

    DEVICE Scalar ModulatorPrimei() // D[f[\[Theta], \[Alpha], \[Omega]], Cos[\[Theta]]]
        {
            Scalar fact = Modulatori();
            // weird way of writing out the derivative of f with respect to doti = Cos[theta] =
            // the -1 comes because we are doing the derivative with respect to ni instead of costhetai
            return Scalar(-1) * params.omega * fast::exp(-params.omega*(costhetai-params.cosalpha)) * fact * fact;
        }

    DEVICE Scalar ModulatorPrimej()
        {
            Scalar fact = Modulatorj();
            return params.omega * fast::exp(-params.omega*(costhetaj-params.cosalpha)) * fact * fact;
        }

    vec3<Scalar> new_cross_fun(quat<Scalar> qp)
        {
            vec3<Scalar> new_cross_term;
            new_cross_term = qp.s * cross(dr, qp.v);

            // {qx, qy, qz}*{{dry,drz}.{qy,qz}, {drx,drz}.{qx,qz}, {drx,dry}.{qx,qy}}
            // Components of dot products:
            Scalar drxqx = dr.x*qp.v.x;
            Scalar dryqy = dr.y*qp.v.y;
            Scalar drzqz = dr.z*qp.v.z;

            new_cross_term.x += Scalar(2) * qp.v.x * (dryqy + drzqz);
            new_cross_term.y += Scalar(2) * qp.v.y * (drxqx + drzqz);
            new_cross_term.z += Scalar(2) * qp.v.z * (drxqx + dryqy);

            // {drx, dry, drz}*{qx, qy, qz}^2 . {{-1, 1, 1}, {1, -1, 1}, {1, 1, -1}}
            Scalar qr2 = qp.s * qp.s;
            Scalar qx2 = qp.v.x * qp.v.x;
            Scalar qy2 = qp.v.y * qp.v.y;
            Scalar qz2 = qp.v.z * qp.v.z;

            new_cross_term.x += dr.x * (qr2 + qx2 - qy2 - qz2);
            new_cross_term.y += dr.y * (qr2 - qx2 + qy2 - qz2);
            new_cross_term.z += dr.z * (qr2 - qx2 - qy2 + qz2);

            // The norm2 comes from the definition of rotation for quaternion
            // The magdr comes from the dot product definition of cos(theta_i)
            new_cross_term = new_cross_term / (magdr * norm2(qp));

            return new_cross_term;
        }
    //! Evaluate the force and energy
    /*
      // TODO update this
      \Param force Output parameter to write the computed force.
      \param envelope Output parameter to write the amount of modulation of the isotropic part
      \param torque_div_energy_i The torque exterted on the i^th particle.
      \param torque_div_energy_j The torque exterted on the j^th particle.
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
            Scalar jPi = modPj*modi/magdr; // something wrong here
            // TODO Jan 4 2023: I don't think this division by s.magdr should be here mathematically, but probably for efficiency


            // NEW way with Philipp Feb 9

            torque_div_energy_i =
                vec_to_scalar3( params.ni.x * cross( vec3<Scalar>(a1), dr)) +
                vec_to_scalar3( params.ni.y * cross( vec3<Scalar>(a2), dr)) +
                vec_to_scalar3( params.ni.z * cross( vec3<Scalar>(a3), dr));
            // std::cout << "torque_i before mult: " + vecString(vec3<Scalar>(torque_div_energy_i));
            // std::cout << "dr: " + vecString(vec3<Scalar>(dr));

            torque_div_energy_i *= Scalar(-1) * Modulatorj() * ModulatorPrimei() / magdr; // this last bit is iPj

            torque_div_energy_j =
                vec_to_scalar3( params.nj.x * cross( vec3<Scalar>(b1), dr)) +
                vec_to_scalar3( params.nj.y * cross( vec3<Scalar>(b2), dr)) +
                vec_to_scalar3( params.nj.z * cross( vec3<Scalar>(b3), dr));

            torque_div_energy_j *= Scalar(-1) * Modulatori() * ModulatorPrimej() / magdr;

            // std::cout << "a1 " + vecString(vec3<Scalar>(a1));
            // std::cout << "a2 " + vecString(vec3<Scalar>(a2));
            // std::cout << "a3 " + vecString(vec3<Scalar>(a3));
      //      std::cout << "ni " + vecString(vec3<Scalar>(ni)); // reset to 0 0 0?

            // std::cout << "torque_i: " + vecString(vec3<Scalar>(torque_div_energy_i));

            // std::cout << "b1 " + vecString(vec3<Scalar>(b1));
            // std::cout << "b2 " + vecString(vec3<Scalar>(b2));
            // std::cout << "b3 " + vecString(vec3<Scalar>(b3));
        //    std::cout << "nj " + vecString(vec3<Scalar>(nj)); // reset to 0 0 0?

            // std::cout << "iPj: " + std::to_string(iPj) + '\n';
            // std::cout << "jPi: " + std::to_string(jPi) + '\n'; // is always 0

            // std::cout << "torque_j: " + vecString(vec3<Scalar>(torque_div_energy_j));
            //

            
            // torque on ith
            // These are not the full torque. The pair energy is multiplied in PairModulator.
            // torque_div_energy_i = vec_to_scalar3(iPj * cross(vec3<Scalar>(s.ei), vec3<Scalar>(s.dr))); // TODO: is all the casting efficient?

            // The component of a2x, a2y, a2z ends up zero because the orientation is tied to the a1 direction.
            // Same for the a3 part.

            // The above comment is accurate when it's a uni-axial potential. Need to add more when I change the patch alignment.


            // New general torque for patch offset from a1 direction of particle

            // comments use Mathematica notation:

            // qr * Cross[{drx, dry, drz}, {qx, qy, qz}]

            // quat<Scalar> qpi = params.qpi; // quaternion representing orientation of patch with respect to particle x direction (a1)

            // quicker calculation and avoiding a bug in the following when patch is aligned
            // a1 = old ei

            // TODO turn new_cross_term into a function


            // I multiply iPj by magdr next for clarity during the derivation bc I did it above -Corwin
//            torque_div_energy_i = vec_to_scalar3( (iPj*magdr) * cross(vec3<Scalar>(a1), -new_cross_fun(params.qpi)));
            // TODO make the new_cross_term for torque_div_energy_j depend on 
            // Previously above, I would have s.ei which is the same, but I'm moving to a1 for clarity.

//                                                             vec3<Scalar> ii;
            // torque on jth - note sign is opposite ith
//            ii = new_cross_fun(params.qpi);
            // std::cout << vecString(ii);
//            vec3<Scalar> jj = new_cross_fun(params.qpj);
            // std::cout << vecString(jj);
            
//            torque_div_energy_j = vec_to_scalar3( (jPi*magdr) * cross(vec3<Scalar>(b1), jj));

            

            // std::cout << vecString(vec3<Scalar>(torque_div_energy_i));
            // std::cout << vecString(vec3<Scalar>(torque_div_energy_j));
            // TODO why is the order different than before?
            
            // compute force contribution
            // not the full force. Just the envelope that will be applied to pair energy

            // For first check, pretend that ei is a1.
            // force.x = -( ModulatorPrimei * Modulatorj /s.magdr *(-s.ei.x - s.doti*s.dr.x/s.magdr)
            //             + jPi*(s.ej.x - s.dotj*s.dr.x/s.magdr));

            // TODO still need to update this with respect to the conj(quat) bug

            // x component
            // hi_i = Scalar(-1) * dot(dr, ni) / norm2(params.qpi);
            // dhi_i = ;
                
            // hi_j = Scalar(1) * dot(dr, nj) / norm2(params.qpj);
            
            force.x = -(iPj*(-ei.x - doti*dr.x/magdr) // iPj includes a factor of 1/magdr
                        + jPi*(ej.x - dotj*dr.x/magdr));
            force.y = -(iPj*(-ei.y - doti*dr.y/magdr)
                        + jPi*(ej.y - dotj*dr.y/magdr));
            force.z = -(iPj*(-ei.z - doti*dr.z/magdr)
                        + jPi*(ej.z - dotj*dr.z/magdr));
            // for force we only care about derivative with respect to dr
            // only doti, dotj depend on dr because ei or ej is fixed from particle orientation

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
    Scalar doti, costhetai;
    Scalar dotj, costhetaj;
};

    } // end namespace md
    } // end namespace hoomd

#endif // __GENERAL_ENVELOPE_H__
