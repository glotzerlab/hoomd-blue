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

                auto ni = vec3<Scalar>(ni_[0].cast<Scalar>(), ni_[1].cast<Scalar>(), ni_[2].cast<Scalar>());
                auto nj = vec3<Scalar>(nj_[0].cast<Scalar>(), nj_[1].cast<Scalar>(), nj_[2].cast<Scalar>());

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
                vec3<Scalar> ni = rotate(qpi, ex);
                vec3<Scalar> nj = rotate(qpj, ex);

                v["ni"] = pybind11::make_tuple(ni.x, ni.y, ni.z);
                v["nj"] = pybind11::make_tuple(nj.x, nj.y, nj.z);

                return v;
            }

        quat<Scalar> qpi;
        quat<Scalar> qpj;
        Scalar cosalpha;
        Scalar omega;
    }__attribute__((aligned(16)));

    DEVICE GeneralEnvelope( // TODO: this is not actually general. It assumes a single off-center patch
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
            ni = rotate(params.qpi, a1);

            // TODO combine rotations ni = rotate(params.qpi * conj(qi), ex)

            ej = rotate(conj(qj), ex);
            b1 = rotate(conj(qj), ex);
            b2 = rotate(conj(qj), ey);
            b3 = rotate(conj(qj), ez);

            // patch points relative to x (b1) direction of particle
            nj = rotate(params.qpj, b1);

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

            costhetai = -dot(vec3<Scalar>(dr), ni) / magdr;
            costhetaj = dot(vec3<Scalar>(dr), nj) / magdr;
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


    DEVICE inline Scalar Modulatori()
        {
            return Scalar(1.0) / ( Scalar(1.0) + fast::exp(-params.omega*(costhetai-params.cosalpha)) );
        }

    DEVICE inline Scalar Modulatorj()
        {
            return Scalar(1.0) / ( Scalar(1.0) + fast::exp(-params.omega*(costhetaj-params.cosalpha)) );
        }

    DEVICE Scalar ModulatorPrimei() // D[f[\[Theta], \[Alpha], \[Omega]], Cos[\[Theta]]]
        {
            Scalar fact = Modulatori();
            // weird way of writing out the derivative of f with respect to doti = Cos[theta] =
            return params.omega * fast::exp(-params.omega*(costhetai-params.cosalpha)) * fact * fact;
        }

    DEVICE Scalar ModulatorPrimej()
        {
            Scalar fact = Modulatorj();
            return params.omega * fast::exp(-params.omega*(costhetaj-params.cosalpha)) * fact * fact;
        }

    //! Evaluate the force and energy
    /*
      \param force Output parameter to write the computed force.
      \param envelope Output parameter to write the amount of modulation of the isotropic part
      \param torque_i The torque exterted on the i^th particle.
      \param torque_j The torque exterted on the j^th particle.
      \note There is no need to check if rsq < rcutsq in this method. Cutoff tests are performed in PotentialPair from the PairModulator.
      \return Always true
    */
    DEVICE bool evaluate(Scalar3& force,
                         Scalar& envelope,
                         Scalar3& torque_i, //torque_modulator
                         Scalar3& torque_j) //torque_modulator
        {
            // common calculations
            Scalar modi = Modulatori();
            Scalar modj = Modulatorj();
            Scalar modPi = ModulatorPrimei();
            Scalar modPj = ModulatorPrimej();

            std::cout << "------------\n";
            // the overall modulation
            envelope = modi*modj;

            // intermediate calculations
            Scalar iPj = modPi*modj/magdr; // TODO: make variable name more descriptive and check if these are correct. Jan 4: They are correct
            Scalar jPi = modPj*modi/magdr;
            // TODO Jan 4 2023: I don't think this division by s.magdr should be here mathematically, but probably for efficiency

            // torque on ith
            // These are not the full torque. The pair energy is multiplied in PairModulator.
            // torque_i = vec_to_scalar3(iPj * cross(vec3<Scalar>(s.ei), vec3<Scalar>(s.dr))); // TODO: is all the casting efficient?

            // The component of a2x, a2y, a2z ends up zero because the orientation is tied to the a1 direction.
            // Same for the a3 part.

            // The above comment is accurate when it's a uni-axial potential. Need to add more when I change the patch alignment.


            // New general torque for patch offset from a1 direction of particle

            // comments use Mathematica notation:

            // qr * Cross[{drx, dry, drz}, {qx, qy, qz}]

            quat<Scalar> qpi = params.qpi; // quaternion representing orientation of patch with respect to particle x direction (a1)

            std::cout << params.qpi.s << params.qpi.v.x << params.qpi.v.y << params.qpi.v.z << '\n';             // this is okay

            // quicker calculation and avoiding a bug in the following when patch is aligned
            // a1 = old ei
            vec3<Scalar> new_cross_term;
            if (qpi.s == Scalar(1) && qpi.v == vec3<Scalar>(0,0,0) )
                {
                    new_cross_term = -dr / magdr;
                }
            else
                {
                    new_cross_term = qpi.s * cross(dr, qpi.v);
                    std::cout << vecString(dr); // this is okay
                    std::cout << vecString(qpi.v); // this is zero. Problem!
                    std::cout << vecString(new_cross_term);

                    // {qx, qy, qz}*{{dry,drz}.{qy,qz}, {drx,drz}.{qx,qz}, {drx,dry}.{qx,qy}}
                    // Components of dot products:
                    Scalar drxqx = dr.x*qpi.v.x;
                    Scalar dryqy = dr.y*qpi.v.y;
                    Scalar drzqz = dr.z*qpi.v.z;

                    new_cross_term.x += qpi.v.x * (dryqy + drzqz);
                    new_cross_term.y += qpi.v.y * (drxqx + drzqz);
                    new_cross_term.z += qpi.v.z * (drxqx + dryqy);

                    // {drx, dry, drz}*{qx, qy, qz}^2 . {{-1, 1, 1}, {1, -1, 1}, {1, 1, -1}}
                    Scalar qx2 = qpi.v.x * qpi.v.x;
                    Scalar qy2 = qpi.v.y * qpi.v.y;
                    Scalar qz2 = qpi.v.z * qpi.v.z;

                    new_cross_term.x += dr.x * (-qx2 + qy2 + qz2);
                    new_cross_term.y += dr.y * ( qx2 - qy2 + qz2);
                    new_cross_term.z += dr.z * ( qx2 + qy2 - qz2);

                    // The norm2 comes from the definition of rotation for quaternion
                    // The magdr comes from the dot product definition of cos(theta_i)
                    new_cross_term = new_cross_term / (-magdr * norm2(qpi));
                }

            // I multiply iPj by magdr next for clarity during the derivation bc I did it above -Corwin
            torque_i = vec_to_scalar3( (iPj*magdr) * cross(vec3<Scalar>(a1), new_cross_term));


            // Previously above, I would have s.ei which is the same, but I'm moving to a1 for clarity.

            // torque on jth - note sign is opposite ith!
            torque_j = vec_to_scalar3( (jPi*magdr) * cross(vec3<Scalar>(b1), -new_cross_term));
            // TODO why is the order different than before?
            
            // compute force contribution
            // not the full force. Just the envelope that will be applied to pair energy

            // For first check, pretend that ei is a1.
            // force.x = -( ModulatorPrimei * Modulatorj /s.magdr *(-s.ei.x - s.doti*s.dr.x/s.magdr)
            //             + jPi*(s.ej.x - s.dotj*s.dr.x/s.magdr));

            // TODO still need to update this with respect to the conj(quat) bug
            force.x = -(iPj*(-ei.x - doti*dr.x/magdr) // iPj includes a factor of magdr
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

    vec3<Scalar> ni; // pointing vector for patch on particle i
    vec3<Scalar> nj; // pointing vector for patch on particle j

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
