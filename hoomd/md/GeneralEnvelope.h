// Copyright (c) 2009-2022 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.


#ifndef __GENERAL_ENVELOPE_H__
#define __GENERAL_ENVELOPE_H__

#ifndef __HIPCC__
#include <string>
#endif

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

class GeneralEnvelope
{
public:
    //typedef typename AngleDependence::param_type param_type;
    struct param_type
    {
        param_type()
            {
            }

        param_type(pybind11::dict params)
            : cosalpha( fast::cos(params["alpha"].cast<Scalar>()) ),
              omega(params["omega"].cast<Scalar>())
            {
            }

        pybind11::dict asDict()
            {
                pybind11::dict v;

                v["alpha"] = fast::acos(cosalpha);
                v["omega"] = omega;

                return v;
            }

        Scalar cosalpha;
        Scalar omega;
    }
#ifdef SINGLE_PRECISION
        __attribute__((aligned(8)));
#else
        __attribute__((aligned(16)));
#endif

    //! Constructor
    DEVICE GeneralEnvelope( // TODO: this is not actually general. It assumes a single off center patch
        const vec3<Scalar>& _dr,
        const quat<Scalar>& _quat_i,
        const quat<Scalar>& _quat_j,
        const quat<Scalar>& _patch_orientation_i,
        const quat<Scalar>& _patch_orientation_j,
        const Scalar& _rcutsq,
        const param_type& _params)
        : dr(_dr), qi(_quat_i), qj(_quat_j), oi(_patch_orientation_i), oj(_patch_orientation_j), params(_params) // patch orientation is per type, so it's not in params
        {
            // compute current janus direction vectors
            vec3<Scalar> ex { make_scalar3(1, 0, 0) };
            vec3<Scalar> ey { make_scalar3(0, 1, 0) };
            vec3<Scalar> ez { make_scalar3(0, 0, 1) };

            //vec3<Scalar> ei, ej;
            //vec3<Scalar> ni, nj;
            // vec3<Scalar> a1, a2, a3, b1, b2, b3;

            ei = rotate(qi, ex);
            a1 = rotate(qi, ex);
            a2 = rotate(qi, ey);
            a3 = rotate(qi, ez);
            // patch points relative to x (a1) direction of particle
            ni = rotate(oi, a1);

            ej = rotate(qj, ex);
            b1 = rotate(qj, ex);
            b2 = rotate(qj, ey);
            b3 = rotate(qj, ez);

            // patch points relative to x (b1) direction of particle
            nj = rotate(oj, b1);

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
            vec3<Scalar> new_cross_term = oi.s * cross(dr, oi.v);

            // {qx, qy, qz}*{{dry,drz}.{qy,qz}, {drx,drz}.{qx,qz}, {drx,dry}.{qx,qy}}
            // Components of dot products:
            Scalar drxqx = dr.x*oi.v.x;
            Scalar dryqy = dr.y*oi.v.y;
            Scalar drzqz = dr.z*oi.v.z;

            new_cross_term.x += oi.v.x * (dryqy + drzqz);
            new_cross_term.y += oi.v.y * (drxqx + drzqz);
            new_cross_term.z += oi.v.z * (drxqx + dryqy);

            // {drx, dry, drz}*{qx, qy, qz}^2 . {{-1, 1, 1}, {1, -1, 1}, {1, 1, -1}}
            Scalar qx2 = oi.v.x * oi.v.x;
            Scalar qy2 = oi.v.y * oi.v.y;
            Scalar qz2 = oi.v.z * oi.v.z;

            new_cross_term.x += dr.x * (-qx2 + qy2 + qz2);
            new_cross_term.y += dr.y * ( qx2 - qy2 + qz2);
            new_cross_term.z += dr.z * ( qx2 + qy2 - qz2);

            // The norm2 comes from the definition of rotation for quaternion
            // The magdr comes from the dot product definition of cos(theta_i)
            new_cross_term = new_cross_term / (-magdr * norm2(oi));
            // I multiply iPj by magdr next for clarity during the derivation bc I did it above -Corwin
            torque_i = vec_to_scalar3( (iPj*magdr) * cross(vec3<Scalar>(a1), new_cross_term));

            // Previously above, I would have s.ei which is the same, but I'm moving to a1 for clarity.

            // torque on jth - note sign is opposite ith!
            torque_j = vec_to_scalar3( (jPi*magdr) * cross(vec3<Scalar>(b1), -new_cross_term));

            // compute force contribution
            // not the full force. Just the envelope that will be applied to pair energy

            // For first check, pretend that ei is a1.
            // force.x = -( ModulatorPrimei * Modulatorj /s.magdr *(-s.ei.x - s.doti*s.dr.x/s.magdr)
            //             + jPi*(s.ej.x - s.dotj*s.dr.x/s.magdr));
            force.x = -(iPj*(-ei.x - doti*dr.x/magdr)
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
    //AngleDependence s;
    vec3<Scalar> dr;
    quat<Scalar> qi;
    quat<Scalar> qj;

    quat<Scalar> oi; // quaternion representing orientation of patch with respect to particle x direction (a1)
    quat<Scalar> oj; // quaternion representing orientation of patch with respect to particle x direction (b1)
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
