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
template <typename AngleDependence> // thing that we modulate the potential by
class GeneralEnvelope
{
public:
    typedef typename AngleDependence::param_type param_type;

    DEVICE GeneralEnvelope(pybind11::dict params)
        : param_type(params)
        {
        }

    //! Constructor
    DEVICE GeneralEnvelope(
        const Scalar3& _dr,
        const Scalar4& _quat_i,
        const Scalar4& _quat_j,
        const Scalar4& _patch_orientation_i,
        const Scalar4& _patch_orientation_j,
        const Scalar& _rcutsq,
        const param_type& _params):
        s(_dr, _quat_i, _quat_j, _patch_orientation_i, _patch_orientation_j, _rcutsq, _params) {}

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
            Scalar modi = s.Modulatori();
            Scalar modj = s.Modulatorj();
            Scalar modPi = s.ModulatorPrimei();
            Scalar modPj = s.ModulatorPrimej();

            // the overall modulation
            envelope = modi*modj;

            // intermediate calculations
            Scalar iPj = modPi*modj/s.magdr; // TODO: make variable name more descriptive and check if these are correct. Jan 4: They are correct
            Scalar jPi = modPj*modi/s.magdr;
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
            vec3<Scalar> new_cross_term = s.oi.s * cross(s.dr, s.oi.v);

            // {qx, qy, qz}*{{dry,drz}.{qy,qz}, {drx,drz}.{qx,qz}, {drx,dry}.{qx,qy}}
            // Components of dot products:
            Scalar drxqx = s.dr.x*s.oi.v.x;
            Scalar dryqy = s.dr.y*s.oi.v.y;
            Scalar drzqz = s.dr.z*s.oi.v.z;

            new_cross_term.x += s.oi.v.x * (dryqy + drzqz);
            new_cross_term.y += s.oi.v.y * (drxqx + drzqz);
            new_cross_term.z += s.oi.v.z * (drxqx + dryqy);

            // {drx, dry, drz}*{qx, qy, qz}^2 . {{-1, 1, 1}, {1, -1, 1}, {1, 1, -1}}
            Scalar qx2 = s.oi.v.x * s.oi.v.x;
            Scalar qy2 = s.oi.v.y * s.oi.v.y;
            Scalar qz2 = s.oi.v.z * s.oi.v.z;

            new_cross_term.x += s.dr.x * (-qx2 + qy2 + qz2);
            new_cross_term.y += s.dr.y * ( qx2 - qy2 + qz2);
            new_cross_term.z += s.dr.z * ( qx2 + qy2 - qz2);

            // The norm2 comes from the definition of rotation for quaternion
            // The magdr comes from the dot product definition of cos(theta_i)
            new_cross_term = new_cross_term / (-s.magdr * norm2(s.oi));
            // I multiply iPj by magdr next for clarity during the derivation bc I did it above -Corwin
            torque_i = vec_to_scalar3( (iPj*s.magdr) * cross(vec3<Scalar>(s.a1), new_cross_term));

            // Previously above, I would have s.ei which is the same, but I'm moving to a1 for clarity.

            // torque on jth - note sign is opposite ith!
            torque_j = vec_to_scalar3( (jPi*s.magdr) * cross(vec3<Scalar>(s.b1), -new_cross_term));

            // compute force contribution
            // not the full force. Just the envelope that will be applied to pair energy

            // For first check, pretend that ei is a1.
            // force.x = -( ModulatorPrimei * Modulatorj /s.magdr *(-s.ei.x - s.doti*s.dr.x/s.magdr)
            //             + jPi*(s.ej.x - s.dotj*s.dr.x/s.magdr));
            force.x = -(iPj*(-s.ei.x - s.doti*s.dr.x/s.magdr)
                        + jPi*(s.ej.x - s.dotj*s.dr.x/s.magdr));
            force.y = -(iPj*(-s.ei.y - s.doti*s.dr.y/s.magdr)
                        + jPi*(s.ej.y - s.dotj*s.dr.y/s.magdr));
            force.z = -(iPj*(-s.ei.z - s.doti*s.dr.z/s.magdr)
                        + jPi*(s.ej.z - s.dotj*s.dr.z/s.magdr));
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
        AngleDependence s;
};

    } // end namespace md
    } // end namespace hoomd

#endif // __GENERAL_ENVELOPE_H__
