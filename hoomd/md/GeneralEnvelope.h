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

        static inline std::string vecString(vec3<Scalar> a) {
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
#ifndef __HIPCC__
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
#endif
        Scalar cosalpha;
        Scalar omega;
    }__attribute__((aligned(16)));

    struct shape_type
    {
        HOSTDEVICE shape_type() { }

#ifndef __HIPCC__

        shape_type(pybind11::object patch_location)
            {
            pybind11::tuple p_py = patch_location;
            if (len(p_py) != 3)
                throw std::runtime_error("Each patch position must have 3 elements");
            vec3<Scalar> p = vec3<Scalar>(pybind11::cast<Scalar>(p_py[0]),
                                          pybind11::cast<Scalar>(p_py[1]),
                                          pybind11::cast<Scalar>(p_py[2]));

            // normalize
            p = p * fast::rsqrt(dot(p, p));
            m_norm_patch_local_dir = vec_to_scalar3(p);
            }

        pybind11::object toPython()
            {
            return pybind11::make_tuple(m_norm_patch_local_dir.x, m_norm_patch_local_dir.y, m_norm_patch_local_dir.z);
            }
#endif

        Scalar3 m_norm_patch_local_dir;
    };

    DEVICE GeneralEnvelope( // TODO replace GeneralEnvelope with PatchesEnvelope
        const Scalar3& _dr,
        const Scalar4& _quat_i, // Note in hoomd, the quaternion is how to get from the particle orientation to align to the world orientation. so World = qi Local qi-1
        const Scalar4& _quat_j,
        const Scalar _rcutsq,
        const param_type& _params,
        const shape_type& shape_i,
        const shape_type& shape_j)
// #ifdef __HIPCC__
        // TODO decide which is faster on GPU and CPU. // use 250_000 particles to saturate GPU // benchmark on brett
        // : dr(_dr), qi(_quat_i), qj(_quat_j), params(_params), p_i(shape_i.m_norm_patch_local_dir), p_j(shape_j.m_norm_patch_local_dir)
// #else
        : dr(_dr), R_i(rotmat3<ShortReal>((quat<ShortReal>)_quat_i)), R_j(rotmat3<ShortReal>((quat<ShortReal>)_quat_j)), params(_params), p_i(shape_i.m_norm_patch_local_dir), p_j(shape_j.m_norm_patch_local_dir)
// #endif
        {
            // compute current particle direction vectors

            // rotate from particle to world frame
            vec3<ShortReal> ex(1,0,0);
            vec3<ShortReal> ey(0,1,0);
            vec3<ShortReal> ez(0,0,1);

            //TODO Real --> ShortReal could help

            // DONE try converting to rotation matrices

            // a1, a2, a3 are orientation vectors of particle a in world frame
            // b1, b2, b3 are orientation vectors of particle b in world frame
            // ni_world is patch direction of particle i in world frame
// #ifdef __HIPCC__
            a1 = R_i * ex;
            a2 = R_i * ey;
            a3 = R_i * ez;
            ni_world = R_i * (vec3<ShortReal>)p_i;
            b1 = R_j * ex;
            b2 = R_j * ey;
            b3 = R_j * ez;
            nj_world = R_j * (vec3<ShortReal>)p_j;
// #else
//             a1 = rotate(qi, ex);
//             a2 = rotate(qi, ey);
//             a3 = rotate(qi, ez);
//             ni_world = rotate(qi, p_i);
//             b1 = rotate(qj, ex);
//             b2 = rotate(qj, ey);
//             b3 = rotate(qj, ez);
//             nj_world = rotate(qj, p_j);
// #endif
            
            // compute distance
            drsq = dot(dr, dr);
            magdr = fast::sqrt(drsq);

            rhat = dr/magdr;

            // cos(angle between dr and pointing vector)
            Scalar costhetai = -dot(vec3<Scalar>(rhat), ni_world); // negative because dr = dx = pi - pj
            Scalar costhetaj = dot(vec3<Scalar>(rhat), nj_world);

            exp_negOmega_times_CosThetaI_minus_CosAlpha = fast::exp(-params.omega*(costhetai - params.cosalpha));
            exp_negOmega_times_CosThetaJ_minus_CosAlpha = fast::exp(-params.omega*(costhetaj - params.cosalpha));
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

            Scalar f_min, f_max, f_max_min_inv;

            f_min = Scalar(1.0) / ( Scalar(1.0 + fast::exp(-params.omega*(-1 - params.cosalpha))) );
            f_max = Scalar(1.0) / ( Scalar(1.0 + fast::exp(-params.omega*(1 - params.cosalpha))) );

            f_max_min_inv = 1 / (f_max - f_min);


            Scalar fi = Scalar(1.0) / ( Scalar(1.0) + exp_negOmega_times_CosThetaI_minus_CosAlpha );
            Scalar dfi_du = params.omega * exp_negOmega_times_CosThetaI_minus_CosAlpha * f_max_min_inv * fi * fi;
            // normalize the modulator function
            fi = (fi - f_min) * f_max_min_inv;

            Scalar fj = Scalar(1.0) / ( Scalar(1.0) + exp_negOmega_times_CosThetaJ_minus_CosAlpha );
            Scalar dfj_du = params.omega * exp_negOmega_times_CosThetaJ_minus_CosAlpha * f_max_min_inv * fj * fj;
            fj = (fj - f_min) * f_max_min_inv;

            // the overall modulation
            envelope = fi * fj;

            vec3<Scalar> dfi_dni = dfi_du * -rhat;

            torque_div_energy_i =
                vec_to_scalar3( p_i.x * cross( vec3<Scalar>(a1), dfi_dni)) +
                vec_to_scalar3( p_i.y * cross( vec3<Scalar>(a2), dfi_dni)) +
                vec_to_scalar3( p_i.z * cross( vec3<Scalar>(a3), dfi_dni));

            torque_div_energy_i *= Scalar(-1) * fj;

            vec3<Scalar> dfj_dnj = dfj_du * rhat; // still positive

            torque_div_energy_j =
                vec_to_scalar3( p_j.x * cross( vec3<Scalar>(b1), dfj_dnj)) +
                vec_to_scalar3( p_j.y * cross( vec3<Scalar>(b2), dfj_dnj)) +
                vec_to_scalar3( p_j.z * cross( vec3<Scalar>(b3), dfj_dnj));

            torque_div_energy_j *= Scalar(-1) * fi;


            // THIS PART in here:
            //     dfj_duj * duj_dr * self.patch.fi(dr, self.ni_world) + dfi_dui * dui_dr * self.patch.fj(dr, self.nj_world)

            
            // find df/dr = df/du * du/dr
            // find du/dr using quotient rule, where u = "hi" / "lo" = dot(dr,n) / magdr
            Scalar lo = magdr;
            vec3<Scalar> dlo = rhat;

            // //something wrong: this has to be a scalar
            // Scalar3 dfi_dui = dfi_dni();
            Scalar dfi_dui = dfi_du;

            Scalar hi = -dot(dr, vec3<Scalar>(ni_world));
            vec3<Scalar> dhi = -ni_world;
            // // quotient rule
            vec3<Scalar> dui_dr = (lo*dhi - hi*dlo) / (lo*lo);
            //           dui_dr = removedrhat * ( magdr* -ni_world - -dot(dr, vec3<Scalar>(ni_world)) * rhat ) / (magdr*magdr)

            Scalar dfj_duj = dfj_du;
            hi = dot(vec3<Scalar>(dr), vec3<Scalar>(nj_world));
            dhi = nj_world;
            // // lo and dlo are the same as above
            vec3<Scalar> duj_dr = (lo*dhi - hi*dlo) / (lo*lo);

            force = -vec_to_scalar3(dfj_duj*duj_dr * fi + dfi_dui*dui_dr * fj);

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

// #ifdef __HIPCC__
    quat<Scalar> qi;
    quat<Scalar> qj;
// #else
    rotmat3<ShortReal> R_i;
    rotmat3<ShortReal> R_j;
// #endif

    const param_type& params;
    vec3<Scalar> ni_world, nj_world;
    vec3<Scalar> p_i, p_j;
    vec3<Scalar> a1, a2, a3;
    vec3<Scalar> b1, b2, b3;


    Scalar drsq;
    Scalar magdr;
    vec3<Scalar> rhat;

    Scalar exp_negOmega_times_CosThetaI_minus_CosAlpha;
    Scalar exp_negOmega_times_CosThetaJ_minus_CosAlpha;
};

    } // end namespace md
    } // end namespace hoomd

#endif // __GENERAL_ENVELOPE_H__
