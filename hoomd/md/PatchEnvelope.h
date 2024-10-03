// Copyright (c) 2009-2024 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

#ifndef __GENERAL_ENVELOPE_H__
#define __GENERAL_ENVELOPE_H__

#ifndef __HIPCC__
#include <string>
#endif
#include <string.h>
#include "hoomd/HOOMDMath.h"
#include "hoomd/VectorMath.h"

/** need to declare these class methods with __device__ qualifiers when building in nvcc
    DEVICE is __host__ __device__ when included in nvcc and blank when included into the host
    compiler
*/
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

/** PatchEnvelope is an angle-dependent multiplier on an isotropic pair force to make it directional.

    Defines the envelopes \f( f_i, f_j \f):
    
    \f{align*}
    f_i(\vec{dr}, \vec{n}_i, \alpha) = \Big(1 + e^{-\omega (\frac{-\vec{dr} \cdot \vec{n_i}}{|\vec{dr}|} - \cos{\alpha})}\Big)^{-1}\\
    f_j(\vec{dr}, \vec{n}_j, \alpha) = \Big(1 + e^{-\omega (\frac{\vec{dr} \cdot \vec{n_j}}{|\vec{dr}|} - \cos{\alpha})}\Big)^{-1}
    \f}

    where \f$ \vec{n}_i, \vec{n}_j \f$ are the patch directions in the world frame,
    \f$ \alpha \f$ is the patch half-angle, and \f$ \omega \f$ is the patch steepness.
*/
class PatchEnvelope
{
public:
    struct param_type
    {
        param_type()
            {
            }
#ifndef __HIPCC__
        param_type(pybind11::dict params) //<! param dict can take any python type
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

    
/**  Constructor

     \param _dr Displacement vector from particle j to particle i
     \param q_i Quaternion of i^{th} particle
     \param q_j Quaternion of j^{th} particle
     \param _rcutsq Squared distance at which the potential goes to 0
     \param _params Per type pair parameters of this potential
     \param shape_i The patch location on the i^{th} particle
     \param shape_j The patch location on the j^{th} particle
*/
    DEVICE PatchEnvelope(
        const Scalar3& _dr,
        const Scalar4& q_i,
        const Scalar4& q_j,
        const Scalar _rcutsq,
        const param_type& _params,
        const shape_type& shape_i,
        const shape_type& shape_j)
        : dr(_dr), params(_params), p_i(shape_i.m_norm_patch_local_dir), p_j(shape_j.m_norm_patch_local_dir)
        {
            // compute current particle direction vectors

            // rotate from particle to world frame
            vec3<LongReal> ex(1,0,0);
            vec3<LongReal> ey(0,1,0);
            vec3<LongReal> ez(0,0,1);

            // a1, a2, a3 are orientation vectors of particle a in world frame
            // b1, b2, b3 are orientation vectors of particle b in world frame
            // ni_world is patch direction of particle i in world frame

#ifndef __HIPCC__
            auto R_i = rotmat3<LongReal>(quat<LongReal>(q_i));
            auto R_j = rotmat3<LongReal>(quat<LongReal>(q_j));
            a1 = R_i * ex;
            a2 = R_i * ey;
            a3 = R_i * ez;
            ni_world = R_i * (vec3<LongReal>)p_i;
            b1 = R_j * ex;
            b2 = R_j * ey;
            b3 = R_j * ez;
            nj_world = R_j * (vec3<LongReal>)p_j;
#else
            a1 = rotate(q_i, ex);
            a2 = rotate(q_i, ey);
            a3 = rotate(q_i, ez);
            ni_world = rotate(q_i, p_i);
            b1 = rotate(q_j, ex);
            b2 = rotate(q_j, ey);
            b3 = rotate(q_j, ez);
            nj_world = rotate(q_j, p_j);
#endif
            
            // compute distance
            drsq = dot(dr, dr);
            magdr = fast::sqrt(drsq);

            rhat = dr/magdr;

            // cos(angle between dr and pointing vector)
            Scalar costhetai = -dot(vec3<Scalar>(rhat), ni_world); // negative because dr = dx = pi - pj
            Scalar costhetaj = dot(vec3<Scalar>(rhat), nj_world);

            exp_neg_omega_times_cos_theta_i_minus_cos_alpha = fast::exp(-params.omega*(costhetai - params.cosalpha));
            exp_neg_omega_times_cos_theta_j_minus_cos_alpha = fast::exp(-params.omega*(costhetaj - params.cosalpha));
        }

    DEVICE static bool needsCharge() { return false; }

    DEVICE void setCharge(Scalar qi, Scalar qj)
    {
    m_charge_i = qi;
    m_charge_j = qj;
    }

    //! Evaluate the force and energy
    /*
      \Param force Output parameter to write the computed force.
      \param envelope Output parameter to write the amount of modulation of the isotropic part
      \param torque_div_energy_i The torque exterted on the i^th particle, divided by energy of interaction.
      \param torque_div_energy_j The torque exterted on the j^th particle, divided by energy of interaction.
      \note There is no need to check if rsq < rcutsq in this method. Cutoff tests are performed in PotentialPair from the PairModulator.
      \return Always true
    */
    DEVICE bool evaluate(Scalar3& force,
                         Scalar& envelope,
                         Scalar3& torque_div_energy_i,
                         Scalar3& torque_div_energy_j)
        {
            // common calculations

            Scalar f_min, f_max, f_max_min_inv;

            f_min = Scalar(1.0) / ( Scalar(1.0 + fast::exp(-params.omega*(-1 - params.cosalpha))) );
            f_max = Scalar(1.0) / ( Scalar(1.0 + fast::exp(-params.omega*(1 - params.cosalpha))) );

            f_max_min_inv = 1 / (f_max - f_min);


            Scalar fi = Scalar(1.0) / ( Scalar(1.0) + exp_neg_omega_times_cos_theta_i_minus_cos_alpha );
            Scalar dfi_du = params.omega * exp_neg_omega_times_cos_theta_i_minus_cos_alpha * f_max_min_inv * fi * fi;
            // normalize the modulator function
            fi = (fi - f_min) * f_max_min_inv;

            Scalar fj = Scalar(1.0) / ( Scalar(1.0) + exp_neg_omega_times_cos_theta_j_minus_cos_alpha );
            Scalar dfj_du = params.omega * exp_neg_omega_times_cos_theta_j_minus_cos_alpha * f_max_min_inv * fj * fj;
            fj = (fj - f_min) * f_max_min_inv;

            // the overall modulation
            envelope = fi * fj;

            vec3<Scalar> dfi_dni = dfi_du * -rhat;

            torque_div_energy_i =
                vec_to_scalar3( p_i.x * cross(a1, dfi_dni)) +
                vec_to_scalar3( p_i.y * cross(a2, dfi_dni)) +
                vec_to_scalar3( p_i.z * cross(a3, dfi_dni));

            torque_div_energy_i *= Scalar(-1) * fj;

            vec3<Scalar> dfj_dnj = dfj_du * rhat; // still positive

            torque_div_energy_j =
                vec_to_scalar3( p_j.x * cross(b1, dfj_dnj)) +
                vec_to_scalar3( p_j.y * cross(b2, dfj_dnj)) +
                vec_to_scalar3( p_j.z * cross(b3, dfj_dnj));

            torque_div_energy_j *= Scalar(-1) * fi;

            // find df/dr = df/du * du/dr (using chain rule)
            // find du/dr using quotient rule, where u = "hi" / "lo" = dot(dr,n) / magdr
            Scalar lo = magdr;
            vec3<Scalar> dlo = rhat;

            Scalar dfi_dui = dfi_du;

            Scalar hi = -dot(dr, vec3<Scalar>(ni_world));
            vec3<Scalar> dhi = -ni_world;
            /// quotient rule
            vec3<Scalar> dui_dr = (lo*dhi - hi*dlo) / (lo*lo);

            Scalar dfj_duj = dfj_du;
            hi = dot(vec3<Scalar>(dr), vec3<Scalar>(nj_world));
            dhi = nj_world;
            // lo and dlo are the same as above
            vec3<Scalar> duj_dr = (lo*dhi - hi*dlo) / (lo*lo);

            force = -vec_to_scalar3(dfj_duj*duj_dr * fi + dfi_dui*dui_dr * fj);

            return true;
        }

#ifndef _HIPCC_
    static std::string getName()
        {
            return std::string("patchenvelope");
        }
#endif

private:
    vec3<Scalar> dr;

    const param_type& params;
    vec3<Scalar> ni_world, nj_world;
    vec3<Scalar> p_i, p_j;
    vec3<Scalar> a1, a2, a3;
    vec3<Scalar> b1, b2, b3;

    Scalar m_charge_i, m_charge_j;

    Scalar drsq;
    Scalar magdr;
    vec3<Scalar> rhat;

    Scalar exp_neg_omega_times_cos_theta_i_minus_cos_alpha;
    Scalar exp_neg_omega_times_cos_theta_j_minus_cos_alpha;
};

    } // end namespace md
    } // end namespace hoomd

#endif // __GENERAL_ENVELOPE_H__
