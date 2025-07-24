// Copyright (c) 2009-2025 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

#ifndef __ALIGN_ENVELOPE_H__
#define __ALIGN_ENVELOPE_H__

#ifndef __HIPCC__
#include <string>
#endif
#include "hoomd/HOOMDMath.h"
#include "hoomd/VectorMath.h"
#include <string.h>

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

/** AlignEnvelope is an angle-dependent multiplier on an isotropic pair force to make it
   directional.

    Defines the envelopes \f( f_i, f_j \f):

    \f{align*}
    f_i(\vec{dr}, \vec{n}_i, \alpha) = \Big(1 + e^{-\omega (\frac{-\vec{dr} \cdot
   \vec{n_i}}{|\vec{dr}|} - \cos{\alpha})}\Big)^{-1}\\ f_j(\vec{dr}, \vec{n}_j, \alpha) = \Big(1 +
   e^{-\omega (\frac{\vec{dr} \cdot \vec{n_j}}{|\vec{dr}|} - \cos{\alpha})}\Big)^{-1} \f}

    where \f$ \vec{n}_i, \vec{n}_j \f$ are the patch directions in the world frame,
    \f$ \alpha \f$ is the patch half-angle, and \f$ \omega \f$ is the patch steepness.
*/
class AlignEnvelope
    {
    public:
    struct param_type
        {
        param_type() : additive(false), anti_align(false) { }
#ifndef __HIPCC__
        param_type(pybind11::dict params) //<! param dict can take any python type
            {
            additive = params["additive"].cast<bool>();
            anti_align = params["anti_align"].cast<bool>();
            }

        pybind11::dict toPython()
            {
            pybind11::dict v;
            v["additive"] = additive;
            v["anti_align"] = anti_align;
            return v;
            }
#endif
	bool additive;
	bool anti_align;
        } __attribute__((aligned(16)));

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
            return pybind11::make_tuple(m_norm_patch_local_dir.x,
                                        m_norm_patch_local_dir.y,
                                        m_norm_patch_local_dir.z);
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
    DEVICE AlignEnvelope(const Scalar3& _dr,
                         const Scalar4& _q_i,
                         const Scalar4& _q_j,
                         const Scalar _rcutsq,
                         const param_type& _params,
                         const shape_type& shape_i,
                         const shape_type& shape_j)
        : params(_params), p_i(shape_i.m_norm_patch_local_dir),
          p_j(shape_j.m_norm_patch_local_dir)
        {
        // compute current particle direction vectors

        // rotate from particle to world frame
        vec3<LongReal> ex(1, 0, 0);
        vec3<LongReal> ey(0, 1, 0);
        vec3<LongReal> ez(0, 0, 1);

        // a1, a2, a3 are orientation vectors of particle a in world frame
        // b1, b2, b3 are orientation vectors of particle b in world frame
        // ni_world is patch direction of particle i in world frame

        auto q_i = quat<LongReal>(_q_i);
        auto q_j = quat<LongReal>(_q_j);

#ifndef __HIPCC__
        auto R_i = rotmat3<LongReal>(q_i);
        auto R_j = rotmat3<LongReal>(q_j);
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
        }

    DEVICE static bool needsCharge()
        {
        return false;
        }

    DEVICE void setCharge(Scalar qi, Scalar qj)
        {
        m_charge_i = qi;
        m_charge_j = qj;
        }

    //! Evaluate the force and energy
    /*
      \Param force Output parameter to write the computed force.
      \param envelope Output parameter to write the amount of modulation of the isotropic part
      \param torque_div_energy_i The torque exterted on the i^th particle, divided by energy of
      interaction. \param torque_div_energy_j The torque exterted on the j^th particle, divided by
      energy of interaction. \note There is no need to check if rsq < rcutsq in this method. Cutoff
      tests are performed in PotentialPair from the PairModulator. \return Always true
    */
    DEVICE bool evaluate(Scalar3& force,
                         Scalar& envelope,
                         Scalar3& torque_div_energy_i,
                         Scalar3& torque_div_energy_j)
        {
        // common calculations

	envelope = dot(nj_world,ni_world);

        torque_div_energy_i = vec_to_scalar3(p_i.x * cross(a1, nj_world))
                              + vec_to_scalar3(p_i.y * cross(a2, nj_world))
                              + vec_to_scalar3(p_i.z * cross(a3, nj_world));

        torque_div_energy_j = vec_to_scalar3(p_j.x * cross(b1, ni_world))
                              + vec_to_scalar3(p_j.y * cross(b2, ni_world))
                              + vec_to_scalar3(p_j.z * cross(b3, ni_world));

	if( params.anti_align)
        	envelope *= Scalar(-1);
	else
	{
        	torque_div_energy_i *= Scalar(-1);
        	torque_div_energy_j *= Scalar(-1);
	}

        force = make_scalar3(0,0,0);

        return true;
        }

#ifndef _HIPCC_
    static std::string getName()
        {
        return std::string("alignenvelope");
        }
#endif

    private:
    const param_type& params;
    vec3<Scalar> ni_world, nj_world;
    vec3<Scalar> p_i, p_j;
    vec3<Scalar> a1, a2, a3;
    vec3<Scalar> b1, b2, b3;

    Scalar m_charge_i, m_charge_j;

    };

    } // end namespace md
    } // end namespace hoomd

#endif // __ALIGN_ENVELOPE_H__
