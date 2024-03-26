// Copyright (c) 2009-2024 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

#include "hoomd/HOOMDMath.h"
#include "hoomd/VectorMath.h"

#ifndef __MINKOWSKI_MATH_H__
#define __MINKOWSKI_MATH_H__

/*! \file MinkowskiMath.h
    \brief Composite support functions for use in Minkowski based methods
*/

// need to declare these class methods with __device__ qualifiers when building in nvcc
// DEVICE is __device__ when included in nvcc and blank when included into the host compiler
#ifdef __HIPCC__
#define DEVICE __device__
#else
#define DEVICE
#endif

namespace hoomd
    {
namespace hpmc
    {
namespace detail
    {
//! Composite support functor
/*! \tparam SupportFuncA Support function class type for shape A
    \tparam SupportFuncB Support function class type for shape B

    Helper functor that computes the support function of the Minkowski difference B-A from the given
   two support functions. The given support functions are kept in local coords and
   translations/rotations are performed going in and out so that the input *n* and final result are
   in the space frame (where a is at the origin).

    This operation is performed many times in XenoCollide, so this convenience class simplifies the
   calling code from having too many rotations/translations.

    \ingroup minkowski
*/
template<class SupportFuncA, class SupportFuncB> class CompositeSupportFunc3D
    {
    public:
    //! Construct a composite support function
    /*! \param _sa Support function for shape A
        \param _sb Support function for shape B
        \param _ab_t Vector pointing from a's center to b's center, in the space frame
        \param _q Orientation of shape B in frame A
    */
    DEVICE CompositeSupportFunc3D(const SupportFuncA& _sa,
                                  const SupportFuncB& _sb,
                                  const vec3<ShortReal>& _ab_t,
                                  const quat<ShortReal>& _q)
#ifdef __HIPCC__
        : sa(_sa), sb(_sb), ab_t(_ab_t), q(_q)
#else
        : sa(_sa), sb(_sb), ab_t(_ab_t), R(rotmat3<ShortReal>(_q))
#endif
        {
        }

    //! Compute the support function
    /*! \param n Normal vector input (in the A frame)
        \returns S_B(n) - S_A(n) in world space coords (transformations put n into local coords for
       S_A and S_b)
    */
    DEVICE vec3<ShortReal> operator()(const vec3<ShortReal>& n) const
        {
            // translation/rotation formula comes from pg 168 of "Games Programming Gems 7"
#ifdef __HIPCC__
        vec3<ShortReal> SB_n = rotate(q, sb(rotate(conj(q), n))) + ab_t;
        vec3<ShortReal> SA_n = sa(-n);
#else
        vec3<ShortReal> SB_n = R * sb(transpose(R) * n) + ab_t;
        vec3<ShortReal> SA_n = sa(-n);
#endif
        return SB_n - SA_n;
        }

    private:
    const SupportFuncA& sa; //!< Support function for shape A
    const SupportFuncB& sb; //!< Support function for shape B
    const vec3<ShortReal>&
        ab_t; //!< Vector pointing from a's center to b's center, in the space frame
#ifdef __HIPCC__
    const quat<ShortReal>& q; //!< Orientation of shape B in frame A

#else
    const rotmat3<ShortReal> R; //!< Orientation of shape B in A frame

#endif
    };

    } // end namespace detail

    } // end namespace hpmc
    } // end namespace hoomd

#endif // __MINKOWSKI_MATH_H__
