// Copyright (c) 2009-2019 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.

#include "hoomd/HOOMDMath.h"
#include "HPMCPrecisionSetup.h"
#include "hoomd/VectorMath.h"

#ifndef __MINKOWSKI_MATH_H__
#define __MINKOWSKI_MATH_H__

/*! \file MinkowskiMath.h
    \brief Composite support functions for use in Minkowski based methods
*/

// need to declare these class methods with __device__ qualifiers when building in nvcc
// DEVICE is __device__ when included in nvcc and blank when included into the host compiler
#ifdef NVCC
#define DEVICE __device__
#else
#define DEVICE
#endif

namespace hpmc
{

namespace detail
{

//! Composite support functor
/*! \tparam SupportFuncA Support function class type for shape A
    \tparam SupportFuncB Support function class type for shape B

    Helper functor that computes the support function of the Minkowski difference B-A from the given two support
    functions. The given support functions are kept in local coords and translations/rotations are performed going in
    and out so that the input *n* and final result are in the space frame (where a is at the origin).

    This operation is performed many times in XenoCollide, so this convenience class simplifies the calling code
    from having too many rotations/translations.

    \ingroup minkowski
*/
template<class SupportFuncA, class SupportFuncB>
class CompositeSupportFunc3D
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
                                    const vec3<OverlapReal>& _ab_t,
                                    const quat<OverlapReal>& _q)
#ifdef NVCC
            : sa(_sa), sb(_sb), ab_t(_ab_t), q(_q)
#else
            : sa(_sa), sb(_sb), ab_t(_ab_t), R(rotmat3<OverlapReal>(_q))
#endif
            {}

        //! Compute the support function
        /*! \param n Normal vector input (in the A frame)
            \returns S_B(n) - S_A(n) in world space coords (transformations put n into local coords for S_A and S_b)
        */
        DEVICE vec3<OverlapReal> operator() (const vec3<OverlapReal>& n) const
            {
            // translation/rotation formula comes from pg 168 of "Games Programming Gems 7"
#ifdef NVCC
            vec3<OverlapReal> SB_n = rotate(q, sb(rotate(conj(q),n))) + ab_t;
            vec3<OverlapReal> SA_n = sa(-n);
#else
            vec3<OverlapReal> SB_n = R * sb(transpose(R)*n) + ab_t;
            vec3<OverlapReal> SA_n = sa(-n);
#endif
            return SB_n - SA_n;
            }

    private:
        const SupportFuncA& sa;    //!< Support function for shape A
        const SupportFuncB& sb;    //!< Support function for shape B
        const vec3<OverlapReal>& ab_t;  //!< Vector pointing from a's center to b's center, in the space frame
#ifdef NVCC
        const quat<OverlapReal>& q; //!< Orientation of shape B in frame A

#else
        const rotmat3<OverlapReal> R; //!< Orientation of shape B in A frame

#endif
    };

}; // end namespace detail

}; // end namespace hpmc

#endif // __MINKOWSKI_MATH_H__
