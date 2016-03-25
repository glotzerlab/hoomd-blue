
#include "hoomd/HOOMDMath.h"
#include "hoomd/BoxDim.h"
#include "hoomd/VectorMath.h"
#include "ShapeSphere.h"    //< For the base template of test_overlap
#include "ShapeSpheropolyhedron.h"

#ifndef __SHAPE_UNION_H__
#define __SHAPE_UNION_H__

/*! \file ShapeUnion.h
    \brief Defines the ShapeUnion templated aggregate shape
*/

// need to declare these class methods with __device__ qualifiers when building in nvcc
// DEVICE is __device__ when included in nvcc and blank when included into the host compiler
#ifdef NVCC
#define DEVICE __device__
#else
#define DEVICE
#include <iostream>
#endif

namespace hpmc
{

namespace detail
{

//! maximum number of constituent shapes
const int MAX_MEMBERS=10;

//! Data structure for shape composed of a union of multiple shapes
template<class Shape>
struct union_params : aligned_struct
    {
    typedef typename Shape::param_type mparam_type;
    unsigned int N;                          //!< Number of member shapes
    vec3<Scalar> mpos[MAX_MEMBERS];          //!< Position vectors of member shapes
    quat<Scalar> morientation[MAX_MEMBERS];  //!< Orientation of member shapes
    mparam_type mparams[MAX_MEMBERS];        //!< Parameters of member shapes
    OverlapReal diameter;                    //!< Precalculated overall circumsphere diameter
    unsigned int ignore;                     //!<  Bitwise ignore flag for stats, overlaps. 1 will ignore, 0 will not ignore
                                             //   First bit is ignore overlaps, Second bit is ignore statistics

    } __attribute__((aligned(32)));

} // end namespace detail

//! Shape consisting of union of shapes of a single type but individual parameters
/*!
    The parameter defining a ShapeUnion is a structure implementing the HPMC shape interface and containing
    parameter objects for its member particles in its own parameters structure

    The purpose of ShapeUnion is to allow an overlap check to iterate through pairs of member shapes between
    two composite particles. The two particles overlap if any of their member shapes overlap.
*/
template<class Shape>
struct ShapeUnion
    {
    //! Define the parameter type
    typedef typename detail::union_params<Shape> param_type;

    //! Initialize a sphere_union
    DEVICE ShapeUnion(const quat<Scalar>& _orientation, const param_type& _params)
        : orientation(_orientation), members(_params)
        {
        }

    //! Does this shape have an orientation
    DEVICE static bool hasOrientation() { return true; }

    //!Ignore flag for acceptance statistics
    DEVICE bool ignoreStatistics() const { return members.ignore>>1 & 0x01; }

    //!Ignore flag for overlaps
    DEVICE bool ignoreOverlaps() const { return members.ignore & 0x01; }

    //! Get the circumsphere diameter
    DEVICE OverlapReal getCircumsphereDiameter() const
        {
        // return the precomputed diameter
        return members.diameter;
        }

    //! Get the in-sphere radius
    DEVICE OverlapReal getInsphereRadius() const
        {
        // not implemented
        return OverlapReal(0.0);
        }

    //! Return the bounding box of the shape in world coordinates
    DEVICE detail::AABB getAABB(const vec3<Scalar>& pos) const
        {
        return detail::AABB(pos, members.diameter/OverlapReal(2.0));
        }

    //! Returns true if this shape splits the overlap check over several threads of a warp using threadIdx.x
    HOSTDEVICE static bool isParallel() { return false; }

    quat<Scalar> orientation;    //!< Orientation of the particle

    const param_type members;     //!< member data
    };

//! Check if circumspheres overlap
/*! \param r_ab Vector defining the position of shape b relative to shape a (r_b - r_a)
    \param a first shape
    \param b second shape
    \returns true if the circumspheres of both shapes overlap

    \ingroup shape
*/
template <class Shape>
DEVICE inline bool check_circumsphere_overlap(const vec3<Scalar>& r_ab, const ShapeUnion<Shape>& a,
    const ShapeUnion<Shape> &b)
    {
    vec3<OverlapReal> dr(r_ab);

    OverlapReal rsq = dot(dr,dr);
    OverlapReal DaDb = a.getCircumsphereDiameter() + b.getCircumsphereDiameter();
    return (rsq*OverlapReal(4.0) <= DaDb * DaDb);
    }

//! ShapeUnion overlap test
/*! \param r_ab vector from a to b: r_b - r_a
    \param a first shape
    \param b second shape
    \param err reference to an int to hold errors counted during overlap check
    \returns true when *a* and *b* overlap, and false when they are disjoint

    \ingroup shape
*/

template <class Shape >
DEVICE inline bool test_overlap(const vec3<Scalar>& r_ab,
                                      const ShapeUnion<Shape>& a,
                                      const ShapeUnion<Shape>& b,
                                      unsigned int& err)
    {
    //! Param type of the member shapes
    typedef typename Shape::param_type mparam_type;

    vec3<Scalar> dr(r_ab);

    bool result = false;
    for (unsigned int i=0; i < a.members.N; i++)
        {
        const mparam_type& params_i = a.members.mparams[i];
        const quat<Scalar> q_i = a.orientation * a.members.morientation[i];
        Shape shape_i(q_i, params_i);
        for (unsigned int j=0; j < b.members.N; j++)
            {
            const mparam_type& params_j = b.members.mparams[j];
            const quat<Scalar> q_j = b.orientation * b.members.morientation[j];
            vec3<Scalar> r_ij = dr + rotate(b.orientation, b.members.mpos[j]) - rotate(a.orientation, a.members.mpos[i]);
            Shape shape_j(q_j, params_j);
            bool temp_result = test_overlap(r_ij, shape_i, shape_j, err);
            if (temp_result == true)
                {
                result = true;
                break;
                }
            }
        }
    return result;
    }

} // end namespace hpmc

#endif // end __SHAPE_UNION_H__
