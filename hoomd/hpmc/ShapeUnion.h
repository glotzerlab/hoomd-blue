
#include "hoomd/HOOMDMath.h"
#include "hoomd/BoxDim.h"
#include "hoomd/VectorMath.h"
#include "ShapeSphere.h"    //< For the base template of test_overlap
#include "ShapeSpheropolyhedron.h"
#include "GPUTree.h"

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
const int MAX_MEMBERS=128;

//! Maximum number of nodes in OBB tree
const unsigned int MAX_UNION_NODES=64;

//! Maximum number of spheres per leaf node
const unsigned int MAX_UNION_CAPACITY=8;

typedef GPUTree<MAX_UNION_NODES,MAX_UNION_CAPACITY> union_gpu_tree_type;

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
    union_gpu_tree_type tree;                //!< OBB tree for constituent shapes
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

template<class Shape>
DEVICE inline bool test_narrow_phase_overlap(vec3<OverlapReal> dr,
                                             const ShapeUnion<Shape>& a,
                                             const ShapeUnion<Shape>& b,
                                             unsigned int cur_node_a,
                                             unsigned int cur_node_b)
    {
    //! Param type of the member shapes
    typedef typename Shape::param_type mparam_type;

    // loop through shape of cur_node_a
    for (unsigned int i= 0; i< detail::union_gpu_tree_type::capacity; i++)
        {
        int ishape = a.members.tree.getParticle(cur_node_a, i);
        if (ishape == -1) break;

        const mparam_type& params_i = a.members.mparams[ishape];
        const quat<Scalar> q_i = a.orientation * a.members.morientation[ishape];
        Shape shape_i(q_i, params_i);

        // loop through shapes of cur_node_b
        for (unsigned int j= 0; j< detail::union_gpu_tree_type::capacity; j++)
            {
            int jshape = b.members.tree.getParticle(cur_node_b, j);
            if (jshape == -1) break;

            const mparam_type& params_j = b.members.mparams[jshape];
            const quat<Scalar> q_j = b.orientation * b.members.morientation[jshape];
            vec3<Scalar> r_ij = vec3<Scalar>(dr) + rotate(b.orientation, b.members.mpos[jshape]) - rotate(a.orientation, a.members.mpos[ishape]);
            Shape shape_j(q_j, params_j);
            unsigned int err =0;
            if (test_overlap(r_ij, shape_i, shape_j, err))
                {
                return true;
                }
            }
        }
    return false;
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
    vec3<Scalar> dr(r_ab);

    unsigned int cur_node_a = 0;
    unsigned int cur_node_b =0;

    unsigned int old_cur_node_a = UINT_MAX;
    unsigned int old_cur_node_b = UINT_MAX;

    unsigned int level_a = 0;
    unsigned int level_b = 0;

    hpmc::detail::OBB obb_a;
    hpmc::detail::OBB obb_b;

    while (cur_node_a < a.members.tree.getNumNodes() && cur_node_b < b.members.tree.getNumNodes())
        {
        if (old_cur_node_a != cur_node_a)
            {
            obb_a = a.members.tree.getOBB(cur_node_a);
            level_a = a.members.tree.getLevel(cur_node_a);

            // rotate and translate a's obb into b's body frame
            obb_a.affineTransform(conj(b.orientation)*a.orientation,
                rotate(conj(b.orientation),-dr));
            old_cur_node_a = cur_node_a;
            }
        if (old_cur_node_b != cur_node_b)
            {
            obb_b = b.members.tree.getOBB(cur_node_b);
            level_b = b.members.tree.getLevel(cur_node_b);
            old_cur_node_b = cur_node_b;
            }

        if (detail::overlap(obb_a, obb_b))
            {
            if (a.members.tree.isLeaf(cur_node_a) && b.members.tree.isLeaf(cur_node_b))
                {
                if (test_narrow_phase_overlap(dr, a, b, cur_node_a, cur_node_b)) return true;
                }
            else
                {
                if (level_a < level_b)
                    {
                    if (a.members.tree.isLeaf(cur_node_a))
                        {
                        unsigned int end_node = cur_node_b;
                        b.members.tree.advanceNode(end_node, true);
                        if (test_subtree(dr, a, b, a.members.tree, b.members.tree, cur_node_a, cur_node_b+1, end_node)) return true;
                        }
                    else
                        {
                        // descend into a's tree
                        cur_node_a = a.members.tree.getLeftChild(cur_node_a);
                        continue;
                        }
                    }
                else
                    {
                    if (b.members.tree.isLeaf(cur_node_b))
                        {
                        unsigned int end_node = cur_node_a;
                        a.members.tree.advanceNode(end_node, true);
                        if (test_subtree(-dr, b, a, b.members.tree, a.members.tree, cur_node_b, cur_node_a+1, end_node)) return true;
                        }
                    else
                        {
                        // descend into b's tree
                        cur_node_b = b.members.tree.getLeftChild(cur_node_b);
                        continue;
                        }
                    }
                }
            } // end if overlap

        // move up in tandem fashion
        detail::moveUp(a.members.tree, cur_node_a, b.members.tree, cur_node_b);
        } // end while
    return false;
    }

} // end namespace hpmc

#endif // end __SHAPE_UNION_H__
