// Copyright (c) 2009-2016 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.

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

//! Data structure for shape composed of a union of multiple shapes
template<class Shape, unsigned int max_n_members, unsigned int capacity, unsigned int max_n_nodes>
struct union_params : aligned_struct
    {
    typedef GPUTree<max_n_nodes,capacity> gpu_tree_type; //!< Handy typedef for GPUTree template
    typedef typename Shape::param_type mparam_type;

    unsigned int N;                          //!< Number of member shapes
    vec3<Scalar> mpos[max_n_members];        //!< Position vectors of member shapes
    quat<Scalar> morientation[max_n_members];//!< Orientation of member shapes
    mparam_type mparams[max_n_members];      //!< Parameters of member shapes
    unsigned int moverlap[max_n_members];    //!< only check overlaps for which moverlap[i] & moverlap[j] 
    OverlapReal diameter;                    //!< Precalculated overall circumsphere diameter
    unsigned int ignore;                     //!<  Bitwise ignore flag for stats. 1 will ignore, 0 will not ignore
    gpu_tree_type tree;                      //!< OBB tree for constituent shapes
    } __attribute__((aligned(32)));

} // end namespace detail

//! Shape consisting of union of shapes of a single type but individual parameters
/*!
    The parameter defining a ShapeUnion is a structure implementing the HPMC shape interface and containing
    parameter objects for its member particles in its own parameters structure

    The purpose of ShapeUnion is to allow an overlap check to iterate through pairs of member shapes between
    two composite particles. The two particles overlap if any of their member shapes overlap.

    ShapeUnion stores an internal OBB tree for fast overlap checks.

    To estimate the maximum number of nodes, we assume that the tree is maximally unbalanced,
    i.e. every second leaf is (almost) empty. Then n_leaf_max = max_n_members/capacity*2 and
    max_n_nodes = n_leaf_max*2-1.
*/
template<class Shape, unsigned int max_n_members=8,
     unsigned int capacity=8,
     unsigned int max_n_nodes=(max_n_members/capacity*2)*2-1 >
struct ShapeUnion
    {
    //! Define the parameter type
    typedef typename detail::union_params<Shape, max_n_members, capacity, max_n_nodes> param_type;

    //! Initialize a sphere_union
    DEVICE ShapeUnion(const quat<Scalar>& _orientation, const param_type& _params)
        : orientation(_orientation), members(_params)
        {
        }

    //! Does this shape have an orientation
    DEVICE bool hasOrientation() const
        {
        if (members.N == 1)
            {
            // if we have only one member in the center, return that shape's anisotropy flag
            const vec3<Scalar>& pos = members.mpos[0];
            if (pos.x == Scalar(0.0) && pos.y == pos.x && pos.z == pos.x)
                {
                Shape s(quat<Scalar>(), members.mparams[0]);
                return s.hasOrientation();
                }
            }

        return true;
        }

    //!Ignore flag for acceptance statistics
    DEVICE bool ignoreStatistics() const { return members.ignore; }

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

    const param_type& members;     //!< member data
    };

//! Check if circumspheres overlap
/*! \param r_ab Vector defining the position of shape b relative to shape a (r_b - r_a)
    \param a first shape
    \param b second shape
    \returns true if the circumspheres of both shapes overlap

    \ingroup shape
*/
template <class Shape, unsigned int max_n_members>
DEVICE inline bool check_circumsphere_overlap(const vec3<Scalar>& r_ab, const ShapeUnion<Shape, max_n_members>& a,
    const ShapeUnion<Shape, max_n_members> &b)
    {
    vec3<OverlapReal> dr(r_ab);

    OverlapReal rsq = dot(dr,dr);
    OverlapReal DaDb = a.getCircumsphereDiameter() + b.getCircumsphereDiameter();
    return (rsq*OverlapReal(4.0) <= DaDb * DaDb);
    }

template<class Shape, unsigned int max_n_members, unsigned int capacity,unsigned int max_n_nodes >
DEVICE inline bool test_narrow_phase_overlap(vec3<OverlapReal> dr,
                                             const ShapeUnion<Shape, max_n_members, capacity, max_n_nodes>& a,
                                             const ShapeUnion<Shape, max_n_members, capacity, max_n_nodes>& b,
                                             unsigned int cur_node_a,
                                             unsigned int cur_node_b)
    {
    vec3<Scalar> r_ab = rotate(conj(b.orientation),vec3<Scalar>(dr));

    //! Param type of the member shapes
    typedef typename Shape::param_type mparam_type;

    // loop through shape of cur_node_a
    for (unsigned int i= 0; i < capacity; i++)
        {
        int ishape = a.members.tree.getParticle(cur_node_a, i);
        if (ishape == -1) break;

        const mparam_type& params_i = a.members.mparams[ishape];
        const quat<Scalar> q_i = conj(b.orientation)*a.orientation * a.members.morientation[ishape];
        Shape shape_i(q_i, params_i);
        vec3<Scalar> pos_i(rotate(conj(b.orientation)*a.orientation,a.members.mpos[ishape])-r_ab);
        unsigned int overlap_i = a.members.moverlap[ishape];

        // loop through shapes of cur_node_b
        for (unsigned int j= 0; j < capacity; j++)
            {
            int jshape = b.members.tree.getParticle(cur_node_b, j);
            if (jshape == -1) break;

            const mparam_type& params_j = b.members.mparams[jshape];
            const quat<Scalar> q_j = b.members.morientation[jshape];
            Shape shape_j(q_j, params_j);
            unsigned int overlap_j = b.members.moverlap[jshape];

            unsigned int err =0;
            if (overlap_i & overlap_j)
                {
                vec3<Scalar> r_ij = b.members.mpos[jshape] - pos_i;
                if (test_overlap(r_ij, shape_i, shape_j, err))
                    {
                    return true;
                    }
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

template <class Shape, unsigned int max_n_members, unsigned int capacity, unsigned int max_n_nodes >
DEVICE inline bool test_overlap(const vec3<Scalar>& r_ab,
                                const ShapeUnion<Shape, max_n_members, capacity, max_n_nodes>& a,
                                const ShapeUnion<Shape, max_n_members, capacity, max_n_nodes>& b,
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
                        if (test_subtree(dr, a, b, a.members.tree, b.members.tree, cur_node_a, cur_node_b, end_node)) return true;
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
                        if (test_subtree(-dr, b, a, b.members.tree, a.members.tree, cur_node_b, cur_node_a, end_node)) return true;
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
