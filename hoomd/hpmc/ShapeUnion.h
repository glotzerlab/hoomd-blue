// Copyright (c) 2009-2019 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.

#include "hoomd/HOOMDMath.h"
#include "hoomd/BoxDim.h"
#include "hoomd/VectorMath.h"
#include "ShapeSphere.h"    //< For the base template of test_overlap
#include "ShapeSpheropolyhedron.h"
#include "ShapeConvexPolyhedron.h"
#include "GPUTree.h"

#include "hoomd/ManagedArray.h"

#ifndef __SHAPE_UNION_H__
#define __SHAPE_UNION_H__

/*! \file ShapeUnion.h
    \brief Defines the ShapeUnion templated aggregate shape
*/

// need to declare these class methods with __device__ qualifiers when building in nvcc
// DEVICE is __device__ when included in nvcc and blank when included into the host compiler
#ifdef NVCC
#define DEVICE __device__
#define HOSTDEVICE __host__ __device__
#else
#define DEVICE
#define HOSTDEVICE
#include <iostream>
#endif

//#define SHAPE_UNION_LEAVES_AGAINST_TREE_TRAVERSAL

namespace hpmc
{

namespace detail
{

//! Data structure for shape composed of a union of multiple shapes
template<class Shape>
struct union_params : param_base
    {
    typedef GPUTree gpu_tree_type; //!< Handy typedef for GPUTree template
    typedef typename Shape::param_type mparam_type;

    //! Default constructor
    DEVICE union_params()
        : diameter(0.0), N(0), ignore(0)
        { }

    //! Load dynamic data members into shared memory and increase pointer
    /*! \param ptr Pointer to load data to (will be incremented)
        \param available_bytes Size of remaining shared memory allocation
     */
    HOSTDEVICE void load_shared(char *& ptr, unsigned int &available_bytes) const
        {
        tree.load_shared(ptr, available_bytes);
        mpos.load_shared(ptr, available_bytes);
        mparams.load_shared(ptr, available_bytes);
        moverlap.load_shared(ptr, available_bytes);
        morientation.load_shared(ptr, available_bytes);
        }

    #ifdef ENABLE_CUDA
    //! Attach managed memory to CUDA stream
    void attach_to_stream(cudaStream_t stream) const
        {
        // attach managed memory arrays to stream
        tree.attach_to_stream(stream);

        mpos.attach_to_stream(stream);
        morientation.attach_to_stream(stream);
        mparams.attach_to_stream(stream);
        moverlap.attach_to_stream(stream);
        }
    #endif

    #ifndef NVCC
    //! Shape constructor
    union_params(unsigned int _N, bool _managed)
        : N(_N)
        {
        mpos = ManagedArray<vec3<OverlapReal> >(N,_managed);
        morientation = ManagedArray<quat<OverlapReal> >(N,_managed);
        mparams = ManagedArray<mparam_type>(N,_managed);
        moverlap = ManagedArray<unsigned int>(N,_managed);
        }
    #endif

    gpu_tree_type tree;                      //!< OBB tree for constituent shapes
    ManagedArray<vec3<OverlapReal> > mpos;         //!< Position vectors of member shapes
    ManagedArray<quat<OverlapReal> > morientation; //!< Orientation of member shapes
    ManagedArray<mparam_type> mparams;        //!< Parameters of member shapes
    ManagedArray<unsigned int> moverlap;      //!< only check overlaps for which moverlap[i] & moverlap[j]
    OverlapReal diameter;                    //!< Precalculated overall circumsphere diameter
    unsigned int N;                           //!< Number of member shapes
    unsigned int ignore;                     //!<  Bitwise ignore flag for stats. 1 will ignore, 0 will not ignore
    } __attribute__((aligned(32)));

} // end namespace detail

//! Shape consisting of union of shapes of a single type but individual parameters
/*!
    The parameter defining a ShapeUnion is a structure implementing the HPMC shape interface and containing
    parameter objects for its member particles in its own parameters structure

    The purpose of ShapeUnion is to allow an overlap check to iterate through pairs of member shapes between
    two composite particles. The two particles overlap if any of their member shapes overlap.

    ShapeUnion stores an internal OBB tree for fast overlap checks.
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

    #ifndef NVCC
    std::string getShapeSpec() const
        {
        throw std::runtime_error("Shape definition not supported for this shape class.");
        }
    #endif

    //! Return the bounding box of the shape in world coordinates
    DEVICE detail::AABB getAABB(const vec3<Scalar>& pos) const
        {
        return detail::AABB(pos, members.diameter/OverlapReal(2.0));
        }

    //! Returns true if this shape splits the overlap check over several threads of a warp using threadIdx.x
    HOSTDEVICE static bool isParallel() {
        #ifdef SHAPE_UNION_LEAVES_AGAINST_TREE_TRAVERSAL
        return true;
        #else
        return false;
        #endif
        }

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
    vec3<OverlapReal> r_ab = rotate(conj(quat<OverlapReal>(b.orientation)),vec3<OverlapReal>(dr));

    //! Param type of the member shapes
    typedef typename Shape::param_type mparam_type;

    // loop through shape of cur_node_a
    unsigned int na = a.members.tree.getNumParticles(cur_node_a);
    unsigned int nb = b.members.tree.getNumParticles(cur_node_b);

    for (unsigned int i= 0; i < na; i++)
        {
        unsigned int ishape = a.members.tree.getParticle(cur_node_a, i);

        const mparam_type& params_i = a.members.mparams[ishape];
        Shape shape_i(quat<Scalar>(), params_i);
        if (shape_i.hasOrientation())
            shape_i.orientation = conj(quat<OverlapReal>(b.orientation))*quat<OverlapReal>(a.orientation) * a.members.morientation[ishape];

        vec3<OverlapReal> pos_i(rotate(conj(quat<OverlapReal>(b.orientation))*quat<OverlapReal>(a.orientation),a.members.mpos[ishape])-r_ab);
        unsigned int overlap_i = a.members.moverlap[ishape];

        // loop through shapes of cur_node_b
        for (unsigned int j= 0; j < nb; j++)
            {
            unsigned int jshape = b.members.tree.getParticle(cur_node_b, j);

            const mparam_type& params_j = b.members.mparams[jshape];
            Shape shape_j(quat<Scalar>(), params_j);
            if (shape_j.hasOrientation())
                shape_j.orientation = b.members.morientation[jshape];

            unsigned int overlap_j = b.members.moverlap[jshape];

            unsigned int err =0;
            if (overlap_i & overlap_j)
                {
                vec3<OverlapReal> r_ij = b.members.mpos[jshape] - pos_i;
                if (test_overlap(r_ij, shape_i, shape_j, err))
                    {
                    return true;
                    }
                }
            }
        }
    return false;
    }

template <class Shape >
DEVICE inline bool test_overlap(const vec3<Scalar>& r_ab,
                                const ShapeUnion<Shape>& a,
                                const ShapeUnion<Shape>& b,
                                unsigned int& err)
    {
    const detail::GPUTree& tree_a = a.members.tree;
    const detail::GPUTree& tree_b = b.members.tree;

    #ifdef SHAPE_UNION_LEAVES_AGAINST_TREE_TRAVERSAL
    #ifdef NVCC
    // Parallel tree traversal
    unsigned int offset = threadIdx.x;
    unsigned int stride = blockDim.x;
    #else
    unsigned int offset = 0;
    unsigned int stride = 1;
    #endif

    if (tree_a.getNumLeaves() <= tree_b.getNumLeaves())
        {
        for (unsigned int cur_leaf_a = offset; cur_leaf_a < tree_a.getNumLeaves(); cur_leaf_a += stride)
            {
            unsigned int cur_node_a = tree_a.getLeafNode(cur_leaf_a);
            hpmc::detail::OBB obb_a = tree_a.getOBB(cur_node_a);
            // rotate and translate a's obb into b's body frame
            obb_a.affineTransform(conj(b.orientation)*a.orientation,
                rotate(conj(b.orientation),-r_ab));

            unsigned cur_node_b = 0;
            while (cur_node_b < tree_b.getNumNodes())
                {
                unsigned int query_node = cur_node_b;
                if (tree_b.queryNode(obb_a, cur_node_b) && test_narrow_phase_overlap(r_ab, a, b, cur_node_a, query_node)) return true;
                }
            }
        }
    else
        {
        for (unsigned int cur_leaf_b = offset; cur_leaf_b < tree_b.getNumLeaves(); cur_leaf_b += stride)
            {
            unsigned int cur_node_b = tree_b.getLeafNode(cur_leaf_b);
            hpmc::detail::OBB obb_b = tree_b.getOBB(cur_node_b);

            // rotate and translate b's obb into a's body frame
            obb_b.affineTransform(conj(a.orientation)*b.orientation,
                rotate(conj(a.orientation),r_ab));

            unsigned cur_node_a = 0;
            while (cur_node_a < tree_a.getNumNodes())
                {
                unsigned int query_node = cur_node_a;
                if (tree_a.queryNode(obb_b, cur_node_a) && test_narrow_phase_overlap(-r_ab, b, a, cur_node_b, query_node)) return true;
                }
            }
        }
    #else
    // perform a tandem tree traversal
    unsigned long int stack = 0;
    unsigned int cur_node_a = 0;
    unsigned int cur_node_b = 0;

    vec3<OverlapReal> dr_rot(rotate(conj(b.orientation),-r_ab));
    quat<OverlapReal> q(conj(b.orientation)*a.orientation);

    detail::OBB obb_a = tree_a.getOBB(cur_node_a);
    obb_a.affineTransform(q, dr_rot);

    detail::OBB obb_b = tree_b.getOBB(cur_node_b);

    while (cur_node_a != tree_a.getNumNodes() && cur_node_b != tree_b.getNumNodes())
        {
        unsigned int query_node_a = cur_node_a;
        unsigned int query_node_b = cur_node_b;

        if (detail::traverseBinaryStack(tree_a, tree_b, cur_node_a, cur_node_b, stack, obb_a, obb_b, q, dr_rot)
            && test_narrow_phase_overlap(r_ab, a, b, query_node_a, query_node_b)) return true;
        }
    #endif

    return false;
    }

} // end namespace hpmc

#undef DEVICE
#undef HOSTDEVICE
#endif // end __SHAPE_UNION_H__
