// Copyright (c) 2009-2019 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.

#include "hoomd/HOOMDMath.h"
#include "hoomd/BoxDim.h"
#include "hoomd/VectorMath.h"
#include "ShapeSphere.h"    //< For the base template of test_overlap
#include "ShapeSpheropolyhedron.h"
#include "ShapeConvexPolyhedron.h"
#include "GPUTree.h"

#include "hoomd/AABB.h"
#include "hoomd/ManagedArray.h"

#ifndef __SHAPE_UNION_H__
#define __SHAPE_UNION_H__

/*! \file ShapeUnion.h
    \brief Defines the ShapeUnion templated aggregate shape
*/

// need to declare these class methods with __device__ qualifiers when building in nvcc
// DEVICE is __device__ when included in nvcc and blank when included into the host compiler
#ifdef __HIPCC__
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

//! Stores the overlapping node pairs from a prior traversal
/* This data structure is used to accelerate the random choice of overlapping
   node pairs when depletants are reinserted, eliminating the need to traverse
   the same tree for all reinsertion attempts.
 */
struct union_depletion_storage
    {
    //! The inclusive prefix sum over previous weights of overlapping node pairs
    OverlapReal accumulated_weight;

    //! The node in tree a
    unsigned int cur_node_a;

    //! The node in tree b
    unsigned int cur_node_b;
    };

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
    DEVICE void load_shared(char *& ptr, unsigned int &available_bytes)
        {
        tree.load_shared(ptr, available_bytes);
        mpos.load_shared(ptr, available_bytes);
        bool params_in_shared_mem = mparams.load_shared(ptr, available_bytes);
        moverlap.load_shared(ptr, available_bytes);
        morientation.load_shared(ptr, available_bytes);

        // load all member parameters
        #if defined (__HIP_DEVICE_COMPILE__)
        __syncthreads();
        #endif

        for (unsigned int i = 0; i < mparams.size(); ++i)
            {
            if (params_in_shared_mem)
                {
                // load only if we are sure that we are not touching any unified memory
                mparams[i].load_shared(ptr, available_bytes);
                }
            else
                {
                // increment pointer only
                mparams[i].allocate_shared(ptr, available_bytes);
                }
            }
        }

    //! Determine size of the shared memory allocaation
    /*! \param ptr Pointer to increment
        \param available_bytes Size of remaining shared memory allocation
     */
    HOSTDEVICE void allocate_shared(char *& ptr, unsigned int &available_bytes) const
        {
        tree.allocate_shared(ptr, available_bytes);
        mpos.allocate_shared(ptr, available_bytes);
        mparams.allocate_shared(ptr, available_bytes);
        moverlap.allocate_shared(ptr, available_bytes);
        morientation.allocate_shared(ptr, available_bytes);

        for (unsigned int i = 0; i < mparams.size(); ++i)
            mparams[i].allocate_shared(ptr, available_bytes);
        }


    #ifdef ENABLE_HIP
    //! Set CUDA memory hints
    void set_memory_hint() const
        {
        tree.set_memory_hint();

        mpos.set_memory_hint();
        morientation.set_memory_hint();
        mparams.set_memory_hint();
        moverlap.set_memory_hint();

        // attach member parameters
        for (unsigned int i = 0; i < mparams.size(); ++i)
            mparams[i].set_memory_hint();
        }
    #endif

    #ifndef __HIPCC__
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

    //! Temporary storage for depletant insertion
    typedef struct detail::union_depletion_storage depletion_storage_type;

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

    #ifndef __HIPCC__
    std::string getShapeSpec() const
        {
        throw std::runtime_error("Shape definition not supported for this shape class.");
        }
    #endif

    //! Return the bounding box of the shape in world coordinates
    DEVICE detail::AABB getAABB(const vec3<Scalar>& pos) const
        {
        return getOBB(pos).getAABB();
        }

    //! Return a tight fitting OBB
    DEVICE detail::OBB getOBB(const vec3<Scalar>& pos) const
        {
        if (members.N > 0)
            {
            // get the root node OBB from the tree
            detail::OBB obb = members.tree.getOBB(0);

            // transform it into world-space
            obb.affineTransform(orientation, pos);

            return obb;
            }
        else
            {
            return detail::OBB(pos, OverlapReal(0.5)*members.diameter);
            }
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

template<class Shape>
DEVICE inline bool test_narrow_phase_overlap(vec3<OverlapReal> dr,
                                             const ShapeUnion<Shape>& a,
                                             const ShapeUnion<Shape>& b,
                                             unsigned int cur_node_a,
                                             unsigned int cur_node_b,
                                             unsigned int &err)
    {
    vec3<OverlapReal> r_ab = rotate(conj(quat<OverlapReal>(b.orientation)),vec3<OverlapReal>(dr));

    //! Param type of the member shapes
    typedef typename Shape::param_type mparam_type;

    // loop through shape of cur_node_a
    unsigned int na = a.members.tree.getNumParticles(cur_node_a);
    unsigned int nb = b.members.tree.getNumParticles(cur_node_b);

    for (unsigned int i= 0; i < na; i++)
        {
        unsigned int ishape = a.members.tree.getParticleByNode(cur_node_a, i);

        const mparam_type& params_i = a.members.mparams[ishape];
        Shape shape_i(quat<Scalar>(), params_i);
        if (shape_i.hasOrientation())
            shape_i.orientation = conj(quat<OverlapReal>(b.orientation))*quat<OverlapReal>(a.orientation) * a.members.morientation[ishape];

        vec3<OverlapReal> pos_i(rotate(conj(quat<OverlapReal>(b.orientation))*quat<OverlapReal>(a.orientation),a.members.mpos[ishape])-r_ab);
        unsigned int overlap_i = a.members.moverlap[ishape];

        // loop through shapes of cur_node_b
        for (unsigned int j= 0; j < nb; j++)
            {
            unsigned int jshape = b.members.tree.getParticleByNode(cur_node_b, j);

            const mparam_type& params_j = b.members.mparams[jshape];
            Shape shape_j(quat<Scalar>(), params_j);
            if (shape_j.hasOrientation())
                shape_j.orientation = b.members.morientation[jshape];

            unsigned int overlap_j = b.members.moverlap[jshape];

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
    #ifdef __HIPCC__
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
                if (tree_b.queryNode(obb_a, cur_node_b) &&
                    test_narrow_phase_overlap(r_ab, a, b, cur_node_a, query_node, err))
                    return true;
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
                if (tree_a.queryNode(obb_b, cur_node_a) &&
                    test_narrow_phase_overlap(-r_ab, b, a, cur_node_b, query_node, err))
                    return true;
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

    unsigned int query_node_a = UINT_MAX;
    unsigned int query_node_b = UINT_MAX;

    while (cur_node_a != tree_a.getNumNodes() && cur_node_b != tree_b.getNumNodes())
        {
        query_node_a = cur_node_a;
        query_node_b = cur_node_b;

        if (detail::traverseBinaryStack(tree_a, tree_b, cur_node_a, cur_node_b, stack, obb_a, obb_b, q, dr_rot)
            && test_narrow_phase_overlap(r_ab, a, b, query_node_a, query_node_b, err))
            return true;
        }
    #endif

    return false;
    }

template<class Shape>
DEVICE inline bool test_narrow_phase_excluded_volume_overlap(vec3<OverlapReal> dr,
                                             const ShapeUnion<Shape>& a,
                                             const ShapeUnion<Shape>& b,
                                             unsigned int cur_node_a,
                                             unsigned int cur_node_b,
                                             OverlapReal r,
                                             unsigned int dim)
    {
    vec3<OverlapReal> r_ab = rotate(conj(quat<OverlapReal>(b.orientation)),vec3<OverlapReal>(dr));

    //! Param type of the member shapes
    typedef typename Shape::param_type mparam_type;

    // loop through shape of cur_node_a
    unsigned int na = a.members.tree.getNumParticles(cur_node_a);
    unsigned int nb = b.members.tree.getNumParticles(cur_node_b);

    for (unsigned int i= 0; i < na; i++)
        {
        unsigned int ishape = a.members.tree.getParticleByNode(cur_node_a, i);

        const mparam_type& params_i = a.members.mparams[ishape];
        Shape shape_i(quat<Scalar>(), params_i);
        if (shape_i.hasOrientation())
            shape_i.orientation = conj(quat<OverlapReal>(b.orientation))*quat<OverlapReal>(a.orientation) * a.members.morientation[ishape];

        vec3<OverlapReal> pos_i(rotate(conj(quat<OverlapReal>(b.orientation))*quat<OverlapReal>(a.orientation),a.members.mpos[ishape])-r_ab);

        // loop through shapes of cur_node_b
        for (unsigned int j= 0; j < nb; j++)
            {
            unsigned int jshape = b.members.tree.getParticleByNode(cur_node_b, j);

            const mparam_type& params_j = b.members.mparams[jshape];
            Shape shape_j(quat<Scalar>(), params_j);
            if (shape_j.hasOrientation())
                shape_j.orientation = b.members.morientation[jshape];

            vec3<OverlapReal> r_ij = b.members.mpos[jshape] - pos_i;
            if (excludedVolumeOverlap(shape_i, shape_j, r_ij, r, dim,
                detail::SamplingMethod::accurate))
                {
                return true;
                }
            }
        }
    return false;
    }

//! Test for overlap of excluded volumes
/*! \param shape_a the first shape
    \param shape_b the second shape
    \param r_ab the separation vector between the two shapes (in the same image)
    \param r excluded volume radius
    \param dim the spatial dimension

    returns true if the covering of the intersection is non-empty
 */
template<class Shape>
DEVICE inline bool excludedVolumeOverlap(
    const ShapeUnion<Shape>& a, const ShapeUnion<Shape>& b, const vec3<Scalar>& r_ab,
    OverlapReal r, unsigned int dim, const detail::SamplingMethod::enumAccurate)
    {
    // perform a tandem tree traversal
    const detail::GPUTree& tree_a = a.members.tree;
    const detail::GPUTree& tree_b = b.members.tree;

    unsigned long int stack = 0;
    unsigned int cur_node_a = 0;
    unsigned int cur_node_b = 0;

    vec3<OverlapReal> dr_rot(rotate(conj(b.orientation),-r_ab));
    quat<OverlapReal> q(conj(b.orientation)*a.orientation);

    detail::OBB obb_a = tree_a.getOBB(cur_node_a);
    obb_a.affineTransform(q, dr_rot);

    detail::OBB obb_b = tree_b.getOBB(cur_node_b);

    unsigned int query_node_a = UINT_MAX;
    unsigned int query_node_b = UINT_MAX;

    while (cur_node_a != tree_a.getNumNodes() && cur_node_b != tree_b.getNumNodes())
        {
        // extend OBBs
        if (query_node_a != cur_node_a)
            {
            obb_a.lengths.x += r;
            obb_a.lengths.y += r;
            obb_a.lengths.z += r;
            query_node_a = cur_node_a;
            }

        if (query_node_b != cur_node_b)
            {
            obb_b.lengths.x += r;
            obb_b.lengths.y += r;
            obb_b.lengths.z += r;
            query_node_b = cur_node_b;
            }

        if (detail::traverseBinaryStack(tree_a, tree_b, cur_node_a, cur_node_b, stack, obb_a, obb_b, q, dr_rot)
            && test_narrow_phase_excluded_volume_overlap(r_ab, a, b, query_node_a, query_node_b, r, dim))
            return true;
        }

    return false;
    }

//! Allocate memory for temporary storage in depletant simulations
/*! \param shape_a the first shape
    \param shape_b the second shape
    \param r_ab the separation vector between the two shapes (in the same image)
    \param r excluded volume radius
    \param dim the spatial dimension

    \returns the number of Shape::depletion_storage_type elements requested for
    temporary storage
 */
template<class Shape>
DEVICE inline unsigned int allocateDepletionTemporaryStorage(
    const ShapeUnion<Shape>& a, const ShapeUnion<Shape>& b, const vec3<Scalar>& r_ab,
    OverlapReal r, unsigned int dim, const detail::SamplingMethod::enumAccurate)
    {
    const detail::GPUTree& tree_a = a.members.tree;
    const detail::GPUTree& tree_b = b.members.tree;

    unsigned long int stack = 0;
    unsigned int cur_node_a = 0;
    unsigned int cur_node_b = 0;

    vec3<OverlapReal> dr_rot(rotate(conj(b.orientation),-r_ab));
    quat<OverlapReal> q(conj(b.orientation)*a.orientation);

    detail::OBB obb_a = tree_a.getOBB(cur_node_a);
    obb_a.affineTransform(q, dr_rot);

    detail::OBB obb_b = tree_b.getOBB(cur_node_b);

    unsigned int query_node_a = UINT_MAX;
    unsigned int query_node_b = UINT_MAX;

    unsigned int nelem = 0;

    while (cur_node_a != tree_a.getNumNodes() && cur_node_b != tree_b.getNumNodes())
        {
        // extend OBBs
        if (query_node_a != cur_node_a)
            {
            obb_a.lengths.x += r;
            obb_a.lengths.y += r;
            obb_a.lengths.z += r;
            query_node_a = cur_node_a;
            }

        if (query_node_b != cur_node_b)
            {
            obb_b.lengths.x += r;
            obb_b.lengths.y += r;
            obb_b.lengths.z += r;
            query_node_b = cur_node_b;
            }

        if (detail::traverseBinaryStack(tree_a, tree_b, cur_node_a, cur_node_b, stack, obb_a, obb_b, q, dr_rot)
            && test_narrow_phase_excluded_volume_overlap(r_ab, a, b, query_node_a, query_node_b, r, dim))
            {
            // count number of overlapping pairs
            nelem++;
            }
        }

    return nelem;
    }

template<class Shape>
DEVICE inline OverlapReal sampling_volume_narrow_phase(vec3<OverlapReal> dr,
                                             const ShapeUnion<Shape>& a,
                                             const ShapeUnion<Shape>& b,
                                             unsigned int cur_node_a,
                                             unsigned int cur_node_b,
                                             OverlapReal r,
                                             unsigned int dim)
    {
    vec3<OverlapReal> r_ab = rotate(conj(quat<OverlapReal>(b.orientation)),vec3<OverlapReal>(dr));

    //! Param type of the member shapes
    typedef typename Shape::param_type mparam_type;

    // loop through shape of cur_node_a
    unsigned int na = a.members.tree.getNumParticles(cur_node_a);
    unsigned int nb = b.members.tree.getNumParticles(cur_node_b);

    OverlapReal V(0.0);

    for (unsigned int i= 0; i < na; i++)
        {
        unsigned int ishape = a.members.tree.getParticleByNode(cur_node_a, i);

        const mparam_type& params_i = a.members.mparams[ishape];
        Shape shape_i(quat<Scalar>(), params_i);
        if (shape_i.hasOrientation())
            shape_i.orientation = conj(quat<OverlapReal>(b.orientation))*quat<OverlapReal>(a.orientation) * a.members.morientation[ishape];

        vec3<OverlapReal> pos_i(rotate(conj(quat<OverlapReal>(b.orientation))*quat<OverlapReal>(a.orientation),a.members.mpos[ishape])-r_ab);

        // loop through shapes of cur_node_b
        for (unsigned int j= 0; j < nb; j++)
            {
            unsigned int jshape = b.members.tree.getParticleByNode(cur_node_b, j);

            const mparam_type& params_j = b.members.mparams[jshape];
            Shape shape_j(quat<Scalar>(), params_j);
            if (shape_j.hasOrientation())
                shape_j.orientation = b.members.morientation[jshape];

            vec3<OverlapReal> r_ij = b.members.mpos[jshape] - pos_i;
            if (excludedVolumeOverlap(shape_i, shape_j, r_ij, r, dim,
                detail::SamplingMethod::accurate))
                {
                V += getSamplingVolumeIntersection(shape_i, shape_j, r_ij, r, dim,
                    detail::SamplingMethod::accurate);
                }
            }
        }
    return V;
    }


//! Initialize temporary storage in depletant simulations
/*! \param shape_a the first shape
    \param shape_b the second shape
    \param r_ab the separation vector between the two shapes (in the same image)
    \param r excluded volume radius
    \param dim the spatial dimension
    \param storage a pointer to a pre-allocated memory region, the size of which has been
        determined by a call to allocateDepletionTemporaryStorage
    \param V_sample the insertion volume
        V_sample has to to be precomputed for the overlapping shapes using
        getSamplingVolumeIntersection()

    \returns the number of Shape::depletion_storage_type elements initialized
 */
template<class Shape>
DEVICE inline unsigned int initializeDepletionTemporaryStorage(
    const ShapeUnion<Shape>& a, const ShapeUnion<Shape>& b, const vec3<Scalar>& r_ab,
    OverlapReal r, unsigned int dim, detail::union_depletion_storage *storage,
    const OverlapReal V_sample, const detail::SamplingMethod::enumAccurate)
    {
    const detail::GPUTree& tree_a = a.members.tree;
    const detail::GPUTree& tree_b = b.members.tree;

    unsigned long int stack = 0;
    unsigned int cur_node_a = 0;
    unsigned int cur_node_b = 0;

    vec3<OverlapReal> dr_rot(rotate(conj(b.orientation),-r_ab));
    quat<OverlapReal> q(conj(b.orientation)*a.orientation);

    detail::OBB obb_a = tree_a.getOBB(cur_node_a);
    obb_a.affineTransform(q, dr_rot);

    detail::OBB obb_b = tree_b.getOBB(cur_node_b);

    unsigned int query_node_a = UINT_MAX;
    unsigned int query_node_b = UINT_MAX;

    unsigned int nelem = 0;
    OverlapReal V_sum(0.0);

    while (cur_node_a != tree_a.getNumNodes() && cur_node_b != tree_b.getNumNodes())
        {
        // extend OBBs
        if (query_node_a != cur_node_a)
            {
            obb_a.lengths.x += r;
            obb_a.lengths.y += r;
            obb_a.lengths.z += r;
            query_node_a = cur_node_a;
            }

        if (query_node_b != cur_node_b)
            {
            obb_b.lengths.x += r;
            obb_b.lengths.y += r;
            obb_b.lengths.z += r;
            query_node_b = cur_node_b;
            }

        if (detail::traverseBinaryStack(tree_a, tree_b, cur_node_a, cur_node_b, stack, obb_a, obb_b, q, dr_rot)
            && test_narrow_phase_excluded_volume_overlap(r_ab, a, b, query_node_a, query_node_b, r, dim))
            {
            V_sum += sampling_volume_narrow_phase(r_ab, a, b, query_node_a, query_node_b, r, dim);
            detail::union_depletion_storage elem;
            elem.accumulated_weight = V_sum/V_sample;
            elem.cur_node_a = query_node_a;
            elem.cur_node_b = query_node_b;
            storage[nelem++] = elem;
            }
        }

    return nelem;
    }

//! Get the sampling volume for an intersection of shapes
/*! \param shape_a the first shape
    \param shape_b the second shape
    \param r_ab the separation vector between the two shapes (in the same image)
    \param r excluded volume radius
    \param p the returned point
    \param dim the spatial dimension

    If the shapes are not overlapping, return zero

    returns the volume of the intersection
 */
template<class Shape>
DEVICE inline OverlapReal getSamplingVolumeIntersection(
    const ShapeUnion<Shape>& a, const ShapeUnion<Shape>& b, const vec3<Scalar>& r_ab,
    OverlapReal r, unsigned int dim, const detail::SamplingMethod::enumAccurate)
    {
    // perform a tandem tree traversal
    const detail::GPUTree& tree_a = a.members.tree;
    const detail::GPUTree& tree_b = b.members.tree;

    unsigned long int stack = 0;
    unsigned int cur_node_a = 0;
    unsigned int cur_node_b = 0;

    vec3<OverlapReal> dr_rot(rotate(conj(b.orientation),-r_ab));
    quat<OverlapReal> q(conj(b.orientation)*a.orientation);

    detail::OBB obb_a = tree_a.getOBB(cur_node_a);
    obb_a.affineTransform(q, dr_rot);

    detail::OBB obb_b = tree_b.getOBB(cur_node_b);

    unsigned int query_node_a = UINT_MAX;
    unsigned int query_node_b = UINT_MAX;

    OverlapReal V_sample(0.0);
    while (cur_node_a != tree_a.getNumNodes() && cur_node_b != tree_b.getNumNodes())
        {
        // extend OBBs
        if (query_node_a != cur_node_a)
            {
            obb_a.lengths.x += r;
            obb_a.lengths.y += r;
            obb_a.lengths.z += r;
            query_node_a = cur_node_a;
            }

        if (query_node_b != cur_node_b)
            {
            obb_b.lengths.x += r;
            obb_b.lengths.y += r;
            obb_b.lengths.z += r;
            query_node_b = cur_node_b;
            }

        if (detail::traverseBinaryStack(tree_a, tree_b, cur_node_a, cur_node_b, stack, obb_a, obb_b, q, dr_rot))
            {
            V_sample += sampling_volume_narrow_phase(r_ab, a, b, query_node_a, query_node_b, r, dim);
            }
        }
    return V_sample;
    }

template<class RNG, class Shape>
DEVICE inline bool sample_narrow_phase(RNG &rng,
                                       vec3<OverlapReal> dr,
                                       const ShapeUnion<Shape>& a,
                                       const ShapeUnion<Shape>& b,
                                       unsigned int cur_node_a,
                                       unsigned int cur_node_b,
                                       OverlapReal r,
                                       vec3<OverlapReal>& p,
                                       unsigned int dim)
    {
    OverlapReal V_sample = sampling_volume_narrow_phase(dr, a, b, cur_node_a, cur_node_b, r, dim);
    OverlapReal u = hoomd::UniformDistribution<OverlapReal>(OverlapReal(0.0),V_sample)(rng);

    OverlapReal V_sum(0.0);

    //! Param type of the member shapes
    typedef typename Shape::param_type mparam_type;

    // loop through shape of cur_node_a
    unsigned int na = a.members.tree.getNumParticles(cur_node_a);
    unsigned int nb = b.members.tree.getNumParticles(cur_node_b);

    unsigned int ishape, jshape;

    bool done = false;
    unsigned int i, j;

    vec3<OverlapReal> r_ab = rotate(conj(quat<OverlapReal>(b.orientation)),vec3<OverlapReal>(dr));

    OverlapReal V;
    for (i= 0; i < na; i++)
        {
        ishape = a.members.tree.getParticleByNode(cur_node_a, i);

        const mparam_type& params_i = a.members.mparams[ishape];
        Shape shape_i(quat<Scalar>(), params_i);
        if (shape_i.hasOrientation())
            shape_i.orientation = conj(quat<OverlapReal>(b.orientation))*quat<OverlapReal>(a.orientation) * a.members.morientation[ishape];

        vec3<OverlapReal> pos_i(rotate(conj(quat<OverlapReal>(b.orientation))*quat<OverlapReal>(a.orientation),a.members.mpos[ishape])-r_ab);

        // loop through shapes of cur_node_b
        for (j= 0; j < nb; j++)
            {
            jshape = b.members.tree.getParticleByNode(cur_node_b, j);

            const mparam_type& params_j = b.members.mparams[jshape];
            Shape shape_j(quat<Scalar>(), params_j);
            if (shape_j.hasOrientation())
                shape_j.orientation = b.members.morientation[jshape];

            vec3<OverlapReal> r_ij = b.members.mpos[jshape] - pos_i;
            if (excludedVolumeOverlap(shape_i, shape_j, r_ij, r, dim,
                detail::SamplingMethod::accurate))
                {
                V = getSamplingVolumeIntersection(shape_i, shape_j, r_ij, r, dim,
                    detail::SamplingMethod::accurate);
                V_sum += V;

                if (u < V_sum)
                    {
                    done = true;
                    break;
                    }
                }
            }

        if (done)
            break;
        }

    if (!done)
        return false;

    // get point in space frame, with a at the origin
    Shape shape_i(a.orientation*quat<Scalar>(a.members.morientation[ishape]), a.members.mparams[ishape]);
    Shape shape_j(b.orientation*quat<Scalar>(b.members.morientation[jshape]), b.members.mparams[jshape]);
    vec3<OverlapReal> pos_i = rotate(quat<OverlapReal>(a.orientation), a.members.mpos[ishape]);
    vec3<Scalar> r_ij = rotate(quat<OverlapReal>(b.orientation), b.members.mpos[jshape]) + dr - pos_i;

    // set up temp storage on stack / in local memory
    unsigned int ntemp = allocateDepletionTemporaryStorage(shape_i, shape_j, r_ij, r, dim,
        detail::SamplingMethod::accurate);
    typename Shape::depletion_storage_type temp[ntemp];
    unsigned int nelem = initializeDepletionTemporaryStorage(shape_i, shape_j, r_ij, r, dim,
        temp, V, detail::SamplingMethod::accurate);

    // sample
    if (!sampleInExcludedVolumeIntersection(rng, shape_i, shape_j, r_ij, r, p, dim, nelem,
        temp, detail::SamplingMethod::accurate))
        return false;

    p += pos_i;

    unsigned int min_i = i;
    unsigned int min_j = j;

    // test if it is overlapping with other shapes with lower indices
    for (i = 0; i <= min_i; i++)
        {
        unsigned int ishape = a.members.tree.getParticleByNode(cur_node_a, i);

        const mparam_type& params_i = a.members.mparams[ishape];
        Shape shape_i(quat<Scalar>(), params_i);
        if (shape_i.hasOrientation())
            shape_i.orientation = conj(quat<OverlapReal>(b.orientation))*quat<OverlapReal>(a.orientation) * a.members.morientation[ishape];

        vec3<OverlapReal> pos_i(rotate(conj(quat<OverlapReal>(b.orientation))*quat<OverlapReal>(a.orientation),a.members.mpos[ishape])-r_ab);

        // loop through shapes of cur_node_b
        for (j= 0; j < ((i == min_i) ? min_j : nb); j++)
            {
            unsigned int jshape = b.members.tree.getParticleByNode(cur_node_b, j);

            const mparam_type& params_j = b.members.mparams[jshape];
            Shape shape_j(quat<Scalar>(), params_j);
            if (shape_j.hasOrientation())
                shape_j.orientation = b.members.morientation[jshape];

            vec3<OverlapReal> r_ij = b.members.mpos[jshape] - pos_i;
            if (excludedVolumeOverlap(shape_i, shape_j, r_ij, r, dim, detail::SamplingMethod::accurate))
                {
                // shift origin to ishape's position, test in space frame
                Shape shape_i_world(a.orientation*quat<Scalar>(a.members.morientation[ishape]), params_i);
                Shape shape_j_world(b.orientation*quat<Scalar>(b.members.morientation[jshape]), params_j);

                vec3<OverlapReal> pos_i = rotate(quat<OverlapReal>(a.orientation), a.members.mpos[ishape]);
                vec3<OverlapReal> r_ij_world = rotate(quat<OverlapReal>(b.orientation), b.members.mpos[jshape]) + dr - pos_i;
                vec3<OverlapReal> q = p - pos_i;

                if (isPointInExcludedVolumeIntersection(shape_i_world,
                    shape_j_world, r_ij_world, r, q, dim, detail::SamplingMethod::accurate))
                    return false;
                }
            }
        }
    return true;
    }

template<class Shape>
DEVICE inline bool pt_in_intersection_narrow_phase(vec3<OverlapReal> dr,
                                       const ShapeUnion<Shape>& a,
                                       const ShapeUnion<Shape>& b,
                                       unsigned int cur_node_a,
                                       unsigned int cur_node_b,
                                       OverlapReal r,
                                       const vec3<OverlapReal>& p,
                                       unsigned int dim)
    {
    vec3<OverlapReal> r_ab = rotate(conj(quat<OverlapReal>(b.orientation)),vec3<OverlapReal>(dr));

    //! Param type of the member shapes
    typedef typename Shape::param_type mparam_type;

    // loop through shape of cur_node_a
    unsigned int na = a.members.tree.getNumParticles(cur_node_a);
    unsigned int nb = b.members.tree.getNumParticles(cur_node_b);

    for (unsigned int i= 0; i < na; i++)
        {
        unsigned int ishape = a.members.tree.getParticleByNode(cur_node_a, i);

        const mparam_type& params_i = a.members.mparams[ishape];
        Shape shape_i(quat<Scalar>(), params_i);
        if (shape_i.hasOrientation())
            shape_i.orientation = conj(quat<OverlapReal>(b.orientation))*quat<OverlapReal>(a.orientation) * a.members.morientation[ishape];

        vec3<OverlapReal> pos_i(rotate(conj(quat<OverlapReal>(b.orientation))*quat<OverlapReal>(a.orientation),a.members.mpos[ishape])-r_ab);

        // loop through shapes of cur_node_b
        for (unsigned int j= 0; j < nb; j++)
            {
            unsigned int jshape = b.members.tree.getParticleByNode(cur_node_b, j);

            const mparam_type& params_j = b.members.mparams[jshape];
            Shape shape_j(quat<Scalar>(), params_j);
            if (shape_j.hasOrientation())
                shape_j.orientation = b.members.morientation[jshape];

            vec3<OverlapReal> r_ij = b.members.mpos[jshape] - pos_i;
            if (excludedVolumeOverlap(shape_i, shape_j, r_ij, r, dim, detail::SamplingMethod::accurate))
                {
                // shift origin to ihape's position, test in space frame
                Shape shape_i_world(a.orientation*quat<Scalar>(a.members.morientation[ishape]), params_i);
                Shape shape_j_world(b.orientation*quat<Scalar>(b.members.morientation[jshape]), params_j);

                vec3<OverlapReal> pos_i = rotate(quat<OverlapReal>(a.orientation), a.members.mpos[ishape]);
                vec3<OverlapReal> r_ij_world = rotate(quat<OverlapReal>(b.orientation), b.members.mpos[jshape]) + dr - pos_i;
                vec3<OverlapReal> q = p - pos_i;

                if (isPointInExcludedVolumeIntersection(
                    shape_i_world, shape_j_world, r_ij_world, r, q, dim,
                    detail::SamplingMethod::accurate))
                    return true;
                }
            }
        }

    return false;
    }

//! Uniform rejection sampling in a volume covering the intersection of two shapes, defined by their Minkowski sums with a sphere of radius r
/*! \param rng random number generator
    \param shape_a the first shape
    \param shape_b the second shape
    \param r_ab the separation vector between the two shapes (in the same image)
    \param r excluded volume radius
    \param p the returned point (relative to the origin == shape_a)
    \param dim the spatial dimension
    \param storage_sz the number of temporary storage elements of type
        Shape::depletion_storage_type passed
    \param storage the array of temporary storage elements

    returns true if the point was not rejected
 */
template<class RNG, class Shape>
DEVICE inline bool sampleInExcludedVolumeIntersection(
    RNG& rng, const ShapeUnion<Shape>& a, const ShapeUnion<Shape>& b, const vec3<Scalar>& r_ab,
    OverlapReal r, vec3<OverlapReal>& p, unsigned int dim,
    unsigned int storage_sz, const detail::union_depletion_storage *storage,
    const detail::SamplingMethod::enumAccurate)
    {
    // perform a tandem tree traversal
    const detail::GPUTree& tree_a = a.members.tree;
    const detail::GPUTree& tree_b = b.members.tree;

    OverlapReal u = hoomd::UniformDistribution<OverlapReal>()(rng);
    bool found = false;
    unsigned int query_node_a;
    unsigned int query_node_b;

    if (storage_sz > 0)
        {
        // binary search for the the first element with accumulated weight > u
        unsigned int l = 0;
        unsigned int r = storage_sz-1;
        while (l <= r)
            {
            unsigned int m = l + (r-l)/2;

            if (storage[m].accumulated_weight > u &&
                (m == 0 || storage[m-1].accumulated_weight <= u))
                {
                query_node_a = storage[m].cur_node_a;
                query_node_b = storage[m].cur_node_b;
                found = true;
                break;
                }

            if (storage[m].accumulated_weight <= u)
                l = m + 1;
            else
                r = m - 1;
            }
        }

    if (!found)
        return false;

    if (!sample_narrow_phase(rng, r_ab, a, b, query_node_a, query_node_b, r, p, dim))
        return false;

    // test if point is in other volumes lying 'below' the current one
    vec3<OverlapReal> dr_rot(rotate(conj(b.orientation),-r_ab));
    quat<OverlapReal> q(conj(b.orientation)*a.orientation);

    unsigned int cur_node_a = 0;
    unsigned int cur_node_b = 0;
    unsigned long int stack = 0;
    detail::OBB obb_a = tree_a.getOBB(cur_node_a);
    obb_a.affineTransform(q, dr_rot);
    detail::OBB obb_b = tree_b.getOBB(cur_node_b);

    detail::OBB obb_query(p, r);
    obb_query.affineTransform(conj(b.orientation), dr_rot);

    unsigned int min_query_node_a = query_node_a;
    unsigned int min_query_node_b = query_node_b;

    while (cur_node_a != tree_a.getNumNodes() && cur_node_b != tree_b.getNumNodes())
        {
        // extend OBBs
        if (query_node_a != cur_node_a)
            {
            obb_a.lengths.x += r;
            obb_a.lengths.y += r;
            obb_a.lengths.z += r;
            query_node_a = cur_node_a;
            }

        if (query_node_b != cur_node_b)
            {
            obb_b.lengths.x += r;
            obb_b.lengths.y += r;
            obb_b.lengths.z += r;
            query_node_b = cur_node_b;
            }

        if (detail::traverseBinaryStackIntersection(tree_a, tree_b, cur_node_a, cur_node_b, stack,
                obb_a, obb_b, q, dr_rot, obb_query) &&
            (query_node_a < min_query_node_a || (query_node_a == min_query_node_a && query_node_b < min_query_node_b)) &&
            pt_in_intersection_narrow_phase(r_ab, a, b, query_node_a, query_node_b, r, p, dim))
            {
            return false;
            }
        }

    return true;
    }


//! Test if a point is in the intersection of two excluded volumes
/*! \param shape_a the first shape
    \param shape_b the second shape
    \param r_ab the separation vector between the two shapes (in the same image)
    \param r excluded volume radius
    \param p the point to test (relative to the origin == shape_a)
    \param dim the spatial dimension

    returns true if the point was not rejected
 */
template<class Shape>
DEVICE inline bool isPointInExcludedVolumeIntersection(
    const ShapeUnion<Shape>& a, const ShapeUnion<Shape>& b, const vec3<Scalar>& r_ab,
    OverlapReal r, const vec3<OverlapReal>& p, unsigned int dim,
    const detail::SamplingMethod::enumAccurate)
    {
    // perform a tandem tree traversal
    const detail::GPUTree& tree_a = a.members.tree;
    const detail::GPUTree& tree_b = b.members.tree;

    unsigned long int stack = 0;
    unsigned int cur_node_a = 0;
    unsigned int cur_node_b = 0;

    vec3<OverlapReal> dr_rot(rotate(conj(b.orientation),-r_ab));
    quat<OverlapReal> q(conj(b.orientation)*a.orientation);

    detail::OBB obb_a = tree_a.getOBB(cur_node_a);
    obb_a.affineTransform(q, dr_rot);

    detail::OBB obb_b = tree_b.getOBB(cur_node_b);

    unsigned int query_node_a = UINT_MAX;
    unsigned int query_node_b = UINT_MAX;

    detail::OBB obb_query(p, r);
    obb_query.affineTransform(conj(b.orientation), dr_rot);

    while (cur_node_a != tree_a.getNumNodes() && cur_node_b != tree_b.getNumNodes())
        {
        // extend OBBs
        if (query_node_a != cur_node_a)
            {
            obb_a.lengths.x += r;
            obb_a.lengths.y += r;
            obb_a.lengths.z += r;
            query_node_a = cur_node_a;
            }

        if (query_node_b != cur_node_b)
            {
            obb_b.lengths.x += r;
            obb_b.lengths.y += r;
            obb_b.lengths.z += r;
            query_node_b = cur_node_b;
            }

        if (detail::traverseBinaryStackIntersection(tree_a, tree_b, cur_node_a, cur_node_b, stack,
                obb_a, obb_b, q, dr_rot, obb_query) &&
            pt_in_intersection_narrow_phase(r_ab, a, b, query_node_a, query_node_b, r, p, dim))
            {
            return true;
            }
        }

    return false;
    }

}; // end namespace hpmc

#undef DEVICE
#undef HOSTDEVICE
#endif // end __SHAPE_UNION_H__
