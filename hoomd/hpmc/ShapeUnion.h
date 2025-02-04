// Copyright (c) 2009-2024 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

#pragma once

#include "GPUTree.h"
#include "ShapeConvexPolyhedron.h"
#include "ShapeSphere.h" //< For the base template of test_overlap
#include "ShapeSpheropolyhedron.h"
#include "hoomd/BoxDim.h"
#include "hoomd/HOOMDMath.h"
#include "hoomd/VectorMath.h"

#include "hoomd/AABB.h"
#include "hoomd/ManagedArray.h"
#ifdef __HIPCC__
#define DEVICE __device__
#define HOSTDEVICE __host__ __device__
#else
#define DEVICE
#define HOSTDEVICE
#include <iostream>
#endif

namespace hoomd
    {
namespace hpmc
    {
namespace detail
    {
/// Stores the overlapping node pairs from a prior traversal
/* This data structure is used to accelerate the random choice of overlapping
   node pairs when depletants are reinserted, eliminating the need to traverse
   the same tree for all reinsertion attempts.
 */
struct union_depletion_storage
    {
    //! The inclusive prefix sum over previous weights of overlapping node pairs
    ShortReal accumulated_weight;

    //! The node in tree a
    unsigned int cur_node_a;

    //! The node in tree b
    unsigned int cur_node_b;
    };

/** Data structure for shape composed of a union of multiple shapes.

    Store N member shapes of the same type at given positions and orientations relative to the
    position and orientation of the parent shape. Use ManagedArray to support shape data types that
    have nested ManagedArray members.
*/
template<class Shape> struct ShapeUnionParams : ShapeParams
    {
    /** Default constructor
     */
    DEVICE ShapeUnionParams() : diameter(0.0), N(0), ignore(0) { }

    /** Load dynamic data members into shared memory and increase pointer

        @param ptr Pointer to load data to (will be incremented)
        @param available_bytes Size of remaining shared memory allocation
     */
    DEVICE inline void load_shared(char*& ptr, unsigned int& available_bytes)
        {
        tree.load_shared(ptr, available_bytes);
        mpos.load_shared(ptr, available_bytes);
        bool params_in_shared_mem = mparams.load_shared(ptr, available_bytes);
        moverlap.load_shared(ptr, available_bytes);
        morientation.load_shared(ptr, available_bytes);

// load all member parameters
#if defined(__HIP_DEVICE_COMPILE__)
        __syncthreads();
#endif

        if (params_in_shared_mem)
            {
            // load only if we are sure that we are not touching any unified memory
            for (unsigned int i = 0; i < mparams.size(); ++i)
                {
                mparams[i].load_shared(ptr, available_bytes);
                }
            }
        }

    /** Determine size of the shared memory allocation

        @param ptr Pointer to increment
        @param available_bytes Size of remaining shared memory allocation
     */
    HOSTDEVICE void allocate_shared(char*& ptr, unsigned int& available_bytes) const
        {
        tree.allocate_shared(ptr, available_bytes);
        mpos.allocate_shared(ptr, available_bytes);
        bool params_in_shared_mem = mparams.allocate_shared(ptr, available_bytes) != nullptr;
        moverlap.allocate_shared(ptr, available_bytes);
        morientation.allocate_shared(ptr, available_bytes);

        if (params_in_shared_mem)
            {
            for (unsigned int i = 0; i < mparams.size(); ++i)
                mparams[i].allocate_shared(ptr, available_bytes);
            }
        }

#ifdef ENABLE_HIP

    /// Set CUDA memory hints
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

    /** Construct with a given number of members

        @note Use of this constructor with N != 0 is intended only for unit tests
    */
    DEVICE ShapeUnionParams(unsigned int _N) // TODO rename mpos to m_pos etc
        : mpos(_N, false), morientation(_N, false), mparams(_N, false), moverlap(_N, false),
          diameter(0.0), N(_N), ignore(0)
        {
        }

    ShapeUnionParams(pybind11::dict v, bool managed = false)
        {
        // list of dicts to set parameters for member shapes
        pybind11::list shapes = v["shapes"];
        // list of 3-tuples that set the position of each member
        pybind11::list positions = v["positions"];
        // list of 4-tuples that set the orientation of each member
        pybind11::object orientations = v["orientations"];
        pybind11::object overlap = v["overlap"];
        ignore = v["ignore_statistics"].cast<unsigned int>();
        unsigned int leaf_capacity = v["capacity"].cast<unsigned int>();

        N = (unsigned int)pybind11::len(shapes);
        mpos = ManagedArray<vec3<ShortReal>>(N, managed);
        morientation = ManagedArray<quat<ShortReal>>(N, managed);
        mparams = ManagedArray<typename Shape::param_type>(N, managed);
        moverlap = ManagedArray<unsigned int>(N, managed);

        if (pybind11::len(positions) != N)
            {
            throw std::runtime_error(std::string("len(positions) != len(shapes): ")
                                     + "positions=" + pybind11::str(positions).cast<std::string>()
                                     + +" shapes=" + pybind11::str(shapes).cast<std::string>());
            }

        pybind11::list orientation_list;
        if (!orientations.is_none())
            {
            if (pybind11::len(orientations) != N)
                {
                throw std::runtime_error(std::string("len(orientations) != len(shapes): ")
                                         + "orientations="
                                         + pybind11::str(orientations).cast<std::string>()
                                         + +" shapes=" + pybind11::str(shapes).cast<std::string>());
                }

            orientation_list = pybind11::list(orientations);
            }

        pybind11::list overlap_list;
        if (!overlap.is_none())
            {
            if (pybind11::len(overlap) != N)
                {
                throw std::runtime_error(std::string("len(overlap) != len(shapes): ")
                                         + "overlaps=" + pybind11::str(overlap).cast<std::string>()
                                         + +" shapes=" + pybind11::str(shapes).cast<std::string>());
                }

            overlap_list = pybind11::list(overlap);
            }

        hpmc::detail::OBB* obbs = new hpmc::detail::OBB[N];

        std::vector<std::vector<vec3<ShortReal>>> internal_coordinates;

        // extract member parameters, positions, and orientations and compute the radius along the
        // way
        diameter = ShortReal(0.0);

        // compute a tight fitting AABB in the body frame
        hoomd::detail::AABB local_aabb(vec3<ShortReal>(0, 0, 0), ShortReal(0.0));

        for (unsigned int i = 0; i < N; i++)
            {
            typename Shape::param_type param(shapes[i], managed);

            pybind11::list position = positions[i];
            if (len(position) != 3)
                throw std::runtime_error("Each position must have 3 elements: found "
                                         + pybind11::str(position).cast<std::string>() + " in "
                                         + pybind11::str(positions).cast<std::string>());

            vec3<ShortReal> pos = vec3<ShortReal>(pybind11::cast<ShortReal>(position[0]),
                                                  pybind11::cast<ShortReal>(position[1]),
                                                  pybind11::cast<ShortReal>(position[2]));

            mparams[i] = param;
            mpos[i] = pos;

            // set default orientation of (1,0,0,0) when orienations is None
            if (orientations.is_none())
                {
                morientation[i] = quat<ShortReal>(1, vec3<ShortReal>(0, 0, 0));
                }
            else
                {
                pybind11::list orientation_l = orientation_list[i];
                ShortReal s = pybind11::cast<ShortReal>(orientation_l[0]);
                ShortReal x = pybind11::cast<ShortReal>(orientation_l[1]);
                ShortReal y = pybind11::cast<ShortReal>(orientation_l[2]);
                ShortReal z = pybind11::cast<ShortReal>(orientation_l[3]);
                morientation[i] = quat<ShortReal>(s, vec3<ShortReal>(x, y, z));
                }

            // set default overlap of 1 when overlaps is None
            if (overlap.is_none())
                {
                moverlap[i] = 1;
                }
            else
                {
                moverlap[i] = pybind11::cast<unsigned int>(overlap_list[i]);
                }

            Shape dummy(morientation[i], param);
            Scalar d = sqrt(dot(pos, pos));
            diameter = max(diameter, ShortReal(2 * d + dummy.getCircumsphereDiameter()));

            if (dummy.hasOrientation())
                {
                // construct OBB
                obbs[i] = dummy.getOBB(pos);
                }
            else
                {
                // construct bounding sphere
                obbs[i] = detail::OBB(pos, ShortReal(0.5) * dummy.getCircumsphereDiameter());
                }

            obbs[i].mask = moverlap[i];

            hoomd::detail::AABB my_aabb = dummy.getAABB(pos);
            local_aabb = merge(local_aabb, my_aabb);
            }

        // set the diameter

        // build tree and store GPU accessible version in parameter structure
        OBBTree tree_obb;
        tree_obb.buildTree(obbs, N, leaf_capacity, false);
        delete[] obbs;
        tree = GPUTree(tree_obb, managed);

        // store local AABB
        lower = local_aabb.getLower();
        upper = local_aabb.getUpper();
        }

    /// Convert parameters to a python dictionary
    pybind11::dict asDict()
        {
        pybind11::dict v;

        pybind11::list positions;
        pybind11::list orientations;
        pybind11::list overlaps;
        pybind11::list shapes;

        for (unsigned int i = 0; i < N; i++)
            {
            pybind11::list pos_l;
            pos_l.append(mpos[i].x);
            pos_l.append(mpos[i].y);
            pos_l.append(mpos[i].z);
            positions.append(pybind11::tuple(pos_l));

            pybind11::list orientation_l;
            orientation_l.append(morientation[i].s);
            orientation_l.append(morientation[i].v.x);
            orientation_l.append(morientation[i].v.y);
            orientation_l.append(morientation[i].v.z);
            orientations.append(pybind11::tuple(orientation_l));

            overlaps.append(moverlap[i]);
            shapes.append(mparams[i].asDict());
            }
        v["shapes"] = shapes;
        v["orientations"] = orientations;
        v["positions"] = positions;
        v["overlap"] = overlaps;
        v["ignore_statistics"] = ignore;
        v["capacity"] = tree.getLeafNodeCapacity();

        return v;
        }
#endif

    /// OBB tree for constituent shapes
    GPUTree tree;

    /// Position vectors of member shapes
    ManagedArray<vec3<ShortReal>> mpos;

    /// Orientation of member shapes
    ManagedArray<quat<ShortReal>> morientation;

    /// Parameters of member shapes
    ManagedArray<typename Shape::param_type> mparams;

    /// only check overlaps for which moverlap[i] & moverlap[j]
    ManagedArray<unsigned int> moverlap;

    /// Precalculated overall circumsphere diameter
    ShortReal diameter;

    /// Number of member shapes
    unsigned int N;

    /// True when move statistics should not be counted
    unsigned int ignore;

    /// Lower corner of local AABB
    vec3<ShortReal> lower;

    /// Upper corner of local AABB
    vec3<ShortReal> upper;
    } __attribute__((aligned(32)));

    } // end namespace detail

/** Shape consisting of union of shapes of a single type but individual parameters.

    The parameter defining a ShapeUnion is a structure implementing the HPMC shape interface and
    containing parameter objects for its member particles in its own parameters structure

    The purpose of ShapeUnion is to allow an overlap check to iterate through pairs of member shapes
    between two composite particles. The two particles overlap if any of their member shapes
    overlap.

    ShapeUnion stores an internal OBB tree for fast overlap checks.
*/
template<class Shape> struct ShapeUnion
    {
    /// Define the parameter type
    typedef typename detail::ShapeUnionParams<Shape> param_type;

    /// Temporary storage for depletant insertion
    typedef struct detail::union_depletion_storage depletion_storage_type;

    /// Construct a shape at a given orientation
    DEVICE ShapeUnion(const quat<Scalar>& _orientation, const param_type& _params)
        : orientation(_orientation), members(_params)
        {
        }

    /// Check if the shape may be rotated
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

    /// Check if this shape should be ignored in the move statistics
    DEVICE bool ignoreStatistics() const
        {
        return members.ignore;
        }

    /// Get the circumsphere diameter of the shape
    DEVICE ShortReal getCircumsphereDiameter() const
        {
        // return the precomputed diameter
        return members.diameter;
        }

    /// Get the in-sphere radius of the shape
    DEVICE ShortReal getInsphereRadius() const
        {
        // not implemented
        return ShortReal(0.0);
        }

    /// Return the bounding box of the shape in world coordinates
    DEVICE hoomd::detail::AABB getAABB(const vec3<Scalar>& pos) const
        {
        return getOBB(pos).getAABB();
        }

    /// Return a tight fitting OBB
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
            return detail::OBB(pos, ShortReal(0.5) * members.diameter);
            }
        }

    /** Returns true if this shape splits the overlap check over several threads of a warp using
        threadIdx.x
    */
    HOSTDEVICE static bool isParallel()
        {
        return true;
        }

    /// Orientation of the sphere
    quat<Scalar> orientation;

    /// Member data
    const param_type& members;
    };

template<class Shape>
DEVICE inline bool test_narrow_phase_overlap(vec3<ShortReal> dr,
                                             const ShapeUnion<Shape>& a,
                                             const ShapeUnion<Shape>& b,
                                             unsigned int cur_node_a,
                                             unsigned int cur_node_b,
                                             unsigned int& err)
    {
    //! Param type of the member shapes
    typedef typename Shape::param_type mparam_type;

    vec3<ShortReal> r_ab = rotate(conj(quat<ShortReal>(b.orientation)), vec3<ShortReal>(dr));

    // loop through leaf particles of cur_node_a
    // parallel loop over N^2 interacting particle pairs
    unsigned int ptl_i = a.members.tree.getLeafNodePtrByNode(cur_node_a);
    unsigned int ptl_j = b.members.tree.getLeafNodePtrByNode(cur_node_b);

    unsigned int ptls_i_end = a.members.tree.getLeafNodePtrByNode(cur_node_a + 1);
    unsigned int ptls_j_end = b.members.tree.getLeafNodePtrByNode(cur_node_b + 1);

    // get starting offset for this thread
    unsigned int na = ptls_i_end - ptl_i;
    unsigned int nb = ptls_j_end - ptl_j;

    unsigned int len = na * nb;

#if defined(__HIP_DEVICE_COMPILE__)
    unsigned int offset = threadIdx.x;
    unsigned int incr = blockDim.x;
#else
    unsigned int offset = 0;
    unsigned int incr = 1;
#endif

    // iterate over (a,b) pairs in row major
    for (unsigned int n = 0; n < len; n += incr)
        {
        if (n + offset < len)
            {
            unsigned int ishape = a.members.tree.getParticleByIndex(ptl_i + (n + offset) / nb);
            unsigned int jshape = b.members.tree.getParticleByIndex(ptl_j + (n + offset) % nb);

            const mparam_type& params_i = a.members.mparams[ishape];
            Shape shape_i(quat<Scalar>(), params_i);
            if (shape_i.hasOrientation())
                shape_i.orientation = conj(quat<ShortReal>(b.orientation))
                                      * quat<ShortReal>(a.orientation)
                                      * a.members.morientation[ishape];

            vec3<ShortReal> pos_i(
                rotate(conj(quat<ShortReal>(b.orientation)) * quat<ShortReal>(a.orientation),
                       a.members.mpos[ishape])
                - r_ab);
            unsigned int overlap_i = a.members.moverlap[ishape];

            const auto& params_j = b.members.mparams[jshape];
            Shape shape_j(quat<Scalar>(), params_j);
            if (shape_j.hasOrientation())
                shape_j.orientation = b.members.morientation[jshape];

            unsigned int overlap_j = b.members.moverlap[jshape];

            if (overlap_i & overlap_j)
                {
                vec3<ShortReal> r_ij = b.members.mpos[jshape] - pos_i;
                if (test_overlap(r_ij, shape_i, shape_j, err))
                    {
                    return true;
                    }
                }
            }
        }

    return false;
    }

template<class Shape>
DEVICE inline bool test_overlap(const vec3<Scalar>& r_ab,
                                const ShapeUnion<Shape>& a,
                                const ShapeUnion<Shape>& b,
                                unsigned int& err)
    {
    const detail::GPUTree& tree_a = a.members.tree;
    const detail::GPUTree& tree_b = b.members.tree;

    // perform a tandem tree traversal
    unsigned long int stack = 0;
    unsigned int cur_node_a = 0;
    unsigned int cur_node_b = 0;

    vec3<ShortReal> dr_rot(rotate(conj(b.orientation), -r_ab));
    quat<ShortReal> q(conj(b.orientation) * a.orientation);

    detail::OBB obb_a = tree_a.getOBB(cur_node_a);
    obb_a.affineTransform(q, dr_rot);

    detail::OBB obb_b = tree_b.getOBB(cur_node_b);

    unsigned int query_node_a = UINT_MAX;
    unsigned int query_node_b = UINT_MAX;

    while (cur_node_a != tree_a.getNumNodes() && cur_node_b != tree_b.getNumNodes())
        {
        query_node_a = cur_node_a;
        query_node_b = cur_node_b;

        if (detail::traverseBinaryStack(tree_a,
                                        tree_b,
                                        cur_node_a,
                                        cur_node_b,
                                        stack,
                                        obb_a,
                                        obb_b,
                                        q,
                                        dr_rot)
            && test_narrow_phase_overlap(r_ab, a, b, query_node_a, query_node_b, err))
            return true;
        }

    return false;
    }

template<class Shape>
DEVICE inline bool test_narrow_phase_excluded_volume_overlap(vec3<ShortReal> dr,
                                                             const ShapeUnion<Shape>& a,
                                                             const ShapeUnion<Shape>& b,
                                                             unsigned int cur_node_a,
                                                             unsigned int cur_node_b,
                                                             ShortReal r,
                                                             unsigned int dim)
    {
    vec3<ShortReal> r_ab = rotate(conj(quat<ShortReal>(b.orientation)), vec3<ShortReal>(dr));

    //! Param type of the member shapes
    typedef typename Shape::param_type mparam_type;

    // loop through shape of cur_node_a
    unsigned int na = a.members.tree.getNumParticles(cur_node_a);
    unsigned int nb = b.members.tree.getNumParticles(cur_node_b);

    for (unsigned int i = 0; i < na; i++)
        {
        unsigned int ishape = a.members.tree.getParticleByNode(cur_node_a, i);

        const mparam_type& params_i = a.members.mparams[ishape];
        Shape shape_i(quat<Scalar>(), params_i);
        if (shape_i.hasOrientation())
            shape_i.orientation = conj(quat<ShortReal>(b.orientation))
                                  * quat<ShortReal>(a.orientation) * a.members.morientation[ishape];

        vec3<ShortReal> pos_i(
            rotate(conj(quat<ShortReal>(b.orientation)) * quat<ShortReal>(a.orientation),
                   a.members.mpos[ishape])
            - r_ab);

        // loop through shapes of cur_node_b
        for (unsigned int j = 0; j < nb; j++)
            {
            unsigned int jshape = b.members.tree.getParticleByNode(cur_node_b, j);

            const mparam_type& params_j = b.members.mparams[jshape];
            Shape shape_j(quat<Scalar>(), params_j);
            if (shape_j.hasOrientation())
                shape_j.orientation = b.members.morientation[jshape];

            vec3<ShortReal> r_ij = b.members.mpos[jshape] - pos_i;
            if (excludedVolumeOverlap(shape_i,
                                      shape_j,
                                      r_ij,
                                      r,
                                      dim,
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
DEVICE inline bool excludedVolumeOverlap(const ShapeUnion<Shape>& a,
                                         const ShapeUnion<Shape>& b,
                                         const vec3<Scalar>& r_ab,
                                         ShortReal r,
                                         unsigned int dim,
                                         const detail::SamplingMethod::enumAccurate)
    {
    // perform a tandem tree traversal
    const detail::GPUTree& tree_a = a.members.tree;
    const detail::GPUTree& tree_b = b.members.tree;

    unsigned long int stack = 0;
    unsigned int cur_node_a = 0;
    unsigned int cur_node_b = 0;

    vec3<ShortReal> dr_rot(rotate(conj(b.orientation), -r_ab));
    quat<ShortReal> q(conj(b.orientation) * a.orientation);

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

        if (detail::traverseBinaryStack(tree_a,
                                        tree_b,
                                        cur_node_a,
                                        cur_node_b,
                                        stack,
                                        obb_a,
                                        obb_b,
                                        q,
                                        dr_rot)
            && test_narrow_phase_excluded_volume_overlap(r_ab,
                                                         a,
                                                         b,
                                                         query_node_a,
                                                         query_node_b,
                                                         r,
                                                         dim))
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
DEVICE inline unsigned int
allocateDepletionTemporaryStorage(const ShapeUnion<Shape>& a,
                                  const ShapeUnion<Shape>& b,
                                  const vec3<Scalar>& r_ab,
                                  ShortReal r,
                                  unsigned int dim,
                                  const detail::SamplingMethod::enumAccurate)
    {
    const detail::GPUTree& tree_a = a.members.tree;
    const detail::GPUTree& tree_b = b.members.tree;

    unsigned long int stack = 0;
    unsigned int cur_node_a = 0;
    unsigned int cur_node_b = 0;

    vec3<ShortReal> dr_rot(rotate(conj(b.orientation), -r_ab));
    quat<ShortReal> q(conj(b.orientation) * a.orientation);

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

        if (detail::traverseBinaryStack(tree_a,
                                        tree_b,
                                        cur_node_a,
                                        cur_node_b,
                                        stack,
                                        obb_a,
                                        obb_b,
                                        q,
                                        dr_rot)
            && test_narrow_phase_excluded_volume_overlap(r_ab,
                                                         a,
                                                         b,
                                                         query_node_a,
                                                         query_node_b,
                                                         r,
                                                         dim))
            {
            // count number of overlapping pairs
            nelem++;
            }
        }

    return nelem;
    }

template<class Shape>
DEVICE inline ShortReal sampling_volume_narrow_phase(vec3<ShortReal> dr,
                                                     const ShapeUnion<Shape>& a,
                                                     const ShapeUnion<Shape>& b,
                                                     unsigned int cur_node_a,
                                                     unsigned int cur_node_b,
                                                     ShortReal r,
                                                     unsigned int dim)
    {
    vec3<ShortReal> r_ab = rotate(conj(quat<ShortReal>(b.orientation)), vec3<ShortReal>(dr));

    // loop through shape of cur_node_a
    unsigned int na = a.members.tree.getNumParticles(cur_node_a);
    unsigned int nb = b.members.tree.getNumParticles(cur_node_b);

    ShortReal V(0.0);

    for (unsigned int i = 0; i < na; i++)
        {
        unsigned int ishape = a.members.tree.getParticleByNode(cur_node_a, i);

        const auto& params_i = a.members.mparams[ishape];
        Shape shape_i(quat<Scalar>(), params_i);
        if (shape_i.hasOrientation())
            shape_i.orientation = conj(quat<ShortReal>(b.orientation))
                                  * quat<ShortReal>(a.orientation) * a.members.morientation[ishape];

        vec3<ShortReal> pos_i(
            rotate(conj(quat<ShortReal>(b.orientation)) * quat<ShortReal>(a.orientation),
                   a.members.mpos[ishape])
            - r_ab);

        // loop through shapes of cur_node_b
        for (unsigned int j = 0; j < nb; j++)
            {
            unsigned int jshape = b.members.tree.getParticleByNode(cur_node_b, j);

            const auto& params_j = b.members.mparams[jshape];
            Shape shape_j(quat<Scalar>(), params_j);
            if (shape_j.hasOrientation())
                shape_j.orientation = b.members.morientation[jshape];

            vec3<ShortReal> r_ij = b.members.mpos[jshape] - pos_i;
            if (excludedVolumeOverlap(shape_i,
                                      shape_j,
                                      r_ij,
                                      r,
                                      dim,
                                      detail::SamplingMethod::accurate))
                {
                V += getSamplingVolumeIntersection(shape_i,
                                                   shape_j,
                                                   r_ij,
                                                   r,
                                                   dim,
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
DEVICE inline unsigned int
initializeDepletionTemporaryStorage(const ShapeUnion<Shape>& a,
                                    const ShapeUnion<Shape>& b,
                                    const vec3<Scalar>& r_ab,
                                    ShortReal r,
                                    unsigned int dim,
                                    detail::union_depletion_storage* storage,
                                    const ShortReal V_sample,
                                    const detail::SamplingMethod::enumAccurate)
    {
    const detail::GPUTree& tree_a = a.members.tree;
    const detail::GPUTree& tree_b = b.members.tree;

    unsigned long int stack = 0;
    unsigned int cur_node_a = 0;
    unsigned int cur_node_b = 0;

    vec3<ShortReal> dr_rot(rotate(conj(b.orientation), -r_ab));
    quat<ShortReal> q(conj(b.orientation) * a.orientation);

    detail::OBB obb_a = tree_a.getOBB(cur_node_a);
    obb_a.affineTransform(q, dr_rot);

    detail::OBB obb_b = tree_b.getOBB(cur_node_b);

    unsigned int query_node_a = UINT_MAX;
    unsigned int query_node_b = UINT_MAX;

    unsigned int nelem = 0;
    ShortReal V_sum(0.0);

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

        if (detail::traverseBinaryStack(tree_a,
                                        tree_b,
                                        cur_node_a,
                                        cur_node_b,
                                        stack,
                                        obb_a,
                                        obb_b,
                                        q,
                                        dr_rot)
            && test_narrow_phase_excluded_volume_overlap(r_ab,
                                                         a,
                                                         b,
                                                         query_node_a,
                                                         query_node_b,
                                                         r,
                                                         dim))
            {
            V_sum += sampling_volume_narrow_phase(r_ab, a, b, query_node_a, query_node_b, r, dim);
            detail::union_depletion_storage elem;
            elem.accumulated_weight = V_sum / V_sample;
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
DEVICE inline ShortReal getSamplingVolumeIntersection(const ShapeUnion<Shape>& a,
                                                      const ShapeUnion<Shape>& b,
                                                      const vec3<Scalar>& r_ab,
                                                      ShortReal r,
                                                      unsigned int dim,
                                                      const detail::SamplingMethod::enumAccurate)
    {
    // perform a tandem tree traversal
    const detail::GPUTree& tree_a = a.members.tree;
    const detail::GPUTree& tree_b = b.members.tree;

    unsigned long int stack = 0;
    unsigned int cur_node_a = 0;
    unsigned int cur_node_b = 0;

    vec3<ShortReal> dr_rot(rotate(conj(b.orientation), -r_ab));
    quat<ShortReal> q(conj(b.orientation) * a.orientation);

    detail::OBB obb_a = tree_a.getOBB(cur_node_a);
    obb_a.affineTransform(q, dr_rot);

    detail::OBB obb_b = tree_b.getOBB(cur_node_b);

    unsigned int query_node_a = UINT_MAX;
    unsigned int query_node_b = UINT_MAX;

    ShortReal V_sample(0.0);
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

        if (detail::traverseBinaryStack(tree_a,
                                        tree_b,
                                        cur_node_a,
                                        cur_node_b,
                                        stack,
                                        obb_a,
                                        obb_b,
                                        q,
                                        dr_rot))
            {
            V_sample
                += sampling_volume_narrow_phase(r_ab, a, b, query_node_a, query_node_b, r, dim);
            }
        }
    return V_sample;
    }

template<class RNG, class Shape>
DEVICE inline bool sample_narrow_phase(RNG& rng,
                                       vec3<ShortReal> dr,
                                       const ShapeUnion<Shape>& a,
                                       const ShapeUnion<Shape>& b,
                                       unsigned int cur_node_a,
                                       unsigned int cur_node_b,
                                       ShortReal r,
                                       vec3<ShortReal>& p,
                                       unsigned int dim)
    {
    ShortReal V_sample = sampling_volume_narrow_phase(dr, a, b, cur_node_a, cur_node_b, r, dim);
    ShortReal u = hoomd::UniformDistribution<ShortReal>(ShortReal(0.0), V_sample)(rng);

    ShortReal V_sum(0.0);

    //! Param type of the member shapes
    typedef typename Shape::param_type mparam_type;

    // loop through shape of cur_node_a
    unsigned int na = a.members.tree.getNumParticles(cur_node_a);
    unsigned int nb = b.members.tree.getNumParticles(cur_node_b);

    unsigned int ishape, jshape;

    bool done = false;
    unsigned int i, j;

    vec3<ShortReal> r_ab = rotate(conj(quat<ShortReal>(b.orientation)), vec3<ShortReal>(dr));

    ShortReal V;
    for (i = 0; i < na; i++)
        {
        ishape = a.members.tree.getParticleByNode(cur_node_a, i);

        const mparam_type& params_i = a.members.mparams[ishape];
        Shape shape_i(quat<Scalar>(), params_i);
        if (shape_i.hasOrientation())
            shape_i.orientation = conj(quat<ShortReal>(b.orientation))
                                  * quat<ShortReal>(a.orientation) * a.members.morientation[ishape];

        vec3<ShortReal> pos_i(
            rotate(conj(quat<ShortReal>(b.orientation)) * quat<ShortReal>(a.orientation),
                   a.members.mpos[ishape])
            - r_ab);

        // loop through shapes of cur_node_b
        for (j = 0; j < nb; j++)
            {
            jshape = b.members.tree.getParticleByNode(cur_node_b, j);

            const mparam_type& params_j = b.members.mparams[jshape];
            Shape shape_j(quat<Scalar>(), params_j);
            if (shape_j.hasOrientation())
                shape_j.orientation = b.members.morientation[jshape];

            vec3<ShortReal> r_ij = b.members.mpos[jshape] - pos_i;
            if (excludedVolumeOverlap(shape_i,
                                      shape_j,
                                      r_ij,
                                      r,
                                      dim,
                                      detail::SamplingMethod::accurate))
                {
                V = getSamplingVolumeIntersection(shape_i,
                                                  shape_j,
                                                  r_ij,
                                                  r,
                                                  dim,
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
    Shape shape_i(a.orientation * quat<Scalar>(a.members.morientation[ishape]),
                  a.members.mparams[ishape]);
    Shape shape_j(b.orientation * quat<Scalar>(b.members.morientation[jshape]),
                  b.members.mparams[jshape]);
    vec3<ShortReal> pos_i = rotate(quat<ShortReal>(a.orientation), a.members.mpos[ishape]);
    vec3<Scalar> r_ij = rotate(quat<ShortReal>(b.orientation), b.members.mpos[jshape]) + dr - pos_i;

    // set up temp storage on stack / in local memory
    unsigned int ntemp = allocateDepletionTemporaryStorage(shape_i,
                                                           shape_j,
                                                           r_ij,
                                                           r,
                                                           dim,
                                                           detail::SamplingMethod::accurate);
    std::vector<typename Shape::depletion_storage_type> temp(ntemp);
    unsigned int nelem = initializeDepletionTemporaryStorage(shape_i,
                                                             shape_j,
                                                             r_ij,
                                                             r,
                                                             dim,
                                                             temp.data(),
                                                             V,
                                                             detail::SamplingMethod::accurate);

    // sample
    if (!sampleInExcludedVolumeIntersection(rng,
                                            shape_i,
                                            shape_j,
                                            r_ij,
                                            r,
                                            p,
                                            dim,
                                            nelem,
                                            temp.data(),
                                            detail::SamplingMethod::accurate))
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
            shape_i.orientation = conj(quat<ShortReal>(b.orientation))
                                  * quat<ShortReal>(a.orientation) * a.members.morientation[ishape];

        vec3<ShortReal> pos_i(
            rotate(conj(quat<ShortReal>(b.orientation)) * quat<ShortReal>(a.orientation),
                   a.members.mpos[ishape])
            - r_ab);

        // loop through shapes of cur_node_b
        for (j = 0; j < ((i == min_i) ? min_j : nb); j++)
            {
            unsigned int jshape = b.members.tree.getParticleByNode(cur_node_b, j);

            const mparam_type& params_j = b.members.mparams[jshape];
            Shape shape_j(quat<Scalar>(), params_j);
            if (shape_j.hasOrientation())
                shape_j.orientation = b.members.morientation[jshape];

            vec3<ShortReal> r_ij = b.members.mpos[jshape] - pos_i;
            if (excludedVolumeOverlap(shape_i,
                                      shape_j,
                                      r_ij,
                                      r,
                                      dim,
                                      detail::SamplingMethod::accurate))
                {
                // shift origin to ishape's position, test in space frame
                Shape shape_i_world(a.orientation * quat<Scalar>(a.members.morientation[ishape]),
                                    params_i);
                Shape shape_j_world(b.orientation * quat<Scalar>(b.members.morientation[jshape]),
                                    params_j);

                vec3<ShortReal> pos_i
                    = rotate(quat<ShortReal>(a.orientation), a.members.mpos[ishape]);
                vec3<ShortReal> r_ij_world
                    = rotate(quat<ShortReal>(b.orientation), b.members.mpos[jshape]) + dr - pos_i;
                vec3<ShortReal> q = p - pos_i;

                if (isPointInExcludedVolumeIntersection(shape_i_world,
                                                        shape_j_world,
                                                        r_ij_world,
                                                        r,
                                                        q,
                                                        dim,
                                                        detail::SamplingMethod::accurate))
                    return false;
                }
            }
        }
    return true;
    }

template<class Shape>
DEVICE inline bool pt_in_intersection_narrow_phase(vec3<ShortReal> dr,
                                                   const ShapeUnion<Shape>& a,
                                                   const ShapeUnion<Shape>& b,
                                                   unsigned int cur_node_a,
                                                   unsigned int cur_node_b,
                                                   ShortReal r,
                                                   const vec3<ShortReal>& p,
                                                   unsigned int dim)
    {
    vec3<ShortReal> r_ab = rotate(conj(quat<ShortReal>(b.orientation)), vec3<ShortReal>(dr));

    //! Param type of the member shapes
    typedef typename Shape::param_type mparam_type;

    // loop through shape of cur_node_a
    unsigned int na = a.members.tree.getNumParticles(cur_node_a);
    unsigned int nb = b.members.tree.getNumParticles(cur_node_b);

    for (unsigned int i = 0; i < na; i++)
        {
        unsigned int ishape = a.members.tree.getParticleByNode(cur_node_a, i);

        const mparam_type& params_i = a.members.mparams[ishape];
        Shape shape_i(quat<Scalar>(), params_i);
        if (shape_i.hasOrientation())
            shape_i.orientation = conj(quat<ShortReal>(b.orientation))
                                  * quat<ShortReal>(a.orientation) * a.members.morientation[ishape];

        vec3<ShortReal> pos_i(
            rotate(conj(quat<ShortReal>(b.orientation)) * quat<ShortReal>(a.orientation),
                   a.members.mpos[ishape])
            - r_ab);

        // loop through shapes of cur_node_b
        for (unsigned int j = 0; j < nb; j++)
            {
            unsigned int jshape = b.members.tree.getParticleByNode(cur_node_b, j);

            const mparam_type& params_j = b.members.mparams[jshape];
            Shape shape_j(quat<Scalar>(), params_j);
            if (shape_j.hasOrientation())
                shape_j.orientation = b.members.morientation[jshape];

            vec3<ShortReal> r_ij = b.members.mpos[jshape] - pos_i;
            if (excludedVolumeOverlap(shape_i,
                                      shape_j,
                                      r_ij,
                                      r,
                                      dim,
                                      detail::SamplingMethod::accurate))
                {
                // shift origin to ihape's position, test in space frame
                Shape shape_i_world(a.orientation * quat<Scalar>(a.members.morientation[ishape]),
                                    params_i);
                Shape shape_j_world(b.orientation * quat<Scalar>(b.members.morientation[jshape]),
                                    params_j);

                vec3<ShortReal> pos_i
                    = rotate(quat<ShortReal>(a.orientation), a.members.mpos[ishape]);
                vec3<ShortReal> r_ij_world
                    = rotate(quat<ShortReal>(b.orientation), b.members.mpos[jshape]) + dr - pos_i;
                vec3<ShortReal> q = p - pos_i;

                if (isPointInExcludedVolumeIntersection(shape_i_world,
                                                        shape_j_world,
                                                        r_ij_world,
                                                        r,
                                                        q,
                                                        dim,
                                                        detail::SamplingMethod::accurate))
                    return true;
                }
            }
        }

    return false;
    }

//! Uniform rejection sampling in a volume covering the intersection of two shapes, defined by their
//! Minkowski sums with a sphere of radius r
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
DEVICE inline bool
sampleInExcludedVolumeIntersection(RNG& rng,
                                   const ShapeUnion<Shape>& a,
                                   const ShapeUnion<Shape>& b,
                                   const vec3<Scalar>& r_ab,
                                   ShortReal r,
                                   vec3<ShortReal>& p,
                                   unsigned int dim,
                                   unsigned int storage_sz,
                                   const detail::union_depletion_storage* storage,
                                   const detail::SamplingMethod::enumAccurate)
    {
    // perform a tandem tree traversal
    const detail::GPUTree& tree_a = a.members.tree;
    const detail::GPUTree& tree_b = b.members.tree;

    ShortReal u = hoomd::UniformDistribution<ShortReal>()(rng);
    bool found = false;
    unsigned int query_node_a;
    unsigned int query_node_b;

    if (storage_sz > 0)
        {
        // binary search for the the first element with accumulated weight > u
        unsigned int l = 0;
        unsigned int r = storage_sz - 1;
        while (l <= r)
            {
            unsigned int m = l + (r - l) / 2;

            if (storage[m].accumulated_weight > u
                && (m == 0 || storage[m - 1].accumulated_weight <= u))
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
    vec3<ShortReal> dr_rot(rotate(conj(b.orientation), -r_ab));
    quat<ShortReal> q(conj(b.orientation) * a.orientation);

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

        if (detail::traverseBinaryStackIntersection(tree_a,
                                                    tree_b,
                                                    cur_node_a,
                                                    cur_node_b,
                                                    stack,
                                                    obb_a,
                                                    obb_b,
                                                    q,
                                                    dr_rot,
                                                    obb_query)
            && (query_node_a < min_query_node_a
                || (query_node_a == min_query_node_a && query_node_b < min_query_node_b))
            && pt_in_intersection_narrow_phase(r_ab, a, b, query_node_a, query_node_b, r, p, dim))
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
DEVICE inline bool isPointInExcludedVolumeIntersection(const ShapeUnion<Shape>& a,
                                                       const ShapeUnion<Shape>& b,
                                                       const vec3<Scalar>& r_ab,
                                                       ShortReal r,
                                                       const vec3<ShortReal>& p,
                                                       unsigned int dim,
                                                       const detail::SamplingMethod::enumAccurate)
    {
    // perform a tandem tree traversal
    const detail::GPUTree& tree_a = a.members.tree;
    const detail::GPUTree& tree_b = b.members.tree;

    unsigned long int stack = 0;
    unsigned int cur_node_a = 0;
    unsigned int cur_node_b = 0;

    vec3<ShortReal> dr_rot(rotate(conj(b.orientation), -r_ab));
    quat<ShortReal> q(conj(b.orientation) * a.orientation);

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

        if (detail::traverseBinaryStackIntersection(tree_a,
                                                    tree_b,
                                                    cur_node_a,
                                                    cur_node_b,
                                                    stack,
                                                    obb_a,
                                                    obb_b,
                                                    q,
                                                    dr_rot,
                                                    obb_query)
            && pt_in_intersection_narrow_phase(r_ab, a, b, query_node_a, query_node_b, r, p, dim))
            {
            return true;
            }
        }

    return false;
    }

#ifndef __HIPCC__
template<> inline std::string getShapeSpec(const ShapeUnion<ShapeSphere>& sphere_union)
    {
    auto& members = sphere_union.members;

    unsigned int n_centers = members.N;

    if (n_centers == 0)
        {
        throw std::runtime_error("Shape definition not supported for 0-center union.");
        }

    std::ostringstream shapedef;
    shapedef << "{\"type\": \"SphereUnion\", \"centers\": [";
    for (unsigned int i = 0; i < n_centers - 1; i++)
        {
        shapedef << "[" << members.mpos[i].x << ", " << members.mpos[i].y << ", "
                 << members.mpos[i].z << "], ";
        }
    shapedef << "[" << members.mpos[n_centers - 1].x << ", " << members.mpos[n_centers - 1].y
             << ", " << members.mpos[n_centers - 1].z << "]], \"diameters\": [";
    for (unsigned int i = 0; i < n_centers - 1; i++)
        {
        shapedef << 2.0 * members.mparams[i].radius << ", ";
        }
    shapedef << 2.0 * members.mparams[n_centers - 1].radius;
    shapedef << "]}";

    return shapedef.str();
    }
#endif

    } // end namespace hpmc
    } // end namespace hoomd
#undef DEVICE
#undef HOSTDEVICE
