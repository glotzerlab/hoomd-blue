// Copyright (c) 2009-2019 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.

# pragma once

#include "hoomd/HOOMDMath.h"
#include "hoomd/BoxDim.h"
#include "hoomd/VectorMath.h"
#include "ShapeSphere.h"    //< For the base template of test_overlap
#include "ShapeSpheropolyhedron.h"
#include "ShapeConvexPolyhedron.h"
#include "GPUTree.h"

#include "hoomd/ManagedArray.h"
#ifdef NVCC
#define DEVICE __device__
#define HOSTDEVICE __host__ __device__
#else
#define DEVICE
#define HOSTDEVICE
#include <iostream>
#endif

namespace hpmc
{

namespace detail
{

/** Data structure for shape composed of a union of multiple shapes.

    Store N member shapes of the same type at given positions and orientations relative to the
    position and orientation of the parent shape. Use ManagedArray to support shape data types that
    have nested ManagedArray members.
*/
template<class Shape>
struct ShapeUnionParams : ShapeParams
    {
    /** Default constructor
    */
    DEVICE ShapeUnionParams()
        : diameter(0.0), N(0), ignore(0)
        {
        }

    /** Load dynamic data members into shared memory and increase pointer

        @param ptr Pointer to load data to (will be incremented)
        @param available_bytes Size of remaining shared memory allocation
     */
    DEVICE void load_shared(char *& ptr, unsigned int &available_bytes)
        {
        tree.load_shared(ptr, available_bytes);
        mpos.load_shared(ptr, available_bytes);
        bool params_in_shared_mem = mparams.load_shared(ptr, available_bytes);
        moverlap.load_shared(ptr, available_bytes);
        morientation.load_shared(ptr, available_bytes);

        // load all member parameters
        #if defined (__CUDA_ARCH__)
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

    /** Determine size of the shared memory allocation

        @param ptr Pointer to increment
        @param available_bytes Size of remaining shared memory allocation
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


    #ifdef ENABLE_CUDA

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

    #ifndef NVCC

    /** Construct with a given number of members

        @note Use of this constructor with N != 0 is intended only for unit tests
    */
    DEVICE ShapeUnionParams(unsigned int _N)
        : mpos(_N, false), morientation(_N, false), mparams(_N, false),
          moverlap(_N, false), diameter(0.0), N(_N), ignore(0)
        {
        }

    ShapeUnionParams(pybind11::dict v, bool managed=false)
        {
        // list of dicts to set parameters for member shapes
        pybind11::list shapes = v["shapes"];
        // list of 3-tuples that set the position of each member
        pybind11::list positions = v["positions"];
        // list of 4-tuples that set the orientation of each member
        pybind11::list orientations = v["orientations"];
        pybind11::list overlap = v["overlap"];
        ignore = v["ignore_statistics"].cast<unsigned int>();
        unsigned int leaf_capacity = v["capacity"].cast<unsigned int>();

        N = pybind11::len(shapes);
        mpos = ManagedArray<vec3<OverlapReal> >(N,managed);
        morientation = ManagedArray<quat<OverlapReal> >(N,managed);
        mparams = ManagedArray<typename Shape::param_type>(N,managed);
        moverlap = ManagedArray<unsigned int>(N,managed);

        if (pybind11::len(positions) != N)
            {
            throw std::runtime_error(std::string("len(positions) != len(shapes): ")
                                     + "positions=" + pybind11::str(positions).cast<std::string>() +
                                     + " shapes=" + pybind11::str(shapes).cast<std::string>() );
            }
        if (pybind11::len(orientations) != N)
            {
            throw std::runtime_error(std::string("len(orientations) != len(shapes): ")
                                     + "orientations="
                                     + pybind11::str(orientations).cast<std::string>() +
                                     + " shapes=" + pybind11::str(shapes).cast<std::string>() );
            }
        if (pybind11::len(overlap) != N)
            {
            throw std::runtime_error(std::string("len(overlap) != len(shapes): ")
                                     + "overlaps=" + pybind11::str(overlap).cast<std::string>() +
                                     + " shapes=" + pybind11::str(shapes).cast<std::string>() );
            }

        hpmc::detail::OBB *obbs = new hpmc::detail::OBB[N];

        std::vector<std::vector<vec3<OverlapReal> > > internal_coordinates;

        // extract member parameters, positions, and orientations and compute the radius along the
        // way
        diameter = OverlapReal(0.0);

        // compute a tight fitting AABB in the body frame
        detail::AABB local_aabb(vec3<OverlapReal>(0,0,0),OverlapReal(0.0));

        for (unsigned int i = 0; i < N; i++)
            {
            typename Shape::param_type param(shapes[i]);

            pybind11::list position = pybind11::cast<pybind11::list>(positions[i]);
            if (len(position) != 3)
                throw std::runtime_error("Each position must have 3 elements: found "
                                        + pybind11::str(position).cast<std::string>()
                                        + " in " + pybind11::str(positions).cast<std::string>());

            vec3<OverlapReal> pos = vec3<OverlapReal>(pybind11::cast<OverlapReal>(position[0]),
                                                      pybind11::cast<OverlapReal>(position[1]),
                                                      pybind11::cast<OverlapReal>(position[2]));
            pybind11::list orientation_l = pybind11::cast<pybind11::list>(orientations[i]);
            OverlapReal s = pybind11::cast<OverlapReal>(orientation_l[0]);
            OverlapReal x = pybind11::cast<OverlapReal>(orientation_l[1]);
            OverlapReal y = pybind11::cast<OverlapReal>(orientation_l[2]);
            OverlapReal z = pybind11::cast<OverlapReal>(orientation_l[3]);
            quat<OverlapReal> orientation(s, vec3<OverlapReal>(x,y,z));

            mparams[i] = param;
            mpos[i] = pos;
            morientation[i] = orientation;
            moverlap[i] = pybind11::cast<unsigned int>(overlap[i]);

            Shape dummy(orientation, param);
            Scalar d = sqrt(dot(pos,pos));
            diameter = max(diameter, OverlapReal(2*d + dummy.getCircumsphereDiameter()));

            if (dummy.hasOrientation())
                {
                // construct OBB
                obbs[i] = dummy.getOBB(pos);
                }
            else
                {
                // construct bounding sphere
                obbs[i] = detail::OBB(pos, OverlapReal(0.5)*dummy.getCircumsphereDiameter());
                }

            obbs[i].mask = moverlap[i];

            detail::AABB my_aabb = dummy.getAABB(pos);
            local_aabb = merge(local_aabb, my_aabb);
            }

        // set the diameter

        // build tree and store GPU accessible version in parameter structure
        OBBTree tree_obb;
        tree_obb.buildTree(obbs, N, leaf_capacity, false);
        delete [] obbs;
        tree = GPUTree(tree_obb, false);

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
    ManagedArray<vec3<OverlapReal> > mpos;

    /// Orientation of member shapes
    ManagedArray<quat<OverlapReal> > morientation;

    /// Parameters of member shapes
    ManagedArray<typename Shape::param_type> mparams;

    /// only check overlaps for which moverlap[i] & moverlap[j]
    ManagedArray<unsigned int> moverlap;

    /// Precalculated overall circumsphere diameter
    OverlapReal diameter;

    /// Number of member shapes
    unsigned int N;

    /// True when move statistics should not be counted
    unsigned int ignore;

    /// Lower corner of local AABB
    vec3<OverlapReal> lower;

    /// Upper corner of local AABB
    vec3<OverlapReal> upper;
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
template<class Shape>
struct ShapeUnion
    {
    /// Define the parameter type
    typedef typename detail::ShapeUnionParams<Shape> param_type;

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
    DEVICE bool ignoreStatistics() const { return members.ignore; }

    /// Get the circumsphere diameter of the shape
    DEVICE OverlapReal getCircumsphereDiameter() const
        {
        // return the precomputed diameter
        return members.diameter;
        }

    /// Get the in-sphere radius of the shape
    DEVICE OverlapReal getInsphereRadius() const
        {
        // not implemented
        return OverlapReal(0.0);
        }

    #ifndef NVCC
    /// Return the shape parameters in the `type_shape` format
    std::string getShapeSpec() const
        {
        throw std::runtime_error("Shape definition not supported for this shape class.");
        }
    #endif

    /// Return the bounding box of the shape in world coordinates
    DEVICE detail::AABB getAABB(const vec3<Scalar>& pos) const
        {
        // rotate local AABB into world coordinates
        vec3<OverlapReal> lower_a = members.lower;
        vec3<OverlapReal> upper_a = members.upper;
        vec3<OverlapReal> lower_b = pos;
        vec3<OverlapReal> upper_b = pos;

        rotmat3<OverlapReal> M(orientation);

        OverlapReal e, f;
        e = M.row0.x*lower_a.x;
        f = M.row0.x*upper_a.x;
        if (e < f)
            {
            lower_b.x += e;
            upper_b.x += f;
            }
        else
            {
            lower_b.x += f;
            upper_b.x += e;
            }

        e = M.row0.y*lower_a.y;
        f = M.row0.y*upper_a.y;
        if (e < f)
            {
            lower_b.x += e;
            upper_b.x += f;
            }
        else
            {
            lower_b.x += f;
            upper_b.x += e;
            }

        e = M.row0.z*lower_a.z;
        f = M.row0.z*upper_a.z;
        if (e < f)
            {
            lower_b.x += e;
            upper_b.x += f;
            }
        else
            {
            lower_b.x += f;
            upper_b.x += e;
            }

        e = M.row1.x*lower_a.x;
        f = M.row1.x*upper_a.x;
        if (e < f)
            {
            lower_b.y += e;
            upper_b.y += f;
            }
        else
            {
            lower_b.y += f;
            upper_b.y += e;
            }

        e = M.row1.y*lower_a.y;
        f = M.row1.y*upper_a.y;
        if (e < f)
            {
            lower_b.y += e;
            upper_b.y += f;
            }
        else
            {
            lower_b.y += f;
            upper_b.y += e;
            }

        e = M.row1.z*lower_a.z;
        f = M.row1.z*upper_a.z;
        if (e < f)
            {
            lower_b.y += e;
            upper_b.y += f;
            }
        else
            {
            lower_b.y += f;
            upper_b.y += e;
            }

        e = M.row2.x*lower_a.x;
        f = M.row2.x*upper_a.x;
        if (e < f)
            {
            lower_b.z += e;
            upper_b.z += f;
            }
        else
            {
            lower_b.z += f;
            upper_b.z += e;
            }

        e = M.row2.y*lower_a.y;
        f = M.row2.y*upper_a.y;
        if (e < f)
            {
            lower_b.z += e;
            upper_b.z += f;
            }
        else
            {
            lower_b.z += f;
            upper_b.z += e;
            }

        e = M.row2.z*lower_a.z;
        f = M.row2.z*upper_a.z;
        if (e < f)
            {
            lower_b.z += e;
            upper_b.z += f;
            }
        else
            {
            lower_b.z += f;
            upper_b.z += e;
            }

        return detail::AABB(lower_b, upper_b);
        }

    /// Return a tight fitting OBB around the shape
    DEVICE detail::OBB getOBB(const vec3<Scalar>& pos) const
        {
        // get the root node OBB from the tree
        detail::OBB obb = members.tree.getOBB(0);

        // transform it into world-space
        obb.affineTransform(orientation, pos);

        return obb;
        }

    /** Returns true if this shape splits the overlap check over several threads of a warp using
        threadIdx.x
    */
    HOSTDEVICE static bool isParallel() {
        #ifdef SHAPE_UNION_LEAVES_AGAINST_TREE_TRAVERSAL
        return true;
        #else
        return false;
        #endif
        }

    /// Returns true if the overlap check supports sweeping both shapes by a sphere of given radius
    HOSTDEVICE static bool supportsSweepRadius()
        {
        return Shape::supportsSweepRadius();
        }

    /// Orientation of the sphere
    quat<Scalar> orientation;

    /// Member data
    const param_type& members;
    };

template<class Shape>
DEVICE inline bool test_narrow_phase_overlap(vec3<OverlapReal> dr,
                                             const ShapeUnion<Shape>& a,
                                             const ShapeUnion<Shape>& b,
                                             unsigned int cur_node_a,
                                             unsigned int cur_node_b,
                                             unsigned int &err,
                                             OverlapReal sweep_radius_a,
                                             OverlapReal sweep_radius_b,
                                             bool ignore_mask)
    {
    vec3<OverlapReal> r_ab = rotate(conj(quat<OverlapReal>(b.orientation)),vec3<OverlapReal>(dr));

    // loop through shape of cur_node_a
    unsigned int na = a.members.tree.getNumParticles(cur_node_a);
    unsigned int nb = b.members.tree.getNumParticles(cur_node_b);

    for (unsigned int i= 0; i < na; i++)
        {
        unsigned int ishape = a.members.tree.getParticle(cur_node_a, i);

        const auto& params_i = a.members.mparams[ishape];
        Shape shape_i(quat<Scalar>(), params_i);
        if (shape_i.hasOrientation())
            shape_i.orientation = conj(quat<OverlapReal>(b.orientation))
                                  * quat<OverlapReal>(a.orientation)
                                  * a.members.morientation[ishape];

        vec3<OverlapReal> pos_i(rotate(conj(quat<OverlapReal>(b.orientation))
                                       * quat<OverlapReal>(a.orientation),
                                       a.members.mpos[ishape])
                                - r_ab);
        unsigned int overlap_i = a.members.moverlap[ishape];

        // loop through shapes of cur_node_b
        for (unsigned int j= 0; j < nb; j++)
            {
            unsigned int jshape = b.members.tree.getParticle(cur_node_b, j);

            const auto& params_j = b.members.mparams[jshape];
            Shape shape_j(quat<Scalar>(), params_j);
            if (shape_j.hasOrientation())
                shape_j.orientation = b.members.morientation[jshape];

            unsigned int overlap_j = b.members.moverlap[jshape];

            if (ignore_mask || (overlap_i & overlap_j))
                {
                vec3<OverlapReal> r_ij = b.members.mpos[jshape] - pos_i;
                if (test_overlap(r_ij, shape_i, shape_j, err, sweep_radius_a, sweep_radius_b))
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
                                unsigned int& err,
                                Scalar sweep_radius_a = Scalar(0.0),
                                Scalar sweep_radius_b = Scalar(0.0))
    {
    const detail::GPUTree& tree_a = a.members.tree;
    const detail::GPUTree& tree_b = b.members.tree;

    bool ignore_mask = sweep_radius_a != Scalar(0.0) || sweep_radius_b != Scalar(0.0);

    #ifdef SHAPE_UNION_LEAVES_AGAINST_TREE_TRAVERSAL
    #ifdef NVCC
    // Parallel tree traversal
    unsigned int offset = threadIdx.x;
    unsigned int stride = blockDim.x;
    #else
    unsigned int offset = 0;
    unsigned int stride = 1;
    #endif

    OverlapReal sab = sweep_radius_a + sweep_radius_b;

    if (tree_a.getNumLeaves() <= tree_b.getNumLeaves())
        {
        for (unsigned int cur_leaf_a = offset;
             cur_leaf_a < tree_a.getNumLeaves();
             cur_leaf_a += stride)
            {
            unsigned int cur_node_a = tree_a.getLeafNode(cur_leaf_a);
            hpmc::detail::OBB obb_a = tree_a.getOBB(cur_node_a);
            // rotate and translate a's obb into b's body frame
            obb_a.affineTransform(conj(b.orientation)*a.orientation,
                rotate(conj(b.orientation),-r_ab));

            // extend OBB
            obb_a.lengths.x += sab;
            obb_a.lengths.y += sab;
            obb_a.lengths.z += sab;

            unsigned cur_node_b = 0;
            while (cur_node_b < tree_b.getNumNodes())
                {
                unsigned int query_node = cur_node_b;
                if (tree_b.queryNode(obb_a, cur_node_b, ignore_mask) &&
                    test_narrow_phase_overlap(r_ab,
                                              a,
                                              b,
                                              cur_node_a,
                                              query_node,
                                              err,
                                              sweep_radius_a,
                                              sweep_radius_b,
                                              ignore_mask))
                    return true;
                }
            }
        }
    else
        {
        for (unsigned int cur_leaf_b = offset;
             cur_leaf_b < tree_b.getNumLeaves();
             cur_leaf_b += stride)
            {
            unsigned int cur_node_b = tree_b.getLeafNode(cur_leaf_b);
            hpmc::detail::OBB obb_b = tree_b.getOBB(cur_node_b);

            // rotate and translate b's obb into a's body frame
            obb_b.affineTransform(conj(a.orientation)*b.orientation,
                rotate(conj(a.orientation),r_ab));

            // extend OBB
            obb_b.lengths.x += sab;
            obb_b.lengths.y += sab;
            obb_b.lengths.z += sab;

            unsigned cur_node_a = 0;
            while (cur_node_a < tree_a.getNumNodes())
                {
                unsigned int query_node = cur_node_a;
                if (tree_a.queryNode(obb_b, cur_node_a, ignore_mask) &&
                    test_narrow_phase_overlap(-r_ab,
                                              b,
                                              a,
                                              cur_node_b,
                                              query_node,
                                              err,
                                              sweep_radius_a,
                                              sweep_radius_b,
                                              ignore_mask))
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
        // extend OBBs
        if (query_node_a != cur_node_a)
            {
            obb_a.lengths.x += sweep_radius_a;
            obb_a.lengths.y += sweep_radius_a;
            obb_a.lengths.z += sweep_radius_a;
            query_node_a = cur_node_a;
            }

        if (query_node_b != cur_node_b)
            {
            obb_b.lengths.x += sweep_radius_b;
            obb_b.lengths.y += sweep_radius_b;
            obb_b.lengths.z += sweep_radius_b;
            query_node_b = cur_node_b;
            }

        if (detail::traverseBinaryStack(tree_a, tree_b, cur_node_a, cur_node_b, stack, obb_a, obb_b, q, dr_rot, ignore_mask)
            && test_narrow_phase_overlap(r_ab, a, b, query_node_a, query_node_b, err, sweep_radius_a, sweep_radius_b, ignore_mask))
            return true;
        }
    #endif

    return false;
    }

template<class Shape>
DEVICE inline bool test_narrow_phase_overlap_intersection(const ShapeUnion<Shape>& a,
                                             const ShapeUnion<Shape>& b,
                                             const ShapeUnion<Shape>& c,
                                             vec3<OverlapReal> ab_t,
                                             vec3<OverlapReal> ac_t,
                                             unsigned int cur_node_a,
                                             unsigned int cur_node_b,
                                             unsigned int cur_node_c,
                                             unsigned int &err,
                                             OverlapReal sweep_radius_a,
                                             OverlapReal sweep_radius_b,
                                             OverlapReal sweep_radius_c)
    {
    unsigned int na = a.members.tree.getNumParticles(cur_node_a);
    unsigned int nb = b.members.tree.getNumParticles(cur_node_b);
    unsigned int nc = c.members.tree.getNumParticles(cur_node_c);

    // loop through shapes of cur_node_a
    for (unsigned int i= 0; i < na; i++)
        {
        unsigned int ishape = a.members.tree.getParticle(cur_node_a, i);

        const auto& params_i = a.members.mparams[ishape];
        Shape shape_i(quat<Scalar>(), params_i);
        if (shape_i.hasOrientation())
            shape_i.orientation = quat<OverlapReal>(a.orientation)*a.members.morientation[ishape];

        vec3<OverlapReal> pos_i(rotate(quat<OverlapReal>(a.orientation),a.members.mpos[ishape]));
        unsigned int overlap_i = a.members.moverlap[ishape];

        // loop through shapes of cur_node_b
        for (unsigned int j= 0; j < nb; j++)
            {
            unsigned int jshape = b.members.tree.getParticle(cur_node_b, j);

            const auto& params_j = b.members.mparams[jshape];
            Shape shape_j(quat<Scalar>(), params_j);
            if (shape_j.hasOrientation())
                shape_j.orientation = quat<OverlapReal>(b.orientation)*b.members.morientation[jshape];

            vec3<OverlapReal> pos_ij(rotate(quat<OverlapReal>(b.orientation),b.members.mpos[jshape]) + ab_t - pos_i);
            unsigned int overlap_j = b.members.moverlap[jshape];

            // loop through shapes of cur_node_c
            for (unsigned int k= 0; k < nc; k++)
                {
                unsigned int kshape = c.members.tree.getParticle(cur_node_c, k);

                const auto& params_k = c.members.mparams[kshape];
                Shape shape_k(quat<Scalar>(), params_k);
                if (shape_k.hasOrientation())
                    shape_k.orientation = quat<OverlapReal>(c.orientation)*c.members.morientation[kshape];

                vec3<OverlapReal> pos_ik(rotate(quat<OverlapReal>(c.orientation),c.members.mpos[kshape]) + ac_t - pos_i);
                unsigned int overlap_k = c.members.moverlap[kshape];

                if (((overlap_i & overlap_k) || (overlap_j & overlap_k)) &&
                    test_overlap_intersection(shape_i, shape_j, shape_k, pos_ij, pos_ik, err, sweep_radius_a, sweep_radius_b, sweep_radius_c))
                    {
                    return true;
                    }
                }
            }
        }
    return false;
    }

/** Test for overlap of a third particle with the intersection of two shapes
    @param a First shape to test
    @param b Second shape to test
    @param c Third shape to test
    @param ab_t Position of second shape relative to first
    @param ac_t Position of third shape relative to first
    @param err Output variable that is incremented upon non-convergence
    @param sweep_radius_a Radius of a sphere to sweep the first shape by
    @param sweep_radius_b Radius of a sphere to sweep the second shape by
    @param sweep_radius_c Radius of a sphere to sweep the third shape by
*/
template <class Shape >
DEVICE inline bool test_overlap_intersection(const ShapeUnion<Shape>& a,
                                const ShapeUnion<Shape>& b,
                                const ShapeUnion<Shape>& c,
                                const vec3<Scalar>& ab_t,
                                const vec3<Scalar>& ac_t,
                                unsigned int& err,
                                Scalar sweep_radius_a = Scalar(0.0),
                                Scalar sweep_radius_b = Scalar(0.0),
                                Scalar sweep_radius_c = Scalar(0.0))
    {
    const detail::GPUTree& tree_a = a.members.tree;
    const detail::GPUTree& tree_b = b.members.tree;
    const detail::GPUTree& tree_c = c.members.tree;

    quat<OverlapReal> qbc(conj(b.orientation)*c.orientation);
    vec3<OverlapReal> rbc_rot(rotate(conj(quat<OverlapReal>(b.orientation)),vec3<OverlapReal>(-ab_t+ac_t)));

    vec3<OverlapReal> rab_rot(rotate(conj(quat<OverlapReal>(b.orientation)),vec3<OverlapReal>(-ab_t)));
    quat<OverlapReal> qab(conj(b.orientation)*a.orientation);

    for (unsigned int cur_leaf_c = 0; cur_leaf_c < tree_c.getNumLeaves(); cur_leaf_c++)
        {
        unsigned int cur_node_c = tree_c.getLeafNode(cur_leaf_c);
        detail::OBB obb_c = tree_c.getOBB(cur_node_c);

        obb_c.lengths.x += sweep_radius_c;
        obb_c.lengths.y += sweep_radius_c;
        obb_c.lengths.z += sweep_radius_c;

        // transform into b's reference frame
        obb_c.affineTransform(qbc, rbc_rot);

        // perform a tandem tree traversal between trees a and b, subject to overlap with c
        unsigned long int stack = 0;
        unsigned int cur_node_a = 0;
        unsigned int cur_node_b = 0;

        detail::OBB obb_a = tree_a.getOBB(cur_node_a);
        obb_a.affineTransform(qab, rab_rot);

        detail::OBB obb_b = tree_b.getOBB(cur_node_b);

        unsigned int query_node_a = UINT_MAX;
        unsigned int query_node_b = UINT_MAX;

        unsigned int mask_a = 0;
        unsigned int mask_b = 0;
        while (cur_node_a != tree_a.getNumNodes() && cur_node_b != tree_b.getNumNodes())
            {
            if (query_node_a != cur_node_a)
                {
                // extend OBBs
                obb_a.lengths.x += sweep_radius_a;
                obb_a.lengths.y += sweep_radius_a;
                obb_a.lengths.z += sweep_radius_a;
                query_node_a = cur_node_a;
                mask_a = obb_a.mask;
                }

            if (query_node_b != cur_node_b)
                {
                obb_b.lengths.x += sweep_radius_b;
                obb_b.lengths.y += sweep_radius_b;
                obb_b.lengths.z += sweep_radius_b;
                query_node_b = cur_node_b;
                mask_b = obb_b.mask;
                }

            // combine masks
            unsigned int combined_mask = mask_a | mask_b;
            obb_a.mask = obb_b.mask = combined_mask;

            if (detail::traverseBinaryStackIntersection(tree_a, tree_b, cur_node_a, cur_node_b, stack, obb_a, obb_b, qab, rab_rot, obb_c)
                && test_narrow_phase_overlap_intersection(a, b, c, ab_t, ac_t,
                    query_node_a, query_node_b, cur_node_c, err, sweep_radius_a, sweep_radius_b, sweep_radius_c))
                        return true;
            }
        }

    return false;
    }
} // end namespace hpmc

#undef DEVICE
#undef HOSTDEVICE
