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

//! Data structure for shape composed of a union of multiple shapes
template<class Shape>
struct union_params : param_base
    {
    typedef GPUTree gpu_tree_type; //!< Handy typedef for GPUTree template
    typedef typename Shape::param_type mparam_type;

    //! Default constructor
    DEVICE union_params()
        : mpos(), morientation(), mparams(),
          moverlap(), diameter(0.0), N(0), ignore(0)
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

    //! Determine size of the shared memory allocation
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

    //! Shape constructor
    union_params(unsigned int _N, bool _managed)
        : N(_N)
        {
        mpos = ManagedArray<vec3<OverlapReal> >(N,_managed);
        morientation = ManagedArray<quat<OverlapReal> >(N,_managed);
        mparams = ManagedArray<mparam_type>(N,_managed);
        moverlap = ManagedArray<unsigned int>(N,_managed);
        }
    #ifndef __HIPCC__
    union_params(pybind11::dict v)
        : union_params(pybind11::len(v["members"]), false)
        {
        pybind11::list _members = v["members"];
        pybind11::list positions = v["positions"];
        pybind11::list orientations = v["orientations"];
        pybind11::list overlap = v["overlap"];
        ignore = v["ignore_statistics"].cast<unsigned int>();
        unsigned int leaf_capacity = v["capacity"].cast<unsigned int>();

        if (pybind11::len(positions) != pybind11::len(_members))
            {
            throw std::runtime_error("Number of member positions not equal to number of members");
            }
        if (pybind11::len(orientations) != pybind11::len(_members))
            {
            throw std::runtime_error("Number of member orientations not equal to number of members");
            }

        if (pybind11::len(overlap) != pybind11::len(_members))
            {
            throw std::runtime_error("Number of member overlap flags not equal to number of members");
            }


        hpmc::detail::OBB *obbs = new hpmc::detail::OBB[pybind11::len(_members)];

        std::vector<std::vector<vec3<OverlapReal> > > internal_coordinates;

        // extract member parameters, positions, and orientations and compute the radius along the way
        diameter = OverlapReal(0.0);

        // compute a tight fitting AABB in the body frame
        detail::AABB local_aabb(vec3<OverlapReal>(0,0,0),OverlapReal(0.0));

        for (unsigned int i = 0; i < pybind11::len(_members); i++)
            {
            typename Shape::param_type param = pybind11::cast<typename Shape::param_type>(_members[i]);
            pybind11::list positions_i = pybind11::cast<pybind11::list>(positions[i]);
            vec3<OverlapReal> pos = vec3<OverlapReal>(pybind11::cast<OverlapReal>(positions_i[0]), pybind11::cast<OverlapReal>(positions_i[1]), pybind11::cast<OverlapReal>(positions_i[2]));
            pybind11::list orientations_i = pybind11::cast<pybind11::list>(orientations[i]);
            OverlapReal s = pybind11::cast<OverlapReal>(orientations_i[0]);
            OverlapReal x = pybind11::cast<OverlapReal>(orientations_i[1]);
            OverlapReal y = pybind11::cast<OverlapReal>(orientations_i[2]);
            OverlapReal z = pybind11::cast<OverlapReal>(orientations_i[3]);
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
        tree_obb.buildTree(obbs, pybind11::len(_members), leaf_capacity, false);
        delete [] obbs;
        tree = gpu_tree_type(tree_obb, false);

        // store local AABB
        lower = local_aabb.getLower();
        upper = local_aabb.getUpper();

        }
    pybind11::dict asDict()
        {
        pybind11::dict v;

        pybind11::list positions;
        pybind11::list orientations;
        pybind11::list overlaps;
        pybind11::list members;

        for (unsigned int i = 0; i < N; i++)
            {
            pybind11::list pos_i;
            pos_i.append(mpos[i].x);
            pos_i.append(mpos[i].y);
            pos_i.append(mpos[i].z);
            pybind11::tuple pos_tuple = pybind11::tuple(pos_i);
            positions.append(pos_tuple);

           // quat<OverlapReal> orientation_i = morientation[i];
            //QuatIterator<OverlapReal> begin = orientation_i.begin();
            //QuatIterator<OverlapReal> end = orientation_i.end();
            //OverlapReal s = quat<OverlapReal>(begin, begin);
           // vec3<OverlapReal> orientation_vec = quat<OverlapReal>(end, end);
            pybind11::list orientation_list;
            //orientation_list.append(s);
            //orientation_list.append(orientation_vec.x);
            //orientation_list.append(orientation_vec.y);
            //orientation_list.append(orientation_vec.z);
            orientation_list.append(morientation[i].s);
            //orientation_vec = morientation[i].v;
            orientation_list.append(morientation[i].v.x);
            orientation_list.append(morientation[i].v.y);
            orientation_list.append(morientation[i].v.z);
            pybind11::tuple orientation_tuple = pybind11::tuple(orientation_list);
            orientations.append(orientation_tuple);

            overlaps.append(moverlap[i]);
            members.append(mparams[i].asDict());
            //members.append(mparams[i]);
            }
        v["members"] = members;
        v["orientations"] = orientations;
        v["positions"] = positions;
        v["overlap"] = overlaps;
        v["ignore_statistics"] = ignore;
        v["capacity"] = tree.getLeafNodeCapacity();

        return v;
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
    vec3<OverlapReal> lower;                 //!< Lower corner of local AABB
    vec3<OverlapReal> upper;                 //!< Upper corner of local AABB
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

    #ifndef __HIPCC__
    std::string getShapeSpec() const
        {
        throw std::runtime_error("Shape definition not supported for this shape class.");
        }
    #endif

    //! Return the bounding box of the shape in world coordinates
    DEVICE detail::AABB getAABB(const vec3<Scalar>& pos) const
        {
        //return detail::AABB(pos, members.diameter/OverlapReal(2.0));

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

    //! Return a tight fitting OBB
    DEVICE detail::OBB getOBB(const vec3<Scalar>& pos) const
        {
        // get the root node OBB from the tree
        detail::OBB obb = members.tree.getOBB(0);

        // transform it into world-space
        obb.affineTransform(orientation, pos);

        return obb;
        }

    //! Returns true if this shape splits the overlap check over several threads of a warp using threadIdx.x
    HOSTDEVICE static bool isParallel() {
        #ifdef SHAPE_UNION_LEAVES_AGAINST_TREE_TRAVERSAL
        return true;
        #else
        return false;
        #endif
        }

    //! Returns true if the overlap check supports sweeping both shapes by a sphere of given radius
    HOSTDEVICE static bool supportsSweepRadius()
        {
        return Shape::supportsSweepRadius();
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
                                             unsigned int &err,
                                             OverlapReal sweep_radius_a,
                                             OverlapReal sweep_radius_b,
                                             bool ignore_mask)
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
    #ifdef __HIPCC__
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
        for (unsigned int cur_leaf_a = offset; cur_leaf_a < tree_a.getNumLeaves(); cur_leaf_a += stride)
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
                    test_narrow_phase_overlap(r_ab, a, b, cur_node_a, query_node, err, sweep_radius_a,sweep_radius_b, ignore_mask))
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

            // extend OBB
            obb_b.lengths.x += sab;
            obb_b.lengths.y += sab;
            obb_b.lengths.z += sab;

            unsigned cur_node_a = 0;
            while (cur_node_a < tree_a.getNumNodes())
                {
                unsigned int query_node = cur_node_a;
                if (tree_a.queryNode(obb_b, cur_node_a, ignore_mask) &&
                    test_narrow_phase_overlap(-r_ab, b, a, cur_node_b, query_node, err, sweep_radius_a,sweep_radius_b, ignore_mask))
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
    //! Param type of the member shapes
    typedef typename Shape::param_type mparam_type;

    unsigned int na = a.members.tree.getNumParticles(cur_node_a);
    unsigned int nb = b.members.tree.getNumParticles(cur_node_b);
    unsigned int nc = c.members.tree.getNumParticles(cur_node_c);

    // loop through shapes of cur_node_a
    for (unsigned int i= 0; i < na; i++)
        {
        unsigned int ishape = a.members.tree.getParticle(cur_node_a, i);

        const mparam_type& params_i = a.members.mparams[ishape];
        Shape shape_i(quat<Scalar>(), params_i);
        if (shape_i.hasOrientation())
            shape_i.orientation = quat<OverlapReal>(a.orientation)*a.members.morientation[ishape];

        vec3<OverlapReal> pos_i(rotate(quat<OverlapReal>(a.orientation),a.members.mpos[ishape]));
        unsigned int overlap_i = a.members.moverlap[ishape];

        // loop through shapes of cur_node_b
        for (unsigned int j= 0; j < nb; j++)
            {
            unsigned int jshape = b.members.tree.getParticle(cur_node_b, j);

            const mparam_type& params_j = b.members.mparams[jshape];
            Shape shape_j(quat<Scalar>(), params_j);
            if (shape_j.hasOrientation())
                shape_j.orientation = quat<OverlapReal>(b.orientation)*b.members.morientation[jshape];

            vec3<OverlapReal> pos_ij(rotate(quat<OverlapReal>(b.orientation),b.members.mpos[jshape]) + ab_t - pos_i);
            unsigned int overlap_j = b.members.moverlap[jshape];

            // loop through shapes of cur_node_c
            for (unsigned int k= 0; k < nc; k++)
                {
                unsigned int kshape = c.members.tree.getParticle(cur_node_c, k);

                const mparam_type& params_k = c.members.mparams[kshape];
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

//! Test for overlap of a third particle with the intersection of two shapes
/*! \param a First shape to test
    \param b Second shape to test
    \param c Third shape to test
    \param ab_t Position of second shape relative to first
    \param ac_t Position of third shape relative to first
    \param err Output variable that is incremented upon non-convergence
    \param sweep_radius_a Radius of a sphere to sweep the first shape by
    \param sweep_radius_b Radius of a sphere to sweep the second shape by
    \param sweep_radius_c Radius of a sphere to sweep the third shape by
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
#endif // end __SHAPE_UNION_H__
