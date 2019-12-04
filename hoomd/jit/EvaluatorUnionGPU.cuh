// Copyright (c) 2009-2019 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.

#pragma once

#include "hoomd/jit/Evaluator.cuh"
#include "hoomd/HOOMDMath.h"
#include "hoomd/VectorMath.h"
#include "hoomd/hpmc/GPUTree.h"
#include "hoomd/ManagedArray.h"

#undef HOSTDEVICE
#ifdef __HIPCC__
#define HOSTDEVICE __host__ __device__
#else
#define HOSTDEVICE
#endif

namespace jit
{

//! Data structure for shape composed of a union of multiple shapes
struct union_params_t
    {
    //! Default constructor
    HOSTDEVICE union_params_t()
        : N(0)
        { }

    // this initial implementation is probably not performant, as it does not use shared memory

    #if 0
    //! Load dynamic data members into shared memory and increase pointer
    /*! \param ptr Pointer to load data to (will be incremented)
        \param available_bytes Size of remaining shared memory allocation
     */
    DEVICE void load_shared(char *& ptr, unsigned int &available_bytes)
        {
        tree.load_shared(ptr, available_bytes);
        mpos.load_shared(ptr, available_bytes);
        morientation.load_shared(ptr, available_bytes);
        mdiameter.load_shared(ptr, available_bytes);
        mcharge.load_shared(ptr, available_bytes);
        mtype.load_shared(ptr, available_bytes);
        }

    //! Determine size of the shared memory allocaation
    /*! \param ptr Pointer to increment
        \param available_bytes Size of remaining shared memory allocation
     */
    HOSTDEVICE void allocate_shared(char *& ptr, unsigned int &available_bytes) const
        {
        tree.allocate_shared(ptr, available_bytes);
        mpos.allocate_shared(ptr, available_bytes);
        morientation.allocate_shared(ptr, available_bytes);
        mdiameter.allocate_shared(ptr, available_bytes);
        mcharge.allocate_shared(ptr, available_bytes);
        mtype.allocate_shared(ptr, available_bytes);
        }

    #ifdef ENABLE_HIP
    //! Set CUDA memory hints
    void set_memory_hint() const
        {
        tree.set_memory_hint();

        mpos.set_memory_hint();
        morientation.set_memory_hint();
        mdiameter.set_memory_hint();
        mcharge.set_memory_hint();
        mtype.set_memory_hint();
        }
    #endif
    #endif

    #ifndef __HIPCC__
    //! Shape constructor
    union_params_t(unsigned int _N, bool _managed)
        : N(_N)
        {
        mpos = ManagedArray<vec3<float> >(N,_managed);
        morientation = ManagedArray<quat<float> >(N,_managed);
        mdiameter = ManagedArray<float>(N,_managed);
        mcharge = ManagedArray<float>(N,_managed);
        mtype = ManagedArray<unsigned int>(N,_managed);
        }
    #endif

    hpmc::detail::GPUTree tree;                    //!< OBB tree for constituent particles
    ManagedArray<vec3<float> > mpos;         //!< Position vectors of constituent particles
    ManagedArray<quat<float> > morientation; //!< Orientation of constituent particles
    ManagedArray<float> mdiameter;           //!< Diameters of constituent particles
    ManagedArray<float> mcharge;             //!< Charges of constituent particles
    ManagedArray<unsigned int> mtype;        //!< Types of constituent particles
    unsigned int N;                           //!< Number of member shapes
    }; //__attribute__((aligned(32))); // aligned to sizeof(double4), for shared memory storage

#ifdef __HIPCC__
// Storage for shape parameters
static __device__ union_params_t *d_union_params;

//! Device storage of rcut value
static __device__ float d_rcut_union;

__device__ inline float compute_leaf_leaf_energy(float r_cut,
                             vec3<float> dr,
                             unsigned int type_a,
                             unsigned int type_b,
                             const quat<float>& orientation_a,
                             const quat<float>& orientation_b,
                             unsigned int cur_node_a,
                             unsigned int cur_node_b)
    {
    float energy = 0.0;
    vec3<float> r_ij = rotate(conj(quat<float>(orientation_b)),vec3<float>(dr));

    // loop through leaf particles of cur_node_a
    unsigned int na = d_union_params[type_a].tree.getNumParticles(cur_node_a);
    unsigned int nb = d_union_params[type_b].tree.getNumParticles(cur_node_b);

    for (unsigned int i= 0; i < na; i++)
        {
        unsigned int ileaf = d_union_params[type_a].tree.getParticle(cur_node_a, i);

        unsigned int type_i = d_union_params[type_a].mtype[ileaf];
        quat<float> orientation_i = conj(quat<float>(orientation_b))*quat<float>(orientation_a) * d_union_params[type_a].morientation[ileaf];
        vec3<float> pos_i(rotate(conj(quat<float>(orientation_b))*quat<float>(orientation_a),d_union_params[type_a].mpos[ileaf])-r_ij);

        // loop through leaf particles of cur_node_b
        for (unsigned int j= 0; j < nb; j++)
            {
            unsigned int jleaf = d_union_params[type_b].tree.getParticle(cur_node_b, j);

            unsigned int type_j = d_union_params[type_b].mtype[jleaf];
            quat<float> orientation_j = d_union_params[type_b].morientation[jleaf];
            vec3<float> r_ij = d_union_params[type_b].mpos[jleaf] - pos_i;

            float rsq = dot(r_ij,r_ij);
            if (rsq <= r_cut*r_cut)
                {
                // evaluate energy via JIT function
                energy += eval(r_ij,
                    type_i,
                    orientation_i,
                    d_union_params[type_a].mdiameter[ileaf],
                    d_union_params[type_a].mcharge[ileaf],
                    type_j,
                    orientation_j,
                    d_union_params[type_b].mdiameter[jleaf],
                    d_union_params[type_b].mcharge[jleaf]);
                }
            }
        }
    return energy;
    }

extern "C" {
__device__ static float eval_union(const vec3<float>& r_ij,
    unsigned int type_i,
    const quat<float>& q_i,
    float d_i,
    float charge_i,
    unsigned int type_j,
    const quat<float>& q_j,
    float d_j,
    float charge_j)
    {
    #if 0
    const hpmc::detail::GPUTree& tree_a = d_union_params[type_i].tree;
    const hpmc::detail::GPUTree& tree_b = d_union_params[type_j].tree;

    // load from device global variable
    float r_cut = d_rcut_union;

    // perform a tandem tree traversal
    unsigned long int stack = 0;
    unsigned int cur_node_a = 0;
    unsigned int cur_node_b = 0;

    vec3<float> dr_rot(rotate(conj(q_j),-r_ij));
    quat<float> q(conj(q_j)*q_i);

    hpmc::detail::OBB obb_a = tree_a.getOBB(cur_node_a);
    obb_a.affineTransform(q, dr_rot);

    hpmc::detail::OBB obb_b = tree_b.getOBB(cur_node_b);

    unsigned int query_node_a = 0xffffffffu;
    unsigned int query_node_b = 0xffffffffu;

    float energy = 0.0f;
    while (cur_node_a != tree_a.getNumNodes() && cur_node_b != tree_b.getNumNodes())
        {
        // extend OBBs
        if (query_node_a != cur_node_a)
            {
            obb_a.lengths.x += r_cut;
            obb_a.lengths.y += r_cut;
            obb_a.lengths.z += r_cut;
            query_node_a = cur_node_a;
            }

        if (query_node_b != cur_node_b)
            {
            obb_b.lengths.x += r_cut;
            obb_b.lengths.y += r_cut;
            obb_b.lengths.z += r_cut;
            query_node_b = cur_node_b;
            }

        if (hpmc::detail::traverseBinaryStack(tree_a, tree_b, cur_node_a, cur_node_b, stack, obb_a, obb_b, q, dr_rot, false))
            {
            energy += compute_leaf_leaf_energy(r_cut, r_ij, type_i, type_j, q_i, q_j, query_node_a, query_node_b);
            }
        }
    return energy;
    #else
    return 0.0f;
    #endif
    }
}
#endif // __HIPCC__

#undef HOSTDEVICE
} // end namespace jit
