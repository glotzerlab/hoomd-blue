// Copyright (c) 2009-2024 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

#pragma once

#include "hoomd/HOOMDMath.h"
#include "hoomd/ManagedArray.h"
#include "hoomd/VectorMath.h"
#include "hoomd/hpmc/Evaluator.cuh"
#include "hoomd/hpmc/GPUTree.h"

#ifdef __HIPCC__
#define HOSTDEVICE __host__ __device__
#define DEVICE __device__
#else
#define HOSTDEVICE
#define DEVICE
#endif

namespace hoomd
    {
namespace hpmc
    {
namespace jit
    {
//! Data structure for shape composed of a union of multiple shapes
struct __attribute__((
    aligned(sizeof(Scalar4)))) // align to largest data type used in shared memory storage
union_params_t
    {
    //! Default constructor
    HOSTDEVICE union_params_t() : N(0) { }

    //! Load dynamic data members into shared memory and increase pointer
    /*! \param ptr Pointer to load data to (will be incremented)
        \param available_bytes Size of remaining shared memory allocation
     */
    DEVICE void load_shared(char*& ptr, unsigned int& available_bytes)
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
    HOSTDEVICE void allocate_shared(char*& ptr, unsigned int& available_bytes) const
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

#ifndef __HIPCC__
    //! Shape constructor
    union_params_t(unsigned int _N, bool _managed) : N(_N)
        {
        mpos = ManagedArray<vec3<float>>(N, _managed);
        morientation = ManagedArray<quat<float>>(N, _managed);
        mdiameter = ManagedArray<float>(N, _managed);
        mcharge = ManagedArray<float>(N, _managed);
        mtype = ManagedArray<unsigned int>(N, _managed);
        }
#endif

    hpmc::detail::GPUTree tree;             //!< OBB tree for constituent particles
    ManagedArray<vec3<float>> mpos;         //!< Position vectors of constituent particles
    ManagedArray<quat<float>> morientation; //!< Orientation of constituent particles
    ManagedArray<float> mdiameter;          //!< Diameters of constituent particles
    ManagedArray<float> mcharge;            //!< Charges of constituent particles
    ManagedArray<unsigned int> mtype;       //!< Types of constituent particles
    unsigned int N;                         //!< Number of member shapes
    };

#ifdef __HIPCC__
// Storage for shape parameters
__device__ union_params_t* d_union_params;

//! Device storage of rcut value
__device__ float d_r_cut_constituent;

__device__ inline float compute_leaf_leaf_energy(const union_params_t* params,
                                                 float r_cut,
                                                 const vec3<float>& dr,
                                                 unsigned int type_a,
                                                 unsigned int type_b,
                                                 const quat<float>& orientation_a,
                                                 const quat<float>& orientation_b,
                                                 unsigned int cur_node_a,
                                                 unsigned int cur_node_b)
    {
    float energy = 0.0;

    // loop through leaf particles of cur_node_a
    // parallel loop over N^2 interacting particle pairs
    unsigned int ptl_i = params[type_a].tree.getLeafNodePtrByNode(cur_node_a);
    unsigned int ptl_j = params[type_b].tree.getLeafNodePtrByNode(cur_node_b);

    unsigned int ptls_i_end = params[type_a].tree.getLeafNodePtrByNode(cur_node_a + 1);
    unsigned int ptls_j_end = params[type_b].tree.getLeafNodePtrByNode(cur_node_b + 1);

    // get starting offset for this thread
    unsigned int nb = ptls_j_end - ptl_j;
    if (nb == 0)
        return 0.0;

    ptl_i += threadIdx.x / nb;
    ptl_j += threadIdx.x % nb;

    vec3<float> r_ij = rotate(conj(orientation_b), dr);

    while ((ptl_i < ptls_i_end) && (ptl_j < ptls_j_end))
        {
        unsigned int ileaf = params[type_a].tree.getParticleByIndex(ptl_i);
        unsigned int type_i = params[type_a].mtype[ileaf];

        quat<float> orientation_i
            = conj(orientation_b) * orientation_a * params[type_a].morientation[ileaf];
        vec3<float> pos_i(rotate(conj(orientation_b) * orientation_a, params[type_a].mpos[ileaf])
                          - r_ij);

        unsigned int jleaf = params[type_b].tree.getParticleByIndex(ptl_j);
        unsigned int type_j = params[type_b].mtype[jleaf];
        quat<float> orientation_j = params[type_b].morientation[jleaf];
        vec3<float> r_ij_local = params[type_b].mpos[jleaf] - pos_i;

        float rsq = dot(r_ij_local, r_ij_local);
        float d_i = params[type_a].mdiameter[ileaf];
        float d_j = params[type_b].mdiameter[jleaf];
        if (rsq <= r_cut * r_cut)
            {
            // evaluate energy via JIT function
            energy += ::eval(r_ij_local,
                             type_i,
                             orientation_i,
                             d_i,
                             params[type_a].mcharge[ileaf],
                             type_j,
                             orientation_j,
                             d_j,
                             params[type_b].mcharge[jleaf]);
            }

        // increment counters
        ptl_j += blockDim.x;
        while (ptl_j >= ptls_j_end)
            {
            ptl_j -= nb;
            ptl_i++;

            if (ptl_i == ptls_i_end)
                break;
            }
        }
    return energy;
    }

__device__ inline float eval_union(const union_params_t* params,
                                   const vec3<float>& r_ij,
                                   unsigned int type_i,
                                   const quat<float>& q_i,
                                   float d_i,
                                   float charge_i,
                                   unsigned int type_j,
                                   const quat<float>& q_j,
                                   float d_j,
                                   float charge_j)
    {
    const hpmc::detail::GPUTree& tree_a = params[type_i].tree;
    const hpmc::detail::GPUTree& tree_b = params[type_j].tree;

    // load from device global variable
    float r_cut = d_r_cut_constituent;
    float r_cut2 = 0.5f * r_cut;

    // perform a tandem tree traversal
    unsigned long int stack = 0;
    unsigned int cur_node_a = 0;
    unsigned int cur_node_b = 0;

    vec3<float> dr_rot(rotate(conj(q_j), -r_ij));
    quat<float> q(conj(q_j) * q_i);

    hpmc::detail::OBB obb_a = tree_a.getOBB(cur_node_a);
    obb_a.affineTransform(q, dr_rot);

    hpmc::detail::OBB obb_b = tree_b.getOBB(cur_node_b);

    unsigned int query_node_a = 0xffffffffu;
    unsigned int query_node_b = 0xffffffffu;

    float energy = 0.0f;
    while (cur_node_a != tree_a.getNumNodes() && cur_node_b != tree_b.getNumNodes())
        {
        // extend OBBs by distributing the cut-off symmetrically
        if (query_node_a != cur_node_a)
            {
            obb_a.lengths.x += r_cut2;
            obb_a.lengths.y += r_cut2;
            obb_a.lengths.z += r_cut2;
            query_node_a = cur_node_a;
            }

        if (query_node_b != cur_node_b)
            {
            obb_b.lengths.x += r_cut2;
            obb_b.lengths.y += r_cut2;
            obb_b.lengths.z += r_cut2;
            query_node_b = cur_node_b;
            }

        if (hpmc::detail::traverseBinaryStack(tree_a,
                                              tree_b,
                                              cur_node_a,
                                              cur_node_b,
                                              stack,
                                              obb_a,
                                              obb_b,
                                              q,
                                              dr_rot))
            {
            energy += compute_leaf_leaf_energy(params,
                                               r_cut,
                                               r_ij,
                                               type_i,
                                               type_j,
                                               q_i,
                                               q_j,
                                               query_node_a,
                                               query_node_b);
            }
        }
    return energy;
    }
#endif // __HIPCC__

#undef HOSTDEVICE
#undef DEVICE
    } // end namespace jit

    } // end namespace hpmc
    } // end namespace hoomd
