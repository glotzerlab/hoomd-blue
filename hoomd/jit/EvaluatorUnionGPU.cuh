// Copyright (c) 2009-2019 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.

#ifndef __EVALUATOR_UNION_CUH__
#define __EVALUATOR_UNION_CUH__

#include "hoomd/jit/Evaluator.cuh"
#include "hoomd/HOOMDMath.h"
#include "hoomd/VectorMath.h"
#include "hoomd/hpmc/GPUTree.h"
#include "hoomd/ManagedArray.h"
#include "hoomd/RandomNumbers.h"
#include "hoomd/RNGIdentifiers.h"

#ifdef __HIPCC__
#define HOSTDEVICE __host__ __device__
#define DEVICE __device__
#else
#define HOSTDEVICE
#define DEVICE
#endif

namespace jit
{

//! Data structure for shape composed of a union of multiple shapes
struct __attribute__((aligned(sizeof(Scalar4)))) // align to largest data type used in shared memory storage
union_params_t
    {
    //! Default constructor
    HOSTDEVICE union_params_t()
        : N(0)
        { }

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
    };

#ifdef __HIPCC__
// Storage for shape parameters
static __device__ union_params_t *d_union_params;

//! Device storage of rcut value
static __device__ float d_rcut_union;
static __device__ float d_rcut_union_repulsive;

__device__ inline float compute_leaf_leaf_energy(const union_params_t* params,
                             float r_cut,
                             const vec3<float>& dr_old,
                             const vec3<float>& dr_new,
                             unsigned int type_a,
                             unsigned int type_b,
                             const quat<float>& orientation_a_old,
                             const quat<float>& orientation_a_new,
                             const quat<float>& orientation_b,
                             unsigned int cur_node_a,
                             unsigned int cur_node_b,
                             bool old_config,
                             unsigned int seed_ij,
                             bool check_early_exit,
                             bool &early_exit)
    {
    float energy = 0.0;
    vec3<float> r_ij_old = rotate(conj(quat<float>(orientation_b)),vec3<float>(dr_old));
    vec3<float> r_ij_new = rotate(conj(quat<float>(orientation_b)),vec3<float>(dr_new));

    // loop through leaf particles of cur_node_a
    unsigned int na = params[type_a].tree.getNumParticles(cur_node_a);
    unsigned int nb = params[type_b].tree.getNumParticles(cur_node_b);

    unsigned int leafptr_i = params[type_a].tree.getLeafNodePtrByNode(cur_node_a);
    unsigned int leafptr_j = params[type_b].tree.getLeafNodePtrByNode(cur_node_b);

    for (unsigned int i = 0; i < na; i++)
        {
        unsigned int ileaf = params[type_a].tree.getParticleByIndex(leafptr_i+i);

        unsigned int type_i = params[type_a].mtype[ileaf];
        quat<float> orientation_i_old = conj(quat<float>(orientation_b))*quat<float>(orientation_a_old) * params[type_a].morientation[ileaf];
        quat<float> orientation_i_new = conj(quat<float>(orientation_b))*quat<float>(orientation_a_new) * params[type_a].morientation[ileaf];
        vec3<float> pos_i_old(rotate(conj(quat<float>(orientation_b))*quat<float>(orientation_a_old),params[type_a].mpos[ileaf])-r_ij_old);
        vec3<float> pos_i_new(rotate(conj(quat<float>(orientation_b))*quat<float>(orientation_a_new),params[type_a].mpos[ileaf])-r_ij_new);

        // loop through leaf particles of cur_node_b
        for (unsigned int j= 0; j < nb; j++)
            {
            unsigned int jleaf = params[type_b].tree.getParticleByIndex(leafptr_j+j);

            unsigned int type_j = params[type_b].mtype[jleaf];
            quat<float> orientation_j = params[type_b].morientation[jleaf];
            vec3<float> r_ij = params[type_b].mpos[jleaf] - (old_config ? pos_i_old : pos_i_new);

            float rsq = dot(r_ij,r_ij);
            float Uij = 0.0f;
            float rcut_total = r_cut+0.5*(params[type_a].mdiameter[ileaf] + params[type_b].mdiameter[jleaf]);
            if (rsq <= rcut_total*rcut_total)
                {
                // evaluate energy via JIT function
                Uij = ::eval(r_ij,
                    type_i,
                    old_config ? orientation_i_old : orientation_i_new,
                    params[type_a].mdiameter[ileaf],
                    params[type_a].mcharge[ileaf],
                    type_j,
                    orientation_j,
                    params[type_b].mdiameter[jleaf],
                    params[type_b].mcharge[jleaf]);
                }


            if (check_early_exit)
                {
                // check the other config of particle i, too
                r_ij = params[type_b].mpos[jleaf] - (old_config ? pos_i_new : pos_i_old);
                rsq = dot(r_ij,r_ij);
                float Vij = 0.0f;
                if (rsq <= rcut_total*rcut_total)
                    {
                    // evaluate energy via JIT function
                    Vij = ::eval(r_ij,
                        type_i,
                        old_config ? orientation_i_new : orientation_i_old,
                        params[type_a].mdiameter[ileaf],
                        params[type_a].mcharge[ileaf],
                        type_j,
                        orientation_j,
                        params[type_b].mdiameter[jleaf],
                        params[type_b].mcharge[jleaf]);
                    }
                float deltaU = Uij - Vij;
                if ((old_config && deltaU < 0.0f) || (!old_config && deltaU > 0.0f))
                    {
                    // factorize this ij contribution to the MH probability out
                    hoomd::RandomGenerator rng_ij(hoomd::RNGIdentifier::HPMCJITPairs+1, seed_ij, ileaf, jleaf);
                    early_exit |= hoomd::detail::generate_canonical<float>(rng_ij) > slow::exp((old_config ? deltaU : -deltaU));
                    if (early_exit)
                        {
                        return energy;
                        }
                    }
                else
                    {
                    // attractive contribution, keep
                    energy += Uij;
                    }
                }
            else
                {
                energy += Uij;
                }
            }
        }
    return energy;
    }

__device__ inline float eval_union(const union_params_t *params,
    const vec3<float>& r_ij_old,
    const vec3<float>& r_ij_new,
    unsigned int type_i,
    const quat<float>& q_i_old,
    const quat<float>& q_i_new,
    float d_i,
    float charge_i,
    unsigned int type_j,
    const quat<float>& q_j,
    float d_j,
    float charge_j,
    bool old_config,
    unsigned int seed_ij,
    bool &early_exit)
    {
    const hpmc::detail::GPUTree& tree_a = params[type_i].tree;
    const hpmc::detail::GPUTree& tree_b = params[type_j].tree;

    // perform a tandem tree traversal
    unsigned long int stack = 0;
    unsigned int cur_node_a = 0;
    unsigned int cur_node_b = 0;

    vec3<float> dr_rot(rotate(conj(q_j),-(old_config ? r_ij_old : r_ij_new)));
    quat<float> q(conj(q_j)*(old_config ? q_i_old : q_i_new));

    hpmc::detail::OBB obb_a = tree_a.getOBB(cur_node_a);
    obb_a.affineTransform(q, dr_rot);

    hpmc::detail::OBB obb_b = tree_b.getOBB(cur_node_b);

    unsigned int query_node_a = 0xffffffffu;
    unsigned int query_node_b = 0xffffffffu;

    // load from device global variable
    float r_cut = d_rcut_union;
    float r_cut2 = 0.5f*r_cut;

    float r_cut_repulsive = d_rcut_union_repulsive;
    float r_cut_repulsive2 = 0.5f*d_rcut_union_repulsive;

    bool check_early_exit = r_cut_repulsive >= 0.0f;
    if (check_early_exit)
        {
        while (cur_node_a != tree_a.getNumNodes() && cur_node_b != tree_b.getNumNodes())
            {
            // extend OBBs by distributing the cut-off symmetrically
            if (query_node_a != cur_node_a)
                {
                obb_a.lengths.x += r_cut_repulsive2;
                obb_a.lengths.y += r_cut_repulsive2;
                obb_a.lengths.z += r_cut_repulsive2;
                query_node_a = cur_node_a;
                }

            if (query_node_b != cur_node_b)
                {
                obb_b.lengths.x += r_cut_repulsive2;
                obb_b.lengths.y += r_cut_repulsive2;
                obb_b.lengths.z += r_cut_repulsive2;
                query_node_b = cur_node_b;
                }

            if (hpmc::detail::traverseBinaryStack(tree_a, tree_b, cur_node_a, cur_node_b, stack, obb_a, obb_b, q, dr_rot, true))
                {
                // check for early exit only, ignore total energy
                compute_leaf_leaf_energy(params, r_cut, r_ij_old, r_ij_new,
                    type_i, type_j, q_i_old, q_i_new,
                    q_j, query_node_a, query_node_b,
                    old_config, seed_ij, check_early_exit, early_exit);
                if (early_exit)
                    {
                    return 0.0f;
                    }
                }
            }
        }

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

        if (hpmc::detail::traverseBinaryStack(tree_a, tree_b, cur_node_a, cur_node_b, stack, obb_a, obb_b, q, dr_rot, true))
            {
            energy += compute_leaf_leaf_energy(params, r_cut, r_ij_old, r_ij_new,
                type_i, type_j, q_i_old, q_i_new,
                q_j, query_node_a, query_node_b,
                old_config, seed_ij, check_early_exit,  early_exit);
            if (early_exit)
                {
                return energy;
                }
            }
        }
    return energy;
    }
#endif // __HIPCC__

#undef HOSTDEVICE
#undef DEVICE
} // end namespace jit

#endif // __EVALUATOR_UNION_CUH__
