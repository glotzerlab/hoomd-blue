// Copyright (c) 2009-2019 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.

// Original license
// Copyright (c) 2018, Michael P. Howard.
// This file is released under the Modified BSD License.

// Maintainer: mphoward

#include "LBVH.h"

namespace neighbor
{

/*!
 * \param exec_conf HOOMD-blue execution configuration
 * \param stream CUDA stream for kernel execution.
 *
 * The constructor defers memory initialization to the first call to ::build.
 */
LBVH::LBVH(std::shared_ptr<const ExecutionConfiguration> exec_conf, cudaStream_t stream)
    : m_exec_conf(exec_conf), m_stream(stream),
      m_root(gpu::LBVHSentinel), m_N(0), m_N_internal(0), m_N_nodes(0)
    {
    m_exec_conf->msg->notice(4) << "Constructing LBVH" << std::endl;

    m_tune_gen_codes.reset(new Autotuner(32, 1024, 32, 5, 100000, "lbvh_gen_codes", m_exec_conf));
    m_tune_gen_tree.reset(new Autotuner(32, 1024, 32, 5, 100000, "lbvh_gen_tree", m_exec_conf));
    m_tune_bubble.reset(new Autotuner(32, 1024, 32, 5, 100000, "lbvh_bubble", m_exec_conf));
    }

LBVH::~LBVH()
    {
    m_exec_conf->msg->notice(4) << "Destroying LBVH" << std::endl;
    }

/*!
 * \param N Number of primitives
 *
 * Initializes the memory for an LBVH holding \a N primitives. The memory
 * requirements are O(N). Every node is allocated 1 integer (4B) holding the parent
 * node and 2 float3s (24B) holding the bounding box. Each internal node additional
 * is allocated 2 integers (8B) holding their children and 1 integer (4B) holding a
 * flag used to backpropagate the bounding boxes.
 *
 * Primitive sorting requires 4N integers of storage, which is allocated persistently
 * to avoid the overhead of repeated malloc / free calls.
 *
 * \note
 * Additional calls to allocate are ignored if \a N has not changed from
 * the previous call.
 *
 */
void LBVH::allocate(unsigned int N)
    {
    m_root = 0;
    m_N = N;
    m_N_internal = (m_N > 0) ? m_N - 1 : 0;
    m_N_nodes = m_N + m_N_internal;

    if (m_N_nodes > m_parent.getNumElements())
        {
        GlobalArray<int> parent(m_N_nodes, m_exec_conf);
        m_parent.swap(parent);
        }

    if (m_N_internal > m_left.getNumElements())
        {
        GlobalArray<int> left(m_N_internal, m_exec_conf);
        m_left.swap(left);

        GlobalArray<int> right(m_N_internal, m_exec_conf);
        m_right.swap(right);

        GlobalArray<unsigned int> locks(m_N_internal, m_exec_conf);
        m_locks.swap(locks);
        }

    if (m_N_nodes > m_lo.getNumElements())
        {
        GlobalArray<float3> lo(m_N_nodes, m_exec_conf);
        m_lo.swap(lo);

        GlobalArray<float3> hi(m_N_nodes, m_exec_conf);
        m_hi.swap(hi);
        }

    if (m_N > m_codes.getNumElements())
        {
        GlobalArray<unsigned int> codes(m_N, m_exec_conf);
        m_codes.swap(codes);

        GlobalArray<unsigned int> indexes(m_N, m_exec_conf);
        m_indexes.swap(indexes);

        GlobalArray<unsigned int> sorted_codes(m_N, m_exec_conf);
        m_sorted_codes.swap(sorted_codes);

        GlobalArray<unsigned int> sorted_indexes(m_N, m_exec_conf);
        m_sorted_indexes.swap(sorted_indexes);
        }
    }

} // end namespace neighbor
