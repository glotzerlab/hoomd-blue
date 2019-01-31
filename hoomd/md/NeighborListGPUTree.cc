// Copyright (c) 2009-2019 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.


// Maintainer: mphoward

/*! \file NeighborListGPUTree.cc
    \brief Defines NeighborListGPUTree
*/

#include "NeighborListGPUTree.h"
#include "NeighborListGPUTree.cuh"

namespace py = pybind11;

#ifdef ENABLE_MPI
#include "hoomd/Communicator.h"
#endif

using namespace std;

NeighborListGPUTree::NeighborListGPUTree(std::shared_ptr<SystemDefinition> sysdef,
                                       Scalar r_cut,
                                       Scalar r_buff)
    : NeighborListGPU(sysdef, r_cut, r_buff), m_type_changed(false), m_box_changed(true),
      m_max_num_changed(false), m_n_leaf(0), m_n_internal(0), m_n_node(0), m_n_images(0)
    {
    m_exec_conf->msg->notice(5) << "Constructing NeighborListGPUTree" << endl;

    m_pdata->getNumTypesChangeSignal().connect<NeighborListGPUTree, &NeighborListGPUTree::slotNumTypesChanged>(this);
    m_pdata->getBoxChangeSignal().connect<NeighborListGPUTree, &NeighborListGPUTree::slotBoxChanged>(this);
    m_pdata->getMaxParticleNumberChangeSignal().connect<NeighborListGPUTree, &NeighborListGPUTree::slotMaxNumChanged>(this);

    m_tuner_morton.reset(new Autotuner(32, 1024, 32, 5, 100000, "nlist_morton_codes", this->m_exec_conf));
    m_tuner_merge.reset(new Autotuner(32, 1024, 32, 5, 100000, "nlist_merge_particles", this->m_exec_conf));
    m_tuner_hierarchy.reset(new Autotuner(32, 1024, 32, 5, 100000, "nlist_gen_hierarchy", this->m_exec_conf));
    m_tuner_bubble.reset(new Autotuner(32, 1024, 32, 5, 100000, "nlist_bubble_aabbs", this->m_exec_conf));
    m_tuner_move.reset(new Autotuner(32, 1024, 32, 5, 100000, "nlist_move_particles", this->m_exec_conf));
    m_tuner_map.reset(new Autotuner(32, 1024, 32, 5, 100000, "nlist_map_particles", this->m_exec_conf));
    m_tuner_traverse.reset(new Autotuner(32, 1024, 32, 5, 100000, "nlist_traverse_tree", this->m_exec_conf));

    allocateTree();

    calcTypeBits();

    m_prev_ntypes = m_pdata->getNTypes();
    }

NeighborListGPUTree::~NeighborListGPUTree()
    {
    m_exec_conf->msg->notice(5) << "Destroying NeighborListGPUTree" << endl;
    m_pdata->getNumTypesChangeSignal().disconnect<NeighborListGPUTree, &NeighborListGPUTree::slotNumTypesChanged>(this);
    m_pdata->getBoxChangeSignal().disconnect<NeighborListGPUTree, &NeighborListGPUTree::slotBoxChanged>(this);
    m_pdata->getMaxParticleNumberChangeSignal().disconnect<NeighborListGPUTree, &NeighborListGPUTree::slotMaxNumChanged>(this);
    }

void NeighborListGPUTree::buildNlist(unsigned int timestep)
    {
    // kernels will crash in strange ways if there are no particles owned by the rank
    // so the build should just be aborted here (there are no neighbors to compute if there are no particles)
    if (!m_pdata->getN())
        {
        // maybe we should clear the arrays here, but really whoever's using the nlist should
        // just be smart enough to not try to use something that shouldn't exist
        return;
        }

    // allocate the tree memory as needed based on the mapping
    setupTree();

    // build the tree
    buildTree();

    // walk with the tree
    traverseTree();
    }

void NeighborListGPUTree::allocateTree()
    {
    // allocate per particle memory
    GPUArray<uint64_t> morton_types(m_pdata->getMaxN(), m_exec_conf);
    m_morton_types.swap(morton_types);
    GPUArray<uint64_t> morton_types_alt(m_pdata->getMaxN(), m_exec_conf);
    m_morton_types_alt.swap(morton_types_alt);

    GPUArray<unsigned int> map_tree_pid(m_pdata->getMaxN(), m_exec_conf);
    m_map_tree_pid.swap(map_tree_pid);
    GPUArray<unsigned int> map_tree_pid_alt(m_pdata->getMaxN(), m_exec_conf);
    m_map_tree_pid_alt.swap(map_tree_pid_alt);

    GPUArray<Scalar4> leaf_xyzf(m_pdata->getMaxN(), m_exec_conf);
    m_leaf_xyzf.swap(leaf_xyzf);

    GPUArray<Scalar2> leaf_db(m_pdata->getMaxN(), m_exec_conf);
    m_leaf_db.swap(leaf_db);

    // allocate per type memory
    GPUArray<unsigned int> leaf_offset(m_pdata->getNTypes(), m_exec_conf);
    m_leaf_offset.swap(leaf_offset);

    GPUArray<unsigned int> tree_roots(m_pdata->getNTypes(), m_exec_conf);
    m_tree_roots.swap(tree_roots);

    GPUArray<unsigned int> num_per_type(m_pdata->getNTypes(), m_exec_conf);
    m_num_per_type.swap(num_per_type);

    GPUArray<unsigned int> type_head(m_pdata->getNTypes(), m_exec_conf);
    m_type_head.swap(type_head);

    // allocate the tree memory to default lengths of 0 (will be resized later)
    // we use a GPUVector instead of GPUArray for the amortized resizing
    GPUVector<uint2> tree_parent_sib(m_exec_conf);
    m_tree_parent_sib.swap(tree_parent_sib);

    // holds two Scalar4s per node in tree
    GPUVector<Scalar4> tree_aabbs(m_exec_conf);
    m_tree_aabbs.swap(tree_aabbs);

    // we really only need as many morton codes as we have leafs
    GPUVector<uint32_t> morton_codes_red(m_exec_conf);
    m_morton_codes_red.swap(morton_codes_red);

    // 1 / 0 locks for traversing up the tree
    GPUVector<unsigned int> node_locks(m_exec_conf);
    m_node_locks.swap(node_locks);

    // conditions
    GPUFlags<int> morton_conditions(m_exec_conf);
    m_morton_conditions.swap(morton_conditions);
    }

/*!
 * \post Tree internal data structures are updated to begin a build.
 */
void NeighborListGPUTree::setupTree()
    {
    // increase arrays that depend on the local number of particles
    if (m_max_num_changed)
        {
        m_morton_types.resize(m_pdata->getMaxN());
        m_morton_types_alt.resize(m_pdata->getMaxN());
        m_map_tree_pid.resize(m_pdata->getMaxN());
        m_map_tree_pid_alt.resize(m_pdata->getMaxN());
        m_leaf_xyzf.resize(m_pdata->getMaxN());
        m_leaf_db.resize(m_pdata->getMaxN());

        // all done with the particle data reallocation
        m_max_num_changed = false;
        }

    // allocate memory that depends on type
    if (m_type_changed)
        {
        m_leaf_offset.resize(m_pdata->getNTypes());
        m_tree_roots.resize(m_pdata->getNTypes());
        m_num_per_type.resize(m_pdata->getNTypes());
        m_type_head.resize(m_pdata->getNTypes());

        // get the number of bits needed to represent all the types
        calcTypeBits();

        // all done with the type reallocation
        m_type_changed = false;
        m_prev_ntypes = m_pdata->getNTypes();
        }

    if (m_box_changed)
        {
        updateImageVectors();
        m_box_changed = false;
        }
    }

/*!
 * Determines the number of bits needed to represent the largest type index for more efficient particle sorting.
 * This is done by taking the ceiling of the log2 of the type index using integers.
 * \sa sortMortonCodes
 */
inline void NeighborListGPUTree::calcTypeBits()
    {
    if (m_pdata->getNTypes() > 1)
        {
        unsigned int n_type_bits = 0;

        // start with the maximum type id that there can be
        unsigned int tmp = m_pdata->getNTypes() - 1;

        // see how many times you can bit shift
        while (tmp >>= 1)
            {
            ++n_type_bits;
            }

        // add one to get the number of bits needed (rounding up int logarithm)
        m_n_type_bits = n_type_bits + 1;
        }
    else
        {
        // if there is only one type, you don't need to do any sorting
        m_n_type_bits = 0;
        }
    }

/*!
 * Determines the number of particles per type (and their starting indexes) in the flat leaf particle order. Also
 * determines the leaf offsets and and tree roots. When there is only one type, most operations are skipped since these
 * values are simple to determine.
 */
void NeighborListGPUTree::countParticlesAndTrees()
    {
    if (m_prof) m_prof->push(m_exec_conf,"map");

    if (m_pdata->getNTypes() > 1)
        {
        // first do the stuff with the particle data on the GPU to avoid a costly copy
            {
            ArrayHandle<unsigned int> d_type_head(m_type_head, access_location::device, access_mode::overwrite);
            ArrayHandle<Scalar4> d_pos(m_pdata->getPositions(), access_location::device, access_mode::read);
            ArrayHandle<unsigned int> d_map_tree_pid(m_map_tree_pid, access_location::device, access_mode::read);
            m_tuner_map->begin();
            gpu_nlist_init_count(d_type_head.data,
                                 d_pos.data,
                                 d_map_tree_pid.data,
                                 m_pdata->getN() + m_pdata->getNGhosts(),
                                 m_pdata->getNTypes(),
                                 m_tuner_map->getParam());
            if (m_exec_conf->isCUDAErrorCheckingEnabled())
                CHECK_CUDA_ERROR();
            m_tuner_map->end();
            }


        // then do the harder to parallelize stuff on the cpu because the number of types is usually small
        // so what's the point of trying this in parallel to save a copy of a few bytes?
            {
            // the number of leafs is the first tree root
            ArrayHandle<unsigned int> h_type_head(m_type_head, access_location::host, access_mode::read);
            ArrayHandle<unsigned int> h_num_per_type(m_num_per_type, access_location::host, access_mode::overwrite);
            ArrayHandle<unsigned int> h_leaf_offset(m_leaf_offset, access_location::host, access_mode::overwrite);
            ArrayHandle<unsigned int> h_tree_roots(m_tree_roots, access_location::host, access_mode::overwrite);

            // loop through the type heads and figure out how many there are of each
            m_n_leaf = 0;
            unsigned int total_offset = 0;
            unsigned int active_types = 0; // tracks the number of types that currently have particles
            for (unsigned int cur_type = 0; cur_type < m_pdata->getNTypes(); ++cur_type)
                {
                const unsigned int head_plus_1 = h_type_head.data[cur_type];

                unsigned int N_i = 0;

                if (head_plus_1 > 0) // there are particles of this type
                    {
                    // so loop over the types (we are ordered), and try to find a match
                    unsigned int next_head_plus_1 = 0;
                    for (unsigned int next_type = cur_type + 1; !next_head_plus_1 && next_type < m_pdata->getNTypes(); ++next_type)
                        {
                        if (h_type_head.data[next_type]) // this head exists
                            {
                            next_head_plus_1 = h_type_head.data[next_type];
                            }
                        }
                    // if we still haven't found a match, then the end index (+1) should be the end of the list
                    if (!next_head_plus_1)
                        {
                        next_head_plus_1 = m_pdata->getN() + m_pdata->getNGhosts() + 1;
                        }
                    N_i = next_head_plus_1 - head_plus_1;
                    }

                // set the number per type
                h_num_per_type.data[cur_type] = N_i;
                if (N_i > 0) ++active_types;

                // compute the number of leafs for this type, and accumulate it
                // temporarily stash the number of leafs in the tree root array
                unsigned int cur_n_leaf = (N_i + NLIST_GPU_PARTICLES_PER_LEAF - 1)/NLIST_GPU_PARTICLES_PER_LEAF;
                h_tree_roots.data[cur_type] = cur_n_leaf;
                m_n_leaf += cur_n_leaf;

                // compute the offset that is needed for this type, set and accumulate the total offset required
                const unsigned int remainder = N_i % NLIST_GPU_PARTICLES_PER_LEAF;
                const unsigned int cur_offset = (remainder > 0) ? (NLIST_GPU_PARTICLES_PER_LEAF - remainder) : 0;
                h_leaf_offset.data[cur_type] = total_offset;
                total_offset += cur_offset;
                }

            // each tree has Nleaf,i - 1 internal nodes
            // so in total we have N_leaf - N_types internal nodes for each type that has at least one particle
            m_n_internal = m_n_leaf - active_types;
            m_n_node = m_n_leaf + m_n_internal;

            // now loop over the roots one more time, and set each of them
            unsigned int leaf_head = 0;
            unsigned int internal_head = m_n_leaf;
            for (unsigned int cur_type = 0; cur_type < m_pdata->getNTypes(); ++cur_type)
                {
                const unsigned int n_leaf_i = h_tree_roots.data[cur_type];
                if (n_leaf_i == 0)
                    {
                    h_tree_roots.data[cur_type] = NLIST_GPU_INVALID_NODE;
                    }
                else if (n_leaf_i == 1)
                    {
                    h_tree_roots.data[cur_type] = leaf_head;
                    }
                else
                    {
                    h_tree_roots.data[cur_type] = internal_head;
                    internal_head += n_leaf_i - 1;
                    }
                leaf_head += n_leaf_i;
                }
            }
        }
    else // only one type
        {
        ArrayHandle<unsigned int> h_type_head(m_type_head, access_location::host, access_mode::overwrite);
        ArrayHandle<unsigned int> h_num_per_type(m_num_per_type, access_location::host, access_mode::overwrite);
        ArrayHandle<unsigned int> h_leaf_offset(m_leaf_offset, access_location::host, access_mode::overwrite);
        ArrayHandle<unsigned int> h_tree_roots(m_tree_roots, access_location::host, access_mode::overwrite);

        // with one type, we don't need to do anything fancy
        // type head is the first particle
        h_type_head.data[0] = 0;

        // num per type is all the particles in the rank
        h_num_per_type.data[0] = m_pdata->getN() + m_pdata->getNGhosts();

        // there is no leaf offset
        h_leaf_offset.data[0] = 0;

        // number of leafs is for all particles
        m_n_leaf = (m_pdata->getN() + m_pdata->getNGhosts() + NLIST_GPU_PARTICLES_PER_LEAF - 1)/NLIST_GPU_PARTICLES_PER_LEAF;

        // number of internal nodes is one less than number of leafs
        m_n_internal = m_n_leaf - 1;
        m_n_node = m_n_leaf + m_n_internal;

        // the root is the end of the leaf list if multiple leafs, otherwise the only leaf
        h_tree_roots.data[0] = (m_n_leaf > 1) ? m_n_leaf : 0;
        }

    if (m_prof) m_prof->pop(m_exec_conf);
    }

/*!
 * Driver to implement the tree build algorithm of Karras,
 * "Maximizing parallelism in the construction of BVHs, octrees, and k-d trees", High Performance Graphics (2012).
 * \post a valid tree is allocated and ready for traversal
 */
void NeighborListGPUTree::buildTree()
    {
    if (m_prof) m_prof->push(m_exec_conf,"Build tree");

    // step one: morton code calculation
    calcMortonCodes();

    // step two: particle sorting
    sortMortonCodes();

    // step three: map the particles by type
    countParticlesAndTrees();

    // (re-) allocate memory that depends on tree size
    // GPUVector should only do this as needed
    m_tree_parent_sib.resize(m_n_node);
    m_tree_aabbs.resize(2*m_n_node);
    m_morton_codes_red.resize(m_n_leaf);
    m_node_locks.resize(m_n_internal);

    // step four: merge leaf particles into aabbs by morton code
    mergeLeafParticles();

    // step five: hierarchy generation from morton codes
    genTreeHierarchy();

    // step six: bubble up the aabbs
    bubbleAABBs();

    if (m_prof) m_prof->pop(m_exec_conf);
    }

/*!
 * \post One morton code-type key is assigned per particle
 * \note Call before sortMortonCodes().
 */
void NeighborListGPUTree::calcMortonCodes()
    {
    if (m_prof) m_prof->push(m_exec_conf,"Morton codes");

    // need a ghost layer width to get the fractional position of particles in the local box
    const BoxDim& box = m_pdata->getBox();

        {
        // particle data and where to write it
        ArrayHandle<Scalar4> d_pos(m_pdata->getPositions(), access_location::device, access_mode::read);
        ArrayHandle<unsigned int> d_map_tree_pid(m_map_tree_pid, access_location::device, access_mode::overwrite);

        ArrayHandle<uint64_t> d_morton_types(m_morton_types, access_location::device, access_mode::overwrite);

        Scalar ghost_layer_width(0.0);
        #ifdef ENABLE_MPI
        if (m_comm) ghost_layer_width = m_comm->getGhostLayerMaxWidth();
        #endif

        Scalar3 ghost_width = make_scalar3(0.0, 0.0, 0.0);
        if (!box.getPeriodic().x) ghost_width.x = ghost_layer_width;
        if (!box.getPeriodic().y) ghost_width.y = ghost_layer_width;
        if (this->m_sysdef->getNDimensions() == 3 && !box.getPeriodic().z)
            {
            ghost_width.z = ghost_layer_width;
            }


        // reset the flag to zero before calling the compute
        m_morton_conditions.resetFlags(0);

        m_tuner_morton->begin();
        gpu_nlist_morton_types(d_morton_types.data,
                               d_map_tree_pid.data,
                               m_morton_conditions.getDeviceFlags(),
                               d_pos.data,
                               m_pdata->getN(),
                               m_pdata->getNGhosts(),
                               box,
                               ghost_width,
                               m_tuner_morton->getParam());
        if (m_exec_conf->isCUDAErrorCheckingEnabled())
            CHECK_CUDA_ERROR();
        m_tuner_morton->end();
        }

    // error check that no local particles are out of bounds
    const unsigned int morton_conditions = m_morton_conditions.readFlags();
    if (morton_conditions > 0)
        {
        ArrayHandle<Scalar4> h_pos(m_pdata->getPositions(), access_location::host, access_mode::read);
        ArrayHandle<unsigned int> h_tag(m_pdata->getTags(), access_location::host, access_mode::read);
        Scalar4 post_i = h_pos.data[morton_conditions-1];
        Scalar3 pos_i = make_scalar3(post_i.x, post_i.y, post_i.z);
        Scalar3 f = box.makeFraction(pos_i);
        m_exec_conf->msg->error() << "nlist.tree(): Particle " << h_tag.data[morton_conditions-1] << " is out of bounds "
                                  << "(x: " << post_i.x << ", y: " << post_i.y << ", z: " << post_i.z
                                  << ", fx: "<< f.x <<", fy: "<<f.y<<", fz:"<<f.z<<")"<<endl;
        throw runtime_error("Error updating neighborlist");
        }

    if (m_prof) m_prof->pop(m_exec_conf);
    }

/*!
 * Invokes the CUB libraries to sort the morton code-type keys.
 * \pre Morton code-keys are in local ParticleData order
 * \post Morton code-keys are sorted by type then position along the Z order curve.
 * \note Call after calcMortonCodes(), but before mergeLeafParticles().
 */
void NeighborListGPUTree::sortMortonCodes()
    {
    if (m_prof) m_prof->push(m_exec_conf,"Sort");

    bool swap_morton = false;
    bool swap_map = false;
        {
        ArrayHandle<uint64_t> d_morton_types(m_morton_types, access_location::device, access_mode::readwrite);
        ArrayHandle<uint64_t> d_morton_types_alt(m_morton_types_alt, access_location::device, access_mode::overwrite);
        ArrayHandle<unsigned int> d_map_tree_pid(m_map_tree_pid, access_location::device, access_mode::readwrite);
        ArrayHandle<unsigned int> d_map_tree_pid_alt(m_map_tree_pid_alt, access_location::device, access_mode::overwrite);
        ArrayHandle<unsigned int> h_num_per_type(m_num_per_type, access_location::host, access_mode::read);

        // size the temporary storage
        void *d_tmp_storage = NULL;
        size_t tmp_storage_bytes = 0;
        gpu_nlist_morton_sort(d_morton_types.data,
                              d_morton_types_alt.data,
                              d_map_tree_pid.data,
                              d_map_tree_pid_alt.data,
                              d_tmp_storage,
                              tmp_storage_bytes,
                              swap_morton,
                              swap_map,
                              m_pdata->getN() + m_pdata->getNGhosts(),
                              m_n_type_bits);

        if (m_exec_conf->isCUDAErrorCheckingEnabled())
            CHECK_CUDA_ERROR();

        /*
         * Always allocate at least 4 bytes. In CUB 1.4.1, sorting N < the tile size (which I believe is a thread block)
         * does not require any temporary storage, and tmp_storage_bytes returns 0. But, d_tmp_storage must be not NULL
         * for the sort to occur on the second pass. C++ standards forbid specifying a pointer to memory that
         * isn't properly allocated / doesn't exist (for example, a pointer to an odd address), so we allocate a small
         * bit of memory as temporary storage that isn't used.
         */
        size_t alloc_size = (tmp_storage_bytes > 0) ? tmp_storage_bytes : 4;
        // unsigned char = 1 B
        ScopedAllocation<unsigned char> d_alloc(m_exec_conf->getCachedAllocator(), alloc_size);
        d_tmp_storage = (void *)d_alloc();

        // perform the sort
        gpu_nlist_morton_sort(d_morton_types.data,
                              d_morton_types_alt.data,
                              d_map_tree_pid.data,
                              d_map_tree_pid_alt.data,
                              d_tmp_storage,
                              tmp_storage_bytes,
                              swap_morton,
                              swap_map,
                              m_pdata->getN() + m_pdata->getNGhosts(),
                              m_n_type_bits);

        if (m_exec_conf->isCUDAErrorCheckingEnabled())
            CHECK_CUDA_ERROR();
        }

    // we want the sorted data in the real data because the alt is just a tmp holder
    if (swap_morton)
        {
        m_morton_types.swap(m_morton_types_alt);
        }

    if (swap_map)
        {
        m_map_tree_pid.swap(m_map_tree_pid_alt);
        }

    if (m_prof) m_prof->pop(m_exec_conf);
    }

/*!
 * \post AABB leafs are constructed for adjacent groupings of particles.
 * \note Call after sortMortonCodes(), but before genTreeHierarchy().
 */
void NeighborListGPUTree::mergeLeafParticles()
    {
    if (m_prof) m_prof->push(m_exec_conf,"Leaf merge");

    // particle position data
    ArrayHandle<Scalar4> d_pos(m_pdata->getPositions(), access_location::device, access_mode::read);
    ArrayHandle<unsigned int> d_num_per_type(m_num_per_type, access_location::device, access_mode::read);
    ArrayHandle<unsigned int> d_type_head(m_type_head, access_location::device, access_mode::read);

    // leaf particle data
    ArrayHandle<uint64_t> d_morton_types(m_morton_types, access_location::device, access_mode::read);
    ArrayHandle<unsigned int> d_map_tree_pid(m_map_tree_pid, access_location::device, access_mode::read);
    ArrayHandle<unsigned int> d_leaf_offset(m_leaf_offset, access_location::device, access_mode::read);

    // tree aabbs and reduced morton codes to overwrite
    ArrayHandle<Scalar4> d_tree_aabbs(m_tree_aabbs, access_location::device, access_mode::overwrite);
    ArrayHandle<uint32_t> d_morton_codes_red(m_morton_codes_red, access_location::device, access_mode::overwrite);
    ArrayHandle<uint2> d_tree_parent_sib(m_tree_parent_sib, access_location::device, access_mode::overwrite);

    m_tuner_merge->begin();
    gpu_nlist_merge_particles(d_tree_aabbs.data,
                              d_morton_codes_red.data,
                              d_tree_parent_sib.data,
                              d_morton_types.data,
                              d_pos.data,
                              d_num_per_type.data,
                              m_pdata->getNTypes(),
                              d_map_tree_pid.data,
                              d_leaf_offset.data,
                              d_type_head.data,
                              m_pdata->getN() + m_pdata->getNGhosts(),
                              m_n_leaf,
                              m_tuner_merge->getParam());
    if (m_exec_conf->isCUDAErrorCheckingEnabled())
        CHECK_CUDA_ERROR();
    m_tuner_merge->end();

    if (m_prof) m_prof->pop(m_exec_conf);
    }

/*!
 * \post Parent-child-sibling relationships are established between nodes.
 * \note This function should always be called alongside bubbleAABBs to generate a complete hierarchy.
 *       genTreeHierarchy saves only the left children of the nodes for downward traversal because bubbleAABBs
 *       saves the right child as a rope to complete the edge graph.
 * \note Call after mergeLeafParticles(), but before bubbleAABBs().
 */
void NeighborListGPUTree::genTreeHierarchy()
    {
    if (m_prof) m_prof->push(m_exec_conf,"Hierarchy");

    // don't bother to process if there are no internal nodes
    if (!m_n_internal)
        return;

    ArrayHandle<uint2> d_tree_parent_sib(m_tree_parent_sib, access_location::device, access_mode::overwrite);

    ArrayHandle<uint32_t> d_morton_codes_red(m_morton_codes_red, access_location::device, access_mode::read);
    ArrayHandle<unsigned int> d_num_per_type(m_num_per_type, access_location::device, access_mode::read);

    m_tuner_hierarchy->begin();
    gpu_nlist_gen_hierarchy(d_tree_parent_sib.data,
                            d_morton_codes_red.data,
                            d_num_per_type.data,
                            m_pdata->getNTypes(),
                            m_n_leaf,
                            m_n_internal,
                            m_tuner_hierarchy->getParam());
    if (m_exec_conf->isCUDAErrorCheckingEnabled())
        CHECK_CUDA_ERROR();
    m_tuner_hierarchy->end();
    if (m_prof) m_prof->pop(m_exec_conf);
    }

//! walk up the tree from the leaves, and assign stackless ropes for traversal, and conservative AABBs
/*!
 * \post Conservative AABBs are assigned to all internal nodes, and stackless "ropes" for downward traversal are
 *       defined between nodes.
 * \note Call after genTreeHierarchy()
 */
void NeighborListGPUTree::bubbleAABBs()
    {
    if (m_prof) m_prof->push(m_exec_conf,"Bubble");
    ArrayHandle<unsigned int> d_node_locks(m_node_locks, access_location::device, access_mode::overwrite);
    ArrayHandle<Scalar4> d_tree_aabbs(m_tree_aabbs, access_location::device, access_mode::readwrite);

    ArrayHandle<uint2> d_tree_parent_sib(m_tree_parent_sib, access_location::device, access_mode::read);

    m_tuner_bubble->begin();
    gpu_nlist_bubble_aabbs(d_node_locks.data,
                           d_tree_aabbs.data,
                           d_tree_parent_sib.data,
                           m_pdata->getNTypes(),
                           m_n_leaf,
                           m_n_internal,
                           m_tuner_bubble->getParam());
    if (m_exec_conf->isCUDAErrorCheckingEnabled())
        CHECK_CUDA_ERROR();
    m_tuner_bubble->end();

    if (m_prof) m_prof->pop(m_exec_conf);
    }

void NeighborListGPUTree::moveLeafParticles()
    {
    if (m_prof) m_prof->push(m_exec_conf,"move");
    ArrayHandle<Scalar4> d_leaf_xyzf(m_leaf_xyzf, access_location::device, access_mode::overwrite);
    ArrayHandle<Scalar2> d_leaf_db(m_leaf_db, access_location::device, access_mode::overwrite);

    ArrayHandle<Scalar4> d_pos(m_pdata->getPositions(), access_location::device, access_mode::read);
    ArrayHandle<Scalar> d_diameter(m_pdata->getDiameters(), access_location::device, access_mode::read);
    ArrayHandle<unsigned int> d_body(m_pdata->getBodies(), access_location::device, access_mode::read);
    ArrayHandle<unsigned int> d_map_tree_pid(m_map_tree_pid, access_location::device, access_mode::read);

    m_tuner_move->begin();
    gpu_nlist_move_particles(d_leaf_xyzf.data,
                             d_leaf_db.data,
                             d_pos.data,
                             d_diameter.data,
                             d_body.data,
                             d_map_tree_pid.data,
                             m_pdata->getN() + m_pdata->getNGhosts(),
                             m_tuner_move->getParam());
    if (m_exec_conf->isCUDAErrorCheckingEnabled())
        CHECK_CUDA_ERROR();
    m_tuner_move->end();

    if (m_prof) m_prof->pop(m_exec_conf);
    }

/*!
 * (Re-)computes the translation vectors for traversing the BVH tree. At most, there are 27 translation vectors
 * when the simulation box is 3D periodic. In 2D, there are at most 9 translation vectors. In MPI runs, a ghost layer
 * of particles is added from adjacent ranks, so there is no need to perform any translations in this direction.
 * The translation vectors are determined by linear combination of the lattice vectors, and must be recomputed any
 * time that the box resizes.
 */
void NeighborListGPUTree::updateImageVectors()
    {
    const BoxDim& box = m_pdata->getBox();
    uchar3 periodic = box.getPeriodic();

    // check if the box is 3d or 2d, and use this to compute number of lattice vectors below
    unsigned char sys3d = (this->m_sysdef->getNDimensions() == 3);

    // check that rcut fits in the box
    Scalar3 nearest_plane_distance = box.getNearestPlaneDistance();
    Scalar rmax = getMaxRCut() + m_r_buff;
    if (m_diameter_shift)
        rmax += m_d_max - Scalar(1.0);

    if (m_filter_body)
        {
        // add the maximum diameter of all composite particles
        Scalar max_d_comp = m_pdata->getMaxCompositeParticleDiameter();
        rmax += 0.5*max_d_comp;
        }

    if ((periodic.x && nearest_plane_distance.x <= rmax * 2.0) ||
        (periodic.y && nearest_plane_distance.y <= rmax * 2.0) ||
        (sys3d && periodic.z && nearest_plane_distance.z <= rmax * 2.0))
        {
        m_exec_conf->msg->error() << "nlist: Simulation box is too small! Particles would be interacting with themselves." << endl;
        throw runtime_error("Error updating neighborlist bins");
        }

    // now compute the image vectors
    // each dimension increases by one power of 3
    unsigned int n_dim_periodic = (periodic.x + periodic.y + sys3d*periodic.z);
    m_n_images = 1;
    for (unsigned int dim = 0; dim < n_dim_periodic; ++dim)
        {
        m_n_images *= 3;
        }

    // reallocate memory if necessary
    if (m_n_images > m_image_list.getPitch())
        {
        GPUArray<Scalar3> image_list(m_n_images, m_exec_conf);
        m_image_list.swap(image_list);
        }

    ArrayHandle<Scalar3> h_image_list(m_image_list, access_location::host, access_mode::overwrite);
    Scalar3 latt_a = box.getLatticeVector(0);
    Scalar3 latt_b = box.getLatticeVector(1);
    Scalar3 latt_c = box.getLatticeVector(2);

    // there is always at least 1 image, which we put as our first thing to look at
    h_image_list.data[0] = make_scalar3(0.0, 0.0, 0.0);

    // iterate over all other combinations of images, skipping those that are
    unsigned int n_images = 1;
    for (int i=-1; i <= 1 && n_images < m_n_images; ++i)
        {
        for (int j=-1; j <= 1 && n_images < m_n_images; ++j)
            {
            for (int k=-1; k <= 1 && n_images < m_n_images; ++k)
                {
                if (!(i == 0 && j == 0 && k == 0))
                    {
                    // skip any periodic images if we don't have periodicity
                    if (i != 0 && !periodic.x) continue;
                    if (j != 0 && !periodic.y) continue;
                    if (k != 0 && (!sys3d || !periodic.z)) continue;

                    h_image_list.data[n_images] = Scalar(i) * latt_a + Scalar(j) * latt_b + Scalar(k) * latt_c;
                    ++n_images;
                    }
                }
            }
        }
    }

/*!
 * \post The neighbor list has been fully generated.
 */
void NeighborListGPUTree::traverseTree()
    {
    if (m_prof) m_prof->push(m_exec_conf,"Traverse");

    // move the leaf particles into leaf_xyzf and leaf_tdb for fast traversal
    moveLeafParticles();

    // neighborlist data
    ArrayHandle<unsigned int> d_nlist(m_nlist, access_location::device, access_mode::overwrite);
    ArrayHandle<unsigned int> d_n_neigh(m_n_neigh, access_location::device, access_mode::overwrite);
    ArrayHandle<Scalar4> d_last_updated_pos(m_last_pos, access_location::device, access_mode::overwrite);
    ArrayHandle<unsigned int> d_conditions(m_conditions, access_location::device, access_mode::readwrite);

    ArrayHandle<unsigned int> d_Nmax(m_Nmax, access_location::device, access_mode::read);
    ArrayHandle<unsigned int> d_head_list(m_head_list, access_location::device, access_mode::read);

    // tree data
    ArrayHandle<unsigned int> d_map_tree_pid(m_map_tree_pid, access_location::device, access_mode::read);
    ArrayHandle<unsigned int> d_leaf_offset(m_leaf_offset, access_location::device, access_mode::read);
    ArrayHandle<unsigned int> d_tree_roots(m_tree_roots, access_location::device, access_mode::read);
    ArrayHandle<Scalar4> d_tree_aabbs(m_tree_aabbs, access_location::device, access_mode::read);

    // tree particle data
    ArrayHandle<Scalar4> d_leaf_xyzf(m_leaf_xyzf, access_location::device, access_mode::read);
    ArrayHandle<Scalar2> d_leaf_db(m_leaf_db, access_location::device, access_mode::read);

    // particle data
    ArrayHandle<Scalar4> d_pos(m_pdata->getPositions(), access_location::device, access_mode::read);

    // images
    ArrayHandle<Scalar3> d_image_list(m_image_list, access_location::device, access_mode::read);

    // pairwise cutoffs
    ArrayHandle<Scalar> d_r_cut(m_r_cut, access_location::device, access_mode::read);

    m_tuner_traverse->begin();
    gpu_nlist_traverse_tree(d_nlist.data,
                            d_n_neigh.data,
                            d_last_updated_pos.data,
                            d_conditions.data,
                            d_Nmax.data,
                            d_head_list.data,
                            m_pdata->getN(),
                            m_pdata->getNGhosts(),
                            d_map_tree_pid.data,
                            d_leaf_offset.data,
                            d_tree_roots.data,
                            d_tree_aabbs.data,
                            m_n_leaf,
                            m_n_internal,
                            m_n_node,
                            d_leaf_xyzf.data,
                            d_leaf_db.data,
                            d_pos.data,
                            d_image_list.data,
                            m_image_list.getPitch(),
                            d_r_cut.data,
                            m_r_buff,
                            m_d_max,
                            m_pdata->getNTypes(),
                            m_filter_body,
                            m_diameter_shift,
                            m_exec_conf->getComputeCapability()/10,
                            m_tuner_traverse->getParam());
    if (m_exec_conf->isCUDAErrorCheckingEnabled())
        CHECK_CUDA_ERROR();
    m_tuner_traverse->end();

    if (m_prof) m_prof->pop(m_exec_conf);
    }

void export_NeighborListGPUTree(py::module& m)
    {
    py::class_<NeighborListGPUTree, std::shared_ptr<NeighborListGPUTree> >(m, "NeighborListGPUTree", py::base<NeighborListGPU>())
    .def(py::init< std::shared_ptr<SystemDefinition>, Scalar, Scalar >());
    }
