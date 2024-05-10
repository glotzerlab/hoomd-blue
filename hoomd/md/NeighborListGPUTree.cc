// Copyright (c) 2009-2024 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

/*! \file NeighborListGPUTree.cc
    \brief Defines NeighborListGPUTree
*/

#include "NeighborListGPUTree.h"
#include "NeighborListGPUTree.cuh"

#ifdef ENABLE_MPI
#include "hoomd/Communicator.h"
#endif

namespace hoomd
    {
namespace md
    {
/*!
 * \param sysdef System definition.
 * \param r_cut The default cutoff.
 * \param r_buff The buffer radius.
 */
NeighborListGPUTree::NeighborListGPUTree(std::shared_ptr<SystemDefinition> sysdef, Scalar r_buff)
    : NeighborListGPU(sysdef, r_buff), m_type_bits(1), m_lbvh_errors(m_exec_conf), m_n_images(0),
      m_types_allocated(false), m_box_changed(true), m_max_num_changed(true), m_max_types(0)
    {
    m_exec_conf->msg->notice(5) << "Constructing NeighborListGPUTree" << std::endl;
    m_pdata->getBoxChangeSignal()
        .connect<NeighborListGPUTree, &NeighborListGPUTree::slotBoxChanged>(this);
    m_pdata->getMaxParticleNumberChangeSignal()
        .connect<NeighborListGPUTree, &NeighborListGPUTree::slotMaxNumChanged>(this);

    m_mark_tuner.reset(new Autotuner<1>({AutotunerBase::makeBlockSizeRange(this->m_exec_conf)},
                                        m_exec_conf,
                                        "nlist_tree_mark"));
    m_count_tuner.reset(new Autotuner<1>({AutotunerBase::makeBlockSizeRange(this->m_exec_conf)},
                                         m_exec_conf,
                                         "nlist_tree_count"));
    m_copy_tuner.reset(new Autotuner<1>({AutotunerBase::makeBlockSizeRange(this->m_exec_conf)},
                                        m_exec_conf,
                                        "nlist_tree_copy"));
    m_autotuners.insert(m_autotuners.end(), {m_mark_tuner, m_count_tuner, m_copy_tuner});
    }

/*!
 * Any existing CUDA streams are destroyed with the object.
 */
NeighborListGPUTree::~NeighborListGPUTree()
    {
    m_exec_conf->msg->notice(5) << "Destroying NeighborListGPUTree" << std::endl;
    m_pdata->getBoxChangeSignal()
        .disconnect<NeighborListGPUTree, &NeighborListGPUTree::slotBoxChanged>(this);
    m_pdata->getMaxParticleNumberChangeSignal()
        .disconnect<NeighborListGPUTree, &NeighborListGPUTree::slotMaxNumChanged>(this);

    // destroy all of the created streams
    for (auto stream = m_streams.begin(); stream != m_streams.end(); ++stream)
        {
        hipStreamDestroy(*stream);
        }
    }

/*!
 * \param timestep Current timestep
 *
 * First, memory is reallocated based on the number of particles and types.
 * The traversal images are also updated if the box has changed. One LBVH is then
 * built for each particle type using buildTree(), and these LBVHs are traversed in
 * traverseTree().
 */
void NeighborListGPUTree::buildNlist(uint64_t timestep)
    {
    if (!m_pdata->getN())
        return;

    // allocate memory that depends on the local number of particles
    if (m_max_num_changed)
        {
        GPUArray<unsigned int> types(m_pdata->getMaxN(), m_exec_conf);
        m_types.swap(types);

        GPUArray<unsigned int> sorted_types(m_pdata->getMaxN(), m_exec_conf);
        m_sorted_types.swap(sorted_types);

        GPUArray<unsigned int> indexes(m_pdata->getMaxN(), m_exec_conf);
        m_indexes.swap(indexes);

        GPUArray<unsigned int> sorted_indexes(m_pdata->getMaxN(), m_exec_conf);
        m_sorted_indexes.swap(sorted_indexes);

        GPUArray<unsigned int> traverse_order(m_pdata->getMaxN(), m_exec_conf);
        m_traverse_order.swap(traverse_order);

        // all done with the particle data reallocation
        m_max_num_changed = false;
        }

    // allocate memory that depends on type
    if (!m_types_allocated)
        {
        if (m_pdata->getNTypes() > m_max_types)
            {
            GPUArray<unsigned int> type_first(m_pdata->getNTypes(), m_exec_conf);
            m_type_first.swap(type_first);

            GPUArray<unsigned int> type_last(m_pdata->getNTypes(), m_exec_conf);
            m_type_last.swap(type_last);

            m_lbvhs.resize(m_pdata->getNTypes());
            m_traversers.resize(m_pdata->getNTypes());
            m_streams.resize(m_pdata->getNTypes());
            for (unsigned int i = m_max_types; i < m_pdata->getNTypes(); ++i)
                {
                m_lbvhs[i].reset(new kernel::LBVHWrapper());
                m_traversers[i].reset(new kernel::LBVHTraverserWrapper());
                hipStreamCreate(&m_streams[i]);
                }

            m_max_types = m_pdata->getNTypes();
            }

        /*
         * Compute the number of bits to sort, which is the number of bits needed to represent the
         * largest type index, plus 1 to account for the ghost sentinel. So, it is the number of
         * bits to represent the number of types.
         */
        m_type_bits = 0;
        // count bit shifts to zero, then round up to get the right counts
        unsigned int tmp = m_pdata->getNTypes() + 1;
        while (tmp >>= 1)
            {
            ++m_type_bits;
            }
        ++m_type_bits;

        // all done with the type reallocation
        m_types_allocated = true;
        }

    // update properties that depend on the box
    if (m_box_changed)
        {
        updateImageVectors();
        m_box_changed = false;
        }

    // ensure build tuner is set
    if (!m_build_tuner)
        {
        m_build_tuner.reset(new Autotuner<1>({m_lbvhs[0]->getTunableParameters()},
                                             m_exec_conf,
                                             "nlist_tree_build"));
        m_autotuners.push_back(m_build_tuner);
        }

    // ensure traverser tuner is set
    if (!m_traverse_tuner)
        {
        m_traverse_tuner.reset(new Autotuner<1>({m_traversers[0]->getTunableParameters()},
                                                m_exec_conf,
                                                "nlist_tree_traverse"));
        m_autotuners.push_back(m_traverse_tuner);
        }

    // build the tree
    buildTree();

    // walk with the tree
    traverseTree();
    }

/*!
 * Builds the LBVHs by first sorting the particles by type (to make one LBVH per type).
 * This method also puts the particles into the right order for traversal, and it prepares
 * each LBVH traverser so that subsequent calls to traverse can safely use the cached version
 * of the traverser internal data.
 *
 * The builds and the traverser setup are done in CUDA streams. I believe that the build has
 * a blocking call for a single CPU thread because of a stream synchronize due to CUB's use of the
 * double buffer. (It must report which buffer holds the sorted data.) However, benchmarks showed
 * that using the CUB API that should be non-blocking had significantly worse performance.
 *
 * I also note that the use of autotuners in neighbor should break concurrency, since these CUDA
 * timing events are placed in the default stream. This might be reconsidered in future if HOOMD
 * makes more use of CUDA streams anywhere.
 */
void NeighborListGPUTree::buildTree()
    {
        // set the data by type
        {
        // also, check particles to filter out ghosts that lie outside the current box
        const BoxDim& box = m_pdata->getBox();
        Scalar ghost_layer_width(0.0);
#ifdef ENABLE_MPI
        if (m_sysdef->isDomainDecomposed())
            ghost_layer_width = m_comm->getGhostLayerMaxWidth();
#endif
        Scalar3 ghost_width = make_scalar3(0.0, 0.0, 0.0);
        if (!box.getPeriodic().x)
            ghost_width.x = ghost_layer_width;
        if (!box.getPeriodic().y)
            ghost_width.y = ghost_layer_width;
        if (!box.getPeriodic().z && m_sysdef->getNDimensions() == 3)
            {
            ghost_width.z = ghost_layer_width;
            }

            {
            ArrayHandle<unsigned int> d_types(m_types,
                                              access_location::device,
                                              access_mode::overwrite);
            ArrayHandle<unsigned int> d_indexes(m_indexes,
                                                access_location::device,
                                                access_mode::overwrite);
            m_lbvh_errors.resetFlags(0);
            ArrayHandle<Scalar4> d_last_pos(m_last_pos,
                                            access_location::device,
                                            access_mode::overwrite);
            ArrayHandle<Scalar4> d_pos(m_pdata->getPositions(),
                                       access_location::device,
                                       access_mode::read);

            m_mark_tuner->begin();
            kernel::gpu_nlist_mark_types(d_types.data,
                                         d_indexes.data,
                                         m_lbvh_errors.getDeviceFlags(),
                                         d_last_pos.data,
                                         d_pos.data,
                                         m_pdata->getN(),
                                         m_pdata->getNGhosts(),
                                         box,
                                         ghost_width,
                                         m_mark_tuner->getParam()[0]);
            if (m_exec_conf->isCUDAErrorCheckingEnabled())
                CHECK_CUDA_ERROR();
            m_mark_tuner->end();
            }

        // error check that no local particles are out of bounds
        const unsigned int lbvh_errors = m_lbvh_errors.readFlags();
        if (lbvh_errors)
            {
            ArrayHandle<Scalar4> h_pos(m_pdata->getPositions(),
                                       access_location::host,
                                       access_mode::read);
            ArrayHandle<unsigned int> h_tag(m_pdata->getTags(),
                                            access_location::host,
                                            access_mode::read);

            const unsigned int error_idx = lbvh_errors - 1;
            const Scalar4 error_pos = h_pos.data[error_idx];
            const unsigned int error_tag = h_tag.data[error_idx];

            m_exec_conf->msg->error()
                << "nlist.tree(): Particle " << error_tag << " is out of bounds " << "("
                << error_pos.x << ", " << error_pos.y << ", " << error_pos.z << ")" << std::endl;
            throw std::runtime_error("Error updating neighborlist");
            }
        }

    // sort the particles by type, pushing out-of-bounds ghosts to the ends
    if (m_pdata->getNTypes() > 1 || m_pdata->getNGhosts() > 0)
        {
        uchar2 swap;
            {
            ArrayHandle<unsigned int> d_types(m_types,
                                              access_location::device,
                                              access_mode::readwrite);
            ArrayHandle<unsigned int> d_sorted_types(m_sorted_types,
                                                     access_location::device,
                                                     access_mode::overwrite);
            ArrayHandle<unsigned int> d_indexes(m_indexes,
                                                access_location::device,
                                                access_mode::readwrite);
            ArrayHandle<unsigned int> d_sorted_indexes(m_sorted_indexes,
                                                       access_location::device,
                                                       access_mode::overwrite);

            void* d_tmp = NULL;
            size_t tmp_bytes = 0;
            kernel::gpu_nlist_sort_types(d_tmp,
                                         tmp_bytes,
                                         d_types.data,
                                         d_sorted_types.data,
                                         d_indexes.data,
                                         d_sorted_indexes.data,
                                         m_pdata->getN() + m_pdata->getNGhosts(),
                                         m_type_bits);

            // make requested temporary allocation (1 char = 1B)
            size_t alloc_size = (tmp_bytes > 0) ? tmp_bytes : 4;
            ScopedAllocation<unsigned char> d_alloc(m_exec_conf->getCachedAllocator(), alloc_size);
            d_tmp = (void*)d_alloc();

            // perform the sort
            swap = kernel::gpu_nlist_sort_types(d_tmp,
                                                tmp_bytes,
                                                d_types.data,
                                                d_sorted_types.data,
                                                d_indexes.data,
                                                d_sorted_indexes.data,
                                                m_pdata->getN() + m_pdata->getNGhosts(),
                                                m_type_bits);
            }
        if (swap.x)
            m_sorted_types.swap(m_types);
        if (swap.y)
            m_sorted_indexes.swap(m_indexes);
        }
    else
        {
        m_sorted_types.swap(m_types);
        m_sorted_indexes.swap(m_indexes);
        }

        // count the number of each type
        {
        ArrayHandle<unsigned int> d_type_first(m_type_first,
                                               access_location::device,
                                               access_mode::overwrite);
        ArrayHandle<unsigned int> d_type_last(m_type_last,
                                              access_location::device,
                                              access_mode::overwrite);
        ArrayHandle<unsigned int> d_sorted_types(m_sorted_types,
                                                 access_location::device,
                                                 access_mode::read);

        m_count_tuner->begin();
        kernel::gpu_nlist_count_types(d_type_first.data,
                                      d_type_last.data,
                                      d_sorted_types.data,
                                      m_pdata->getNTypes(),
                                      m_pdata->getN() + m_pdata->getNGhosts(),
                                      m_count_tuner->getParam()[0]);
        if (m_exec_conf->isCUDAErrorCheckingEnabled())
            CHECK_CUDA_ERROR();
        m_count_tuner->end();
        }

        // build a lbvh for each type
        {
        ArrayHandle<unsigned int> h_type_first(m_type_first,
                                               access_location::host,
                                               access_mode::read);
        ArrayHandle<unsigned int> h_type_last(m_type_last,
                                              access_location::host,
                                              access_mode::read);

        ArrayHandle<Scalar4> d_pos(m_pdata->getPositions(),
                                   access_location::device,
                                   access_mode::read);
        ArrayHandle<unsigned int> d_sorted_indexes(m_sorted_indexes,
                                                   access_location::device,
                                                   access_mode::read);

        const BoxDim lbvh_box = getLBVHBox();

        // first, setup memory (these do not actually execute in a stream)
        for (unsigned int i = 0; i < m_pdata->getNTypes(); ++i)
            {
            const unsigned int first = h_type_first.data[i];
            const unsigned int last = h_type_last.data[i];
            if (first != kernel::NeighborListTypeSentinel)
                {
                m_lbvhs[i]->setup(d_pos.data,
                                  d_sorted_indexes.data + first,
                                  last - first,
                                  m_streams[i]);
                }
            else
                {
                // effectively destroy the lbvh
                m_lbvhs[i]->setup(d_pos.data, NULL, 0, m_streams[i]);
                }
            }

        // then, launch all of the builds in their own streams
        hipDeviceSynchronize();
        m_build_tuner->begin();
        const unsigned int block_size = m_build_tuner->getParam()[0];

        for (unsigned int i = 0; i < m_pdata->getNTypes(); ++i)
            {
            const unsigned int first = h_type_first.data[i];
            const unsigned int last = h_type_last.data[i];

            if (first != kernel::NeighborListTypeSentinel)
                {
                m_lbvhs[i]->build(d_pos.data,
                                  d_sorted_indexes.data + first,
                                  last - first,
                                  lbvh_box.getLo(),
                                  lbvh_box.getHi(),
                                  m_streams[i],
                                  block_size);
                }
            else
                {
                // effectively destroy the lbvh
                m_lbvhs[i]->build(d_pos.data,
                                  NULL,
                                  0,
                                  lbvh_box.getLo(),
                                  lbvh_box.getHi(),
                                  m_streams[i],
                                  block_size);
                }
            }
        m_build_tuner->end();
        // wait for all builds to finish
        hipDeviceSynchronize();
        }

        // put particles in primitive order for traversal and compress the lbvhs so that the data is
        // ready for traversal
        {
        ArrayHandle<unsigned int> h_type_first(m_type_first,
                                               access_location::host,
                                               access_mode::read);
        ArrayHandle<unsigned int> d_traverse_order(m_traverse_order,
                                                   access_location::device,
                                                   access_mode::overwrite);
        ArrayHandle<unsigned int> d_sorted_indexes(m_sorted_indexes,
                                                   access_location::device,
                                                   access_mode::read);

        for (unsigned int i = 0; i < m_pdata->getNTypes(); ++i)
            {
            const unsigned int Ni = m_lbvhs[i]->getN();
            if (Ni > 0)
                {
                const unsigned int first = h_type_first.data[i];
                auto d_primitives = m_lbvhs[i]->getPrimitives();
                m_copy_tuner->begin();
                kernel::gpu_nlist_copy_primitives(d_traverse_order.data + first,
                                                  d_sorted_indexes.data + first,
                                                  d_primitives,
                                                  Ni,
                                                  m_copy_tuner->getParam()[0]);
                if (m_exec_conf->isCUDAErrorCheckingEnabled())
                    CHECK_CUDA_ERROR();
                m_copy_tuner->end();
                }
            }

        // loops are not fused to avoid streams or syncing in kernel loop above, but could be done
        // if necessary
        hipDeviceSynchronize();
        for (unsigned int i = 0; i < m_pdata->getNTypes(); ++i)
            {
            if (m_lbvhs[i]->getN() == 0)
                continue;
            m_traversers[i]->setup(d_sorted_indexes.data + h_type_first.data[i],
                                   *(m_lbvhs[i]->get()),
                                   m_streams[i]);
            }
        hipDeviceSynchronize();
        }
    }

/*!
 * Traversal is performed for each particle type against all LBVHs. This is done using one CUDA
 * stream for each particle type, and traversal of each LBVH is loaded into the stream so that there
 * are no race conditions. The traversal should have good concurrency, as there are no blocking
 * calls on the host. For efficiency, body filtering is templated out, and the correct template is
 * selected at dispatch.
 *
 * As for the build, I note that the use of autotuners in neighbor should break concurrency, since
 * these CUDA timing events are placed in the default stream. This might be reconsidered in future
 * if HOOMD makes more use of CUDA streams anywhere.
 */
void NeighborListGPUTree::traverseTree()
    {
    ArrayHandle<unsigned int> d_nlist(m_nlist, access_location::device, access_mode::overwrite);
    ArrayHandle<unsigned int> d_n_neigh(m_n_neigh, access_location::device, access_mode::overwrite);
    ArrayHandle<unsigned int> d_conditions(m_conditions,
                                           access_location::device,
                                           access_mode::readwrite);
    ArrayHandle<size_t> d_head_list(m_head_list, access_location::device, access_mode::read);

    ArrayHandle<unsigned int> h_type_first(m_type_first, access_location::host, access_mode::read);
    ArrayHandle<unsigned int> h_type_last(m_type_last, access_location::host, access_mode::read);

    ArrayHandle<unsigned int> d_sorted_indexes(m_sorted_indexes,
                                               access_location::device,
                                               access_mode::read);

    ArrayHandle<Scalar4> d_pos(m_pdata->getPositions(), access_location::device, access_mode::read);
    ArrayHandle<unsigned int> d_body(m_pdata->getBodies(),
                                     access_location::device,
                                     access_mode::read);
    ArrayHandle<Scalar> d_diam(m_pdata->getDiameters(), access_location::device, access_mode::read);
    ArrayHandle<unsigned int> d_traverse_order(m_traverse_order,
                                               access_location::device,
                                               access_mode::read);
    ArrayHandle<Scalar3> d_image_list(m_image_list, access_location::device, access_mode::read);

    ArrayHandle<Scalar> h_r_cut(m_r_cut, access_location::host, access_mode::read);
    ArrayHandle<unsigned int> h_Nmax(m_Nmax, access_location::host, access_mode::read);

    // clear the neighbor counts
    hipMemset(d_n_neigh.data, 0, sizeof(unsigned int) * m_pdata->getN());

    const BoxDim& box = m_pdata->getBox();

    // traverse all pairs in (now-transposed) streams
    hipDeviceSynchronize();
    m_traverse_tuner->begin();
    const unsigned int block_size = m_traverse_tuner->getParam()[0];
    for (unsigned int i = 0; i < m_pdata->getNTypes(); ++i)
        {
        // skip this type if there are no particles
        const unsigned int first = h_type_first.data[i];
        if (first == kernel::NeighborListTypeSentinel)
            continue;
        const unsigned int Ni = h_type_last.data[i] - first;

        // traverse it against all trees, using the same stream for type i to avoid race conditions
        // on writing
        for (unsigned int j = 0; j < m_pdata->getNTypes(); ++j)
            {
            // skip this lbvh if there are no particles in it
            if (m_lbvhs[j]->getN() == 0)
                continue;

            // search radii for this type pair
            Scalar rcut = h_r_cut.data[m_typpair_idx(i, j)];
            Scalar rlist;
            if (rcut > Scalar(0))
                {
                rcut += m_r_buff;
                rlist = rcut;
                }
            else
                {
                // skip interaction completely if turned off
                continue;
                }

            // pack args to the traverser
            kernel::LBVHTraverserWrapper::TraverserArgs args;

            // the transform operator is for the particles in this LBVH (j)
            args.map = d_sorted_indexes.data + h_type_first.data[j];

            // particles
            args.positions = d_pos.data;
            args.bodies = (m_filter_body) ? d_body.data : NULL;
            args.order = d_traverse_order.data + first;
            args.N = Ni;
            args.Nown = m_pdata->getN();
            args.rcut = rcut;
            args.rlist = rlist;
            args.box = box;

            // neighbor list write op for this type
            args.neigh_list = d_nlist.data;
            args.nneigh = d_n_neigh.data;
            args.new_max_neigh = d_conditions.data + i;
            args.first_neigh = d_head_list.data;
            args.max_neigh = h_Nmax.data[i];

            m_traversers[j]->traverse(args,
                                      *(m_lbvhs[j]->get()),
                                      d_image_list.data,
                                      (unsigned int)m_image_list.getNumElements(),
                                      m_streams[i],
                                      block_size);
            }
        }
    m_traverse_tuner->end();
    // wait for all traversals to finish
    hipDeviceSynchronize();
    }

/*!
 * (Re-)computes the translation vectors for traversing the BVH tree. At most, there are 27
 * translation vectors when the simulation box is 3D periodic (self-image included). In 2D, there
 * are at most 9 translation vectors. In MPI runs, a ghost layer of particles is added from adjacent
 * ranks, so there is no need to perform any translations in this direction. The translation vectors
 * are determined by linear combination of the lattice vectors, and must be recomputed any time that
 * the box resizes.
 */
void NeighborListGPUTree::updateImageVectors()
    {
    const BoxDim& box = m_pdata->getBox();
    uchar3 periodic = box.getPeriodic();
    unsigned char sys3d = (m_sysdef->getNDimensions() == 3);

    // now compute the image vectors
    // each dimension increases by one power of 3
    unsigned int n_dim_periodic = (periodic.x + periodic.y + sys3d * periodic.z);
    m_n_images = 1;
    for (unsigned int dim = 0; dim < n_dim_periodic; ++dim)
        {
        m_n_images *= 3;
        }

    // reallocate memory if necessary
    if (m_n_images > m_image_list.getNumElements())
        {
        GlobalVector<Scalar3> image_list(m_n_images, m_exec_conf);
        m_image_list.swap(image_list);
        }

    ArrayHandle<Scalar3> h_image_list(m_image_list, access_location::host, access_mode::overwrite);
    Scalar3 latt_a = box.getLatticeVector(0);
    Scalar3 latt_b = box.getLatticeVector(1);
    Scalar3 latt_c = box.getLatticeVector(2);

    // iterate over all other combinations of images, skipping those that are
    h_image_list.data[0] = make_scalar3(0, 0, 0);
    unsigned int n_images = 1;
    for (int i = -1; i <= 1 && n_images < m_n_images; ++i)
        {
        for (int j = -1; j <= 1 && n_images < m_n_images; ++j)
            {
            for (int k = -1; k <= 1 && n_images < m_n_images; ++k)
                {
                if (!(i == 0 && j == 0 && k == 0))
                    {
                    // skip any periodic images if we don't have periodicity
                    if (i != 0 && !periodic.x)
                        continue;
                    if (j != 0 && !periodic.y)
                        continue;
                    if (k != 0 && (!sys3d || !periodic.z))
                        continue;

                    h_image_list.data[n_images]
                        = Scalar(i) * latt_a + Scalar(j) * latt_b + Scalar(k) * latt_c;
                    ++n_images;
                    }
                }
            }
        }
    }

namespace detail
    {
void export_NeighborListGPUTree(pybind11::module& m)
    {
    pybind11::class_<NeighborListGPUTree, NeighborListGPU, std::shared_ptr<NeighborListGPUTree>>(
        m,
        "NeighborListGPUTree")
        .def(pybind11::init<std::shared_ptr<SystemDefinition>, Scalar>());
    }

    } // end namespace detail
    } // end namespace md
    } // end namespace hoomd
