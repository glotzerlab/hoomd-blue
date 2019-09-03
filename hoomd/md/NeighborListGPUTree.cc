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

NeighborListGPUTree::NeighborListGPUTree(std::shared_ptr<SystemDefinition> sysdef,
                                       Scalar r_cut,
                                       Scalar r_buff)
    : NeighborListGPU(sysdef, r_cut, r_buff), m_lbvh_errors(m_exec_conf),
      m_n_images(0),
      m_type_changed(true), m_box_changed(true), m_max_num_changed(true), m_max_types(0)
    {
    m_exec_conf->msg->notice(5) << "Constructing NeighborListGPUTree" << std::endl;
    m_pdata->getNumTypesChangeSignal().connect<NeighborListGPUTree, &NeighborListGPUTree::slotNumTypesChanged>(this);
    m_pdata->getBoxChangeSignal().connect<NeighborListGPUTree, &NeighborListGPUTree::slotBoxChanged>(this);
    m_pdata->getMaxParticleNumberChangeSignal().connect<NeighborListGPUTree, &NeighborListGPUTree::slotMaxNumChanged>(this);
    }

NeighborListGPUTree::~NeighborListGPUTree()
    {
    m_exec_conf->msg->notice(5) << "Destroying NeighborListGPUTree" << std::endl;
    m_pdata->getNumTypesChangeSignal().disconnect<NeighborListGPUTree, &NeighborListGPUTree::slotNumTypesChanged>(this);
    m_pdata->getBoxChangeSignal().disconnect<NeighborListGPUTree, &NeighborListGPUTree::slotBoxChanged>(this);
    m_pdata->getMaxParticleNumberChangeSignal().disconnect<NeighborListGPUTree, &NeighborListGPUTree::slotMaxNumChanged>(this);
    }

void NeighborListGPUTree::buildNlist(unsigned int timestep)
    {
    if (!m_pdata->getN()) return;

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
    if (m_type_changed)
        {
        if (m_pdata->getNTypes() > m_max_types)
            {
            GPUArray<unsigned int> type_first(m_pdata->getNTypes(), m_exec_conf);
            m_type_first.swap(type_first);

            GPUArray<unsigned int> type_last(m_pdata->getNTypes(), m_exec_conf);
            m_type_last.swap(type_last);

            m_lbvhs.resize(m_pdata->getNTypes());
            m_traversers.resize(m_pdata->getNTypes());
            for (unsigned int i=m_max_types; i < m_pdata->getNTypes(); ++i)
                {
                m_lbvhs[i].reset(new neighbor::LBVH(m_exec_conf));
                m_traversers[i].reset(new neighbor::LBVHTraverser(m_exec_conf));
                }

            m_max_types = m_pdata->getNTypes();
            }

        // all done with the type reallocation
        m_type_changed = false;
        }

    // update properties that depend on the box
    if (m_box_changed)
        {
        updateImageVectors();
        m_box_changed = false;
        }

    // build the tree
    buildTree();

    // walk with the tree
    traverseTree();

    // memcpy the current positions of local particles
        {
        ArrayHandle<Scalar4> d_pos(m_pdata->getPositions(), access_location::device, access_mode::read);
        ArrayHandle<Scalar4> d_last_updated_pos(m_last_pos, access_location::device, access_mode::overwrite);
        cudaMemcpy(d_last_updated_pos.data, d_pos.data, sizeof(Scalar4)*m_pdata->getN(), cudaMemcpyDeviceToDevice);
        }
    }

void NeighborListGPUTree::buildTree()
    {
    // set the data by type
        {
        // also, check particles to filter out ghosts that lie outside the current box
        const BoxDim& box = m_pdata->getBox();
        Scalar ghost_layer_width(0.0);
        #ifdef ENABLE_MPI
        if (m_comm) ghost_layer_width = m_comm->getGhostLayerMaxWidth();
        #endif
        Scalar3 ghost_width = make_scalar3(0.0, 0.0, 0.0);
        if (!box.getPeriodic().x) ghost_width.x = ghost_layer_width;
        if (!box.getPeriodic().y) ghost_width.y = ghost_layer_width;
        if (!box.getPeriodic().z && m_sysdef->getNDimensions() == 3) ghost_width.z = ghost_layer_width;
            {
            ArrayHandle<unsigned int> d_types(m_types, access_location::device, access_mode::overwrite);
            ArrayHandle<unsigned int> d_indexes(m_indexes, access_location::device, access_mode::overwrite);
            m_lbvh_errors.resetFlags(0);
            ArrayHandle<Scalar4> d_pos(m_pdata->getPositions(), access_location::device, access_mode::read);

            gpu_nlist_mark_types(d_types.data,
                                 d_indexes.data,
                                 m_lbvh_errors.getDeviceFlags(),
                                 d_pos.data,
                                 m_pdata->getN(),
                                 m_pdata->getNGhosts(),
                                 box,
                                 ghost_width,
                                 128);
            if (m_exec_conf->isCUDAErrorCheckingEnabled()) CHECK_CUDA_ERROR();
            }

        // error check that no local particles are out of bounds
        const unsigned int lbvh_errors = m_lbvh_errors.readFlags();
        if (lbvh_errors)
            {
            ArrayHandle<Scalar4> h_pos(m_pdata->getPositions(), access_location::host, access_mode::read);
            ArrayHandle<unsigned int> h_tag(m_pdata->getTags(), access_location::host, access_mode::read);

            const unsigned int error_idx = lbvh_errors-1;
            const Scalar4 error_pos = h_pos.data[error_idx];
            const unsigned int error_tag = h_tag.data[error_idx];

            m_exec_conf->msg->error() << "nlist.tree(): Particle " << error_tag << " is out of bounds "
                                      << "(" << error_pos.x << ", " << error_pos.y << ", " << error_pos.z << ")" << std::endl;
            throw std::runtime_error("Error updating neighborlist");
            }
        }

    // sort the particles by type, pushing out-of-bounds ghosts to the ends
        {
        uchar2 swap;
            {
            ArrayHandle<unsigned int> d_types(m_types, access_location::device, access_mode::readwrite);
            ArrayHandle<unsigned int> d_sorted_types(m_sorted_types, access_location::device, access_mode::overwrite);
            ArrayHandle<unsigned int> d_indexes(m_indexes, access_location::device, access_mode::readwrite);
            ArrayHandle<unsigned int> d_sorted_indexes(m_sorted_indexes, access_location::device, access_mode::overwrite);

            void *d_tmp = NULL;
            size_t tmp_bytes = 0;
            gpu_nlist_sort_types(d_tmp,
                                 tmp_bytes,
                                 d_types.data,
                                 d_sorted_types.data,
                                 d_indexes.data,
                                 d_sorted_indexes.data,
                                 m_pdata->getN() + m_pdata->getNGhosts());

            // make requested temporary allocation (1 char = 1B)
            size_t alloc_size = (tmp_bytes > 0) ? tmp_bytes : 4;
            ScopedAllocation<unsigned char> d_alloc(m_exec_conf->getCachedAllocator(), alloc_size);
            d_tmp = (void *)d_alloc();

            // perform the sort
            swap = gpu_nlist_sort_types(d_tmp,
                                        tmp_bytes,
                                        d_types.data,
                                        d_sorted_types.data,
                                        d_indexes.data,
                                        d_sorted_indexes.data,
                                        m_pdata->getN() + m_pdata->getNGhosts());
            }
        if (swap.x) m_sorted_types.swap(m_types);
        if (swap.y) m_sorted_indexes.swap(m_indexes);
        }

    // count the number of each type
        {
        ArrayHandle<unsigned int> d_type_first(m_type_first, access_location::device, access_mode::overwrite);
        ArrayHandle<unsigned int> d_type_last(m_type_last, access_location::device, access_mode::overwrite);
        ArrayHandle<unsigned int> d_sorted_types(m_sorted_types, access_location::device, access_mode::read);

        gpu_nlist_count_types(d_type_first.data,
                              d_type_last.data,
                              d_sorted_types.data,
                              m_pdata->getNTypes(),
                              m_pdata->getN()+m_pdata->getNGhosts(),
                              128);
        if (m_exec_conf->isCUDAErrorCheckingEnabled()) CHECK_CUDA_ERROR();
        }

    // build a lbvh for each type
        {
        ArrayHandle<unsigned int> h_type_first(m_type_first, access_location::host, access_mode::read);
        ArrayHandle<unsigned int> h_type_last(m_type_last, access_location::host, access_mode::read);

        ArrayHandle<Scalar4> d_pos(m_pdata->getPositions(), access_location::device, access_mode::read);
        ArrayHandle<unsigned int> d_sorted_indexes(m_sorted_indexes, access_location::device, access_mode::read);

        const BoxDim lbvh_box = getLBVHBox();

        for (unsigned int i=0; i < m_pdata->getNTypes(); ++i)
            {
            const unsigned int first = h_type_first.data[i];
            const unsigned int last = h_type_last.data[i];

            if (first != NeigborListTypeSentinel)
                {
                m_lbvhs[i]->build(PointMapInsertOp(d_pos.data, d_sorted_indexes.data + first, last-first),
                                  lbvh_box.getLo(),
                                  lbvh_box.getHi());
                }
            else
                {
                // effectively destroy the lbvh
                m_lbvhs[i]->build(PointMapInsertOp(d_pos.data, NULL, 0), lbvh_box.getLo(), lbvh_box.getHi());
                }
            }
        }

    // put particles in primitive order for traversal, filtering out ghosts
        {
        ArrayHandle<unsigned int> h_type_first(m_type_first, access_location::host, access_mode::read);
        ArrayHandle<unsigned int> d_traverse_order(m_traverse_order, access_location::device, access_mode::overwrite);
        ArrayHandle<unsigned int> d_sorted_indexes(m_sorted_indexes, access_location::device, access_mode::read);

        unsigned int Ntotal = 0;
        for (unsigned int i=0; i < m_pdata->getNTypes(); ++i)
            {
            const unsigned int Ni = m_lbvhs[i]->getN();
            if (Ni)
                {
                const unsigned int first = h_type_first.data[i];
                ArrayHandle<unsigned int> d_primitives(m_lbvhs[i]->getPrimitives(), access_location::device, access_mode::read);
                gpu_nlist_copy_primitives(d_traverse_order.data + first,
                                          d_sorted_indexes.data + first,
                                          d_primitives.data,
                                          Ni,
                                          128);
                Ntotal += Ni;
                }
            }

        #if ENABLE_MPI
        // stream compact to get rid of ghosts
        if (m_pdata->getNGhosts() > 0)
            {
            Ntotal = gpu_nlist_remove_ghosts(d_traverse_order.data, Ntotal, m_pdata->getN());
            }
        #endif // ENABLE_MPI

        // check that all primitives are going to be traversed
        if (Ntotal != m_pdata->getN())
            {
            m_exec_conf->msg->error() << "Wrong number of particles in nlist.tree() arrays!" << std::endl;
            throw std::runtime_error("Wrong number of particles in nlist.tree() arrays!");
            }
        }
    }

void NeighborListGPUTree::traverseTree()
    {
    ArrayHandle<unsigned int> d_nlist(m_nlist, access_location::device, access_mode::overwrite);
    ArrayHandle<unsigned int> d_n_neigh(m_n_neigh, access_location::device, access_mode::overwrite);
    ArrayHandle<unsigned int> d_conditions(m_conditions, access_location::device, access_mode::readwrite);
    ArrayHandle<unsigned int> d_head_list(m_head_list, access_location::device, access_mode::read);
    ArrayHandle<unsigned int> d_Nmax(m_Nmax, access_location::device, access_mode::read);
    NeighborListOp nlist_op(d_nlist.data, d_n_neigh.data, d_conditions.data, d_head_list.data, d_Nmax.data);

    ArrayHandle<unsigned int> h_type_first(m_type_first, access_location::host, access_mode::read);
    ArrayHandle<unsigned int> d_sorted_indexes(m_sorted_indexes, access_location::device, access_mode::read);

    ArrayHandle<Scalar4> d_pos(m_pdata->getPositions(), access_location::device, access_mode::read);
    ArrayHandle<unsigned int> d_body(m_pdata->getBodies(), access_location::device, access_mode::read);
    ArrayHandle<Scalar> d_diam(m_pdata->getDiameters(), access_location::device, access_mode::read);
    ArrayHandle<unsigned int> d_traverse_order(m_traverse_order, access_location::device, access_mode::read);
    ArrayHandle<Scalar> d_r_cut(m_r_cut, access_location::device, access_mode::read);

    Scalar rpad = (m_diameter_shift) ? m_d_max - Scalar(1.0) : Scalar(0.0);

    // clear the neighbor counts
    cudaMemset(d_n_neigh.data, 0, sizeof(unsigned int)*m_pdata->getN());

    for (unsigned int i=0; i < m_pdata->getNTypes(); ++i)
        {
        if (m_lbvhs[i]->getN())
            {
            const unsigned int first = h_type_first.data[i];
            neighbor::MapTransformOp map(d_sorted_indexes.data + first);

            ParticleQueryOp query_op(d_pos.data,
                                     (m_filter_body) ? d_body.data : NULL,
                                     (m_diameter_shift) ? d_diam.data : NULL,
                                     d_traverse_order.data,
                                     m_pdata->getN(),
                                     d_r_cut.data,
                                     m_r_buff,
                                     rpad,
                                     m_typpair_idx,
                                     i);

            m_traversers[i]->traverse(nlist_op, query_op, map, *m_lbvhs[i], m_image_list);
            }
        }
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
    unsigned char sys3d = (m_sysdef->getNDimensions() == 3);

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
        m_exec_conf->msg->error() << "nlist: Simulation box is too small! Particles would be interacting with themselves." << std::endl;
        throw std::runtime_error("Error updating neighborlist bins");
        }

    // now compute the image vectors
    // each dimension increases by one power of 3
    unsigned int n_dim_periodic = (periodic.x + periodic.y + sys3d*periodic.z);
    m_n_images = 1;
    for (unsigned int dim = 0; dim < n_dim_periodic; ++dim)
        {
        m_n_images *= 3;
        }
    m_n_images -= 1; // remove the self image

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
    unsigned int n_images = 0;
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

void export_NeighborListGPUTree(py::module& m)
    {
    py::class_<NeighborListGPUTree, std::shared_ptr<NeighborListGPUTree> >(m, "NeighborListGPUTree", py::base<NeighborListGPU>())
    .def(py::init< std::shared_ptr<SystemDefinition>, Scalar, Scalar >());
    }
