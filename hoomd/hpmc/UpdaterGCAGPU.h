// Copyright (c) 2009-2025 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

#pragma once

#ifdef ENABLE_HIP

#include "UpdaterGCA.h"
#include "UpdaterGCAGPU.cuh"

#include <hip/hip_runtime.h>

namespace hoomd
    {
namespace hpmc
    {
/*!
   Implementation of UpdaterGCA on the GPU
*/

template<class Shape> class UpdaterGCAGPU : public UpdaterGCA<Shape>
    {
    public:
    //! Constructor
    /*! \param sysdef System definition
        \param mc HPMC integrator
        \param seed PRNG seed
    */
    UpdaterGCAGPU(std::shared_ptr<SystemDefinition> sysdef,
                  std::shared_ptr<Trigger> trigger,
                  std::shared_ptr<IntegratorHPMCMono<Shape>> mc,
                  std::shared_ptr<CellList> cl);

    //! Destructor
    virtual ~UpdaterGCAGPU();

    //! Take one timestep forward
    /*! \param timestep timestep at which update is being evaluated
     */
    virtual void update(uint64_t timestep);

    protected:
    GPUArray<unsigned int> m_adjacency; //!< List of overlaps between old and new configuration
    GPUVector<uint2>
        m_adjacency_copy;        //!< List of overlaps between old and new configuration, contiguous
    GPUVector<int> m_components; //!< The connected component labels per particle

    std::shared_ptr<CellList> m_cl; //!< Cell list
    uint3 m_last_dim;               //!< Dimensions of the cell list on the last call to update
    unsigned int m_last_nmax;       //!< Last cell list NMax value allocated in excell

    /// Autotuner for excell block_size.
    std::shared_ptr<Autotuner<1>> m_tuner_excell_block_size;

    /// Autotuner for overlap checks.
    std::shared_ptr<Autotuner<3>> m_tuner_overlaps;

    /// Autotuner for contenating the per-particle neighbor lists.
    std::shared_ptr<Autotuner<2>> m_tuner_concatenate;

    /// Autotuner for transforming particles.
    std::shared_ptr<Autotuner<1>> m_tuner_transform;

    /// Autotuner for flipping clusters.
    std::shared_ptr<Autotuner<1>> m_tuner_flip;

    GPUArray<unsigned int> m_excell_idx;  //!< Particle indices in expanded cells
    GPUArray<unsigned int> m_excell_size; //!< Number of particles in each expanded cell
    Index2D m_excell_list_indexer;        //!< Indexer to access elements of the excell_idx list

    GPUVector<unsigned int> m_nneigh;      //!< Number of neighbors
    GPUVector<unsigned int> m_nneigh_scan; //!< Exclusive prefix sum over number of neighbors
    unsigned int m_maxn;                   //!< Max number of neighbors
    GPUArray<unsigned int> m_overflow;     //!< Overflow condition for neighbor list

    hipStream_t m_overlaps_stream; //!< Stream for overlaps kernel

    //! Determine connected components of the interaction graph
    virtual void connectedComponents();

    // backup current particle data
    virtual void backupState();

    /*! \param timestep Current time step
     */
    virtual void
    findInteractions(uint64_t timestep, const quat<Scalar> q, const vec3<Scalar> pivot, bool line);

    //! Transform particles
    virtual void transform(const quat<Scalar>& q, const vec3<Scalar>& pivot, bool line);

    //! Flip clusters randomly
    virtual void flip(uint64_t timestep);

    //! Set up excell_list
    virtual void initializeExcellMem();

    //! Check if memory reallocation for the adjacency list is necessary
    virtual bool checkReallocate();
    };

template<class Shape>
UpdaterGCAGPU<Shape>::UpdaterGCAGPU(std::shared_ptr<SystemDefinition> sysdef,
                                    std::shared_ptr<Trigger> trigger,
                                    std::shared_ptr<IntegratorHPMCMono<Shape>> mc,
                                    std::shared_ptr<CellList> cl)
    : UpdaterGCA<Shape>(sysdef, trigger, mc), m_cl(cl)
    {
    this->m_exec_conf->msg->notice(5) << "Constructing UpdaterGCAGPU" << std::endl;

    this->m_cl->setRadius(1);
    this->m_cl->setComputeTypeBody(false);
    this->m_cl->setFlagType();
    this->m_cl->setComputeIdx(true);

    // set last dim to a bogus value so that it will re-init on the first call
    m_last_dim = make_uint3(0xffffffff, 0xffffffff, 0xffffffff);
    m_last_nmax = 0xffffffff;

    hipDeviceProp_t dev_prop = this->m_exec_conf->dev_prop;
    m_tuner_excell_block_size.reset(
        new Autotuner<1>({AutotunerBase::makeBlockSizeRange(this->m_exec_conf)},
                         this->m_exec_conf,
                         "clusters_excell_block_size"));

    m_tuner_transform.reset(new Autotuner<1>({AutotunerBase::makeBlockSizeRange(this->m_exec_conf)},
                                             this->m_exec_conf,
                                             "clusters_transform"));

    m_tuner_flip.reset(new Autotuner<1>({AutotunerBase::makeBlockSizeRange(this->m_exec_conf)},
                                        this->m_exec_conf,
                                        "clusters_flip"));

    // Tuning parameters for overlap checks:
    // 0: block size
    // 1: threads per particle
    // 2: overlap threads

    // Only widen the parallelism if the shape supports it, and limit parallelism to fit within the
    // warp.
    std::function<bool(const std::array<unsigned int, 3>&)> is_overlaps_parameter_valid
        = [](const std::array<unsigned int, 3>& parameter) -> bool
    {
        unsigned int block_size = parameter[0];
        unsigned int threads_per_particle = parameter[1];
        unsigned int overlap_threads = parameter[2];
        return (threads_per_particle == 1 || Shape::isParallel())
               && (threads_per_particle * overlap_threads <= block_size)
               && (block_size % (threads_per_particle * overlap_threads)) == 0;
    };

    const unsigned int overlaps_max_tpp = dev_prop.maxThreadsDim[2];

    m_tuner_overlaps.reset(
        new Autotuner<3>({AutotunerBase::makeBlockSizeRange(this->m_exec_conf),
                          AutotunerBase::getTppListPow2(this->m_exec_conf, overlaps_max_tpp),
                          AutotunerBase::getTppListPow2(this->m_exec_conf)},
                         this->m_exec_conf,
                         "clusters_overlaps",
                         3,
                         true,
                         is_overlaps_parameter_valid));

    // Tuning parameters for nlist concatenation kernel:
    // 0: block size
    // 1: threads per particle

    std::function<bool(const std::array<unsigned int, 3>&)> is_dconcatenate_parameter_valid
        = [](const std::array<unsigned int, 3>& parameter) -> bool
    {
        unsigned int block_size = parameter[0];
        unsigned int threads_per_particle = parameter[1];
        return (block_size % threads_per_particle) == 0;
    };

    m_tuner_concatenate.reset(
        new Autotuner<2>({AutotunerBase::makeBlockSizeRange(this->m_exec_conf),
                          AutotunerBase::getTppListPow2(this->m_exec_conf, overlaps_max_tpp)},
                         this->m_exec_conf,
                         "clusters_concatenate"));

    this->m_autotuners.insert(this->m_autotuners.end(),
                              {m_tuner_excell_block_size,
                               m_tuner_transform,
                               m_tuner_flip,
                               m_tuner_overlaps,
                               m_tuner_concatenate});

    GPUArray<unsigned int> excell_size(0, this->m_exec_conf);
    m_excell_size.swap(excell_size);

    GPUArray<unsigned int> excell_idx(0, this->m_exec_conf);
    m_excell_idx.swap(excell_idx);

    // allocate memory for connected components
    GPUArray<unsigned int>(1, this->m_exec_conf).swap(m_adjacency);
    GPUVector<int>(this->m_exec_conf).swap(m_components);

    GPUVector<uint2>(this->m_exec_conf).swap(m_adjacency_copy);

    this->m_exec_conf->setDevice();
    hipStreamCreate(&m_overlaps_stream);

    GPUArray<unsigned int>(1, this->m_exec_conf).swap(m_nneigh);

    GPUArray<unsigned int>(1, this->m_exec_conf).swap(m_nneigh_scan);

    m_maxn = 0;
    GPUArray<unsigned int>(1, this->m_exec_conf).swap(m_overflow);

        {
        ArrayHandle<unsigned int> h_overflow(m_overflow,
                                             access_location::host,
                                             access_mode::overwrite);
        *h_overflow.data = 0;
        }
    }

template<class Shape> UpdaterGCAGPU<Shape>::~UpdaterGCAGPU()
    {
    this->m_exec_conf->msg->notice(5) << "Destroying UpdaterGCAGPU" << std::endl;

    this->m_exec_conf->setDevice();
    hipStreamDestroy(m_overlaps_stream);
    }

/*! Perform a cluster move
    \param timestep Current time step of the simulation
*/
template<class Shape> void UpdaterGCAGPU<Shape>::update(uint64_t timestep)
    {
    Updater::update(timestep);
    // compute nominal cell width
    Scalar nominal_width = this->m_mc->getMaxCoreDiameter();
    if (this->m_cl->getNominalWidth() != nominal_width)
        this->m_cl->setNominalWidth(nominal_width);

    // update the cell list before re-initializing
    this->m_cl->compute(timestep);

    // perform the update
    UpdaterGCA<Shape>::update(timestep);
    }

template<class Shape> void UpdaterGCAGPU<Shape>::connectedComponents()
    {
    // this will contain the number of strongly connected components
    unsigned int num_components = 0;

    m_components.resize(this->m_pdata->getN());

    // access edges of adajacency matrix
    ArrayHandle<unsigned int> d_adjacency(m_adjacency, access_location::device, access_mode::read);
    ArrayHandle<unsigned int> d_nneigh(m_nneigh, access_location::device, access_mode::read);
    ArrayHandle<unsigned int> d_nneigh_scan(m_nneigh_scan,
                                            access_location::device,
                                            access_mode::overwrite);

    // determine total size of adjacency list, and do prefix sum
    unsigned int nneigh_total;
    this->m_exec_conf->setDevice();
    gpu::get_num_neighbors(d_nneigh.data,
                           d_nneigh_scan.data,
                           nneigh_total,
                           this->m_pdata->getN(),
                           this->m_exec_conf->getCachedAllocatorManaged());
    if (this->m_exec_conf->isCUDAErrorCheckingEnabled())
        CHECK_CUDA_ERROR();

    // allocate memory for contiguous adjacency list, factor of two for symmetry
    m_adjacency_copy.resize(nneigh_total * 2);

    ArrayHandle<uint2> d_adjacency_copy(m_adjacency_copy,
                                        access_location::device,
                                        access_mode::overwrite);

    // concatenate per-particle neighbor lists into a single one
    auto param = m_tuner_concatenate->getParam();
    unsigned int block_size = param[0];
    unsigned int group_size = param[1];
    m_tuner_concatenate->begin();
    gpu::concatenate_adjacency_list(d_adjacency.data,
                                    d_nneigh.data,
                                    d_nneigh_scan.data,
                                    m_maxn,
                                    d_adjacency_copy.data,
                                    this->m_pdata->getN(),
                                    block_size,
                                    group_size);
    if (this->m_exec_conf->isCUDAErrorCheckingEnabled())
        CHECK_CUDA_ERROR();
    m_tuner_concatenate->end();

        {
        // access the output array
        ArrayHandle<int> d_components(m_components,
                                      access_location::device,
                                      access_mode::overwrite);

        gpu::connected_components(d_adjacency_copy.data,
                                  this->m_pdata->getN(),
                                  nneigh_total * 2,
                                  d_components.data,
                                  num_components,
                                  this->m_exec_conf->dev_prop,
                                  this->m_exec_conf->getCachedAllocator());

        if (this->m_exec_conf->isCUDAErrorCheckingEnabled())
            CHECK_CUDA_ERROR();
        }

    // count clusters
    this->m_count_total.n_particles_in_clusters += this->m_pdata->getN();
    this->m_count_total.n_clusters += num_components;
    }

template<class Shape> void UpdaterGCAGPU<Shape>::initializeExcellMem()
    {
    this->m_exec_conf->msg->notice(4) << "hpmc resizing expanded cells" << std::endl;

    // get the current cell dimensions
    unsigned int num_cells = this->m_cl->getCellIndexer().getNumElements();
    unsigned int num_adj = this->m_cl->getCellAdjIndexer().getW();
    unsigned int num_max = this->m_cl->getNmax();

    // make the excell dimensions the same, but with room for Nmax*Nadj in each cell
    m_excell_list_indexer = Index2D(num_max * num_adj, num_cells);

    // reallocate memory
    m_excell_idx.resize(m_excell_list_indexer.getNumElements());
    m_excell_size.resize(num_cells);
    }

template<class Shape> void UpdaterGCAGPU<Shape>::backupState()
    {
    unsigned int nptl = this->m_pdata->getN();

    // resize as necessary
    this->m_postype_backup.resize(nptl);
    this->m_orientation_backup.resize(nptl);
    this->m_image_backup.resize(nptl);

        {
        ArrayHandle<Scalar4> d_postype(this->m_pdata->getPositions(),
                                       access_location::device,
                                       access_mode::read);
        ArrayHandle<Scalar4> d_orientation(this->m_pdata->getOrientationArray(),
                                           access_location::device,
                                           access_mode::read);
        ArrayHandle<int3> d_image(this->m_pdata->getImages(),
                                  access_location::device,
                                  access_mode::read);

        ArrayHandle<Scalar4> d_postype_backup(this->m_postype_backup,
                                              access_location::device,
                                              access_mode::overwrite);
        ArrayHandle<Scalar4> d_orientation_backup(this->m_orientation_backup,
                                                  access_location::device,
                                                  access_mode::overwrite);
        ArrayHandle<int3> d_image_backup(this->m_image_backup,
                                         access_location::device,
                                         access_mode::overwrite);

        // copy over data
        this->m_exec_conf->setDevice();

        if (nptl != 0)
            {
            hipMemcpyAsync(d_postype_backup.data,
                           d_postype.data,
                           sizeof(Scalar4) * nptl,
                           hipMemcpyDeviceToDevice);
            hipMemcpyAsync(d_orientation_backup.data,
                           d_orientation.data,
                           sizeof(Scalar4) * nptl,
                           hipMemcpyDeviceToDevice);
            hipMemcpyAsync(d_image_backup.data,
                           d_image.data,
                           sizeof(int3) * nptl,
                           hipMemcpyDeviceToDevice);
            }
        if (this->m_exec_conf->isCUDAErrorCheckingEnabled())
            CHECK_CUDA_ERROR();
        }
    }

template<class Shape>
void UpdaterGCAGPU<Shape>::transform(const quat<Scalar>& q, const vec3<Scalar>& pivot, bool line)
    {
    ArrayHandle<Scalar4> d_postype(this->m_pdata->getPositions(),
                                   access_location::device,
                                   access_mode::readwrite);
    ArrayHandle<Scalar4> d_orientation(this->m_pdata->getOrientationArray(),
                                       access_location::device,
                                       access_mode::readwrite);
    ArrayHandle<int3> d_image(this->m_pdata->getImages(),
                              access_location::device,
                              access_mode::readwrite);

    auto params = this->m_mc->getParams();
    unsigned int block_size = m_tuner_transform->getParam()[0];
    gpu::clusters_transform_args_t args(d_postype.data,
                                        d_orientation.data,
                                        d_image.data,
                                        pivot,
                                        q,
                                        line,
                                        this->m_pdata->getN(),
                                        this->m_pdata->getGlobalBox(),
                                        this->m_pdata->getNTypes(),
                                        block_size);

    m_tuner_transform->begin();
    gpu::transform_particles<Shape>(args, params.data());
    if (this->m_exec_conf->isCUDAErrorCheckingEnabled())
        CHECK_CUDA_ERROR();
    m_tuner_transform->end();
    }

template<class Shape> void UpdaterGCAGPU<Shape>::flip(uint64_t timestep)
    {
    ArrayHandle<Scalar4> d_postype(this->m_pdata->getPositions(),
                                   access_location::device,
                                   access_mode::readwrite);
    ArrayHandle<Scalar4> d_orientation(this->m_pdata->getOrientationArray(),
                                       access_location::device,
                                       access_mode::readwrite);
    ArrayHandle<int3> d_image(this->m_pdata->getImages(),
                              access_location::device,
                              access_mode::readwrite);

    ArrayHandle<Scalar4> d_postype_backup(this->m_postype_backup,
                                          access_location::device,
                                          access_mode::read);
    ArrayHandle<Scalar4> d_orientation_backup(this->m_orientation_backup,
                                              access_location::device,
                                              access_mode::read);
    ArrayHandle<int3> d_image_backup(this->m_image_backup,
                                     access_location::device,
                                     access_mode::read);

    ArrayHandle<int> d_components(m_components, access_location::device, access_mode::read);

    m_tuner_flip->begin();
    gpu::flip_clusters(d_postype.data,
                       d_orientation.data,
                       d_image.data,
                       d_postype_backup.data,
                       d_orientation_backup.data,
                       d_image_backup.data,
                       d_components.data,
                       (float)this->m_flip_probability,
                       this->m_sysdef->getSeed(),
                       timestep,
                       this->m_pdata->getN(),
                       m_tuner_flip->getParam()[0]);
    if (this->m_exec_conf->isCUDAErrorCheckingEnabled())
        CHECK_CUDA_ERROR();
    m_tuner_flip->end();
    }

template<class Shape>
void UpdaterGCAGPU<Shape>::findInteractions(uint64_t timestep,
                                            const quat<Scalar> q,
                                            const vec3<Scalar> pivot,
                                            bool line)
    {
    const auto& params = this->m_mc->getParams();

    if (this->m_pdata->getN() == 0)
        {
        return;
        }
    // check if we are below a minimum image convention box size
    // the minimum image convention comes from the global box, not the local one
    BoxDim global_box = this->m_pdata->getGlobalBox();
    Scalar3 nearest_plane_distance = global_box.getNearestPlaneDistance();

    Scalar nominal_width = this->m_cl->getNominalWidth();

    if ((global_box.getPeriodic().x && nearest_plane_distance.x <= nominal_width * 2)
        || (global_box.getPeriodic().y && nearest_plane_distance.y <= nominal_width * 2)
        || (this->m_sysdef->getNDimensions() == 3 && global_box.getPeriodic().z
            && nearest_plane_distance.z <= nominal_width * 2))
        {
        this->m_exec_conf->msg->error()
            << "Simulation box too small for GPU accelerated HPMC execution - increase it so "
               "the minimum image convention works"
            << std::endl;
        throw std::runtime_error("Error performing HPMC update");
        }

    // if the cell list is a different size than last time, reinitialize the expanded cell list
    uint3 cur_dim = this->m_cl->getDim();
    if (m_last_dim.x != cur_dim.x || m_last_dim.y != cur_dim.y || m_last_dim.z != cur_dim.z
        || m_last_nmax != this->m_cl->getNmax())
        {
        initializeExcellMem();

        m_last_dim = cur_dim;
        m_last_nmax = this->m_cl->getNmax();
        }

    // access the cell list data
    ArrayHandle<unsigned int> d_cell_size(this->m_cl->getCellSizeArray(),
                                          access_location::device,
                                          access_mode::read);
    ArrayHandle<unsigned int> d_cell_idx(this->m_cl->getIndexArray(),
                                         access_location::device,
                                         access_mode::read);
    ArrayHandle<unsigned int> d_cell_adj(this->m_cl->getCellAdjArray(),
                                         access_location::device,
                                         access_mode::read);

    // access the parameters and interaction matrix
    ArrayHandle<unsigned int> d_overlaps(this->m_mc->getInteractionMatrix(),
                                         access_location::device,
                                         access_mode::read);

    BoxDim box = this->m_pdata->getBox();

    Scalar3 ghost_width = this->m_cl->getGhostWidth();

    // expanded cells & neighbor list
    ArrayHandle<unsigned int> d_excell_idx(m_excell_idx,
                                           access_location::device,
                                           access_mode::overwrite);
    ArrayHandle<unsigned int> d_excell_size(m_excell_size,
                                            access_location::device,
                                            access_mode::overwrite);

    // update the expanded cells
    this->m_tuner_excell_block_size->begin();
    gpu::hpmc_excell(d_excell_idx.data,
                     d_excell_size.data,
                     m_excell_list_indexer,
                     d_cell_idx.data,
                     d_cell_size.data,
                     d_cell_adj.data,
                     this->m_cl->getCellIndexer(),
                     this->m_cl->getCellListIndexer(),
                     this->m_cl->getCellAdjIndexer(),
                     this->m_tuner_excell_block_size->getParam()[0]);
    if (this->m_exec_conf->isCUDAErrorCheckingEnabled())
        CHECK_CUDA_ERROR();
    this->m_tuner_excell_block_size->end();

    bool reallocate = false;

    // allocate memory for number of neighbors
    m_nneigh.resize(this->m_pdata->getN());
    m_nneigh_scan.resize(this->m_pdata->getN());

    do
        {
            { // ArrayHandle scope
            ArrayHandle<unsigned int> d_adjacency(m_adjacency,
                                                  access_location::device,
                                                  access_mode::overwrite);
            ArrayHandle<unsigned int> d_nneigh(m_nneigh,
                                               access_location::device,
                                               access_mode::overwrite);

            // access backup particle data
            ArrayHandle<Scalar4> d_postype_backup(this->m_postype_backup,
                                                  access_location::device,
                                                  access_mode::read);
            ArrayHandle<Scalar4> d_orientation_backup(this->m_orientation_backup,
                                                      access_location::device,
                                                      access_mode::read);

            // access the particle data
            ArrayHandle<Scalar4> d_postype(this->m_pdata->getPositions(),
                                           access_location::device,
                                           access_mode::read);
            ArrayHandle<Scalar4> d_orientation(this->m_pdata->getOrientationArray(),
                                               access_location::device,
                                               access_mode::read);

            ArrayHandle<unsigned int> d_overflow(m_overflow,
                                                 access_location::device,
                                                 access_mode::readwrite);

            // fill the parameter structure for the GPU kernel
            gpu::cluster_args_t args(d_postype_backup.data,
                                     d_orientation_backup.data,
                                     this->m_cl->getCellIndexer(),
                                     this->m_cl->getDim(),
                                     ghost_width,
                                     this->m_pdata->getN(),
                                     this->m_pdata->getNTypes(),
                                     this->m_sysdef->getSeed(),
                                     d_overlaps.data,
                                     this->m_mc->getOverlapIndexer(),
                                     timestep,
                                     box,
                                     0, // block size
                                     0, // tpp
                                     0, // overlap_threads
                                     d_postype.data,
                                     d_orientation.data,
                                     d_excell_idx.data,
                                     d_excell_size.data,
                                     m_excell_list_indexer,
                                     d_adjacency.data,
                                     d_nneigh.data,
                                     m_maxn,
                                     d_overflow.data,
                                     this->m_sysdef->getNDimensions(),
                                     line,
                                     pivot,
                                     q,
                                     true,
                                     this->m_exec_conf->dev_prop,
                                     m_overlaps_stream);

            // reset number of neighbors
            if (this->m_pdata->getN() != 0)
                hipMemsetAsync(d_nneigh.data, 0, sizeof(unsigned int) * this->m_pdata->getN());
            if (this->m_exec_conf->isCUDAErrorCheckingEnabled())
                CHECK_CUDA_ERROR();

            /*
             *  check overlaps, new configuration simultaneously against old configuration
             */

            m_tuner_overlaps->begin();
            auto param = m_tuner_overlaps->getParam();
            args.block_size = param[0];
            args.tpp = param[1];
            args.overlap_threads = param[2];
            gpu::hpmc_cluster_overlaps<Shape>(args, params.data());
            if (this->m_exec_conf->isCUDAErrorCheckingEnabled())
                CHECK_CUDA_ERROR();
            m_tuner_overlaps->end();
            }
        reallocate = checkReallocate();
        } while (reallocate);
    }

template<class Shape> bool UpdaterGCAGPU<Shape>::checkReallocate()
    {
    // read back overflow condition and resize as necessary
    ArrayHandle<unsigned int> h_overflow(m_overflow, access_location::host, access_mode::read);
    unsigned int req_maxn = *h_overflow.data;

    bool maxn_changed = false;
    if (req_maxn > m_maxn)
        {
        m_maxn = req_maxn;
        maxn_changed = true;
        }

    unsigned int req_size_nlist = m_maxn * this->m_pdata->getN();

    // resize
    bool reallocate = req_size_nlist > m_adjacency.getNumElements();
    if (reallocate)
        {
        this->m_exec_conf->msg->notice(9)
            << "hpmc clusters resizing neighbor list " << m_adjacency.getNumElements() << " -> "
            << req_size_nlist << std::endl;

        GPUArray<unsigned int> adjacency(req_size_nlist, this->m_exec_conf);
        m_adjacency.swap(adjacency);
        }
    return reallocate || maxn_changed;
    }

namespace detail
    {
template<class Shape> void export_UpdaterGCAGPU(pybind11::module& m, const std::string& name)
    {
    pybind11::class_<UpdaterGCAGPU<Shape>,
                     UpdaterGCA<Shape>,
                     std::shared_ptr<UpdaterGCAGPU<Shape>>>(m, name.c_str())
        .def(pybind11::init<std::shared_ptr<SystemDefinition>,
                            std::shared_ptr<Trigger>,
                            std::shared_ptr<IntegratorHPMCMono<Shape>>,
                            std::shared_ptr<CellList>>());
    }

    } // end namespace detail
    } // end namespace hpmc
    } // end namespace hoomd

#endif // ENABLE_CUDA
