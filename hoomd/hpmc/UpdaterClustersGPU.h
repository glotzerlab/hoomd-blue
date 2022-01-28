// Copyright (c) 2009-2022 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

#ifndef _UPDATER_HPMC_CLUSTERS_GPU_
#define _UPDATER_HPMC_CLUSTERS_GPU_

/*! \file UpdaterBoxClusters.h
    \brief Declaration of UpdaterBoxClusters
*/

#ifdef ENABLE_HIP

#include "IntegratorHPMCMonoGPUDepletants.cuh"
#include "UpdaterClusters.h"
#include "UpdaterClustersGPU.cuh"
#include "UpdaterClustersGPUDepletants.cuh"

#include <hip/hip_runtime.h>

namespace hoomd
    {
namespace hpmc
    {
/*!
   Implementation of UpdaterClusters on the GPU
*/

template<class Shape> class UpdaterClustersGPU : public UpdaterClusters<Shape>
    {
    public:
    //! Constructor
    /*! \param sysdef System definition
        \param mc HPMC integrator
        \param seed PRNG seed
    */
    UpdaterClustersGPU(std::shared_ptr<SystemDefinition> sysdef,
                       std::shared_ptr<IntegratorHPMCMono<Shape>> mc,
                       std::shared_ptr<CellList> cl);

    //! Destructor
    virtual ~UpdaterClustersGPU();

    //! Set autotuner parameters
    /*! \param enable Enable/disable autotuning
        \param period period (approximate) in time steps when returning occurs
    */
    virtual void setAutotunerParams(bool enable, unsigned int period)
        {
        m_tuner_excell_block_size->setPeriod(period);
        m_tuner_excell_block_size->setEnabled(enable);

        m_tuner_overlaps->setPeriod(period);
        m_tuner_overlaps->setEnabled(enable);

        m_tuner_depletants->setPeriod(period);
        m_tuner_depletants->setEnabled(enable);

        m_tuner_num_depletants->setPeriod(period);
        m_tuner_num_depletants->setEnabled(enable);

        m_tuner_concatenate->setPeriod(period);
        m_tuner_concatenate->setEnabled(enable);

        m_tuner_transform->setPeriod(period);
        m_tuner_transform->setEnabled(enable);

        m_tuner_flip->setPeriod(period);
        m_tuner_flip->setEnabled(enable);
        }

    //! Take one timestep forward
    /*! \param timestep timestep at which update is being evaluated
     */
    virtual void update(uint64_t timestep);

    protected:
    GlobalArray<unsigned int> m_adjacency; //!< List of overlaps between old and new configuration
    GlobalVector<uint2>
        m_adjacency_copy; //!< List of overlaps between old and new configuration, contiguous
    GlobalVector<int> m_components; //!< The connected component labels per particle

    std::shared_ptr<CellList> m_cl; //!< Cell list
    uint3 m_last_dim;               //!< Dimensions of the cell list on the last call to update
    unsigned int m_last_nmax;       //!< Last cell list NMax value allocated in excell

    std::unique_ptr<Autotuner> m_tuner_excell_block_size; //!< Autotuner for excell block_size
    std::unique_ptr<Autotuner> m_tuner_overlaps;          //!< Autotuner for overlap checks
    std::unique_ptr<Autotuner> m_tuner_depletants;        //!< Autotuner for inserting depletants
    std::unique_ptr<Autotuner>
        m_tuner_num_depletants; //!< Autotuner for calculating number of depletants
    std::unique_ptr<Autotuner>
        m_tuner_concatenate; //!< Autotuner for contenating the per-particle neighbor lists
    std::unique_ptr<Autotuner> m_tuner_transform; //!< Autotuner for transforming particles
    std::unique_ptr<Autotuner> m_tuner_flip;      //!< Autotuner for flipping clusters

    GlobalArray<unsigned int> m_excell_idx;  //!< Particle indices in expanded cells
    GlobalArray<unsigned int> m_excell_size; //!< Number of particles in each expanded cell
    Index2D m_excell_list_indexer;           //!< Indexer to access elements of the excell_idx list

    GlobalVector<unsigned int> m_nneigh;      //!< Number of neighbors
    GlobalVector<unsigned int> m_nneigh_scan; //!< Exclusive prefix sum over number of neighbors
    unsigned int m_maxn;                      //!< Max number of neighbors
    GlobalArray<unsigned int> m_overflow;     //!< Overflow condition for neighbor list

    GPUPartition m_old_gpu_partition;          //!< The partition in the old configuration
    GlobalVector<unsigned int> m_n_depletants; //!< List of number of depletants, per particle

    std::vector<hipStream_t> m_overlaps_streams; //!< Stream for overlaps kernel, per device
    std::vector<std::vector<hipStream_t>>
        m_depletant_streams; //!< Stream for every particle type, and device

    //!< Variables for implicit depletants
    GlobalArray<Scalar> m_lambda; //!< Poisson means, per type pair

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

    //! Update GPU memory hints
    virtual void updateGPUAdvice();

    //! Check if memory reallocation for the adjacency list is necessary
    virtual bool checkReallocate();
    };

template<class Shape>
UpdaterClustersGPU<Shape>::UpdaterClustersGPU(std::shared_ptr<SystemDefinition> sysdef,
                                              std::shared_ptr<IntegratorHPMCMono<Shape>> mc,
                                              std::shared_ptr<CellList> cl)
    : UpdaterClusters<Shape>(sysdef, mc), m_cl(cl)
    {
    this->m_exec_conf->msg->notice(5) << "Constructing UpdaterClustersGPU" << std::endl;

    this->m_cl->setRadius(1);
    this->m_cl->setComputeTDB(false);
    this->m_cl->setFlagType();
    this->m_cl->setComputeIdx(true);

    // with multiple GPUs, request a cell list per device
    m_cl->setPerDevice(this->m_exec_conf->allConcurrentManagedAccess());

    // set last dim to a bogus value so that it will re-init on the first call
    m_last_dim = make_uint3(0xffffffff, 0xffffffff, 0xffffffff);
    m_last_nmax = 0xffffffff;

    hipDeviceProp_t dev_prop = this->m_exec_conf->dev_prop;
    m_tuner_excell_block_size.reset(new Autotuner(dev_prop.warpSize,
                                                  dev_prop.maxThreadsPerBlock,
                                                  dev_prop.warpSize,
                                                  5,
                                                  1000000,
                                                  "clusters_excell_block_size",
                                                  this->m_exec_conf));
    m_tuner_num_depletants.reset(new Autotuner(dev_prop.warpSize,
                                               dev_prop.maxThreadsPerBlock,
                                               dev_prop.warpSize,
                                               5,
                                               1000000,
                                               "clusters_num_depletants",
                                               this->m_exec_conf));
    m_tuner_transform.reset(new Autotuner(dev_prop.warpSize,
                                          dev_prop.maxThreadsPerBlock,
                                          dev_prop.warpSize,
                                          5,
                                          1000000,
                                          "clusters_transform",
                                          this->m_exec_conf));
    m_tuner_flip.reset(new Autotuner(dev_prop.warpSize,
                                     dev_prop.maxThreadsPerBlock,
                                     dev_prop.warpSize,
                                     5,
                                     1000000,
                                     "clusters_flip",
                                     this->m_exec_conf));

    // tuning parameters for overlap checks
    std::vector<unsigned int> valid_params;
    unsigned int warp_size = this->m_exec_conf->dev_prop.warpSize;
    const unsigned int overlaps_max_tpp = dev_prop.maxThreadsDim[2];
    for (unsigned int block_size = warp_size;
         block_size <= (unsigned int)dev_prop.maxThreadsPerBlock;
         block_size += warp_size)
        {
        for (auto s : Autotuner::getTppListPow2(overlaps_max_tpp))
            {
            for (auto t : Autotuner::getTppListPow2(warp_size))
                {
                // only widen the parallelism if the shape supports it
                if (t == 1 || Shape::isParallel())
                    {
                    if ((s * t <= block_size) && ((block_size % (s * t)) == 0))
                        valid_params.push_back(block_size * 1000000 + s * 100 + t);
                    }
                }
            }
        }

    m_tuner_overlaps.reset(
        new Autotuner(valid_params, 5, 100000, "clusters_overlaps", this->m_exec_conf));

    // tuning parameters for depletants
    std::vector<unsigned int> valid_params_depletants;
    for (unsigned int block_size = dev_prop.warpSize;
         block_size <= (unsigned int)dev_prop.maxThreadsPerBlock;
         block_size += dev_prop.warpSize)
        {
        for (unsigned int group_size = 1; group_size <= overlaps_max_tpp; group_size *= 2)
            {
            for (unsigned int depletants_per_thread = 1; depletants_per_thread <= 32;
                 depletants_per_thread *= 2)
                {
                if ((block_size % group_size) == 0)
                    valid_params_depletants.push_back(block_size * 1000000
                                                      + depletants_per_thread * 10000 + group_size);
                }
            }
        }
    m_tuner_depletants.reset(new Autotuner(valid_params_depletants,
                                           5,
                                           100000,
                                           "clusters_depletants",
                                           this->m_exec_conf));

    // tuning parameters for nlist concatenation kernel
    std::vector<unsigned int> valid_params_concatenate;
    for (unsigned int block_size = dev_prop.warpSize;
         block_size <= (unsigned int)dev_prop.maxThreadsPerBlock;
         block_size += dev_prop.warpSize)
        {
        for (unsigned int group_size = 1; group_size <= overlaps_max_tpp; group_size *= 2)
            {
            if ((block_size % group_size) == 0)
                valid_params_concatenate.push_back(block_size * 10000 + group_size);
            }
        }
    m_tuner_concatenate.reset(new Autotuner(valid_params_concatenate,
                                            5,
                                            100000,
                                            "clusters_concatenate",
                                            this->m_exec_conf));

    GlobalArray<unsigned int> excell_size(0, this->m_exec_conf);
    m_excell_size.swap(excell_size);
    TAG_ALLOCATION(m_excell_size);

    GlobalArray<unsigned int> excell_idx(0, this->m_exec_conf);
    m_excell_idx.swap(excell_idx);
    TAG_ALLOCATION(m_excell_idx);

    // allocate memory for connected components
    GlobalArray<unsigned int>(1, this->m_exec_conf).swap(m_adjacency);
    TAG_ALLOCATION(m_adjacency);
    GlobalVector<int>(this->m_exec_conf).swap(m_components);
    TAG_ALLOCATION(m_components);

    GlobalVector<uint2>(this->m_exec_conf).swap(m_adjacency_copy);
    TAG_ALLOCATION(m_adjacency_copy);

    m_overlaps_streams.resize(this->m_exec_conf->getNumActiveGPUs());
    for (int idev = this->m_exec_conf->getNumActiveGPUs() - 1; idev >= 0; --idev)
        {
        hipSetDevice(this->m_exec_conf->getGPUIds()[idev]);
        hipStreamCreate(&m_overlaps_streams[idev]);
        }

    GlobalArray<unsigned int>(1, this->m_exec_conf).swap(m_nneigh);
    TAG_ALLOCATION(m_nneigh);

    GlobalArray<unsigned int>(1, this->m_exec_conf).swap(m_nneigh_scan);
    TAG_ALLOCATION(m_nneigh_scan);

    m_maxn = 0;
    GlobalArray<unsigned int>(1, this->m_exec_conf).swap(m_overflow);
    TAG_ALLOCATION(m_overflow);

        {
        ArrayHandle<unsigned int> h_overflow(m_overflow,
                                             access_location::host,
                                             access_mode::overwrite);
        *h_overflow.data = 0;
        }

    // Depletants
    const Index2D& depletant_idx = this->m_mc->getDepletantIndexer();
    unsigned int ntypes = this->m_pdata->getNTypes();
    GlobalArray<Scalar> lambda(ntypes * depletant_idx.getNumElements(), this->m_exec_conf);
    m_lambda.swap(lambda);
    TAG_ALLOCATION(m_lambda);

    GlobalArray<unsigned int>(1, this->m_exec_conf).swap(m_n_depletants);
    TAG_ALLOCATION(m_n_depletants);

    m_depletant_streams.resize(depletant_idx.getNumElements());
    for (unsigned int itype = 0; itype < this->m_pdata->getNTypes(); ++itype)
        {
        for (unsigned int jtype = 0; jtype < this->m_pdata->getNTypes(); ++jtype)
            {
            m_depletant_streams[depletant_idx(itype, jtype)].resize(
                this->m_exec_conf->getNumActiveGPUs());
            for (int idev = this->m_exec_conf->getNumActiveGPUs() - 1; idev >= 0; --idev)
                {
                hipSetDevice(this->m_exec_conf->getGPUIds()[idev]);
                hipStreamCreate(&m_depletant_streams[depletant_idx(itype, jtype)][idev]);
                }
            }
        }
    }

template<class Shape> UpdaterClustersGPU<Shape>::~UpdaterClustersGPU()
    {
    this->m_exec_conf->msg->notice(5) << "Destroying UpdaterClustersGPU" << std::endl;

    for (auto s : m_depletant_streams)
        {
        for (int idev = this->m_exec_conf->getNumActiveGPUs() - 1; idev >= 0; --idev)
            {
            hipSetDevice(this->m_exec_conf->getGPUIds()[idev]);
            hipStreamDestroy(s[idev]);
            }
        }

    for (int idev = this->m_exec_conf->getNumActiveGPUs() - 1; idev >= 0; --idev)
        {
        hipSetDevice(this->m_exec_conf->getGPUIds()[idev]);
        hipStreamDestroy(m_overlaps_streams[idev]);
        }
    }

/*! Perform a cluster move
    \param timestep Current time step of the simulation
*/
template<class Shape> void UpdaterClustersGPU<Shape>::update(uint64_t timestep)
    {
    Updater::update(timestep);
    // compute nominal cell width
    auto& params = this->m_mc->getParams();
    Scalar nominal_width = this->m_mc->getMaxCoreDiameter();
    Scalar max_d(0.0);
    for (unsigned int type_i = 0; type_i < this->m_pdata->getNTypes(); ++type_i)
        {
        for (unsigned int type_j = 0; type_j < this->m_pdata->getNTypes(); ++type_j)
            {
            quat<Scalar> o;
            Shape tmp_i(o, params[type_i]);
            if (this->m_mc->getDepletantFugacity(type_i, type_j) != Scalar(0.0))
                {
                // add range of depletion interaction
                Shape tmp_j(o, params[type_j]);
                max_d = std::max(
                    max_d,
                    (Scalar)0.5
                        * (tmp_i.getCircumsphereDiameter() + tmp_j.getCircumsphereDiameter()));
                }
            }
        }
    nominal_width += max_d;
    if (this->m_cl->getNominalWidth() != nominal_width)
        this->m_cl->setNominalWidth(nominal_width);

    // update the cell list before re-initializing
    this->m_cl->compute(timestep);

    m_old_gpu_partition = this->m_pdata->getGPUPartition();

    // reinitialize poisson means array

    // this is slightly inelegant because it involves a sync / page migration, maybe this data
    // could be cached or computed inside a kernel?
    ArrayHandle<Scalar> h_lambda(m_lambda, access_location::host, access_mode::overwrite);

    for (unsigned int i_type = 0; i_type < this->m_pdata->getNTypes(); ++i_type)
        {
        Shape shape_i(quat<Scalar>(), params[i_type]);
        OverlapReal d_i(shape_i.getCircumsphereDiameter());

        for (unsigned int j_type = 0; j_type < this->m_pdata->getNTypes(); ++j_type)
            {
            Shape shape_j(quat<Scalar>(), params[j_type]);
            OverlapReal d_j(shape_j.getCircumsphereDiameter());

            // we use the larger of the two diameters for insertion
            OverlapReal range = std::max(d_i, d_j);

            for (unsigned int k_type = 0; k_type < this->m_pdata->getNTypes(); ++k_type)
                {
                // parameter for Poisson distribution
                Shape shape_k(quat<Scalar>(), params[k_type]);

                // get OBB and extend by depletant radius
                detail::OBB obb = shape_k.getOBB(vec3<Scalar>(0, 0, 0));
                obb.lengths.x += 0.5f * range;
                obb.lengths.y += 0.5f * range;
                if (this->m_sysdef->getNDimensions() == 3)
                    obb.lengths.z += 0.5f * range;
                else
                    obb.lengths.z = 0.5f; // unit length

                Scalar lambda = std::abs(this->m_mc->getDepletantFugacity(i_type, j_type)
                                         * obb.getVolume(this->m_sysdef->getNDimensions()));
                h_lambda.data[k_type * this->m_mc->getDepletantIndexer().getNumElements()
                              + this->m_mc->getDepletantIndexer()(i_type, j_type)]
                    = lambda;
                }
            }
        }

    // perform the update
    UpdaterClusters<Shape>::update(timestep);
    }

template<class Shape> void UpdaterClustersGPU<Shape>::connectedComponents()
    {
    if (this->m_prof)
        this->m_prof->push(this->m_exec_conf, "connected components");

    // this will contain the number of strongly connected components
    unsigned int num_components = 0;

    m_components.resize(this->m_pdata->getN());

    // access edges of adajacency matrix
    ArrayHandle<unsigned int> d_adjacency(m_adjacency, access_location::device, access_mode::read);
    ArrayHandle<unsigned int> d_nneigh(m_nneigh, access_location::host, access_mode::read);
    ArrayHandle<unsigned int> d_nneigh_scan(m_nneigh_scan,
                                            access_location::host,
                                            access_mode::overwrite);

    // determine total size of adjacency list, and do prefix sum
    unsigned int nneigh_total;
    this->m_exec_conf->beginMultiGPU();
    gpu::get_num_neighbors(d_nneigh.data,
                           d_nneigh_scan.data,
                           nneigh_total,
                           this->m_pdata->getGPUPartition(),
                           this->m_exec_conf->getCachedAllocatorManaged());
    if (this->m_exec_conf->isCUDAErrorCheckingEnabled())
        CHECK_CUDA_ERROR();
    this->m_exec_conf->endMultiGPU();

    // allocate memory for contiguous adjacency list, factor of two for symmetry
    m_adjacency_copy.resize(nneigh_total * 2);

    ArrayHandle<uint2> d_adjacency_copy(m_adjacency_copy,
                                        access_location::device,
                                        access_mode::overwrite);

    // concatenate per-particle neighbor lists into a single one
    this->m_exec_conf->beginMultiGPU();
    unsigned int param = m_tuner_concatenate->getParam();
    unsigned int block_size = param / 10000;
    unsigned int group_size = param % 10000;
    m_tuner_concatenate->begin();
    gpu::concatenate_adjacency_list(d_adjacency.data,
                                    d_nneigh.data,
                                    d_nneigh_scan.data,
                                    m_maxn,
                                    d_adjacency_copy.data,
                                    this->m_pdata->getGPUPartition(),
                                    block_size,
                                    group_size);
    if (this->m_exec_conf->isCUDAErrorCheckingEnabled())
        CHECK_CUDA_ERROR();
    m_tuner_concatenate->end();
    this->m_exec_conf->endMultiGPU();

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

    if (this->m_prof)
        this->m_prof->pop(this->m_exec_conf);
    }

template<class Shape> void UpdaterClustersGPU<Shape>::initializeExcellMem()
    {
    this->m_exec_conf->msg->notice(4) << "hpmc resizing expanded cells" << std::endl;

    // get the current cell dimensions
    unsigned int num_cells = this->m_cl->getCellIndexer().getNumElements();
    unsigned int num_adj = this->m_cl->getCellAdjIndexer().getW();
    unsigned int n_cell_list
        = this->m_cl->getPerDevice() ? this->m_exec_conf->getNumActiveGPUs() : 1;
    unsigned int num_max = this->m_cl->getNmax() * n_cell_list;

    // make the excell dimensions the same, but with room for Nmax*Nadj in each cell
    m_excell_list_indexer = Index2D(num_max * num_adj, num_cells);

    // reallocate memory
    m_excell_idx.resize(m_excell_list_indexer.getNumElements());
    m_excell_size.resize(num_cells);

#if defined(__HIP_PLATFORM_NVCC__) \
    && 0 // excell is currently not multi-GPU optimized, let the CUDA driver figure this out
    if (this->m_exec_conf->allConcurrentManagedAccess())
        {
        // set memory hints
        auto gpu_map = this->m_exec_conf->getGPUIds();
        for (unsigned int idev = 0; idev < this->m_exec_conf->getNumActiveGPUs(); ++idev)
            {
            cudaMemAdvise(m_excell_idx.get(),
                          sizeof(unsigned int) * m_excell_idx.getNumElements(),
                          cudaMemAdviseSetAccessedBy,
                          gpu_map[idev]);
            cudaMemAdvise(m_excell_size.get(),
                          sizeof(unsigned int) * m_excell_size.getNumElements(),
                          cudaMemAdviseSetAccessedBy,
                          gpu_map[idev]);
            CHECK_CUDA_ERROR();
            }
        }
#endif
    }

template<class Shape> void UpdaterClustersGPU<Shape>::backupState()
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
        this->m_exec_conf->beginMultiGPU();
        for (int idev = this->m_exec_conf->getNumActiveGPUs() - 1; idev >= 0; --idev)
            {
            hipSetDevice(this->m_exec_conf->getGPUIds()[idev]);

            auto range = this->m_pdata->getGPUPartition().getRange(idev);
            if (range.second - range.first != 0)
                {
                hipMemcpyAsync(d_postype_backup.data + range.first,
                               d_postype.data + range.first,
                               sizeof(Scalar4) * (range.second - range.first),
                               hipMemcpyDeviceToDevice);
                hipMemcpyAsync(d_orientation_backup.data + range.first,
                               d_orientation.data + range.first,
                               sizeof(Scalar4) * (range.second - range.first),
                               hipMemcpyDeviceToDevice);
                hipMemcpyAsync(d_image_backup.data + range.first,
                               d_image.data + range.first,
                               sizeof(int3) * (range.second - range.first),
                               hipMemcpyDeviceToDevice);
                }
            if (this->m_exec_conf->isCUDAErrorCheckingEnabled())
                CHECK_CUDA_ERROR();
            }
        this->m_exec_conf->endMultiGPU();
        }
    }

template<class Shape>
void UpdaterClustersGPU<Shape>::transform(const quat<Scalar>& q,
                                          const vec3<Scalar>& pivot,
                                          bool line)
    {
    if (this->m_prof)
        this->m_prof->push(this->m_exec_conf, "Transform");

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
    unsigned int block_size = m_tuner_transform->getParam();
    gpu::clusters_transform_args_t args(d_postype.data,
                                        d_orientation.data,
                                        d_image.data,
                                        pivot,
                                        q,
                                        line,
                                        this->m_pdata->getGPUPartition(),
                                        this->m_pdata->getGlobalBox(),
                                        this->m_pdata->getNTypes(),
                                        block_size);

    this->m_exec_conf->beginMultiGPU();
    m_tuner_transform->begin();
    gpu::transform_particles<Shape>(args, params.data());
    if (this->m_exec_conf->isCUDAErrorCheckingEnabled())
        CHECK_CUDA_ERROR();
    m_tuner_transform->end();
    this->m_exec_conf->endMultiGPU();

    if (this->m_prof)
        this->m_prof->pop(this->m_exec_conf);
    }

template<class Shape> void UpdaterClustersGPU<Shape>::flip(uint64_t timestep)
    {
    if (this->m_prof)
        this->m_prof->push(this->m_exec_conf, "flip");

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

    this->m_exec_conf->beginMultiGPU();
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
                       this->m_pdata->getGPUPartition(),
                       m_tuner_flip->getParam());
    if (this->m_exec_conf->isCUDAErrorCheckingEnabled())
        CHECK_CUDA_ERROR();
    m_tuner_flip->end();
    this->m_exec_conf->endMultiGPU();

    if (this->m_prof)
        this->m_prof->pop(this->m_exec_conf);
    }

template<class Shape>
void UpdaterClustersGPU<Shape>::findInteractions(uint64_t timestep,
                                                 const quat<Scalar> q,
                                                 const vec3<Scalar> pivot,
                                                 bool line)
    {
    const auto& params = this->m_mc->getParams();

    // start the profile
    if (this->m_prof)
        this->m_prof->push(this->m_exec_conf, "Interactions");

    if (this->m_pdata->getN() > 0)
        {
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

        // per-device cell list data
        const ArrayHandle<unsigned int>& d_cell_size_per_device
            = m_cl->getPerDevice() ? ArrayHandle<unsigned int>(m_cl->getCellSizeArrayPerDevice(),
                                                               access_location::device,
                                                               access_mode::read)
                                   : ArrayHandle<unsigned int>(GlobalArray<unsigned int>(),
                                                               access_location::device,
                                                               access_mode::read);
        const ArrayHandle<unsigned int>& d_cell_idx_per_device
            = m_cl->getPerDevice() ? ArrayHandle<unsigned int>(m_cl->getIndexArrayPerDevice(),
                                                               access_location::device,
                                                               access_mode::read)
                                   : ArrayHandle<unsigned int>(GlobalArray<unsigned int>(),
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
                         m_cl->getPerDevice() ? d_cell_idx_per_device.data : d_cell_idx.data,
                         m_cl->getPerDevice() ? d_cell_size_per_device.data : d_cell_size.data,
                         d_cell_adj.data,
                         this->m_cl->getCellIndexer(),
                         this->m_cl->getCellListIndexer(),
                         this->m_cl->getCellAdjIndexer(),
                         this->m_exec_conf->getNumActiveGPUs(),
                         this->m_tuner_excell_block_size->getParam());
        if (this->m_exec_conf->isCUDAErrorCheckingEnabled())
            CHECK_CUDA_ERROR();
        this->m_tuner_excell_block_size->end();

        bool reallocate = false;

        // allocate memory for number of neighbors
        size_t old_size = m_nneigh.size();
        m_nneigh.resize(this->m_pdata->getN());
        m_nneigh_scan.resize(this->m_pdata->getN());

        if (m_nneigh.size() != old_size)
            {
            // update memory hints
            updateGPUAdvice();
            }

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

                // depletants
                auto depletant_idx = this->m_mc->getDepletantIndexer();
                m_n_depletants.resize(this->m_pdata->getN() * depletant_idx.getNumElements()
                                      * this->m_pdata->getMaxN());

                ArrayHandle<Scalar> d_lambda(m_lambda, access_location::device, access_mode::read);
                ArrayHandle<unsigned int> d_n_depletants(m_n_depletants,
                                                         access_location::device,
                                                         access_mode::overwrite);

                // depletant parameters
                ArrayHandle<Scalar> h_fugacity(this->m_mc->getFugacityArray(),
                                               access_location::host,
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
                                         m_old_gpu_partition,
                                         &m_overlaps_streams.front());

                this->m_exec_conf->beginMultiGPU();

                // reset number of neighbors
                for (int idev = this->m_exec_conf->getNumActiveGPUs() - 1; idev >= 0; --idev)
                    {
                    hipSetDevice(this->m_exec_conf->getGPUIds()[idev]);

                    auto range = this->m_pdata->getGPUPartition().getRange(idev);
                    if (range.second - range.first != 0)
                        hipMemsetAsync(d_nneigh.data + range.first,
                                       0,
                                       sizeof(unsigned int) * (range.second - range.first));
                    if (this->m_exec_conf->isCUDAErrorCheckingEnabled())
                        CHECK_CUDA_ERROR();
                    }

                /*
                 *  check overlaps, new configuration simultaneously against old configuration
                 */

                m_tuner_overlaps->begin();
                unsigned int param = m_tuner_overlaps->getParam();
                args.block_size = param / 1000000;
                args.tpp = (param % 1000000) / 100;
                args.overlap_threads = param % 100;
                gpu::hpmc_cluster_overlaps<Shape>(args, params.data());
                if (this->m_exec_conf->isCUDAErrorCheckingEnabled())
                    CHECK_CUDA_ERROR();
                m_tuner_overlaps->end();

                /*
                 * Insert depletants
                 */

                // allow concurrency between depletant types in multi GPU block
                for (unsigned int itype = 0; itype < this->m_pdata->getNTypes(); ++itype)
                    {
                    for (unsigned int jtype = itype; jtype < this->m_pdata->getNTypes(); ++jtype)
                        {
                        if (h_fugacity.data[depletant_idx(itype, jtype)] == 0)
                            continue;

                        if (itype != jtype)
                            throw std::runtime_error(
                                "Non-additive depletants are not supported by UpdaterClustersGPU.");

                        if (h_fugacity.data[depletant_idx(itype, jtype)] < 0)
                            throw std::runtime_error(
                                "Negative fugacities are not supported by UpdaterClustersGPU.");

                        // draw random number of depletant insertions per particle from Poisson
                        // distribution
                        m_tuner_num_depletants->begin();
                        gpu::generate_num_depletants(
                            this->m_sysdef->getSeed(),
                            timestep,
                            0, // is select really always 0 here?
                            this->m_exec_conf->getRank(),
                            itype,
                            jtype,
                            depletant_idx,
                            d_lambda.data,
                            d_postype.data,
                            d_n_depletants.data
                                + depletant_idx(itype, jtype) * this->m_pdata->getMaxN(),
                            m_tuner_num_depletants->getParam(),
                            &m_depletant_streams[depletant_idx(itype, jtype)].front(),
                            m_old_gpu_partition);
                        if (this->m_exec_conf->isCUDAErrorCheckingEnabled())
                            CHECK_CUDA_ERROR();
                        m_tuner_num_depletants->end();

                        // max reduce over result
                        unsigned int max_n_depletants[this->m_exec_conf->getNumActiveGPUs()];
                        gpu::get_max_num_depletants(
                            d_n_depletants.data
                                + depletant_idx(itype, jtype) * this->m_pdata->getMaxN(),
                            &max_n_depletants[0],
                            &m_depletant_streams[depletant_idx(itype, jtype)].front(),
                            m_old_gpu_partition,
                            this->m_exec_conf->getCachedAllocatorManaged());
                        if (this->m_exec_conf->isCUDAErrorCheckingEnabled())
                            CHECK_CUDA_ERROR();

                        // insert depletants on-the-fly
                        m_tuner_depletants->begin();
                        unsigned int param = m_tuner_depletants->getParam();
                        args.block_size = param / 1000000;
                        unsigned int depletants_per_thread = (param % 1000000) / 10000;
                        args.tpp = param % 10000;

                        gpu::hpmc_implicit_args_t implicit_args(
                            itype,
                            jtype,
                            depletant_idx,
                            0,     // implicit_counters
                            0,     // implicit_counters pitch
                            false, // repulsive
                            d_n_depletants.data
                                + depletant_idx(itype, jtype) * this->m_pdata->getMaxN(),
                            &max_n_depletants[0],
                            depletants_per_thread,
                            &m_depletant_streams[depletant_idx(itype, jtype)].front());
                        gpu::hpmc_clusters_depletants<Shape>(args, implicit_args, params.data());
                        if (this->m_exec_conf->isCUDAErrorCheckingEnabled())
                            CHECK_CUDA_ERROR();
                        m_tuner_depletants->end();
                        }
                    }
                this->m_exec_conf->endMultiGPU();
                } // end ArrayHandle scope

            reallocate = checkReallocate();
            } while (reallocate);
        }

    // start the profile
    if (this->m_prof)
        this->m_prof->pop(this->m_exec_conf);
    }

template<class Shape> bool UpdaterClustersGPU<Shape>::checkReallocate()
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

        GlobalArray<unsigned int> adjacency(req_size_nlist, this->m_exec_conf);
        m_adjacency.swap(adjacency);
        TAG_ALLOCATION(m_adjacency);

#ifdef __HIP_PLATFORM_NVCC__
        // update memory hints
        if (this->m_exec_conf->allConcurrentManagedAccess())
            {
            // set memory hints
            auto gpu_map = this->m_exec_conf->getGPUIds();
            for (unsigned int idev = 0; idev < this->m_exec_conf->getNumActiveGPUs(); ++idev)
                {
                auto range = this->m_pdata->getGPUPartition().getRange(idev);

                unsigned int nelem = range.second - range.first;
                if (nelem == 0)
                    continue;

                cudaMemAdvise(m_adjacency.get() + range.first * m_maxn,
                              sizeof(unsigned int) * nelem * m_maxn,
                              cudaMemAdviseSetPreferredLocation,
                              gpu_map[idev]);
                cudaMemPrefetchAsync(m_adjacency.get() + range.first * m_maxn,
                                     sizeof(unsigned int) * nelem * m_maxn,
                                     gpu_map[idev]);
                CHECK_CUDA_ERROR();
                }
            }
#endif
        }
    return reallocate || maxn_changed;
    }

template<class Shape> void UpdaterClustersGPU<Shape>::updateGPUAdvice()
    {
#ifdef __HIP_PLATFORM_NVCC__
    // update memory hints
    if (this->m_exec_conf->allConcurrentManagedAccess())
        {
        // set memory hints
        auto gpu_map = this->m_exec_conf->getGPUIds();
        for (unsigned int idev = 0; idev < this->m_exec_conf->getNumActiveGPUs(); ++idev)
            {
            auto range = m_old_gpu_partition.getRange(idev);

            unsigned int nelem = range.second - range.first;
            if (nelem == 0)
                continue;

            cudaMemAdvise(this->m_postype_backup.get() + range.first,
                          sizeof(Scalar4) * nelem,
                          cudaMemAdviseSetPreferredLocation,
                          gpu_map[idev]);
            cudaMemPrefetchAsync(this->m_postype_backup.get() + range.first,
                                 sizeof(Scalar4) * nelem,
                                 gpu_map[idev]);

            cudaMemAdvise(this->m_orientation_backup.get() + range.first,
                          sizeof(Scalar4) * nelem,
                          cudaMemAdviseSetPreferredLocation,
                          gpu_map[idev]);
            cudaMemPrefetchAsync(this->m_orientation_backup.get() + range.first,
                                 sizeof(Scalar4) * nelem,
                                 gpu_map[idev]);

            cudaMemAdvise(this->m_nneigh.get() + range.first,
                          sizeof(unsigned int) * nelem,
                          cudaMemAdviseSetPreferredLocation,
                          gpu_map[idev]);
            cudaMemPrefetchAsync(this->m_nneigh.get() + range.first,
                                 sizeof(unsigned int) * nelem,
                                 gpu_map[idev]);

            cudaMemAdvise(this->m_nneigh_scan.get() + range.first,
                          sizeof(unsigned int) * nelem,
                          cudaMemAdviseSetPreferredLocation,
                          gpu_map[idev]);
            cudaMemPrefetchAsync(this->m_nneigh_scan.get() + range.first,
                                 sizeof(unsigned int) * nelem,
                                 gpu_map[idev]);
            }
        }
#endif
    }

namespace detail
    {
template<class Shape> void export_UpdaterClustersGPU(pybind11::module& m, const std::string& name)
    {
    pybind11::class_<UpdaterClustersGPU<Shape>,
                     UpdaterClusters<Shape>,
                     std::shared_ptr<UpdaterClustersGPU<Shape>>>(m, name.c_str())
        .def(pybind11::init<std::shared_ptr<SystemDefinition>,
                            std::shared_ptr<IntegratorHPMCMono<Shape>>,
                            std::shared_ptr<CellList>>());
    }

    } // end namespace detail
    } // end namespace hpmc
    } // end namespace hoomd

#endif // ENABLE_CUDA
#endif // _UPDATER_HPMC_CLUSTERS_GPU_
