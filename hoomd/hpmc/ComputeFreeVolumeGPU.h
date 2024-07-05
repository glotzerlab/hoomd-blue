// Copyright (c) 2009-2024 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

#ifndef __COMPUTE_FREE_VOLUME_GPU_H__
#define __COMPUTE_FREE_VOLUME_GPU_H__

#ifdef ENABLE_HIP

#include "hoomd/Autotuner.h"
#include "hoomd/CellList.h"
#include "hoomd/Compute.h"

#include "ComputeFreeVolume.h"
#include "ComputeFreeVolumeGPU.cuh"
#include "IntegratorHPMCMono.h"
#include "IntegratorHPMCMonoGPU.cuh"

/*! \file ComputeFreeVolumeGPU.h
    \brief Defines the template class for an approximate free volume integration
    \note This header cannot be compiled by nvcc
*/

#ifdef __HIPCC__
#error This header cannot be compiled by nvcc
#endif

#include <pybind11/pybind11.h>

namespace hoomd
    {
namespace hpmc
    {
//! Template class for a free volume integration analyzer
/*!
    \ingroup hpmc_integrators
*/
template<class Shape> class ComputeFreeVolumeGPU : public ComputeFreeVolume<Shape>
    {
    public:
    //! Construct the integrator
    ComputeFreeVolumeGPU(std::shared_ptr<SystemDefinition> sysdef,
                         std::shared_ptr<IntegratorHPMCMono<Shape>> mc,
                         std::shared_ptr<CellList> cl);
    //! Destructor
    virtual ~ComputeFreeVolumeGPU();

    //! Return an estimate of the overlap volume
    virtual void computeFreeVolume(uint64_t timestep);

    protected:
    uint3 m_last_dim;         //!< Dimensions of the cell list on the last call to update
    unsigned int m_last_nmax; //!< Last cell list NMax value allocated in excell

    GPUArray<unsigned int> m_excell_idx;  //!< Particle indices in expanded cells
    GPUArray<unsigned int> m_excell_size; //!< Number of particles in each expanded cell
    Index2D m_excell_list_indexer;        //!< Indexer to access elements of the excell_idx list

    /// Autotuner for the overlap/free volume counter
    std::shared_ptr<Autotuner<3>> m_tuner_free_volume;

    /// Autotuner for excell block_size
    std::shared_ptr<Autotuner<1>> m_tuner_excell_block_size;

    void initializeExcellMem();
    };

template<class Shape>
ComputeFreeVolumeGPU<Shape>::ComputeFreeVolumeGPU(std::shared_ptr<SystemDefinition> sysdef,
                                                  std::shared_ptr<IntegratorHPMCMono<Shape>> mc,
                                                  std::shared_ptr<CellList> cl)
    : ComputeFreeVolume<Shape>(sysdef, mc, cl)
    {
    // Autotuner parameters:
    // 0: block size
    // 1: stride
    // 2: group size

    // Only widen the parallelism if the shape supports it, and limit parallelism to fit within the
    // warp.
    std::function<bool(const std::array<unsigned int, 3>&)> is_parameter_valid
        = [](const std::array<unsigned int, 3>& parameter) -> bool
    {
        unsigned int block_size = parameter[0];
        unsigned int stride = parameter[1];
        unsigned int group_size = parameter[2];
        return (stride == 1 || Shape::isParallel()) && (stride * group_size <= block_size)
               && (block_size % (stride * group_size)) == 0;
    };

    m_tuner_free_volume.reset(
        new Autotuner<3>({AutotunerBase::makeBlockSizeRange(this->m_exec_conf),
                          AutotunerBase::getTppListPow2(this->m_exec_conf),
                          AutotunerBase::getTppListPow2(this->m_exec_conf)},
                         this->m_exec_conf,
                         "hpmc_free_volume",
                         3,
                         false,
                         is_parameter_valid));
    GPUArray<unsigned int> excell_size(0, this->m_exec_conf);
    m_excell_size.swap(excell_size);

    GPUArray<unsigned int> excell_idx(0, this->m_exec_conf);
    m_excell_idx.swap(excell_idx);

    // set last dim to a bogus value so that it will re-init on the first call
    m_last_dim = make_uint3(0xffffffff, 0xffffffff, 0xffffffff);
    m_last_nmax = 0xffffffff;

    m_tuner_excell_block_size.reset(
        new Autotuner<1>({AutotunerBase::makeBlockSizeRange(this->m_exec_conf)},
                         this->m_exec_conf,
                         "hpmc_free_volume_excell_block_size"));

    this->m_autotuners.insert(this->m_autotuners.end(),
                              {m_tuner_free_volume, m_tuner_excell_block_size});
    }

template<class Shape> ComputeFreeVolumeGPU<Shape>::~ComputeFreeVolumeGPU() { }

/*! \return the current free volume (by MC integration)
 */
template<class Shape> void ComputeFreeVolumeGPU<Shape>::computeFreeVolume(uint64_t timestep)
    {
    this->m_exec_conf->msg->notice(5) << "HPMC computing free volume " << timestep << std::endl;

    // set nominal width
    Scalar nominal_width = this->m_mc->getMaxCoreDiameter();

    if (this->m_cl->getNominalWidth() != nominal_width)
        this->m_cl->setNominalWidth(nominal_width);

    const BoxDim box = this->m_pdata->getBox();
    Scalar3 npd = box.getNearestPlaneDistance();

    if ((box.getPeriodic().x && npd.x <= nominal_width * 2)
        || (box.getPeriodic().y && npd.y <= nominal_width * 2)
        || (this->m_sysdef->getNDimensions() == 3 && box.getPeriodic().z
            && npd.z <= nominal_width * 2))
        {
        throw std::runtime_error("Simulation box too small");
        }

    // compute cell list
    this->m_cl->compute(timestep);

    // if the cell list is a different size than last time, reinitialize expanded cell list
    uint3 cur_dim = this->m_cl->getDim();
    if (this->m_last_dim.x != cur_dim.x || this->m_last_dim.y != cur_dim.y
        || this->m_last_dim.z != cur_dim.z || this->m_last_nmax != this->m_cl->getNmax())
        {
        this->initializeExcellMem();
        m_last_dim = cur_dim;
        m_last_nmax = this->m_cl->getNmax();
        }

    // access the cell list data
    ArrayHandle<unsigned int> d_cell_size(this->m_cl->getCellSizeArray(),
                                          access_location::device,
                                          access_mode::read);
    ArrayHandle<Scalar4> d_cell_xyzf(this->m_cl->getXYZFArray(),
                                     access_location::device,
                                     access_mode::read);
    ArrayHandle<Scalar4> d_cell_orientation(this->m_cl->getOrientationArray(),
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
        = this->m_cl->getPerDevice()
              ? ArrayHandle<unsigned int>(this->m_cl->getCellSizeArrayPerDevice(),
                                          access_location::device,
                                          access_mode::read)
              : ArrayHandle<unsigned int>(GlobalArray<unsigned int>(),
                                          access_location::device,
                                          access_mode::read);
    const ArrayHandle<unsigned int>& d_cell_idx_per_device
        = this->m_cl->getPerDevice()
              ? ArrayHandle<unsigned int>(this->m_cl->getIndexArrayPerDevice(),
                                          access_location::device,
                                          access_mode::read)
              : ArrayHandle<unsigned int>(GlobalArray<unsigned int>(),
                                          access_location::device,
                                          access_mode::read);

    ArrayHandle<unsigned int> d_excell_idx(this->m_excell_idx,
                                           access_location::device,
                                           access_mode::readwrite);
    ArrayHandle<unsigned int> d_excell_size(this->m_excell_size,
                                            access_location::device,
                                            access_mode::readwrite);

    // update the expanded cells
    this->m_tuner_excell_block_size->begin();
    gpu::hpmc_excell(d_excell_idx.data,
                     d_excell_size.data,
                     m_excell_list_indexer,
                     this->m_cl->getPerDevice() ? d_cell_idx_per_device.data : d_cell_idx.data,
                     this->m_cl->getPerDevice() ? d_cell_size_per_device.data : d_cell_size.data,
                     d_cell_adj.data,
                     this->m_cl->getCellIndexer(),
                     this->m_cl->getCellListIndexer(),
                     this->m_cl->getCellAdjIndexer(),
                     this->m_cl->getPerDevice() ? this->m_exec_conf->getNumActiveGPUs() : 1,
                     this->m_tuner_excell_block_size->getParam()[0]);
    if (this->m_exec_conf->isCUDAErrorCheckingEnabled())
        CHECK_CUDA_ERROR();
    this->m_tuner_excell_block_size->end();

    // access the particle data
    ArrayHandle<Scalar4> d_postype(this->m_pdata->getPositions(),
                                   access_location::device,
                                   access_mode::readwrite);
    ArrayHandle<Scalar4> d_orientation(this->m_pdata->getOrientationArray(),
                                       access_location::device,
                                       access_mode::readwrite);
    ArrayHandle<unsigned int> d_overlaps(this->m_mc->getInteractionMatrix(),
                                         access_location::device,
                                         access_mode::read);

    const Index2D& overlap_idx = this->m_mc->getOverlapIndexer();

    // access the parameters
    auto& params = this->m_mc->getParams();

        {
        // access counter
        ArrayHandle<unsigned int> d_n_overlap_all(this->m_n_overlap_all,
                                                  access_location::device,
                                                  access_mode::overwrite);

        m_tuner_free_volume->begin();
        auto param = m_tuner_free_volume->getParam();
        unsigned int block_size = param[0];
        unsigned int stride = param[1];
        unsigned int group_size = param[2];

        unsigned int n_sample = this->m_n_sample;

#ifdef ENABLE_MPI
        n_sample /= this->m_exec_conf->getNRanks();
#endif

        detail::hpmc_free_volume_args_t free_volume_args(n_sample,
                                                         this->m_type,
                                                         d_postype.data,
                                                         d_orientation.data,
                                                         d_cell_idx.data,
                                                         d_cell_size.data,
                                                         this->m_cl->getCellIndexer(),
                                                         this->m_cl->getCellListIndexer(),
                                                         d_excell_idx.data,
                                                         d_excell_size.data,
                                                         this->m_excell_list_indexer,
                                                         this->m_cl->getDim(),
                                                         this->m_pdata->getN(),
                                                         this->m_pdata->getNTypes(),
                                                         this->m_sysdef->getSeed(),
                                                         this->m_exec_conf->getRank(),
                                                         0,
                                                         timestep,
                                                         this->m_sysdef->getNDimensions(),
                                                         box,
                                                         block_size,
                                                         stride,
                                                         group_size,
                                                         this->m_pdata->getMaxN(),
                                                         d_n_overlap_all.data,
                                                         this->m_cl->getGhostWidth(),
                                                         d_overlaps.data,
                                                         overlap_idx,
                                                         this->m_exec_conf->dev_prop);

        // invoke kernel for counting total overlap volume
        detail::gpu_hpmc_free_volume<Shape>(free_volume_args, params.data());

        if (this->m_exec_conf->isCUDAErrorCheckingEnabled())
            CHECK_CUDA_ERROR();

        m_tuner_free_volume->end();
        }

#ifdef ENABLE_MPI
    if (this->m_sysdef->isDomainDecomposed())
        {
        ArrayHandle<unsigned int> h_n_overlap_all(this->m_n_overlap_all,
                                                  access_location::host,
                                                  access_mode::readwrite);
        MPI_Allreduce(MPI_IN_PLACE,
                      h_n_overlap_all.data,
                      1,
                      MPI_UNSIGNED,
                      MPI_SUM,
                      this->m_exec_conf->getMPICommunicator());
        }
#endif
    }

template<class Shape> void ComputeFreeVolumeGPU<Shape>::initializeExcellMem()
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

namespace detail
    {
//! Export this hpmc analyzer to python
/*! \param name Name of the class in the exported python module
    \tparam Shape An instantiation of IntegratorHPMCMono<Shape> will be exported
*/
template<class Shape> void export_ComputeFreeVolumeGPU(pybind11::module& m, const std::string& name)
    {
    pybind11::class_<ComputeFreeVolumeGPU<Shape>,
                     ComputeFreeVolume<Shape>,
                     std::shared_ptr<ComputeFreeVolumeGPU<Shape>>>(m, name.c_str())
        .def(pybind11::init<std::shared_ptr<SystemDefinition>,
                            std::shared_ptr<IntegratorHPMCMono<Shape>>,
                            std::shared_ptr<CellList>>());
    }

    } // end namespace detail
    } // end namespace hpmc

    } // end namespace hoomd

#endif // ENABLE_HIP

#endif // __COMPUTE_FREE_VOLUME_GPU_H__
