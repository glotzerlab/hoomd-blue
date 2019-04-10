#ifndef __POTENTIAL_PAIR_GPU_H__
#define __POTENTIAL_PAIR_GPU_H__

#ifdef ENABLE_CUDA

#include "IntegratorHPMCMono.h"
#include "IntegratorHPMCMonoGPU.cuh"
#include "hoomd/Autotuner.h"

#include <cuda_runtime.h>

/*! \file IntegratorHPMCMonoGPU.h
    \brief Defines the template class for HPMC on the GPU
    \note This header cannot be compiled by nvcc
*/

#ifdef NVCC
#error This header cannot be compiled by nvcc
#endif

#include <hoomd/extern/pybind/include/pybind11/pybind11.h>

namespace hpmc
{

//! Template class for HPMC update on the GPU
/*!
    \ingroup hpmc_integrators
*/
template< class Shape >
class IntegratorHPMCMonoGPU : public IntegratorHPMCMono<Shape>
    {
    public:
        //! Construct the integrator
        IntegratorHPMCMonoGPU(std::shared_ptr<SystemDefinition> sysdef,
                              std::shared_ptr<CellList> cl,
                              unsigned int seed);
        //! Destructor
        virtual ~IntegratorHPMCMonoGPU();

        //! Set autotuner parameters
        /*! \param enable Enable/disable autotuning
            \param period period (approximate) in time steps when returning occurs
        */
        virtual void setAutotunerParams(bool enable, unsigned int period)
            {
            unsigned int ndim = this->m_sysdef->getNDimensions();
            if (ndim == 3)
                m_tuner_update->setPeriod(period*this->m_nselect*8);
            else
                m_tuner_update->setPeriod(period*this->m_nselect*4);
            m_tuner_update->setEnabled(enable);

            m_tuner_excell_block_size->setPeriod(period);
            m_tuner_excell_block_size->setEnabled(enable);
            }

        //! Enable deterministic simulations
        virtual void setDeterministic(bool deterministic)
            {
            this->m_exec_conf->msg->notice(2) << "hpmc: Sorting cell list to enable deterministic simulations." << std::endl;
            m_cl->setSortCellList(deterministic);
            }

    protected:
        std::shared_ptr<CellList> m_cl;           //!< Cell list
        GPUArray<unsigned int> m_cell_sets;   //!< List of cells active during each subsweep
        Index2D m_cell_set_indexer;           //!< Indexer into the cell set array
        uint3 m_last_dim;                     //!< Dimensions of the cell list on the last call to update
        unsigned int m_last_nmax;             //!< Last cell list NMax value allocated in excell
        detail::UpdateOrder m_cell_set_order; //!< Update order for cell sets

        GPUArray<unsigned int> m_excell_idx;  //!< Particle indices in expanded cells
        GPUArray<unsigned int> m_excell_size; //!< Number of particles in each expanded cell
        Index2D m_excell_list_indexer;        //!< Indexer to access elements of the excell_idx list

        std::unique_ptr<Autotuner> m_tuner_update;             //!< Autotuner for the update step group and block sizes
        std::unique_ptr<Autotuner> m_tuner_excell_block_size;  //!< Autotuner for excell block_size

        cudaStream_t m_stream;                //!< CUDA stream for update kernel

        //! Take one timestep forward
        virtual void update(unsigned int timestep);

        //! Set up cell sets
        virtual void initializeCellSets();

        //! Set up excell_list
        virtual void initializeExcellMem();

        //! Set the nominal width appropriate for looped moves
        virtual void updateCellWidth();
    };

template< class Shape >
IntegratorHPMCMonoGPU< Shape >::IntegratorHPMCMonoGPU(std::shared_ptr<SystemDefinition> sysdef,
                                                                   std::shared_ptr<CellList> cl,
                                                                   unsigned int seed)
    : IntegratorHPMCMono<Shape>(sysdef, seed), m_cl(cl), m_cell_set_order(seed+this->m_exec_conf->getRank())
    {
    this->m_cl->setRadius(1);
    this->m_cl->setComputeTDB(false);
    this->m_cl->setFlagType();
    this->m_cl->setComputeIdx(true);

    // require that cell lists have an even number of cells along each direction
    this->m_cl->setMultiple(2);

    // set last dim to a bogus value so that it will re-init on the first call
    m_last_dim = make_uint3(0xffffffff, 0xffffffff, 0xffffffff);
    m_last_nmax = 0xffffffff;

    GPUArray<unsigned int> excell_size(0, this->m_exec_conf);
    m_excell_size.swap(excell_size);

    GPUArray<unsigned int> excell_idx(0, this->m_exec_conf);
    m_excell_idx.swap(excell_idx);

    // initialize the autotuners
    // the full block size, stride and group size matrix is searched,
    // encoded as block_size*1000000 + stride*100 + group_size.
    std::vector<unsigned int> valid_params;
    cudaDeviceProp dev_prop = this->m_exec_conf->dev_prop;
    if (Shape::isParallel())
        {
        for (unsigned int block_size =dev_prop.warpSize; block_size <= (unsigned int) dev_prop.maxThreadsPerBlock; block_size +=dev_prop.warpSize)
            {
            unsigned int s=1;
            while (s <= (unsigned int)dev_prop.warpSize)
                {
                unsigned int stride = 1;
                while (stride <= block_size)
                    {
                    // for parallel overlap checks, use 3d-layout where blockDim.z is limited
                    if (block_size % (s*stride) == 0 && block_size/(s*stride) <= (unsigned int) dev_prop.maxThreadsDim[2])
                        valid_params.push_back(block_size*1000000 + stride*100 + s);

                    // increment stride in powers of two
                    stride *= 2;
                    }
                s++;
                }
            }
        }
    else
        {
        // for serial overlap checks, force stride=1. And groups no longer need to evenly divide into warps: only into
        // blocks
        unsigned int stride = 1;

        for (unsigned int block_size = dev_prop.warpSize; block_size <= (unsigned int) dev_prop.maxThreadsPerBlock; block_size += dev_prop.warpSize)
            {
            for (unsigned int group_size=1; group_size <= (unsigned int)dev_prop.warpSize; group_size++)
                {
                if ((block_size % group_size) == 0)
                    valid_params.push_back(block_size*1000000 + stride*100 + group_size);
                }
            }
        }

    m_tuner_update.reset(new Autotuner(valid_params, 5, 1000000, "hpmc_update", this->m_exec_conf));
    m_tuner_excell_block_size.reset(new Autotuner(dev_prop.warpSize, dev_prop.maxThreadsPerBlock, dev_prop.warpSize, 5, 1000000, "hpmc_excell_block_size", this->m_exec_conf));

    // create a CUDA stream
    // streams are used to ensure memory coherency until concurrent host-gpu access is fully supported (such as for compute 6.x devices
    // with appropriate kernel drivers)
    cudaStreamCreate(&m_stream);
    CHECK_CUDA_ERROR();
    }

template< class Shape >
IntegratorHPMCMonoGPU< Shape>::~IntegratorHPMCMonoGPU()
    {
    cudaStreamDestroy(m_stream);
    CHECK_CUDA_ERROR();
    }

template< class Shape >
void IntegratorHPMCMonoGPU< Shape >::update(unsigned int timestep)
    {
    if (this->m_patch && !this->m_patch_log)
        {
        this->m_exec_conf->msg->error() << "GPU simulations with patches are unsupported." << std::endl;
        throw std::runtime_error("Error during HPMC integration\n");
        }

    IntegratorHPMC::update(timestep);
    // compute the width of the active region
    Scalar3 npd = this->m_pdata->getBox().getNearestPlaneDistance();
    Scalar3 ghost_fraction = this->m_nominal_width / npd;

    // check if we are below a minimum image convention box size
    // the minimum image convention comes from the global box, not the local one
    BoxDim box = this->m_pdata->getGlobalBox();
    Scalar3 nearest_plane_distance = box.getNearestPlaneDistance();

    if ((box.getPeriodic().x && nearest_plane_distance.x <= this->m_nominal_width*2) ||
        (box.getPeriodic().y && nearest_plane_distance.y <= this->m_nominal_width*2) ||
        (this->m_sysdef->getNDimensions() == 3 && box.getPeriodic().z && nearest_plane_distance.z <= this->m_nominal_width*2))
        {
        this->m_exec_conf->msg->error() << "Simulation box too small for GPU accelerated HPMC execution - increase it so the minimum image convention works" << std::endl;
        throw std::runtime_error("Error performing HPMC update");
        }

    // update the cell list
    this->m_cl->compute(timestep);

    // start the profile
    if (this->m_prof) this->m_prof->push(this->m_exec_conf, "HPMC");

    // rng for shuffle and grid shift
    hoomd::RandomGenerator rng(hoomd::RNGIdentifier::HPMCMonoShift, this->m_seed, timestep);

    // if the cell list is a different size than last time, reinitialize the cell sets list
    uint3 cur_dim = this->m_cl->getDim();
    if (m_last_dim.x != cur_dim.x || m_last_dim.y != cur_dim.y || m_last_dim.z != cur_dim.z)
        {
        initializeCellSets();
        initializeExcellMem();

        m_last_dim = cur_dim;
        m_last_nmax = this->m_cl->getNmax();

        // initialize the cell set update order
        this->m_cell_set_order.resize(m_cell_set_indexer.getH());
        }

    // if only NMax changed, only need to reallocate excell memory
    if (m_last_nmax != this->m_cl->getNmax())
        {
        initializeExcellMem();
        m_last_nmax = this->m_cl->getNmax();
        }

    // test if we are in domain decomposition mode
    bool domain_decomposition = false;
#ifdef ENABLE_MPI
    if (this->m_comm)
        domain_decomposition = true;
#endif

    {
    // access the particle data
    ArrayHandle<Scalar4> d_postype(this->m_pdata->getPositions(), access_location::device, access_mode::readwrite);
    ArrayHandle<Scalar4> d_orientation(this->m_pdata->getOrientationArray(), access_location::device, access_mode::readwrite);
    ArrayHandle<int3> d_image(this->m_pdata->getImages(), access_location::device, access_mode::readwrite);

    // access the cell list data
    ArrayHandle<unsigned int> d_cell_size(this->m_cl->getCellSizeArray(), access_location::device, access_mode::read);
    ArrayHandle<Scalar4> d_cell_xyzf(this->m_cl->getXYZFArray(), access_location::device, access_mode::read);
    ArrayHandle<unsigned int> d_cell_idx(this->m_cl->getIndexArray(), access_location::device, access_mode::read);
    ArrayHandle<unsigned int> d_cell_adj(this->m_cl->getCellAdjArray(), access_location::device, access_mode::read);

    ArrayHandle< unsigned int > d_cell_sets(m_cell_sets, access_location::device, access_mode::read);

    ArrayHandle< unsigned int > d_excell_idx(m_excell_idx, access_location::device, access_mode::readwrite);
    ArrayHandle< unsigned int > d_excell_size(m_excell_size, access_location::device, access_mode::readwrite);

    ArrayHandle<hpmc_counters_t> d_counters(this->m_count_total, access_location::device, access_mode::readwrite);

    // access the parameters and interaction matrix
    const std::vector<typename Shape::param_type, managed_allocator<typename Shape::param_type> > & params = this->getParams();

    ArrayHandle<unsigned int> d_overlaps(this->m_overlaps, access_location::device, access_mode::read);

    // access the move sizes by type
    ArrayHandle<Scalar> d_d(this->m_d, access_location::device, access_mode::read);
    ArrayHandle<Scalar> d_a(this->m_a, access_location::device, access_mode::read);

    BoxDim box = this->m_pdata->getBox();

    Scalar3 ghost_width = this->m_cl->getGhostWidth();

    // compute the ceiling of the average number of particles in each cell, accounting for ghost particles
    int ncells = cur_dim.x * cur_dim.y * cur_dim.z;
    int particles_per_cell = int(ceil(double(this->m_pdata->getN() + this->m_pdata->getNGhosts()) / double(ncells)));

    // update the expanded cells
    this->m_tuner_excell_block_size->begin();
    detail::gpu_hpmc_excell(d_excell_idx.data,
                            d_excell_size.data,
                            m_excell_list_indexer,
                            d_cell_idx.data,
                            d_cell_size.data,
                            d_cell_adj.data,
                            this->m_cl->getCellIndexer(),
                            this->m_cl->getCellListIndexer(),
                            this->m_cl->getCellAdjIndexer(),
                            this->m_tuner_excell_block_size->getParam());
    if (this->m_exec_conf->isCUDAErrorCheckingEnabled())
        CHECK_CUDA_ERROR();

    this->m_tuner_excell_block_size->end();

    // on the first iteration, shape parameters are updated
    bool first = true;

    // loop over cell sets in a shuffled order
    this->m_cell_set_order.shuffle(timestep);
    for (unsigned int i = 0; i < this->m_nselect * particles_per_cell; i++)
        {
        for (unsigned int j = 0; j < m_cell_set_indexer.getH(); j++)
            {
            unsigned cur_set = this->m_cell_set_order[j];
            this->m_tuner_update->begin();

            unsigned int param = m_tuner_update->getParam();
            unsigned int block_size = param / 1000000;
            unsigned int stride = (param % 1000000 ) / 100;
            unsigned int group_size = param % 100;

            detail::gpu_hpmc_update<Shape> (detail::hpmc_args_t(d_postype.data,
                                                                d_orientation.data,
                                                                d_counters.data,
                                                                d_cell_idx.data,
                                                                d_cell_size.data,
                                                                d_excell_idx.data,
                                                                d_excell_size.data,
                                                                this->m_cl->getCellIndexer(),
                                                                this->m_cl->getCellListIndexer(),
                                                                m_excell_list_indexer,
                                                                this->m_cl->getDim(),
                                                                ghost_width,
                                                                &d_cell_sets.data[m_cell_set_indexer(0,cur_set)],
                                                                m_cell_set_indexer.getW(),
                                                                this->m_pdata->getN(),
                                                                this->m_pdata->getNTypes(),
                                                                this->m_seed,
                                                                d_d.data,
                                                                d_a.data,
                                                                d_overlaps.data,
                                                                this->m_overlap_idx,
                                                                this->m_move_ratio,
                                                                timestep,
                                                                this->m_sysdef->getNDimensions(),
                                                                box,
                                                                this->m_exec_conf->getRank()*this->m_nselect*particles_per_cell + i,
                                                                ghost_fraction,
                                                                domain_decomposition,
                                                                block_size,
                                                                stride,
                                                                group_size,
                                                                this->m_hasOrientation,
                                                                this->m_pdata->getMaxN(),
                                                                this->m_exec_conf->dev_prop,
                                                                first,
                                                                m_stream),
                                            params.data());

            if (this->m_exec_conf->isCUDAErrorCheckingEnabled())
                CHECK_CUDA_ERROR();

            this->m_tuner_update->end();

            first = false;
            }
        }

    // shift particles
    Scalar3 shift = make_scalar3(0,0,0);
    hoomd::UniformDistribution<Scalar> uniform(-this->m_nominal_width/Scalar(2.0),this->m_nominal_width/Scalar(2.0));
    shift.x = uniform(rng);
    shift.y = uniform(rng);
    if (this->m_sysdef->getNDimensions() == 3)
        {
        shift.z = uniform(rng);
        }

    detail::gpu_hpmc_shift(d_postype.data,
                           d_image.data,
                           this->m_pdata->getN(),
                           box,
                           shift,
                           128);

    // update the particle data origin
    this->m_pdata->translateOrigin(shift);

    if (this->m_exec_conf->isCUDAErrorCheckingEnabled())
        CHECK_CUDA_ERROR();

    if (this->m_prof) this->m_prof->pop(this->m_exec_conf);
    }

    this->communicate(true);

    // all particle have been moved, the aabb tree is now invalid
    this->m_aabb_tree_invalid = true;
    }

template< class Shape >
void IntegratorHPMCMonoGPU< Shape >::initializeCellSets()
    {
    this->m_exec_conf->msg->notice(4) << "hpmc recomputing active cells" << std::endl;
    // "ghost cells" might contain active particles. So they must be included in the active cell sets
    // we should not run into a multiple issue since the base multiple is 2 and the ghost cells added are 2 in each
    // direction. Check just to be on the safe side

    // compute the number of cells in each set
    // every other cell is active along each direction, excluding ghost cells
    uint3 dim = this->m_cl->getDim();
    const Index3D& cell_indexer = this->m_cl->getCellIndexer();
    unsigned int n_active = dim.x / 2 * dim.y / 2;
    unsigned int n_sets = 4;

    if (this->m_sysdef->getNDimensions() == 3)
        {
        n_active *= dim.z / 2;
        n_sets = 8;
        }

    GPUArray< unsigned int > cell_sets(n_active, n_sets, this->m_exec_conf);
    m_cell_sets.swap(cell_sets);
    m_cell_set_indexer = Index2D(n_active, n_sets);

    // build a list of active cells
    ArrayHandle< unsigned int > h_cell_sets(m_cell_sets, access_location::host, access_mode::overwrite);

    // offsets for x and y based on the set index
    unsigned int ox[] = {0, 1, 0, 1, 0, 1, 0, 1};
    unsigned int oy[] = {0, 0, 1, 1, 0, 0, 1, 1};
    unsigned int oz[] = {0, 0, 0, 0, 1, 1, 1, 1};

    for (unsigned int cur_set = 0; cur_set < n_sets; cur_set++)
        {
        unsigned int active_idx = 0;
        // loop over all cells in the active region, using information from num_ghost cells to avoid adding ghost cells
        // to the active set
        for (int k = oz[cur_set]; k < int(dim.z); k+=2)
            for (int j = oy[cur_set]; j < int(dim.y); j+=2)
                for (int i = ox[cur_set]; i < int(dim.x); i+=2)
                    {
                    h_cell_sets.data[m_cell_set_indexer(active_idx, cur_set)] = cell_indexer(i,j,k);
                    active_idx++;
                    }
        }
    }

template< class Shape >
void IntegratorHPMCMonoGPU< Shape >::initializeExcellMem()
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

template< class Shape >
void IntegratorHPMCMonoGPU< Shape >::updateCellWidth()
    {
    // changing the cell width means that the particle shapes have changed, assume this invalidates the
    // image list and aabb tree
    this->m_image_list_valid = false;
    this->m_aabb_tree_invalid = true;

    this->m_nominal_width = this->getMaxCoreDiameter();
    this->m_cl->setNominalWidth(this->m_nominal_width);

    // attach the parameters to the kernel stream so that they are visible
    // when other kernels are called
    cudaStreamAttachMemAsync(m_stream, this->m_params.data(), 0, cudaMemAttachSingle);
    CHECK_CUDA_ERROR();
    #if (CUDART_VERSION >= 8000)
    cudaMemAdvise(this->m_params.data(), this->m_params.size()*sizeof(typename Shape::param_type), cudaMemAdviseSetReadMostly, 0);
    CHECK_CUDA_ERROR();
    #endif

    for (unsigned int i = 0; i < this->m_pdata->getNTypes(); ++i)
        {
        // attach nested memory regions
        this->m_params[i].attach_to_stream(m_stream);
        }
    }

//! Export this hpmc integrator to python
/*! \param name Name of the class in the exported python module
    \tparam Shape An instantiation of IntegratorHPMCMono<Shape> will be exported
*/
template < class Shape > void export_IntegratorHPMCMonoGPU(pybind11::module& m, const std::string& name)
    {
     pybind11::class_<IntegratorHPMCMonoGPU<Shape>, std::shared_ptr< IntegratorHPMCMonoGPU<Shape> > >(m, name.c_str(), pybind11::base< IntegratorHPMCMono<Shape> >())
              .def(pybind11::init< std::shared_ptr<SystemDefinition>, std::shared_ptr<CellList>, unsigned int >())
              ;
    }

} // end namespace hpmc

#endif // ENABLE_CUDA

#endif // __POTENTIAL_PAIR_GPU_H__
