// inclusion guard
#ifndef _UPDATER_HPMC_CLUSTERS_GPU_
#define _UPDATER_HPMC_CLUSTERS_GPU_

/*! \file UpdaterBoxClusters.h
    \brief Declaration of UpdaterBoxClusters
*/

#ifdef ENABLE_HIP

#include "UpdaterClusters.h"
#include "UpdaterClustersGPU.cuh"

#include <cuda_runtime.h>

namespace hpmc
{

/*!
   Implementation of UpdaterClusters on the GPU
*/

template< class Shape >
class UpdaterClustersGPU : public UpdaterClusters<Shape>
    {
    public:
        //! Constructor
        /*! \param sysdef System definition
            \param mc HPMC integrator
            \param seed PRNG seed
        */
        UpdaterClustersGPU(std::shared_ptr<SystemDefinition> sysdef,
                        std::shared_ptr<IntegratorHPMCMono<Shape> > mc,
                        unsigned int seed);

        //! Destructor
        virtual ~UpdaterClustersGPU();

        //! Set autotuner parameters
        /*! \param enable Enable/disable autotuning
            \param period period (approximate) in time steps when returning occurs
        */
        virtual void setAutotunerParams(bool enable, unsigned int period)
            {
            }


    protected:
        GlobalVector<uint2> m_adjacency;     //!< List of overlaps between old and new configuration
        GlobalVector<int> m_components;      //!< The connected component labels per particle

        //! Determine connected components of the interaction graph
        #ifdef ENABLE_TBB
        virtual void connectedComponents(unsigned int N, std::vector<tbb::concurrent_vector<unsigned int> >& clusters);
        #else
        virtual void connectedComponents(unsigned int N, std::vector<std::vector<unsigned int> >& clusters);
        #endif
    };

template< class Shape >
UpdaterClustersGPU<Shape>::UpdaterClustersGPU(std::shared_ptr<SystemDefinition> sysdef,
                             std::shared_ptr<IntegratorHPMCMono<Shape> > mc,
                             unsigned int seed)
    : UpdaterClusters<Shape>(sysdef, mc, seed)
    {
    this->m_exec_conf->msg->notice(5) << "Constructing UpdaterClustersGPU" << std::endl;

    // allocate memory for connected components
    GlobalVector<uint2>(this->m_exec_conf).swap(m_adjacency);
    GlobalVector<int>(this->m_exec_conf).swap(m_components);
    }

template< class Shape >
UpdaterClustersGPU<Shape>::~UpdaterClustersGPU()
    {
    this->m_exec_conf->msg->notice(5) << "Destroying UpdaterClustersGPU" << std::endl;
    }


template<class Shape>
#ifdef ENABLE_TBB
void UpdaterClustersGPU<Shape>::connectedComponents(unsigned int N, std::vector<tbb::concurrent_vector<unsigned int> >& clusters)
#else
void UpdaterClustersGPU<Shape>::connectedComponents(unsigned int N, std::vector<std::vector<unsigned int> >& clusters)
#endif
    {
    // collect interactions on rank 0
    bool master = !this->m_exec_conf->getRank();

    // copy overlaps into GPU array
    #ifdef ENABLE_MPI
    if (this->m_comm)
        {
        // combine lists from different ranks
        #ifdef ENABLE_TBB
        std::vector< tbb::concurrent_unordered_set<std::pair<unsigned int, unsigned int> > > all_overlaps;
        #else
        std::vector< std::set<std::pair<unsigned int, unsigned int> > > all_overlaps;
        #endif

        gather_v(this->m_overlap, all_overlaps, 0, this->m_exec_conf->getMPICommunicator());

        if (master)
            {
            #ifdef ENABLE_MPI
            // determine new size for overlaps list
            unsigned int n_overlaps = 0;
            for (auto it = all_overlaps.begin(); it != all_overlaps.end(); ++it)
                n_overlaps += it->size();

            // resize local adjacency list
            m_adjacency.resize(n_overlaps);

                {
                ArrayHandle<uint2> h_adjacency(m_adjacency, access_location::host, access_mode::overwrite);

                // collect adjacency matrix
                unsigned int offs = 0;
                for (auto it = all_overlaps.begin(); it != all_overlaps.end(); ++it)
                    {
                    for (auto p : *it)
                        {
                        h_adjacency.data[offs++] = make_uint2(p.first, p.second);
                        h_adjacency.data[offs++] = make_uint2(p.second, p.first);
                        }
                    }
                }
            }
        #endif
        }
    else
    #endif
        {
        // resize local adjacency list
        m_adjacency.resize(this->m_overlap.size());

        ArrayHandle<uint2> h_adjacency(m_adjacency, access_location::host, access_mode::overwrite);

        // collect adjacency matrix
        unsigned int offs = 0;
        for (auto p : this->m_overlap)
            {
            // undirected edge
            h_adjacency.data[offs++] = make_uint2(p.first, p.second);
            }
        }

    if (this->m_prof)
        this->m_prof->push(this->m_exec_conf, "connected components");

    if (master)
        {
        m_components.resize(N);

        // access edges of adajacency matrix
        ArrayHandle<uint2> d_adjacency(m_adjacency, access_location::device, access_mode::read);

        // this will contain the number of strongly connected components
        unsigned int num_components = 0;

            {
            // access the output array
            ArrayHandle<int> d_components(m_components, access_location::device, access_mode::overwrite);

            detail::gpu_connected_components(
                d_adjacency.data,
                N,
                m_adjacency.size(),
                d_components.data,
                num_components,
                this->m_exec_conf->dev_prop,
                this->m_exec_conf->getCachedAllocator());

            if (this->m_exec_conf->isCUDAErrorCheckingEnabled())
                CHECK_CUDA_ERROR();
            }

        clusters.clear();
        clusters.resize(num_components);

        // copy back to host
        ArrayHandle<int> h_components(m_components, access_location::host, access_mode::read);

        for (unsigned int i = 0; i < N; ++i)
            {
            clusters[h_components.data[i]].push_back(i);
            }
        }

    if (this->m_prof)
        this->m_prof->pop(this->m_exec_conf);
    }


template <class Shape>
void export_UpdaterClustersGPU(pybind11::module& m, const std::string& name)
    {
    pybind11::class_< UpdaterClustersGPU<Shape>, UpdaterClusters<Shape>,
        std::shared_ptr< UpdaterClustersGPU<Shape> > >(m, name.c_str())
        .def( pybind11::init< std::shared_ptr<SystemDefinition>,
                         std::shared_ptr< IntegratorHPMCMono<Shape> >,
                         unsigned int >())
    ;
    }

} // end namespace hpmc

#endif // ENABLE_CUDA
#endif // _UPDATER_HPMC_CLUSTERS_GPU_
