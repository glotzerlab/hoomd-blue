#pragma once

#ifdef ENABLE_CUDA

#include <string>
#include <vector>
#include <stdexcept>
#include <algorithm>

#include <cuda_runtime.h>

class GPUPartition
    {
    public:
        // Empty constructor
        GPUPartition() {}

        //! Constructor
        /*! \param gpu_id Mapping of contiguous device IDs onto CUDA devices
         */
        GPUPartition(const std::vector<unsigned int>& gpu_id)
            : m_gpu_map(gpu_id), m_N(0)
            {
            m_gpu_range.resize(gpu_id.size());

            // reset to defaults
            for (unsigned int i = 0; i < gpu_id.size(); ++i)
                m_gpu_range[i] = std::make_pair(0,0);
            }

        //! Update the number of particles
        void setN(unsigned int N)
            {
            m_N = N;
            }

        //! Update the number of particles and distribute particles equally between GPUs
        void setNMax(unsigned int Nmax)
            {
            unsigned int n_per_gpu = Nmax/m_gpu_map.size();
            unsigned int offset = 0;
            for (unsigned int i = 0; i < m_gpu_map.size(); ++i)
                {
                m_gpu_range[i] = std::make_pair(offset, offset+n_per_gpu);
                offset += n_per_gpu;
                }

            // fill last GPU with remaining particles
            m_gpu_range[m_gpu_map.size()-1].second = Nmax;
            }

        //! Returns the current maximum number of particles
        unsigned int getNMax() const
            {
            return m_gpu_range[m_gpu_map.size()-1].second;
            }

        //! Get the number of active GPUs
        unsigned int getNumActiveGPUs() const
            {
            return m_gpu_map.size();
            }

        //! Get the index range for a given GPU
        /*! \param igpu The logical ID of the GPU
         */
        std::pair<unsigned int, unsigned int> getRange(unsigned int igpu) const
            {
            if (igpu > m_gpu_map.size())
                throw std::runtime_error("GPU "+std::to_string(igpu)+" not in execution configuration");

            unsigned int n_remaining = m_N;
            unsigned int offset = 0;
            std::pair<unsigned int, unsigned int> range;
            unsigned int ncur;
            for (unsigned int i = 0; i <= igpu; ++i)
                {
                unsigned int nelem = m_gpu_range[i].second-m_gpu_range[i].first;
                ncur = std::min(nelem,n_remaining);
                n_remaining -= ncur;

                if (i < igpu)
                    offset += ncur;
                }

            range.first = offset;
            range.second = offset+ncur;
            return range;
            }

        //! Get the index range for a given GPU
        /*! \param igpu The logical ID of the GPU
         */
        std::pair<unsigned int, unsigned int> getRangeAndSetGPU(unsigned int igpu) const
            {
            if (igpu > m_gpu_map.size())
                throw std::runtime_error("GPU "+std::to_string(igpu)+" not in execution configuration");

            unsigned int gpu_id = m_gpu_map[igpu];

            // set the active GPU
            cudaSetDevice(gpu_id);

            return getRange(igpu);
            };

    private:
        std::vector<unsigned int> m_gpu_map;

        unsigned int m_N; //!< Current number of particles

        std::vector<std::pair<unsigned int, unsigned int> > m_gpu_range;
    };

#endif
