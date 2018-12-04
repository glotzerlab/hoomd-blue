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
            : m_gpu_map(gpu_id)
            {
            m_gpu_range.resize(gpu_id.size());

            // reset to defaults
            for (unsigned int i = 0; i < gpu_id.size(); ++i)
                m_gpu_range[i] = std::make_pair(0,0);
            }

        //! Update the number of particles and distribute particles equally between GPUs
        void setN(unsigned int N)
            {
            unsigned int n_per_gpu = N/m_gpu_map.size();
            unsigned int offset = 0;
            for (unsigned int i = 0; i < m_gpu_map.size(); ++i)
                {
                m_gpu_range[i] = std::make_pair(offset, offset+n_per_gpu);
                offset += n_per_gpu;
                }

            // fill last GPU with remaining particles
            m_gpu_range[m_gpu_map.size()-1].second = N;
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

            return m_gpu_range[igpu];
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

        std::vector<std::pair<unsigned int, unsigned int> > m_gpu_range;
    };

#endif
