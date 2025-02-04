// Copyright (c) 2009-2024 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

#pragma once

#ifdef ENABLE_HIP

#include <algorithm>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <hip/hip_runtime.h>

namespace hoomd
    {
//! A thin data structure to hold the split of particles across GPUs
/* We intentionally do not use STL containers such as std::vector<> here, as these are known to
   cause problems when passed across shared library boundaries, such as to a GPU driver function.
 */

class __attribute__((visibility("default"))) GPUPartition
    {
    public:
    // Empty constructor
    GPUPartition() : m_n_gpu(0), m_gpu_map(nullptr), m_gpu_range(nullptr) { }

    //! Constructor
    /*! \param gpu_id Mapping of contiguous device IDs onto CUDA devices
     */
    GPUPartition(const std::vector<unsigned int>& gpu_id)
        {
        m_gpu_range = nullptr;
        m_gpu_map = nullptr;
        m_n_gpu = (unsigned int)gpu_id.size();
        if (m_n_gpu != 0)
            {
            m_gpu_map = new unsigned int[gpu_id.size()];
            for (unsigned int i = 0; i < m_n_gpu; ++i)
                m_gpu_map[i] = gpu_id[i];
            m_gpu_range = new std::pair<unsigned int, unsigned int>[gpu_id.size()];

            // reset to defaults
            for (unsigned int i = 0; i < gpu_id.size(); ++i)
                m_gpu_range[i] = std::make_pair(0, 0);
            }
        }

    //! Destructor
    virtual ~GPUPartition()
        {
        if (m_n_gpu)
            {
            delete[] m_gpu_map;
            delete[] m_gpu_range;
            }
        }

    //! Copy constructor
    GPUPartition(const GPUPartition& other)
        {
        m_gpu_range = nullptr;
        m_gpu_map = nullptr;
        m_n_gpu = other.m_n_gpu;

        if (m_n_gpu != 0)
            {
            m_gpu_range = new std::pair<unsigned int, unsigned int>[m_n_gpu];
            m_gpu_map = new unsigned int[m_n_gpu];
            }
        for (unsigned int i = 0; i < m_n_gpu; ++i)
            {
            m_gpu_range[i] = other.m_gpu_range[i];
            m_gpu_map[i] = other.m_gpu_map[i];
            }
        }

    //! Copy assignment operator
    GPUPartition& operator=(const GPUPartition& rhs)
        {
        if (&rhs != this)
            {
            m_n_gpu = rhs.m_n_gpu;
            if (m_n_gpu != 0)
                {
                m_gpu_range = new std::pair<unsigned int, unsigned int>[m_n_gpu];
                m_gpu_map = new unsigned int[m_n_gpu];
                }
            else
                {
                m_gpu_range = nullptr;
                m_gpu_map = nullptr;
                }
            for (unsigned int i = 0; i < m_n_gpu; ++i)
                {
                m_gpu_range[i] = rhs.m_gpu_range[i];
                m_gpu_map[i] = rhs.m_gpu_map[i];
                }
            }
        return *this;
        }

    //! Update the number of particles and distribute particles equally between GPUs
    void setN(unsigned int N, unsigned int offset = 0)
        {
        unsigned int n_per_gpu = N / m_n_gpu;
        unsigned int ini_offset = offset;
        for (unsigned int i = 0; i < m_n_gpu; ++i)
            {
            m_gpu_range[i] = std::make_pair(offset, offset + n_per_gpu);
            offset += n_per_gpu;
            }

        // fill last GPU with remaining particles
        m_gpu_range[m_n_gpu - 1].second = ini_offset + N;
        }

    //! Get the number of active GPUs
    inline unsigned int getNumActiveGPUs() const
        {
        return m_n_gpu;
        }

    //! Get the index range for a given GPU
    /*! \param igpu The logical ID of the GPU
     */
    inline std::pair<unsigned int, unsigned int> getRange(unsigned int igpu) const
        {
        if (igpu > m_n_gpu)
            throw std::runtime_error("GPU " + std::to_string(igpu)
                                     + " not in execution configuration");

        return m_gpu_range[igpu];
        }

    //! Get the index range for a given GPU
    /*! \param igpu The logical ID of the GPU
     */
    inline std::pair<unsigned int, unsigned int> getRangeAndSetGPU(unsigned int igpu) const
        {
        if (igpu > m_n_gpu)
            throw std::runtime_error("GPU " + std::to_string(igpu)
                                     + " not in execution configuration");

        unsigned int gpu_id = m_gpu_map[igpu];

        // set the active GPU
        hipSetDevice(gpu_id);

        return getRange(igpu);
        };

    private:
    unsigned int m_n_gpu;
    unsigned int* m_gpu_map;

    std::pair<unsigned int, unsigned int>* m_gpu_range;
    };

    } // end namespace hoomd

#endif
