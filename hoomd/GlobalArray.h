// Copyright (c) 2009-2017 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.


// Maintainers: jglaser, pschwende

/*! \file GlobalPUArray.h
    \brief Defines the GlobalArray class
*/

#pragma once
#include "ManagedArray.h"
#include "GPUArray.h"
#include <type_traits>

template<class T>
class GlobalArray : public GPUArray<T>
    {
    public:
        //! Empty constructor
        GlobalArray()
            : m_pitch(0), m_height(0)
            { }

        /*! Allocate a 1D array in managed memory
            \param num_elements Number of elements in array
            \param exec_conf The current execution configuration
         */
        GlobalArray(unsigned int num_elements, std::shared_ptr<const ExecutionConfiguration> exec_conf)
            : m_pitch(num_elements), m_height(1), m_exec_conf(exec_conf)
            {
            m_array = ManagedArray<T>(num_elements, exec_conf->isCUDAEnabled());
            }

        /*! Allocate a 2D array in managed memory
            \param width Width of the 2-D array to allocate (in elements)
            \param height Number of rows to allocate in the 2D array
            \param exec_conf Shared pointer to the execution configuration for managing CUDA initialization and shutdown
         */
        GlobalArray(unsigned int width, unsigned int height, std::shared_ptr<const ExecutionConfiguration> exec_conf) :
            m_height(height), m_exec_conf(exec_conf)
            {
            // make m_pitch the next multiple of 16 larger or equal to the given width
            m_pitch = (width + (16 - (width & 15)));

            unsigned int num_elements = m_pitch * m_height;
            m_array = ManagedArray<T>(num_elements, exec_conf->isCUDAEnabled());
            }


        //! Swap the pointers of two GlobalArrays
        inline void swap(GlobalArray &from)
            {
            std::swap(m_pitch,from.m_pitch);
            std::swap(m_height,from.m_height);
            std::swap(m_array, from.m_array);
            std::swap(m_exec_conf, from.m_exec_conf);
            }

        //! Swap the pointers of two equally sized GPUArrays
        inline void swap(GPUArray<T>& from)
            {
            throw std::runtime_error("GlobalArray::swap() not supported with GPUArray()");
            }

        //! Swap the pointers of two equally sized GPUArrays
        // inline void swap(GPUArray& from) const;

        T *get() const
            {
            return m_array.get();
            }

        //! Get the number of elements
        /*!
         - For 1-D allocated GPUArrays, this is the number of elements allocated.
         - For 2-D allocated GPUArrays, this is the \b total number of elements (\a pitch * \a height) allocated
        */
        virtual unsigned int getNumElements() const
            {
            return m_array.size() > 0;
            }

        //! Test if the GPUArray is NULL
        virtual bool isNull() const
            {
            return m_array.size() > 0;
            }

        //! Get the width of the allocated rows in elements
        /*!
         - For 2-D allocated GPUArrays, this is the total width of a row in memory (including the padding added for coalescing)
         - For 1-D allocated GPUArrays, this is the simply the number of elements allocated.
        */
        virtual unsigned int getPitch() const
            {
            return m_pitch;
            }

        //! Get the number of rows allocated
        /*!
         - For 2-D allocated GPUArrays, this is the height given to the constructor
         - For 1-D allocated GPUArrays, this is the simply 1.
        */
        virtual unsigned int getHeight() const
            {
            return m_height;
            }

        //! Resize the GlobalArray
        /*! This method resizes the array by allocating a new array and copying over the elements
            from the old array. Resizing is a slow operation.
        */
        virtual void resize(unsigned int num_elements)
            {
            ManagedArray<T> new_array(num_elements, m_array.isManaged());

            #ifdef ENABLE_CUDA
            if (m_exec_conf->isCUDAEnabled())
                {
                // synchronize all active GPUs
                auto gpu_map = m_exec_conf->getGPUIds();
                for (int idev = m_exec_conf->getNumActiveGPUs() - 1; idev >= 0; --idev)
                    {
                    cudaSetDevice(gpu_map[idev]);
                    cudaDeviceSynchronize();
                    }
                }
            #endif

            std :: copy(m_array.get(), m_array.get() + m_array.size(), new_array.get());
            m_array = new_array;
            m_pitch = m_array.size();
            m_height = 1;
            }

        //! Resize a 2D GlobalArray
        virtual void resize(unsigned int width, unsigned int height)
            {
            // make m_pitch the next multiple of 16 larger or equal to the given width
            unsigned int pitch = (width + (16 - (width & 15)));

            unsigned int num_elements = pitch * height;
            assert(num_elements > 0);

            ManagedArray<T> new_array(num_elements, m_array.isManaged());

            #ifdef ENABLE_CUDA
            if (m_exec_conf->isCUDAEnabled())
                {
                // synchronize all active GPUs
                auto gpu_map = m_exec_conf->getGPUIds();
                for (int idev = m_exec_conf->getNumActiveGPUs() - 1; idev >= 0; --idev)
                    {
                    cudaSetDevice(gpu_map[idev]);
                    cudaDeviceSynchronize();
                    }
                }
            #endif

            // copy over data
            // every column is copied separately such as to align with the new pitch
            unsigned int num_copy_rows = m_height > height ? height : m_height;
            unsigned int num_copy_columns = m_pitch > pitch ? pitch : m_pitch;
            for (unsigned int i = 0; i < num_copy_rows; i++)
                std::copy(m_array.get() + i*m_pitch, m_array.get() + i*m_pitch + num_copy_columns, new_array.get() + i * pitch);

            m_height = height;
            m_pitch  = pitch;

            m_array = new_array;
            }

    protected:
        mutable ManagedArray<T> m_array; //!< Data storage in managed or host memory

        unsigned int m_pitch;  //!< Pitch of 2D array
        unsigned int m_height; //!< Height of 2D array

        std::shared_ptr<const ExecutionConfiguration> m_exec_conf; //!< Handle to the current execution configuration

        virtual inline T* acquire(const access_location::Enum location, const access_mode::Enum mode
        #ifdef ENABLE_CUDA
                         , bool async = false
        #endif
                        ) const
            {
            #ifdef ENABLE_CUDA
            if (m_exec_conf->isCUDAEnabled() && location == access_location::host)
                {
                // synchronize all active GPUs
                auto gpu_map = m_exec_conf->getGPUIds();
                for (int idev = m_exec_conf->getNumActiveGPUs() - 1; idev >= 0; --idev)
                    {
                    cudaSetDevice(gpu_map[idev]);
                    cudaDeviceSynchronize();
                    }
                }
            #endif

            return get();
            }
    };
