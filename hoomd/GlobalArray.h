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
            { }

        GlobalArray(unsigned int num_elements, std::shared_ptr<const ExecutionConfiguration> exec_conf)
            {
            m_array = ManagedArray<T>(num_elements, exec_conf->isCUDAEnabled());
            }

        //! Swap the pointers of two equally sized GPUArrays
        inline void swap(GlobalArray &from)
            {
            std :: swap(m_array, from.m_array);
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
            return getNumElements();
            }

        //! Get the number of rows allocated
        /*!
         - For 2-D allocated GPUArrays, this is the height given to the constructor
         - For 1-D allocated GPUArrays, this is the simply 1.
        */
        virtual unsigned int getHeight() const
            {
            return 1;
            }

        //! Resize the GPUArray
        /*! This method resizes the array by allocating a new array and copying over the elements
            from the old array. This is a slow process.
            Only data from the currently active memory location (gpu/cpu) is copied over to the resized
            memory area.
        */
        virtual void resize(unsigned int num_elements)
            {
            ManagedArray<T> new_array(num_elements, m_array.isManaged());
            std :: copy(m_array.get(), m_array.get() + m_array.size(), new_array.get());
            m_array = new_array;
            }

        //! Resize a 2D GPUArray
        virtual void resize(unsigned int width, unsigned int height)
            {
            throw std::runtime_error("2D Resize not implemented for GlobalArray");
            }

    protected:
        mutable ManagedArray<T> m_array;

        virtual inline T* acquire(const access_location::Enum location, const access_mode::Enum mode
        #ifdef ENABLE_CUDA
                         , bool async = false
        #endif
                        ) const
            {
            if (location == access_location::host)
                cudaDeviceSynchronize();

            return get();
            }
    };
