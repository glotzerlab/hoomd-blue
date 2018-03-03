// Copyright (c) 2009-2017 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.


// Maintainers: jglaser, pschwende

/*! \file GlobalPUArray.h
    \brief Defines the GlobalArray class
*/


#include "ManagedArray.h"
#include "GPUArray.h"

template<class T>
class GlobalArray : public GPUArray<T>
    {
    public:
        //! Empty constructor
        GlobalArray()
            : data(nullptr)
            { }

        GlobalArray(unsigned int num_elements, std::shared_ptr<const ExecutionConfiguration> exec_conf)
            {
            m_array = ManagedArray<T>(num_elements, exec_conf->isCUDAEnabled());
            data = m_array.get();
            }

        T *get()
            {
            return data;
            }

    protected:
        ManagedArray<T> m_array;
        T *data;

        inline T* aquire(const access_location::Enum location, const access_mode::Enum mode
        #ifdef ENABLE_CUDA
                         , bool async = false
        #endif
                        ) const
            {
            return data;
            }
    };
