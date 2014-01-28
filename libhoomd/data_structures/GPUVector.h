/*
Highly Optimized Object-oriented Many-particle Dynamics -- Blue Edition
(HOOMD-blue) Open Source Software License Copyright 2008-2011 Ames Laboratory
Iowa State University and The Regents of the University of Michigan All rights
reserved.

HOOMD-blue may contain modifications ("Contributions") provided, and to which
copyright is held, by various Contributors who have granted The Regents of the
University of Michigan the right to modify and/or distribute such Contributions.

You may redistribute, use, and create derivate works of HOOMD-blue, in source
and binary forms, provided you abide by the following conditions:

* Redistributions of source code must retain the above copyright notice, this
list of conditions, and the following disclaimer both in the code and
prominently in any materials provided with the distribution.

* Redistributions in binary form must reproduce the above copyright notice, this
list of conditions, and the following disclaimer in the documentation and/or
other materials provided with the distribution.

* All publications and presentations based on HOOMD-blue, including any reports
or published results obtained, in whole or in part, with HOOMD-blue, will
acknowledge its use according to the terms posted at the time of submission on:
http://codeblue.umich.edu/hoomd-blue/citations.html

* Any electronic documents citing HOOMD-Blue will link to the HOOMD-Blue website:
http://codeblue.umich.edu/hoomd-blue/

* Apart from the above required attributions, neither the name of the copyright
holder nor the names of HOOMD-blue's contributors may be used to endorse or
promote products derived from this software without specific prior written
permission.

Disclaimer

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDER AND CONTRIBUTORS ``AS IS'' AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, AND/OR ANY
WARRANTIES THAT THIS SOFTWARE IS FREE OF INFRINGEMENT ARE DISCLAIMED.

IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT,
INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE
OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF
ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

// Maintainer: jglaser

/*! \file GPUVector.h
    \brief Defines the GPUVector class
*/

#ifdef NVCC
#error This header cannot be compiled by nvcc
#endif

#ifndef __GPUVECTOR_H__
#define __GPUVECTOR_H__

#include "GPUArray.h"
#include <algorithm>

// The factor with which the array size is incremented
#define RESIZE_FACTOR 9.f/8.f

template<class T> class GPUArray;
//! Class for managing a vector of elements on the GPU mirrored to the CPU
/*! The GPUVector class is a simple container for a variable number of elements. Its interface is inspired
    by std::vector and it offers methods to insert new elements at the end of the list, and to remove
    them from there.

    It uses a GPUArray as the underlying storage class, thus the data in a GPUVector can also be accessed
    directly using ArrayHandles.

    In the current implementation, a GPUVector can only grow (but not shrink) in size until it is destroyed.

    \ingroup data_structs
*/
template<class T> class GPUVector : public GPUArray<T>
    {
    public:
        //! Default constructor
        GPUVector();

        //! Constructs an empty GPUVector
        GPUVector(boost::shared_ptr<const ExecutionConfiguration> exec_conf);

        //! Constructs a GPUVector
        GPUVector(unsigned int size, boost::shared_ptr<const ExecutionConfiguration> exec_conf);

    #ifdef ENABLE_CUDA
        //! Constructs an empty GPUVector
        GPUVector(boost::shared_ptr<const ExecutionConfiguration> exec_conf, bool mapped);

        //! Constructs a GPUVector
        GPUVector(unsigned int size, boost::shared_ptr<const ExecutionConfiguration> exec_conf, bool mapped);
    #endif

        //! Frees memory
        virtual ~GPUVector() {}

        //! Copy constructor
        GPUVector(const GPUVector& from);
        //! = operator
        GPUVector& operator=(const GPUVector& rhs);

        //! swap this GPUVector with another
        inline void swap(GPUVector &from);

        /*!
          \returns the current size of the vector
        */
        unsigned int size() const
            {
            return m_size;
            }

        //! Resize the GPUVector
        /*! \param new_size New number of elements
        */
        virtual void resize(unsigned int new_size);

        //! Resizing a vector as a matrix is not supported
        /*!
        */
        virtual void resize(unsigned int width, unsigned int height)
            {
            this->m_exec_conf->msg->error() << "Cannot change a GPUVector into a matrix." << std::endl;
            throw std::runtime_error("Error resizing GPUVector.");
            }

        //! Insert an element at the end of the vector
        /*! \param val The new element
         */
        virtual void push_back(const T& val);

        //! Remove an element from the end of the list
        virtual void pop_back();

        //! Remove an element by index
        virtual void erase(unsigned int i);

        //! Clear the list
        virtual void clear();

        //! Proxy class to provide access to the data elements of the vector
        class data_proxy
            {
            public:
                //! Constructor
                data_proxy(const GPUVector<T> & _vec, const unsigned int _n)
                    : vec(_vec), n(_n) { }

                //! Type cast
                operator T() const
                    {
                    T *data  = vec.acquireHost(access_mode::read);
                    T val = data[n];
                    vec.release();
                    return val;
                    }

                //! Assignment
                data_proxy& operator= (T rhs)
                    {
                    T *data  = vec.acquireHost(access_mode::readwrite);
                    data[n] = rhs;
                    vec.release();
                    return *this;
                    }

            private:
                const GPUVector<T>& vec; //!< The vector that is accessed
                unsigned int n;          //!< The index of the element to access
            };

        //! Get a proxy-reference to a list element
        data_proxy operator [] (unsigned int n)
            {
            assert(n < m_size);
            return data_proxy(*this, n);
            }

        //! Get a proxy-reference to a list element (const version)
        data_proxy operator [] (unsigned int n) const
            {
            assert(n < m_size);
            return data_proxy(*this, n);
            }


    private:
        unsigned int m_size;                    //!< Number of elements

        //! Helper function to reallocate the GPUArray (using amortized array resizing)
        void reallocate(unsigned int new_size);

        //! Acquire the underlying GPU array on the host
        T *acquireHost(const access_mode::Enum mode) const;

        friend class data_proxy;
    };


//******************************************
// GPUVector implementation
// *****************************************

//! Default constructor
/*! \warning When using this constructor, a properly initialized GPUVector with an exec_conf needs
             to be swapped in later, after construction of the GPUVector.
 */
template<class T> GPUVector<T>::GPUVector()
    : GPUArray<T>(), m_size(0)
    {
    }

/*! \param exec_conf Shared pointer to the execution configuration
 */
template<class T> GPUVector<T>::GPUVector(boost::shared_ptr<const ExecutionConfiguration> exec_conf)
    : GPUArray<T>(0,exec_conf), m_size(0)
    {
    }

/*! \param size Number of elements to allocate initial memory for in the array
    \param exec_conf Shared pointer to the execution configuration
*/
template<class T> GPUVector<T>::GPUVector(unsigned int size, boost::shared_ptr<const ExecutionConfiguration> exec_conf)
     : GPUArray<T>(size, exec_conf), m_size(size)
    {
    }

#ifdef ENABLE_CUDA
//! Constructs an empty GPUVector
/*! \param exec_conf Shared pointer to the execution configuration
 *  \param mapped True if using mapped-pinned memory
 */
template<class T> GPUVector<T>::GPUVector(boost::shared_ptr<const ExecutionConfiguration> exec_conf, bool mapped)
    : GPUArray<T>(0,exec_conf, mapped), m_size(0)
    {
    }

/*! \param size Number of elements to allocate initial memory for in the array
    \param exec_conf Shared pointer to the execution configuration
    \param mapped True if using mapped-pinned memory
*/
template<class T> GPUVector<T>::GPUVector(unsigned int size, boost::shared_ptr<const ExecutionConfiguration> exec_conf, bool mapped)
     : GPUArray<T>(size, exec_conf, mapped), m_size(size)
    {
    }
#endif

template<class T> GPUVector<T>::GPUVector(const GPUVector& from) : GPUArray<T>(from), m_size(from.m_size)
    {
    }

template<class T> GPUVector<T>& GPUVector<T>::operator=(const GPUVector& rhs)
    {
    if (this != &rhs) // protect against invalid self-assignment
        {
        m_size = rhs.m_size;
        // invoke base class operator
        (GPUArray<T>) *this = rhs;
        }

    return *this;
    }

/*! \param from GPUVector to swap \a this with
*/
template<class T> void GPUVector<T>::swap(GPUVector<T>& from)
    {
    std::swap(m_size, from.m_size);
    GPUArray<T>::swap(from);
    }

/*! \param size New requested size of allocated memory
 *
 * Internally, this method uses amortized resizing of allocated memory to
 * avoid excessive copying of data. The GPUArray is only reallocated if necessary,
 * i.e. if the requested size is larger than the current size, which is a power of two.
 */
template<class T> void GPUVector<T>::reallocate(unsigned int size)
    {
    if (size > GPUArray<T>::getNumElements())
        {
        // reallocate
        unsigned int new_allocated_size = GPUArray<T>::getNumElements() ? GPUArray<T>::getNumElements() : 1;

        // double the size as often as necessary
        while (size > new_allocated_size)
            new_allocated_size = ((unsigned int) (((float) new_allocated_size) * RESIZE_FACTOR)) + 1 ;

        // actually resize the underlying GPUArray
        GPUArray<T>::resize(new_allocated_size);
        }
    }

/*! \param new_size New size of vector
 \post The GPUVector will be re-allocated if necessary to hold the new elements.
       The newly allocated memory is \b not initialized. It is responsbility of the caller to ensure correct initialiation,
       e.g. using clear()
*/
template<class T> void GPUVector<T>::resize(unsigned int new_size)
    {
    // allocate memory by amortized O(N) resizing
    if (new_size > 0)
        reallocate(new_size);
    else
        // for zero size, we at least allocate the memory
        reallocate(1);

    // set new size
    m_size = new_size;
    }


//! Insert an element at the end of the vector
template<class T> void GPUVector<T>::push_back(const T& val)
    {
    reallocate(m_size+1);

    T *data = acquireHost(access_mode::readwrite);
    data[m_size++] = val;
    GPUArray<T>::release();
    }

//! Remove an element from the end of the list
template<class T> void GPUVector<T>::pop_back()
    {
    assert(m_size);
    m_size--;
    }

//! Remove an element in the middle
template<class T> void GPUVector<T>::erase(unsigned int i)
    {
    assert(i < m_size);
    T *data = acquireHost(access_mode::readwrite);
    T *res = data;
    for (unsigned int n = 0; n < m_size; ++n)
        {
        if (n != i)
            {
            *res = *data;
            res++;
            }
        data++;
        }
    m_size--;
    GPUArray<T>::release();
    }

//! Clear the list
template<class T> void GPUVector<T>::clear()
    {
    m_size = 0;
    }

/*! \param mode Access mode for the GPUArray
 */
template<class T> T * GPUVector<T>::acquireHost(const access_mode::Enum mode) const
    {
    #ifdef ENABLE_CUDA
    return GPUArray<T>::aquire(access_location::host, access_mode::readwrite, false);
    #else
    return GPUArray<T>::aquire(access_location::host, access_mode::readwrite);
    #endif
    }

#endif
