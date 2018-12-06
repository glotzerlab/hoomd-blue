// Copyright (c) 2009-2018 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.


// Maintainer: jglaser

/*! \file GlobalVector.h
    \brief Defines the GlobalVector class
*/

#ifdef NVCC
#error This header cannot be compiled by nvcc
#endif

#ifndef __GLOBALVECTOR_H__
#define __GLOBALVECTOR_H__

#include "GlobalArray.h"
#include <algorithm>

// The factor with which the array size is incremented
#define RESIZE_FACTOR 9.f/8.f

//! Class for managing a vector of elements on the GPU, residing in managed memory
/*! The GlobalVector class is a simple container for a variable number of elements. Its interface is inspired
    by std::vector and it offers methods to insert new elements at the end of the list, and to remove
    them from there.

    It uses a GlobalArray as the underlying storage class, thus the data in a GlobalVector can also be accessed
    directly using ArrayHandles.

    In the current implementation, a GlobalVector can only grow (but not shrink) in size until it is destroyed.

    \ingroup data_structs
*/
template<class T> class GlobalVector : public GlobalArray<T>
    {
    public:
        //! Default constructor
        GlobalVector();

        //! Constructs an empty GlobalVector
        GlobalVector(std::shared_ptr<const ExecutionConfiguration> exec_conf, const std::string& tag = std::string());

        //! Constructs a GlobalVector
        GlobalVector(unsigned int size, std::shared_ptr<const ExecutionConfiguration> exec_conf, const std::string& tag = std::string());

        //! Frees memory
        virtual ~GlobalVector() {}

        //! Copy constructor
        GlobalVector(const GlobalVector& from);
        //! = operator
        GlobalVector& operator=(const GlobalVector& rhs);

        //! swap this GlobalVector with another
        inline void swap(GlobalVector &from);

        /*!
          \returns the current size of the vector
        */
        unsigned int size() const
            {
            return m_size;
            }

        //! Resize the GlobalVector
        /*! \param new_size New number of elements
        */
        virtual void resize(unsigned int new_size);

        //! Resizing a vector as a matrix is not supported
        /*!
        */
        virtual void resize(unsigned int width, unsigned int height)
            {
            this->m_exec_conf->msg->error() << "Cannot change a GlobalVector into a matrix." << std::endl;
            throw std::runtime_error("Error resizing GlobalVector.");
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
                data_proxy(const GlobalVector<T> & _vec, const unsigned int _n)
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
                const GlobalVector<T>& vec; //!< The vector that is accessed
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

        //! Helper function to reallocate the GlobalArray (using amortized array resizing)
        void reallocate(unsigned int new_size);

        //! Acquire the underlying GPU array on the host
        T *acquireHost(const access_mode::Enum mode) const;

        friend class data_proxy;
    };


//******************************************
// GlobalVector implementation
// *****************************************

//! Default constructor
/*! \warning When using this constructor, a properly initialized GlobalVector with an exec_conf needs
             to be swapped in later, after construction of the GlobalVector.
 */
template<class T> GlobalVector<T>::GlobalVector()
    : GlobalArray<T>(), m_size(0)
    {
    }

/*! \param exec_conf Shared pointer to the execution configuration
 */
template<class T> GlobalVector<T>::GlobalVector(std::shared_ptr<const ExecutionConfiguration> exec_conf,
    const std::string& tag)
    : GlobalArray<T>(0,exec_conf,tag), m_size(0)
    {
    }

/*! \param size Number of elements to allocate initial memory for in the array
    \param exec_conf Shared pointer to the execution configuration
*/
template<class T> GlobalVector<T>::GlobalVector(unsigned int size, std::shared_ptr<const ExecutionConfiguration> exec_conf,
    const std::string& tag)
     : GlobalArray<T>(size, exec_conf,tag), m_size(size)
    {
    }

template<class T> GlobalVector<T>::GlobalVector(const GlobalVector& from) : GlobalArray<T>(from), m_size(from.m_size)
    {
    }

template<class T> GlobalVector<T>& GlobalVector<T>::operator=(const GlobalVector& rhs)
    {
    if (this != &rhs) // protect against invalid self-assignment
        {
        m_size = rhs.m_size;
        // invoke base class operator
        (GlobalArray<T>) *this = rhs;
        }

    return *this;
    }

/*! \param from GlobalVector to swap \a this with
*/
template<class T> void GlobalVector<T>::swap(GlobalVector<T>& from)
    {
    std::swap(m_size, from.m_size);
    GlobalArray<T>::swap(from);
    }

/*! \param size New requested size of allocated memory
 *
 * Internally, this method uses amortized resizing of allocated memory to
 * avoid excessive copying of data. The GlobalArray is only reallocated if necessary,
 * i.e. if the requested size is larger than the current size, which is a power of two.
 */
template<class T> void GlobalVector<T>::reallocate(unsigned int size)
    {
    if (size > GlobalArray<T>::getNumElements())
        {
        // reallocate
        unsigned int new_allocated_size = GlobalArray<T>::getNumElements() ? GlobalArray<T>::getNumElements() : 1;

        // double the size as often as necessary
        while (size > new_allocated_size)
            new_allocated_size = ((unsigned int) (((float) new_allocated_size) * RESIZE_FACTOR)) + 1 ;

        // actually resize the underlying GlobalArray
        GlobalArray<T>::resize(new_allocated_size);
        }
    }

/*! \param new_size New size of vector
 \post The GlobalVector will be re-allocated if necessary to hold the new elements.
       The newly allocated memory is \b not initialized. It is responsbility of the caller to ensure correct initialiation,
       e.g. using clear()
*/
template<class T> void GlobalVector<T>::resize(unsigned int new_size)
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
template<class T> void GlobalVector<T>::push_back(const T& val)
    {
    reallocate(m_size+1);

    T *data = acquireHost(access_mode::readwrite);
    data[m_size++] = val;
    GlobalArray<T>::release();
    }

//! Remove an element from the end of the list
template<class T> void GlobalVector<T>::pop_back()
    {
    assert(m_size);
    m_size--;
    }

//! Remove an element in the middle
template<class T> void GlobalVector<T>::erase(unsigned int i)
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
    GlobalArray<T>::release();
    }

//! Clear the list
template<class T> void GlobalVector<T>::clear()
    {
    m_size = 0;
    }

/*! \param mode Access mode for the GlobalArray
 */
template<class T> T * GlobalVector<T>::acquireHost(const access_mode::Enum mode) const
    {
    #ifdef ENABLE_CUDA
    return GlobalArray<T>::acquire(access_location::host, access_mode::readwrite, false);
    #else
    return GlobalArray<T>::acquire(access_location::host, access_mode::readwrite);
    #endif
    }

#endif
