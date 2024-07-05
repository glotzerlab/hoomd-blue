// Copyright (c) 2009-2024 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

/*! \file GPUVector.h
    \brief Defines the GPUVector and GlobalVector classes
*/

#ifdef __HIPCC__
#error This header cannot be compiled by nvcc
#endif

#pragma once
#include "GPUArray.h"
#include <algorithm>

// The factor with which the array size is incremented
#define RESIZE_FACTOR 9.f / 8.f

namespace hoomd
    {
//! Forward declarations
template<class T> class GPUVector;

template<class T> class GlobalVector;

//! Class for managing a vector of elements on the GPU mirrored to the CPU
/*! The GPUVectorBase class is a simple container for a variable number of elements. Its interface
   is inspired by std::vector and it offers methods to insert new elements at the end of the list,
   and to remove them from there.

    It uses a GPUArray as the underlying storage class, thus the data in a GPUVectorBase can also be
   accessed directly using ArrayHandles.

    In the current implementation, a GPUVectorBase can only grow (but not shrink) in size until it
   is destroyed.

    \ingroup data_structs
*/
template<class T, class Array> class GPUVectorBase : public Array
    {
    public:
    //! Frees memory
    virtual ~GPUVectorBase() = default;

    //! Copy constructor
    GPUVectorBase(const GPUVectorBase& from);
    //! Move constructor
    GPUVectorBase(GPUVectorBase&& other);
    //! = operator
    GPUVectorBase& operator=(const GPUVectorBase& rhs);
    //! Move assignment operator
    GPUVectorBase& operator=(GPUVectorBase&& other);

    //! swap this GPUVectorBase with another
    inline void swap(GPUVectorBase& from);

    /*!
      \returns the current size of the vector
    */
    size_t size() const
        {
        return m_size;
        }

    //! Resize the GPUVectorBase
    /*! \param new_size New number of elements
     */
    virtual void resize(size_t new_size);

    //! Resize the GPUVectorBase
    /*! \param new_size New number of elements
     *  \param const value to initialize newly added elements with
     */
    virtual void resize(size_t new_size, const T& value);

    //! Insert an element at the end of the vector
    /*! \param val The new element
     */
    virtual void push_back(const T& val);

    //! Remove an element from the end of the list
    virtual void pop_back();

    //! Remove an element by index
    virtual void erase(size_t i);

    //! Clear the list
    virtual void clear();

    //! Proxy class to provide access to the data elements of the vector
    class data_proxy
        {
        public:
        //! Constructor
        data_proxy(const GPUVectorBase<T, Array>& _vec, const size_t _n) : vec(_vec), n(_n) { }

        //! Type cast
        operator T() const
            {
            auto dispatch = vec.acquireHost(access_mode::read);
            T* data = dispatch.get();
            T val = data[n];
            return val;
            }

        //! Assignment
        data_proxy& operator=(T rhs)
            {
            auto dispatch = vec.acquireHost(access_mode::readwrite);
            T* data = dispatch.get();
            data[n] = rhs;
            return *this;
            }

        private:
        const GPUVectorBase<T, Array>& vec; //!< The vector that is accessed
        size_t n;                           //!< The index of the element to access
        };

    //! Get a proxy-reference to a list element
    data_proxy operator[](size_t n)
        {
        assert(n < m_size);
        return data_proxy(*this, n);
        }

    //! Get a proxy-reference to a list element (const version)
    data_proxy operator[](size_t n) const
        {
        assert(n < m_size);
        return data_proxy(*this, n);
        }

    protected:
    //! Default constructor
    GPUVectorBase();

    //! Constructs an empty GPUVectorBase
    GPUVectorBase(std::shared_ptr<const ExecutionConfiguration> exec_conf);

    //! Constructs a GPUVectorBase
    GPUVectorBase(size_t size, std::shared_ptr<const ExecutionConfiguration> exec_conf);

    //! Constructs a GPUVectorBase of given size, initialized with a constant value
    GPUVectorBase(unsigned int size,
                  const T& value,
                  std::shared_ptr<const ExecutionConfiguration> exec_conf);

#ifdef ENABLE_HIP
    //! Constructs an empty GPUVectorBase
    GPUVectorBase(std::shared_ptr<const ExecutionConfiguration> exec_conf, bool mapped);

    //! Constructs a GPUVectorBase
    GPUVectorBase(size_t size,
                  std::shared_ptr<const ExecutionConfiguration> exec_conf,
                  bool mapped);
#endif

    private:
    size_t m_size; //!< Number of elements

    //! Helper function to reallocate the GPUArray (using amortized array resizing)
    void reallocate(size_t new_size);

    //! Acquire the underlying GPU array on the host
    ArrayHandleDispatch<T> acquireHost(const access_mode::Enum mode) const;

    friend class data_proxy;
    };

//******************************************
// GPUVectorBase implementation
// *****************************************

//! Default constructor
/*! \warning When using this constructor, a properly initialized GPUVectorBase with an exec_conf
   needs to be swapped in later, after construction of the GPUVectorBase.
 */
template<class T, class Array> GPUVectorBase<T, Array>::GPUVectorBase() : Array(), m_size(0) { }

/*! \param exec_conf Shared pointer to the execution configuration
 */
template<class T, class Array>
GPUVectorBase<T, Array>::GPUVectorBase(std::shared_ptr<const ExecutionConfiguration> exec_conf)
    : Array(0, exec_conf), m_size(0)
    {
    }

/*! \param size Number of elements to allocate initial memory for in the array
    \param exec_conf Shared pointer to the execution configuration
*/
template<class T, class Array>
GPUVectorBase<T, Array>::GPUVectorBase(size_t size,
                                       std::shared_ptr<const ExecutionConfiguration> exec_conf)
    : Array(size, exec_conf), m_size(size)
    {
    }

/*! \param size Number of elements to allocate initial memory for in the array
    \param value constant value to initialize the array with
    \param exec_conf Shared pointer to the execution configuration
*/
template<class T, class Array>
GPUVectorBase<T, Array>::GPUVectorBase(unsigned int size,
                                       const T& value,
                                       std::shared_ptr<const ExecutionConfiguration> exec_conf)
    : Array(size, exec_conf), m_size(size)
    {
    auto dispatch = acquireHost(access_mode::readwrite);
    T* data = dispatch.get();
    for (unsigned int i = 0; i < size; ++i)
        data[i] = value;
    }

template<class T, class Array>
GPUVectorBase<T, Array>::GPUVectorBase(const GPUVectorBase& from) : Array(from), m_size(from.m_size)
    {
    }

template<class T, class Array>
GPUVectorBase<T, Array>::GPUVectorBase(GPUVectorBase&& other)
    : Array(std::move(other)), m_size(std::move(other.m_size))
    {
    }

template<class T, class Array>
GPUVectorBase<T, Array>& GPUVectorBase<T, Array>::operator=(const GPUVectorBase& rhs)
    {
    if (this != &rhs) // protect against invalid self-assignment
        {
        m_size = rhs.m_size;
        // invoke base class operator
        Array::operator=(rhs);
        }

    return *this;
    }

template<class T, class Array>
GPUVectorBase<T, Array>& GPUVectorBase<T, Array>::operator=(GPUVectorBase&& other)
    {
    if (this != &other)
        {
        m_size = std::move(other.m_size);
        // invoke move assignment for the base class
        Array::operator=(std::move(other));
        }

    return *this;
    }

/*! \param from GPUVectorBase to swap \a this with
 */
template<class T, class Array> void GPUVectorBase<T, Array>::swap(GPUVectorBase<T, Array>& from)
    {
    std::swap(m_size, from.m_size);
    Array::swap(from);
    }

/*! \param size New requested size of allocated memory
 *
 * Internally, this method uses amortized resizing of allocated memory to
 * avoid excessive copying of data. The GPUArray is only reallocated if necessary,
 * i.e. if the requested size is larger than the current size, which is a power of two.
 */
template<class T, class Array> void GPUVectorBase<T, Array>::reallocate(size_t size)
    {
    if (size > Array::getNumElements())
        {
        // reallocate
        size_t new_allocated_size = Array::getNumElements() ? Array::getNumElements() : 1;

        // double the size as often as necessary
        while (size > new_allocated_size)
            new_allocated_size = ((size_t)(((double)new_allocated_size) * RESIZE_FACTOR)) + 1;

        // actually resize the underlying GPUArray
        Array::resize(new_allocated_size);
        }
    }

/*! \param new_size New size of vector
 \post The GPUVectorBase will be re-allocated if necessary to hold the new elements.
       The newly allocated memory is \b not initialized. It is responsibility of the caller to
 ensure correct initialization, e.g. using clear()
*/
template<class T, class Array> void GPUVectorBase<T, Array>::resize(size_t new_size)
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

/*!
 \post The GPUVectorBase will be re-allocated if necessary to hold the new elements.
       The newly allocated memory is initialized with a constant value.
*/
template<class T, class Array> void GPUVectorBase<T, Array>::resize(size_t new_size, const T& value)
    {
    size_t old_size = m_size;
    resize(new_size);
    auto dispatch = acquireHost(access_mode::readwrite);
    T* data = dispatch.get();
    for (size_t i = old_size; i < new_size; ++i)
        data[i] = value;
    }

//! Insert an element at the end of the vector
template<class T, class Array> void GPUVectorBase<T, Array>::push_back(const T& val)
    {
    reallocate(m_size + 1);

    auto dispatch = acquireHost(access_mode::readwrite);
    T* data = dispatch.get();
    data[m_size++] = val;
    }

//! Remove an element from the end of the list
template<class T, class Array> void GPUVectorBase<T, Array>::pop_back()
    {
    assert(m_size);
    m_size--;
    }

//! Remove an element in the middle
template<class T, class Array> void GPUVectorBase<T, Array>::erase(size_t i)
    {
    assert(i < m_size);
    auto dispatch = acquireHost(access_mode::readwrite);

    T* data = dispatch.get();
    T* res = data;
    for (size_t n = 0; n < m_size; ++n)
        {
        if (n != i)
            {
            *res = *data;
            res++;
            }
        data++;
        }
    m_size--;
    }

//! Clear the list
template<class T, class Array> void GPUVectorBase<T, Array>::clear()
    {
    m_size = 0;
    }

/*! \param mode Access mode for the GPUArray
 */
template<class T, class Array>
ArrayHandleDispatch<T> GPUVectorBase<T, Array>::acquireHost(const access_mode::Enum mode) const
    {
#ifdef ENABLE_HIP
    return GPUArrayBase<T, Array>::acquire(access_location::host, access_mode::readwrite, false);
#else
    return Array::acquire(access_location::host, access_mode::readwrite);
#endif
    }

//! Forward declarations
template<class T> class GPUArray;

template<class T> class GlobalArray;

//******************************************
// Specialization for zero copy memory (GPUArray)
//******************************************

template<class T> class GPUVector : public GPUVectorBase<T, GPUArray<T>>
    {
    public:
    //! Default constructor
    GPUVector() { }

    //! Constructs an empty GPUVector
    GPUVector(std::shared_ptr<const ExecutionConfiguration> exec_conf)
        : GPUVectorBase<T, GPUArray<T>>(exec_conf)
        {
        }

    //! Constructs a GPUVector
    GPUVector(size_t size, std::shared_ptr<const ExecutionConfiguration> exec_conf)
        : GPUVectorBase<T, GPUArray<T>>(size, exec_conf)
        {
        }

    //! Constructs a GPUVector
    GPUVector(unsigned int size,
              const T& value,
              std::shared_ptr<const ExecutionConfiguration> exec_conf)
        : GPUVectorBase<T, GPUArray<T>>(size, value, exec_conf)
        {
        }
    };

//******************************************
// Specialization for managed memory (GlobalArray)
//******************************************

template<class T> class GlobalVector : public GPUVectorBase<T, GlobalArray<T>>
    {
    public:
    //! Default constructor
    GlobalVector() { }

    //! Constructs an empty GlobalVector
    GlobalVector(std::shared_ptr<const ExecutionConfiguration> exec_conf)
        : GPUVectorBase<T, GlobalArray<T>>(exec_conf)
        {
        }

    //! Constructs a GPUVector
    GlobalVector(size_t size, std::shared_ptr<const ExecutionConfiguration> exec_conf)
        : GPUVectorBase<T, GlobalArray<T>>(size, exec_conf)
        {
        }

    //! Constructs a GlobalVector
    GlobalVector(unsigned int size,
                 const T& value,
                 std::shared_ptr<const ExecutionConfiguration> exec_conf)
        : GPUVectorBase<T, GlobalArray<T>>(size, value, exec_conf)
        {
        }

    //! Set the tag
    void setTag(const std::string& tag)
        {
        static_cast<GlobalArray<T>&>(*this).setTag(tag);
        }

    //! Get the underlying raw pointer
    const T* get() const
        {
        return static_cast<GlobalArray<T> const&>(*this).get();
        }
    };

    } // end namespace hoomd
