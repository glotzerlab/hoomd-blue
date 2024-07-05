// Copyright (c) 2009-2024 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

/*!
 * \file GPUPolymorph.h
 * \brief Defines the GPUPolymorph class for polymorphic device objects.
 */

#ifndef HOOMD_GPU_POLYMORPH_H_
#define HOOMD_GPU_POLYMORPH_H_

#include "ExecutionConfiguration.h"
#include "GPUArray.h"

#include <cstdlib>
#include <functional>
#include <memory>
#include <type_traits>

#ifdef ENABLE_HIP
#include "GPUPolymorph.cuh"
#endif

namespace hoomd
    {
//! Wrapper for a polymorphic object accessible on the host or device.
/*!
 * Polymorphic objects cannot be passed directly to a CUDA kernel because the host and device have
 * different virtual tables. Moreover, simple copies of objects between host and device memory also
 * do not work because they could write over those tables. One possible solution is implemented
 * here. Both a host-memory and a device-memory copy of the polymorphic object are allocated and
 * initialized. The device-memory object must be constructed inside a CUDA kernel, and it is placed
 * into the allocated global memory. Both of these objects are managed by std::unique_ptr, and so
 * they are safely destroyed when the wrapper is reset or goes out of scope.
 *
 * The wrapper is constructed using the \a Base class as a template parameter. On construction, the
 * host and device memory is not allocated. The objects can be allocated as type \a Derived using
 * the ::reset method, with an appropriate set of arguments for the constructor of \a Derived. Each
 * pointer can then be obtained using the
 * ::get method with the appropriate access mode.
 *
 * There are a few underlying assumptions of this wrapper that must be satisfied:
 *
 * 1. Constructor arguments must be forwarded to a device kernel from the host. Don't make them
 * anything too bulky.
 * 2. Device-memory objects are allocated and initialized by hoomd::gpu::device_new. You will need
 * to explicitly instantiate this template in a *.cu file, or you will get undefined symbol errors.
 * 3. When the allocations are reset or the object goes out of scope, the device-memory object will
 * be destructed and freed. The base destructor is first called from inside a kernel; it should be
 * virtual to ensure proper destructor chaining. Afterwards, the memory is simply freed. You will
 * need to explicitly instantiate hoomd::gpu::device_delete for the base class in a *.cu file, or
 * you will get undefined symbol errors.
 *
 * This wrapper essentially acts like a factory class that also manages the necessary objects.
 *
 * When instantiating the CUDA functions, you will need to use separable CUDA compilation because of
 * the way HOOMD handles the device object virtual tables. This may place some restrictions on what
 * can be implemented through a plugin interface since the device functions are all resolved at
 * compile-time.
 *
 * \tparam Base Base class for the polymorphic object.
 */
template<class Base> class GPUPolymorph
    {
    static_assert(std::is_polymorphic<Base>::value, "Base should be a polymorphic class.");

    private:
    typedef std::unique_ptr<Base> host_ptr;

#ifdef ENABLE_HIP
    //! Simple custom deleter for the base class.
    /*!
     * This custom deleter is used by std::unique_ptr to free the memory allocation.
     */
    class CUDADeleter
        {
        public:
        //! Default constructor (gives null exec. conf)
        CUDADeleter() { }

        //! Constructor with an execution configuration
        CUDADeleter(std::shared_ptr<const ExecutionConfiguration> exec_conf)
            : m_exec_conf(exec_conf)
            {
            }

        //! Delete operator
        /*!
         * \param p Pointer to a possibly polymorphic \a Base object
         *
         * If the allocation for a device pointer is still valid, the deleter will first
         * call the destructor for the \a Base class from inside a kernel. Then, the memory
         * will be freed by cudaFree().
         */
        void operator()(Base* p) const
            {
            if (m_exec_conf && m_exec_conf->isCUDAEnabled() && p)
                {
                m_exec_conf->msg->notice(5)
                    << "Freeing device memory from GPUPolymorph [Base = " << typeid(Base).name()
                    << "]" << std::endl;
                gpu::device_delete(p);
                }
            }

        private:
        std::shared_ptr<const ExecutionConfiguration>
            m_exec_conf; //!< HOOMD execution configuration
        };

    typedef std::unique_ptr<Base, CUDADeleter> device_ptr;
#endif // ENABLE_HIP

    public:
    //! Constructor
    /*!
     * \param exec_conf HOOMD execution configuration.
     * \post The host-memory and device-memory pointers are not allocated (null values).
     */
    GPUPolymorph(std::shared_ptr<const ExecutionConfiguration> exec_conf) noexcept(true)
        : m_exec_conf(exec_conf), m_host_data(nullptr)
        {
        m_exec_conf->msg->notice(4)
            << "Constructing GPUPolymorph [Base = " << typeid(Base).name() << "]" << std::endl;
#ifdef ENABLE_HIP
        m_device_data = device_ptr(nullptr, CUDADeleter(m_exec_conf));
#endif // ENABLE_HIP
        }

    //! Destructor
    ~GPUPolymorph()
        {
        m_exec_conf->msg->notice(4)
            << "Destroying GPUPolymorph [Base = " << typeid(Base).name() << "]" << std::endl;
        }

    //! Copy constructor.
    /*!
     * Copy is not supported for underlying unique_ptr data.
     */
    GPUPolymorph(const GPUPolymorph& other) = delete;

    //! Copy assignment.
    /*!
     * Copy is not supported for underlying unique_ptr data.
     */
    GPUPolymorph& operator=(const GPUPolymorph& other) = delete;

    //! Move constructor.
    /*!
     * \param other Another GPUPolymorph.
     * \a The underlying data of \a other is pilfered to construct this object.
     */
    GPUPolymorph(GPUPolymorph&& other) noexcept(true)
        : m_exec_conf(std::move(other.m_exec_conf)), m_host_data(std::move(other.m_host_data))
        {
#ifdef ENABLE_HIP
        m_device_data = std::move(other.m_device_data);
#endif // ENABLE_HIP
        }

    //! Move assignment.
    /*!
     * \param other Another GPUPolymorph.
     * \a The underlying data of \a other is pilfered and assigned to this object.
     */
    GPUPolymorph& operator=(GPUPolymorph&& other) noexcept(true)
        {
        if (*this != other)
            {
            m_exec_conf = std::move(other.m_exec_conf);
            m_host_data = std::move(other.m_host_data);
#ifdef ENABLE_HIP
            m_device_data = std::move(other.m_device_data);
#endif // ENABLE_HIP
            }
        return *this;
        }

    //! Reset (and allocate) the host-memory and device-memory objects.
    /*!
     * \tparam Derived Polymorphic object type to create.
     * \tparam Args Argument types for constructor of \a Derived object to call.
     * \param args Argument values to construct \a Derived object.
     *
     * The host-memory copy is allocated and initialized using the new keyword. If CUDA is available
     * for the execution configuration, the device-memory object is created by
     * hoomd::gpu::device_new.
     *
     * \a Derived must be derived from \a Base for this wrapper to make sense. A compile-time
     * assertion will fail otherwise.
     */
    template<class Derived, typename... Args> void reset(Args... args)
        {
        static_assert(std::is_base_of<Base, Derived>::value,
                      "Polymorph must be derived from Base.");

        m_exec_conf->msg->notice(4)
            << "Resetting GPUPolymorph [Derived = " << typeid(Derived).name()
            << ", Base = " << typeid(Base).name() << "] (" << sizeof(Derived) << " bytes)"
            << std::endl;

        m_host_data.reset(new Derived(args...));
#ifdef ENABLE_HIP
        if (m_exec_conf->isCUDAEnabled())
            {
            m_device_data.reset(gpu::device_new<Derived>(args...));
            if (m_exec_conf->isCUDAErrorCheckingEnabled())
                CHECK_CUDA_ERROR();
            }
#endif // ENABLE_HIP
        }

    //! Get the raw pointer associated with a given copy.
    /*!
     * \param location Access location (host, device) for object.
     * \returns Pointer to polymorphic object with type \a Base.
     *
     * If the object has not been initialized or the access location is not recognized,
     * a nullptr is returned.
     */
    Base* get(const access_location::Enum location) const
        {
        if (location == access_location::host)
            {
            return m_host_data.get();
            }
#ifdef ENABLE_HIP
        else if (location == access_location::device)
            {
            return m_device_data.get();
            }
#endif // ENABLE_HIP
        else
            {
            return nullptr;
            }
        }

    private:
    std::shared_ptr<const ExecutionConfiguration> m_exec_conf; //!< HOOMD execution configuration
    host_ptr m_host_data;                                      //!< Host-memory copy
#ifdef ENABLE_HIP
    device_ptr m_device_data; //!< Device-memory copy
#endif                        // ENABLE_HIP
    };

    } // end namespace hoomd

#endif // HOOMD_GPU_POLYMORPH_H_
