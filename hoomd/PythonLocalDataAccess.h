#ifndef __PYTHON_LOCAL_DATA_ACCESS_H__
#define __PYTHON_LOCAL_DATA_ACCESS_H__

#include "GlobalArray.h"
#include <string>
#include <type_traits>
#include <utility>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>


/// Base class for buffers for LocalDataAccess template class type checking.
/** HOOMDBuffer classes need to implement a templated make method that takes
 *  data: pointer to type T
 *  shape: shape of the array
 *  stride: the strides to move one index in each dimension of shape
 *  read_only: whether the buffer is meant to be read_only
 */
struct HOOMDBuffer {};


/// Represents the data required to specify a CPU buffer object in Python.
/** Stores the data to create pybind11::buffer_info objects. This is necessary
 *  since buffer_info objects cannot be reused. Once the buffer is read many of
 *  the members are moved. In addition, this class allows for a uniform way of
 *  specifying a CPU(Host) or GPU(Device) buffer.
 */
struct HOOMDHostBuffer : public HOOMDBuffer
    {
    static const auto device = access_location::host;
    void* m_data;
    ssize_t m_itemsize;
    std::string m_typestr;
    int m_dimensions;
    std::vector<ssize_t> m_shape;
    std::vector<ssize_t> m_strides;
    bool m_read_only;

    HOOMDHostBuffer(void* data, ssize_t itemsize, std::string typestr,
                      int dimensions, std::vector<ssize_t> shape,
                      std::vector<ssize_t> strides, bool read_only)
        : m_data(data), m_itemsize(itemsize), m_typestr(typestr),
          m_dimensions(dimensions), m_shape(shape), m_strides(strides),
          m_read_only(read_only)
        {
        if (m_shape.size() != m_strides.size())
            {
            throw std::runtime_error("GPU buffer shape != strides.");
            }
        }

    template<class T>
    static HOOMDHostBuffer make(T* data, std::vector<ssize_t> shape,
                                std::vector<ssize_t> strides, bool read_only)
        {
        return HOOMDHostBuffer(
            data, sizeof(T), pybind11::format_descriptor<T>::format(),
            shape.size(), shape, strides, read_only);
        }

    pybind11::buffer_info new_buffer()
        {
        return pybind11::buffer_info(
                m_data,
                m_itemsize,
                m_typestr,
                m_dimensions,
                std::vector<ssize_t>(m_shape),
                std::vector<ssize_t>(m_strides)
                );
        }

    bool getReadOnly() { return m_read_only; }
    };


#if ENABLE_HIP
/// Represents the data required to implement the __cuda_array_interface__.
/** Creates the Python dictionary to represent a GPU array through the
 *  __cuda_array_interface__. Currently supports version 2 of the protocol.
 */
struct HOOMDDeviceBuffer : public HOOMDBuffer
    {
    static const auto device = access_location::device;
    void* m_data;
    std::string m_typestr;
    std::vector<ssize_t> m_shape;
    std::vector<ssize_t> m_strides;
    bool m_read_only;

    HOOMDDeviceBuffer(void* data, std::string typestr,
                      std::vector<ssize_t> shape, std::vector<ssize_t> strides,
                      bool read_only)
        : m_data(data), m_typestr(typestr), m_shape(shape), m_strides(strides),
          m_read_only(read_only)
        {
        if (m_shape.size() != m_strides.size())
            {
            throw std::runtime_error("GPU buffer shape != strides.");
            }
        }

    template<class T>
    static HOOMDDeviceBuffer make(T* data, std::vector<ssize_t> shape,
                                  std::vector<ssize_t> strides, bool read_only)
        {
        return HOOMDDeviceBuffer(data, pybind11::format_descriptor<T>::format(),
                                 shape, strides, read_only);
        }

    /// Convert object to a __cuda_array_interface__ v2 compliant Python dict.
    /// We can't only add the existing values in the HOOMDDeviceBuffer because
    /// CuPy and potentially other packages that use the interface can't handle
    /// a shape where the first dimension is zero and the rest are non-zero. In
    /// addition, strides must always be the same length as shape, meaning we
    /// need to ensure that we change strides to (0,) when the shape is (0,).
    /// Likewise, we need to ensure that the int that serves as the device
    /// pointer is zero for size 0 arrays.
    pybind11::dict getCudaArrayInterface()
        {
        auto interface = pybind11::dict();
        interface["typestr"] = m_typestr;
        interface["version"] = 2;
        std::pair<intptr_t, bool> data{};
        pybind11::list shape{};
        pybind11::list strides{};
        if (m_shape.size() == 0 || m_shape[0] == 0)
            {
            data = std::pair<intptr_t, bool>(0, m_read_only);
            shape.append(0);
            strides.append(0);
            }
        else
            {
            data = std::pair<intptr_t, bool>((intptr_t)m_data, m_read_only);
            for (auto s : m_shape)
                {
                shape.append(s);
                }
            for (auto s : m_strides)
                {
                strides.append(s);
                }
            }
        interface["data"] = data;
        interface["shape"] = pybind11::tuple(shape);
        interface["strides"] = pybind11::tuple(strides);
        return interface;
        }

    bool getReadOnly() { return m_read_only; }

    };
#endif


/// Base class for accessing Global or GPU arrays/vectors in Python.
/** Template Parameters:
 *  Output - the output buffer class for the class should be HOOMDDeviceBuffer
 *  or HOOMDHostBuffer
 *  Data - the class of the object we wish to expose data from
 *
 *  This class only allows access when the m_in_manager flag is true. The flag
 *  should only be changed when entering or exiting a Python context manager.
 *  The design of Python access is to restrict access to within a context
 *  manager to prevent invalid reads/writes in Python (and SEGFAULTS).
 *
 *  The main methods of LocalDataAccess are getBuffer and getGlobalBuffer which
 *  provide a way to automatically convert an Global/GPUArray into an object of
 *  type Output. All classes that expose arrays in Python should use these
 *  classes.
 *
 *  This class stores ArrayHandles using a unique pointer to prevent a resource
 *  from being dropped before the object is destroyed.
 */
template <class Output, class Data>
class LocalDataAccess
    {
    static_assert(
        std::is_base_of<HOOMDBuffer, Output>::value,
        "Output template parameter for LocalDataAccess must be a subclass of HOOMDBuffer."
    );

    public:
        inline LocalDataAccess(Data& data) :
            m_data(data), m_in_manager(false) {}

        virtual ~LocalDataAccess() = default;

        /// signifies entering into a Python context manager
        void enter() {m_in_manager = true;}

        /// signifies exiting a Python context manager
        void exit()
            {
            clear();
            m_in_manager = false;
            }

    protected:
        /// Convert Global/GPUArray or vector into an Ouput object for Python
        /** This function is for arrays that are of a size less than or equal to
         *  their global size. An example is particle positions. On each MPI
         *  rank or GPU, the number of positions a ranks knows about (including
         *  ghost particles) is less than or equal to the number of total
         *  particles in the system. For arrays that are the sized according to
         *  the global number, use getGlobalBuffer (quantities such as rtags).
         *  Template parameters:
         *  T - the value stored in the by the internal array (i.e. the template
         *  parameter of the ArrayHandle
         *  S - the exposed type of data to Python
         *  U - the array class returned by the parameter get_array_func
         *
         *  Arguments:
         *  handle: a reference to the unique_ptr that holds the ArrayHandle.
         *  get_array_func: the method of m_data to use to access the array.
         *  ghost: whether to return only ghost data (defaults to false)
         *  include_both: whether to return all known data (defaults to false)
         *  second_dimension_size: the size of the second dimension (defaults to
         *  0)
         *  offest: the offset in bytes from the start of the array to the
         *  start of the exposed array in Python (defaults to no offset).
         *  strides: the strides in bytes of the array (defaults to sizeof(T) or
         *  {sizeof(S), sizeof(T)} depending on dimension).
         */
        template<class T, class S, template<class> class U=GlobalArray>
        Output getBuffer(
            std::unique_ptr<ArrayHandle<T> >& handle,
            const U<T>& (Data::*get_array_func)() const,
            bool ghost = false,
            bool include_both = false,
            unsigned int second_dimension_size = 0,
            ssize_t offset = 0,
            std::vector<ssize_t> strides = {}
        )
            {
            checkManager();

            if (ghost && include_both)
                {
                throw std::runtime_error(
                    "Cannot specify both the ghost and include_both flags.");
                }

            bool read_only = ghost || include_both;

            updateHandle(handle, get_array_func, read_only);

            auto N = m_data.getN();
            auto ghostN = m_data.getNGhosts();
            auto size = N;
            T* _data = handle.get()->data;

            if (include_both)
                {
                size += ghostN;
                }
            else if (ghost)
                {
                _data += N;
                size = ghostN;
                }
            S* data = (S*)(((char*)_data) + offset);

            std::vector<ssize_t> shape{size, second_dimension_size};
            if (strides.size() == 0 && second_dimension_size == 0)
                {
                shape.pop_back();
                strides = std::vector<ssize_t>({sizeof(T)});
                }
            else if (strides.size() == 0 && second_dimension_size != 0)
                {
                strides = std::vector<ssize_t>({sizeof(T), sizeof(S)});
                }
            return Output::make(data, shape, strides, read_only);
            }

        /// Convert Global/GPUArray or vector into an Ouput object for Python
        /** This function is for arrays that are of a size equal to their global
         *  size. An example is the reverse tag index. On each MPI rank or GPU,
         *  the size of the reverse tag index is equal to the entire number of
         *  particles in the system.  For arrays that are the sized according to
         *  the local box, use getBuffer (quantities such as particle positions).
         *  Template parameters:
         *  T - the value stored in the by the internal array (i.e. the template
         *  parameter of the ArrayHandle
         *  U - the array class returned by the parameter get_array_func
         *
         *  Arguments:
         *  handle: a reference to the unique_ptr that holds the ArrayHandle.
         *  get_array_func: the method of m_data to use to access the array.
         *  of the exposed array in Python.
         *  read_only: whether the array should be read only (defaults to True).
         */
        template<class T, template<class> class U=GlobalArray>
        Output getGlobalBuffer(std::unique_ptr<ArrayHandle<T> >& handle,
                               const U<T>& (Data::*get_array_func)() const,
                               bool read_only=true)
            {
            checkManager();
            updateHandle(handle, get_array_func, read_only);

            auto size = m_data.getNGlobal();
            unsigned int* data = handle.get()->data;

            return Output::make(data, std::vector<ssize_t>({size}),
                                std::vector<ssize_t>({sizeof(T)}), true);
            }


        // clear should remove any references to ArrayHandle objects so the
        // handle can be released for other objects.
        virtual void clear() = 0;

    private:
        /// Ensure that arrays are not accessed outside context manager.
        inline void checkManager()
            {
            if (!m_in_manager)
                {
                throw std::runtime_error(
                    "Cannot access arrays outside context manager.");
                }
            }

        /// Helper function to acquire array handle if not already acquired.
        template<class T, template<class> class U>
        void updateHandle(std::unique_ptr<ArrayHandle<T> >& handle,
                          const U<T>& (Data::*get_array_func)() const,
                          bool read_only)
            {
            if (!handle)
                {
                auto mode = read_only ? access_mode::read : access_mode::readwrite;
                std::unique_ptr<ArrayHandle<T> > new_handle(
                    new ArrayHandle<T>((m_data.*get_array_func)(),
                    Output::device,
                    mode));
                handle = std::move(new_handle);
                }
            }

        /// object to access array data from
        Data& m_data;
        /// flag for being inside Python context manager
        bool m_in_manager;
    };

void export_HOOMDHostBuffer(pybind11::module &m);

#if ENABLE_HIP
void export_HOOMDDeviceBuffer(pybind11::module &m);
#endif

#endif
