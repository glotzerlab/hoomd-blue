#ifndef __PYTHON_LOCAL_DATA_ACCESS_H__
#define __PYTHON_LOCAL_DATA_ACCESS_H__

#include "GlobalArray.h"
#include <string>
#include <utility>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>


struct HOOMDHostBuffer
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
          m_read_only(read_only) {}

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
struct HOOMDDeviceBuffer
    {
    static const auto device = access_location::device;
    void* m_data;
    std::string m_typestr;
    std::vector<ssize_t> m_shape;
    std::vector<ssize_t> m_strides;
    bool m_read_only;

    HOOMDDeviceBuffer(void* data, ssize_t unused1, std::string typestr,
                      ssize_t unused2, std::vector<ssize_t> shape,
                      std::vector<ssize_t> strides, bool read_only)
        : m_data(data), m_typestr(typestr), m_shape(shape), m_strides(strides),
          m_read_only(read_only) {}

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
            stride = pybind11::none;
        else
            {
            data = std::pair<intptr_t, bool>((intptr_t)m_data, m_read_only);
            for (auto s : m_shape)
                {
                shape.append(s);
                }
            pybind11::list strides;
            for (auto s : m_strides)
                {
                strides.append(s);
                }
            }
        interface["data"]
        interface["shape"] = pybind11::tuple(shape);
        interface["strides"] = pybind11::tuple(strides);
        return interface;
        }

    bool getReadOnly() { return m_read_only; }

    };
#endif


// LocalDataAccess is a base class for allowing the converting of arrays stored
// using Global or GPU arrays/vectors.
// Output - the output buffer class for the class should be HOOMDDeviceBuffer or
// HOOMDHostBuffer
// Data - the class of the object we wish to expose data from
template <class Output, class Data>
class LocalDataAccess
    {
    public:
        inline LocalDataAccess(Data& data);

        virtual ~LocalDataAccess() = default;

        // signifies entering into a Python context manager
        void enter() {m_in_manager = true;}

        // signifies exiting a Python context manager
        void exit()
            {
            clear();
            m_in_manager = false;
            }

    protected:
        // T - the value stored in the by the internal array (i.e. the template
        // parameter of the ArrayHandle
        // S - the exposed type of data to Python
        // U - the array class returned by the parameter get_array_func
        template<class T, class S, template<class> class U=GlobalArray>
        Output getBuffer(
            // handle to operate on
            std::unique_ptr<ArrayHandle<T> >& handle,
            // function to use for getting array
            const U<T>& (Data::*get_array_func)() const,
            // Whether to return ghost particles properties
            bool ghost = false,
            // Whether to return normal and ghost particles
            bool include_both = false,
            // Size of the second dimension
            unsigned int second_dimension_size = 0,
            // offset from beginning of pointer
            ssize_t offset = 0,
            // Strides in each dimension
            std::vector<ssize_t> strides = {}
        )
            {
            if (!m_in_manager)
                {
                throw std::runtime_error(
                    "Cannot access arrays outside context manager.");
                }

            if (ghost && include_both)
                {
                throw std::runtime_error(
                    "Cannot specify both the ghost and include_both flags.");
                }

            auto read_only = ghost || include_both;

            if (!handle)
                {
                auto mode = read_only ? access_mode::read : access_mode::readwrite;
                std::unique_ptr<ArrayHandle<T> > new_handle(
                    new ArrayHandle<T>((m_data.*get_array_func)(),
                    Output::device,
                    mode));
                handle = std::move(new_handle);
                }

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

            if (strides.size() == 0 && second_dimension_size == 0)
                {
                return Output(
                    data,
                    sizeof(S),
                    pybind11::format_descriptor<S>::format(),
                    1,
                    std::vector<ssize_t>({size}),
                    std::vector<ssize_t>({sizeof(T)}),
                    read_only);
                }
            else if (strides.size() == 0 && second_dimension_size != 0)
                {
                strides = std::vector<ssize_t>({sizeof(T), sizeof(S)});
                }
            return Output(
                data,
                sizeof(S),
                pybind11::format_descriptor<S>::format(),
                2,
                std::vector<ssize_t>({size, second_dimension_size}),
                strides,
                read_only);
            }

        // Helper function for when the exposed type and the internal type are
        // the same.
        template<class T, template<class> class U>
        Output getBufferSameType(std::unique_ptr<ArrayHandle<T> >& handle,
                                 const U<T>& (Data::*get_array_func)() const,
                                 bool ghost = false,
                                 bool include_both = false,
                                 unsigned int second_dimension_size = 0,
                                 ssize_t offset = 0,
                                 std::vector<ssize_t> strides = {})
            {
            return this->template getBuffer<T, T, U>(
                handle, get_array_func, ghost, include_both,
                second_dimension_size, offset, strides
                );
            }

        // clear should remove any references to ArrayHandle objects so the
        // handle can be released for other objects.
        virtual void clear() = 0;

    private:
        Data& m_data;
        bool m_in_manager;
    };

template <class Output, class Data>
LocalDataAccess<Output, Data>::LocalDataAccess(Data& data)
    : m_data(data), m_in_manager(false) {}

void export_HOOMDHostBuffer(pybind11::module &m);

#if ENABLE_HIP
void export_HOOMDDeviceBuffer(pybind11::module &m);
#endif

#endif
