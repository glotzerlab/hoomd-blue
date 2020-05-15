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
        interface["data"] = std::pair<intptr_t, bool>(
            (intptr_t)m_data, m_read_only);
        interface["typestr"] = m_typestr;
        interface["version"] = 2;
        pybind11::list shape;
        for (auto s : m_shape) {shape.append(s);} 
        interface["shape"] = pybind11::tuple(shape);
        pybind11::list strides;
        for (auto s : m_strides) {strides.append(s);}
        interface["strides"] = pybind11::tuple(strides);
        return interface;
        }

    bool getReadOnly() { return m_read_only; }

    };
#endif


template <class OUTPUT, class DATA>
class LocalDataAccess
    {
    public:
        inline LocalDataAccess(DATA& data);

        virtual ~LocalDataAccess() = default;

        void enter() {m_in_manager = true;}

        void exit()
            {
            clear();
            m_in_manager = false;
            }

    protected:
        template <class T, class S, template<class> class U>
        OUTPUT getBuffer(
            // handle to operate on
            std::unique_ptr<ArrayHandle<T> >& handle,
            // function to use for getting array
            const U<T>& (DATA::*get_array_func)() const,
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
                    OUTPUT::device,
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
            S* data = (S*)_data;

            if (strides.size() == 0 && second_dimension_size == 0)
                {
                return OUTPUT(
                    data + offset,
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
            return OUTPUT(
                data + offset,
                sizeof(S),
                pybind11::format_descriptor<S>::format(),
                2,
                std::vector<ssize_t>({size, second_dimension_size}),
                strides,
                read_only);
            }

        virtual void clear() = 0;

    private:
        DATA& m_data;
        bool m_in_manager;
    };

template <class OUTPUT, class DATA>
LocalDataAccess<OUTPUT, DATA>::LocalDataAccess(DATA& data)
    : m_data(data), m_in_manager(false) {}

void export_HOOMDHostBuffer(pybind11::module &m);

#if ENABLE_HIP
void export_HOOMDDeviceBuffer(pybind11::module &m);
#endif

#endif
