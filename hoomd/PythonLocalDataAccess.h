#ifndef __PYTHON_LOCAL_DATA_ACCESS_H__
#define __PYTHON_LOCAL_DATA_ACCESS_H__

#include "GlobalArray.h"
#include <string>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>


struct HOOMDHostBuffer
    {
    pybind11::buffer_info m_buffer_info;

    HOOMDHostBuffer(pybind11::buffer_info buff)
        : m_buffer_info(std::move(buff)) {}

    pybind11::buffer_info new_buffer()
        {
        return pybind11::buffer_info(
            m_buffer_info.ptr,
            m_buffer_info.itemsize,
            m_buffer_info.format,
            m_buffer_info.ndim,
            std::vector<ssize_t>(m_buffer_info.shape),
            std::vector<ssize_t>(m_buffer_info.strides),
            m_buffer_info.readonly
            );
        }
    pybind11::tuple getShape()
        {
        auto shape = pybind11::list();
        for (auto dim : m_buffer_info.shape)
            {
            shape.append(dim);
            }
        return pybind11::tuple(shape);
        }

    std::string getType()
        {
        return m_buffer_info.format;
        }
    };


template <class DATA>
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
        HOOMDHostBuffer getBuffer(
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
            size_t offset = 0,
            // Strides in each dimension
            std::vector<size_t> strides = {}
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
                    access_location::host,
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
                return HOOMDHostBuffer(pybind11::buffer_info(
                    data + offset,
                    sizeof(S),
                    pybind11::format_descriptor<S>::format(),
                    1,
                    std::vector<size_t>({size}),
                    std::vector<size_t>({sizeof(T)}),
                    read_only)
                    );
                }
            else if (strides.size() == 0 && second_dimension_size != 0)
                {
                strides = std::vector<size_t>({sizeof(T), sizeof(S)});
                }
            return HOOMDHostBuffer(pybind11::buffer_info(
                data + offset,
                sizeof(S),
                pybind11::format_descriptor<S>::format(),
                2,
                std::vector<size_t>({size, second_dimension_size}),
                strides,
                read_only)
                );
            }

        virtual void clear() = 0;

    private:
        DATA& m_data;
        bool m_in_manager;
    };

template <class DATA>
LocalDataAccess<DATA>::LocalDataAccess(DATA& data)
    : m_data(data), m_in_manager(false) {}

void export_HOOMDHostBuffer(pybind11::module &m);

#endif
