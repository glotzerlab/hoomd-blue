#ifndef __PYTHON_LOCAL_DATA_ACCESS_H__
#define __PYTHON_LOCAL_DATA_ACCESS_H__

#include "GlobalArray.h"
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

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
        pybind11::array_t<S> getArray(
            // handle to operate on
            std::unique_ptr<ArrayHandle<T> >& handle,
            // function to use for getting array
            const U<T>& (DATA::*get_array_func)() const,
            // Whether to return ghost particles properties
            bool ghost = false,
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

            if (!handle)
                {
                auto mode = ghost ? access_mode::read : access_mode::readwrite;
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

            if (ghost)
                {
                _data += N;
                size = ghostN;
                }
            S* data = (S*)_data;

            if (strides.size() == 0 && second_dimension_size == 0)
                {
                return pybind11::array_t<S>(
                    pybind11::buffer_info(
                        data + offset,
                        sizeof(S),
                        pybind11::format_descriptor<S>::format(),
                        1,
                        std::vector<size_t>({size}),
                        std::vector<size_t>({sizeof(T)}),
                        !ghost));
                }
            else if (strides.size() == 0 && second_dimension_size != 0)
                {
                strides = std::vector<size_t>({sizeof(T), sizeof(S)});
                }
            return pybind11::array_t<S>(
                pybind11::buffer_info(
                    data + offset,
                    sizeof(S),
                    pybind11::format_descriptor<S>::format(),
                    2,
                    std::vector<size_t>({size, second_dimension_size}),
                    strides,
                    !ghost)
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

#endif
