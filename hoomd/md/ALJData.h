// Copyright (c) 2009-2018 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.

#ifndef __ALJ_DATA_H__
#define __ALJ_DATA_H__

#include "hoomd/ManagedArray.h"
#include "hoomd/VectorMath.h"
#include <iostream>
#include <algorithm> 
#include <vector>

#ifndef NVCC
#include <hoomd/extern/pybind/include/pybind11/pybind11.h>
#include <hoomd/extern/pybind/include/pybind11/stl.h>
#endif

#ifdef SINGLE_PRECISION
//! Max floating point type (single precision)
#define SCALAR_MAX FLT_MAX;
#else
//! Max floating point type (double precision)
#define SCALAR_MAX DBL_MAX;
#endif

struct shape_table
    {
    DEVICE shape_table()
        : epsilon(0.0), sigma_i(0.0), sigma_j(0.0), alpha(0.0)
        {}

    #ifndef NVCC
    //! Shape constructor
    shape_table(Scalar _epsilon, Scalar _sigma_i, Scalar _sigma_j, Scalar _alpha, bool use_device)
        : epsilon(_epsilon), sigma_i(_sigma_i), sigma_j(_sigma_j), alpha(_alpha) {}

    #endif

    //! Load dynamic data members into shared memory and increase pointer
    /*! \param ptr Pointer to load data to (will be incremented)
        \param available_bytes Size of remaining shared memory allocation
     */
    HOSTDEVICE void load_shared(char *& ptr, unsigned int &available_bytes) const {}

    #ifdef ENABLE_CUDA
    //! Attach managed memory to CUDA stream
    void attach_to_stream(cudaStream_t stream) const {}
    #endif

    //! Potential parameters
    Scalar epsilon;                      //! interaction parameter.
    Scalar sigma_i;                      //! size of i^th particle.
    Scalar sigma_j;                      //! size of j^th particle.
    Scalar alpha;                        //! toggle switch of attractive branch of potential.
    };


//! Helper function to build shape structure from python
#ifndef NVCC
shape_table make_shape_table(Scalar epsilon, Scalar sigma_i, Scalar sigma_j, Scalar alpha, std::shared_ptr<const ExecutionConfiguration> exec_conf)
    {
    shape_table result(epsilon, sigma_i, sigma_j, alpha, exec_conf->isCUDAEnabled());
    return result;
    }

//! Function to export the ALJ parameter type to python
inline void export_shape_params(pybind11::module& m)
{
    pybind11::class_<shape_table>(m, "shape_table")
        .def(pybind11::init<>())
        .def_readwrite("alpha", &shape_table::alpha)
        .def_readwrite("epsilon", &shape_table::epsilon)
        .def_readwrite("sigma_i", &shape_table::sigma_i)
        .def_readwrite("sigma_j", &shape_table::sigma_j);

    m.def("make_shape_table", &make_shape_table);
}
#endif
#endif // end __ALJ_DATA_H__
