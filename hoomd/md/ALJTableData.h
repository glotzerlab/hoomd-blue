// Copyright (c) 2009-2018 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.

#ifndef __ALJ_TABLE_DATA_H__
#define __ALJ_TABLE_DATA_H__

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
        : epsilon(0.0), sigma_i(0.0), sigma_j(0.0), alpha(0.0), ki_max(0.0), kj_max(0.0), Ni(0), Nj(0)
        {}

    #ifndef NVCC
    //! Shape constructor
    shape_table(Scalar _epsilon, Scalar _sigma_i, Scalar _sigma_j, Scalar _alpha, pybind11::list shape_i, pybind11::list shape_j, bool use_device)
        : epsilon(_epsilon), sigma_i(_sigma_i), sigma_j(_sigma_j), alpha(_alpha), ki_max(0.0), kj_max(0.0), Ni(0), Nj(0)
        {
        Scalar kmax = SCALAR_MAX;

        //! Construct table for particle i
        Ni = len(shape_i);
        verts_i = ManagedArray<vec3<Scalar> >(Ni, use_device);
        for (unsigned int i = 0; i < Ni; ++i)
            {
            pybind11::list shape_tmp = pybind11::cast<pybind11::list>(shape_i[i]);
            verts_i[i] = vec3<Scalar>(pybind11::cast<Scalar>(shape_tmp[0]), pybind11::cast<Scalar>(shape_tmp[1]), pybind11::cast<Scalar>(shape_tmp[2]));

            Scalar ktest = dot(verts_i[i], verts_i[i]);
            if (ktest < kmax)
                {
                kmax = ktest;
                }
            }
        ki_max = sqrt(kmax);

        kmax = SCALAR_MAX;

        //! Construct table for particle j
        Nj = len(shape_j);
        verts_j = ManagedArray<vec3<Scalar> >(Nj, use_device);
        for (unsigned int i = 0; i < Nj; ++i)
            {
            pybind11::list shape_tmp = pybind11::cast<pybind11::list>(shape_j[i]);
            verts_j[i] = vec3<Scalar>(pybind11::cast<Scalar>(shape_tmp[0]), pybind11::cast<Scalar>(shape_tmp[1]), pybind11::cast<Scalar>(shape_tmp[2]));

            Scalar ktest = dot(verts_j[i], verts_j[i]);
            if (ktest < kmax)
                {
                kmax = ktest;
                }
            }
        kj_max = sqrt(kmax);
        }

    #endif

    //! Load dynamic data members into shared memory and increase pointer
    /*! \param ptr Pointer to load data to (will be incremented)
        \param available_bytes Size of remaining shared memory allocation
     */
    HOSTDEVICE void load_shared(char *& ptr, unsigned int &available_bytes) const
        {
        verts_i.load_shared(ptr, available_bytes);
        verts_j.load_shared(ptr, available_bytes);
        }

    #ifdef ENABLE_CUDA
    //! Attach managed memory to CUDA stream
    void attach_to_stream(cudaStream_t stream) const
        {
        verts_i.attach_to_stream(stream);
        verts_j.attach_to_stream(stream);
        }
    #endif

    //! Shape parameters particle i^th
    ManagedArray<vec3<Scalar> > verts_i;          //! Vertices of shape i.
    ManagedArray<vec3<Scalar> > verts_j;          //! Vertices of shape j.
    //! Potential parameters
    Scalar epsilon;                      //! interaction parameter
    Scalar sigma_i;                      //! size of i^th particle
    Scalar sigma_j;                      //! size of j^th particle
    Scalar alpha;                        //! toggle switch fo attractive branch of potential
    Scalar ki_max;                       //! largest kernel value for shape i
    Scalar kj_max;                       //! largest kernel value for shape j
    unsigned int Ni;                           //! number of vertices i^th particle
    unsigned int Nj;                           //! number of vertices j^th particle
    };


//! Helper function to build shape structure from python
#ifndef NVCC
shape_table make_shape_table(Scalar epsilon, Scalar sigma_i, Scalar sigma_j, Scalar alpha, pybind11::list shape_i, pybind11::list shape_j, std::shared_ptr<const ExecutionConfiguration> exec_conf)
    {
    shape_table result(epsilon, sigma_i, sigma_j, alpha, shape_i, shape_j, exec_conf->isCUDAEnabled());
    return result;
    }

//! Function to export the LJ parameter type to python

inline void export_shape_params(pybind11::module& m)
{
    pybind11::class_<shape_table>(m, "shape_table")
        .def(pybind11::init<>())
        .def_readwrite("alpha", &shape_table::alpha)
        .def_readwrite("epsilon", &shape_table::epsilon)
        .def_readwrite("sigma_i", &shape_table::sigma_i)
        .def_readwrite("sigma_j", &shape_table::sigma_j)
        .def_readwrite("ki_max", &shape_table::ki_max)
        .def_readwrite("kj_max", &shape_table::kj_max)
        .def_readwrite("Ni", &shape_table::Ni)
        .def_readwrite("Nj", &shape_table::Nj);

    m.def("make_shape_table", &make_shape_table);
}
#endif
#endif // end __ALJ_TABLE_DATA_H__
