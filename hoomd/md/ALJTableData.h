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

struct shape_table
    {
    DEVICE shape_table()
        : alpha(0.0), epsilon(0.0), sigma_i(0.0), sigma_j(0.0), ki_max(0.0), kj_max(0.0), Ni(0), Nj(0)
        {}

    #ifndef NVCC
    //! Shape constructor
    shape_table(pybind11::list shape_i, pybind11::list shape_j, bool use_device)
        : alpha(0.0), epsilon(0.0), sigma_i(0.0), sigma_j(0.0), ki_max(0.0), kj_max(0.0), Ni(0), Nj(0)
        {
        //! Construct table for particle i
        unsigned int Ni = len(shape_i);
        verts_i = ManagedArray<vec3<Scalar> >(Ni, use_device);
        for (unsigned int i = 0; i < Ni; ++i)
            {
            verts_i[i] = vec3<Scalar>();
            }

        //! Construct table for particle j
        unsigned int Nj = len(shape_j);
        verts_j = ManagedArray<vec3<Scalar> >(Nj, use_device);
        for (unsigned int i = 0; i < Nj; ++i)
            {
            verts_j[i] = vec3<Scalar>();
            }
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
    Scalar alpha;                        //! toggle switch fo attractive branch of potential
    Scalar epsilon;                      //! interaction parameter
    Scalar sigma_i;                      //! size of i^th particle
    Scalar sigma_j;                      //! size of j^th particle
    Scalar ki_max;
    Scalar kj_max;
    unsigned int Ni;                           //! number of vertices i^th particle
    unsigned int Nj;                           //! number of vertices j^th particle
    };

//! Helper function to build shape structure from python
#ifndef NVCC
shape_table make_shape_table(Scalar epsilon, Scalar sigma_i, Scalar sigma_j, Scalar alpha, pybind11::list shape_i, pybind11::list shape_j, std::shared_ptr<const ExecutionConfiguration> exec_conf)
    {
    shape_table result(shape_i, shape_j, exec_conf->isCUDAEnabled());
    result.epsilon = epsilon;
    result.alpha = alpha;
    result.sigma_i = sigma_i;
    result.sigma_j = sigma_j;

    ///////////////////////////////////////////
    /// Define parameters for i^th particle ///
    ///////////////////////////////////////////

    //! Length of vertices list
    unsigned int Ni = len(shape_i);
    //! Extract omega from python list for i^th particle
    Scalar kmax = 10000.0;
    Scalar ktest = 100.0;  
    for (unsigned int i = 0; i < Ni; i++)
        {
        pybind11::list shape_tmp = pybind11::cast<pybind11::list>(shape_i[i]);
        result.verts_i[i] = vec3<Scalar>(pybind11::cast<Scalar>(shape_tmp[0]), pybind11::cast<Scalar>(shape_tmp[1]), pybind11::cast<Scalar>(shape_tmp[2]));
        // Calculate kmax on the fly
        ktest = result.verts_i[i].x*result.verts_i[i].x + result.verts_i[i].y*result.verts_i[i].y + result.verts_i[i].z*result.verts_i[i].z;
        if (ktest < kmax)
            {
            kmax = ktest;
            }
        }
    result.Ni = len(shape_i);
    result.ki_max = sqrt(kmax);

    ///////////////////////////////////////////
    /// Define parameters for j^th particle ///
    ///////////////////////////////////////////

    //! Length of vertices list
    unsigned int Nj = len(shape_j);
    //! Extract omega from python list for i^th particle
    kmax = 10000.0;
    ktest = 100.0;  
    for (unsigned int i = 0; i < Nj; i++)
        {
        pybind11::list shape_tmp = pybind11::cast<pybind11::list>(shape_j[i]);
        result.verts_j[i] = vec3<Scalar>(pybind11::cast<Scalar>(shape_tmp[0]), pybind11::cast<Scalar>(shape_tmp[1]), pybind11::cast<Scalar>(shape_tmp[2]));
        // Calculate kmax on the fly
        ktest = result.verts_j[i].x*result.verts_j[i].x + result.verts_j[i].y*result.verts_j[i].y + result.verts_j[i].z*result.verts_j[i].z;
        if (ktest < kmax)
            {
            kmax = ktest;
            }
        }
    result.Nj = len(shape_j);
    result.kj_max = sqrt(kmax);
    ///////////////////////////////////////////
    ///////////////////////////////////////////     

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
