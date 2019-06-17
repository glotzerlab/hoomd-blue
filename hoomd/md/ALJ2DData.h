// Copyright (c) 2009-2018 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.

#ifndef __ALJ_2D_DATA_H__
#define __ALJ_2D_DATA_H__

#include "EvaluatorPair2DALJ.h"
#include "hoomd/ManagedArray.h"
#include "hoomd/VectorMath.h"
#include <iostream>
#include <algorithm> 
#include <vector>

#ifndef NVCC
#include <hoomd/extern/pybind/include/pybind11/pybind11.h>
#include <hoomd/extern/pybind/include/pybind11/stl.h>
#endif

#include <hoomd/extern/pybind/include/pybind11/pybind11.h>
#include <hoomd/extern/pybind/include/pybind11/stl.h>

struct shape_2D
    {
    DEVICE shape_2D()
        : alpha(0.0),
          epsilon(0.0),
          sigma_i(0.0),
          sigma_j(0.0),          
          ki_max(0.0),
          kj_max(0.0),
          Ni(0.0),
          Nj(0.0)
        { }

    #ifndef NVCC
    //! Shape constructor
    shape_2D(pybind11::list shape_i, pybind11::list shape_j)
        : alpha(0.0), epsilon(0.0), sigma_i(0.0), sigma_j(0.0), ki_max(0.0), kj_max(0.0), Ni(0.0), Nj(0.0)
        {
        //! Construct table for particle i
        unsigned int Ni = len(shape_i)+1;
        xi  = ManagedArray<float>(Ni,1);
        yi  = ManagedArray<float>(Ni,1);
        for (unsigned int i = 0; i < Ni; ++i)
            {
            xi[i] = float(0.0);
            yi[i] = float(0.0);
            }

        //! Construct table for particle j
        unsigned int Nj = len(shape_j)+1;
        xj  = ManagedArray<float>(Nj,1);
        yj  = ManagedArray<float>(Nj,1);
        for (unsigned int i = 0; i < Nj; ++i)
            {
            xj[i] = float(0.0);
            yj[i] = float(0.0);
            }
        }
    #endif

    //! Load dynamic data members into shared memory and increase pointer
    /*! \param ptr Pointer to load data to (will be incremented)
        \param available_bytes Size of remaining shared memory allocation
     */
    HOSTDEVICE void load_shared(char *& ptr, unsigned int &available_bytes) const
        {
        xi.load_shared(ptr,available_bytes);
        yi.load_shared(ptr,available_bytes);
        xj.load_shared(ptr,available_bytes);
        yj.load_shared(ptr,available_bytes);
        }

    #ifdef ENABLE_CUDA
    //! Attach managed memory to CUDA stream
    void attach_to_stream(cudaStream_t stream) const
        {
        xi.attach_to_stream(stream);
        yi.attach_to_stream(stream);
        xj.attach_to_stream(stream);
        yj.attach_to_stream(stream);
        }
    #endif

    // #ifdef ENABLE_CUDA
    // //! Shape constructor
    // shape_table(pybind11::list shape_i)
    //     : alpha(0.0), epsilon(0.0), sigma_i(0.0), sigma_j(0.0), dphi(0.0), dtheta(0.0), theta_init(0.0), phi_init(0.0), Ntheta(0.0)
    //     {
    //     //! Construct table
    //     unsigned int Ni = len(shape_i);
    //     omegai  = ManagedArray<float>(Ni, 1);
    //     for (unsigned int i = 0; i < Ni; ++i)
    //         {
    //         omegai[i] = float(0.0);
    //         }
    //     }
    // #endif

    // // //! Load dynamic data members into shared memory and increase pointer
    // // /*! \param ptr Pointer to load data to (will be incremented)
    // //     \param available_bytes Size of remaining shared memory allocation
    // //  */
    // // HOSTDEVICE void load_shared(char *& ptr, unsigned int &available_bytes) const
    // //     {
    // //     ddthetai.load_shared(ptr,available_bytes);
    // //     ddthetaj.load_shared(ptr,available_bytes);
    // //     ddphii.load_shared(ptr,available_bytes);
    // //     ddphij.load_shared(ptr,available_bytes);
    // //     ddomegai.load_shared(ptr,available_bytes);
    // //     ddomegaj.load_shared(ptr,available_bytes);
    // //     }

    //! Shape parameters particles
    ManagedArray<float> xi;          //! omega coordinates of kernel
    ManagedArray<float> yi;          //! omega coordinates of kernel
    ManagedArray<float> xj;          //! omega coordinates of kernel
    ManagedArray<float> yj;          //! omega coordinates of kernel
    //! Potential parameters
    float alpha;                        //! toggle switch fo attractive branch of potential
    float epsilon;                      //! interaction parameter
    float sigma_i;                      //! size of i^th particle
    float sigma_j;                      //! size of j^th particle
    float ki_max;
    float kj_max;
    float Ni;                           //! number of vertices i^th particle
    float Nj;                           //! number of vertices j^th particle
    };

//! Helper function to build shape structure from python
#ifndef NVCC
shape_2D make_shape_2D(float epsilon, float sigma_i, float sigma_j, float alpha, pybind11::list shape_i, pybind11::list shape_j)
    {

    shape_2D result(shape_i, shape_j);
    result.epsilon = epsilon;
    result.alpha = alpha;
    result.sigma_i = sigma_i;
    result.sigma_j = sigma_j;

    ///////////////////////////////////////////
    /// Define parameters for i^th particle ///
    ///////////////////////////////////////////

    //! Length of vertices list
    int Ni = len(shape_i);
    //! Extract omega from python list for i^th particle
    Scalar kmax = 10000.0;
    Scalar ktest = 100.0;  
    for (unsigned int i = 0; i < Ni; i++)
        {
        pybind11::list shape_tmp = pybind11::cast<pybind11::list>(shape_i[i]);
        vec2<float> vert = vec2<float>(pybind11::cast<float>(shape_tmp[0]), pybind11::cast<float>(shape_tmp[1]));
        result.xi[i] = vert.x;
        result.yi[i] = vert.y;
        // Calculate kmax on the fly
        ktest = vert.x*vert.x + vert.y*vert.y;
        if (ktest < kmax)
            {
            kmax = ktest;
            }
        }
    // Loop back to first value
    pybind11::list shape_tmp = pybind11::cast<pybind11::list>(shape_i[0]);
    vec2<float> vert = vec2<float>(pybind11::cast<float>(shape_tmp[0]), pybind11::cast<float>(shape_tmp[1]));
    result.xi[Ni] = vert.x;
    result.yi[Ni] = vert.y;
    result.Ni = len(shape_i) + 1;
    result.ki_max = kmax;
    ///////////////////////////////////////////
    ///////////////////////////////////////////

    ///////////////////////////////////////////
    /// Define parameters for j^th particle ///
    ///////////////////////////////////////////

    //! Length of vertices list
    int Nj = len(shape_j);
    //! Extract omega from python list for i^th particle
    kmax = 10000.0;
    ktest = 100.0;
    for (unsigned int i = 0; i < Nj; i++)
        {
        pybind11::list shape_tmp = pybind11::cast<pybind11::list>(shape_j[i]);
        vec2<float> vert = vec2<float>(pybind11::cast<float>(shape_tmp[0]), pybind11::cast<float>(shape_tmp[1]));
        result.xj[i] = vert.x;
        result.yj[i] = vert.y;
        // Calculate kmax on the fly
        ktest = vert.x*vert.x + vert.y*vert.y;
        if (ktest < kmax)
            {
            kmax = ktest;
            }
        }
    // Loop back to first value
    shape_tmp = pybind11::cast<pybind11::list>(shape_j[0]);
    vert = vec2<float>(pybind11::cast<float>(shape_tmp[0]), pybind11::cast<float>(shape_tmp[1]));
    result.xj[Nj] = vert.x;
    result.yj[Nj] = vert.y;
    result.Nj = len(shape_j) + 1;
    result.kj_max = kmax;

    return result;
    }
    ///////////////////////////////////////////
    ///////////////////////////////////////////
    

//! Function to export the LJ parameter type to python

inline void export_shape_params2D(pybind11::module& m)
{
    pybind11::class_<shape_2D>(m, "shape_2D")
        .def(pybind11::init<>())
        .def_readwrite("alpha", &shape_2D::alpha)
        .def_readwrite("epsilon", &shape_2D::epsilon)
        .def_readwrite("sigma_i", &shape_2D::sigma_i)
        .def_readwrite("sigma_j", &shape_2D::sigma_j)
        .def_readwrite("ki_max", &shape_2D::ki_max)
        .def_readwrite("kj_max", &shape_2D::kj_max)
        .def_readwrite("Ni", &shape_2D::Ni)
        .def_readwrite("Nj", &shape_2D::Nj);

    m.def("make_shape_2D", &make_shape_2D);
}
#endif
#endif // end __ALJ_2D_DATA_H__
