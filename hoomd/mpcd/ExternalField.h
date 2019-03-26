// Copyright (c) 2009-2019 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.

// Maintainer: mphoward

/*!
 * \file mpcd/ExternalField.h
 * \brief Definition of mpcd::ExternalField.
 */

#ifndef MPCD_EXTERNAL_FIELD_H_
#define MPCD_EXTERNAL_FIELD_H_

#include "hoomd/HOOMDMath.h"

#ifdef NVCC
#define HOSTDEVICE __host__ __device__
#else
#define HOSTDEVICE
#include "hoomd/extern/pybind/include/pybind11/pybind11.h"
#endif

namespace mpcd
{

class ExternalField
    {
    public:
        HOSTDEVICE virtual ~ExternalField() {}
        HOSTDEVICE virtual Scalar3 evaluate(const Scalar3& r) const = 0;
    };

class ConstantForce : public ExternalField
    {
    public:
        HOSTDEVICE ConstantForce(Scalar3 field) : m_field(field) {}

        HOSTDEVICE virtual Scalar3 evaluate(const Scalar3& r) const override
            {
            return m_field;
            }

    private:
        Scalar3 m_field;
    };

#ifndef NVCC
namespace detail
{
void export_ExternalFieldPolymorph(pybind11::module& m);
} // end namespace detail
#endif // NVCC

} // end namespace mpcd

#undef HOSTDEVICE

#endif // MPCD_EXTERNAL_FIELD_H_
