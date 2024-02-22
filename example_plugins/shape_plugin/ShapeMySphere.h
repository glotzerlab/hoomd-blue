// Copyright (c) 2009-2024 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

#pragma once

#include "hoomd/AABB.h"
#include "hoomd/BoxDim.h"
#include "hoomd/HOOMDMath.h"
#include "hoomd/VectorMath.h"
#include "hoomd/hpmc/HPMCMiscFunctions.h"
#include "hoomd/hpmc/Moves.h"
#include "hoomd/hpmc/OBB.h"
#include "hoomd/hpmc/ShapeSphere.h"

#include <sstream>

#include <stdexcept>

#ifdef __HIPCC__
#define DEVICE __device__
#define HOSTDEVICE __host__ __device__
#else
#define DEVICE
#define HOSTDEVICE
#include <pybind11/pybind11.h>
#endif

namespace hoomd
    {
namespace hpmc
    {

struct MySphereParams : ShapeParams
    {
    /// The radius of the sphere
    ShortReal radius;

    /// True when move statistics should not be counted
    bool ignore;

    /// True when the shape may be oriented
    bool isOriented;

#ifdef ENABLE_HIP
    /// Set CUDA memory hints
    void set_memory_hint() const { }
#endif

#ifndef __HIPCC__

    /// Default constructor
    MySphereParams() { }

    /// Construct from a Python dictionary
    MySphereParams(pybind11::dict v, bool managed = false)
        {
        ignore = v["ignore_statistics"].cast<bool>();
        radius = v["radius"].cast<ShortReal>();
        isOriented = v["orientable"].cast<bool>();
        }

    /// Convert parameters to a python dictionary
    pybind11::dict asDict()
        {
        pybind11::dict v;
        v["radius"] = radius;
        v["orientable"] = isOriented;
        v["ignore_statistics"] = ignore;
        return v;
        }

#endif
    } __attribute__((aligned(32)));

struct ShapeMySphere
    {
    /// Define the parameter type
    typedef MySphereParams param_type;

    /// Temporary storage for depletant insertion
    typedef struct
        {
        } depletion_storage_type;

    /// Construct a shape at a given orientation
    DEVICE ShapeMySphere(const quat<Scalar>& _orientation, const param_type& _params)
        : orientation(_orientation), params(_params)
        {
        }

    /// Check if the shape may be rotated
    DEVICE bool hasOrientation() const
        {
        return params.isOriented;
        }

    /// Check if this shape should be ignored in the move statistics
    DEVICE bool ignoreStatistics() const
        {
        return params.ignore;
        }

    /// Get the circumsphere diameter of the shape
    DEVICE ShortReal getCircumsphereDiameter() const
        {
        return params.radius * ShortReal(2.0);
        }

    /// Get the in-sphere radius of the shape
    DEVICE ShortReal getInsphereRadius() const
        {
        return params.radius;
        }

    /// Return the bounding box of the shape in world coordinates
    DEVICE hoomd::detail::AABB getAABB(const vec3<Scalar>& pos) const
        {
        return hoomd::detail::AABB(pos, params.radius);
        }

    /// Return a tight fitting OBB around the shape
    DEVICE detail::OBB getOBB(const vec3<Scalar>& pos) const
        {
        return detail::OBB(pos, params.radius);
        }

    /// Returns true if this shape splits the overlap check over several threads of a warp using
    /// threadIdx.x
    HOSTDEVICE static bool isParallel()
        {
        return false;
        }

    /// Returns true if the overlap check supports sweeping both shapes by a sphere of given radius
    HOSTDEVICE static bool supportsSweepRadius()
        {
        return true;
        }

    quat<Scalar> orientation; //!< Orientation of the sphere (unused)

    /// MySphere parameters
    const MySphereParams& params;
    };

//! MySphere-MySphere overlap
/*! \param r_ab Vector defining the position of shape b relative to shape a (r_b - r_a)
    \param a first shape
    \param b second shape
    \param err in/out variable incremented when error conditions occur in the overlap test
    \returns true when *a* and *b* overlap, and false when they are disjoint

    \ingroup shape
*/
template<>
DEVICE inline bool test_overlap<ShapeMySphere, ShapeMySphere>(const vec3<Scalar>& r_ab,
                                                              const ShapeMySphere& a,
                                                              const ShapeMySphere& b,
                                                              unsigned int& err)
    {
    vec3<ShortReal> dr(r_ab);

    ShortReal rsq = dot(dr, dr);

    ShortReal RaRb = a.params.radius + b.params.radius;
    if (rsq < RaRb * RaRb)
        {
        return true;
        }
    else
        {
        return false;
        }
    }

    } // end namespace hpmc
    } // end namespace hoomd

#undef DEVICE
#undef HOSTDEVICE
