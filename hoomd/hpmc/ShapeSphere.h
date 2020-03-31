// Copyright (c) 2009-2019 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.

#include "hoomd/HOOMDMath.h"
#include "hoomd/BoxDim.h"
#include "hoomd/Hypersphere.h"
#include "HPMCPrecisionSetup.h"
#include "hoomd/VectorMath.h"
#include "Moves.h"
#include "hoomd/AABB.h"
#include <sstream>

#include <stdexcept>

#ifndef __SHAPE_SPHERE_H__
#define __SHAPE_SPHERE_H__

/*! \file ShapeSphere.h
    \brief Defines the sphere shape
*/

// need to declare these class methods with __device__ qualifiers when building in nvcc
// DEVICE is __device__ when included in nvcc and blank when included into the host compiler
#ifdef NVCC
#define DEVICE __device__
#define HOSTDEVICE __host__ __device__
#else
#define DEVICE
#define HOSTDEVICE
#endif

#define SMALL 1e-5

namespace hpmc
{

// put a few misc math functions here as they don't have any better home
namespace detail
    {
    //! Compute the bounding sphere of a hypersphere cap on the 3-sphere
    /*! \param R_circumsphere radius of circum-sphere, defined on 3-sphere
        \param R radius of sphere

        The sphere radius that we compute takes into account the curvature of the supporting hypersphere, which results in a more
        optimal bounding sphere in 4d for large curvature
     */
    template <class Real>
    HOSTDEVICE Real get_bounding_sphere_radius_4d(const Real R_circumsphere, const Real R)
        {
        return R_circumsphere;

        #if 0
        // doesn't seem to work in 4d
        // rotate the bounding sphere center by an arc of length R_circumsphere, around the x-axis
        Real phi = R_circumsphere/R;

        // the transformation quaternion
        quat<Real> p(fast::cos(Real(0.5)*phi),fast::sin(Real(0.5)*phi)*vec3<Real>(1,0,0));

        // apply the translation to the standard position
        quat<Real> v0 = quat<Real>(0,vec3<Real>(0,0,R));
        quat<Real> v1;
        v1 = p*v0*p;

        // v1 is the outermost extent of the hypersphere cap in 4d space
        // fit a 3-sphere of radius R around the surface center v0 containing v1
        quat<Real> dr(v1.s-v0.s, v1.v-v0.v);

        return fast::sqrt(norm2(dr));
        #endif
        }

        //! Compute the arc-length between two positions on the hypersphere
        /* \param quat_l_a left quaternion of position a
           \param quat_r_a right quaternion of position a
           \param quat_l_b left quaternion of position b
           \param quat_r_b left quaternion of position b
           \param sphere Bounding sphere

           \param returns the arc-length
         */
        HOSTDEVICE inline OverlapReal get_arclength_hypersphere(const quat<Scalar>& quat_l_a, const quat<Scalar>& quat_r_a,
            const quat<Scalar>& quat_l_b, const quat<Scalar>& quat_r_b, const Hypersphere& hypersphere)
            {
            // transform spherical coordinates into 4d-cartesian ones
            quat<OverlapReal> pos_a = hypersphere.hypersphericalToCartesian(quat<OverlapReal>(quat_l_a),quat<OverlapReal>(quat_r_a));
            quat<OverlapReal> pos_b = hypersphere.hypersphericalToCartesian(quat<OverlapReal>(quat_l_b),quat<OverlapReal>(quat_r_b));

            // normalize
            OverlapReal inv_norm_a = fast::rsqrt(dot(pos_a,pos_a));
            OverlapReal inv_norm_b = fast::rsqrt(dot(pos_b,pos_b));

            OverlapReal arg = dot(pos_a,pos_b)*inv_norm_a*inv_norm_b;

            // numerical robustness
            OverlapReal arc_length;
            if (arg >= OverlapReal(1.0))
                arc_length = OverlapReal(0.0);
            else if (arg <= OverlapReal(-1.0))
                arc_length = OverlapReal(M_PI)*hypersphere.getR(); 
            else
                arc_length = hypersphere.getR()*fast::acos(dot(pos_a,pos_b)*inv_norm_a*inv_norm_b);

	    return arc_length;
            }

    //! helper to call CPU or GPU signbit
    template <class T> HOSTDEVICE inline int signbit(const T& a)
        {
        #ifdef __CUDA_ARCH__
        return ::signbit(a);
        #else
        return std::signbit(a);
        #endif
        }

    template <class T> HOSTDEVICE inline T min(const T& a, const T& b)
        {
        #ifdef __CUDA_ARCH__
        return ::min(a,b);
        #else
        return std::min(a,b);
        #endif
        }

    template <class T> HOSTDEVICE inline T max(const T& a, const T& b)
        {
        #ifdef __CUDA_ARCH__
        return ::max(a,b);
        #else
        return std::max(a,b);
        #endif
        }

    template<class T> HOSTDEVICE inline void swap(T& a, T&b)
        {
        T c;
        c = a;
        a = b;
        b = c;
        }
    }

//! Base class for parameter structure data types
struct param_base
    {
    //! Custom new operator
    static void* operator new(std::size_t sz)
        {
        void *ret = 0;
        int retval = posix_memalign(&ret, 32, sz);
        if (retval != 0)
            {
            throw std::runtime_error("Error allocating aligned memory");
            }

        return ret;
        }

    //! Custom new operator for arrays
    static void* operator new[](std::size_t sz)
        {
        void *ret = 0;
        int retval = posix_memalign(&ret, 32, sz);
        if (retval != 0)
            {
            throw std::runtime_error("Error allocating aligned memory");
            }

        return ret;
        }

    //! Custom delete operator
    static void operator delete(void *ptr)
        {
        free(ptr);
        }

    //! Custom delete operator for arrays
    static void operator delete[](void *ptr)
        {
        free(ptr);
        }

    //! Load dynamic data members into shared memory and increase pointer
    /*! \param ptr Pointer to load data to (will be incremented)
        \param available_bytes Size of remaining shared memory allocation
     */
    HOSTDEVICE void load_shared(char *& ptr,unsigned int &available_bytes) const
        {
        // default implementation does nothing
        }
    };


//! Sphere shape template
/*! ShapeSphere implements IntegratorHPMC's shape protocol. It serves at the simplest example of a shape for HPMC

    The parameter defining a sphere is just a single Scalar, the sphere radius.

    \ingroup shape
*/
struct sph_params : param_base
    {
    OverlapReal radius;                 //!< radius of sphere
    unsigned int ignore;                //!< Bitwise ignore flag for stats, overlaps. 1 will ignore, 0 will not ignore
                                        //   First bit is ignore overlaps, Second bit is ignore statistics
    bool isOriented;                    //!< Flag to specify whether a sphere has orientation or not. Intended for
                                        //!  for use with anisotropic/patchy pair potentials.

    #ifdef ENABLE_CUDA
    //! Attach managed memory to CUDA stream
    void attach_to_stream(cudaStream_t stream) const
        {
        // default implementation does nothing
        }
    #endif
    } __attribute__((aligned(32)));

struct ShapeSphere
    {
    //! Define the parameter type
    typedef sph_params param_type;

    //! Initialize a shape at a given position
    DEVICE ShapeSphere(const quat<Scalar>& _orientation, const param_type& _params)
        : orientation(_orientation), params(_params) {}

    //! Initialize a shape with a given left and right quaternion (hyperspherical coordinates)
    DEVICE ShapeSphere(const quat<Scalar>& _quat_l, const quat<Scalar>& _quat_r, const param_type& _params)
        : quat_l(_quat_l), quat_r(_quat_r), params(_params) {}


    //! Does this shape have an orientation
    DEVICE bool hasOrientation() const
        {
        return params.isOriented;
        }

    //! Ignore flag for acceptance statistics
    DEVICE bool ignoreStatistics() const { return params.ignore; }

    //! Get the circumsphere diameter
    DEVICE OverlapReal getCircumsphereDiameter() const
        {
        return params.radius*OverlapReal(2.0);
        }

    //! Get the in-sphere radius
    DEVICE OverlapReal getInsphereRadius() const
        {
        return params.radius;
        }

    //! Return the bounding box of the shape in world coordinates
    DEVICE detail::AABB getAABB(const vec3<Scalar>& pos) const
        {
        return detail::AABB(pos, params.radius);
        }

    //! Return the bounding box of the shape, defined on the hyperhypersphere, in world coordinates
    DEVICE detail::AABB getAABBHypersphere(const Hypersphere& hypersphere)
        {
        return detail::AABB(hypersphere.hypersphericalToCartesian(quat_l, quat_r),
            detail::get_bounding_sphere_radius_4d((Scalar)params.radius, hypersphere.getR()));
        }

    #ifndef NVCC
    std::string getShapeSpec() const
        {
        std::ostringstream shapedef;
        shapedef << "{\"type\": \"Sphere\", \"diameter\": " << params.radius*OverlapReal(2.0) << "}";
        return shapedef.str();
        }
    #endif

    //! Returns true if this shape splits the overlap check over several threads of a warp using threadIdx.x
    HOSTDEVICE static bool isParallel() { return false; }

    quat<Scalar> orientation;    //!< Orientation of the sphere (unused)
    quat<Scalar> quat_l;         //!< Left quaternion (for hyperspherical coordinates)
    quat<Scalar> quat_r;         //!< Left quaternion (for hyperspherical coordinates)

    const sph_params &params;        //!< Sphere and ignore flags
    };

//! Check if circumspheres overlap (cartesian)
/*! \param r_ab Vector defining the position of shape b relative to shape a (r_b - r_a)
    \param a first shape
    \param b second shape
    \returns true if the circumspheres of both shapes overlap

    \ingroup shape
*/
DEVICE inline bool check_circumsphere_overlap(const vec3<Scalar>& r_ab, const ShapeSphere& a,
    const ShapeSphere &b)
    {
    // for now, always return true
    return true;
    }

//! Check if circumspheres overlap (hyperspherical coordinates)
/*! \param a first shape
    \param b second shape
    \returns true if the circumspheres of both shapes overlap

    \ingroup shape
*/
template<class Shape>
DEVICE inline bool check_circumsphere_overlap_hypersphere(const Shape& a, const Shape &b, const Hypersphere& hypersphere)
    {
    // default implementation returns true, other shapes will have to implement this for broad-phase
    return true;
    }

//! Check if circumspheres overlap (hyperspherical coordinates)
/*! \param a first shape
    \param b second shape
    \returns true if the circumspheres of both shapes overlap

    \ingroup shape
*/
template<>
DEVICE inline bool check_circumsphere_overlap_hypersphere(const ShapeSphere& a, const ShapeSphere &b, const Hypersphere& hypersphere)
    {
    // for now, always return true
    return true;
    }

//! Define the general overlap function (cartesian)
/*! This is just a convenient spot to put this to make sure it is defined early
    \param r_ab Vector defining the position of shape b relative to shape a (r_b - r_a)
    \param a first shape
    \param b second shape
    \param err Incremented if there is an error condition. Left unchanged otherwise.
    \returns true when *a* and *b* overlap, and false when they are disjoint
*/
template <class ShapeA, class ShapeB>
DEVICE inline bool test_overlap(const vec3<Scalar>& r_ab, const ShapeA &a, const ShapeB& b, unsigned int& err)
    {
    // default implementation returns true, will make it obvious if something calls this
    return true;
    }

//! Returns true if the shape overlaps with itself
template<class Shape>
DEVICE inline bool test_self_overlap_hypersphere(const Shape& shape, const Hypersphere& hypersphere)
    {
    // default implementation returns true, will make it obvious if something calls this
    return true;
    }

//! Define the general overlap function (hyperspherical version)
/*! This is just a convenient spot to put this to make sure it is defined early
    \param a first shape
    \param b second shape
    \param hypersphere Boundary conditions
    \param err Incremented if there is an error condition. Left unchanged otherwise.
    \returns true when *a* and *b* overlap, and false when they are disjoint
*/
template <class ShapeA, class ShapeB>
DEVICE inline bool test_overlap_hypersphere(const ShapeA& a, const ShapeB& b, const Hypersphere& hypersphere, unsigned int& err)
    {
    // default implementation returns true, will make it obvious if something calls this
    return true;
    }

//! Sphere-Sphere overlap
/*! \param r_ab Vector defining the position of shape b relative to shape a (r_b - r_a)
    \param a first shape
    \param b second shape
    \param err in/out variable incremented when error conditions occur in the overlap test
    \returns true when *a* and *b* overlap, and false when they are disjoint

    \ingroup shape
*/
template <>
DEVICE inline bool test_overlap<ShapeSphere, ShapeSphere>(const vec3<Scalar>& r_ab, const ShapeSphere& a, const ShapeSphere& b, unsigned int& err)
    {
    vec3<OverlapReal> dr(r_ab);

    OverlapReal rsq = dot(dr,dr);

    if (rsq < (a.params.radius + b.params.radius)*(a.params.radius + b.params.radius))
        {
        return true;
        }
    else
        {
        return false;
        }
    }

//! Returns true if the shape overlaps with itself
DEVICE inline bool test_self_overlap_hypersphere(const ShapeSphere& shape, const Hypersphere& hypersphere)
    {
    return shape.params.radius >= Scalar(M_PI)*hypersphere.getR();
    }

//! Sphere-Sphere overlap on a hyperhypersphere
/*!  \param a first shape
    \param b second shape
    \param hypersphere Boundary conditions
    \param err in/out variable incremented when error conditions occur in the overlap test
    \returns true when *a* and *b* overlap, and false when they are disjoint

    \ingroup shape
*/
template <>
DEVICE inline bool test_overlap_hypersphere<ShapeSphere, ShapeSphere>(const ShapeSphere& a, const ShapeSphere& b,
    const Hypersphere& hypersphere, unsigned int& err)
    {
    // arc-length along a geodesic
    OverlapReal arc_length = detail::get_arclength_hypersphere(a.quat_l,a.quat_r,b.quat_l,b.quat_r, hypersphere);

    if (arc_length < (a.params.radius + b.params.radius))
        {
        return true;
        }
    else
        {
        return false;
        }
    }

}; // end namespace hpmc

#undef DEVICE
#undef HOSTDEVICE
#endif //__SHAPE_SPHERE_H__
