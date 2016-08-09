// Copyright (c) 2009-2016 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.

#include "hoomd/HOOMDMath.h"
#include "hoomd/BoxDim.h"
#include "HPMCPrecisionSetup.h"
#include "hoomd/VectorMath.h"
#include "ShapeSphere.h"
#include "XenoCollide3D.h"
#include "ShapeConvexPolyhedron.h"
#include "hoomd/AABB.h"

#ifndef __SHAPE_FACETED_SPHERE_H__
#define __SHAPE_FACETED_SPHERE_H__

/*! \file ShapeFacetedSphere.h
    \brief Defines the sphere shape
*/

/*! The faceted sphere is defined by the intersection of a sphere of radius R with planes given by the face
 * normals n and offsets b, obeying the equation:
 *
 * 1) r.n + b <= 0
 * 2) |r| <= R
 *
 * Intersections of planes within the sphere radius are accounted for.
 *
 */
// need to declare these class methods with __device__ qualifiers when building in nvcc
// DEVICE is __device__ when included in nvcc and blank when included into the host compiler
#ifdef NVCC
#define DEVICE __device__
#else
#define DEVICE
#endif

namespace hpmc
{

namespace detail
{

//! maximum number of plane normal vectors that can be stored
/*! \ingroup hpmc_data_structs */
const unsigned int MAX_SPHERE_FACETS = 8;
const unsigned int MAX_FPOLY3D_VERTS = 24;

//! maximum number of intersection vertices between two planes and the sphere
/*! \ingroup hpmc_data_structs */
const unsigned int MAX_SPHERE_VERTICES = MAX_SPHERE_FACETS*(MAX_SPHERE_FACETS-1);


//! Data structure for intersection planes
/*! \ingroup hpmc_data_structs */
struct faceted_sphere_params : aligned_struct
    {
    poly3d_verts<MAX_FPOLY3D_VERTS> verts;           //!< Vertices of the polyhedron
    poly3d_verts<MAX_FPOLY3D_VERTS> additional_verts;//!< Vertices of the polyhedron edge-sphere intersection
    vec3<OverlapReal> n[MAX_SPHERE_FACETS];          //!< Normal vectors of planes
    OverlapReal offset[MAX_SPHERE_FACETS];           //!< Offset of every plane
    OverlapReal diameter;                            //!< Sphere diameter
    OverlapReal insphere_radius;                     //!< Precomputed radius of in-sphere
    vec3<OverlapReal> origin;                        //!< Origin shift
    unsigned int N;                                  //!< Number of cut planes
    unsigned int ignore;                             //!< Bitwise ignore flag for stats, overlaps. 1 will ignore, 0 will not ignore
                                                     //   First bit is ignore overlaps, Second bit is ignore statistics
    } __attribute__((aligned(32)));

#define SMALL 1e-5

//! Support function for ShapeFacetedSphere
/* \ingroup minkowski
*/
class SupportFuncFacetedSphere
    {
    public:
        //! Construct a support function for a faceted sphere
        /*! \param _params Parameters of the faceted sphere
        */
        DEVICE SupportFuncFacetedSphere(const faceted_sphere_params& _params)
            : params(_params)
            {
            }

        //! Compute the support function
        /*! \param n Normal vector input (in the local frame)
            \returns Local coords of the point furthest in the direction of n
        */
        DEVICE vec3<OverlapReal> operator() (const vec3<OverlapReal>& n) const
            {
            OverlapReal R(params.diameter/OverlapReal(2.0));
            OverlapReal nsq = dot(n,n);
            vec3<OverlapReal> max_vec = n*fast::rsqrt(nsq)*R;
            bool intersecting = false;
            bool valid = true;

            // iterate over intersecting planes
            for (unsigned int i = 0; i < params.N; i++)
                {
                const vec3<OverlapReal> &n_p = params.n[i];
                OverlapReal np_sq = dot(n_p,n_p);
                OverlapReal b = params.offset[i];

                // is current supporting vertex outside the half-space defined by this plane?
                if (dot(n_p,max_vec) + b >= OverlapReal(0.0))
                    {
                    // yes, compute supporting vertex on intersection boundary (circle)
                    // between plane and sphere
                    OverlapReal alpha = dot(n,n_p);
                    OverlapReal arg = (nsq-alpha*alpha/np_sq);
                    vec3<OverlapReal> v;
                    if (arg >= OverlapReal(SMALL)*nsq)
                        {
                        OverlapReal arg2 = R*R-b*b/np_sq;
                        OverlapReal invgamma = fast::sqrt(arg2/arg);

                        // Intersection vertex that maximizes support function
                        v = invgamma*(n-alpha/np_sq*n_p)-n_p*b/np_sq;
                        }
                    else
                        {
                        // degenerate case
                        v = -b*n_p/np_sq;
                        }

                    intersecting = true;

                    // is this vertex masked by another plane?
                    valid = true;
                    for (unsigned int j = 0; j < params.N; j++)
                        {
                        const vec3<OverlapReal> &np_2 = params.n[j];
                        OverlapReal b_2 = params.offset[j];

                        // is current supporting vertex outside the half-space defined by this plane?
                        if (dot(np_2,v) + b_2 >= OverlapReal(0.0) && j != i)
                            {
                            valid = false;
                            }
                        }

                    if (valid)
                        {
                        max_vec = v;
                        }
                    }
                }

            // do we have to take into account plane-plane intersection vertices?
            if (intersecting && params.additional_verts.N)
                {
                detail::SupportFuncConvexPolyhedron<MAX_FPOLY3D_VERTS> s(params.additional_verts);
                vec3<OverlapReal> v = s(n);

                if (!valid)
                    {
                    max_vec = v;
                    }
                if (params.verts.N)
                    {
                    // determine polyhedron support
                    detail::SupportFuncConvexPolyhedron<MAX_FPOLY3D_VERTS> t(params.verts);
                    vec3<OverlapReal> p  = t(n);

                    // does the shape intersect within the sphere?
                    if (dot(p,p) <= R*R && dot(p,n) > dot(max_vec,n))
                        {
                        max_vec = p;
                        }
                    }
                }

            return max_vec - params.origin;
            }

    private:
        const faceted_sphere_params& params;      //!< Definition of faceted sphere
    };



} // end namespace detail


//! Faceted sphere shape template
/*! ShapeFacetedSphere implements IntegragorHPMC's shape protocol for a sphere that is truncated
    by a set of planes, defined through their plane equations n_i*x = n_i^2.

    The parameter defining the sphere is just a single Scalar, the sphere radius.

    \ingroup shape
*/
struct ShapeFacetedSphere
    {
    //! Define the parameter type
    typedef detail::faceted_sphere_params param_type;

    //! Initialize a shape at a given position
    DEVICE ShapeFacetedSphere(const quat<Scalar>& _orientation, const param_type& _params)
        : orientation(_orientation), params(_params)
        { }

    //! Does this shape have an orientation
    DEVICE bool hasOrientation() { return params.N > 0; }

    //!Ignore flag for acceptance statistics
    DEVICE bool ignoreStatistics() const { return params.ignore; }

    //! Get the circumsphere diameter
    DEVICE OverlapReal getCircumsphereDiameter() const
        {
        return params.diameter;
        }

    //! Get the in-sphere radius
    DEVICE OverlapReal getInsphereRadius() const
        {
        return params.insphere_radius;
        }

    //! Return the bounding box of the shape in world coordinates
    DEVICE detail::AABB getAABB(const vec3<Scalar>& pos) const
        {
        return detail::AABB(pos, params.diameter/Scalar(2.0));
        }

    //! Returns true if this shape splits the overlap check over several threads of a warp using threadIdx.x
    HOSTDEVICE static bool isParallel() { return false; }

    /*!
     * Generate the intersections points of polyhedron edges with the sphere
     */
    DEVICE static void initializeVertices(param_type& _params)
        {
        #ifndef NVCC
        _params.additional_verts.diameter = _params.diameter;
        _params.additional_verts.N = 0;
        for (unsigned int i = 0; i < detail::MAX_FPOLY3D_VERTS; ++i)
            {
            _params.additional_verts.x[i] = OverlapReal(0.0);
            _params.additional_verts.y[i] = OverlapReal(0.0);
            _params.additional_verts.z[i] = OverlapReal(0.0);
            }

        OverlapReal R = (_params.diameter)/OverlapReal(2.0);

        // iterate over unique pairs of planes
        for (unsigned int i = 0; i < _params.N; ++i)
            {
            vec3<OverlapReal> n_p(_params.n[i]);
            OverlapReal b(_params.offset[i]);

            for (unsigned int j = i+1; j < _params.N; ++j)
                {
                vec3<OverlapReal> np2(_params.n[j]);
                OverlapReal b2 = _params.offset[j];
                OverlapReal np2_sq = dot(np2,np2);

                // determine intersection line between plane i and plane j

                // point on intersection line closest to center of sphere
                OverlapReal dotp = dot(np2,n_p);
                OverlapReal denom = dotp*dotp-dot(n_p,n_p)*np2_sq;
                OverlapReal lambda0 = OverlapReal(2.0)*(b2*dotp - b*np2_sq)/denom;
                OverlapReal lambda1 = OverlapReal(2.0)*(-b2*dot(n_p,n_p)+b*dotp)/denom;

                vec3<OverlapReal> r = -(lambda0*n_p+lambda1*np2)/OverlapReal(2.0);

                // if the line does not intersect the sphere, ignore
                if (dot(r,r) > R*R)
                    continue;

                // the line intersects with the sphere at two points, one of
                // maximizes the support function
                vec3<OverlapReal> c01 = cross(n_p,np2);
                OverlapReal s = fast::sqrt((R*R-dot(r,r))*dot(c01,c01));

                OverlapReal t0 = (-dot(r,c01)-s)/dot(c01,c01);
                OverlapReal t1 = (-dot(r,c01)+s)/dot(c01,c01);

                vec3<OverlapReal> v1 = r+t0*c01;
                vec3<OverlapReal> v2 = r+t1*c01;

                // check first point
                bool allowed = true;
                for (unsigned int k = 0; k < _params.N; ++k)
                    {
                    if (k == i || k == j) continue;

                    vec3<OverlapReal> np3(_params.n[k]);
                    OverlapReal b3(_params.offset[k]);

                    // is this vertex inside the volume bounded by all halfspaces?
                    if (dot(np3,v1) + b3 > OverlapReal(0.0))
                        {
                        allowed = false;
                        break;
                        }
                    }

                if (allowed)
                    {
                    if (_params.additional_verts.N >= detail::MAX_FPOLY3D_VERTS)
                        throw std::runtime_error("Max number of vertices exceeded.\n");

                    _params.additional_verts.x[_params.additional_verts.N] = v1.x;
                    _params.additional_verts.y[_params.additional_verts.N] = v1.y;
                    _params.additional_verts.z[_params.additional_verts.N] = v1.z;
                    _params.additional_verts.N++;
                    }

                // check second point
                allowed = true;
                for (unsigned int k = 0; k < _params.N; ++k)
                    {
                    if (k == i || k == j) continue;

                    vec3<OverlapReal> np3(_params.n[k]);
                    OverlapReal b3(_params.offset[k]);

                    // is this vertex inside the volume bounded by all halfspaces?
                    if (dot(np3,v2) + b3 > OverlapReal(0.0))
                        {
                        allowed = false;
                        break;
                        }
                    }

                if (allowed)
                    {
                    if (_params.additional_verts.N >= detail::MAX_FPOLY3D_VERTS)
                        throw std::runtime_error("Max number of vertices exceeded.\n");

                    _params.additional_verts.x[_params.additional_verts.N] = v2.x;
                    _params.additional_verts.y[_params.additional_verts.N] = v2.y;
                    _params.additional_verts.z[_params.additional_verts.N] = v2.z;
                    _params.additional_verts.N++;
                    }
                }
            }
        #endif
        }

    quat<Scalar> orientation;    //!< Orientation of the sphere (unused)

    const param_type& params;           //!< Faceted sphere parameters
    };

//! Check if circumspheres overlap
/*! \param r_ab Vector defining the position of shape b relative to shape a (r_b - r_a)
    \param a first shape
    \param b second shape
    \returns true if the circumspheres of both shapes overlap

    \ingroup shape
*/
DEVICE inline bool check_circumsphere_overlap(const vec3<Scalar>& r_ab, const ShapeFacetedSphere& a,
    const ShapeFacetedSphere &b)
    {
    OverlapReal DaDb = a.getCircumsphereDiameter() + b.getCircumsphereDiameter();
    vec3<OverlapReal> dr(r_ab);

    return (dot(dr,dr) <= DaDb*DaDb/OverlapReal(4.0));
    }

//! Overlap of faceted spheres
/*! \param r_ab Vector defining the position of shape b relative to shape a (r_b - r_a)
    \param a first shape
    \param b second shape
    \param err in/out variable incremented when error conditions occur in the overlap test
    \returns true when *a* and *b* overlap, and false when they are disjoint

    \ingroup shape
*/
template <>
DEVICE inline bool test_overlap<ShapeFacetedSphere, ShapeFacetedSphere>(const vec3<Scalar>& r_ab, const ShapeFacetedSphere& a, const ShapeFacetedSphere& b, unsigned int& err)
    {
    vec3<OverlapReal> dr(r_ab);

    OverlapReal RaRb = a.params.insphere_radius + b.params.insphere_radius;

    if (dot(dr,dr) < RaRb*RaRb)
        {
        // trivial rejection
        return true;
        }

    OverlapReal DaDb = a.getCircumsphereDiameter() + b.getCircumsphereDiameter();
    return detail::xenocollide_3d(detail::SupportFuncFacetedSphere(a.params),
                           detail::SupportFuncFacetedSphere(b.params),
                           rotate(conj(quat<OverlapReal>(a.orientation)), dr + rotate(quat<OverlapReal>(b.orientation),b.params.origin))-a.params.origin,
                           conj(quat<OverlapReal>(a.orientation))* quat<OverlapReal>(b.orientation),
                           DaDb/2.0,
                           err);

    /*
    return detail::gjke_3d(detail::SupportFuncFacetedSphere(a.params),
                           detail::SupportFuncFacetedSphere(b.params),
                           rotate(conj(quat<OverlapReal>(a.orientation)), dr),
                           conj(quat<OverlapReal>(a.orientation))* quat<OverlapReal>(b.orientation),
                           DaDb/2.0,
                           err);
    */
    }

}; // end namespace hpmc

#endif //__SHAPE_FACETED_SPHERE_H__
