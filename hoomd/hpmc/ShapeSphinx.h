// Copyright (c) 2009-2019 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.

#include "hoomd/HOOMDMath.h"
#include "HPMCPrecisionSetup.h"
#include "hoomd/BoxDim.h"
#include "hoomd/VectorMath.h"
#include "ShapeSphere.h"    //< For the base template of test_overlap

#ifndef __SHAPE_SPHINX_H__
#define __SHAPE_SPHINX_H__

/*! \file ShapeSphinx.h
    \brief Defines the sphinx shape - an intersection of spheres
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

#include "SphinxOverlap.h"  //< This is the main overlap function.

namespace hpmc
{

namespace detail
{

//! maximum number of sphere centers that can be stored
/*! \ingroup hpmc_data_structs */
const unsigned int MAX_SPHERE_CENTERS = 8;

//! Data structure for sphere centers and diameters
/*! \ingroup hpmc_data_structs */
struct sphinx3d_params : param_base
    {
    OverlapReal circumsphereDiameter;               //!< Circumsphere Diameter of all spheres defined in intersection
    unsigned int N;                                //!< Number of spheres
    OverlapReal diameter[MAX_SPHERE_CENTERS];      //!< Sphere Diameters
    vec3<OverlapReal> center[MAX_SPHERE_CENTERS];  //!< Sphere Centers (in local frame)
    unsigned int ignore;    //!< 0: Process overlaps - if (a.ignore == True) and (b.ignore == True) then test_overlap(a,b) = False

    #ifdef ENABLE_CUDA
    //! Attach managed memory to CUDA stream
    void attach_to_stream(cudaStream_t stream) const
        {
        // default implementation does nothing
        }
    #endif
    } __attribute__((aligned(32)));

}; // end namespace detail

namespace detail
    {
    DEVICE inline OverlapReal initVolume(bool disjoint, OverlapReal r[MAX_SPHERE_CENTERS], int n,
         OverlapReal d[MAX_SPHERE_CENTERS*(MAX_SPHERE_CENTERS-1)/2]);
    }

//! Sphinx shape template
/*! ShapeSphinx represents the intersection of spheres. A  positive sphere is one where the volume inside
    the sphere is considered, and a negative sphere is one where the volume outside the sphere is considered.
    This shape is defined using a struct called sphinx3d_params whose detail is named spheres. ShapeSphinx requires
    an incoming orientation vector, followed by parameters for the struct sphinx3d_params.
    The parameter defining a sphinx is a structure containing the circumsphere diameter of all spheres defined in the
    shape. This is followed the number of spheres in the shape, their diameters and then a list of the centers of the
    spheres. It is recommended that the list of spheres begin with a positive sphere placed at the origin,
    and the other sphere centers are relative to it. Positive spheres are defined by a positive value in their
    diameter. Negative spheres are defined by a negative value in their diameter.

    \ingroup shape
*/
struct ShapeSphinx
    {
    //! Define the parameter type
    typedef detail::sphinx3d_params param_type;

    DEVICE inline ShapeSphinx(const quat<Scalar>& _orientation, const param_type& _params)
        : orientation(_orientation), convex(true), spheres(_params)
        {
        volume = 0.0;
        radius = spheres.circumsphereDiameter/(2.0);
        n = spheres.N;
        for(unsigned int i = 0; i<n;i++)
            {
            r[i] = spheres.diameter[i]/(2.0);
            R[i] = r[i]*r[i];
            s[i] = (r[i]<0) ? -1 : 1;
            u[i] = spheres.center[i];
            }
        for(unsigned int i=0; i<n;i++)
            {
            for(unsigned int j=0; j<i; j++)
                {
                D[(i-1)*i/2+j] = dot(u[i]-u[j],u[i]-u[j]);
                d[(i-1)*i/2+j] = s[i]*s[j]*sqrt(D[(i-1)*i/2+j]);
                }
            }
        disjoint = ((n > 0) && (s[0] > 0));
        if(disjoint)
            for(unsigned int i = 1; i < n; i++)
                {
                 if(s[i] > 0) disjoint = false;
                 if(disjoint)
                 for(unsigned int j = 1; j < i; j++)
                    if(!detail::seq2(1,1,R[i],R[j],D[(i-1)*i/2+j])) disjoint = false;
                }
        volume = detail::initVolume(disjoint,r,n,d);
        }

    //! Does this shape have an orientation
    DEVICE static bool hasOrientation() { return true; }

    //! Get the circumsphere diameter
    DEVICE OverlapReal getCircumsphereDiameter() const
        {
        // return the diameter of the parent sphere - TODO: recalculate for convex particles
        return spheres.diameter[0];
        }

    //! Get the insphere radius
    DEVICE Scalar getInsphereRadius() const
        {
        return Scalar(0.0);
        }

    #ifndef NVCC
    std::string getShapeSpec() const
        {
        throw std::runtime_error("Shape definition not supported for this shape class.");
        }
    #endif

    //! Return the bounding box of the shape in world coordinates
    DEVICE detail::AABB getAABB(const vec3<Scalar>& pos) const
        {
        return detail::AABB(pos, getCircumsphereDiameter()/Scalar(2.0));
        }

    //!Ignore flag for acceptance statistics
    DEVICE bool ignoreStatistics() const { return spheres.ignore; }

    //!Ignore flag for overlaps
    HOSTDEVICE static bool isParallel() {return false; }

    quat<Scalar> orientation;                   //!< Orientation of the sphinx

    unsigned int n;              //!< Number of spheres
    bool convex;
    bool disjoint;
    OverlapReal r[detail::MAX_SPHERE_CENTERS];                 //!< radius of each sphere
    OverlapReal R[detail::MAX_SPHERE_CENTERS];                 //!< radius^2
    int s[detail::MAX_SPHERE_CENTERS];                         //!< sign of radius of each sphere
    vec3<OverlapReal> u[detail::MAX_SPHERE_CENTERS];           //!< original center of each sphere
    //vec3<OverlapReal> v[MAX_SPHERE_CENTERS];           //!< rotated center - having this in the overlap check
    OverlapReal D[detail::MAX_SPHERE_CENTERS*(detail::MAX_SPHERE_CENTERS-1)/2];   //!< distance^2 between every pair of spheres
    OverlapReal d[detail::MAX_SPHERE_CENTERS*(detail::MAX_SPHERE_CENTERS-1)/2];   //!< distance with sign bet. every pair of spheres
    OverlapReal radius;
    OverlapReal volume;

    const detail::sphinx3d_params& spheres;     //!< Vertices
    };

//! Check if circumspheres overlap
/*! \param r_ab Vector defining the position of shape b relative to shape a (r_b - r_a)
    \param a first shape
    \param b second shape
    \returns true if the circumspheres of both shapes overlap

    \ingroup shape
*/
DEVICE inline bool check_circumsphere_overlap(const vec3<Scalar>& r_ab, const ShapeSphinx& a,
    const ShapeSphinx &b)
    {
    OverlapReal DaDb = a.getCircumsphereDiameter() + b.getCircumsphereDiameter();
    vec3<OverlapReal> dr(r_ab);

    return (dot(dr,dr) <= DaDb*DaDb/OverlapReal(4.0));
    }


template <>
DEVICE inline bool test_overlap<ShapeSphinx,ShapeSphinx>(const vec3<Scalar>& r_ab,
                                                          const ShapeSphinx& p,
                                                          const ShapeSphinx& q, unsigned int& err)
    {
    vec3<OverlapReal> pv[detail::MAX_SPHERE_CENTERS];           //!< rotated centers of p
    vec3<OverlapReal> qv[detail::MAX_SPHERE_CENTERS];           //!< rotated centers of q

    quat<OverlapReal> qp(p.orientation);
    quat<OverlapReal> qq(q.orientation);

    // update the positions of the spheres according to the rotations of the center for p
    for(unsigned int i=0; i < p.n; i++)
        {
        pv[i] = rotate(qp,p.u[i]);
        }

    // update the positions of the spheres according to the rotations of the center for p
    for(unsigned int i=0; i < q.n; i++)
        {
        qv[i] = rotate(qq,q.u[i]);
        }

    vec3<OverlapReal> x(0.0,0.0,0.0);
    vec3<OverlapReal> y(r_ab);

    if(p.disjoint && q.disjoint)
            {
            if((p.n == 1) && (q.n == 1))
                {
                vec3<OverlapReal> a = x + pv[0],b = y+qv[0];
                if(detail::sep2(false,
                        p.s[0],q.s[0],
                        p.R[0],q.R[0],
                        detail::norm2(a-b))) return false;
                }
            if((p.n > 1) && (q.n == 1))
                {
                vec3<OverlapReal> a = x+pv[0],c = y+qv[0];
                for(unsigned int i = 1; i < p.n; i++)
                    {
                    int k = (i-1)*i/2;
                    vec3<OverlapReal> b = x+pv[i];
                    if(detail::sep3(false,
                            p.s[0],p.s[i],q.s[0],
                            p.R[0],p.R[i],q.R[0],
                            p.D[k],detail::norm2(a-c),
                            detail::norm2(b-c))) return false;
                    }
                }
            if((p.n == 1) && (q.n > 1))
                {
                vec3<OverlapReal> a = x+pv[0],b = y+qv[0];
                for(unsigned int j = 1; j < q.n; j++)
                    {
                    int l = (j-1)*j/2;
                    vec3<OverlapReal> c = y+qv[j];
                    if(detail::sep3(false,
                            p.s[0],q.s[0],q.s[j],
                            p.R[0],q.R[0],q.R[j],
                            detail::norm2(a-b),detail::norm2(a-c),
                            q.D[l])) return false;
                    }
                }
            if((p.n > 1) && (q.n > 1))
                {
                vec3<OverlapReal> a = x+pv[0],c = y+qv[0];
                for(unsigned int i = 1; i < p.n; i++)
                    {
                    int k = (i-1)*i/2;
                    for(unsigned int j = 1; j < q.n; j++)
                        {
                        int l = (j-1)*j/2;
                        vec3<OverlapReal> b = x+pv[i],d = y+qv[j];
                        if(detail::sep4(false,
                                p.s[0],p.s[i],q.s[0],q.s[j],
                                p.R[0],p.R[i],q.R[0],q.R[j],
                                p.D[k],detail::norm2(a-c),detail::norm2(a-d),
                                detail::norm2(b-c),detail::norm2(b-d),
                                q.D[l])) return false;
                        }
                    }
                }
            return true;
            }

        if((p.n == 1) && (q.n == 1))
            {
            vec3<OverlapReal> a = x+pv[0],b = y+qv[0];
            return !detail::sep2(p.convex && q.convex,
                         p.s[0],q.s[0],
                         p.R[0],q.R[0],
                         detail::norm2(a-b));
            }

        if((p.n == 2) && (q.n == 1))
            {
            vec3<OverlapReal> a = x+pv[0],b = x+pv[1],c = y+qv[0];
            return !detail::sep3(p.convex && q.convex,
                         p.s[0],p.s[1],q.s[0],
                         p.R[0],p.R[1],q.R[0],
                         p.D[0],detail::norm2(a-c),
                         detail::norm2(b-c));
            }
        if((p.n == 1) && (q.n == 2))
            {
            vec3<OverlapReal> a = x+pv[0],b = y+qv[0],c = y+qv[1];
            return !detail::sep3(p.convex && q.convex,
                         p.s[0],q.s[0],q.s[1],
                         p.R[0],q.R[0],q.R[1],
                         detail::norm2(a-b),detail::norm2(a-c),
                         q.D[0]);
            }

        if((p.n == 3) && (q.n == 1))
            {
            vec3<OverlapReal> a = x+pv[0],b = x+pv[1],c = x+pv[2],d = y+qv[0];
            return !detail::sep4(p.convex && q.convex,
                         p.s[0],p.s[1],p.s[2],q.s[0],
                         p.R[0],p.R[1],p.R[2],q.R[0],
                         p.D[0],p.D[1],detail::norm2(a-d),
                         p.D[2],detail::norm2(b-d),
                         detail::norm2(c-d));
            }
        if((p.n == 2) && (q.n == 2))
            {
            vec3<OverlapReal> a = x+pv[0],b = x+pv[1],c = y+qv[0],d = y+qv[1];
            return !detail::sep4(p.convex && q.convex,
                         p.s[0],p.s[1],q.s[0],q.s[1],
                         p.R[0],p.R[1],q.R[0],q.R[1],
                         p.D[0],detail::norm2(a-c),detail::norm2(a-d),
                         detail::norm2(b-c),detail::norm2(b-d),
                         q.D[0]);
            }
        if((p.n == 1) && (q.n == 3))
            {
            vec3<OverlapReal> a = x+pv[0],b = y+qv[0],c = y+qv[1],d = y+qv[2];
            return !detail::sep4(p.convex && q.convex,
                         p.s[0],q.s[0],q.s[1],q.s[2],
                         p.R[0],q.R[0],q.R[1],q.R[2],
                         detail::norm2(a-b),detail::norm2(a-c),detail::norm2(a-d),
                         q.D[0],q.D[1],
                         q.D[2]);
            }

        if((p.n == 4) && (q.n == 1))
            {
            vec3<OverlapReal> a = x+pv[0],b = x+pv[1],c = x+pv[2],d = x+pv[3],e = y+qv[0];
            return !detail::sep5(p.convex && q.convex,
                         p.s[0],p.s[1],p.s[2],p.s[3],q.s[0],
                         p.R[0],p.R[1],p.R[2],p.R[3],q.R[0],
                         p.D[0],p.D[1],p.D[3],detail::norm2(a-e),
                         p.D[2],p.D[4],detail::norm2(b-e),
                         p.D[5],detail::norm2(c-e),
                         detail::norm2(d-e));
            }
        if((p.n == 3) && (q.n == 2))
            {
            vec3<OverlapReal> a = x+pv[0],b = x+pv[1],c = x+pv[2],d = y+qv[0],e = y+qv[1];
            return !detail::sep5(p.convex && q.convex,
                         p.s[0],p.s[1],p.s[2],q.s[0],q.s[1],
                         p.R[0],p.R[1],p.R[2],q.R[0],q.R[1],
                         p.D[0],p.D[1],detail::norm2(a-d),detail::norm2(a-e),
                         p.D[2],detail::norm2(b-d),detail::norm2(b-e),
                         detail::norm2(c-d),detail::norm2(c-e),
                         q.D[0]);
            }
        if((p.n == 2) && (q.n == 3))
            {
            vec3<OverlapReal> a = x+pv[0],b = x+pv[1],c = y+qv[0],d = y+qv[1],e = y+qv[2];
            return !detail::sep5(p.convex && q.convex,
                         p.s[0],p.s[1],q.s[0],q.s[1],q.s[2],
                         p.R[0],p.R[1],q.R[0],q.R[1],q.R[2],
                         p.D[0],detail::norm2(a-c),detail::norm2(a-d),detail::norm2(a-e),
                         detail::norm2(b-c),detail::norm2(b-d),detail::norm2(b-e),
                         q.D[0],q.D[1],
                         q.D[2]);
            }
        if((p.n == 1) && (q.n == 4))
            {
            vec3<OverlapReal> a = x+pv[0],b = y+qv[0],c = y+qv[1],d = y+qv[2],e = y+qv[3];
            return !detail::sep5(p.convex && q.convex,
                         p.s[0],q.s[0],q.s[1],q.s[2],q.s[3],
                         p.R[0],q.R[0],q.R[1],q.R[2],q.R[3],
                         detail::norm2(a-b),detail::norm2(a-c),detail::norm2(a-d),detail::norm2(a-e),
                         q.D[0],q.D[1],q.D[3],
                         q.D[2],q.D[4],
                         q.D[5]);
            }

        if((p.n == 5) && (q.n == 1))
            {
            vec3<OverlapReal> a = x+pv[0],b = x+pv[1],c = x+pv[2],d = x+pv[3],e = x+pv[4],f = y+qv[0];
            return !detail::sep6(p.convex && q.convex,
                         p.s[0],p.s[1],p.s[2],p.s[3],p.s[4],q.s[0],
                         p.R[0],p.R[1],p.R[2],p.R[3],p.R[4],q.R[0],
                         p.D[0],p.D[1],p.D[3],p.D[6],detail::norm2(a-f),
                         p.D[2],p.D[4],p.D[7],detail::norm2(b-f),
                         p.D[5],p.D[8],detail::norm2(c-f),
                         p.D[9],detail::norm2(d-f),
                         detail::norm2(e-f));
            }
        if((p.n == 4) && (q.n == 2))
            {
            vec3<OverlapReal> a = x+pv[0],b = x+pv[1],c = x+pv[2],d = x+pv[3],e = y+qv[0],f = y+qv[1];
            return !detail::sep6(p.convex && q.convex,
                         p.s[0],p.s[1],p.s[2],p.s[3],q.s[0],q.s[1],
                         p.R[0],p.R[1],p.R[2],p.R[3],q.R[0],q.R[1],
                         p.D[0],p.D[1],p.D[3],detail::norm2(a-e),detail::norm2(a-f),
                         p.D[2],p.D[4],detail::norm2(b-e),detail::norm2(b-f),
                         p.D[5],detail::norm2(c-e),detail::norm2(c-f),
                         detail::norm2(d-e),detail::norm2(d-f),
                         q.D[0]);
            }
        if((p.n == 3) && (q.n == 3))
            {
            vec3<OverlapReal> a = x+pv[0],b = x+pv[1],c = x+pv[2],d = y+qv[0],e = y+qv[1],f = y+qv[2];
            return !detail::sep6(p.convex && q.convex,
                         p.s[0],p.s[1],p.s[2],q.s[0],q.s[1],q.s[2],
                         p.R[0],p.R[1],p.R[2],q.R[0],q.R[1],q.R[2],
                         p.D[0],p.D[1],detail::norm2(a-d),detail::norm2(a-e),detail::norm2(a-f),
                         p.D[2],detail::norm2(b-d),detail::norm2(b-e),detail::norm2(b-f),
                         detail::norm2(c-d),detail::norm2(c-e),detail::norm2(c-f),
                         q.D[0],q.D[1],
                         q.D[2]);
            }
        if((p.n == 2) && (q.n == 4))
            {
            vec3<OverlapReal> a = x+pv[0],b = x+pv[1],c = y+qv[0],d = y+qv[1],e = y+qv[2],f = y+qv[3];
            return !detail::sep6(p.convex && q.convex,
                         p.s[0],p.s[1],q.s[0],q.s[1],q.s[2],q.s[3],
                         p.R[0],p.R[1],q.R[0],q.R[1],q.R[2],q.R[3],
                         p.D[0],detail::norm2(a-c),detail::norm2(a-d),detail::norm2(a-e),detail::norm2(a-f),
                         detail::norm2(b-c),detail::norm2(b-d),detail::norm2(b-e),detail::norm2(b-f),
                         q.D[0],q.D[1],q.D[3],
                         q.D[2],q.D[4],
                         q.D[5]);
            }
        if((p.n == 1) && (q.n == 5))
            {
            vec3<OverlapReal> a = x+pv[0],b = y+qv[0],c = y+qv[1],d = y+qv[2],e = y+qv[3],f = y+qv[4];
            return !detail::sep6(p.convex && q.convex,
                         p.s[0],q.s[0],q.s[1],q.s[2],q.s[3],q.s[4],
                         p.R[0],q.R[0],q.R[1],q.R[2],q.R[3],q.R[4],
                         detail::norm2(a-b),detail::norm2(a-c),detail::norm2(a-d),detail::norm2(a-e),detail::norm2(a-f),
                         q.D[0],q.D[1],q.D[3],q.D[6],
                         q.D[2],q.D[4],q.D[7],
                         q.D[5],q.D[8],
                         q.D[9]);
            }

        if((p.n == 5) && (q.n == 2))
            {
            vec3<OverlapReal> a = x+pv[0],b = x+pv[1],c = x+pv[2],d = x+pv[3],e = x+pv[4],f = y+qv[0],g = y+qv[1];
            return !detail::sep7(p.convex && q.convex,
                         p.s[0],p.s[1],p.s[2],p.s[3],p.s[4],q.s[0],q.s[1],
                         p.R[0],p.R[1],p.R[2],p.R[3],p.R[4],q.R[0],q.R[1],
                         p.D[0],p.D[1],p.D[3],p.D[6],detail::norm2(a-f),detail::norm2(a-g),
                         p.D[2],p.D[4],p.D[7],detail::norm2(b-f),detail::norm2(b-g),
                         p.D[5],p.D[8],detail::norm2(c-f),detail::norm2(c-g),
                         p.D[9],detail::norm2(d-f),detail::norm2(d-g),
                         detail::norm2(e-f),detail::norm2(e-g),
                         q.D[0]);
            }
        if((p.n == 4) && (q.n == 3))
            {
            vec3<OverlapReal> a = x+pv[0],b = x+pv[1],c = x+pv[2],d = x+pv[3],e = y+qv[0],f = y+qv[1],g = y+qv[2];
            return !detail::sep7(p.convex && q.convex,
                         p.s[0],p.s[1],p.s[2],p.s[3],q.s[0],q.s[1],q.s[2],
                         p.R[0],p.R[1],p.R[2],p.R[3],q.R[0],q.R[1],q.R[2],
                         p.D[0],p.D[1],p.D[3],detail::norm2(a-e),detail::norm2(a-f),detail::norm2(a-g),
                         p.D[2],p.D[4],detail::norm2(b-e),detail::norm2(b-f),detail::norm2(b-g),
                         p.D[5],detail::norm2(c-e),detail::norm2(c-f),detail::norm2(c-g),
                         detail::norm2(d-e),detail::norm2(d-f),detail::norm2(d-g),
                         q.D[0],q.D[1],
                         q.D[2]);
            }
        if((p.n == 3) && (q.n == 4))
            {
            vec3<OverlapReal> a = x+pv[0],b = x+pv[1],c = x+pv[2],d = y+qv[0],e = y+qv[1],f = y+qv[2],g = y+qv[3];
            return !detail::sep7(p.convex && q.convex,
                         p.s[0],p.s[1],p.s[2],q.s[0],q.s[1],q.s[2],q.s[3],
                         p.R[0],p.R[1],p.R[2],q.R[0],q.R[1],q.R[2],q.R[3],
                         p.D[0],p.D[1],detail::norm2(a-d),detail::norm2(a-e),detail::norm2(a-f),detail::norm2(a-g),
                         p.D[2],detail::norm2(b-d),detail::norm2(b-e),detail::norm2(b-f),detail::norm2(b-g),
                         detail::norm2(c-d),detail::norm2(c-e),detail::norm2(c-f),detail::norm2(c-g),
                         q.D[0],q.D[1],q.D[3],
                         q.D[2],q.D[4],
                         q.D[5]);
            }
        if((p.n == 2) && (q.n == 5))
            {
            vec3<OverlapReal> a = x+pv[0],b = x+pv[1],c = y+qv[0],d = y+qv[1],e = y+qv[2],f = y+qv[3],g = y+qv[4];
            return !detail::sep7(p.convex && q.convex,
                         p.s[0],p.s[1],q.s[0],q.s[1],q.s[2],q.s[3],q.s[4],
                         p.R[0],p.R[1],q.R[0],q.R[1],q.R[2],q.R[3],q.R[4],
                         p.D[0],detail::norm2(a-c),detail::norm2(a-d),detail::norm2(a-e),detail::norm2(a-f),detail::norm2(a-g),
                         detail::norm2(b-c),detail::norm2(b-d),detail::norm2(b-e),detail::norm2(b-f),detail::norm2(b-g),
                         q.D[0],q.D[1],q.D[3],q.D[6],
                         q.D[2],q.D[4],q.D[7],
                         q.D[5],q.D[8],
                         q.D[9]);
            }

        if((p.n == 5) && (q.n == 3))
            {
            vec3<OverlapReal> a = x+pv[0],b = x+pv[1],c = x+pv[2],d = x+pv[3],e = x+pv[4],f = y+qv[0],g = y+qv[1],h = y+qv[2];
            return !detail::sep8(p.convex && q.convex,
                         p.s[0],p.s[1],p.s[2],p.s[3],p.s[4],q.s[0],q.s[1],q.s[2],
                         p.R[0],p.R[1],p.R[2],p.R[3],p.R[4],q.R[0],q.R[1],q.R[2],
                         p.D[0],p.D[1],p.D[3],p.D[6],detail::norm2(a-f),detail::norm2(a-g),detail::norm2(a-h),
                         p.D[2],p.D[4],p.D[7],detail::norm2(b-f),detail::norm2(b-g),detail::norm2(b-h),
                         p.D[5],p.D[8],detail::norm2(c-f),detail::norm2(c-g),detail::norm2(c-h),
                         p.D[9],detail::norm2(d-f),detail::norm2(d-g),detail::norm2(d-h),
                         detail::norm2(e-f),detail::norm2(e-g),detail::norm2(e-h),
                         q.D[0],q.D[1],
                         q.D[2]);
            }
        if((p.n == 4) && (q.n == 4))
            {
            vec3<OverlapReal> a = x+pv[0],b = x+pv[1],c = x+pv[2],d = x+pv[3],e = y+qv[0],f = y+qv[1],g = y+qv[2],h = y+qv[3];
            return !detail::sep8(p.convex && q.convex,
                         p.s[0],p.s[1],p.s[2],p.s[3],q.s[0],q.s[1],q.s[2],q.s[3],
                         p.R[0],p.R[1],p.R[2],p.R[3],q.R[0],q.R[1],q.R[2],q.R[3],
                         p.D[0],p.D[1],p.D[3],detail::norm2(a-e),detail::norm2(a-f),detail::norm2(a-g),detail::norm2(a-h),
                         p.D[2],p.D[4],detail::norm2(b-e),detail::norm2(b-f),detail::norm2(b-g),detail::norm2(b-h),
                         p.D[5],detail::norm2(c-e),detail::norm2(c-f),detail::norm2(c-g),detail::norm2(c-h),
                         detail::norm2(d-e),detail::norm2(d-f),detail::norm2(d-g),detail::norm2(d-h),
                         q.D[0],q.D[1],q.D[3],
                         q.D[2],q.D[4],
                         q.D[5]);
            }
        if((p.n == 3) && (q.n == 5))
            {
            vec3<OverlapReal> a = x+pv[0],b = x+pv[1],c = x+pv[2],d = y+qv[0],e = y+qv[1],f = y+qv[2],g = y+qv[3],h = y+qv[4];
            return !detail::sep8(p.convex && q.convex,
                         p.s[0],p.s[1],p.s[2],q.s[0],q.s[1],q.s[2],q.s[3],q.s[4],
                         p.R[0],p.R[1],p.R[2],q.R[0],q.R[1],q.R[2],q.R[3],q.R[4],
                         p.D[0],p.D[1],detail::norm2(a-d),detail::norm2(a-e),detail::norm2(a-f),detail::norm2(a-g),detail::norm2(a-h),
                         p.D[2],detail::norm2(b-d),detail::norm2(b-e),detail::norm2(b-f),detail::norm2(b-g),detail::norm2(b-h),
                         detail::norm2(c-d),detail::norm2(c-e),detail::norm2(c-f),detail::norm2(c-g),detail::norm2(c-h),
                         q.D[0],q.D[1],q.D[3],q.D[6],
                         q.D[2],q.D[4],q.D[7],
                         q.D[5],q.D[8],
                         q.D[9]);
            }
        /*if((p.n == 5) && (q.n == 4))
            {
            vec3<OverlapReal> a = x+pv[0],b = x+pv[1],c = x+pv[2],d = x+pv[3],e = x+pv[4],f = y+qv[0],g = y+qv[1],h = y+qv[2],i = y+qv[3];
            return !detail::sep9(p.convex && q.convex,
                          p.s[0],p.s[1],p.s[2],p.s[3],p.s[4],q.s[0],q.s[1],q.s[2],q.s[3],
                          p.R[0],p.R[1],p.R[2],p.R[3],p.R[4],q.R[0],q.R[1],q.R[2],q.R[3],
                          p.D[0],p.D[1],p.D[3],p.D[6],detail::norm2(a-f),detail::norm2(a-g),detail::norm2(a-h),detail::norm2(a-i),
                          p.D[2],p.D[4],p.D[7],detail::norm2(b-f),detail::norm2(b-g),detail::norm2(b-h),detail::norm2(b-i),
                          p.D[5],p.D[8],detail::norm2(c-f),detail::norm2(c-g),detail::norm2(c-h),detail::norm2(c-i),
                          p.D[9],detail::norm2(d-f),detail::norm2(d-g),detail::norm2(d-h),detail::norm2(d-i),
                          detail::norm2(e-f),detail::norm2(e-g),detail::norm2(e-h),detail::norm2(e-i),
                          q.D[0],q.D[1],q.D[3],
                          q.D[2],q.D[4],
                          q.D[5]);
            }
        if((p.n == 4) && (q.n == 5))
            {
            vec3<OverlapReal> a = x+pv[0],b = x+pv[1],c = x+pv[2],d = x+pv[3],e = y+qv[0],f = y+qv[1],g = y+qv[2],h = y+qv[3],i = y+qv[4];
            return !detail::sep9(p.convex && q.convex,
                          p.s[0],p.s[1],p.s[2],p.s[3],q.s[0],q.s[1],q.s[2],q.s[3],q.s[4],
                          p.R[0],p.R[1],p.R[2],p.R[3],q.R[0],q.R[1],q.R[2],q.R[3],q.R[4],
                          p.D[0],p.D[1],p.D[3],detail::norm2(a-e),detail::norm2(a-f),detail::norm2(a-g),detail::norm2(a-h),detail::norm2(a-i),
                          p.D[2],p.D[4],detail::norm2(b-e),detail::norm2(b-f),detail::norm2(b-g),detail::norm2(b-h),detail::norm2(b-i),
                          p.D[5],detail::norm2(c-e),detail::norm2(c-f),detail::norm2(c-g),detail::norm2(c-h),detail::norm2(c-i),
                          detail::norm2(d-e),detail::norm2(d-f),detail::norm2(d-g),detail::norm2(d-h),detail::norm2(d-i),
                          q.D[0],q.D[1],q.D[3],q.D[6],
                          q.D[2],q.D[4],q.D[7],
                          q.D[5],q.D[8],
                          q.D[9]);
            }

        if((p.n == 5) && (q.n == 5))
            {
            vec3<OverlapReal> a = x+pv[0],b = x+pv[1],c = x+pv[2],d = x+pv[3],e = x+pv[4],f = y+qv[0],g = y+qv[1],h = y+qv[2],i = y+qv[3],j = y+qv[4];
            return !detail::sep10(p.convex && q.convex,
                          p.s[0],p.s[1],p.s[2],p.s[3],p.s[4],q.s[0],q.s[1],q.s[2],q.s[3],q.s[4],
                          p.R[0],p.R[1],p.R[2],p.R[3],p.R[4],q.R[0],q.R[1],q.R[2],q.R[3],q.R[4],
                          p.D[0],p.D[1],p.D[3],p.D[6],detail::norm2(a-f),detail::norm2(a-g),detail::norm2(a-h),detail::norm2(a-i),detail::norm2(a-j),
                          p.D[2],p.D[4],p.D[7],detail::norm2(b-f),detail::norm2(b-g),detail::norm2(b-h),detail::norm2(b-i),detail::norm2(b-j),
                          p.D[5],p.D[8],detail::norm2(c-f),detail::norm2(c-g),detail::norm2(c-h),detail::norm2(c-i),detail::norm2(c-j),
                          p.D[9],detail::norm2(d-f),detail::norm2(d-g),detail::norm2(d-h),detail::norm2(d-i),detail::norm2(d-j),
                          detail::norm2(e-f),detail::norm2(e-g),detail::norm2(e-h),detail::norm2(e-i),detail::norm2(e-j),
                          q.D[0],q.D[1],q.D[3],q.D[6],
                          q.D[2],q.D[4],q.D[7],
                          q.D[5],q.D[8],
                          q.D[9]);
            }*/

        return true;
    }

namespace detail
{
DEVICE inline OverlapReal initVolume(bool disjoint, OverlapReal r[MAX_SPHERE_CENTERS], int n,
     OverlapReal d[MAX_SPHERE_CENTERS*(MAX_SPHERE_CENTERS-1)/2])
    {
    if(disjoint)
        {
        OverlapReal vol = uol1(r[0]);
        for(int i = 1; i < n; i++)
            vol += uol2(r[0],r[i],d[(i-1)*i/2])-uol1(r[0]);
        return vol;
        }

    if(n == 1)
        return uol1(r[0]);

    if(n == 2)
        return uol2(r[0],r[1],
                    d[0]);

    if(n == 3)
        return uol3(r[0],r[1],r[2],
                    d[0],d[1],
                    d[2]);

    if(n == 4)
        return uol4(r[0],r[1],r[2],r[3],
                    d[0],d[1],d[3],
                    d[2],d[4],
                    d[5]);

    /*if(n == 5)
        return uol5(r[0],r[1],r[2],r[3],r[4],
                    d[0],d[1],d[3],d[6],
                    d[2],d[4],d[7],
                    d[5],d[8],
                    d[9]);
    */
    return 0;
    }

} // detail

}; // end namespace hpmc

#undef DEVICE
#undef HOSTDEVICE
#endif // __SHAPE_SPHINX_H__
