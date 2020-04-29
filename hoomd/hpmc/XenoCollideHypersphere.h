// Copyright (c) 2009-2019 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.

#include "hoomd/HOOMDMath.h"
#include "hoomd/Hypersphere.h"
#include "HPMCPrecisionSetup.h"
#include "hoomd/VectorMath.h"
#include "MinkowskiMath.h"
#include <cstdio>

#ifndef __XENOCOLLIDE_HYPERSPHERE_H__
#define __XENOCOLLIDE_HYPERSPHERE_H__

/*! \file XenoCollideHypersphere.h
    \brief Implements XenoCollide on the Hypersphere
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

const unsigned int XENOCOLLIDE_HYPERSPHERE_MAX_ITERATIONS = 1024;

//! XenoCollide overlap check in Hypersphere
/*! \tparam SupportFuncA Support function class type for shape A
    \tparam SupportFuncB Support function class type for shape B
    \param sa Support function for shape A
    \param sb Support function for shape B
    \param quat_l Left quaternion of shape B in frame A
    \param quat_r Right quaternion of shape B in frame A
    \param hypersphere Hypersphere the particles are living on
    \param R Approximate radius of Minkowski difference for scaling tolerance value
    \param err_count Error counter to increment whenever an infinite loop is encountered
    \returns true when the two shapes overlap and false when they are disjoint.

    XenoCollide is a generic algorithm for detecting overlaps between two shapes. It operates with the support function
    of each of the two shapes. To enable generic use of this algorithm on a variety of shapes, those support functions
    are passed in as templated functors. Each functor might store a reference to data (i.e. polyhedron verts), but the only
    public interface that XenoCollide will use is to call the operator() on the functor and give it the normal vector
    n *in the **local** coordinates* of that shape. Local coordinates are used to avoid massive memory usage needed to
    store a translated copy of each shape.

    The initial implementation is designed primarily for polygons. Shapes with curved surfaces could be used,
    but they require an additional termination condition that comes with a tolerance. When and if such shapes are
    needed, we can update this function to optionally implement that tolerance (via another template parameter).

    The parameters of this class closely follow those of test_overlap_separating_planes, since they were found to be a
    good breakdown of the problem into coordinate systems. Specifically, overlaps are checked in a coordinate system
    where particle *A* is at (R,0,0,0), and particle *B* is at position quat_l(R,0,0,0)quat_r. Particle A has orientation (0,0,0,1)
    and particle B has orientation quat_l(0,0,0,1)quat_r*.

    The recommended way of using this code is to specify the support functor in the same file as the shape data
    (e.g. ShapeConvexPolyhedron.h). Then include XenoCollideHypersphere.h and call xenocollide_3d where needed.

    **Normalization**
    In _Games Programming Gems_, the book normalizes all vectors passed into S. This is unnecessary in some circumstances
    and we avoid it for performance reasons. Support functions that require the use of normal n vectors should normalize
    it when needed.

    \ingroup minkowski
*/
template<class SupportFuncA, class SupportFuncB>
DEVICE inline bool xenocollide_hypersphere(const SupportFuncA& a,
                                  const SupportFuncB& b,
                                  const quat<OverlapReal>& quat_l,
                                  const quat<OverlapReal>& quat_r,
                                  const Hypersphere& hypersphere,
                                  const OverlapReal Ra,
                                  unsigned int& err_count)
    {
    // This implementation of XenoCollide is hand-written from the description of the algorithm on page 171 of _Games
    // Programming Gems 7_


     quat<OverlapReal> a_p1 =  hypersphere.cartesianToHyperspherical(vec3<OverlapReal>(a.x[0],a.y[0],a.z[0]));
     quat<OverlapReal> a_p2 =  hypersphere.cartesianToHyperspherical(vec3<OverlapReal>(a.x[1],a.y[1],a.z[1]));
     quat<OverlapReal> a_p3 =  hypersphere.cartesianToHyperspherical(vec3<OverlapReal>(a.x[2],a.y[2],a.z[2]));
     quat<OverlapReal> a_p4 =  hypersphere.cartesianToHyperspherical(vec3<OverlapReal>(a.x[3],a.y[3],a.z[3]));

     quat<OverlapReal> b_p1 =  hypersphere.cartesianToHyperspherical(vec3<OverlapReal>(b.x[0],b.y[0],b.z[0]));
     quat<OverlapReal> b_p2 =  hypersphere.cartesianToHyperspherical(vec3<OverlapReal>(b.x[1],b.y[1],b.z[1]));
     quat<OverlapReal> b_p3 =  hypersphere.cartesianToHyperspherical(vec3<OverlapReal>(b.x[2],b.y[2],b.z[2]));
     quat<OverlapReal> b_p4 =  hypersphere.cartesianToHyperspherical(vec3<OverlapReal>(b.x[3],b.y[3],b.z[3]));


    //move to Vertex 1 of A
     quat<OverlapReal> pos_u = conj(a_p1)*conj(a_p1);
    std::cout << "pos_u: " << pos_u.s << " " << pos_u.v.x << " " << pos_u.v.y << " " << pos_u.v.z << std::endl;

     quat<OverlapReal> pos_a1 = conj(a_p1)*a_p1*a_p1*conj(a_p1);
     std::cout << "pos_a1: " << pos_a1.s << " " << pos_a1.v.x << " " << pos_a1.v.y << " " << pos_a1.v.z << std::endl;
     quat<OverlapReal> pos_a2 = conj(a_p1)*a_p2*a_p2*conj(a_p1);
     std::cout << "pos_a2: " << pos_a2.s << " " << pos_a2.v.x << " " << pos_a2.v.y << " " << pos_a2.v.z << std::endl;
     quat<OverlapReal> pos_a3 = conj(a_p1)*a_p3*a_p3*conj(a_p1);
     std::cout << "pos_a3: " << pos_a3.s << " " << pos_a3.v.x << " " << pos_a3.v.y << " " << pos_a3.v.z << std::endl;
     quat<OverlapReal> pos_a4 = conj(a_p1)*a_p4*a_p4*conj(a_p1);
     std::cout << "pos_a4: " << pos_a4.s << " " << pos_a4.v.x << " " << pos_a4.v.y << " " << pos_a4.v.z << std::endl;

     std::vector< quat<OverlapReal> > pos_b;
     pos_b.push_back(conj(a_p1)*quat_l*b_p1*b_p1*quat_r*conj(a_p1));
     std::cout << "pos_b[0]: " << pos_b[0].s << " " << pos_b[0].v.x << " " << pos_b[0].v.y << " " << pos_b[0].v.z << std::endl;
     pos_b.push_back(conj(a_p1)*quat_l*b_p2*b_p2*quat_r*conj(a_p1));
     std::cout << "pos_b[1]: " << pos_b[1].s << " " << pos_b[1].v.x << " " << pos_b[1].v.y << " " << pos_b[1].v.z << std::endl;
     pos_b.push_back(conj(a_p1)*quat_l*b_p3*b_p3*quat_r*conj(a_p1));
     std::cout << "pos_b[2]: " << pos_b[2].s << " " << pos_b[2].v.x << " " << pos_b[2].v.y << " " << pos_b[2].v.z << std::endl;
     pos_b.push_back(conj(a_p1)*quat_l*b_p4*b_p4*quat_r*conj(a_p1));
     std::cout << "pos_b[3]: " << pos_b[3].s << " " << pos_b[3].v.x << " " << pos_b[3].v.y << " " << pos_b[3].v.z << std::endl;

    // Face (1,2,3) of A 
    vec3<OverlapReal> n123 = cross(pos_a2.v, pos_a3.v);
    if(dot(n123,pos_u.v) > 0) n123 = -n123;
     std::cout << "n123: " << " " << n123.x << " " << n123.y << " " << n123.z << std::endl;

    //inside or outside halfspace of (1,2,3)
    std::vector<bool> side123(4,false);

    for( int i = 0; i < side123.size(); i++){
        if(dot(n123,pos_b[i].v) > 0) side123[i] = true;
        std::cout << "side123 " << i << ":  " << side123[i] << " " << dot(n123,pos_b[i].v) << std::endl;
    }

    if(side123[0] && side123[1] && side123[2] && side123[3]) return false;

    // Face (1,2,4) of A 
    vec3<OverlapReal> n124 = cross(pos_a2.v, pos_a4.v);
    if(dot(n124,pos_u.v) > 0) n124 = -n124;
     std::cout << "n124: " << " " << n124.x << " " << n124.y << " " << n124.z << std::endl;

    //inside or outside halfspace of (1,2,4)
    std::vector<bool> side124(4,false);

    for( int i = 0; i < side124.size(); i++){
        if(dot(n124,pos_b[i].v) > 0) side124[i] = true;
        std::cout << "side124 " << i << ":  " << side124[i] << " " << dot(n124,pos_b[i].v) << std::endl;
    }

    if(side124[0] && side124[1] && side124[2] && side124[3]) return false;


    // Edge (1,2) of A 
    std::vector< vec3<OverlapReal> > pm;
    std::vector< vec3<OverlapReal> > mp;
    bool mm = false;

    for( int i = 0; i < side124.size(); i++){
        if(!side123[i] && !side124[i] ){ 
            mm=true;
            break;
        }
        if(side123[i] && !side124[i] ) pm.push_back(pos_b[i].v);
        else if(!side123[i] && side124[i]) mp.push_back(pos_b[i].v) ;
    }

    // See if edge of B goes through A 
    if(mp.size() > 0  && pm.size() > 0 && !mm){
        for (int i =0; i < mp.size(); i++){
            vec3<OverlapReal> n = cross(pm[i],pos_a2.v);
            if(dot(n,n123)<0) n = -n;
            for (int j =0; j < mp.size(); j++){
                if(dot(n,mp[j]) < 0 ) mm = true;
            }
        }

        if(!mm) return false;
    }

    // Face (1,3,4) of A 
    vec3<OverlapReal> n134 = cross(pos_a3.v, pos_a4.v);
    if(dot(n134,pos_u.v) > 0) n134 = -n134;

    //inside or outside halfspace of (1,3,4)
    std::vector<bool> side134(4,false);

    for( int i = 0; i < side134.size(); i++){
        if(dot(n134,pos_b[i].v) > 0) side134[i] = true;
        std::cout << "side134 " << i << ":  " << side134[i] << std::endl;
    }

    if(side134[0] && side134[1] && side134[2] && side134[3]) return false;

    // Edge (1,3) of A 
    pm.resize(0);
    mp.resize(0);

    for( int i = 0; i < side123.size(); i++){
        if(!side123[i] && !side134[i] ){ 
            mm=true;
            break;
        }
        if(side123[i] && !side134[i] ) pm.push_back(pos_b[i].v);
        else if(!side123[i] && side134[i]) mp.push_back(pos_b[i].v) ;
    }

    // See if edge of B goes through A 
    if(mp.size() > 0  && pm.size() > 0 && !mm){
        for (int i =0; i < mp.size(); i++){
            vec3<OverlapReal> n = cross(pm[i],pos_a3.v);
            if(dot(n,n123)<0) n = -n;
            for (int j =0; j < mp.size(); j++){
                if(dot(n,mp[j]) < 0 ) mm = true;
            }
        }

        if(!mm) return false;
    }

    // Edge (1,4) of A 
    pm.resize(0);
    mp.resize(0);

    for( int i = 0; i < side124.size(); i++){
        if(!side124[i] && !side134[i] ){ 
            mm=true;
            break;
        }
        if(side124[i] && !side134[i] ) pm.push_back(pos_b[i].v);
        else if(!side124[i] && side134[i]) mp.push_back(pos_b[i].v) ;
    }

    // See if edge of B goes through A 
    if(mp.size() > 0  && pm.size() > 0){
        for (int i =0; i < mp.size(); i++){
            vec3<OverlapReal> n = cross(pm[i],pos_a4.v);
            if(dot(n,n124)<0) n = -n;
            for (int j =0; j < mp.size(); j++){
                if(dot(n,mp[j]) < 0 ) mm = true;
            }
        }

        if(!mm) return false;
    }


    //move to Vertex 2 of A
     pos_u = conj(a_p2)*conj(a_p2);
     pos_a2 = conj(a_p2)*a_p2*a_p2*conj(a_p2);
     pos_a3 = conj(a_p2)*a_p3*a_p3*conj(a_p2);
     pos_a4 = conj(a_p2)*a_p4*a_p4*conj(a_p2);


     pos_b.resize(0);
     pos_b.push_back(conj(a_p2)*quat_l*b_p1*b_p1*quat_r*conj(a_p2));
     pos_b.push_back(conj(a_p2)*quat_l*b_p2*b_p2*quat_r*conj(a_p2));
     pos_b.push_back(conj(a_p2)*quat_l*b_p3*b_p3*quat_r*conj(a_p2));
     pos_b.push_back(conj(a_p2)*quat_l*b_p4*b_p4*quat_r*conj(a_p2));


    // Face (2,3,4) of A 
    vec3<OverlapReal> n234 = cross(pos_a3.v, pos_a4.v);
    if(dot(n234,pos_u.v) > 0) n234 = -n234;

    //inside or outside halfspace of (2,3,4)
    std::vector<bool> side234(4,false);

    for( int i = 0; i < side234.size(); i++){
        if(dot(n234,pos_b[i].v) > 0) side234[i] = true;
        std::cout << "side234 " << i << ":  " << side234[i] << std::endl;
    }

    if(side234[0] && side234[1] && side234[2] && side234[3]) return false;


    // See if vertex of B inside A 
    if( (!side123[0] && !side124[0] && !side134[0] && !side234[0] ) || (!side123[1] && !side124[1] && !side134[1] && !side234[1]) || (!side123[2] && !side124[2] && !side134[2] && !side234[2]) || (!side123[3] && !side124[3] && !side134[3] && !side234[3]) ) return true; 

    // Edge (2,3) of A 
    pm.resize(0);
    mp.resize(0);

    for( int i = 0; i < side123.size(); i++){
        if(!side123[i] && !side234[i] ){ 
            mm=true;
            break;
        }
        if(side123[i] && !side234[i] ) pm.push_back(pos_b[i].v);
        else if(!side123[i] && side234[i]) mp.push_back(pos_b[i].v) ;
    }

    // See if edge of B goes through A 
    if(mp.size() > 0 && pm.size() > 0){
        for (int i =0; i < mp.size(); i++){
            vec3<OverlapReal> n = cross(pm[i],pos_a3.v);
            if(dot(n,n234)<0) n = -n;
            for (int j =0; j < mp.size(); j++){
                if(dot(n,mp[j]) < 0 ) mm = true;
            }
        }

        if(!mm) return false;
    }
    
    // Edge (2,4) of A 
    pm.resize(0);
    mp.resize(0);

    for( int i = 0; i < side124.size(); i++){
        if(!side124[i] && !side234[i] ){ 
            mm=true;
            break;
        }
        if(side124[i] && !side234[i] ) pm.push_back(pos_b[i].v);
        else if(!side124[i] && side234[i]) mp.push_back(pos_b[i].v) ;
    }

    // See if edge of B goes through A 
    if(mp.size() > 0 && pm.size() > 0){
        for (int i =0; i < mp.size(); i++){
            vec3<OverlapReal> n = cross(pm[i],pos_a4.v);
            if(dot(n,n234)<0) n = -n;
            for (int j =0; j < mp.size(); j++){
                if(dot(n,mp[j]) < 0 ) mm = true;
            }
        }

        if(!mm) return false;
    }

    //move to Vertex 3 of A
     pos_u = conj(a_p3)*conj(a_p3);
     pos_a3 = conj(a_p3)*a_p3*a_p3*conj(a_p3);
     pos_a4 = conj(a_p3)*a_p4*a_p4*conj(a_p3);


     pos_b.resize(0);
     pos_b.push_back(conj(a_p3)*quat_l*b_p1*b_p1*quat_r*conj(a_p3));
     pos_b.push_back(conj(a_p3)*quat_l*b_p2*b_p2*quat_r*conj(a_p3));
     pos_b.push_back(conj(a_p3)*quat_l*b_p3*b_p3*quat_r*conj(a_p3));
     pos_b.push_back(conj(a_p3)*quat_l*b_p4*b_p4*quat_r*conj(a_p3));

     n234 = cross(pos_a3.v, pos_a4.v);
     if(dot(n234,pos_u.v) > 0) n234 = -n234;

    // Edge (3,4) of A 
    pm.resize(0);
    mp.resize(0);

    for( int i = 0; i < side134.size(); i++){
        if(!side134[i] && !side234[i] ){ 
            mm=true;
            break;
        }
        if(side134[i] && !side234[i] ) pm.push_back(pos_b[i].v);
        else if(!side134[i] && side234[i]) mp.push_back(pos_b[i].v) ;
    }

    // See if edge of B goes through A 
    if(mp.size() > 0 && pm.size() > 0){
        for (int i =0; i < mp.size(); i++){
            vec3<OverlapReal> n = cross(pm[i],pos_a4.v);
            if(dot(n,n234)<0) n = -n;
            for (int j =0; j < mp.size(); j++){
                if(dot(n,mp[j]) < 0 ) mm = true;
            }
        }

        if(!mm) return false;
    }

    return true;

    //exit(0);

//    CompositeSupportFuncHypersphere<SupportFuncA, SupportFuncB> S(sa, sb, quat_l, quat_r, hypersphere);
//    quat<OverlapReal> pos_u, pos_v0, pos_v1, pos_v2, pos_v3, pos_v4;
//    quat<OverlapReal> pos_ve1, pos_ve2, pos_ve3, pos_ve4;
//    quat<OverlapReal> pos_vs1, pos_vs2, pos_vs3, pos_vs4;
//    quat<OverlapReal> pos_v2_ref1, pos_v3_ref1, pos_v4_ref1;
//    quat<OverlapReal> v0, v1, v2, v3, v4;
//    quat<OverlapReal> v1_atu, v2_atu, v3_atu, v4_atu;
//    quat<OverlapReal> v01, v02, v03, v04, v12, v13, v14;
//    quat<OverlapReal> v01_atu, v02_atu, v03_atu, v04_atu, v12_atu, v13_atu, v14_atu;
//    quat<OverlapReal> pA,pB,n;
//    OverlapReal d;
//    OverlapReal R = hypersphere.getR();
//    const OverlapReal precision_tol = 1e-7;        // precision tolerance for single-precision floats near 1.0
//    const OverlapReal root_tol = 3e-4;   // square root of precision tolerance
//
//
//    pos_v0 = quat_l*quat_r;
//    if (fabs(pos_v0.v.x) < root_tol && fabs(pos_v0.v.y) < root_tol && fabs(pos_v0.v.z) < root_tol)
//        {
//        // Interior point is at origin => particles overlap
//        return true;
//        }
//        std::cout << "Passed check 1" << std::endl;
//
//    // Phase 1: Portal Discovery
//    // ------
//    // Find the origin ray v0 from the origin to an interior point of the Minkowski difference.
//    // The easiest origin ray is the position of b minus the position of a, or more simply:
//
//    pos_u = quat<OverlapReal>();
//    v0 = R*pos_v0; //pos_v0 - (R,0,0,0) but perp to (R,0,0,0)
//    v0.s =0;
//    n = -v0;
//
//    std::cout << "pos_v0: "<< pos_v0.s << " " << pos_v0.v.x << " " << pos_v0.v.y << " " << pos_v0.v.z << std::endl;
//    std::cout << "v0: "<< v0.s << " " << v0.v.x << " " << v0.v.y << " " << v0.v.z << std::endl;
//
//    // ------
//    // find a candidate portal of three support points
//    //
//    // find support v1 in the direction of the origin
//    pA = S(n,true); // should be guaranteed ||v1|| > 0 TO DOOOOOO
//    pB = S(n,false); // should be guaranteed ||v1|| > 0 TO DOOOOOO
//
//    pos_vs1 = pA*pA;
//    pos_ve1 = (quat_l*pB)*(pB*quat_r);
//    pos_v1 = conj(pA)*pos_ve1*conj(pA);
//    v1 = R*(pos_ve1 - dot(pos_vs1,pos_ve1)*pos_vs1);
//    v1_atu = parallel_transport(v1, pos_vs1, pos_u);
//    
//    std::cout << "pos_ve1: " << pos_ve1.s << " " << pos_ve1.v.x << " " << pos_ve1.v.y << " " << pos_ve1.v.z << std::endl;
//    std::cout << "pos_vs1: " << pos_vs1.s << " " << pos_vs1.v.x << " " << pos_vs1.v.y << " " << pos_vs1.v.z << std::endl;
//    std::cout << "v1: " << v1.s << " " << v1.v.x << " " << v1.v.y << " " << v1.v.z << std::endl;
//    std::cout << "v1_atu: " << v1_atu.s << " " << v1_atu.v.x << " " << v1_atu.v.y << " " << v1_atu.v.z << std::endl;
//    std::cout << "dot(v1_atu, v0): " << dot(v1_atu, v0) << std::endl;
//
//
//    /* if (dot(v1, v1 - v0) <= 0) // by convexity */
//
//    if (dot(v1_atu, v0) > OverlapReal(0.0))
//        return false;   // origin is outside v1 support plane
//
//        std::cout << "Passed check 2" << std::endl;
//
//    // find support v2 perpendicular to v0, v1 plane
//    n.v = cross(v1_atu.v, v0.v);
//    // cross product is zero if v0,v1 colinear with origin, but we have already determined origin is within v1 support
//    // plane. If origin is on a line between v1 and v0, particles overlap.
//    //if (dot(n, n) < tol)
//    if (fabs(n.v.x) < precision_tol && fabs(n.v.y) < precision_tol && fabs(n.v.z) < precision_tol)
//        return true;
//
//        std::cout << "Passed check 3" << std::endl;
//
//    std::cout << "n: " << n.s << " " << n.v.x << " " << n.v.y << " " << n.v.z << std::endl;
//
//    pA = S(n,true); // Convexity should guarantee ||v2|| > 0, but v2 == v1 may be possible in edge cases of {B}-{A}
//    pB = S(n,false); // Convexity should guarantee ||v2|| > 0, but v2 == v1 may be possible in edge cases of {B}-{A}
//
//    pos_vs2 = pA*pA;
//    pos_ve2 = (quat_l*pB)*(pB*quat_r);
//    pos_v2 = conj(pA)*pos_ve2*conj(pA);
//    v2 = R*(pos_ve2 - dot(pos_vs2,pos_ve2)*pos_vs2);
//    v2_atu = parallel_transport(v2, pos_vs2, pos_u);
//
//    std::cout << "pos_ve2: " << pos_ve2.s << " " << pos_ve2.v.x << " " << pos_ve2.v.y << " " << pos_ve2.v.z << std::endl;
//    std::cout << "pos_vs2: " << pos_vs2.s << " " << pos_vs2.v.x << " " << pos_vs2.v.y << " " << pos_vs2.v.z << std::endl;
//    std::cout << "v2: " << v2.s << " " << v2.v.x << " " << v2.v.y << " " << v2.v.z << std::endl;
//    std::cout << "v2_atu: " << v2_atu.s << " " << v2_atu.v.x << " " << v2_atu.v.y << " " << v2_atu.v.z << std::endl;
//    std::cout << "dot(v2_atu, v0): " << dot(v2_atu, v0) << std::endl;
//
//
//    // particles do not overlap if origin outside v2 support plane
//    if (dot(v2_atu, n) < OverlapReal(0.0))
//        return false;
//
//        std::cout << "Passed check 4" << std::endl;
//
//    // Find next support direction perpendicular to plane (v1,v0,v2)
//
//    v01 = R*(pos_v1 - dot(pos_v0,pos_v1)*pos_v0);
//    v02 = R*(pos_v2 - dot(pos_v0,pos_v2)*pos_v0);
//
//    v01_atu = parallel_transport(v01,pos_v0,pos_u);
//    v02_atu = parallel_transport(v02,pos_v0,pos_u);
//
//    n.v = cross(v01_atu.v, v02_atu.v);
//
//    std::cout << "v01: " << v01.s << " " << v01.v.x << " " << v01.v.y << " " << v01.v.z << std::endl;
//    std::cout << "v02: " << v02.s << " " << v02.v.x << " " << v02.v.y << " " << v02.v.z << std::endl;
//    std::cout << "v01_atu: " << v01_atu.s << " " << v01_atu.v.x << " " << v01_atu.v.y << " " << v01_atu.v.z << std::endl;
//    std::cout << "v02_atu: " << v02_atu.s << " " << v02_atu.v.x << " " << v02_atu.v.y << " " << v02_atu.v.z << std::endl;
//    std::cout << "n: " << n.s << " " << n.v.x << " " << n.v.y << " " << n.v.z << std::endl;
//    std::cout << "dot(v0,n): " << dot(v0, n) << std::endl;
//
//    // Maintain known handedness of the portal: make sure plane normal points towards origin
//    if (dot(n, v0) > OverlapReal(0.0))
//        {
//        v1.swap(v2);
//        pos_v1.swap(pos_v2);
//        pos_vs1.swap(pos_vs2);
//        pos_ve1.swap(pos_ve2);
//        v1_atu.swap(v2_atu);
//        v01.swap(v02);
//        v01_atu.swap(v02_atu);
//        n = -n;
//        }
//
//    // ------
//    // while (origin ray does not intersect candidate) choose new candidate
//    bool intersects = false;
//    unsigned int count = 0;
//    while (!intersects)
//        {
//        count++;
//
//        if (count >= XENOCOLLIDE_HYPERSPHERE_MAX_ITERATIONS)
//            {
//            err_count++;
//            return true;
//            }
//
//        std::cout << "Passed check 5" << std::endl;
//
//        // Get the next support point
//        pA = S(n,true);
//        pB = S(n,false);
//
//        pos_vs3 = pA*pA;
//        pos_ve3 = (quat_l*pB)*(pB*quat_r);
//        pos_v3 = conj(pA)*pos_ve3*conj(pA);
//        v3 = R*(pos_ve3 - dot(pos_vs3,pos_ve3)*pos_vs3);
//        v3_atu = parallel_transport(v3, pos_vs3, pos_u);
//
//        std::cout << "pos_ve3: " << pos_ve3.s << " " << pos_ve3.v.x << " " << pos_ve3.v.y << " " << pos_ve3.v.z << std::endl;
//        std::cout << "pos_vs3: " << pos_vs3.s << " " << pos_vs3.v.x << " " << pos_vs3.v.y << " " << pos_vs3.v.z << std::endl;
//        std::cout << "v3: " << v3.s << " " << v3.v.x << " " << v3.v.y << " " << v3.v.z << std::endl;
//        std::cout << "v3_atu: " << v3_atu.s << " " << v3_atu.v.x << " " << v3_atu.v.y << " " << v3_atu.v.z << std::endl;
//        std::cout << "dot(v3_atu, v0): " << dot(v2_atu, v0) << std::endl;
//
//        if (dot(v3_atu, n) <= 0)
//            return false; // check if origin outside v3 support plane
//
//        std::cout << "Passed check 6" << std::endl;
//
//        v03 = R*(pos_v3 - dot(pos_v0,pos_v3)*pos_v0);
//
//        v03_atu = parallel_transport(v03,pos_v0,pos_u);
//
//        std::cout << "v03: " << v03.s << " " << v03.v.x << " " << v03.v.y << " " << v03.v.z << std::endl;
//        std::cout << "v03_atu: " << v03_atu.s << " " << v03_atu.v.x << " " << v03_atu.v.y << " " << v03_atu.v.z << std::endl;
//        std::cout << dot(cross(v01_atu.v, v03_atu.v), v0.v) << std::endl;
//
//
//        // If origin lies on opposite side of a plane from the third support point, use outer-facing plane normal
//        // to find a new support point.
//        // Check (v3,v0,v1)
//        // if (dot(cross(v3 - v0, v1 - v0), -v0) < 0)
//        // -> if (dot(cross(v1 - v0, v3 - v0), v0) < 0)
//        // A little bit of algebra shows that dot(cross(a - c, b - c), c) == dot(cross(a, b), c)
//        if (dot(cross(v01_atu.v, v03_atu.v), v0.v) < OverlapReal(0.0))
//            {
//            // replace v2 and find new support direction
//            v2 = v3; // preserve handedness
//
//            pos_v2 = pos_v3;
//            pos_vs2 = pos_vs3;
//            v2_atu = v3_atu;
//            v02 = v03;
//            v02_atu = v03_atu;
//
//            n.v = cross(v01_atu.v, v02_atu.v);
//            continue; // continue iterating to find valid portal
//            }
//        // Check (v2, v0, v3)
//        if (dot(cross(v03_atu.v, v02_atu.v), v0.v) < OverlapReal(0.0))
//            {
//            // replace v1 and find new support direction
//            v1 = v3;
//            pos_v1 = pos_v3;
//            pos_vs1 = pos_vs3;
//            v1_atu = v3_atu;
//            v01 = v03;
//            v01_atu = v03_atu;
//
//            n.v = cross(v01_atu.v, v02_atu.v);
//            continue;
//            }
//
//        // If we've made it this far, we have a valid portal and can proceed to refine the portal
//        intersects = true;
//        }
//
//    // Phase 2: Portal Refinement
//    count = 0;
//
//
//    while (true)
//        {
//        count++;
//
//        // ----
//        // if (origin inside portal) return true
//
//        //at v1
//
//        pos_v2_ref1 = parallel_transport(pos_ve2, pos_vs2, pos_vs1);
//        //pos_v2_ref1 = parallel_transport(pos_v2_ref1, pos_ve1, pos_u);
//
//        pos_v3_ref1 = parallel_transport(pos_ve3, pos_vs3, pos_vs1);
//        //pos_v3_ref1 = parallel_transport(pos_v3_ref1, pos_ve1, pos_u);
//
//        v12 = R*(pos_v2_ref1 - dot(pos_ve1,pos_v2_ref1)*pos_ve1);
//        v12_atu = parallel_transport(v12,pos_ve1,pos_u);
//
//        v13 = R*(pos_v3_ref1 - dot(pos_ve1,pos_v3_ref1)*pos_ve1);
//        v13_atu = parallel_transport(v13,pos_ve1,pos_u);
//
//        n.v = cross(v12_atu.v, v13_atu.v); // by construction, this is the outer-facing normal
//
//        std::cout << "pos_v2_ref1: " << pos_v2_ref1.s << " " << pos_v2_ref1.v.x << " " << pos_v2_ref1.v.y << " " << pos_v2_ref1.v.z << std::endl;
//        std::cout << "pos_v3_ref1: " << pos_v3_ref1.s << " " << pos_v3_ref1.v.x << " " << pos_v3_ref1.v.y << " " << pos_v3_ref1.v.z << std::endl;
//        std::cout << "v12: " << v12.s << " " << v12.v.x << " " << v12.v.y << " " << v12.v.z << std::endl;
//        std::cout << "v13: " << v13.s << " " << v13.v.x << " " << v13.v.y << " " << v13.v.z << std::endl;
//        std::cout << "v12_atu: " << v12_atu.s << " " << v12_atu.v.x << " " << v12_atu.v.y << " " << v12_atu.v.z << std::endl;
//        std::cout << "v13_atu: " << v13_atu.s << " " << v13_atu.v.x << " " << v13_atu.v.y << " " << v13_atu.v.z << std::endl;
//        std::cout << "n: " << n.s << " " << n.v.x << " " << n.v.y << " " << n.v.z << std::endl;
//        std::cout << "dot(v1_atu,n): " << dot(v1_atu, n) << std::endl;
//
//
//        // check if origin is inside (or overlapping)
//        // the = is important, because in an MC simulation you are guaranteed to find cases where edges and or vertices
//        // touch exactly
//        if (dot(v1_atu, n) >= OverlapReal(0.0))
//            {
//            return true;
//            }
//
//        std::cout << "Passed check 7" << std::endl;
//
//        // ----
//        // find support in direction of portal's outer facing normal
//        pA = S(n,true);
//        pB = S(n,false);
//
//        pos_vs4 = pA*pA;
//        pos_ve4 = (quat_l*pB)*(pB*quat_r);
//        pos_v4 = conj(pA)*pos_ve4*conj(pA);
//        v4 = R*(pos_ve4 - dot(pos_vs4,pos_ve4)*pos_vs4);
//        v4_atu = parallel_transport(v4, pos_vs4, pos_u);
//
//        //std::cout << "pos_ve: " << pos_ve.s << " " << pos_ve.v.x << " " << pos_ve.v.y << " " << pos_ve.v.z << std::endl;
//        //std::cout << "pos_vs: " << pos_vs.s << " " << pos_vs.v.x << " " << pos_vs.v.y << " " << pos_vs.v.z << std::endl;
//        //std::cout << "v4: " << v4.s << " " << v4.v.x << " " << v4.v.y << " " << v4.v.z << std::endl;
//        //std::cout << "v4_atu: " << v4_atu.s << " " << v4_atu.v.x << " " << v4_atu.v.y << " " << v4_atu.v.z << std::endl;
//        //std::cout << dot(v4_atu, n) << std::endl;
//
//        // ----
//        // if (origin outside support plane) return false
//        if (dot(v4_atu, n) < OverlapReal(0.0))
//            {
//            return false;
//            }
//
//        std::cout << "Passed check 8" << std::endl;
//
//        pos_v4_ref1 = parallel_transport(pos_ve4, pos_vs4, pos_vs1);
//        //pos_v4_ref1 = parallel_transport(pos_v4_ref1, pos_vs1, pos_u);
//
//        v14 = R*(pos_v4_ref1 - dot(pos_ve1,pos_v4_ref1)*pos_ve1);
//
//        v14_atu = parallel_transport(v14,pos_ve1,pos_u);
//
//        // Perform tolerance checks
//        // are we within an epsilon of the surface of the shape? If yes, done, one way or another
//        const OverlapReal tol_multiplier = 10000;
//
//        n.v = cross(v12_atu.v, v13_atu.v);
//        d = dot(v14_atu * tol_multiplier, n);
//        OverlapReal tol = precision_tol * tol_multiplier * Ra * fast::sqrt(dot(n,n));
//
//        // First, check if v4 is on plane (v2,v1,v3)
//        if (fabs(d) < tol)
//            return false; // no more refinement possible, but not intersection detected
//
//        std::cout << "Passed check 9" << std::endl;
//
//        // Second, check if origin is on plane (v2,v1,v3) and has been missed by other checks
//        d = dot(v1_atu * tol_multiplier, n);
//        if (fabs(d) < tol)
//            return true;
//
//        std::cout << "Passed check 10" << std::endl;
//
//        if (count >= XENOCOLLIDE_HYPERSPHERE_MAX_ITERATIONS)
//            {
//            err_count++;
//            return true;
//            }
//
//        std::cout << "Passed check 11" << std::endl;
//
//        // ----
//        // Choose new portal. Two of its edges will be from the planes (v4,v0,v1), (v4,v0,v2), (v4,v0,v3). Find which
//        // two have the origin on the same side.
//        // MEI: As I understand this statement, I don't believe it is correct. An _inside_ needs to be defined and used.
//        // The only way I can think to do this is to consider all three pairs of planes to find which pair has the
//        // origin between them.
//        // Need to better understand and document this. The following code was directly adapted from example code.
//
//        // Test origin against the three planes that separate the new portal candidates: (v1,v4,v0) (v2,v4,v0) (v3,v4,v0)
//        // Note:  We're taking advantage of the triple product identities here as an optimization
//        //        (v1 % v4) * v0 == v1 * (v4 % v0)    > 0 if origin inside (v1, v4, v0)
//        //        (v2 % v4) * v0 == v2 * (v4 % v0)    > 0 if origin inside (v2, v4, v0)
//        //        (v3 % v4) * v0 == v3 * (v4 % v0)    > 0 if origin inside (v3, v4, v0)
//        vec3<OverlapReal> x = cross(v01_atu.v, v04_atu.v);
//        if (dot(v1_atu.v, x) > OverlapReal(0.0))
//            {
//            x = cross(v02_atu.v, v04_atu.v);
//            if (dot(v2_atu.v, x) > OverlapReal(0.0))
//            {
//                v1 = v4;    // Inside v1 & inside v2 ==> eliminate v1
//                pos_v1 = pos_v4;
//                pos_vs1 = pos_vs4;
//                pos_ve1 = pos_ve4;
//                v1_atu = v4_atu;
//                v01 = v04;
//                v01_atu = v04_atu;
//            }
//            else
//            {
//                v3 = v4;                   // Inside v1 & outside v2 ==> eliminate v3
//                pos_v3 = pos_v4;
//                pos_vs3 = pos_vs4;
//                pos_ve3 = pos_ve4;
//                v3_atu = v4_atu;
//                v03 = v04;
//                v03_atu = v04_atu;
//            }
//            }
//        else
//            {
//            x = cross(v03_atu.v, v04_atu.v);
//            if (dot(v3_atu.v, x) > OverlapReal(0.0))
//            {
//                v2 = v4;    // Outside v1 & inside v3 ==> eliminate v2
//                pos_v2 = pos_v4;
//                pos_vs2 = pos_vs4;
//                pos_ve2 = pos_ve4;
//                v2_atu = v4_atu;
//                v02 = v04;
//                v02_atu = v04_atu;
//            }
//            else
//            {
//                v1 = v4;                   // Outside v1 & outside v3 ==> eliminate v1
//                pos_v1 = pos_v4;
//                pos_vs1 = pos_vs4;
//                pos_ve1 = pos_ve4;
//                v1_atu = v4_atu;
//                v01 = v04;
//                v01_atu = v04_atu;
//            }
//            }
//
//        }
    }
} // end namespace hpmc::detail

}; // end namespace hpmc

#endif // __XENOCOLLIDE_HYPERSPHERE_H__
