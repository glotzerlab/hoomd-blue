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


     quat<OverlapReal> a_p1, a_p2, a_p3, a_p4, b_p1, b_p2, b_p3, b_p4;
     quat<OverlapReal> pos_u, pos_a2, pos_a3, pos_a4;
     std::vector< quat<OverlapReal> > pos_b;
     vec3<OverlapReal> n123, n124, n134, n234;
     std::vector<bool> side123(4,false);
     std::vector<bool> side124(4,false);
     std::vector<bool> side134(4,false);
     std::vector<bool> side234(4,false);
    std::vector< vec3<OverlapReal> > pm;
    std::vector< vec3<OverlapReal> > mp;
    bool mm;

     a_p1 =  hypersphere.cartesianToHyperspherical(vec3<OverlapReal>(a.x[0],a.y[0],a.z[0]));
     a_p2 =  hypersphere.cartesianToHyperspherical(vec3<OverlapReal>(a.x[1],a.y[1],a.z[1]));
     a_p3 =  hypersphere.cartesianToHyperspherical(vec3<OverlapReal>(a.x[2],a.y[2],a.z[2]));
     a_p4 =  hypersphere.cartesianToHyperspherical(vec3<OverlapReal>(a.x[3],a.y[3],a.z[3]));

     b_p1 =  hypersphere.cartesianToHyperspherical(vec3<OverlapReal>(b.x[0],b.y[0],b.z[0]));
     b_p2 =  hypersphere.cartesianToHyperspherical(vec3<OverlapReal>(b.x[1],b.y[1],b.z[1]));
     b_p3 =  hypersphere.cartesianToHyperspherical(vec3<OverlapReal>(b.x[2],b.y[2],b.z[2]));
     b_p4 =  hypersphere.cartesianToHyperspherical(vec3<OverlapReal>(b.x[3],b.y[3],b.z[3]));


    //move to Vertex 1 of A
     pos_u =  hypersphere.hypersphericalToCartesian(conj(a_p1),conj(a_p1));
     pos_a2 = hypersphere.hypersphericalToCartesian(conj(a_p1)*a_p2,a_p2*conj(a_p1));
     pos_a3 = hypersphere.hypersphericalToCartesian(conj(a_p1)*a_p3,a_p3*conj(a_p1));
     pos_a4 = hypersphere.hypersphericalToCartesian(conj(a_p1)*a_p4,a_p4*conj(a_p1));

     pos_b.push_back(hypersphere.hypersphericalToCartesian(conj(a_p1)*quat_l*b_p1,b_p1*quat_r*conj(a_p1)));
     pos_b.push_back(hypersphere.hypersphericalToCartesian(conj(a_p1)*quat_l*b_p2,b_p2*quat_r*conj(a_p1)));
     pos_b.push_back(hypersphere.hypersphericalToCartesian(conj(a_p1)*quat_l*b_p3,b_p3*quat_r*conj(a_p1)));
     pos_b.push_back(hypersphere.hypersphericalToCartesian(conj(a_p1)*quat_l*b_p4,b_p4*quat_r*conj(a_p1)));


    // Face (1,2,3) of A 
    n123 = cross(pos_a2.v, pos_a3.v);
    if(dot(n123,pos_u.v) > 0) n123 = -n123;

    //inside or outside halfspace of (1,2,3)
    for( int i = 0; i < side123.size(); i++){
        if(dot(n123,pos_b[i].v) > 0) side123[i] = true;
    }

    if(side123[0] && side123[1] && side123[2] && side123[3]) return false;

    // Face (1,2,4) of A 
    n124 = cross(pos_a2.v, pos_a4.v);
    if(dot(n124,pos_u.v) > 0) n124 = -n124;

    //inside or outside halfspace of (1,2,4)
    for( int i = 0; i < side124.size(); i++){
        if(dot(n124,pos_b[i].v) > 0) side124[i] = true;
    }

    if(side124[0] && side124[1] && side124[2] && side124[3]) return false;


    // Edge (1,2) of A 
    mm = false;

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
        for (int i =0; i < pm.size(); i++){
            vec3<OverlapReal> n = cross(pm[i],pos_a2.v);
            if(dot(n,pos_u.v)>0) n = -n;
            for (int j =0; j < mp.size(); j++){
                if(dot(n,mp[j]) < 0 ){ 
                    mm = true;
                }
            }
        }

        if(!mm) return false;
    }

    // Face (1,3,4) of A 
    n134 = cross(pos_a3.v, pos_a4.v);
    if(dot(n134,pos_u.v) > 0) n134 = -n134;

    //inside or outside halfspace of (1,3,4)
    for( int i = 0; i < side134.size(); i++){
        if(dot(n134,pos_b[i].v) > 0) side134[i] = true;
    }

    if(side134[0] && side134[1] && side134[2] && side134[3]) return false;

    // Edge (1,3) of A 
    pm.resize(0);
    mp.resize(0);
    mm = false;

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
        for (int i =0; i < pm.size(); i++){
            vec3<OverlapReal> n = cross(pm[i],pos_a3.v);
            if(dot(n,pos_u.v)>0) n = -n;
            for (int j =0; j < mp.size(); j++){
                if(dot(n,mp[j]) < 0 ){ 
                    mm = true;
                }
            }
        }

        if(!mm) return false;
    }

    // Edge (1,4) of A 
    pm.resize(0);
    mp.resize(0);
    mm = false;

    for( int i = 0; i < side124.size(); i++){
        if(!side124[i] && !side134[i] ){ 
            mm=true;
            break;
        }
        if(side124[i] && !side134[i] ) pm.push_back(pos_b[i].v);
        else if(!side124[i] && side134[i]) mp.push_back(pos_b[i].v) ;
    }

    // See if edge of B goes through A 
    if(mp.size() > 0  && pm.size() > 0 && !mm){
        for (int i =0; i < pm.size(); i++){
            vec3<OverlapReal> n = cross(pm[i],pos_a4.v);
            if(dot(n,pos_u.v)>0) n = -n;
            for (int j =0; j < mp.size(); j++){
                if(dot(n,mp[j]) < 0 ){ 
                    mm = true;
                }
            }
        }

        if(!mm) return false;
    }


    //move to Vertex 2 of A
     pos_u =  hypersphere.hypersphericalToCartesian(conj(a_p2),conj(a_p2));
     pos_a3 = hypersphere.hypersphericalToCartesian(conj(a_p2)*a_p3,a_p3*conj(a_p2));
     pos_a4 = hypersphere.hypersphericalToCartesian(conj(a_p2)*a_p4,a_p4*conj(a_p2));

     pos_b.resize(0);
     pos_b.push_back(hypersphere.hypersphericalToCartesian(conj(a_p2)*quat_l*b_p1,b_p1*quat_r*conj(a_p2)));
     pos_b.push_back(hypersphere.hypersphericalToCartesian(conj(a_p2)*quat_l*b_p2,b_p2*quat_r*conj(a_p2)));
     pos_b.push_back(hypersphere.hypersphericalToCartesian(conj(a_p2)*quat_l*b_p3,b_p3*quat_r*conj(a_p2)));
     pos_b.push_back(hypersphere.hypersphericalToCartesian(conj(a_p2)*quat_l*b_p4,b_p4*quat_r*conj(a_p2)));




    // Face (2,3,4) of A 
    n234 = cross(pos_a3.v, pos_a4.v);
    if(dot(n234,pos_u.v) > 0) n234 = -n234;

    //inside or outside halfspace of (2,3,4)
    for( int i = 0; i < side234.size(); i++){
        if(dot(n234,pos_b[i].v) > 0) side234[i] = true;
    }

    if(side234[0] && side234[1] && side234[2] && side234[3]) return false;


    // See if vertex of B inside A 
    if( (!side123[0] && !side124[0] && !side134[0] && !side234[0] ) || (!side123[1] && !side124[1] && !side134[1] && !side234[1]) || (!side123[2] && !side124[2] && !side134[2] && !side234[2]) || (!side123[3] && !side124[3] && !side134[3] && !side234[3]) ) return true; 

    // Edge (2,3) of A 
    pm.resize(0);
    mp.resize(0);
    mm = false;

    for( int i = 0; i < side234.size(); i++){
        if(!side234[i] && !side123[i] ){ 
            mm=true;
            break;
        }
        if(side234[i] && !side123[i] ) pm.push_back(pos_b[i].v);
        else if(!side234[i] && side123[i]) mp.push_back(pos_b[i].v) ;
    }

    // See if edge of B goes through A 
    if(mp.size() > 0 && pm.size() > 0 && !mm){
        for (int i =0; i < pm.size(); i++){
            vec3<OverlapReal> n = cross(pm[i],pos_a3.v);
            if(dot(n,pos_u.v)>0) n = -n;
            for (int j =0; j < mp.size(); j++){
                if(dot(n,mp[j]) < 0 ){ 
                    mm = true;
                }
            }
        }

        if(!mm) return false;
    }
    
    // Edge (2,4) of A 
    pm.resize(0);
    mp.resize(0);
    mm = false;

    for( int i = 0; i < side234.size(); i++){
        if(!side234[i] && !side124[i] ){ 
            mm=true;
            break;
        }
        if(side234[i] && !side124[i] ) pm.push_back(pos_b[i].v);
        else if(!side234[i] && side124[i]) mp.push_back(pos_b[i].v) ;
    }

    // See if edge of B goes through A 
    if(mp.size() > 0 && pm.size() > 0 && !mm){
        for (int i =0; i < pm.size(); i++){
            vec3<OverlapReal> n = cross(pm[i],pos_a4.v);
            if(dot(n,pos_u.v)>0) n = -n;
            for (int j =0; j < mp.size(); j++){
                if(dot(n,mp[j]) < 0 ){ 
                    mm = true;
                }
            }
        }

        if(!mm) return false;
    }

    //move to Vertex 3 of A
     pos_u =  hypersphere.hypersphericalToCartesian(conj(a_p3),conj(a_p3));
     pos_a2 = hypersphere.hypersphericalToCartesian(conj(a_p3)*a_p2,a_p2*conj(a_p3));
     pos_a4 = hypersphere.hypersphericalToCartesian(conj(a_p3)*a_p4,a_p4*conj(a_p3));

     pos_b.resize(0);
     pos_b.push_back(hypersphere.hypersphericalToCartesian(conj(a_p3)*quat_l*b_p1,b_p1*quat_r*conj(a_p3)));
     pos_b.push_back(hypersphere.hypersphericalToCartesian(conj(a_p3)*quat_l*b_p2,b_p2*quat_r*conj(a_p3)));
     pos_b.push_back(hypersphere.hypersphericalToCartesian(conj(a_p3)*quat_l*b_p3,b_p3*quat_r*conj(a_p3)));
     pos_b.push_back(hypersphere.hypersphericalToCartesian(conj(a_p3)*quat_l*b_p4,b_p4*quat_r*conj(a_p3)));


    // Edge (3,4) of A 
    pm.resize(0);
    mp.resize(0);
    mm = false;

    for( int i = 0; i < side234.size(); i++){
        if(!side234[i] && !side134[i] ){ 
            mm=true;
            break;
        }
        if(side234[i] && !side134[i] ) pm.push_back(pos_b[i].v);
        else if(!side234[i] && side134[i]) mp.push_back(pos_b[i].v) ;
    }

    // See if edge of B goes through A 
    if(mp.size() > 0 && pm.size() > 0 && !mm){
        for (int i =0; i < pm.size(); i++){
            vec3<OverlapReal> n = cross(pm[i],pos_a4.v);
            if(dot(n,pos_u.v)>0) n = -n;
            for (int j =0; j < mp.size(); j++){
                if(dot(n,mp[j]) < 0 ){ 
                    mm = true;
                }
            }
        }

        if(!mm) return false;
    }


    //move to Vertex 1 of B
     pos_u =  hypersphere.hypersphericalToCartesian(conj(b_p1),conj(b_p1));
     pos_a2 = hypersphere.hypersphericalToCartesian(conj(b_p1)*b_p2,b_p2*conj(b_p1));
     pos_a3 = hypersphere.hypersphericalToCartesian(conj(b_p1)*b_p3,b_p3*conj(b_p1));
     pos_a4 = hypersphere.hypersphericalToCartesian(conj(b_p1)*b_p4,b_p4*conj(b_p1));

     pos_b.resize(0);
     pos_b.push_back(hypersphere.hypersphericalToCartesian(conj(quat_l*b_p1)*a_p1,a_p1*conj(b_p1*quat_r)));
     pos_b.push_back(hypersphere.hypersphericalToCartesian(conj(quat_l*b_p1)*a_p2,a_p2*conj(b_p1*quat_r)));
     pos_b.push_back(hypersphere.hypersphericalToCartesian(conj(quat_l*b_p1)*a_p3,a_p3*conj(b_p1*quat_r)));
     pos_b.push_back(hypersphere.hypersphericalToCartesian(conj(quat_l*b_p1)*a_p4,a_p4*conj(b_p1*quat_r)));

    // Face (1,2,3) of AB
    n123 = cross(pos_a2.v, pos_a3.v);
    if(dot(n123,pos_u.v) > 0) n123 = -n123;

    //inside or outside halfspace of (1,2,3)
    for( int i = 0; i < side123.size(); i++){
        if(dot(n123,pos_b[i].v) > 0) side123[i] = true;
        else side123[i] = false;
    }

    if(side123[0] && side123[1] && side123[2] && side123[3]) return false;

    // Face (1,2,4) of A 
    n124 = cross(pos_a2.v, pos_a4.v);
    if(dot(n124,pos_u.v) > 0) n124 = -n124;

    //inside or outside halfspace of (1,2,4)
    for( int i = 0; i < side124.size(); i++){
        if(dot(n124,pos_b[i].v) > 0) side124[i] = true;
        else side124[i] = false;
    }

    if(side124[0] && side124[1] && side124[2] && side124[3]) return false;

    // Face (1,3,4) of A 
    n134 = cross(pos_a3.v, pos_a4.v);
    if(dot(n134,pos_u.v) > 0) n134 = -n134;

    //inside or outside halfspace of (1,2,4)
    for( int i = 0; i < side134.size(); i++){
        if(dot(n134,pos_b[i].v) > 0) side134[i] = true;
        else side134[i] = false;
    }

    if(side134[0] && side134[1] && side134[2] && side134[3]) return false;

    //move to Vertex 2 of B
     pos_u = conj(b_p2)*conj(b_p2);
     pos_a3 = conj(b_p2)*b_p3*b_p3*conj(b_p2);
     pos_a4 = conj(b_p2)*b_p4*b_p4*conj(b_p2);

     pos_b.resize(0);
     pos_b.push_back(conj(quat_l*b_p2)*a_p1*a_p1*conj(b_p2*quat_r));
     pos_b.push_back(conj(quat_l*b_p2)*a_p2*a_p2*conj(b_p2*quat_r));
     pos_b.push_back(conj(quat_l*b_p2)*a_p3*a_p3*conj(b_p2*quat_r));
     pos_b.push_back(conj(quat_l*b_p2)*a_p4*a_p4*conj(b_p2*quat_r));

    // Face (2,3,4) of B
    n234 = cross(pos_a3.v, pos_a4.v);
    if(dot(n234,pos_u.v) > 0) n234 = -n234;

    //inside or outside halfspace of (2,3,4)
    for( int i = 0; i < side234.size(); i++){
        if(dot(n234,pos_b[i].v) > 0) side234[i] = true;
        else side234[i] = false;
    }

    if(side234[0] && side234[1] && side234[2] && side234[3]) return false;

    return true;


    }
} // end namespace hpmc::detail

}; // end namespace hpmc

#endif // __XENOCOLLIDE_HYPERSPHERE_H__
