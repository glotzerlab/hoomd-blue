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
DEVICE inline bool xenocollide_hypersphere(const SupportFuncA& a,const SupportFuncB& b,const quat<OverlapReal>& quat_l,const quat<OverlapReal>& quat_r,const Hypersphere& hypersphere,const OverlapReal Ra,unsigned int& err_count)
    {
    // This implementation of XenoCollide is hand-written from the description of the algorithm on page 171 of _Games
    // Programming Gems 7_


    std::vector<quat<OverlapReal> > a_p, b_p, pos_a, pos_b;
    std::vector<std::vector<unsigned int> > faces;
    quat<OverlapReal> pos_u;
    vec3<OverlapReal> n1;
    std::vector<bool> side_used (b.N,false);
    std::vector<std::vector<bool> > side;

    for ( int i = 0; i < a.N; i++) a_p.push_back( hypersphere.cartesianToHyperspherical(vec3<OverlapReal>(a.x[i],a.y[i],a.z[i])) );

    for ( int i = 0; i < b.N; i++) b_p.push_back( hypersphere.cartesianToHyperspherical(vec3<OverlapReal>(b.x[i],b.y[i],b.z[i])) );


    int j = 0;
    while (j < a.Nf){
        int jj = a.faces[j];
        std::vector<unsigned int> face;
        for (int i=1; i <= jj; i++)
    		face.push_back(a.faces[j+i]);
        faces.push_back(face);
        side.push_back(side_used);
        j = j + jj + 1;
    }
   

    unsigned int Nf = faces.size();
    side_used.resize(Nf, false);

    unsigned int all_faces = 0;

    pos_a.resize(a.N);
    pos_b.resize(b.N);

    unsigned int current_vertex = 100000;
    bool separate = false;
    for ( int ei = 0; ei < a.Ne; ei++){
        unsigned int vertex = a.edges[ei].x;
        unsigned int vertex1 = a.edges[ei].y;

        if( vertex != current_vertex){
            current_vertex = vertex;

            pos_u = hypersphere.hypersphericalToCartesian(conj(a_p[vertex]),conj(a_p[vertex]));
            for( int i = vertex+1; i < a.N; i++){
                pos_a[i] = hypersphere.hypersphericalToCartesian(conj(a_p[vertex])*a_p[i],a_p[i]*conj(a_p[vertex]));
            }

            for( int i = 0; i < b.N; i++){
                pos_b[i] = hypersphere.hypersphericalToCartesian(conj(a_p[vertex])*quat_l*b_p[i],b_p[i]*quat_r*conj(a_p[vertex]));
            }
        }


        // Face of A 
        unsigned int face1 = a.boundary_edges[ei].x;
        if(!side_used[face1]){
        	side_used[face1] = true;
                separate = true;
        	unsigned int vertex2=vertex1;
                int vi = 0;
        	while (vertex2 == vertex1 || vertex2 == vertex) {
                    vertex2 = faces[face1][vi];
                    vi++;
        	}
        	
        	n1 = cross(pos_a[vertex1].v, pos_a[vertex2].v);
                if(dot(n1,pos_u.v) > 0) n1 = -n1;

                for( int i = 0; i < b.N; i++){
                    if(dot(n1,pos_b[i].v) > 0) side[face1][i] = true;
                    else separate = false;
                }
                if(separate) return false;

        	all_faces++;
        	if(all_faces == Nf){
         	    for( int i = 0; i < b.N && !separate; i++){
         		separate = true;
         		for( int j = 0; j < Nf && separate; j++)
         		    if(side[j][i]) separate = false;
         	    }
         	    if(separate) return true;
        	}
        }


        unsigned int face2 = a.boundary_edges[ei].y;
        if(!side_used[face2]){
        	side_used[face2] = true;
                separate = true;
        	unsigned int vertex2=vertex1;
                int vi = 0;
        	while (vertex2 == vertex1 || vertex2 == vertex) {
                    vertex2 = faces[face2][vi];
                    vi++;
        	}
        	
        	n1 = cross(pos_a[vertex1].v, pos_a[vertex2].v);
                if(dot(n1,pos_u.v) > 0) n1 = -n1;

                for( int i = 0; i < b.N; i++){
                    if(dot(n1,pos_b[i].v) > 0) side[face2][i] = true;
                    else separate = false;
                }
                if(separate) return false;

        	all_faces++;
        	if(all_faces == Nf){
         	    for( int i = 0; i < b.N && !separate; i++){
         		separate = true;
         		for( int j = 0; j < Nf && separate; j++)
         		    if(side[j][i]) separate = false;
         	    }
         	    if(separate) return true;
                }
        }


        // Edge of A 
        separate = true;
        std::vector< vec3<OverlapReal> > pm;
        std::vector< vec3<OverlapReal> > mp;

        for( int i = 0; i < b.N; i++){
            if(!side[face1][i] && !side[face2][i] ){ 
                separate=false;
                break;
            }
            if(side[face1][i] && !side[face2][i] ) pm.push_back(pos_b[i].v);
            else if(!side[face1][i] && side[face2][i]) mp.push_back(pos_b[i].v) ;
        }


        // See if edge of B goes through A 
        if(separate){
            for (int i =0; i < pm.size() && separate; i++){
                vec3<OverlapReal> n = cross(pm[i],pos_a[vertex1].v);
                if(dot(n,pos_u.v)>0) n = -n;
                for (int j =0; j < mp.size() && separate; j++){
                    if(dot(n,mp[j]) < 0 ){ 
                        separate = false;
                    }
                }
            }

            if(separate) return false;
        }
    }
   
    current_vertex = 100000;
    for ( int fi = 0; fi < Nf; fi++){
        unsigned int vertex = faces[fi][0];
        unsigned int vertex1 = faces[fi][1];
        unsigned int vertex2 = faces[fi][2];
        if( vertex != current_vertex){
            current_vertex = vertex;

            pos_u = hypersphere.hypersphericalToCartesian(conj(b_p[vertex]),conj(b_p[vertex]));
            for( int i = 0; i < a.N; i++)
                pos_a[i] = hypersphere.hypersphericalToCartesian(conj(quat_l*b_p[vertex])*a_p[i],a_p[i]*conj(b_p[vertex]*quat_r));

            for( int i = 0; i < b.N; i++)
                pos_b[i] = hypersphere.hypersphericalToCartesian(conj(b_p[vertex])*b_p[i],b_p[i]*conj(b_p[vertex]));
        }

        separate = true;
        
        n1 = cross(pos_b[vertex1].v, pos_b[vertex2].v);
        if(dot(n1,pos_u.v) > 0) n1 = -n1;

        for( int i = 0; i < a.N && separate; i++){
            if(dot(n1,pos_a[i].v) < 0) separate=false;
        }
        if(separate) return false;
    }

    return true;

    }
} // end namespace hpmc::detail

}; // end namespace hpmc

#endif // __XENOCOLLIDE_HYPERSPHERE_H__
