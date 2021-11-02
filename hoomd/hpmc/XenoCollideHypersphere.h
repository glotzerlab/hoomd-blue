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
DEVICE inline bool xenocollide_hypersphere(const SupportFuncA& a,const SupportFuncB& b,quat<OverlapReal>& quat_l,quat<OverlapReal>& quat_r,const Hypersphere& hypersphere,const OverlapReal Ra,unsigned int& err_count)
    {
    // This implementation of XenoCollide is hand-written from the description of the algorithm on page 171 of _Games
    // Programming Gems 7_


    std::vector<quat<OverlapReal> > a_p, b_p, pos_a, pos_b;
    std::vector<std::vector<unsigned int> > faces;
    quat<OverlapReal> pos_u;
    vec3<OverlapReal> n1;
    std::vector<bool> side_used (b.N,false);
    std::vector<std::vector<bool> > side;
    bool nearlyoverlap= false;

    for ( int i = 0; i < a.N; i++) a_p.push_back( hypersphere.cartesianToHyperspherical(vec3<OverlapReal>(a.x[i],a.y[i],a.z[i])) );

    for ( int i = 0; i < b.N; i++) b_p.push_back( hypersphere.cartesianToHyperspherical(vec3<OverlapReal>(b.x[i],b.y[i],b.z[i])) );


    int j = 0;
    while (j < a.Nf){
        int jj = a.faces[j];
        std::vector<unsigned int> face;
        for (int i=1; i <= jj; i++){
    		face.push_back(a.faces[j+i]);
		}
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
            for( int i = 0; i < a.N; i++){
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
                if(separate){
		 	//std::cout << "Return 1" << std::endl;
			return false;
			}

        	all_faces++;
        	if(all_faces == Nf){
         	    for( int i = 0; i < b.N && !separate; i++){
         		separate = true;
         		for( int j = 0; j < Nf && separate; j++)
         		    if(side[j][i]) separate = false;
         	    }
         	    if(separate){ 
		 	//std::cout << "Return 2" << std::endl;
			return true;
			}
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
                if(separate){ 
		 	//std::cout << "Return 3" << std::endl;
			return false;
		}

        	all_faces++;
        	if(all_faces == Nf){
         	    for( int i = 0; i < b.N && !separate; i++){
         		separate = true;
         		for( int j = 0; j < Nf && separate; j++)
         		    if(side[j][i]) separate = false;
         	    }
         	    if(separate){ 
		 	//std::cout << "Return 4" << std::endl;
			return true;
			}
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
                    OverlapReal dnmp = dot(n,mp[j]);
                    if(fabs(dnmp) < 1e-6) nearlyoverlap = true;
                    if(dnmp < 0 ){ 
                        separate = false;
                    }
                }
            }

            if(separate){
		 //std::cout << "Return 5" << std::endl;
		 return false;
		}
        }
    }

    quat_l = conj(quat_l);
    quat_r = conj(quat_r);

    current_vertex = 100000;

    for( int i = 0; i < side_used.size(); i++) side_used[i] =false;

    for( int i = 0; i < side.size(); i++) 
        	for( int j = 0; j < side[i].size(); j++) side[i][j] =false;


    if(nearlyoverlap){
        all_faces = 0;

        for ( int ei = 0; ei < b.Ne; ei++){
            unsigned int vertex = b.edges[ei].x;
            unsigned int vertex1 = b.edges[ei].y;

            if( vertex != current_vertex){
                current_vertex = vertex;

                pos_u = hypersphere.hypersphericalToCartesian(conj(b_p[vertex]),conj(b_p[vertex]));
                for( int i = 0; i < b.N; i++){
                    pos_b[i] = hypersphere.hypersphericalToCartesian(conj(b_p[vertex])*b_p[i],b_p[i]*conj(b_p[vertex]));
                }

                for( int i = 0; i < a.N; i++){
                    pos_a[i] = hypersphere.hypersphericalToCartesian(conj(b_p[vertex])*quat_l*a_p[i],a_p[i]*quat_r*conj(b_p[vertex]));
                }
            }


            // Face of A 
            unsigned int face1 = b.boundary_edges[ei].x;
            if(!side_used[face1]){
            	side_used[face1] = true;
                    separate = true;
            	unsigned int vertex2=vertex1;
                    int vi = 0;
            	while (vertex2 == vertex1 || vertex2 == vertex) {
                        vertex2 = faces[face1][vi];
                        vi++;
            	}
            	
            	n1 = cross(pos_b[vertex1].v, pos_b[vertex2].v);
                    if(dot(n1,pos_u.v) > 0) n1 = -n1;

                    for( int i = 0; i < a.N; i++){
                        if(dot(n1,pos_a[i].v) > 0) side[face1][i] = true;
                        else separate = false;
                    }
                    if(separate){ 
			return false;
		 	//std::cout << "Return 6" << std::endl;
		    }

            	all_faces++;
            	if(all_faces == Nf){
             	    for( int i = 0; i < a.N && !separate; i++){
             		separate = true;
             		for( int j = 0; j < Nf && separate; j++)
             		    if(side[j][i]) separate = false;
             	    }
             	    if(separate){
		 	//std::cout << "Return 7" << std::endl;
			return true;
			}
            	}
            }


            unsigned int face2 = b.boundary_edges[ei].y;
            if(!side_used[face2]){
            	side_used[face2] = true;
                    separate = true;
            	unsigned int vertex2=vertex1;
                    int vi = 0;
            	while (vertex2 == vertex1 || vertex2 == vertex) {
                        vertex2 = faces[face2][vi];
                        vi++;
            	}
            	
            	n1 = cross(pos_b[vertex1].v, pos_b[vertex2].v);
                    if(dot(n1,pos_u.v) > 0) n1 = -n1;

                    for( int i = 0; i < a.N; i++){
                        if(dot(n1,pos_a[i].v) > 0) side[face2][i] = true;
                        else separate = false;
                    }
                    if(separate){ 
		 	//std::cout << "Return 8" << std::endl;
			return false;
			}

            	all_faces++;
            	if(all_faces == Nf){
             	    for( int i = 0; i < a.N && !separate; i++){
             		separate = true;
             		for( int j = 0; j < Nf && separate; j++)
             		    if(side[j][i]) separate = false;
             	    }
             	    if(separate){ 
		 	//std::cout << "Return 9" << std::endl;
			return true;
			}
                    }
            }

            // Edge of A 
            separate = true;
            std::vector< vec3<OverlapReal> > pm;
            std::vector< vec3<OverlapReal> > mp;

            for( int i = 0; i < a.N; i++){
                if(!side[face1][i] && !side[face2][i] ){ 
                    separate=false;
                    break;
                }
                if(side[face1][i] && !side[face2][i] ) pm.push_back(pos_a[i].v);
                else if(!side[face1][i] && side[face2][i]) mp.push_back(pos_a[i].v) ;
            }


            // See if edge of B goes through A 
            if(separate){
                for (int i =0; i < pm.size() && separate; i++){
                    vec3<OverlapReal> n = cross(pm[i],pos_b[vertex1].v);
                    if(dot(n,pos_u.v)>0) n = -n;
                    for (int j =0; j < mp.size() && separate; j++){
                        if(dot(n,mp[j]) < 0 ){ 
                            separate = false;
                        }
                    }
                }

                if(separate){ 
		 	//std::cout << "Return 10" << std::endl;
			return false;
		}
            }
        }
    }else{
        for ( int ei = 0; ei < b.Ne; ei++){
            unsigned int vertex = b.edges[ei].x;
            unsigned int vertex1 = b.edges[ei].y;

            if( vertex != current_vertex){
                current_vertex = vertex;

                pos_u = hypersphere.hypersphericalToCartesian(conj(b_p[vertex]),conj(b_p[vertex]));
                for( int i = 0; i < b.N; i++){
                    pos_b[i] = hypersphere.hypersphericalToCartesian(conj(b_p[vertex])*b_p[i],b_p[i]*conj(b_p[vertex]));
                }

                for( int i = 0; i < a.N; i++){
                    pos_a[i] = hypersphere.hypersphericalToCartesian(conj(b_p[vertex])*quat_l*a_p[i],a_p[i]*quat_r*conj(b_p[vertex]));
                }
            }


            // Face of A 
            unsigned int face1 = b.boundary_edges[ei].x;
            if(!side_used[face1]){
            	side_used[face1] = true;
                separate = true;
            	unsigned int vertex2=vertex1;
                int vi = 0;
            	while (vertex2 == vertex1 || vertex2 == vertex) {
                        vertex2 = faces[face1][vi];
                        vi++;
            	}
            	
            	n1 = cross(pos_b[vertex1].v, pos_b[vertex2].v);
                if(dot(n1,pos_u.v) > 0) n1 = -n1;

                for( int i = 0; i < a.N && separate; i++){
                    if(dot(n1,pos_a[i].v) < 0) separate = false;
                }
                if(separate){ 
		 	//std::cout << "Return 11" << std::endl;
			return false;
			}

           }


            unsigned int face2 = b.boundary_edges[ei].y;
            if(!side_used[face2]){
            	side_used[face2] = true;
                separate = true;
            	unsigned int vertex2=vertex1;
                int vi = 0;
            	while (vertex2 == vertex1 || vertex2 == vertex) {
                        vertex2 = faces[face2][vi];
                        vi++;
            	}
            	
            	n1 = cross(pos_b[vertex1].v, pos_b[vertex2].v);
                if(dot(n1,pos_u.v) > 0) n1 = -n1;

                for( int i = 0; i < a.N && separate; i++){
                    if(dot(n1,pos_a[i].v) < 0) separate = false;
                }
                if(separate){ 
		 	//std::cout << "Return 12" << std::endl;
			return false;
			}
            }

        }
    }
//std::cout << "Return 13" << std::endl;
    return true;

    }



template<class SupportFuncA, class SupportFuncB>
DEVICE inline bool xenocollide_hypersphereTetra(const SupportFuncA& a,const SupportFuncB& b,quat<OverlapReal>& quat_l,quat<OverlapReal>& quat_r,const Hypersphere& hypersphere,const OverlapReal Ra,unsigned int& err_count)
    {
    // This implementation of XenoCollide is hand-written from the description of the algorithm on page 171 of _Games
    // Programming Gems 7_

     quat<OverlapReal>  pos_a2, pos_a3, pos_a4;
     vec3<OverlapReal> n123, n124, n134, n234;
     std::vector<bool> side123(4,false);
     std::vector<bool> side124(4,false);
     std::vector<bool> side134(4,false);
     std::vector<bool> side234(4,false);
     std::vector< vec3<OverlapReal> > pm;
     std::vector< vec3<OverlapReal> > mp;
     bool mm, outside123, outside124, outside134, outside234;

     std::vector<quat<OverlapReal> > pos_b;
     quat<OverlapReal> pos_u;


     quat<OverlapReal> a_p1 =  hypersphere.cartesianToHyperspherical(vec3<OverlapReal>(a.x[0],a.y[0],a.z[0]));
     quat<OverlapReal> a_p2 =  hypersphere.cartesianToHyperspherical(vec3<OverlapReal>(a.x[1],a.y[1],a.z[1]));
     quat<OverlapReal> a_p3 =  hypersphere.cartesianToHyperspherical(vec3<OverlapReal>(a.x[2],a.y[2],a.z[2]));
     quat<OverlapReal> a_p4 =  hypersphere.cartesianToHyperspherical(vec3<OverlapReal>(a.x[3],a.y[3],a.z[3]));
    
     quat<OverlapReal> b_p1 =  hypersphere.cartesianToHyperspherical(vec3<OverlapReal>(b.x[0],b.y[0],b.z[0]));
     quat<OverlapReal> b_p2 =  hypersphere.cartesianToHyperspherical(vec3<OverlapReal>(b.x[1],b.y[1],b.z[1]));
     quat<OverlapReal> b_p3 =  hypersphere.cartesianToHyperspherical(vec3<OverlapReal>(b.x[2],b.y[2],b.z[2]));
     quat<OverlapReal> b_p4 =  hypersphere.cartesianToHyperspherical(vec3<OverlapReal>(b.x[3],b.y[3],b.z[3]));


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
     
     mm = true;
     outside123 = false;
     //inside or outside halfspace of (1,2,3)
     for( int i = 0; i < side123.size(); i++){
         if(dot(n123,pos_b[i].v) > 0){ 
		side123[i] = true;
		outside123 = true;
	 }
	 else mm = false;
     }
     
     if(mm) return false;
     
     // Face (1,2,4) of A 
     n124 = cross(pos_a2.v, pos_a4.v);
     if(dot(n124,pos_u.v) > 0) n124 = -n124;
     
     mm = true;
     outside124 = false;
     //inside or outside halfspace of (1,2,4)
     for( int i = 0; i < side124.size(); i++){
         if(dot(n124,pos_b[i].v) > 0){ 
		side124[i] = true;
		outside124 = true;
	 }
	 else mm = false;
     }
     
     if(mm) return false;


     // Edge (1,2) of A 

     if(outside123 && outside124){

     	mm = true;
     	
     	
     	for( int i = 0; i < side124.size(); i++){
     	    if(!side123[i] && !side124[i] ){ 
     	        mm=false;
     	        break;
     	    }
     	    if(side123[i] && !side124[i] ) pm.push_back(pos_b[i].v);
     	    else if(!side123[i] && side124[i]) mp.push_back(pos_b[i].v) ;
     	}
     	
     	// See if edge of B goes through A 
     	if(mp.size() > 0  && pm.size() > 0 && mm){
     	    for (int i =0; i < pm.size() && mm; i++){
     	        vec3<OverlapReal> n = cross(pm[i],pos_a2.v);
     	        if(dot(n,pos_u.v)>0) n = -n;
     	        for (int j =0; j < mp.size() && mm; j++){
     	            if(dot(n,mp[j]) < 0 ){ 
     	                mm = false;
     	            }
     	        }
     	    }
     	
     	    if(mm) return false;
     	}
     	pm.resize(0);
     	mp.resize(0);
     }

     // Face (1,3,4) of A 
     n134 = cross(pos_a3.v, pos_a4.v);
     if(dot(n134,pos_u.v) > 0) n134 = -n134;
     
     mm = true;
     outside134 = false;
     //inside or outside halfspace of (1,3,4)
     for( int i = 0; i < side134.size(); i++){
         if(dot(n134,pos_b[i].v) > 0){ 
		side134[i] = true;
		outside134 = true;
	 }
	 else mm = false;
     }
     
     if(mm) return false;
     
     if(outside134){

     	// Edge (1,3) of A 
	if(outside123){
     		mm = true;
     		
     		for( int i = 0; i < side123.size(); i++){
     		    if(!side123[i] && !side134[i] ){ 
     		        mm=false;
     		        break;
     		    }
     		    if(side123[i] && !side134[i] ) pm.push_back(pos_b[i].v);
     		    else if(!side123[i] && side134[i]) mp.push_back(pos_b[i].v) ;
     		}

     		// See if edge of B goes through A 
     		if(mp.size() > 0  && pm.size() > 0 && mm){
     		    for (int i =0; i < pm.size() && mm; i++){
     		        vec3<OverlapReal> n = cross(pm[i],pos_a3.v);
     		        if(dot(n,pos_u.v)>0) n = -n;
     		        for (int j =0; j < mp.size() && mm; j++){
     		            if(dot(n,mp[j]) < 0 ){ 
     		                mm = false;
     		            }
     		        }
     		    }
     		
     		    if(mm) return false;
     		}
     		pm.resize(0);
     		mp.resize(0);
	}

     	// Edge (1,4) of A 
	if(outside124){
     		mm = true;
     		
     		for( int i = 0; i < side124.size(); i++){
     		    if(!side124[i] && !side134[i] ){ 
     		        mm=false;
     		        break;
     		    }
     		    if(side124[i] && !side134[i] ) pm.push_back(pos_b[i].v);
     		    else if(!side124[i] && side134[i]) mp.push_back(pos_b[i].v) ;
     		}
     		
     		// See if edge of B goes through A 
     		if(mp.size() > 0  && pm.size() > 0 && mm){
     		    for (int i =0; i < pm.size() && mm; i++){
     		        vec3<OverlapReal> n = cross(pm[i],pos_a4.v);
     		        if(dot(n,pos_u.v)>0) n = -n;
     		        for (int j =0; j < mp.size() && mm; j++){
     		            if(dot(n,mp[j]) < 0 ){ 
     		                mm = false;
     		            }
     		        }
     		    }
     		
     		    if(mm) return false;
     		}
     		pm.resize(0);
     		mp.resize(0);
	}
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
     
     mm = true;
     outside234 = false;
     //inside or outside halfspace of (2,3,4)
     for( int i = 0; i < side234.size(); i++){
         if(dot(n234,pos_b[i].v) > 0){ 
		side234[i] = true;
		outside234 = true;
	 }
	 else mm = false;
     }
     
     if(mm) return false;
     
     
     // See if vertex of B inside A 
     if( (!side123[0] && !side124[0] && !side134[0] && !side234[0] ) || (!side123[1] && !side124[1] && !side134[1] && !side234[1]) || (!side123[2] && !side124[2] && !side134[2] && !side234[2]) || (!side123[3] && !side124[3] && !side134[3] && !side234[3]) ) return true; 
     
     
     

     if(outside234){

        // Edge (2,3) of A 
     	if(outside123){
     		mm = true;
     		
     		for( int i = 0; i < side234.size(); i++){
     		    if(!side234[i] && !side123[i] ){ 
     		        mm=false;
     		        break;
     		    }
     		    if(side234[i] && !side123[i] ) pm.push_back(pos_b[i].v);
     		    else if(!side234[i] && side123[i]) mp.push_back(pos_b[i].v) ;
     		}
     		
     		// See if edge of B goes through A 
     		if(mp.size() > 0 && pm.size() > 0 && mm){
     		    for (int i =0; i < pm.size() && mm; i++){
     		        vec3<OverlapReal> n = cross(pm[i],pos_a3.v);
     		        if(dot(n,pos_u.v)>0) n = -n;
     		        for (int j =0; j < mp.size() && mm; j++){
     		            if(dot(n,mp[j]) < 0 ){ 
     		                mm = false;
     		            }
     		        }
     		    }
     		
     		    if(mm) return false;
     		}
     		pm.resize(0);
     		mp.resize(0);
	}

     	// Edge (2,4) of A 
     	if(outside124){
     		mm = true;
     		
     		for( int i = 0; i < side234.size(); i++){
     		    if(!side234[i] && !side124[i] ){ 
     		        mm=false;
     		        break;
     		    }
     		    if(side234[i] && !side124[i] ) pm.push_back(pos_b[i].v);
     		    else if(!side234[i] && side124[i]) mp.push_back(pos_b[i].v) ;
     		}
     		
     		// See if edge of B goes through A 
     		if(mp.size() > 0 && pm.size() > 0 && mm){
     		    for (int i =0; i < pm.size() && mm; i++){
     		        vec3<OverlapReal> n = cross(pm[i],pos_a4.v);
     		        if(dot(n,pos_u.v)>0) n = -n;
     		        for (int j =0; j < mp.size() && mm; j++){
     		            if(dot(n,mp[j]) < 0 ){ 
     		                mm = false;
     		            }
     		        }
     		    }
     		
     		    if(mm) return false;
     		}
     		pm.resize(0);
     		mp.resize(0);
	}

     	// Edge (3,4) of A 
     	if(outside134){
     		//move to Vertex 3 of A
     		pos_u =  hypersphere.hypersphericalToCartesian(conj(a_p3),conj(a_p3));
     		pos_a4 = hypersphere.hypersphericalToCartesian(conj(a_p3)*a_p4,a_p4*conj(a_p3));
     		
     		pos_b.resize(0);
     		pos_b.push_back(hypersphere.hypersphericalToCartesian(conj(a_p3)*quat_l*b_p1,b_p1*quat_r*conj(a_p3)));
     		pos_b.push_back(hypersphere.hypersphericalToCartesian(conj(a_p3)*quat_l*b_p2,b_p2*quat_r*conj(a_p3)));
     		pos_b.push_back(hypersphere.hypersphericalToCartesian(conj(a_p3)*quat_l*b_p3,b_p3*quat_r*conj(a_p3)));
     		pos_b.push_back(hypersphere.hypersphericalToCartesian(conj(a_p3)*quat_l*b_p4,b_p4*quat_r*conj(a_p3)));
     		
     		
     		mm = true;
     		
     		for( int i = 0; i < side234.size(); i++){
     		    if(!side234[i] && !side134[i] ){ 
     		        mm=false;
     		        break;
     		    }
     		    if(side234[i] && !side134[i] ) pm.push_back(pos_b[i].v);
     		    else if(!side234[i] && side134[i]) mp.push_back(pos_b[i].v) ;
     		}
     		
     		// See if edge of B goes through A 
     		if(mp.size() > 0 && pm.size() > 0 && mm){
     		    for (int i =0; i < pm.size() && mm; i++){
     		        vec3<OverlapReal> n = cross(pm[i],pos_a4.v);
     		        if(dot(n,pos_u.v)>0) n = -n;
     		        for (int j =0; j < mp.size() && mm; j++){
     		            if(dot(n,mp[j]) < 0 ){ 
     		                mm = false;
     		            }
     		        }
     		    }
     		
     		    if(mm) return false;
     		}
	}
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
     
     // Face (1,2,3) of B
     n123 = cross(pos_a2.v, pos_a3.v);
     if(dot(n123,pos_u.v) > 0) n123 = -n123;
     
     //inside or outside halfspace of (1,2,3)
     mm = true;
     for( int i = 0; i < side123.size() && mm; i++){
         if(dot(n123,pos_b[i].v) < 0) mm = false;
     }
     
     if(mm) return false;
     
     // Face (1,2,4) of B
     n124 = cross(pos_a2.v, pos_a4.v);
     if(dot(n124,pos_u.v) > 0) n124 = -n124;
     
     //inside or outside halfspace of (1,2,4)
     mm = true;
     for( int i = 0; i < side124.size() && mm; i++){
         if(dot(n124,pos_b[i].v) < 0) mm = false;
     }
     
     if(mm) return false;
     
     // Face (1,3,4) of B 
     n134 = cross(pos_a3.v, pos_a4.v);
     if(dot(n134,pos_u.v) > 0) n134 = -n134;
     
     //inside or outside halfspace of (1,3,4)
     mm = true;
     for( int i = 0; i < side134.size() && mm; i++){
         if(dot(n134,pos_b[i].v) < 0) mm = false;
     }
     
     if(mm) return false;
     
     //move to Vertex 2 of B
     pos_u =  hypersphere.hypersphericalToCartesian(conj(b_p2),conj(b_p2));
     pos_a3 = hypersphere.hypersphericalToCartesian(conj(b_p2)*b_p3,b_p3*conj(b_p2));
     pos_a4 = hypersphere.hypersphericalToCartesian(conj(b_p2)*b_p4,b_p4*conj(b_p2));
    
     pos_b.resize(0);
     pos_b.push_back(hypersphere.hypersphericalToCartesian(conj(quat_l*b_p2)*a_p1,a_p1*conj(b_p2*quat_r)));
     pos_b.push_back(hypersphere.hypersphericalToCartesian(conj(quat_l*b_p2)*a_p2,a_p2*conj(b_p2*quat_r)));
     pos_b.push_back(hypersphere.hypersphericalToCartesian(conj(quat_l*b_p2)*a_p3,a_p3*conj(b_p2*quat_r)));
     pos_b.push_back(hypersphere.hypersphericalToCartesian(conj(quat_l*b_p2)*a_p4,a_p4*conj(b_p2*quat_r)));
     
     // Face (2,3,4) of B
     n234 = cross(pos_a3.v, pos_a4.v);
     if(dot(n234,pos_u.v) > 0) n234 = -n234;
     
     //inside or outside halfspace of (2,3,4)
     mm = true;
     for( int i = 0; i < side234.size() && mm; i++){
         if(dot(n234,pos_b[i].v) < 0) mm = false;
     }
     
     if(mm) return false;
     
     return true;


    }



template<class SupportFuncA, class SupportFuncB>
DEVICE inline bool xenocollide_hypersphereBiTetra(const SupportFuncA& a,const SupportFuncB& b,quat<OverlapReal>& quat_l,quat<OverlapReal>& quat_r,const Hypersphere& hypersphere,const OverlapReal Ra,unsigned int& err_count)
    {
    // This implementation of XenoCollide is hand-written from the description of the algorithm on page 171 of _Games
    // Programming Gems 7_

     quat<OverlapReal>  pos_a2, pos_a3, pos_a4, pos_a5;
     vec3<OverlapReal> n124, n125, n134, n135, n234, n235;
     std::vector<bool> side124(5,false);
     std::vector<bool> side125(5,false);
     std::vector<bool> side134(5,false);
     std::vector<bool> side135(5,false);
     std::vector<bool> side234(5,false);
     std::vector<bool> side235(5,false);
     std::vector< vec3<OverlapReal> > pm;
     std::vector< vec3<OverlapReal> > mp;
     bool mm, outside124, outside125, outside134, outside135, outside234, outside235;

     std::vector<quat<OverlapReal> > pos_b;
     quat<OverlapReal> pos_u;


     quat<OverlapReal> a_p1 =  hypersphere.cartesianToHyperspherical(vec3<OverlapReal>(a.x[0],a.y[0],a.z[0]));
     quat<OverlapReal> a_p2 =  hypersphere.cartesianToHyperspherical(vec3<OverlapReal>(a.x[1],a.y[1],a.z[1]));
     quat<OverlapReal> a_p3 =  hypersphere.cartesianToHyperspherical(vec3<OverlapReal>(a.x[2],a.y[2],a.z[2]));
     quat<OverlapReal> a_p4 =  hypersphere.cartesianToHyperspherical(vec3<OverlapReal>(a.x[3],a.y[3],a.z[3]));
     quat<OverlapReal> a_p5 =  hypersphere.cartesianToHyperspherical(vec3<OverlapReal>(a.x[4],a.y[4],a.z[4]));
    
     quat<OverlapReal> b_p1 =  hypersphere.cartesianToHyperspherical(vec3<OverlapReal>(b.x[0],b.y[0],b.z[0]));
     quat<OverlapReal> b_p2 =  hypersphere.cartesianToHyperspherical(vec3<OverlapReal>(b.x[1],b.y[1],b.z[1]));
     quat<OverlapReal> b_p3 =  hypersphere.cartesianToHyperspherical(vec3<OverlapReal>(b.x[2],b.y[2],b.z[2]));
     quat<OverlapReal> b_p4 =  hypersphere.cartesianToHyperspherical(vec3<OverlapReal>(b.x[3],b.y[3],b.z[3]));
     quat<OverlapReal> b_p5 =  hypersphere.cartesianToHyperspherical(vec3<OverlapReal>(b.x[4],b.y[4],b.z[4]));


     //move to Vertex 1 of A
     pos_u =  hypersphere.hypersphericalToCartesian(conj(a_p1),conj(a_p1));
     pos_a2 = hypersphere.hypersphericalToCartesian(conj(a_p1)*a_p2,a_p2*conj(a_p1));
     pos_a3 = hypersphere.hypersphericalToCartesian(conj(a_p1)*a_p3,a_p3*conj(a_p1));
     pos_a4 = hypersphere.hypersphericalToCartesian(conj(a_p1)*a_p4,a_p4*conj(a_p1));
     pos_a5 = hypersphere.hypersphericalToCartesian(conj(a_p1)*a_p5,a_p5*conj(a_p1));

     pos_b.push_back(hypersphere.hypersphericalToCartesian(conj(a_p1)*quat_l*b_p1,b_p1*quat_r*conj(a_p1)));
     pos_b.push_back(hypersphere.hypersphericalToCartesian(conj(a_p1)*quat_l*b_p2,b_p2*quat_r*conj(a_p1)));
     pos_b.push_back(hypersphere.hypersphericalToCartesian(conj(a_p1)*quat_l*b_p3,b_p3*quat_r*conj(a_p1)));
     pos_b.push_back(hypersphere.hypersphericalToCartesian(conj(a_p1)*quat_l*b_p4,b_p4*quat_r*conj(a_p1)));
     pos_b.push_back(hypersphere.hypersphericalToCartesian(conj(a_p1)*quat_l*b_p5,b_p5*quat_r*conj(a_p1)));


     // Face (1,2,4) of A 
     n124 = cross(pos_a2.v, pos_a4.v);
     if(dot(n124,pos_u.v) > 0) n124 = -n124;
     
     mm = true;
     outside124 = false;
     //inside or outside halfspace of (1,2,4)
     for( int i = 0; i < side124.size(); i++){
         if(dot(n124,pos_b[i].v) > 0){ 
		side124[i] = true;
		outside124 = true;
	 }
	 else mm = false;
     }
     
     if(mm) return false;
     
     // Face (1,2,5) of A 
     n125 = cross(pos_a2.v, pos_a5.v);
     if(dot(n125,pos_u.v) > 0) n125 = -n125;
     
     mm = true;
     outside125 = false;
     //inside or outside halfspace of (1,2,5)
     for( int i = 0; i < side125.size(); i++){
         if(dot(n125,pos_b[i].v) > 0){ 
		side125[i] = true;
		outside125 = true;
	 }
	 else mm = false;
     }
     
     if(mm) return false;


     // Edge (1,2) of A 

     if(outside124 && outside125){

     	mm = true;
     	
     	
     	for( int i = 0; i < side125.size(); i++){
     	    if(!side124[i] && !side125[i] ){ 
     	        mm=false;
     	        break;
     	    }
     	    if(side124[i] && !side125[i] ) pm.push_back(pos_b[i].v);
     	    else if(!side124[i] && side125[i]) mp.push_back(pos_b[i].v) ;
     	}
     	
     	// See if edge of B goes through A 
     	if(mp.size() > 0  && pm.size() > 0 && mm){
     	    for (int i =0; i < pm.size() && mm; i++){
     	        vec3<OverlapReal> n = cross(pm[i],pos_a2.v);
     	        if(dot(n,pos_u.v)>0) n = -n;
     	        for (int j =0; j < mp.size() && mm; j++){
     	            if(dot(n,mp[j]) < 0 ){ 
     	                mm = false;
     	            }
     	        }
     	    }
     	
     	    if(mm) return false;
     	}
     	pm.resize(0);
     	mp.resize(0);
     }

     // Face (1,3,4) of A 
     n134 = cross(pos_a3.v, pos_a4.v);
     if(dot(n134,pos_u.v) > 0) n134 = -n134;
     
     mm = true;
     outside134 = false;
     //inside or outside halfspace of (1,3,4)
     for( int i = 0; i < side134.size(); i++){
         if(dot(n134,pos_b[i].v) > 0){ 
		side134[i] = true;
		outside134 = true;
	 }
	 else mm = false;
     }
     
     if(mm) return false;
     
     // Edge (1,4) of A 
     if(outside134 && outside124){
     	mm = true;
     	
     	for( int i = 0; i < side124.size(); i++){
     	    if(!side124[i] && !side134[i] ){ 
     	        mm=false;
     	        break;
     	    }
     	    if(side124[i] && !side134[i] ) pm.push_back(pos_b[i].v);
     	    else if(!side124[i] && side134[i]) mp.push_back(pos_b[i].v) ;
     	}

     	// See if edge of B goes through A 
     	if(mp.size() > 0  && pm.size() > 0 && mm){
     	    for (int i =0; i < pm.size() && mm; i++){
     	        vec3<OverlapReal> n = cross(pm[i],pos_a4.v);
     	        if(dot(n,pos_u.v)>0) n = -n;
     	        for (int j =0; j < mp.size() && mm; j++){
     	            if(dot(n,mp[j]) < 0 ){ 
     	                mm = false;
     	            }
     	        }
     	    }
     	
     	    if(mm) return false;
     	}
     	pm.resize(0);
     	mp.resize(0);
     }

     // Face (1,3,5) of A 
     n135 = cross(pos_a3.v, pos_a5.v);
     if(dot(n135,pos_u.v) > 0) n135 = -n135;
     
     mm = true;
     outside135 = false;
     //inside or outside halfspace of (1,3,5)
     for( int i = 0; i < side135.size(); i++){
         if(dot(n135,pos_b[i].v) > 0){ 
		side135[i] = true;
		outside135 = true;
	 }
	 else mm = false;
     }
     
     if(mm) return false;


     if(outside135){

     	// Edge (1,5) of A 
	if(outside125){
     		mm = true;
     		
     		for( int i = 0; i < side125.size(); i++){
     		    if(!side125[i] && !side135[i] ){ 
     		        mm=false;
     		        break;
     		    }
     		    if(side125[i] && !side135[i] ) pm.push_back(pos_b[i].v);
     		    else if(!side125[i] && side135[i]) mp.push_back(pos_b[i].v) ;
     		}

     		// See if edge of B goes through A 
     		if(mp.size() > 0  && pm.size() > 0 && mm){
     		    for (int i =0; i < pm.size() && mm; i++){
     		        vec3<OverlapReal> n = cross(pm[i],pos_a5.v);
     		        if(dot(n,pos_u.v)>0) n = -n;
     		        for (int j =0; j < mp.size() && mm; j++){
     		            if(dot(n,mp[j]) < 0 ){ 
     		                mm = false;
     		            }
     		        }
     		    }
     		
     		    if(mm) return false;
     		}
     		pm.resize(0);
     		mp.resize(0);
	}

     	// Edge (1,3) of A 
	if(outside134){
     		mm = true;
     		
     		for( int i = 0; i < side134.size(); i++){
     		    if(!side134[i] && !side135[i] ){ 
     		        mm=false;
     		        break;
     		    }
     		    if(side134[i] && !side135[i] ) pm.push_back(pos_b[i].v);
     		    else if(!side134[i] && side135[i]) mp.push_back(pos_b[i].v) ;
     		}
     		
     		// See if edge of B goes through A 
     		if(mp.size() > 0  && pm.size() > 0 && mm){
     		    for (int i =0; i < pm.size() && mm; i++){
     		        vec3<OverlapReal> n = cross(pm[i],pos_a3.v);
     		        if(dot(n,pos_u.v)>0) n = -n;
     		        for (int j =0; j < mp.size() && mm; j++){
     		            if(dot(n,mp[j]) < 0 ){ 
     		                mm = false;
     		            }
     		        }
     		    }
     		
     		    if(mm) return false;
     		}
     		pm.resize(0);
     		mp.resize(0);
	}
     }
     


     //move to Vertex 2 of A
     pos_u =  hypersphere.hypersphericalToCartesian(conj(a_p2),conj(a_p2));
     pos_a3 = hypersphere.hypersphericalToCartesian(conj(a_p2)*a_p3,a_p3*conj(a_p2));
     pos_a4 = hypersphere.hypersphericalToCartesian(conj(a_p2)*a_p4,a_p4*conj(a_p2));
     pos_a5 = hypersphere.hypersphericalToCartesian(conj(a_p2)*a_p5,a_p5*conj(a_p2));

     pos_b.resize(0);
     pos_b.push_back(hypersphere.hypersphericalToCartesian(conj(a_p2)*quat_l*b_p1,b_p1*quat_r*conj(a_p2)));
     pos_b.push_back(hypersphere.hypersphericalToCartesian(conj(a_p2)*quat_l*b_p2,b_p2*quat_r*conj(a_p2)));
     pos_b.push_back(hypersphere.hypersphericalToCartesian(conj(a_p2)*quat_l*b_p3,b_p3*quat_r*conj(a_p2)));
     pos_b.push_back(hypersphere.hypersphericalToCartesian(conj(a_p2)*quat_l*b_p4,b_p4*quat_r*conj(a_p2)));
     pos_b.push_back(hypersphere.hypersphericalToCartesian(conj(a_p2)*quat_l*b_p5,b_p5*quat_r*conj(a_p2)));

     // Face (2,3,4) of A 
     n234 = cross(pos_a3.v, pos_a4.v);
     if(dot(n234,pos_u.v) > 0) n234 = -n234;
     
     mm = true;
     outside234 = false;
     //inside or outside halfspace of (2,3,4)
     for( int i = 0; i < side234.size(); i++){
         if(dot(n234,pos_b[i].v) > 0){ 
		side234[i] = true;
		outside234 = true;
	 }
	 else mm = false;
     }
     
     if(mm) return false;
     
     // Edge (2,4) of A 
     if(outside234 && outside124){

     	mm = true;
     	
     	for( int i = 0; i < side234.size(); i++){
     	    if(!side234[i] && !side124[i] ){ 
     	        mm=false;
     	        break;
     	    }
     	    if(side234[i] && !side124[i] ) pm.push_back(pos_b[i].v);
     	    else if(!side234[i] && side124[i]) mp.push_back(pos_b[i].v) ;
     	}
     	
     	// See if edge of B goes through A 
     	if(mp.size() > 0 && pm.size() > 0 && mm){
     	    for (int i =0; i < pm.size() && mm; i++){
     	        vec3<OverlapReal> n = cross(pm[i],pos_a4.v);
     	        if(dot(n,pos_u.v)>0) n = -n;
     	        for (int j =0; j < mp.size() && mm; j++){
     	            if(dot(n,mp[j]) < 0 ){ 
     	                mm = false;
     	            }
     	        }
     	    }
     	
     	    if(mm) return false;
     	}
     	pm.resize(0);
     	mp.resize(0);
     }

     // Face (2,3,5) of A 
     n235 = cross(pos_a3.v, pos_a5.v);
     if(dot(n235,pos_u.v) > 0) n235 = -n235;
     
     mm = true;
     outside235 = false;
     //inside or outside halfspace of (2,3,5)
     for( int i = 0; i < side235.size(); i++){
         if(dot(n235,pos_b[i].v) > 0){ 
		side235[i] = true;
		outside235 = true;
	 }
	 else mm = false;
     }
     
     if(mm) return false;

     // See if vertex of B inside A 
     if( (!side124[0] && !side125[0] && !side134[0] && !side135[0] && !side234[0] && !side235[0]) || (!side124[1] && !side125[1] && !side134[1] && !side135[1] && !side234[1] && !side235[1]) || (!side124[2] && !side125[2] && !side134[2] && !side135[2] && !side234[2] && !side235[2]) || (!side124[3] && !side125[3] && !side134[3] && !side135[3] && !side234[3] && !side235[3]) || (!side124[4] && !side125[4] && !side134[4] && !side135[4] && !side234[4] && !side235[4])) return true; 
     

     bool nchanged = true;
     if(outside235){

        // Edge (2,5) of A 
     	if(outside125){
     		mm = true;
     		
     		for( int i = 0; i < side235.size(); i++){
     		    if(!side235[i] && !side125[i] ){ 
     		        mm=false;
     		        break;
     		    }
     		    if(side235[i] && !side125[i] ) pm.push_back(pos_b[i].v);
     		    else if(!side235[i] && side125[i]) mp.push_back(pos_b[i].v) ;
     		}
     		
     		// See if edge of B goes through A 
     		if(mp.size() > 0 && pm.size() > 0 && mm){
     		    for (int i =0; i < pm.size() && mm; i++){
     		        vec3<OverlapReal> n = cross(pm[i],pos_a5.v);
     		        if(dot(n,pos_u.v)>0) n = -n;
     		        for (int j =0; j < mp.size() && mm; j++){
     		            if(dot(n,mp[j]) < 0 ){ 
     		                mm = false;
     		            }
     		        }
     		    }
     		
     		    if(mm) return false;
     		}
     		pm.resize(0);
     		mp.resize(0);
	}

        // Edge (2,3) of A 
     	if(outside234){
     		mm = true;
     		
     		for( int i = 0; i < side235.size(); i++){
     		    if(!side235[i] && !side234[i] ){ 
     		        mm=false;
     		        break;
     		    }
     		    if(side235[i] && !side234[i] ) pm.push_back(pos_b[i].v);
     		    else if(!side235[i] && side234[i]) mp.push_back(pos_b[i].v) ;
     		}
     		
     		// See if edge of B goes through A 
     		if(mp.size() > 0 && pm.size() > 0 && mm){
     		    for (int i =0; i < pm.size() && mm; i++){
     		        vec3<OverlapReal> n = cross(pm[i],pos_a3.v);
     		        if(dot(n,pos_u.v)>0) n = -n;
     		        for (int j =0; j < mp.size() && mm; j++){
     		            if(dot(n,mp[j]) < 0 ){ 
     		                mm = false;
     		            }
     		        }
     		    }
     		
     		    if(mm) return false;
     		}
     		pm.resize(0);
     		mp.resize(0);
	}

        // Edge (3,5) of A 
     	if(outside135){
                nchanged = false;
     		pos_u =  hypersphere.hypersphericalToCartesian(conj(a_p3),conj(a_p3));
     		pos_a5 = hypersphere.hypersphericalToCartesian(conj(a_p3)*a_p5,a_p5*conj(a_p3));
     		
     		pos_b.resize(0);
     		pos_b.push_back(hypersphere.hypersphericalToCartesian(conj(a_p3)*quat_l*b_p1,b_p1*quat_r*conj(a_p3)));
     		pos_b.push_back(hypersphere.hypersphericalToCartesian(conj(a_p3)*quat_l*b_p2,b_p2*quat_r*conj(a_p3)));
     		pos_b.push_back(hypersphere.hypersphericalToCartesian(conj(a_p3)*quat_l*b_p3,b_p3*quat_r*conj(a_p3)));
     		pos_b.push_back(hypersphere.hypersphericalToCartesian(conj(a_p3)*quat_l*b_p4,b_p4*quat_r*conj(a_p3)));
     		pos_b.push_back(hypersphere.hypersphericalToCartesian(conj(a_p3)*quat_l*b_p5,b_p5*quat_r*conj(a_p3)));

     		mm = true;
     		
     		for( int i = 0; i < side235.size(); i++){
     		    if(!side235[i] && !side135[i] ){ 
     		        mm=false;
     		        break;
     		    }
     		    if(side235[i] && !side135[i] ) pm.push_back(pos_b[i].v);
     		    else if(!side235[i] && side135[i]) mp.push_back(pos_b[i].v) ;
     		}
     		
     		// See if edge of B goes through A 
     		if(mp.size() > 0 && pm.size() > 0 && mm){
     		    for (int i =0; i < pm.size() && mm; i++){
     		        vec3<OverlapReal> n = cross(pm[i],pos_a5.v);
     		        if(dot(n,pos_u.v)>0) n = -n;
     		        for (int j =0; j < mp.size() && mm; j++){
     		            if(dot(n,mp[j]) < 0 ){ 
     		                mm = false;
     		            }
     		        }
     		    }
     		
     		    if(mm) return false;
     		}
     		pm.resize(0);
     		mp.resize(0);
	}
     }

     // Edge (3,4) of A 
     if(outside134 && outside234){
     	
        if(nchanged){
     	    pos_b.resize(0);
     	    pos_b.push_back(hypersphere.hypersphericalToCartesian(conj(a_p3)*quat_l*b_p1,b_p1*quat_r*conj(a_p3)));
     	    pos_b.push_back(hypersphere.hypersphericalToCartesian(conj(a_p3)*quat_l*b_p2,b_p2*quat_r*conj(a_p3)));
     	    pos_b.push_back(hypersphere.hypersphericalToCartesian(conj(a_p3)*quat_l*b_p3,b_p3*quat_r*conj(a_p3)));
     	    pos_b.push_back(hypersphere.hypersphericalToCartesian(conj(a_p3)*quat_l*b_p4,b_p4*quat_r*conj(a_p3)));
     	    pos_b.push_back(hypersphere.hypersphericalToCartesian(conj(a_p3)*quat_l*b_p5,b_p5*quat_r*conj(a_p3)));
     	    pos_u =  hypersphere.hypersphericalToCartesian(conj(a_p3),conj(a_p3));
	}
     	pos_a4 = hypersphere.hypersphericalToCartesian(conj(a_p3)*a_p4,a_p4*conj(a_p3));
     
     	mm = true;
     	
     	for( int i = 0; i < side234.size(); i++){
     	    if(!side234[i] && !side134[i] ){ 
     	        mm=false;
     	        break;
     	    }
     	    if(side234[i] && !side134[i] ) pm.push_back(pos_b[i].v);
     	    else if(!side234[i] && side134[i]) mp.push_back(pos_b[i].v) ;
     	}
     	
     	// See if edge of B goes through A 
     	if(mp.size() > 0 && pm.size() > 0 && mm){
     	    for (int i =0; i < pm.size() && mm; i++){
     	        vec3<OverlapReal> n = cross(pm[i],pos_a4.v);
     	        if(dot(n,pos_u.v)>0) n = -n;
     	        for (int j =0; j < mp.size() && mm; j++){
     	            if(dot(n,mp[j]) < 0 ){ 
     	                mm = false;
     	            }
     	        }
     	    }
     	
     	    if(mm) return false;
     	}
     	pm.resize(0);
     	mp.resize(0);
     }
     
     //move to Vertex 1 of B
      pos_u =  hypersphere.hypersphericalToCartesian(conj(b_p1),conj(b_p1));
      pos_a2 = hypersphere.hypersphericalToCartesian(conj(b_p1)*b_p2,b_p2*conj(b_p1));
      pos_a3 = hypersphere.hypersphericalToCartesian(conj(b_p1)*b_p3,b_p3*conj(b_p1));
      pos_a4 = hypersphere.hypersphericalToCartesian(conj(b_p1)*b_p4,b_p4*conj(b_p1));
      pos_a5 = hypersphere.hypersphericalToCartesian(conj(b_p1)*b_p5,b_p5*conj(b_p1));
     
      pos_b.resize(0);
      pos_b.push_back(hypersphere.hypersphericalToCartesian(conj(quat_l*b_p1)*a_p1,a_p1*conj(b_p1*quat_r)));
      pos_b.push_back(hypersphere.hypersphericalToCartesian(conj(quat_l*b_p1)*a_p2,a_p2*conj(b_p1*quat_r)));
      pos_b.push_back(hypersphere.hypersphericalToCartesian(conj(quat_l*b_p1)*a_p3,a_p3*conj(b_p1*quat_r)));
      pos_b.push_back(hypersphere.hypersphericalToCartesian(conj(quat_l*b_p1)*a_p4,a_p4*conj(b_p1*quat_r)));
      pos_b.push_back(hypersphere.hypersphericalToCartesian(conj(quat_l*b_p1)*a_p5,a_p5*conj(b_p1*quat_r)));
     
     // Face (1,2,4) of B
     n124 = cross(pos_a2.v, pos_a4.v);
     if(dot(n124,pos_u.v) > 0) n124 = -n124;
     
     //inside or outside halfspace of (1,2,4)
     mm = true;
     for( int i = 0; i < side124.size() && mm; i++){
         if(dot(n124,pos_b[i].v) < 0) mm = false;
     }
     
     if(mm) return false;
     
     // Face (1,2,5) of B
     n125 = cross(pos_a2.v, pos_a5.v);
     if(dot(n125,pos_u.v) > 0) n125 = -n125;
     
     //inside or outside halfspace of (1,2,5)
     mm = true;
     for( int i = 0; i < side125.size() && mm; i++){
         if(dot(n125,pos_b[i].v) < 0) mm = false;
     }
     
     if(mm) return false;
     
     // Face (1,3,4) of B 
     n134 = cross(pos_a3.v, pos_a4.v);
     if(dot(n134,pos_u.v) > 0) n134 = -n134;
     
     //inside or outside halfspace of (1,3,4)
     mm = true;
     for( int i = 0; i < side134.size() && mm; i++){
         if(dot(n134,pos_b[i].v) < 0) mm = false;
     }
     
     if(mm) return false;

     // Face (1,3,5) of B 
     n135 = cross(pos_a3.v, pos_a5.v);
     if(dot(n135,pos_u.v) > 0) n135 = -n135;
     
     //inside or outside halfspace of (1,3,5)
     mm = true;
     for( int i = 0; i < side135.size() && mm; i++){
         if(dot(n135,pos_b[i].v) < 0) mm = false;
     }
     
     if(mm) return false;
     
     //move to Vertex 2 of B
     pos_u =  hypersphere.hypersphericalToCartesian(conj(b_p2),conj(b_p2));
     pos_a3 = hypersphere.hypersphericalToCartesian(conj(b_p2)*b_p3,b_p3*conj(b_p2));
     pos_a4 = hypersphere.hypersphericalToCartesian(conj(b_p2)*b_p4,b_p4*conj(b_p2));
     pos_a5 = hypersphere.hypersphericalToCartesian(conj(b_p2)*b_p5,b_p5*conj(b_p2));
    
     pos_b.resize(0);
     pos_b.push_back(hypersphere.hypersphericalToCartesian(conj(quat_l*b_p2)*a_p1,a_p1*conj(b_p2*quat_r)));
     pos_b.push_back(hypersphere.hypersphericalToCartesian(conj(quat_l*b_p2)*a_p2,a_p2*conj(b_p2*quat_r)));
     pos_b.push_back(hypersphere.hypersphericalToCartesian(conj(quat_l*b_p2)*a_p3,a_p3*conj(b_p2*quat_r)));
     pos_b.push_back(hypersphere.hypersphericalToCartesian(conj(quat_l*b_p2)*a_p4,a_p4*conj(b_p2*quat_r)));
     pos_b.push_back(hypersphere.hypersphericalToCartesian(conj(quat_l*b_p2)*a_p5,a_p5*conj(b_p2*quat_r)));
     
     // Face (2,3,4) of B
     n234 = cross(pos_a3.v, pos_a4.v);
     if(dot(n234,pos_u.v) > 0) n234 = -n234;
     
     //inside or outside halfspace of (2,3,4)
     mm = true;
     for( int i = 0; i < side234.size() && mm; i++){
         if(dot(n234,pos_b[i].v) < 0) mm = false;
     }
     
     if(mm) return false;

     // Face (2,3,5) of B
     n235 = cross(pos_a3.v, pos_a5.v);
     if(dot(n235,pos_u.v) > 0) n235 = -n235;
     
     //inside or outside halfspace of (2,3,5)
     mm = true;
     for( int i = 0; i < side235.size() && mm; i++){
         if(dot(n235,pos_b[i].v) < 0) mm = false;
     }
     
     if(mm) return false;
     
     return true;


    }





template<class SupportFuncA, class SupportFuncB>
DEVICE inline bool xenocollide_hypersphere2(const SupportFuncA& a,const SupportFuncB& b,quat<OverlapReal>& quat_l,quat<OverlapReal>& quat_r,const Hypersphere& hypersphere,const OverlapReal Ra,unsigned int& err_count)
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
            std::cout << "posu " << pos_u.s << " " << pos_u.v.x << " " << pos_u.v.y << " " << pos_u.v.z << std::endl;
            for( int i = 0; i < a.N; i++){
                pos_a[i] = hypersphere.hypersphericalToCartesian(conj(a_p[vertex])*a_p[i],a_p[i]*conj(a_p[vertex]));
                std::cout << "pos_a" << i << " " << pos_a[i].s << " " << pos_a[i].v.x << " " << pos_a[i].v.y << " " << pos_a[i].v.z << std::endl;
            }

            for( int i = 0; i < b.N; i++){
                pos_b[i] = hypersphere.hypersphericalToCartesian(conj(a_p[vertex])*quat_l*b_p[i],b_p[i]*quat_r*conj(a_p[vertex]));
                std::cout << "pos_b" << i << " " << pos_b[i].s << " " << pos_b[i].v.x << " " << pos_b[i].v.y << " " << pos_b[i].v.z << std::endl;
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
                std::cout << "FACE " <<  vertex << " " << vertex1 << " " << vertex2 << std::endl;
        	
        	n1 = cross(pos_a[vertex1].v, pos_a[vertex2].v);
                if(dot(n1,pos_u.v) > 0) n1 = -n1;

                for( int i = 0; i < b.N; i++){
		    std::cout << i << ": " << dot(n1,pos_b[i].v) << std::endl;
                    if(dot(n1,pos_b[i].v) > 0) side[face1][i] = true;
                    else separate = false;
                }
                //if(separate) return false;
		
        	all_faces++;
        	if(all_faces == Nf){
                    std::cout << "VOLUME" << std::endl;
         	    for( int i = 0; i < b.N && !separate; i++){
         		separate = true;
         		for( int j = 0; j < Nf && separate; j++)
         		    if(side[j][i]) separate = false;
         	    }
         	    //if(separate) return true;
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
                std::cout << "FACE " <<  vertex << " " << vertex1 << " " << vertex2 << std::endl;
        	
        	n1 = cross(pos_a[vertex1].v, pos_a[vertex2].v);
                if(dot(n1,pos_u.v) > 0) n1 = -n1;

                for( int i = 0; i < b.N; i++){
		    std::cout << i << ": " << dot(n1,pos_b[i].v) << std::endl;
                    if(dot(n1,pos_b[i].v) > 0) side[face2][i] = true;
                    else separate = false;
                }
                //if(separate) return false;

        	all_faces++;
        	if(all_faces == Nf){
                    std::cout << "VOLUME" << std::endl;
         	    for( int i = 0; i < b.N && !separate; i++){
         		separate = true;
         		for( int j = 0; j < Nf && separate; j++)
         		    if(side[j][i]) separate = false;
         	    }
         	    //if(separate) return true;
                }
        }


        // Edge of A 
        separate = true;
        std::vector< vec3<OverlapReal> > pm;
        std::vector< vec3<OverlapReal> > mp;

        std::cout << "EDGE " <<  vertex << " " << vertex1 << std::endl;

        for( int i = 0; i < b.N; i++){
            std::cout << side[face1][i] << " " << side[face2][i] << std::endl;
            if(!side[face1][i] && !side[face2][i] ){ 
                separate=false;
                break;
            }
            if(side[face1][i] && !side[face2][i] ) pm.push_back(pos_b[i].v);
            else if(!side[face1][i] && side[face2][i]) mp.push_back(pos_b[i].v) ;
        }


        // See if edge of B goes through A 
        if(separate){
            for (int i =0; i < pm.size(); i++){
                vec3<OverlapReal> n = cross(pm[i],pos_a[vertex1].v);
                if(dot(n,pos_u.v)>0) n = -n;
                for (int j =0; j < mp.size(); j++){
                    std::cout << dot(n,mp[j]) << std::endl;
                    if(dot(n,mp[j]) < 0 ){ 
                        separate = false;
                    }
                }
            }

            //if(separate) return false;
        }
    }

    quat_l = conj(quat_l);
    quat_r = conj(quat_r);
   
    for( int i = 0; i < side_used.size(); i++) side_used[i] =false;

    for( int i = 0; i < side.size(); i++) 
        	for( int j = 0; j < side[i].size(); j++) side[i][j] =false;

    all_faces = 0;

    current_vertex = 100000;
    for ( int ei = 0; ei < b.Ne; ei++){
        unsigned int vertex = b.edges[ei].x;
        unsigned int vertex1 = b.edges[ei].y;

        if( vertex != current_vertex){
            current_vertex = vertex;

            pos_u = hypersphere.hypersphericalToCartesian(conj(b_p[vertex]),conj(b_p[vertex]));
            std::cout << "posu " << pos_u.s << " " << pos_u.v.x << " " << pos_u.v.y << " " << pos_u.v.z << std::endl;
            for( int i = 0; i < b.N; i++){
                pos_b[i] = hypersphere.hypersphericalToCartesian(conj(b_p[vertex])*b_p[i],b_p[i]*conj(b_p[vertex]));
                std::cout << "pos_b" << i << " " << pos_b[i].s << " " << pos_b[i].v.x << " " << pos_b[i].v.y << " " << pos_b[i].v.z << std::endl;
            }

            for( int i = 0; i < a.N; i++){
                pos_a[i] = hypersphere.hypersphericalToCartesian(conj(b_p[vertex])*quat_l*a_p[i],a_p[i]*quat_r*conj(b_p[vertex]));
                std::cout << "pos_a" << i << " " << pos_a[i].s << " " << pos_a[i].v.x << " " << pos_a[i].v.y << " " << pos_a[i].v.z << std::endl;
            }
        }


        // Face of A 
        unsigned int face1 = b.boundary_edges[ei].x;
        if(!side_used[face1]){
        	side_used[face1] = true;
                separate = true;
        	unsigned int vertex2=vertex1;
                int vi = 0;
        	while (vertex2 == vertex1 || vertex2 == vertex) {
                    vertex2 = faces[face1][vi];
                    vi++;
        	}
                std::cout << "FACEB " <<  vertex << " " << vertex1 << " " << vertex2 << std::endl;
        	
        	n1 = cross(pos_b[vertex1].v, pos_b[vertex2].v);
                if(dot(n1,pos_u.v) > 0) n1 = -n1;

                for( int i = 0; i < a.N; i++){
        	    std::cout << i << ": " << dot(n1,pos_a[i].v) << std::endl;
                    if(dot(n1,pos_a[i].v) > 0) side[face1][i] = true;
                    else separate = false;
                }
                //if(separate) return false;

        	all_faces++;
        	if(all_faces == Nf){
                    std::cout << "VOLUME" << std::endl;
         	    for( int i = 0; i < a.N && !separate; i++){
         		separate = true;
         		for( int j = 0; j < Nf && separate; j++)
         		    if(side[j][i]) separate = false;
         	    }
         	    //if(separate) return true;
        	}
        }


        unsigned int face2 = b.boundary_edges[ei].y;
        if(!side_used[face2]){
        	side_used[face2] = true;
                separate = true;
        	unsigned int vertex2=vertex1;
                int vi = 0;
        	while (vertex2 == vertex1 || vertex2 == vertex) {
                    vertex2 = faces[face2][vi];
                    vi++;
        	}
                std::cout << "FACEB " <<  vertex << " " << vertex1 << " " << vertex2 << std::endl;
        	
        	n1 = cross(pos_b[vertex1].v, pos_b[vertex2].v);
                if(dot(n1,pos_u.v) > 0) n1 = -n1;

                for( int i = 0; i < a.N; i++){
        	    std::cout << i << ": " << dot(n1,pos_a[i].v) << std::endl;
                    if(dot(n1,pos_a[i].v) > 0) side[face2][i] = true;
                    else separate = false;
                }
                //if(separate) return false;

        	all_faces++;
        	if(all_faces == Nf){
                    std::cout << "VOLUME" << std::endl;
         	    for( int i = 0; i < a.N && !separate; i++){
         		separate = true;
         		for( int j = 0; j < Nf && separate; j++)
         		    if(side[j][i]) separate = false;
         	    }
         	    //if(separate) return true;
                }
        }


        // Edge of A 
        separate = true;
        std::vector< vec3<OverlapReal> > pm;
        std::vector< vec3<OverlapReal> > mp;

        std::cout << "EDGEB " <<  vertex << " " << vertex1 <<  std::endl;

        for( int i = 0; i < a.N; i++){
            std::cout << side[face1][i] << " " << side[face2][i] << std::endl;
            if(!side[face1][i] && !side[face2][i] ){ 
                separate=false;
                break;
            }
            if(side[face1][i] && !side[face2][i] ) pm.push_back(pos_a[i].v);
            else if(!side[face1][i] && side[face2][i]) mp.push_back(pos_a[i].v) ;
        }


        // See if edge of B goes through A 
        if(separate){
            for (int i =0; i < pm.size() && separate; i++){
                vec3<OverlapReal> n = cross(pm[i],pos_b[vertex1].v);
                if(dot(n,pos_u.v)>0) n = -n;
                for (int j =0; j < mp.size() && separate; j++){
                    std::cout << dot(n,mp[j]) << std::endl;
                    if(dot(n,mp[j]) < 0 ){ 
                        separate = false;
                    }
                }
            }

            //if(separate) return false;
        }
    }

    return true;

    }

template<class SupportFuncA, class SupportFuncB>
DEVICE inline bool xenocollide_hypersphereTetra2(const SupportFuncA& a,const SupportFuncB& b,quat<OverlapReal>& quat_l,quat<OverlapReal>& quat_r,const Hypersphere& hypersphere,const OverlapReal Ra,unsigned int& err_count)
    {
    // This implementation of XenoCollide is hand-written from the description of the algorithm on page 171 of _Games
    // Programming Gems 7_

     quat<OverlapReal>  pos_a2, pos_a3, pos_a4;
     vec3<OverlapReal> n123, n124, n134, n234;
     std::vector<bool> side123(4,false);
     std::vector<bool> side124(4,false);
     std::vector<bool> side134(4,false);
     std::vector<bool> side234(4,false);
     std::vector< vec3<OverlapReal> > pm;
     std::vector< vec3<OverlapReal> > mp;
     bool mm;

     std::vector<quat<OverlapReal> > pos_b;
     quat<OverlapReal> pos_u;


     quat<OverlapReal> a_p1 =  hypersphere.cartesianToHyperspherical(vec3<OverlapReal>(a.x[0],a.y[0],a.z[0]));
     quat<OverlapReal> a_p2 =  hypersphere.cartesianToHyperspherical(vec3<OverlapReal>(a.x[1],a.y[1],a.z[1]));
     quat<OverlapReal> a_p3 =  hypersphere.cartesianToHyperspherical(vec3<OverlapReal>(a.x[2],a.y[2],a.z[2]));
     quat<OverlapReal> a_p4 =  hypersphere.cartesianToHyperspherical(vec3<OverlapReal>(a.x[3],a.y[3],a.z[3]));
    
     quat<OverlapReal> b_p1 =  hypersphere.cartesianToHyperspherical(vec3<OverlapReal>(b.x[0],b.y[0],b.z[0]));
     quat<OverlapReal> b_p2 =  hypersphere.cartesianToHyperspherical(vec3<OverlapReal>(b.x[1],b.y[1],b.z[1]));
     quat<OverlapReal> b_p3 =  hypersphere.cartesianToHyperspherical(vec3<OverlapReal>(b.x[2],b.y[2],b.z[2]));
     quat<OverlapReal> b_p4 =  hypersphere.cartesianToHyperspherical(vec3<OverlapReal>(b.x[3],b.y[3],b.z[3]));


     //move to Vertex 1 of A
     pos_u =  hypersphere.hypersphericalToCartesian(conj(a_p1),conj(a_p1));
     std::cout << "posu " << pos_u.s << " " << pos_u.v.x << " " << pos_u.v.y << " " << pos_u.v.z << std::endl;
     pos_a2 = hypersphere.hypersphericalToCartesian(conj(a_p1)*a_p2,a_p2*conj(a_p1));
     std::cout << "pos_a2 " << pos_a2.s << " " << pos_a2.v.x << " " << pos_a2.v.y << " " << pos_a2.v.z << std::endl;
     pos_a3 = hypersphere.hypersphericalToCartesian(conj(a_p1)*a_p3,a_p3*conj(a_p1));
     std::cout << "pos_a3 " << pos_a3.s << " " << pos_a3.v.x << " " << pos_a3.v.y << " " << pos_a3.v.z << std::endl;
     pos_a4 = hypersphere.hypersphericalToCartesian(conj(a_p1)*a_p4,a_p4*conj(a_p1));
     std::cout << "pos_a4 " << pos_a4.s << " " << pos_a4.v.x << " " << pos_a4.v.y << " " << pos_a4.v.z << std::endl;

     pos_b.push_back(hypersphere.hypersphericalToCartesian(conj(a_p1)*quat_l*b_p1,b_p1*quat_r*conj(a_p1)));
     pos_b.push_back(hypersphere.hypersphericalToCartesian(conj(a_p1)*quat_l*b_p2,b_p2*quat_r*conj(a_p1)));
     pos_b.push_back(hypersphere.hypersphericalToCartesian(conj(a_p1)*quat_l*b_p3,b_p3*quat_r*conj(a_p1)));
     pos_b.push_back(hypersphere.hypersphericalToCartesian(conj(a_p1)*quat_l*b_p4,b_p4*quat_r*conj(a_p1)));
     for( int i = 0; i < b.N; i++) std::cout << "pos_b" << i << " " << pos_b[i].s << " " << pos_b[i].v.x << " " << pos_b[i].v.y << " " << pos_b[i].v.z << std::endl;


     // Face (1,2,3) of A 
     n123 = cross(pos_a2.v, pos_a3.v);
     if(dot(n123,pos_u.v) > 0) n123 = -n123;
     
     std::cout << "0-1-2" <<std::endl;
     //inside or outside halfspace of (1,2,3)
     for( int i = 0; i < side123.size(); i++){
         if(dot(n123,pos_b[i].v) > 0) side123[i] = true;
         std::cout << i << ": " << dot(n123,pos_b[i].v) << std::endl;
     }
     
     //if(side123[0] && side123[1] && side123[2] && side123[3]) return false;
     
     // Face (1,2,4) of A 
     n124 = cross(pos_a2.v, pos_a4.v);
     if(dot(n124,pos_u.v) > 0) n124 = -n124;
     
     std::cout << "0-1-3" <<std::endl;
     //inside or outside halfspace of (1,2,4)
     for( int i = 0; i < side124.size(); i++){
         if(dot(n124,pos_b[i].v) > 0) side124[i] = true;
         std::cout << i << ": " << dot(n124,pos_b[i].v) << std::endl;
     }
     
     //if(side124[0] && side124[1] && side124[2] && side124[3]) return false;


     // Edge (1,2) of A 
     mm = false;
     
     std::cout << "0-1" <<std::endl;
     
     for( int i = 0; i < side124.size(); i++){
       	std::cout << side123[i] << " " << side124[i] << std::endl;
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
                 std::cout << dot(n,mp[j]) << std::endl;
                 if(dot(n,mp[j]) < 0 ){ 
                     mm = true;
                 }
             }
         }
     
         //if(!mm) return false;
     }

     // Face (1,3,4) of A 
     n134 = cross(pos_a3.v, pos_a4.v);
     if(dot(n134,pos_u.v) > 0) n134 = -n134;
     
     std::cout << "0-2-3" <<std::endl;
     //inside or outside halfspace of (1,3,4)
     for( int i = 0; i < side134.size(); i++){
         if(dot(n134,pos_b[i].v) > 0) side134[i] = true;
         std::cout << i << ": " << dot(n134,pos_b[i].v) << std::endl;
     }
     
     //if(side134[0] && side134[1] && side134[2] && side134[3]) return false;
     
     // Edge (1,3) of A 
     pm.resize(0);
     mp.resize(0);
     mm = false;
     std::cout << "0-2" <<std::endl;
     
     for( int i = 0; i < side123.size(); i++){
       	std::cout << side123[i] << " " << side134[i] << std::endl;
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
                 std::cout << dot(n,mp[j]) << std::endl;
                 if(dot(n,mp[j]) < 0 ){ 
                     mm = true;
                 }
             }
         }
     
         //if(!mm) return false;
     }
     
     // Edge (1,4) of A 
     pm.resize(0);
     mp.resize(0);
     mm = false;
     std::cout << "0-3" <<std::endl;
     
     for( int i = 0; i < side124.size(); i++){
       	std::cout << side124[i] << " " << side134[i] << std::endl;
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
                 std::cout << dot(n,mp[j]) << std::endl;
                 if(dot(n,mp[j]) < 0 ){ 
                     mm = true;
                 }
             }
         }
     
         //if(!mm) return false;
     }


     //move to Vertex 2 of A
     pos_u =  hypersphere.hypersphericalToCartesian(conj(a_p2),conj(a_p2));
     std::cout << "posu " << pos_u.s << " " << pos_u.v.x << " " << pos_u.v.y << " " << pos_u.v.z << std::endl;
     pos_a3 = hypersphere.hypersphericalToCartesian(conj(a_p2)*a_p3,a_p3*conj(a_p2));
     std::cout << "pos_a3 " << pos_a3.s << " " << pos_a3.v.x << " " << pos_a3.v.y << " " << pos_a3.v.z << std::endl;
     pos_a4 = hypersphere.hypersphericalToCartesian(conj(a_p2)*a_p4,a_p4*conj(a_p2));
     std::cout << "pos_a4 " << pos_a4.s << " " << pos_a4.v.x << " " << pos_a4.v.y << " " << pos_a4.v.z << std::endl;

     pos_b.resize(0);
     pos_b.push_back(hypersphere.hypersphericalToCartesian(conj(a_p2)*quat_l*b_p1,b_p1*quat_r*conj(a_p2)));
     pos_b.push_back(hypersphere.hypersphericalToCartesian(conj(a_p2)*quat_l*b_p2,b_p2*quat_r*conj(a_p2)));
     pos_b.push_back(hypersphere.hypersphericalToCartesian(conj(a_p2)*quat_l*b_p3,b_p3*quat_r*conj(a_p2)));
     pos_b.push_back(hypersphere.hypersphericalToCartesian(conj(a_p2)*quat_l*b_p4,b_p4*quat_r*conj(a_p2)));
     for( int i = 0; i < b.N; i++) std::cout << "pos_b" << i << " " << pos_b[i].s << " " << pos_b[i].v.x << " " << pos_b[i].v.y << " " << pos_b[i].v.z << std::endl;

     // Face (2,3,4) of A 
     n234 = cross(pos_a3.v, pos_a4.v);
     if(dot(n234,pos_u.v) > 0) n234 = -n234;
     
     std::cout << "1-2-3" <<std::endl;
     //inside or outside halfspace of (2,3,4)
     for( int i = 0; i < side234.size(); i++){
         if(dot(n234,pos_b[i].v) > 0) side234[i] = true;
         std::cout << i << ": " << dot(n234,pos_b[i].v) << std::endl;
     }
     
     //if(side234[0] && side234[1] && side234[2] && side234[3]) return false;
     
     
     // See if vertex of B inside A 
     //if( (!side123[0] && !side124[0] && !side134[0] && !side234[0] ) || (!side123[1] && !side124[1] && !side134[1] && !side234[1]) || (!side123[2] && !side124[2] && !side134[2] && !side234[2]) || (!side123[3] && !side124[3] && !side134[3] && !side234[3]) ) return true; 
     
     
     // Edge (2,3) of A 
     pm.resize(0);
     mp.resize(0);
     mm = false;
     std::cout << "1-2" <<std::endl;
     
     for( int i = 0; i < side234.size(); i++){
       	std::cout << side123[i] << " " << side234[i] << std::endl;
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
                 std::cout << dot(n,mp[j]) << std::endl;
                 if(dot(n,mp[j]) < 0 ){ 
                     mm = true;
                 }
             }
         }
     
         //if(!mm) return false;
     }

     // Edge (2,4) of A 
     pm.resize(0);
     mp.resize(0);
     mm = false;
     std::cout << "1-3" <<std::endl;
     
     for( int i = 0; i < side234.size(); i++){
       	std::cout << side124[i] << " " << side234[i] << std::endl;
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
                 std::cout << dot(n,mp[j]) << std::endl;
                 if(dot(n,mp[j]) < 0 ){ 
                     mm = true;
                 }
             }
         }
     
         //if(!mm) return false;
     }

     //move to Vertex 3 of A
     pos_u =  hypersphere.hypersphericalToCartesian(conj(a_p3),conj(a_p3));
     std::cout << "posu " << pos_u.s << " " << pos_u.v.x << " " << pos_u.v.y << " " << pos_u.v.z << std::endl;
     pos_a4 = hypersphere.hypersphericalToCartesian(conj(a_p3)*a_p4,a_p4*conj(a_p3));
     std::cout << "pos_a4 " << pos_a4.s << " " << pos_a4.v.x << " " << pos_a4.v.y << " " << pos_a4.v.z << std::endl;
     
     pos_b.resize(0);
     pos_b.push_back(hypersphere.hypersphericalToCartesian(conj(a_p3)*quat_l*b_p1,b_p1*quat_r*conj(a_p3)));
     pos_b.push_back(hypersphere.hypersphericalToCartesian(conj(a_p3)*quat_l*b_p2,b_p2*quat_r*conj(a_p3)));
     pos_b.push_back(hypersphere.hypersphericalToCartesian(conj(a_p3)*quat_l*b_p3,b_p3*quat_r*conj(a_p3)));
     pos_b.push_back(hypersphere.hypersphericalToCartesian(conj(a_p3)*quat_l*b_p4,b_p4*quat_r*conj(a_p3)));
     for( int i = 0; i < b.N; i++) std::cout << "pos_b" << i << " " << pos_b[i].s << " " << pos_b[i].v.x << " " << pos_b[i].v.y << " " << pos_b[i].v.z << std::endl;
     
     
     // Edge (3,4) of A 
     pm.resize(0);
     mp.resize(0);
     mm = false;
     std::cout << "2-3" <<std::endl;
     
     for( int i = 0; i < side234.size(); i++){
       	 std::cout << side134[i] << " " << side234[i] << std::endl;
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
                 std::cout << dot(n,mp[j]) << std::endl;
                 if(dot(n,mp[j]) < 0 ){ 
                     mm = true;
                 }
             }
         }
     
         //if(!mm) return false;
     }
     
     
     //move to Vertex 1 of B
      pos_u =  hypersphere.hypersphericalToCartesian(conj(b_p1),conj(b_p1));
     std::cout << "posu " << pos_u.s << " " << pos_u.v.x << " " << pos_u.v.y << " " << pos_u.v.z << std::endl;
      pos_a2 = hypersphere.hypersphericalToCartesian(conj(b_p1)*b_p2,b_p2*conj(b_p1));
     std::cout << "pos_a2 " << pos_a2.s << " " << pos_a2.v.x << " " << pos_a2.v.y << " " << pos_a2.v.z << std::endl;
      pos_a3 = hypersphere.hypersphericalToCartesian(conj(b_p1)*b_p3,b_p3*conj(b_p1));
     std::cout << "pos_a3 " << pos_a3.s << " " << pos_a3.v.x << " " << pos_a3.v.y << " " << pos_a3.v.z << std::endl;
      pos_a4 = hypersphere.hypersphericalToCartesian(conj(b_p1)*b_p4,b_p4*conj(b_p1));
     std::cout << "pos_a4 " << pos_a4.s << " " << pos_a4.v.x << " " << pos_a4.v.y << " " << pos_a4.v.z << std::endl;
     
      pos_b.resize(0);
      pos_b.push_back(hypersphere.hypersphericalToCartesian(conj(quat_l*b_p1)*a_p1,a_p1*conj(b_p1*quat_r)));
      pos_b.push_back(hypersphere.hypersphericalToCartesian(conj(quat_l*b_p1)*a_p2,a_p2*conj(b_p1*quat_r)));
      pos_b.push_back(hypersphere.hypersphericalToCartesian(conj(quat_l*b_p1)*a_p3,a_p3*conj(b_p1*quat_r)));
      pos_b.push_back(hypersphere.hypersphericalToCartesian(conj(quat_l*b_p1)*a_p4,a_p4*conj(b_p1*quat_r)));
     for( int i = 0; i < b.N; i++) std::cout << "pos_b" << i << " " << pos_b[i].s << " " << pos_b[i].v.x << " " << pos_b[i].v.y << " " << pos_b[i].v.z << std::endl;
     
     // Face (1,2,3) of B
     n123 = cross(pos_a2.v, pos_a3.v);
     if(dot(n123,pos_u.v) > 0) n123 = -n123;
     std::cout << "B0-1-2" <<std::endl;
     
     //inside or outside halfspace of (1,2,3)
     for( int i = 0; i < side123.size(); i++){
         if(dot(n123,pos_b[i].v) > 0) side123[i] = true;
         else side123[i] = false;
        std::cout << i << ": " << dot(n123,pos_b[i].v) << std::endl;
     }
     
     //if(side123[0] && side123[1] && side123[2] && side123[3]) return false;
     
     // Face (1,2,4) of B
     n124 = cross(pos_a2.v, pos_a4.v);
     if(dot(n124,pos_u.v) > 0) n124 = -n124;
     std::cout << "B0-1-3" <<std::endl;
     
     //inside or outside halfspace of (1,2,4)
     for( int i = 0; i < side124.size(); i++){
         if(dot(n124,pos_b[i].v) > 0) side124[i] = true;
         else side124[i] = false;
        std::cout << i << ": " << dot(n124,pos_b[i].v) << std::endl;
     }
     
     //if(side124[0] && side124[1] && side124[2] && side124[3]) return false;
     
     // Face (1,3,4) of B 
     n134 = cross(pos_a3.v, pos_a4.v);
     if(dot(n134,pos_u.v) > 0) n134 = -n134;
     std::cout << "B0-2-3" <<std::endl;
     
     //inside or outside halfspace of (1,2,4)
     for( int i = 0; i < side134.size(); i++){
         if(dot(n134,pos_b[i].v) > 0) side134[i] = true;
         else side134[i] = false;
        std::cout << i << ": " << dot(n134,pos_b[i].v) << std::endl;
     }
     
     //if(side134[0] && side134[1] && side134[2] && side134[3]) return false;

     //move to Vertex 2 of B
      pos_u =  hypersphere.hypersphericalToCartesian(conj(b_p2),conj(b_p2));
     pos_u = conj(b_p2)*conj(b_p2);
     std::cout << "posu " << pos_u.s << " " << pos_u.v.x << " " << pos_u.v.y << " " << pos_u.v.z << std::endl;
      pos_a3 = hypersphere.hypersphericalToCartesian(conj(b_p2)*b_p3,b_p3*conj(b_p2));
     std::cout << "pos_a3 " << pos_a3.s << " " << pos_a3.v.x << " " << pos_a3.v.y << " " << pos_a3.v.z << std::endl;
      pos_a4 = hypersphere.hypersphericalToCartesian(conj(b_p2)*b_p4,b_p4*conj(b_p2));
     std::cout << "pos_a4 " << pos_a4.s << " " << pos_a4.v.x << " " << pos_a4.v.y << " " << pos_a4.v.z << std::endl;
     
     pos_b.resize(0);
      pos_b.push_back(hypersphere.hypersphericalToCartesian(conj(quat_l*b_p2)*a_p1,a_p1*conj(b_p2*quat_r)));
      pos_b.push_back(hypersphere.hypersphericalToCartesian(conj(quat_l*b_p2)*a_p2,a_p2*conj(b_p2*quat_r)));
      pos_b.push_back(hypersphere.hypersphericalToCartesian(conj(quat_l*b_p2)*a_p3,a_p3*conj(b_p2*quat_r)));
      pos_b.push_back(hypersphere.hypersphericalToCartesian(conj(quat_l*b_p2)*a_p4,a_p4*conj(b_p2*quat_r)));
     for( int i = 0; i < b.N; i++) std::cout << "pos_b" << i << " " << pos_b[i].s << " " << pos_b[i].v.x << " " << pos_b[i].v.y << " " << pos_b[i].v.z << std::endl;
     
     // Face (2,3,4) of B
     n234 = cross(pos_a3.v, pos_a4.v);
     if(dot(n234,pos_u.v) > 0) n234 = -n234;
     std::cout << "B1-2-3" <<std::endl;
     
     //inside or outside halfspace of (2,3,4)
     for( int i = 0; i < side234.size(); i++){
         if(dot(n234,pos_b[i].v) > 0) side234[i] = true;
         else side234[i] = false;
        std::cout << i << ": " << dot(n234,pos_b[i].v) << std::endl;
     }
     
     //if(side234[0] && side234[1] && side234[2] && side234[3]) return false;
     
     return true;


    }


} // end namespace hpmc::detail

}; // end namespace hpmc

#endif // __XENOCOLLIDE_HYPERSPHERE_H__
