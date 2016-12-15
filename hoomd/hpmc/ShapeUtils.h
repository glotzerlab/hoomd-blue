#ifndef _SHAPE_UTILS_H
#define _SHAPE_UTILS_H

#include <queue>
#include <set>
#include <iostream>
#include <fstream>

#include "ShapeSphere.h"
#include "ShapeConvexPolygon.h"
#include "ShapeSpheropolygon.h"
#include "ShapePolyhedron.h"
#include "ShapeConvexPolyhedron.h"
#include "ShapeSpheropolyhedron.h"
#include "ShapeSimplePolygon.h"
#include "ShapeEllipsoid.h"
#include "ShapeFacetedSphere.h"
#include "ShapeSphinx.h"

namespace hpmc{

template < unsigned int old_max_verts, unsigned int new_max_verts >
detail::poly3d_verts<new_max_verts> cast_poly3d_verts(const detail::poly3d_verts<old_max_verts>& old_verts)
    {
    // restricting this cast from small arrays to larger ones ok because it can not invalidate
    // any of the data. The otherway is not true and I don't want to worry about that
    // right now.
    #if  old_max_verts > new_max_verts
        #error "must cast to a larger number of vertices"
    #endif

    // All data guaranteed to be valid because of static_assert above
    detail::poly3d_verts<new_max_verts> verts;
    verts.N = old_verts.N;
    verts.diameter = old_verts.diameter;
    verts.sweep_radius = old_verts.sweep_radius;
    verts.ignore = old_verts.ignore;

    // initialize because we have observed strange behaviour if we don't.
    for (unsigned int i = 0; i < new_max_verts; i++)
        {
        if( i < old_verts.N )
            {
                verts.x[i] = old_verts.x[i];
                verts.y[i] = old_verts.y[i];
                verts.z[i] = old_verts.z[i];
            }
        else
            {
            verts.x[i] = verts.y[i] = verts.z[i] = OverlapReal(0);
            }
        }

    return verts;
    }


namespace detail{
// TODO: template Scalar type.

template<class Shape>
class mass_properties_base
{
public:
    mass_properties_base() : m_volume(0.0), m_center_of_mass(0.0, 0.0, 0.0)
        {
        for(unsigned int i = 0; i < 6; i++) m_inertia[i] = 0.0;
        }

    Scalar getVolume() { return m_volume; }

    const vec3<Scalar>& getCenterOfMass() { return m_center_of_mass; }

    Scalar getCenterOfMassElement(unsigned int i)
        {
        if(i == 0 )
            return m_center_of_mass.x;
        else if(i == 1 )
            return m_center_of_mass.y;
        else if (i == 2)
            return m_center_of_mass.z;
        else
            throw std::runtime_error("index out of range");
        }

    Scalar getInertiaTensor(unsigned int i)
        {
        if(i >= 6 )
            throw std::runtime_error("index out of range");
        return m_inertia[i];
        }

    Scalar getDeterminant()
        {
        vec3<Scalar> a(m_inertia[0], m_inertia[3], m_inertia[5]), b(m_inertia[3], m_inertia[1], m_inertia[4]), c(m_inertia[5], m_inertia[4], m_inertia[2]);
        return dot(a, cross(b,c));
        }

    virtual void updateParam(const typename Shape::param_type& param, bool force = true) { }

protected:
    virtual void compute() {throw std::runtime_error("mass_properties::compute() is not implemented for this shape.");}
    Scalar m_volume;
    vec3<Scalar> m_center_of_mass;
    Scalar m_inertia[6]; // xx, yy, zz, xy, yz, xz
};

template<class Shape>
class mass_properties : public mass_properties_base<Shape>
{
public:
    mass_properties() : mass_properties_base<Shape>() {}

    mass_properties(const typename Shape::param_type& shape) :  mass_properties_base<Shape>()
    {
    this->compute();
    }
};

inline void normalize_inplace(vec3<Scalar>& v) { v /= sqrt(dot(v,v)); }

inline vec3<Scalar> normalize(const vec3<Scalar>& v) { return v / sqrt(dot(v,v)); }
// face is assumed to be an array of indices of triangular face of a convex body.
// points may contain points inside or outside the body defined by faces.
// faces may include faces that contain vertices that are inside the body.
inline vec3<Scalar> getOutwardNormal(const std::vector< vec3<Scalar> >& points, const vec3<Scalar>& inside_point, const std::vector< std::vector<unsigned int> >& faces, const unsigned int& faceid, Scalar thresh = 0.0001)
    {
    const std::vector<unsigned int>& face = faces[faceid];
    vec3<Scalar> a = points[face[0]], b = points[face[1]], c = points[face[2]];
    vec3<Scalar> di = (inside_point - a), n;
    n = cross((b - a),(c - a));
    Scalar d = dot(n, di);
    if(fabs(d) < thresh)
        throw(std::runtime_error("ShapeUtils.h::getOutwardNormal -- inner point is in the plane"));
    return (d > 0) ? -n : n;
    }

inline void sortFace(const std::vector< vec3<Scalar> >& points, const vec3<Scalar>& inside_point, std::vector< std::vector<unsigned int> >& faces, const unsigned int& faceid, Scalar thresh = 0.0001)
    {
    assert(faces[faceid].size() == 3);
    vec3<Scalar> a = points[faces[faceid][0]], b = points[faces[faceid][1]], c = points[faces[faceid][2]], n, nout;
    nout = getOutwardNormal(points, inside_point, faces, faceid, thresh);
    n = cross((b - a),(c - a));
    if ( dot(nout, n) < 0 )
        std::reverse(faces[faceid].begin(), faces[faceid].end());
    }

inline void sortFaces(const std::vector< vec3<Scalar> >& points, std::vector< std::vector<unsigned int> >& faces, Scalar thresh = 0.0001)
    {
    vec3<Scalar> inside_point(0.0,0.0,0.0);
    for(size_t i = 0; i < points.size(); i++)
        {
        inside_point += points[i];
        }
    inside_point /= Scalar(points.size());

    for( unsigned int f = 0; f < faces.size(); f++ )
        sortFace(points, inside_point, faces, f, thresh);
    }

// Right now I am just solving the problem in 3d but I think that it should be easy to generalize to 2d as well.
class ConvexHull
{
    static const unsigned int invalid_index;
    static const Scalar       zero;
public:
    ConvexHull() { m_ravg = vec3<Scalar>(0.0,0.0,0.0); }
    template<unsigned int _max_verts>
    ConvexHull(const poly3d_verts<_max_verts>& param)
        {
        m_points.reserve(param.N);
        m_ravg = vec3<Scalar>(0.0,0.0,0.0);
        m_points.clear();
        for(unsigned int i = 0; i < param.N; i++)
            {
            m_points.push_back(vec3<Scalar>(param.x[i], param.y[i], param.z[i]));
            // m_ravg += m_points[i];
            }
        // m_ravg /= Scalar(param.N);
        }

    void compute()
        {
        if(m_points.size() < 4) // problem is not well posed.
            return;
        // My thoughts:
        // A note here is that the m_faces will be append only and we will mark it
        // for delete and remove all of the ones at the end. there could be a memory issue
        // with this but since max_verts is only 128 then there is really no reason to worry about that
        // since 128*127*126 = 2,048,256 maximum number of facets.

        // step 1: create a tetrahedron from the first 4 points.
        initialize(); // makes the tetrahedron
        std::vector<bool> inside(m_points.size(), false); // all points are outside.
        std::vector<unsigned int> outside(m_points.size(), invalid_index);
        for(unsigned int i = 0; i < m_points.size(); i++)
            {
            // step 2: initialize the outside and inside sets
            for(unsigned int f = 0; f < m_faces.size() && !inside[i]; f++)
                {
                if(m_deleted[f])
                    continue;
                if(outside[i] == invalid_index && is_above(i,f))
                    {
                    outside[i] = f;
                    break;
                    }
                }
            if(!inside[i] && outside[i] == invalid_index)
                {
                inside[i] = true;
                }
            }

        unsigned int faceid = 0;
        // write_pos_frame(inside);
        while(faceid < m_faces.size())
            {
            if(m_deleted[faceid]) // this facet is deleted so we can skip it.
                {
                faceid++;
                continue;
                }

            Scalar dist = 0.0;
            unsigned int _id = invalid_index;
            for(unsigned int out = 0; out < outside.size(); out++)
                {
                if(outside[out] == faceid)
                    {
                    Scalar sd = signed_distance(out, faceid);
                    assert(sd > zero);
                    if( sd > dist)
                        {
                        dist = sd;
                        _id = out;
                        }
                    }
                }
            if(_id == invalid_index) // no point found.
                {
                faceid++;
                continue;
                }

            // step 3: Find the visible set
            std::vector< unsigned int > visible;
            build_visible_set(_id, faceid, visible);
            // step 4: Build the new faces
            std::vector< std::vector<unsigned int> > new_faces;
            build_horizon_set(visible, new_faces); // boundary of the visible set
            assert(visible[0] == faceid);
            for(unsigned int dd = 0; dd < visible.size(); dd++)
                {
                m_deleted[visible[dd]] = true;
                }

            for(unsigned int i = 0; i < new_faces.size(); i++)
                {
                new_faces[i].push_back(_id);
                std::sort(new_faces[i].begin(), new_faces[i].end());
                unsigned int fnew = m_faces.size();
                m_faces.push_back(new_faces[i]);
                m_deleted.push_back(false);
                build_adjacency_for_face(fnew);
                for(unsigned int out = 0; out < outside.size(); out++)
                    {
                    for(unsigned int v = 0; v < visible.size() && !inside[out]; v++)
                        {
                        if(outside[out] == visible[v])
                            {
                            if(is_above(out, fnew))
                                {
                                outside[out] = fnew;
                                }
                            break;
                            }
                        }
                    }
                }
            // update the inside set for fun.
            for(unsigned int out = 0; out < outside.size(); out++)
                {
                if(outside[out] != invalid_index && m_deleted[outside[out]])
                    {
                    outside[out] = invalid_index;
                    inside[out] = true;
                    }
                }
            inside[_id] = true;
            assert(m_deleted.size() == m_faces.size() && m_faces.size() == m_adjacency.size());
            faceid++;
            // write_pos_frame(inside);
            #ifndef NDEBUG
            for(size_t i = 0; i < m_faces.size(); i++)
                {
                if(m_deleted[i]) continue;
                for(size_t j = i+1; j < m_faces.size(); j++)
                    {
                    if(m_deleted[j]) continue;
                    for(size_t k = 0; k < m_faces[j].size(); k++)
                        {
                        if(is_above(m_faces[j][k], i))
                            {
                            std::cout << "ERROR!!! point " << m_faces[j][k] << ": [" << m_points[m_faces[j][k]].x << ", " << m_points[m_faces[j][k]].y << m_points[m_faces[j][k]].z << "]" << std::endl
                                      << "         is above face " << i << ": [" << m_faces[i][0] << ", " << m_faces[i][1] << ", " << m_faces[i][2] << "]" << std::endl
                                      << "         from the face " << j << ": [" << m_faces[j][0] << ", " << m_faces[j][1] << ", " << m_faces[j][2] << "]" << std::endl;
                            throw std::runtime_error("ERROR in ConvexHull::compute() !");
                            }
                        }
                    }
                }
            #endif
            }
        remove_deleted_faces(); // actually remove the deleted faces.
        build_edge_list();
        sortFaces(m_points, m_faces, zero);
        }

private:
    void write_pos_frame(const std::vector<bool>& inside)
        {
        std::ofstream file("convex_hull.pos", std::ios_base::out | std::ios_base::app);
        std::string inside_sphere  = "def In \"sphere 0.1 005F5F5F\"";
        std::string outside_sphere  = "def Out \"sphere 0.1 00FF5F5F\"";
        std::string avg_sphere  = "def avg \"sphere 0.2 00981C1D\"";
        std::stringstream ss, connections;
        std::set<unsigned int> verts;
        for(size_t f = 0; f < m_faces.size(); f++)
            {
            if(m_deleted[f]) continue;
            verts.insert(m_faces[f].begin(), m_faces[f].end());
            for(size_t k = 0; k < 3; k++)
                connections << "connection 0.05 005F5FFF "<< m_points[m_faces[f][k]].x << " "<< m_points[m_faces[f][k]].y << " "<< m_points[m_faces[f][k]].z << " "
                                                          << m_points[m_faces[f][(k+1)%3]].x << " "<< m_points[m_faces[f][(k+1)%3]].y << " "<< m_points[m_faces[f][(k+1)%3]].z << std::endl;
            }
        ss << "def hull \"poly3d " << verts.size() << " ";
        for(std::set<unsigned int>::iterator iter = verts.begin(); iter != verts.end(); iter++)
            ss << m_points[*iter].x << " " << m_points[*iter].y << " " << m_points[*iter].z << " ";
        ss << "505984FF\"";
        std::string hull  = ss.str();

        // file<< "boxMatrix 10 0 0 0 10 0 0 0 10" << std::endl;
        file<< inside_sphere << std::endl;
        file<< outside_sphere << std::endl;
        file<< avg_sphere << std::endl;
        // file<< hull << std::endl;
        // file << "hull 0 0 0 1 0 0 0" << std::endl;
        file << connections.str();
        file << "avg "<< m_ravg.x << " " << m_ravg.y << " " << m_ravg.z << " " << std::endl;
        for(size_t i = 0; i < m_points.size(); i++)
            {
            if(inside[i])
                file << "In ";
            else
                file << "Out ";
            file << m_points[i].x << " " << m_points[i].y << " " << m_points[i].z << " " << std::endl;
            }
        file << "eof" << std::endl;
        }

    Scalar signed_distance(const unsigned int& i, const unsigned int& faceid)
        {
        vec3<Scalar> n = getOutwardNormal(m_points, m_ravg, m_faces, faceid, zero);
        vec3<Scalar> dx = m_points[i] -  m_points[m_faces[faceid][0]];
        normalize_inplace(dx);
        return dot(dx, n); // signed distance. eiter in the plane or outside.
        }

    bool is_above(const unsigned int& i, const unsigned int& faceid)
        {
        if(i == m_faces[faceid][0] || i == m_faces[faceid][1] || i == m_faces[faceid][2])
            return false;
        return (signed_distance(i, faceid) > zero); // signed distance. either in the plane or outside.
        }

    bool is_coplanar(const unsigned int& i, const unsigned int& j, const unsigned int& k, const unsigned int& l)
        {
        if( i == j || i == k || i == l || j == k || j == l || k == l)
            return true;

        vec3<Scalar> d1 = m_points[j] - m_points[i];
        normalize_inplace(d1);
        vec3<Scalar> d2 = m_points[k] - m_points[i];
        normalize_inplace(d2);
        vec3<Scalar> d3 = m_points[l] - m_points[i];
        normalize_inplace(d3);
        Scalar d = dot(d3, cross(d1, d2));
        return fabs(d) <= zero;
        }

    void edges_from_face(const unsigned int& faceid, std::vector< std::vector<unsigned int> >& edges)
        {
        assert(faceid < m_faces.size());
        unsigned int N = m_faces[faceid].size();
        assert(N == 3);
        assert(!m_deleted[faceid]);
        for(unsigned int i = 0; i < m_faces[faceid].size(); i++)
            {
            std::vector<unsigned int> edge;
            unsigned int e1 = m_faces[faceid][i], e2 = m_faces[faceid][(i+1) % N];
            assert(e1 < m_points.size() && e2 < m_points.size());
            edge.push_back(min(e1, e2));
            edge.push_back(max(e1, e2));
            edges.push_back(edge);
            }
        }

    void initialize()
        {
        const unsigned int Nsym = 4; // number of points in the simplex.
        unsigned int ik[Nsym] = {invalid_index, invalid_index, invalid_index, invalid_index}; // indices of the four points.
        ik[0] = ik[1] = ik[2] = ik[3] = invalid_index;
        m_faces.clear(); m_faces.reserve(100000);
        m_edges.clear(); m_edges.reserve(100000);
        m_deleted.clear(); m_deleted.reserve(100000);
        m_adjacency.clear(); m_adjacency.reserve(100000);

        if(m_points.size() < Nsym) // TODO: the problem is basically done. but need to set up the data structures and return. not common in our use case so put it off until later.
            {
            throw(std::runtime_error("Could not initialize ConvexHull: need 4 points to take the convex hull in 3D"));
            }

        ik[0] = 0;
        bool coplanar = true;
        while( coplanar )
            {
            for(size_t k = 1; k < Nsym; k++)
                {
                std::vector<Scalar> min_dsq(m_points.size());
                for(size_t p = 0; p < m_points.size(); p++)
                    {
                    for(size_t i = 0; i < k; i++)
                        {
                        vec3<Scalar> dr = m_points[p] - m_points[ik[i]];
                        Scalar dsq = dot(dr, dr);
                        if(i == 0)
                            {
                            min_dsq[p] = dsq;
                            }
                        else
                            {
                            if(dsq < min_dsq[p])
                                {
                                min_dsq[p] = dsq;
                                }

                            if(k == Nsym-1 && is_coplanar(ik[0], ik[1], ik[2], p))
                                {
                                min_dsq[p] = 0; // take the point out of the running.
                                }
                            }
                        }
                    }
                ik[k] = std::distance(min_dsq.begin(), std::max_element(min_dsq.begin(), min_dsq.end()));
                }

            if(!is_coplanar(ik[0], ik[1], ik[2], ik[3]))
                {
                coplanar = false;
                }
            else
                {
                ik[0]++;
                ik[1] = ik[2] = ik[3] = invalid_index;
                if( ik[0] >= m_points.size() ) // tried all of the points and this will not.
                    {
                    ik[0] = invalid_index; // exit loop and throw an error.
                    coplanar = false;
                    }
                }
            }
        if(ik[0] == invalid_index || ik[1] == invalid_index || ik[2] == invalid_index || ik[3] == invalid_index)
            {
            std::cerr << std::endl << std::endl<< "*************************" << std::endl;
            for(size_t i = 0; i < m_points.size(); i++)
                {
                std::cerr << "point " << i << ": [" << m_points[i].x << ", " << m_points[i].y << ", " << m_points[i].z << "]" << std::endl;
                }
            throw(std::runtime_error("Could not initialize ConvexHull: found only nearly coplanar points"));
            }
        m_ravg = vec3<Scalar>(0,0,0);
        for(size_t i = 0; i < Nsym; i++)
            {
            m_ravg += m_points[ik[i]];
            }
        m_ravg /= Scalar(Nsym);

        std::vector<unsigned int> face(3);
        // face 0
        face[0] = ik[0]; face[1] = ik[1]; face[2] = ik[2];
        std::sort(face.begin(), face.end());
        m_faces.push_back(face);
        // face 1
        face[0] = ik[0]; face[1] = ik[1]; face[2] = ik[3];
        std::sort(face.begin(), face.end());
        m_faces.push_back(face);
        // face 2
        face[0] = ik[0]; face[1] = ik[2]; face[2] = ik[3];
        std::sort(face.begin(), face.end());
        m_faces.push_back(face);
        // face 3
        face[0] = ik[1]; face[1] = ik[2]; face[2] = ik[3];
        std::sort(face.begin(), face.end());
        m_faces.push_back(face);
        m_deleted.resize(4, false); // we have 4 facets at this point.
        build_adjacency_for_face(0);
        build_adjacency_for_face(1);
        build_adjacency_for_face(2);
        build_adjacency_for_face(3);
        }

    void build_adjacency_for_face(const unsigned int& f)
        {
        if(f >= m_faces.size())
            throw std::runtime_error("index out of range!");
        if(m_deleted[f]) return; // don't do anything there.

        m_adjacency.resize(m_faces.size());
        for(unsigned int g = 0; g < m_faces.size(); g++)
            {
            if(m_deleted[g] || g == f) continue;
            // note this is why we need the faces to be sorted here.
            std::vector<unsigned int> intersection(3, 0);
            assert(m_faces[f].size() == 3 && m_faces[g].size() == 3);
            std::vector<unsigned int>::iterator it = std::set_intersection(m_faces[f].begin(), m_faces[f].end(), m_faces[g].begin(), m_faces[g].end(), intersection.begin());
            intersection.resize(it-intersection.begin());
            if(intersection.size() == 2)
                {
                m_adjacency[f].insert(g);  // always insert both ways
                m_adjacency[g].insert(f);
                }
            }
        }

    void build_visible_set(const unsigned int& pointid, const unsigned int& faceid, std::vector< unsigned int >& visible)
        {
        // std::cout << "building visible set point: "<< pointid << " face: " << faceid << std::endl;
        visible.clear();
        visible.push_back(faceid);
        std::queue<unsigned int> worklist;
        std::vector<bool> found(m_deleted);
        worklist.push(faceid);
        found[faceid] = true;
        // std::cout << "point id: " << pointid << ", face id: " << faceid << std::endl;
        while(!worklist.empty())
            {
            unsigned int f = worklist.front();
            worklist.pop();
            // std::cout << "face " << f << ": " << "[ " << m_faces[f][0] << ", " << m_faces[f][1] << ", " << m_faces[f][2] << "] "<< std::endl;
            if(m_deleted[f]) continue;
            // std::cout << " m_adjacency.size = "<< m_adjacency.size() << " m_adjacency["<<f<<"].size = "<< m_adjacency[f].size() << std::endl;
            for(std::set<unsigned int>::iterator i = m_adjacency[f].begin(); i != m_adjacency[f].end(); i++)
                {
                // std::cout << "found: " << found[*i] << " - neighbor "<< *i << ": " << "[ " << m_faces[*i][0] << ", " << m_faces[*i][1] << ", " << m_faces[*i][2] << "] "<< std::endl;
                if(!found[*i]) // face was not found yet and the point is above the face.
                    {
                    found[*i] = true;
                    if( is_above(pointid, *i) )
                        {
                        assert(!m_deleted[*i]);
                        worklist.push(*i);
                        visible.push_back(*i);
                        }
                    }
                }
            }
        }

    void build_horizon_set(const std::vector< unsigned int >& visible, std::vector< std::vector<unsigned int> >& horizon)
        {
        std::vector< std::vector<unsigned int> > edges;
        for(unsigned int i = 0; i < visible.size(); i++)
            edges_from_face(visible[i], edges); // all visible edges.
        std::vector<bool> unique(edges.size(), true);
        for(unsigned int i = 0; i < edges.size(); i++)
            {
            for(unsigned int j = i+1; j < edges.size() && unique[i]; j++)
                {
                if( (edges[i][0] == edges[j][0] && edges[i][1] == edges[j][1]) ||
                    (edges[i][1] == edges[j][0] && edges[i][0] == edges[j][1]) )
                    {
                    unique[i] = false;
                    unique[j] = false;
                    }
                }
            if(unique[i])
                {
                horizon.push_back(edges[i]);
                }
            }
        }

    void build_edge_list()
        {
        return;
        std::vector< std::vector<unsigned int> > edges;
        for(unsigned int i = 0; i < m_faces.size(); i++)
            edges_from_face(i, edges); // all edges.

        for(unsigned int i = 0; i < edges.size(); i++)
            {
            bool unique = true;
            for(unsigned int j = i+1; j < edges.size(); j++)
                {
                if(edges[i][0] == edges[j][0] && edges[i][1] == edges[j][1])
                    {
                    unique = false;
                    }
                }
            if(unique)
                {
                m_edges.push_back(edges[i]);
                }
            }
        }

    void remove_deleted_faces()
        {
        std::vector< std::vector<unsigned int> >::iterator f;
        std::vector< bool >::iterator d;
        bool bContinue = true;
        while(bContinue)
            {
            bContinue = false;
            d = m_deleted.begin();
            f = m_faces.begin();
            for(; f != m_faces.end() && d != m_deleted.end(); f++, d++)
                {
                if(*d)
                    {
                    m_faces.erase(f);
                    m_deleted.erase(d);
                    bContinue = true;
                    break;
                    }
                }
            }
        m_adjacency.clear(); // the id's of the faces are all different so just clear the list.
        }

public:
    const std::vector< std::vector<unsigned int> >& getFaces() { return m_faces; }

    const std::vector< std::vector<unsigned int> >& getEdges() { return m_edges; }

    const std::vector< vec3<Scalar> >& getPoints() { return m_points; }

    void moveData(std::vector< std::vector<unsigned int> >& faces, std::vector< vec3<Scalar> >& points)
        {
        // NOTE: *this is not valid after using this method!
        faces = std::move(m_faces);
        points = std::move(m_points);
        }

protected:
    vec3<Scalar>                                m_ravg;
    std::vector< vec3<Scalar> >                 m_points;
    std::vector< std::vector<unsigned int> >    m_faces; // Always have 3 vertices in a face.
    std::vector< std::vector<unsigned int> >    m_edges; // Always have 2 vertices in an edge.
    std::vector< std::set<unsigned int> >       m_adjacency; // the face adjacency list.
    std::vector<bool>                           m_deleted;
};

template<unsigned int max_verts>
class mass_properties< ShapeConvexPolyhedron<max_verts> > : public mass_properties_base< ShapeConvexPolyhedron<max_verts> >
{
using mass_properties_base< ShapeConvexPolyhedron<max_verts> >::m_volume;
using mass_properties_base< ShapeConvexPolyhedron<max_verts> >::m_center_of_mass;
using mass_properties_base< ShapeConvexPolyhedron<max_verts> >::m_inertia;

public:
    mass_properties() {}

    mass_properties(const typename ShapeConvexPolyhedron<max_verts>::param_type& param)
        {
        ConvexHull hull(param);
        hull.compute();
        hull.moveData(faces, points);
        compute();
        }

    mass_properties(const std::vector< vec3<Scalar> >& p, const std::vector<std::vector<unsigned int> >& f) :  points(p), faces(f)
        {
        compute();
        }

    unsigned int getFaceIndex(unsigned int i, unsigned int j) { return faces[i][j]; }

    unsigned int getNumFaces() { return faces.size(); }

    void updateParam(const typename ShapeConvexPolyhedron<max_verts>::param_type& param, bool force = true)
        {
        if(force || param.N != points.size())
            {
            ConvexHull hull(param);
            hull.compute();
            hull.moveData(faces, points);
            }
        else
            {
            // assumes that the faces are still good.
            for(unsigned int i = 0; i < param.N; i++)
                {
                points[i] = vec3<Scalar>(param.x[i], param.y[i], param.z[i]);
                }
            }
        compute();
        }

protected:
/*
    algorithm taken from
    http://www.geometrictools.com/Documentation/PolyhedralMassProperties.pdf
*/
    virtual void compute()
        {
        // const std::vector<std::vector<unsigned int> >& faces = convex_hull.getFaces();
        // const std::vector< vec3<Scalar> >& points = convex_hull.getPoints();
        const Scalar mult[10] = {1.0/6.0 ,1.0/24.0 ,1.0/24.0 ,1.0/24.0 ,1.0/60.0 ,1.0/60.0 ,1.0/60.0 ,1.0/120.0 ,1.0/120.0 ,1.0/120.0};
        Scalar intg[10] = {0,0,0,0,0,0,0,0,0,0}; // order: 1, x, y, z, xˆ2, yˆ2, zˆ2, xy, yz, zx
        for (unsigned int t=0; t<faces.size(); t++)
            {
            //get vertices of triangle
            vec3<Scalar> v0, v1, v2;
            vec3<Scalar> a1, a2, d;
            v0 = points[faces[t][0]];
            v1 = points[faces[t][1]];
            v2 = points[faces[t][2]];
            // get edges and cross product of edges
            a1 = v1 - v0;
            a2 = v2 - v0;
            d = cross(a1, a2);

            vec3<Scalar> temp0, temp1, temp2, f1, f2, f3, g0, g1, g2;
            temp0 = v0 + v1;
            f1 = temp0 + v2;
            temp1 = v0*v0;
            temp2 = temp1 + v1*temp0;
            f2 = temp2 + v2*f1;
            f3 = v0*temp1 + v1*temp2 + v2*f2;
            g0 = f2 + v0*(f1 + v0);
            g1 = f2 + v1*(f1 + v1);
            g2 = f2 + v2*(f1 + v2);

            intg[0] += d.x*f1.x;
            intg[1] += d.x*f2.x; intg[2] += d.y*f2.y; intg[3] += d.z*f2.z;
            intg[4] += d.x*f3.x; intg[5] += d.y*f3.y; intg[6] += d.z*f3.z;
            intg[7] += d.x*(v0.y*g0.x + v1.y*g1.x + v2.y*g2.x);
            intg[8] += d.y*(v0.z*g0.y + v1.z*g1.y + v2.z*g2.y);
            intg[9] += d.z*(v0.x*g0.z + v1.x*g1.z + v2.x*g2.z);
            }
        for(unsigned int i = 0; i < 10; i++ )
            {
            intg[i] *= mult[i];
            }

        m_volume = intg[0];

        m_center_of_mass.x = intg[1];
        m_center_of_mass.y = intg[2];
        m_center_of_mass.z = intg[3];
        m_center_of_mass /= m_volume;

        Scalar cx2 = m_center_of_mass.x*m_center_of_mass.x, cy2 = m_center_of_mass.y*m_center_of_mass.y, cz2 = m_center_of_mass.z*m_center_of_mass.z;
        Scalar cxy = m_center_of_mass.x*m_center_of_mass.y, cyz = m_center_of_mass.y*m_center_of_mass.z, cxz = m_center_of_mass.x*m_center_of_mass.z;
        m_inertia[0] = intg[5] + intg[6] - m_volume*(cy2 + cz2);
        m_inertia[1] = intg[4] + intg[6] - m_volume*(cz2 + cx2);
        m_inertia[2] = intg[4] + intg[5] - m_volume*(cx2 + cy2);
        m_inertia[3] = -(intg[7] - m_volume*cxy);
        m_inertia[4] = -(intg[8] - m_volume*cyz);
        m_inertia[5] = -(intg[9] - m_volume*cxz);
        }
private:
    std::vector< vec3<Scalar> > points;
    std::vector<std::vector<unsigned int> > faces;
};

} // end namespace detail

template<class Shape>
void export_massPropertiesBase(pybind11::module& m, std::string name);

template<class Shape>
void export_massProperties(pybind11::module& m, std::string name);

} // end namespace hpmc
#endif // end inclusion guard
