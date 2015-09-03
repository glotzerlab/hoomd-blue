/*
Highly Optimized Object-oriented Many-particle Dynamics -- Blue Edition
(HOOMD-blue) Open Source Software License Copyright 2009-2015 The Regents of
the University of Michigan All rights reserved.

HOOMD-blue may contain modifications ("Contributions") provided, and to which
copyright is held, by various Contributors who have granted The Regents of the
University of Michigan the right to modify and/or distribute such Contributions.

You may redistribute, use, and create derivate works of HOOMD-blue, in source
and binary forms, provided you abide by the following conditions:

* Redistributions of source code must retain the above copyright notice, this
list of conditions, and the following disclaimer both in the code and
prominently in any materials provided with the distribution.

* Redistributions in binary form must reproduce the above copyright notice, this
list of conditions, and the following disclaimer in the documentation and/or
other materials provided with the distribution.

* All publications and presentations based on HOOMD-blue, including any reports
or published results obtained, in whole or in part, with HOOMD-blue, will
acknowledge its use according to the terms posted at the time of submission on:
http://codeblue.umich.edu/hoomd-blue/citations.html

* Any electronic documents citing HOOMD-Blue will link to the HOOMD-Blue website:
http://codeblue.umich.edu/hoomd-blue/

* Apart from the above required attributions, neither the name of the copyright
holder nor the names of HOOMD-blue's contributors may be used to endorse or
promote products derived from this software without specific prior written
permission.

Disclaimer

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDER AND CONTRIBUTORS ``AS IS'' AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, AND/OR ANY
WARRANTIES THAT THIS SOFTWARE IS FREE OF INFRINGEMENT ARE DISCLAIMED.

IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT,
INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE
OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF
ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

// Maintainer: jproc

/*! \file WallData.h
    \brief Contains declarations for WallData.
 */
#ifndef WALL_DATA_H
#define WALL_DATA_H


#include "HOOMDMath.h"
#include "VectorMath.h"
#include <cstdlib>
#include <vector>
#include <boost/python.hpp>
#include <string.h>




//#include "QuaternionMath.h"//doesn't seem to work, might use just vector math

//is the inside portion of the data structure needed?
//how easy would applying potentials to the outside be as well? could we make easy?

struct SphereWall
    {
    SphereWall() {}
    SphereWall(Scalar r, vec3<Scalar> orig, bool ins = true) : inside(ins), origin(orig) {}
    Scalar          r;
    bool            inside;
    vec3<Scalar>    origin;
    };

struct CylinderWall
    {
    CylinderWall() {}
    CylinderWall(Scalar r, vec3<Scalar> orig, vec3<Scalar> zorient, bool ins=true) : inside(ins), origin(orig), orientation(zorient)
        {
        vec3<Scalar> zvec;
        zvec=zorient;
        vec3<Scalar> znorm(0,0,1);

        //method source: http://lolengine.net/blog/2014/02/24/quaternion-from-two-vectors-final
        Scalar norm_uv=sqrt(dot(znorm,znorm) * dot(zvec,zvec));
        Scalar real_part=norm_uv + dot(znorm,zvec);
        vec3<Scalar> w;

        if (real_part < Scalar(1.0e-6) * norm_uv)
            {
                real_part=Scalar(0.0);
                w=fabs(znorm.x) > fabs(znorm.z) ? vec3<Scalar>(-znorm.y, znorm.x, 0.0)
                : vec3<Scalar>(0.0, -znorm.z, znorm.y);
            }
        else
            {
                w=cross(znorm,zvec);
                real_part=Scalar(real_part);
            }
            q_reorientation=quat<Scalar>(real_part,w);
            Scalar norm=fast::rsqrt(norm2(q_reorientation));
            q_reorientation=norm*q_reorientation;
        }
    Scalar          r;
    bool            inside;
    vec3<Scalar>    origin;
    vec3<Scalar>    orientation;
    quat<Scalar>    q_reorientation;
    };

struct PlaneWall
    {
    PlaneWall(){}
    PlaneWall(vec3<Scalar> nvec, vec3<Scalar> pt) : normal(nvec), origin(pt), inside(true)
        {
        Scalar n_length;
        n_length=fast::rsqrt(nvec.x*nvec.x + nvec.y*nvec.y + nvec.z*nvec.z);
        normal=nvec*n_length;
        }
    vec3<Scalar>    normal;
    vec3<Scalar>    origin;
    bool            inside; //TODO: Figure out if it can be removed without messing up HPMC walls. Meaningless usage since it could simply be defined by the negative of the given normal.
    };

/*template <class WallShape>
DEVICE inline Scalar wall_dist_eval(const WallShape& wall, const vec3<Scalar>& position, const vec3<Scalar>& box_origin, const BoxDim& box)
    {
    return false;//something else needs to happen here and particle shape was removed, anything need to be added?
    };

template <class evlauator>
DEVICE inline Scalar wall_dist_eval<SphereWall>(const SphereWall& wall, const vec3<Scalar>& position, const vec3<Scalar>& box_origin, const BoxDim& box)
    {
    vec3<Scalar> t = position - box_origin;
    box.minImage(t);
    t-=wall.origin;
    vec3<Scalar> shifted_pos(t);
    Scalar rxyz_sq = shifted_pos.x*shifted_pos.x + shifted_pos.y*shifted_pos.y + shifted_pos.z*shifted_pos.z; // sq distance from the container origin.
    Scalar wall_dist = wall.r - sqrt(rxyz_sq);
    return wall_dist;
    };

template <class evlauator>
DEVICE inline Scalar wall_dist_eval<CylinderWall>(const CylinderWall& wall, const vec3<Scalar>& position, const vec3<Scalar>& box_origin, const BoxDim& box)
    {
    vec3<Scalar> t = position - box_origin;
    box.minImage(t);
    t-=wall.origin;
    vec3<Scalar> shifted_pos=rotate(wall.q_reorientation,t);
    Scalar rxy_sq= shifted_pos.x*shifted_pos.x + shifted_pos.y*shifted_pos.y; // sq distance from the container central axis.
    Scalar wall_dist = wall.r - sqrt(rxy_sq); //any way around the sqrt use? probably not, could only do if pair potentials were based off r_sq
    return wall_dist;
    };

template <class evlauator>  //make clear in documentation that plane placement is only considered inside 1 box?
DEVICE inline Scalar wall_dist_eval<PlaneWall>(const PlaneWall& wall, const vec3<Scalar>& position, const vec3<Scalar>& box_origin, const BoxDim& box)
    {
    vec3<Scalar> t = position - box_origin;
    box.minImage(t);
    Scalar wall_dist =dot(wall.normal,t)-dot(wall.normal,wall.origin);
    //Scalar n_length=sqrt(wall.normal.x*wall.normal.x + wall.normal.y*wall.normal.y + wall.normal.z*wall.normal.z); taken out because it makes more sense to normalize the vector once rather than many times
    //wall_dist = wall_dist/n_length;
    return wall_dist;
    };*/

class WallDataNew : boost::noncopyable
    {
    public:

        WallDataNew() : m_Spheres(), m_Cylinders(), m_Planes() {}

        WallDataNew(const std::vector<SphereWall>& walls)
            {
            m_Spheres = walls;
            }

        WallDataNew(const std::vector<CylinderWall>& walls)
            {
            m_Cylinders = walls;
            }

        WallDataNew(const std::vector<PlaneWall>& walls)
            {
            m_Planes= walls;
            }

        void setSphereWallParameter(size_t index, SphereWall wall)
            {
            m_Spheres[index] = wall;
            }

        void setCylinderWallParameter(size_t index, CylinderWall wall)
            {
            m_Cylinders[index] = wall;
            }

        void setPlaneWallParameter(size_t index, PlaneWall wall)
            {
            m_Planes[index] = wall;
            }

        void addSphereWall(const SphereWall& wall);

        void addCylinderWall(const CylinderWall& wall);

        void addPlaneWall(const PlaneWall& wall);

        unsigned int getNumSphereWalls() const
            {
            return (unsigned int)m_Spheres.size();
            }

        unsigned int getNumCylinderWalls() const
            {
            return (unsigned int)m_Cylinders.size();
            }

        unsigned int getNumPlaneWalls() const
            {
            return (unsigned int)m_Planes.size();
            }

        const SphereWall& getSWall(unsigned int idx) const
            {
            assert(idx < m_Spheres.size());
            return m_Spheres[idx];
            }

        const CylinderWall& getCWall(unsigned int idx) const
            {
            assert(idx < m_Cylinders.size());
            return m_Cylinders[idx];
            }

        const PlaneWall& getPWall(unsigned int idx) const
            {
            assert(idx < m_Planes.size());
            return m_Planes[idx];
            }

        void removeSphereWalls()
            {
            m_Spheres.clear();
            }

        void removeCylinderWalls()
            {
            m_Cylinders.clear();
            }

        void removePlaneWalls()
            {
            m_Planes.clear();
            }

        void removeAllWalls()
            {
            m_Spheres.clear();
            m_Cylinders.clear();
            m_Planes.clear();
            }

    protected:
        std::vector<SphereWall>     m_Spheres;
        std::vector<CylinderWall>   m_Cylinders;
        std::vector<PlaneWall>      m_Planes;
    };



// template<class Wall>
// void export_Wall(const std::string& name)
// {
//     class_<Wall<WallShape>, boost::shared_ptr< Wall<WallShape>>, bases<ForceCompute, Compute >, boost::noncopyable>
//     (name.c_str(), init< boost::shared_ptr<SystemDefinition> >())
//     .def("SetSphereWallParameter", &ExternalFieldWall<Shape>::SetSphereWallParameter)
//     .def("SetCylinderWallParameter", &ExternalFieldWall<Shape>::SetCylinderWallParameter)
//     .def("SetPlaneWallParameter", &ExternalFieldWall<Shape>::SetPlaneWallParameter)
//     .def("AddSphereWall", &ExternalFieldWall<Shape>::AddSphereWall)
//     .def("AddCylinderWall", &ExternalFieldWall<Shape>::AddCylinderWall)
//     .def("AddPlaneWall", &ExternalFieldWall<Shape>::AddPlaneWall)
//     ;
// };
//
// SphereWall make_sphere_wall(Scalar r, boost::python::list origin, bool inside)
//     {
//     vec3<Scalar> orig;
//     orig.x = boost::python::extract<Scalar>(origin[0]);
//     orig.y = boost::python::extract<Scalar>(origin[1]);
//     orig.z = boost::python::extract<Scalar>(origin[2]);
//     return SphereWall(r, orig, inside);
//     };
//
// CylinderWall make_cylinder_wall(Scalar r, boost::python::list origin, boost::python::list zorientation, bool inside)
// 	{
// 	vec3<Scalar> orig;
//     orig.x = boost::python::extract<Scalar>(origin[0]);
//     orig.y = boost::python::extract<Scalar>(origin[1]);
//     orig.z = boost::python::extract<Scalar>(origin[2]);
//     vec3<Scalar> zorient;
//     zorient.x = boost::python::extract<Scalar>(zorientation[0]);
//     zorient.y = boost::python::extract<Scalar>(zorientation[1]);
//     zorient.z = boost::python::extract<Scalar>(zorientation[2]);
//     return CylinderWall(r, orig, zorient, inside);
// 	};
//
// PlaneWall make_plane_wall(boost::python::list norm, boost::python::list origin)
//     {
//     vec3<Scalar> orig;
//     orig.x = boost::python::extract<Scalar>(origin[0]);
//     orig.y = boost::python::extract<Scalar>(origin[1]);
//     orig.z = boost::python::extract<Scalar>(origin[2]);
//     vec3<Scalar> normal;
//     normal.x = boost::python::extract<Scalar>(norm[0]);
//     normal.y = boost::python::extract<Scalar>(norm[1]);
//     normal.z = boost::python::extract<Scalar>(norm[2]);
//     return PlaneWall(normal, orig);
//     };
//
// void export_walls()
//     {
//     // export wall structs.
//     class_<SphereWall, boost::shared_ptr<SphereWall> >("sphere_wall_parms", no_init);
//     class_<CylinderWall, boost::shared_ptr<CylinderWall> >("cylinder_wall_params", no_init);
//     class_<PlaneWall, boost::shared_ptr<PlaneWall> >("plane_wall_parms", no_init);
//
//     // export helper functions.
//     def("make_sphere_wall", &make_sphere_wall);
//     def("make_cylinder_wall", &make_cylinder_wall);
//     def("make_plane_wall", &make_plane_wall);
//
//     };

//! Simple structure representing a single wall
/*! Walls are represented by an origin and a unit length normal.
    \ingroup data_structs
*/
struct Wall
    {
    //! Constructor
    /*! \param ox Origin x-component
        \param oy Origin y-component
        \param oz Origin z-component
        \param nx Origin x-component
        \param ny Normal y-component
        \param nz Normal z-component
    */
    Wall(Scalar ox=0.0, Scalar oy=0.0, Scalar oz=0.0, Scalar nx=1.0, Scalar ny=0.0, Scalar nz=0.0)
            : origin_x(ox), origin_y(oy), origin_z(oz)
        {
        // normalize nx,ny,nz
        Scalar len = sqrt(nx*nx + ny*ny + nz*nz);
        normal_x = nx / len;
        normal_y = ny / len;
        normal_z = nz / len;
        }

    Scalar origin_x;    //!< x-component of the origin
    Scalar origin_y;    //!< y-component of the origin
    Scalar origin_z;    //!< z-component of the origin

    Scalar normal_x;    //!< x-component of the normal
    Scalar normal_y;    //!< y-component of the normal
    Scalar normal_z;    //!< z-component of the normal
    };

//! Stores information about all the walls defined in the simulation
/*! WallData is responsible for storing all of the walls in the simulation.
    Walls are specified by the Wall struct and any number can be added.

    On the CPU, walls can be accessed with getWall()

    An optimized data structure for the GPU will be written later.
    It will most likely take the form of a 2D texture.
    \ingroup data_structs
*/
class WallData : boost::noncopyable
    {
    public:
        //! Creates an empty structure with no walls
        WallData() : m_walls() {}

        //! Creates a WallData from a list of walls
        WallData(const std::vector<Wall>& walls)
            {
            m_walls = walls;
            }

        //! Get the number of walls in the data
        /*! \return Number of walls
        */
        unsigned int getNumWalls() const
            {
            return (unsigned int)m_walls.size();
            }

        //! Access a specific wall
        /*! \param idx Index of the wall to retrieve
            \return Wall stored at index \a idx
        */
        const Wall& getWall(unsigned int idx) const
            {
            assert(idx < m_walls.size());
            return m_walls[idx];
            }

        //! Adds a wall to the data structure
        void addWall(const Wall& wall);

        //! Removes all walls
        void removeAllWalls()
            {
            m_walls.clear();
            }

    private:
        //! Storage for the walls
        std::vector<Wall> m_walls;
    };
#endif
//
// void export_WallDataNew(const std::string& name)
//     {
//     boost::python::class_<T, boost::shared_ptr<T>, boost::python::bases<ForceCompute>, boost::noncopyable >
//         (name.c_str(), boost::python::init< boost::shared_ptr<SystemDefinition>, const std::string& >())
//         .def("SetSphereWallParameter", &WallDataNew::setSphereWallParameter)
//         .def("SetCylinderWallParameter", &WallDataNew::setCylinderWallParameter)
//         .def("SetPlaneWallParameter", &WallDataNew::setPlaneWallParameter)
//         .def("AddSphereWall", &WallDataNew::addSphereWall)
//         .def("AddCylinderWall", &WallDataNew::addCylinderWall)
//         .def("AddPlaneWall", &WallDataNew::addPlaneWall)
//         .def("RemoveSphereWall", &WallDataNew::removeSphereWall)
//         .def("RemoveCylinderWall", &WallDataNew::removeCylinderWall)
//         .def("RemovePlaneWall", &WallDataNew::removePlaneWall)
//         ;
//     }
