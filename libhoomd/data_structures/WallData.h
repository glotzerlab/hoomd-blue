/*
Highly Optimized Object-oriented Many-particle Dynamics -- Blue Edition
(HOOMD-blue) Open Source Software License Copyright 2009-2016 The Regents of
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
    \brief Contains declarations for all types (currently Sphere, Cylinder, and
    Plane) of WallData and associated utilities.
 */
#ifndef WALL_DATA_H
#define WALL_DATA_H

#include "HOOMDMath.h"
#include "VectorMath.h"
#include <cstdlib>
#include <vector>
#include <string.h>

#ifdef NVCC
#define DEVICE __device__
#else
#define DEVICE
#endif

//! SphereWall Constructor
/*! \param r Radius of the sphere
    \param origin The x,y,z coordinates of the center of the sphere
    \param inside Determines which half space is evaluated.
*/
struct SphereWall
    {
    SphereWall(Scalar rad = 0.0, Scalar3 orig = make_scalar3(0.0,0.0,0.0), bool ins = true) : r(rad), origin(vec3<Scalar>(orig)), inside(ins) {}
    Scalar          r;
    vec3<Scalar>    origin;
    bool            inside;
    };

//! CylinderWall Constructor
/*! \param r Radius of the sphere
    \param origin The x,y,z coordinates of a point on the cylinder axis
    \param axis A x,y,z vector along the cylinder axis used to define the axis
    \param quatAxisToZRot (Calculated not input) The quaternion which rotates the simulation space such that the axis of the cylinder is parallel to the z' axis
    \param inside Determines which half space is evaluated.
*/
struct CylinderWall
    {
    CylinderWall(Scalar rad = 0.0, Scalar3 orig = make_scalar3(0.0,0.0,0.0), Scalar3 zorient = make_scalar3(0.0,0.0,1.0), bool ins=true) : r(rad),  origin(vec3<Scalar>(orig)), axis(vec3<Scalar>(zorient)), inside(ins)
        {
        vec3<Scalar> zVec=axis;
        vec3<Scalar> zNorm(0.0,0.0,1.0);

        // method source: http://lolengine.net/blog/2014/02/24/quaternion-from-two-vectors-final
        // easily simplified due to zNorm being a normalized vector
        Scalar normVec=sqrt(dot(zVec,zVec));
        Scalar realPart=normVec + dot(zNorm,zVec);
        vec3<Scalar> w;

        if (realPart < Scalar(1.0e-6) * normVec)
            {
                realPart=Scalar(0.0);
                w=vec3<Scalar>(0.0, -1.0, 0.0);
            }
        else
            {
                w=cross(zNorm,zVec);
                realPart=Scalar(realPart);
            }
            quatAxisToZRot=quat<Scalar>(realPart,w);
            Scalar norm=fast::rsqrt(norm2(quatAxisToZRot));
            quatAxisToZRot=norm*quatAxisToZRot;
        }
    Scalar          r;
    vec3<Scalar>    origin;
    vec3<Scalar>    axis;
    quat<Scalar>    quatAxisToZRot;
    bool            inside;
    };

//! PlaneWall Constructor
/*! \param origin The x,y,z coordinates of a point on the cylinder axis
    \param normal The x,y,z normal vector of the plane (normalized upon input)
    \param inside Determines which half space is evaluated.
*/
struct PlaneWall
    {
    PlaneWall(Scalar3 orig = make_scalar3(0.0,0.0,0.0), Scalar3 norm = make_scalar3(0.0,0.0,1.0), bool ins = true) : normal(vec3<Scalar>(norm)), origin(vec3<Scalar>(orig)), inside(ins)
        {
        vec3<Scalar> nVec;
        nVec = normal;
        Scalar invNormLength;
        invNormLength=fast::rsqrt(nVec.x*nVec.x + nVec.y*nVec.y + nVec.z*nVec.z);
        normal=nVec*invNormLength;
        }
    vec3<Scalar>    normal;
    vec3<Scalar>    origin;
    bool            inside;
    };

//! Point to wall vector for a sphere wall geometry
DEVICE inline vec3<Scalar> vecPtToWall(const SphereWall& wall, const vec3<Scalar>& position, bool& inside)
    {
    vec3<Scalar> t = position;
    t-=wall.origin;
    vec3<Scalar> shiftedPos(t);
    Scalar rxyz = sqrt(dot(shiftedPos,shiftedPos));
    if (rxyz > 0.0)
        {
        inside = (((rxyz <= wall.r) && wall.inside) || ((rxyz > wall.r) && !(wall.inside))) ? true : false;
        t *= wall.r/rxyz;
        vec3<Scalar> dx = t - shiftedPos;
        return dx;
        }
    else
        {
        inside = (wall.inside) ? true : false;
        return vec3<Scalar>(0.0,0.0,0.0);
        }
    };

//! Point to wall vector for a cylinder wall geometry
DEVICE inline vec3<Scalar> vecPtToWall(const CylinderWall& wall, const vec3<Scalar>& position, bool& inside)
    {
    vec3<Scalar> t = position;
    t-=wall.origin;
    vec3<Scalar> shiftedPos = rotate(wall.quatAxisToZRot,t);
    shiftedPos.z = 0.0;
    Scalar rxy = sqrt(dot(shiftedPos,shiftedPos));
    if (rxy > 0.0)
        {
        inside = (((rxy <= wall.r) && wall.inside) || ((rxy > wall.r) && !(wall.inside))) ? true : false;
        t = (wall.r / rxy) * shiftedPos;
        vec3<Scalar> dx = t - shiftedPos;
        dx = rotate(conj(wall.quatAxisToZRot),dx);
        return dx;
        }
    else
        {
        inside = (wall.inside) ? true : false;
        return vec3<Scalar>(0.0,0.0,0.0);
        }
    };

//! Point to wall vector for a plane wall geometry
DEVICE inline vec3<Scalar> vecPtToWall(const PlaneWall& wall, const vec3<Scalar>& position, bool& inside)
    {
    vec3<Scalar> t = position;
    Scalar d = dot(wall.normal,t) - dot(wall.normal,wall.origin);
    inside = (((d >= 0.0) && wall.inside) || ((d < 0.0) && !(wall.inside))) ? true : false;
    vec3<Scalar> dx = -d * wall.normal;
    return dx;
    };

//! Distance of point to inside sphere wall geometry, not really distance, +- based on if it's inside or not
DEVICE inline Scalar distWall(const SphereWall& wall, const vec3<Scalar>& position)
    {
    vec3<Scalar> t = position;
    t-=wall.origin;
    vec3<Scalar> shiftedPos(t);
    Scalar rxyz2 = shiftedPos.x*shiftedPos.x + shiftedPos.y*shiftedPos.y + shiftedPos.z*shiftedPos.z;
    Scalar d = wall.r - sqrt(rxyz2);
    d = (wall.inside) ? d : -d;
    return d;
    };

//! Distance of point to inside cylinder wall geometry, not really distance, +- based on if it's inside or not
DEVICE inline Scalar distWall(const CylinderWall& wall, const vec3<Scalar>& position)
    {
    vec3<Scalar> t = position;
    t-=wall.origin;
    vec3<Scalar> shiftedPos=rotate(wall.quatAxisToZRot,t);
    Scalar rxy2= shiftedPos.x*shiftedPos.x + shiftedPos.y*shiftedPos.y;
    Scalar d = wall.r - sqrt(rxy2);
    d = (wall.inside) ? d : -d;
    return d;
    };

//! Distance of point to inside plane wall geometry, not really distance, +- based on if it's inside or not
DEVICE inline Scalar distWall(const PlaneWall& wall, const vec3<Scalar>& position)
    {
    vec3<Scalar> t = position;
    Scalar d = dot(wall.normal,t) - dot(wall.normal,wall.origin);
    d = (wall.inside) ? d : -d;
    return d;
    };
#endif
