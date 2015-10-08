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
    \brief Contains declarations for all types (currently Sphere, Cylinder, and
    Plane) of WallData and associated utilities.
 */
#ifndef WALL_DATA_H
#define WALL_DATA_H


#include "HOOMDMath.h"
#include "VectorMath.h"
#include <cstdlib>
#include <vector>

//! SphereWall Constructor
struct SphereWall
    {
    SphereWall(Scalar rad = 0.0, Scalar3 orig = make_scalar3(0.0,0.0,0.0), bool ins = true) : r(rad), inside(ins), origin(vec3<Scalar>(orig)) {}
    Scalar          r;
    bool            inside;
    vec3<Scalar>    origin;
    };

//! CylinderWall Constructor
struct CylinderWall
    {
    CylinderWall(Scalar rad = 0.0, Scalar3 orig = make_scalar3(0.0,0.0,0.0), Scalar3 zorient = make_scalar3(0.0,0.0,1.0), bool ins=true) : r(rad), inside(ins),  origin(vec3<Scalar>(orig)), axis(vec3<Scalar>(zorient))
        {
        vec3<Scalar> zvec;
        zvec=axis;
        vec3<Scalar> znorm(0,0,1);

        //method source: http://lolengine.net/blog/2014/02/24/quaternion-from-two-vectors-final
        Scalar norm_uv=sqrt(dot(znorm,znorm) * dot(zvec,zvec));
        Scalar real_part=norm_uv + dot(znorm,zvec);
        vec3<Scalar> w;

        if (real_part < Scalar(1.0e-6) * norm_uv)
            {
                real_part=Scalar(0.0);
                w=fabs(znorm.x) > fabs(znorm.z) ? vec3<Scalar>(-znorm.y, znorm.x, 0.0) : vec3<Scalar>(0.0, -znorm.z, znorm.y);
            }
        else
            {
                w=cross(znorm,zvec);
                real_part=Scalar(real_part);
            }
            quatAxisToZRot=quat<Scalar>(real_part,w);
            Scalar norm=fast::rsqrt(norm2(quatAxisToZRot));
            quatAxisToZRot=norm*quatAxisToZRot;
        }
    Scalar          r;
    bool            inside;
    vec3<Scalar>    origin;
    vec3<Scalar>    axis;
    quat<Scalar>    quatAxisToZRot;
    };

//! PlaneWall Constructor
struct PlaneWall
    {
    PlaneWall(Scalar3 orig = make_scalar3(0.0,0.0,0.0), Scalar3 norm = make_scalar3(0.0,0.0,1.0)) : normal(vec3<Scalar>(norm)), origin(vec3<Scalar>(orig))
        {
        vec3<Scalar> nvec;
        nvec = normal;
        Scalar n_length;
        n_length=fast::rsqrt(nvec.x*nvec.x + nvec.y*nvec.y + nvec.z*nvec.z);
        normal=nvec*n_length;
        }
    vec3<Scalar>    normal;
    vec3<Scalar>    origin;
    };

DEVICE inline vec3<Scalar> vecInsPtToWall(const SphereWall& wall, const vec3<Scalar>& position)
    {
    vec3<Scalar> t = position;
    t-=wall.origin;
    vec3<Scalar> shifted_pos(t);
    Scalar rxyz = sqrt(dot(shifted_pos,shifted_pos));
    if (((rxyz < wall.r) && wall.inside) || ((rxyz > wall.r) && !(wall.inside)))
        {
        t *= wall.r/rxyz;
        vec3<Scalar> dx = t - shifted_pos;
        return dx;
        }
    else
        {
        return vec3<Scalar>(0.0,0.0,0.0);
        }
    };

DEVICE inline vec3<Scalar> vecInsPtToWall(const CylinderWall& wall, const vec3<Scalar>& position)
    {
    vec3<Scalar> t = position;
    t-=wall.origin;
    vec3<Scalar> shifted_pos = rotate(wall.quatAxisToZRot,t);
    shifted_pos.z = 0.0;
    Scalar rxy = sqrt(dot(shifted_pos,shifted_pos));
    if (((rxy < wall.r) && wall.inside) || ((rxy > wall.r) && !(wall.inside)))
        {
        t = (wall.r / rxy) * shifted_pos;
        vec3<Scalar> dx = t - shifted_pos;
        dx = rotate(conj(wall.quatAxisToZRot),dx);
        return dx;
        }
    else
        {
        return vec3<Scalar>(0.0,0.0,0.0);
        }
    };

DEVICE inline vec3<Scalar> vecInsPtToWall(const PlaneWall& wall, const vec3<Scalar>& position)
    {
    vec3<Scalar> t = position;
    Scalar wall_dist = dot(wall.normal,wall.origin) - dot(wall.normal,t);
    if (wall_dist > 0.0)
        {
        vec3<Scalar> dx = wall_dist * wall.normal;
        return dx;
        }
    else
        {
        return vec3<Scalar>(0.0,0.0,0.0);
        }
    };

DEVICE inline bool insideWall(const SphereWall& wall, const vec3<Scalar>& position)
    {
    t-=wall.origin;
    vec3<Scalar> shifted_pos(t);
    Scalar rxyz_sq = shifted_pos.x*shifted_pos.x + shifted_pos.y*shifted_pos.y + shifted_pos.z*shifted_pos.z;
    Scalar d = wall.r - sqrt(rxyz_sq);
    bool inside = (d > 0.0) ? true : false;
    return inside;
    };

DEVICE inline bool insideWall(const CylinderWall& wall, const vec3<Scalar>& position)
    {
    vec3<Scalar> t = position;
    t-=wall.origin;
    vec3<Scalar> shifted_pos=rotate(wall.quatAxisToZRot,t);
    Scalar rxy_sq= shifted_pos.x*shifted_pos.x + shifted_pos.y*shifted_pos.y;
    Scalar d = wall.r - sqrt(rxy_sq);
    bool inside = (d > 0.0) ? true : false;
    return inside;
    };

DEVICE inline bool insideWall(const PlaneWall& wall, const vec3<Scalar>& position)
    {
    vec3<Scalar> t = position;
    Scalar d = dot(wall.normal,wall.origin) - dot(wall.normal,t);
    bool inside = (d > 0.0) ? true : false;
    return inside;
    };

#endif
