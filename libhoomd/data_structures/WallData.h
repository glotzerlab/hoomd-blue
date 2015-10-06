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


struct SphereWall
    {
    SphereWall(Scalar rad = 0.0, Scalar3 orig = make_scalar3(0.0,0.0,0.0), bool ins = true) : r(rad), inside(ins), origin(vec3<Scalar>(orig)) {}
    Scalar          r;
    bool            inside;
    vec3<Scalar>    origin;
    };

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
    vec3<Scalar>    axis;
    quat<Scalar>    q_reorientation;
    };

struct PlaneWall
    {
    PlaneWall(Scalar3 orig = make_scalar3(0.0,0.0,0.0), Scalar3 norm = make_scalar3(0.0,0.0,1.0)) : normal(vec3<Scalar>(norm)), origin(vec3<Scalar>(orig)), inside(true)
        {
        vec3<Scalar> nvec;
        nvec = normal;
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
#endif
