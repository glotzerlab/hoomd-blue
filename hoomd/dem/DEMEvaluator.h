/*
  Highly Optimized Object-oriented Many-particle Dynamics -- Blue Edition
  (HOOMD-blue) Open Source Software License Copyright 2008-2011 Ames Laboratory
  Iowa State University and The Regents of the University of Michigan All rights
  reserved.

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

// Maintainer: mspells

/*! \file DEMEvaluator.h
  \brief Declares the pure virtual DEMEvaluator class
*/

#ifndef __DEMEVALUATOR_H__
#define __DEMEVALUATOR_H__

#include "VectorMath.h"

// need to declare these class methods with __device__ qualifiers when building in nvcc
// DEVICE is __host__ __device__ when included in nvcc and blank when included into the host compiler
#undef DEVICE
#ifdef NVCC
#define DEVICE __host__ __device__
#else
#define DEVICE
#endif

/*! Wrapper class to evaluate potentials between features of shapes */
template<typename Real, typename Real4, typename Potential>
class DEMEvaluator
{
public:
    DEMEvaluator(const Potential &potential):
        m_potential(potential)
    {
        // ratio of vertex radius to edge rounding radius
        // Real edge_vertex_ratio(0.5);

        // Give the edge/edge interaction a shorter lengthscale
        // m_edgePotential.scale(edge_vertex_ratio);

        // surfaces are thinner by the edge/vertex factor
        // m_potential.scale(Real(0.5)+Real(0.5)*edge_vertex_ratio);
    }

    Real getRcutSq() const {return m_potential.getRcutSq();}

    /*! Evaluate the force and torque contributions for particles i
      and j, with centers of mass separated by rij. The appropriate
      forces and torques for particles i and j will be added to
      {force, torque}_{i, j}. Should only be used with one-vertex shapes
    */
    template<typename Vec, typename Torque>
    DEVICE inline void comCOM(
        const Vec &rij, Real &potential, Vec &force_i,
        Torque &torque_i, Vec &force_j, Torque &torque_j) const;

    /*! Evaluate the force and torque contributions for particles i
      and j, with centers of mass separated by rij. The appropriate
      forces and torques for particles i and j will be added to
      {force, torque}_{i, j}.
    */
    template<typename Vec, typename Torque>
    DEVICE inline void vertexVertex(
        const Vec &rij, const Vec &r0, const Vec &r1,
        Real &potential, Vec &force_i, Torque &torque_i,
        Vec &force_j, Torque &torque_j, float modFactor=1) const;

    /*! Evaluate the force and torque contributions for particles i
      and j, with centers of mass separated by rij. r0 is a vertex
      in particle i, r1 and r2 are two vertices on an edge of particle
      j. r0, r1, and r2 should be relative to the centers of mass of
      the corresponding particle. The appropriate forces and torques
      for particles i and j will be added to {force, torque}_{i, j}.
    */
    template<typename Vec, typename Torque>
    DEVICE inline void vertexEdge(
        const Vec &rij, const Vec &r0, const Vec &r1,
        const Vec &r2, Real &potential, Vec &force_i,
        Torque &torque_i, Vec &force_j, Torque &torque_j, float modFactor=1) const;

    /*! Evaluate the force and torque contributions for particles i
      and j, with centers of mass separated by rij. r0 is a vertex
      in particle i, verticesj is a list of the vertices of j, and facesj
      is a list of the indices of the polygon(s) that make up each face,
      and vertex0 is the first vertex in that particular face.
      r0 and calculations should be relative to the centers of mass of
      the corresponding particle. The appropriate forces and torques
      for particles i and j will be added to {force, torque}_{i, j}.
    */
    DEVICE inline void vertexFace(
        const vec3<Real> &rij, const vec3<Real> &r0, const quat<Real> quatj, const Real4 *verticesj,
        const unsigned int *realIndicesj, const unsigned int *facesj, const unsigned int vertex0, Real &potential,
        vec3<Real> &force_i, vec3<Real> &torque_i, vec3<Real> &force_j, vec3<Real> &torque_j) const;

    /*! Evaluate the force and torque contributions for particles i
        and j between two edges, specified by points r00 (first vertex
        of the edge in particle i), r01 (second vertex in the edge in
        particle i), r10 (first vertex of the edge in particle j), and
        r11 (second vertex of the edge in particle j), all specified
        with respect to particle i's center of mass.
    */
    DEVICE inline void edgeEdge(
        const vec3<Real> &rij, const vec3<Real> &p00, const vec3<Real> &p01, const vec3<Real> &p10,
        const vec3<Real> &p11, Real &potential, vec3<Real> &force_i,
        vec3<Real> &torque_i, vec3<Real> &force_j, vec3<Real> &torque_j) const;

    /*! Test if we need to evalute this potential evaluator
    */
    DEVICE inline bool withinCutoff(const Real rsq, const Real r_cut_sq)
    {return m_potential.withinCutoff(rsq,r_cut_sq);}

    DEVICE static bool needsDiameter() {return Potential::needsDiameter();}

    DEVICE inline void setDiameter(const Real di,const Real dj)
    {m_potential.setDiameter(di, dj);}

    DEVICE inline void swapij() {m_potential.swapij();}

    DEVICE static bool needsVelocity() {return Potential::needsVelocity();}
    DEVICE inline void setVelocity(const vec3<Real> &v)
    {
        m_potential.setVelocity(v);
    }

private:
    //! Vertex/face potential parameters
    Potential m_potential;
};

#include "DEMEvaluator.cc"

#endif
