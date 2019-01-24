// Copyright (c) 2009-2019 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.

// Maintainer: mspells

#ifndef __DEMEVALUATOR_CC__
#define __DEMEVALUATOR_CC__

#include "DEMEvaluator.h"
#include "WCAPotential.h"
#include <assert.h>

/*! Clip a value between 0 and 1 */
template<typename Real>
DEVICE inline Real clip(const Real &x)
    {
    return (x >= 0)*(x + (x > 1)*(1 - x));
    }

/*! Evaluate the force and torque contributions for particles i and j,
  with centers of mass separated by rij. r0 is a vertex in particle
  i, r1 and r2 are two vertices on an edge of particle j. r0, r1, and
  r2 should be relative to the centers of mass of the corresponding
  particle. The appropriate forces and torques for particles i and j
  will be added into {force, torque}_{i, j}.
*/
template<typename Real, typename Real4, typename Potential> template<typename Vec, typename Torque>
DEVICE inline void DEMEvaluator<Real, Real4, Potential>::comCOM(
    const Vec &rij, Real &potential, Vec &force_i,
    Torque &torque_i, Vec &force_j, Torque &torque_j) const
    {
    m_potential.evaluate(rij, Vec(), rij, potential, force_i, torque_i, force_j, torque_j);
    }

/*! Evaluate the force and torque contributions for particles i and j,
  with centers of mass separated by rij. r0 is a vertex in particle
  i, r1 and r2 are two vertices on an edge of particle j. r0, r1, and
  r2 should be relative to the centers of mass of the corresponding
  particle. The appropriate forces and torques for particles i and j
  will be added into {force, torque}_{i, j}.
*/
template<typename Real, typename Real4, typename Potential> template<typename Vec, typename Torque>
DEVICE inline void DEMEvaluator<Real, Real4, Potential>::vertexVertex(
    const Vec &rij, const Vec &r0, const Vec &r1, Real &potential, Vec &force_i,
    Torque &torque_i, Vec &force_j, Torque &torque_j, float modFactor) const
    {
    m_potential.evaluate(rij, r0, r1, potential, force_i, torque_i, force_j, torque_j, modFactor);
    }

template<typename Real, typename Real4, typename Potential> template<typename Vec, typename Torque>
DEVICE inline void DEMEvaluator<Real, Real4, Potential>::vertexEdge(
    const Vec &rij, const Vec &r0, const Vec &r1,
    const Vec &r2, Real &potential, Vec &force_i,
    Torque &torque_i, Vec &force_j, Torque &torque_j, float modFactor) const
    {
    // Work relative to particle i's COM
        const Vec r10(r0 - (r1 + rij));
        const Vec r12(r2 - r1);

        // Find the closest point to r0 in the line r2-r1
        Real lambda(dot(r10, r12)/dot(r12, r12));
        // Clip lambda between [0, 1] so we get a point on the line segment
        lambda = clip(lambda);

        // rPrime is the point in r2-r1 closest to r0
        const Vec rPrime(r1 + rij + lambda*r12);

        m_potential.evaluate(rij, r0, rPrime, potential, force_i, torque_i, force_j, torque_j, modFactor);
        }

template<typename Real, typename Real4, typename Potential>
DEVICE inline void DEMEvaluator<Real, Real4, Potential>::vertexFace(
    const vec3<Real> &rij, const vec3<Real> &r0, const quat<Real> quatj, const Real4 *verticesj,
    const unsigned int *realIndicesj, const unsigned int *facesj, const unsigned int vertex0Index, Real &potential,
    vec3<Real> &force_i, vec3<Real> &torque_i, vec3<Real> &force_j, vec3<Real> &torque_j) const
    {
    // distsq will be used to hold the square distance from r0 to the
    // face of interest; work relative to particle j's center of mass
    Real distsq(0);
    // r0 from particle j's frame of reference
    const vec3<Real> r0j(r0 - rij);
    // rPrime is the closest point to r0
    vec3<Real> rPrime;

    // vertex0 is the reference point in particle j to "fan out" from
    const vec3<Real> vertex0(rotate(quatj, vec3<Real>(verticesj[realIndicesj[vertex0Index]])));

    // r0r0: vector from vertex0 to r0 relative to particle j
    const vec3<Real> r0r0(r0j - vertex0);

    // check distance for first edge of polygon
    const vec3<Real> secondVertex(rotate(quatj, vec3<Real>(verticesj[realIndicesj[facesj[vertex0Index]]])));
    const vec3<Real> rsec(secondVertex - vertex0);
    Real lambda(dot(r0r0, rsec)/dot(rsec, rsec));
    lambda = clip(lambda);
    vec3<Real> closest(vertex0 + lambda*rsec);
    vec3<Real> closestr0(closest - r0j);
    Real closestDistsq(dot(closestr0, closestr0));
    distsq = closestDistsq;
    rPrime = closest;

    // indices of three points in triangle of interest: vertex0Index, i, facesj[i]
    // p01 and p02: two edge vectors of the triangle of interest
    vec3<Real> p1, p2(secondVertex), p01, p02(secondVertex - vertex0);

    // iterate through all fan triangles
    unsigned int i(facesj[vertex0Index]);
    for(; facesj[i] != vertex0Index; i = facesj[i])
        {
        Real alpha(0), beta(0);

        p1 = p2;
        p2 = rotate(quatj, vec3<Real>(verticesj[realIndicesj[facesj[i]]]));
        p01 = p02;
        p02 = p2 - vertex0;

        // pc: vector normal to the triangle of interest
        const vec3<Real> pc(cross(p01, p02));

        // distance matrix A is:
        // [ p01.x p02.x pc.x ]
        // [ p01.y p02.y pc.y ]
        // [ p01.z p02.z pc.z ]
        Real magA(p01.x*(p02.y*pc.z - pc.y*p02.z) - p02.x*(p01.y*pc.z - pc.y*p01.z) +
            pc.x*(p01.y*p02.z - p02.y*p01.z));

        alpha = ((p02.y*pc.z - pc.y*p02.z)*r0r0.x + (pc.x*p02.z - p02.x*pc.z)*r0r0.y +
            (p02.x*pc.y - pc.x*p02.y)*r0r0.z)/magA;
        beta = ((pc.y*p01.z - p01.y*pc.z)*r0r0.x + (p01.x*pc.z - pc.x*p01.z)*r0r0.y +
            (pc.x*p01.y - p01.x*pc.y)*r0r0.z)/magA;

        alpha = clip(alpha);
        beta = clip(beta);
        const Real k(alpha + beta);

        if(k > 1)
            {
            alpha /= k;
            beta /= k;
            }

        // check distance for exterior edge of polygon
        const vec3<Real> p12(p2 - p1);
        Real lambda(dot(r0j - p1, p12)/dot(p12, p12));
        lambda = clip(lambda);
        vec3<Real> closest(p1 + lambda*p12);
        vec3<Real> closestr0(closest - r0j);
        Real closestDistsq(dot(closestr0, closestr0));
        if(closestDistsq < distsq)
            {
            distsq = closestDistsq;
            rPrime = closest;
            }

        // closest: closest point in triangle (in particle j's reference frame)
        closest = vertex0 + alpha*p01 + beta*p02;
        // closestr0: vector between r0 and closest
        closestr0 = closest - r0j;
        closestDistsq = dot(closestr0, closestr0);
        if(closestDistsq < distsq)
            {
            distsq = closestDistsq;
            rPrime = closest;
            }

        // if(k > 1 or beta <= 0.)
        //     break;
        }

    // check distance for last edge of polygon
    const vec3<Real> rlast(p2 - vertex0);
    lambda = dot(r0r0, rlast)/dot(rlast, rlast);
    lambda = clip(lambda);
    closest = vertex0 + lambda*rlast;
    closestr0 = closest - r0j;
    closestDistsq = dot(closestr0, closestr0);
    if(closestDistsq < distsq)
        {
        distsq = closestDistsq;
        rPrime = closest;
        }

    if(distsq > 0)
        m_potential.evaluate(rij, r0, rPrime + rij, potential, force_i, torque_i,
            force_j, torque_j);
    }

// convenience function for edge/edge calculation
template<typename Real>
DEVICE inline Real detp(const vec3<Real> &m, const vec3<Real> &n, const vec3<Real> o, const vec3<Real> p)
    {
    return dot(m - n, o - p);
    }

template<typename Real, typename Real4, typename Potential>
DEVICE inline void DEMEvaluator<Real, Real4, Potential>::edgeEdge(
    const vec3<Real> &rij,
    const vec3<Real> &p00, const vec3<Real> &p01, const vec3<Real> &p10,
    const vec3<Real> &p11, Real &potential, vec3<Real> &force_i,
    vec3<Real> &torque_i, vec3<Real> &force_j, vec3<Real> &torque_j) const
    {
    // in the style of http://paulbourke.net/geometry/pointlineplane/
    Real denominator(detp(p01, p00, p01, p00)*detp(p11, p10, p11, p10) -
        detp(p11, p10, p01, p00)*detp(p11, p10, p01, p00));
    Real lambda0((detp(p00, p10, p11, p10)*detp(p11, p10, p01, p00) -
            detp(p00, p10, p01, p00)*detp(p11, p10, p11, p10))/denominator);
    Real lambda1((detp(p00, p10, p11, p10) +
            lambda0*detp(p11, p10, p01, p00))/detp(p11, p10, p11, p10));

    lambda0 = clip(lambda0);
    lambda1 = clip(lambda1);

    const vec3<Real> r0(p01 - p00);
    const Real r0sq(dot(r0, r0));
    const vec3<Real> r1(p11 - p10);
    const Real r1sq(dot(r1, r1));

    vec3<Real> closestI(p00 + lambda0*r0);
    vec3<Real> closestJ(p10 + lambda1*r1);
    vec3<Real> rContact(closestJ - closestI);
    Real closestDistsq(dot(rContact, rContact));

    Real lambda(clip(dot(p10 - p00, r0)/r0sq));
    vec3<Real> candidateI(p00 + lambda*r0);
    vec3<Real> candidateJ(p10);
    rContact = candidateJ - candidateI;
    Real distsq(dot(rContact, rContact));
    if(distsq < closestDistsq)
        {
        closestI = candidateI;
        closestJ = candidateJ;
        closestDistsq = distsq;
        }

    lambda = clip(dot(p11 - p00, r0)/r0sq);
    candidateI = p00 + lambda*r0;
    candidateJ = p11;
    rContact = candidateJ - candidateI;
    distsq = dot(rContact, rContact);
    if(distsq < closestDistsq)
        {
        closestI = candidateI;
        closestJ = candidateJ;
        closestDistsq = distsq;
        }

    lambda = clip(dot(p00 - p10, r1)/r1sq);
    candidateI = p00;
    candidateJ = p10 + lambda*r1;
    rContact = candidateJ - candidateI;
    distsq = dot(rContact, rContact);
    if(distsq < closestDistsq)
        {
        closestI = candidateI;
        closestJ = candidateJ;
        closestDistsq = distsq;
        }

    lambda = clip(dot(p01 - p10, r1)/r1sq);
    candidateI = p01;
    candidateJ = p10 + lambda*r1;
    rContact = candidateJ - candidateI;
    distsq = dot(rContact, rContact);
    if(distsq < closestDistsq)
        {
        closestI = candidateI;
        closestJ = candidateJ;
        closestDistsq = distsq;
        }

    if(fabs(1 - dot(r0, r1)*dot(r0, r1)/r0sq/r1sq) < 1e-6)
        {
        const Real lambda00(clip(dot(p10 - p00, r0)/r0sq));
        const Real lambda01(clip(dot(p11 - p00, r0)/r0sq));
        const Real lambda10(clip(dot(p00 - p10, r1)/r1sq));
        const Real lambda11(clip(dot(p01 - p10, r1)/r1sq));

        lambda0 = Real(.5)*(lambda00 + lambda01);
        lambda1 = Real(.5)*(lambda10 + lambda11);

        closestI = p00 + lambda0*r0;
        closestJ = p10 + lambda1*r1;
        }

    m_potential.evaluate(rij, closestI, closestJ, potential, force_i, torque_i,
        force_j, torque_j);
    }

#endif
