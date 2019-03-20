// Copyright (c) 2009-2019 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.

// Maintainer: mspells

#ifndef __WCAPOTENTIAL_CC__
#define __WCAPOTENTIAL_CC__

#include "WCAPotential.h"
#include "DEMEvaluator.h"

/*! Evaluate the potential between two points

  Parameters:
  - rij: vector from particle i's COM to particle j's
  - r0: vector from particle i's COM to the interaction point on particle i
  - rPrime: vector from particle i's COM to the interaction point on particle j
*/
template<typename Real, typename Real4, typename FrictionModel> template<typename Vec, typename Torque>
DEVICE inline void WCAPotential<Real, Real4, FrictionModel>::evaluate(
    const Vec &rij, const Vec &r0, const Vec &rPrime, Real &potential, Vec &force_i,
    Torque &torque_i, Vec &force_j, Torque &torque_j, float modFactor) const
    {
    // r0Prime is the vector from the interaction point on particle i
    // to that on particle j
    const Vec r0Prime(rPrime - r0);

    // Use distance to calculate WCA force
    const Real rsq(dot(r0Prime, r0Prime));

    if(rsq <= m_rcutsq)
        {
        const Real rsqInv(Real(1.0)/rsq);
        const Real rsq3Inv(rsqInv*rsqInv*rsqInv);
        // Force between r0 and rPrime is prefactor*r0Prime
        const Real prefactor(modFactor*24*m_sigma6*rsq3Inv*rsqInv*(1 - 2*m_sigma6*rsq3Inv));
        const Vec conservativeForce(prefactor*r0Prime);

        const Vec force(m_frictionParams.modifiedForce(r0Prime, conservativeForce));

        potential += modFactor*(4*m_sigma6*rsq3Inv*(m_sigma6*rsq3Inv - 1) + 1);

        // rjPrime is rPrime relative to particle j's COM
        const Vec rjPrime(rPrime - rij);

        // Add forces and torques
        force_i += force;
        torque_i += cross(r0, force);
        force_j -= force;
        torque_j += cross(rjPrime, -force);
        }
    }

#endif
