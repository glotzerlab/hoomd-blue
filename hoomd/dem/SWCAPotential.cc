// Copyright (c) 2009-2019 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.

// Maintainer: mspells

#ifndef __SWCAPOTENTIAL_CC__
#define __SWCAPOTENTIAL_CC__

#include "SWCAPotential.h"
#include "DEMEvaluator.h"
#include <algorithm>

using namespace std;

/*! evaluate the potential between two points */
// rij: vector from particle i's COM to particle j's
// r0: vector from particle i's COM to the interaction point on particle i
// rPrime: vector from particle i's COM to the interaction point on particle j
template<typename Real, typename Real4, typename FrictionModel> template<typename Vec, typename Torque>
DEVICE inline void SWCAPotential<Real, Real4, FrictionModel>::evaluate(
    const Vec &rij, const Vec &r0, const Vec &rPrime, Real &potential, Vec &force_i,
    Torque &torque_i, Vec &force_j, Torque &torque_j, float modFactor) const
    {
    const Vec r0Prime(rPrime - r0);

    // Use rmd to calculate SWCA force
    const Real magr(sqrt(dot(r0Prime, r0Prime)));
    const Real rmd(magr - m_delta);

    if(rmd*rmd <= m_rcutsq)
        {
        const Real rmdsqInv(Real(1.0)/(rmd*rmd));
        const Real rmdsq3Inv(rmdsqInv*rmdsqInv*rmdsqInv);
        // Force between r0 and rPrime is prefactor*r0Prime
        const Real prefactor(modFactor*24*m_sigma6*rmdsq3Inv*rmdsqInv*(1 - 2*m_sigma6*rmdsq3Inv));
        const Vec conservativeForce(prefactor*r0Prime);

        const Vec force(m_frictionParams.modifiedForce(r0Prime, conservativeForce));

        potential += modFactor*(4*m_sigma6*rmdsq3Inv*(m_sigma6*rmdsq3Inv - 1) + 1);
        //potential = m_delta;

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
