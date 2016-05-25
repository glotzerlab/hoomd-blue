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

// WCAPotential.cc
// by Matthew Spellings <mspells@umich.edu>

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
