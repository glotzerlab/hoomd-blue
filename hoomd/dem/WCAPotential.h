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

/*! \file WCAPotential.h
  \brief Declares the pure virtual WCAPotential class
*/

#ifndef __WCAPOTENTIAL_H__
#define __WCAPOTENTIAL_H__


// need to declare these class methods with __device__ qualifiers when building in nvcc
// DEVICE is __host__ __device__ when included in nvcc and blank when included into the host compiler
#undef DEVICE
#ifdef NVCC
#define DEVICE __host__ __device__
#else
#define DEVICE
#endif

/*! Pluggable potential for a WCA interaction.

    The potential evaluated between contact points is:
    \f{eqnarray*}
    V_{\mathrm{WCA}}(r)  = & 4 \varepsilon \left[ \left( \frac{\sigma}{r} \right)^{12} -
                           \left( \frac{\sigma}{r} \right)^{6} \right] + \varepsilon & r < 2^{1/6}\sigma \\
                         = & 0 & r \ge 2^{1/6}\sigma \\
    \f}
*/
template<typename Real, typename Real4, typename FrictionModel>
class WCAPotential
{
public:
    // Ctor; set sigma of the interaction to be twice the given value
    // so the given value acts as a radius of interaction as might be
    // expected for a spheropolygon. Also takes a parameter eta for
    // the optional viscous term (it is ignored if this potential was
    // not built with viscosity enabled).
    WCAPotential(Real radius, const FrictionModel &frictionParams):
        m_sigma6(radius*radius*radius*radius*radius*radius*64.0),
        m_rcutsq(radius*radius*4.0*pow(2.0, 1./3.0)),
        m_frictionParams(frictionParams) {}

    // Energy scale sigma accessors
    DEVICE Real getSigma6() const {return m_sigma6;}
    void setRadius(Real radius)
    {
        m_sigma6 = radius*radius*radius*radius*radius*radius*64.0;
        m_rcutsq = radius*radius*4.0*pow(2.0, 1./3.0);
    }

    // Get this potential's cutoff radius
    Real getRcutSq() const {return m_rcutsq;}

    // Mutate this object by adjusting its lengthscale
    void scale(Real factor)
    {
        m_sigma6 *= factor*factor*factor*factor*factor*factor;
        m_rcutsq *= factor*factor;
    }

    /*! evaluate the potential between two points */
    template<typename Vec, typename Torque>
    DEVICE inline void evaluate(
        const Vec &rij, const Vec &r0, const Vec &rPrime,
        Real &potential, Vec &force_i, Torque &torque_i,
        Vec &force_j, Torque &torque_j, float modFactor=1) const;

    /*! test if particles are within cutoff of this potential*/
    DEVICE inline bool withinCutoff(const Real rsq, const Real r_cutsq) {return rsq<r_cutsq;}

    /*! Test if potential needs the diameter (It doesn't) */
    DEVICE static bool needsDiameter() {return false;}

    /*! Dummy function to set diameter*/
    DEVICE inline void setDiameter(const Real di,const Real dj) {}

    //! Swap the sense of particle i and j for the friction params
    DEVICE inline void swapij() {m_frictionParams.swapij();}

    /*! Test if the potential needs the velocity */
    DEVICE static bool needsVelocity() {return FrictionModel::needsVelocity();}

    /*! Set the velocity state */
    DEVICE inline void setVelocity(const vec3<Real> &v) {m_frictionParams.setVelocity(v);}

private:
    // Length scale sigma, raised to the sixth power for convenience
    Real m_sigma6;
    // Cutoff radius
    Real m_rcutsq;
    //! Parameters for friction (including relative velocity state, if necessary)
    FrictionModel m_frictionParams;
};

#include "WCAPotential.cc"

#endif
