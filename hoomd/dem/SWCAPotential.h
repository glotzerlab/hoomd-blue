// Copyright (c) 2009-2019 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.

// Maintainer: mspells

/*! \file SWCAPotential.h
  \brief Declares the pure virtual SWCAPotential class
*/

#ifndef __SWCAPOTENTIAL_H__
#define __SWCAPOTENTIAL_H__


// need to declare these class methods with __device__ qualifiers when building in nvcc
// DEVICE is __host__ __device__ when included in nvcc and blank when included into the host compiler
#undef DEVICE
#ifdef NVCC
#define DEVICE __host__ __device__
#else
#define DEVICE
#endif

/*! Pluggable potential for a WCA interaction, shifted by the diameter of each particle.

  The potential evaluated between contact points is:
  \f{eqnarray*}
  V_{\mathrm{SWCA}}(r)  = & 4 \varepsilon \left[ \left( \frac{\sigma}{r - \Delta} \right)^{12} -
  \left( \frac{\sigma}{r - \Delta} \right)^{6} \right] + \varepsilon & r < 2^{1/6}\sigma + \Delta \\
  = & 0 & r \ge 2^{1/6}\sigma + \Delta \\
  \f}
  where \f$ \Delta = (d_i + d_j)/2 - 1 \f$ and \f$ d_i \f$ is the assigned diameter of particle \f$ i \f$.
*/
template<typename Real, typename Real4, typename FrictionModel>
class SWCAPotential
    {
    public:
        // Ctor; set sigma of the interaction to be twice the given value
        // so the given value acts as a radius of interaction as might be
        // expected for a spheropolygon
        SWCAPotential(Real radius, const FrictionModel &frictionParams):
            m_radius(radius),
            m_sigma6(radius*radius*radius*radius*radius*radius*64),
            m_rcutsq(radius*radius*4*pow(2.0, 1./3)),
            m_frictionParams(frictionParams) {}

        // Get this potential's rounding radius
        Real getRadius() const {return m_radius;}

        // Length scale sigma accessors
        Real getSigma6() const {return m_sigma6;}
        void setRadius(Real radius)
            {
            m_radius = radius;
            m_sigma6 = radius*radius*radius*radius*radius*radius*64;
            m_rcutsq = radius*radius*4*pow(2.0, 1./3.0);
            }

        /*! evaluate the potential between two points */
        template<typename Vec, typename Torque>
        DEVICE inline void evaluate(
            const Vec &rij, const Vec &r0, const Vec &rPrime,
            Real &potential, Vec &force_i, Torque &torque_i,
            Vec &force_j, Torque &torque_j, float modFactor=1) const;

        /*! Test to see if we need to evaluate this potential
         */
        DEVICE inline bool withinCutoff(Real rsq, Real r_cut_sq)
            {
            float rmd = sqrt(rsq) - m_delta;
            return rmd*rmd < r_cut_sq;
            }

        //! Test if potential needs the diameter
        DEVICE static bool needsDiameter() {return true;}
        DEVICE void setDiameter(Real di, Real dj) {m_delta = 0.5*(di+dj) - 1;}

        //! Swap the sense of particle i and j for the friction params
        DEVICE inline void swapij() {m_frictionParams.swapij();}
        DEVICE static bool needsVelocity() {return FrictionModel::needsVelocity();}
        DEVICE inline void setVelocity(const vec3<Real> &v) {m_frictionParams.setVelocity(v);}

    private:
        // Rounding radius
        Real m_radius;
        // Length scale sigma, raised to the sixth power for convenience
        Real m_sigma6;
        // Cutoff radius
        Real m_rcutsq;
        // Cutoff shift parameter
        Real m_delta;
        //! Parameters for friction (including relative velocity state, if necessary)
        FrictionModel m_frictionParams;
    };

#include "SWCAPotential.cc"

#endif
