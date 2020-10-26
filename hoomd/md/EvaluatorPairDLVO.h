// Copyright (c) 2009-2019 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.

// Maintainer: jglaser

#ifndef __PAIR_EVALUATOR_DLVO_H__
#define __PAIR_EVALUATOR_DLVO_H__

#ifndef NVCC
#include <string>
#endif

#include "hoomd/HOOMDMath.h"

/*! \file EvaluatorPairDLVO.h
    \brief Defines the pair evaluator class for DLVO potential
*/

// need to declare these class methods with __device__ qualifiers when building in nvcc
//! DEVICE is __host__ __device__ when included in nvcc and blank when included into the host compiler
#ifdef NVCC
#define DEVICE __device__
#else
#define DEVICE
#endif

// call different optimized exp functions on the host / device
//! fast::exp is expf when included in nvcc and exp when included into the host compiler
#ifdef NVCC
#define LOG logf
#else
#define LOG log
#endif

//! Class for evaluating the DLVO pair potential
/*! <b>General Overview</b>

    <b>DLVO specifics</b>

    EvaluatorPairDLVO evaluates the function:
    \f{eqnarray*}
    V_{\mathrm{DLVO}}(r)  = & - \frac{A}{6} \left[ \frac{2a_1a_2}{r^2 - (a_1+a_2)^2} + \frac{2a_1a_2}{r^2 - (a_1-a_2)^2}
                            + \log \left( \frac{r^2 - (a_1+a_2)^2}{r^2 - (a_1+a_2)^2} \right) \right] + \frac{r_1r_2}{r_1+r_2} Z e^{-\kappa(r - (a_1+a_2))} & r < (r_{\mathrm{cut}} + \Delta) \\
                         = & 0 & r \ge (r_{\mathrm{cut}} + \Delta) \\
    \f}
    where \f $a_i \f$ is the radius of particle \f$ i \f$; \f$ \Delta = (d_i + d_j)/2  \f$ and \f$ d_i \f$ is the diameter of particle \f$ i \f$.

    See Israelachvili 2011, pp. 317.

    The DLVO potential does not need charge, but does need diameter. Three parameters are specified and stored in a
    Scalar4. \a kappa is placed in \a params.x and \a Z is in \a params.y and \a A in \a params.z

    Due to the way that DLVO modifies the cutoff condition, it will not function properly with the xplor shifting mode.
*/
class EvaluatorPairDLVO
    {
    public:
        //! Define the parameter type used by this pair potential evaluator
        //May need to be changed to Scalar3 later
		typedef Scalar3 param_type;

        //! Constructs the pair potential evaluator
        /*! \param _rsq Squared distance between the particles
            \param _rcutsq Squared distance at which the potential goes to 0
            \param _params Per type pair parameters of this potential
        */
        DEVICE EvaluatorPairDLVO(Scalar _rsq, Scalar _rcutsq, const param_type& _params)
            : rsq(_rsq), rcutsq(_rcutsq), kappa(_params.x), Z(_params.y), A(_params.z)
            {
            }

        //! DLVO uses diameter
        DEVICE static bool needsDiameter() { return true; }

        //! Accept the optional diameter values
        /*! \param di Diameter of particle i
            \param dj Diameter of particle j
        */
        DEVICE void setDiameter(Scalar di, Scalar dj)
            {
            radsum = (di + dj) / Scalar(2.0);
            radsub = (di - dj) / Scalar(2.0);
            radprod = (di * dj) / Scalar(4.0);
            radsumsq = (di*di + dj*dj) / Scalar(4.0);
            radsubsq = (di*di - dj*dj) / Scalar(4.0);
            delta = radsum - Scalar(1.0);
            }

        //! DLVO doesn't use charge
        DEVICE static bool needsCharge() { return false; }
        //! Accept the optional diameter values
        /*! \param qi Charge of particle i
            \param qj Charge of particle j
        */
        DEVICE void setCharge(Scalar qi, Scalar qj) { }

        //! Evaluate the force and energy
        /*! \param force_divr Output parameter to write the computed force divided by r.
            \param pair_eng Output parameter to write the computed pair energy
            \param energy_shift If true, the potential must be shifted so that V(r) is continuous at the cutoff
            \note There is no need to check if rsq < rcutsq in this method. Cutoff tests are performed
                  in PotentialPair.

            \return True if they are evaluated or false if they are not because we are beyond the cutoff
        */
        DEVICE bool evalForceAndEnergy(Scalar& force_divr, Scalar& pair_eng, bool energy_shift)
            {
            // precompute some quantities
            Scalar rinv = fast::rsqrt(rsq);
            Scalar r = Scalar(1.0) / rinv;
            Scalar rcutinv = fast::rsqrt(rcutsq);
            Scalar rcut = Scalar(1.0) / rcutinv;

            // compute the force divided by r in force_divr
            if (r < (rcut + delta) && kappa != 0)
                {
                Scalar rmds = r - radsum;
                Scalar rmdsqs = r*r - radsum*radsum;
                Scalar rmdsqm = r*r - radsub*radsub;
                Scalar radsuminv = Scalar(1.0) / radsum;
                Scalar rmdsqsinv = Scalar(1.0) / rmdsqs;
                Scalar rmdsqminv = Scalar(1.0) / rmdsqm;
                Scalar exp_val = fast::exp(-kappa * rmds);
                Scalar forcerep_divr = kappa * radprod * radsuminv * Z * exp_val/r;
                Scalar fatrterm1 = r*r*r*r + radsubsq*radsubsq - Scalar(2.0)*r*r*radsumsq;
                Scalar fatrterm1inv = Scalar(1.0) / fatrterm1 * Scalar(1.0) / fatrterm1;
                Scalar forceatr_divr = -Scalar(32.0) * A / Scalar(3.0) * radprod * radprod * radprod * fatrterm1inv;
                force_divr = forcerep_divr + forceatr_divr;

                Scalar engt1 = radprod * rmdsqsinv * A / Scalar(3.0);
                Scalar engt2 = radprod * rmdsqminv * A / Scalar(3.0);
                Scalar engt3 = LOG(rmdsqs * rmdsqminv) * A / Scalar(6.0);
                pair_eng = r * forcerep_divr / kappa - engt1 - engt2 - engt3;
                if (energy_shift)
                    {
                    Scalar rcutt = rcut + delta;
                    Scalar rmdscut = rcutt - radsum;
                    Scalar rmdsqscut = rcutt * rcutt - radsum*radsum;
                    Scalar rmdsqmcut = rcutt * rcutt - radsub*radsub;
                    Scalar rmdsqsinvcut = Scalar(1.0) / rmdsqscut;
                    Scalar rmdsqminvcut = Scalar(1.0) / rmdsqmcut;

                    Scalar engt1cut = radprod * rmdsqsinvcut * A / Scalar(3.0);
                    Scalar engt2cut = radprod * rmdsqminvcut * A / Scalar(3.0);
                    Scalar engt3cut = LOG(rmdsqscut * rmdsqminvcut) * A / Scalar(6.0);
                    Scalar exp_valcut = fast::exp(-kappa * rmdscut);
                    Scalar forcerepcut_divr = kappa * radprod * radsuminv * Z * exp_valcut/rcutt;
                    pair_eng -= rcutt*forcerepcut_divr / kappa - engt1cut - engt2cut - engt3cut;
                    }
                return true;
                }
            else
                return false;
            }

        #ifndef NVCC
        //! Get the name of this potential
        /*! \returns The potential name. Must be short and all lowercase, as this is the name energies will be logged as
            via analyze.log.
        */
        static std::string getName()
            {
            return std::string("dlvo");
            }

        std::string getShapeSpec() const
            {
            throw std::runtime_error("Shape definition not supported for this pair potential.");
            }
        #endif

    protected:
        Scalar rsq;     //!< Stored rsq from the constructor
        Scalar rcutsq;  //!< Stored rcutsq from the constructor
        Scalar kappa;     //!< kappa parameter extracted from the params passed to the constructor
        Scalar Z;     //!< Z parameter extracted from the params passed to the constructor
        Scalar A;       //!< A parameter extracted from the params passed to the constructor
        Scalar radsum;  //!< radsum parameter extracted from the call to setDiameter
        Scalar radsub;  //!< radsub parameter extracted from the call to setDiameter
        Scalar radprod; //!< radprod parameter extracted from the call to setDiameter
        Scalar radsumsq;    //!< radsumsq parameter extracted from the call to setDiameter
        Scalar radsubsq;    //!< radsubsq parameter extracted from the call to setDiameter
        Scalar delta;       //!< Diameter sum minus one
    };


#endif // __PAIR_EVALUATOR_DLVO_H__
