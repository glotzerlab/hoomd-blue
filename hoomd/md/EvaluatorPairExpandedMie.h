// Copyright (c) 2009-2021 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.


// Maintainer: joaander

#ifndef __PAIR_EVALUATOR_ExpandedMie_H__
#define __PAIR_EVALUATOR_ExapndedMie_H__

#ifndef __HIPCC__
#include <string>
#endif

#include "hoomd/HOOMDMath.h"
#include "EvaluatorPairLJ.h"

/*! \file EvaluatorPairExapndedMie.h
    \brief Defines the pair evaluator class for Expanded Mie potentials
*/

// need to declare these class methods with __device__ qualifiers when building in nvcc
// DEVICE is __host__ __device__ when included in nvcc and blank when included into the host compiler
#ifdef __HIPCC__
#define DEVICE __device__
#else
#define DEVICE
#endif

//! Class for evaluating the Expanded Mie pair potential
/*! <b>General Overview</b>

    See EvaluatorPairSLJ and EvaluatorPairMie

    <b>ExpandedMie specifics</b>

    EvaluatorPairExapndedMie evaluates the function:
    \f{eqnarray*}
    V_{\mathrm{ExpMie}}(r)  = \left( \frac{n}{n-m} \right) {\left( \frac{n}{m} \right)}^{\frac{m}{n-m}} \varepsilon \left[ \left( \frac{\sigma}{r-\Delta} \right)^{n} -
                \left( \frac{\sigma}{r-\Delta} \right)^{m} \right] r \l (r_{\mathrm{cut}} + \Delta)\\
                = & 0 & r \ge (r_{\mathrm{cut}} + \Delta) \\
    \f}
    where  \f$ \Delta is specified by the user, conventionally calcualted as, \f$ \Delta = (d_i + d_j)/2 - \sigma \f$ and \f$ d_i \f$ is the diameter of particle \f$ i \f$.

    The ExpandedMie potential needs neither charge nor diameter. Two parameters are specified and stored in a
    Scalar5. \a rep \a att \a npow \a mpow \a delta are stored in \a params.x \a params.y \a params.z and \a params.w \a params.v respectively.


    These are related to the standard lj parameters sigma and epsilon and the variable exponents n and m by:
    - \a rep = epsilon * pow(sigma,n) * (n/(n-m)) * power(n/m,m/(n-m))
    - \a att = epsilon * pow(sigma,m) * (n/(n-m)) * power(n/m,m/(n-m))
    - \a npow = n
    - \a mpow = m
    - \a delta = delta

    Due to the way that ExpandedMie modifies the cutoff condition, it will not function properly with the xplor shifting mode.
*/
class EvaluatorPairExpandedMie
    {
    public:
        //! Define the parameter type used by this pair potential evaluator
        struct param_type
            {
            Scalar rep;
            Scalar att;
            Scalar npow;
            Scalar mpow;
            Scalar delta;

            #ifndef ENABLE_HIP
            //! set CUDA memory hints
            void set_memory_hint() const {}
            #endif

            #ifndef __HIPCC__
            param_type() : rep(0), att(0), npow(0), mpow(0), delta(0) {}

            param_type(pybind11::dict v)
                {
                npow = v["n"].cast<Scalar>();
                mpow = v["m"].cast<Scalar>();

                auto sigma(v["sigma"].cast<Scalar>());
                auto epsilon(v["epsilon"].cast<Scalar>());

                Scalar outFront = (npow/(npow-mpow)) * fast::pow(npow/mpow, mpow/(npow-mpow));
                rep = outFront * epsilon * fast::pow(sigma, npow);
                att = outFront * epsilon * fast::pow(sigma, mpow);

                delta = v["delta"].cast<Scalar>();
                }

            pybind11::dict asDict()
                {
                pybind11::dict v;
                v["n"] = npow;
                v["m"] = mpow;

                Scalar sigma = fast::pow(rep / att, 1 / (npow - mpow));
                Scalar epsilon = rep / fast::pow(sigma, npow) * (npow - mpow) / npow * fast::pow(npow / mpow, mpow / (mpow - npow));

                v["epsilon"] = epsilon;
                v["sigma"] = sigma;

                v["delta"] = delta;
                return v;
                }
            #endif
            }
            __attribute__((aligned(16)));

        //! Constructs the pair potential evaluator
        /*! \param _rsq Squared distance between the particles
            \param _rcutsq Squared distance at which the potential goes to 0
            \param _n First, larger exponent that captures hard-core repulsion
            \param _m Second, smaller exponent that captures attraction
            \param _params Per type pair parameters of this potential
            \param _delta Horizontal shift in r
        */
        DEVICE EvaluatorPairExpandedMie(Scalar _rsq, Scalar _rcutsq, const param_type& _params)
            : rsq(_rsq), rcutsq(_rcutsq), rep(_params.rep), att(_params.att), npow(_params.npow), mpow(_params.mpow), delta(_params.delta)
            {
            }

        //! ExpandedMie doesn't use diameter
        DEVICE static bool needsDiameter() { return false; }
        //! Accept the optional diameter values
        /*! \param di Diameter of particle i
            \param dj Diameter of particle j
        */
        DEVICE void setDiameter(Scalar di, Scalar dj) { }

        //! ExpandedMie doesn't use charge
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
            Scalar r = fast::sqrt(rsq);
            Scalar rinv = fast::rsqrt(rsq);

            // compute the force divided by r in force_divr
            if (rsq < rcutsq && rep != 0)
                {
                Scalar rmd = r - delta;
                Scalar rmdinv = Scalar(1.0) / rmd;
                Scalar rmd2inv = rmdinv * rmdinv;
                Scalar rmdninv = fast::pow(rmd2inv,npow/Scalar(2.0));
                Scalar rmdminv = fast::pow(rmd2inv,mpow/Scalar(2.0));
                force_divr= rinv * rmdinv * (npow*rep*rmdninv-mpow*att*rmdminv);

                pair_eng = rep * rmdninv - att * rmdminv;

                if (energy_shift)
                    {
                    Scalar rcutninv = fast::pow(rcutsq,-npow/Scalar(2.0));
                    Scalar rcutminv = fast::pow(rcutsq,-mpow/Scalar(2.0));
                    pair_eng -= rep * rcutninv - att * rcutminv;
                    }
                return true;
                }
            else
                return false;
            }

        #ifndef __HIPCC__
        //! Get the name of this potential
        /*! \returns The potential name. Must be short and all lowercase, as this is the name energies will be logged as
            via analyze.log.
        */
        static std::string getName()
            {
            return std::string("expanded_mie");
            }

        std::string getShapeSpec() const
            {
            throw std::runtime_error("Shape definition not supported for this pair potential.");
            }
        #endif

    protected:
        Scalar rsq;     //!< Stored rsq from the constructor
        Scalar rcutsq;  //!< Stored rcutsq from the constructor
        Scalar rep;     //!< mie1 parameter extracted from the params passed to the constructor
        Scalar att;     //!< mie2 parameter extracted from the params passed to the constructor
        Scalar npow;     //!< mie3 parameter extracted from the params passed to the constructor
        Scalar mpow;     //!< mie4 parameter extracted from the params passed to the constructor
        Scalar delta;   //!< delta parameter extracted from the call to setDiameter
    };

#endif // __PAIR_EVALUATOR_EXPANDEDMIE_H__
