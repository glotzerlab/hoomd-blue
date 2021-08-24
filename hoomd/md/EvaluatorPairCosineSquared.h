// Copyright (c) 2009-2021 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.

#ifndef __PAIR_EVALUATOR_COSINESQUARED_H__
#define __PAIR_EVALUATOR_COSINESQUARED_H__

#ifndef __HIPCC__
#include <string>
#endif

#include "hoomd/HOOMDMath.h"

/*! \file EvaluatorPairCosineSquared.h
    \brief Defines the pair evaluator class for Cosine Squared potentials
*/

// need to declare these class methods with __device__ qualifiers when building in nvcc
// DEVICE is __host__ __device__ when included in nvcc and blank when included into the host
// compiler
#ifdef __HIPCC__
#define DEVICE __device__
#define HOSTDEVICE __host__ __device__
#else
#define DEVICE
#define HOSTDEVICE
#endif

//! Class for evaluating the Cosine Squared pair potential
/*! <b>General Overview</b>

     EvaluatorPairCosineSquared is a low level computation class that computes the Cosine Squared
    pair potential V(r). As defined in Cooke and Deserno, Journal of Chemical Physics 123, 224710,
    2005 (https://doi.org/10.1063/1.2135785). 
    
     The potential consists of an attractive Cosine Squared potential, optionally combined with a 
    repulsive Weeks-Chandler-Anderson potential. The WCA potential is on by default and defined by
    the mode parameter in the Python class.

     <b> Cosine Squared specifics </b>
    
    EvaluatorPairCosineSquared evaluates the function:

    r < sigma and wca = True
        \f[ V_{\mathrm{CosSq}}(r) = \varepsilon \left[ \left(\frac{\sigma}{r} \right)^{12} - 
                2 \left(\frac{\sigma}{r} \right)^{6} \right]]
        \f[ -\frac{1}{r} \frac{\partial V_{\mathrm{CosSq}}}{\partial r} = 12 \varepsilon 
                \left[\sigma^{12} \cdot r^{-14} - \sigma^{6} \cdot r^{-8} \right]]
    r < sigma and wca = False
        \f[ V_{\mathrm{CosSq}}(r) = -\varepsilon]
        \f[ -\frac{1}{r}\frac{\partial V_{\mathrm{CosSq}}(r)}{\partial r} = 0]
    sigma < r < r_cut
        \f[ V_{\mathrm{CosSq}}(r) = -\varepsilon 
                cos^{2} \left[ \frac{\pi(r - \sigma)}{2(r_{c} - \sigma)} \right]]
        \f[ -\frac{1}{r} \frac{\partial \mathrm{V_{CosSq}}}{\partial r} = 
                -\varepsilon \frac{\pi}{2(r_{c} - \sigma)} 
                sin \left[ \frac{\pi(r - \sigma)}{r_{c} - \sigma} \right]]
    r >= r_cut
        \f[ V_{\mathrm{CosSq}}(r) = 0]
        \f[ -\frac{1}{r} \frac{\partial \mathrm{V_{CosSq}}}{\partial r} = 0]

     Similar to the LJ potential wca parameters are defined as
    - \a wca1 = epsilon * pow(sigma, 12)
    - \a wca2 = epsilon * pow(sigma, 6)
   
*/
class EvaluatorPairCosineSquared
    {
    public:
    //! Define the parameter type used by this pair potential evaluator
    struct param_type
        {
        Scalar epsilon;
        Scalar sigma;
        Scalar wca1;
        Scalar wca2;
        bool wca;

        DEVICE void load_shared(char*& ptr, unsigned int& available_bytes) { }

        HOSTDEVICE void allocate_shared(char*& ptr, unsigned int& available_bytes) const { }

#ifdef ENABLE_HIP
        //! Set CUDA memory hints
        void set_memory_hint() const
            {
            // default implementation does nothing
            }
#endif

#ifndef __HIPCC__
        param_type() : epsilon(0), sigma(0), wca1(0), wca2(0), wca(true) { }

        param_type(pybind11::dict v, bool managed = false)
            {
            sigma = v["sigma"].cast<Scalar>();
            epsilon = v["epsilon"].cast<Scalar>();
            wca = v["wca"].cast<bool>();
            wca1 = epsilon * pow(sigma, 12.0);
            wca2 = epsilon * pow(sigma, 6.0);
            }

        // this constructor facilitates unit testing
        param_type(Scalar sigma, Scalar epsilon, bool managed = false)
            {
            wca1 = epsilon * pow(sigma, 12.0);
            wca2 = epsilon * pow(sigma, 6.0);
            }

        pybind11::dict asDict()
            {
            pybind11::dict v;
            v["sigma"] = sigma;
            v["epsilon"] = epsilon;
            v["wca"] = wca;
            return v;
            }
#endif
        }
#ifdef SINGLE_PRECISION
    __attribute__((aligned(8)));
#else
    __attribute__((aligned(16)));
#endif

    //! Constructs the pair potential evaluator
    /*! \param _rsq Squared distance between the particles
        \param _rcutsq Squared distance at which the potential goes to 0
        \param _params Per type pair parameters of this potential
    */
    DEVICE EvaluatorPairCosineSquared(Scalar _rsq, Scalar _rcutsq, const param_type& _params)
        : rsq(_rsq), rcutsq(_rcutsq), wca1(_params.wca1), wca2(_params.wca2), 
        sigma(_params.sigma), epsilon(_params.epsilon), wca(_params.wca)
        {
        }

    //! CosineSquared doesn't use diameter
    DEVICE static bool needsDiameter()
        {
        return false;
        }
    //! Accept the optional diameter values
    /*! \param di Diameter of particle i
        \param dj Diameter of particle j
    */
    DEVICE void setDiameter(Scalar di, Scalar dj) { }

    //! CosineSquared doesn't use charge
    DEVICE static bool needsCharge()
        {
        return false;
        }
    //! Accept the optional diameter values
    /*! \param qi Charge of particle i
        \param qj Charge of particle j
    */
    DEVICE void setCharge(Scalar qi, Scalar qj) { }

    //! Evaluate the force and energy
    /*! \param force_divr Output parameter to write the computed force divided by r.
        \param pair_eng Output parameter to write the computed pair energy
        \param energy_shift energy_shift is not used in this potential

        \return True if they are evaluated or false if they are not because
        we are beyond the cutoff
    */
    DEVICE bool evalForceAndEnergy(Scalar& force_divr, Scalar& pair_eng, bool energy_shift)
        {
        // compute the force divided by r in force_divr

        // Use sigmasq to check case to prevent r, r_cut calculation if rsq < sigmasq
        Scalar sigmasq = sigma * sigma;

        if (rsq < sigmasq && sigmasq != 0)
            {
            if (wca)
                {
                Scalar r2inv = Scalar(1.0) / rsq;
                Scalar r6inv = r2inv * r2inv * r2inv;

                force_divr = Scalar(12.0) * r2inv * r6inv * (wca1 * r6inv - wca2);
                pair_eng = r6inv * (wca1 * r6inv - Scalar(2.0) * wca2);

                return true;
                }
            else
                {
                force_divr = Scalar(0.0);
                pair_eng = -epsilon;
                return true;
                }
            }
        else if (rsq < rcutsq) 
            {
            Scalar rinv = fast::rsqrt(rsq);
            Scalar r = Scalar(1.0) / rinv;
            Scalar rcutinv = fast::rsqrt(rcutsq);
            Scalar rcut = Scalar(1.0) / rcutinv;

            Scalar wc = rcut - sigma;
            Scalar piwcinv = M_PI / wc;

            Scalar sinterm = piwcinv * (r - sigma);
            Scalar cosres = fast::cos(Scalar(0.5) * sinterm);
            Scalar cossquared = cosres * cosres;
            Scalar sinres = fast::sin(sinterm);

            force_divr = -piwcinv * epsilon * rinv * sinres;
            pair_eng = -epsilon * cossquared;
            
            return true;
            }
        else
            return false;
        }

#ifndef __HIPCC__
    //! Get the name of this potential
    /*! \returns The potential name.
     */
    static std::string getName()
        {
        return std::string("cosinesquared");
        }

    std::string getShapeSpec() const
        {
        throw std::runtime_error("Shape definition not supported for this pair potential.");
        }
#endif

    protected:
    Scalar rsq;    //!< Stored rsq from the constructor
    Scalar rcutsq; //!< Stored rcutsq from the constructor
    Scalar wca1; //!< parameters passed to the constructor
    Scalar wca2; //!< parameters passed to the constructor
    Scalar sigma; //!< parameters passed to the constructor
    Scalar epsilon; //!< parameters passed to the constructor
    bool wca;
    };

#endif // __PAIR_EVALUATOR_COSINESQUARED_H__
