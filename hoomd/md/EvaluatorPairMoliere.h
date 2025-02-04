// Copyright (c) 2009-2024 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

#ifndef __PAIR_EVALUATOR_MOLIERE__
#define __PAIR_EVALUATOR_MOLIERE__

#ifndef __HIPCC__
#include <string>
#endif

#include "hoomd/HOOMDMath.h"

/*! \file EvaluatorPairMoliere.h
    \brief Defines the pair evaluator class for Moliere potentials
*/

// need to declare these class methods with __device__ qualifiers when building in nvcc
#ifdef __HIPCC__
#define DEVICE __device__
#define HOSTDEVICE __host__ __device__
#else
#define DEVICE
#define HOSTDEVICE
#endif

namespace hoomd
    {
namespace md
    {
//! Class for evaluating the Moliere pair potential.
/*! EvaluatorPairMoliere evaluates the function
    \f[ V_{\mathrm{Moliere}}(r) = \frac{Z_i Z_j e^2}{4 \pi \varepsilon_0 r_{ij}} \left[ 0.35 \exp
   \left(-0.3 \frac{r_{ij}}{a_F} \right) + 0.55 \exp \left( -1.2 \frac{r_{ij}}{a_F} \right) + 0.10
   \exp \left( -6.0 \frac{r_{ij}}{a_F} \right) \right] \f]

    where
    \f[ a_F = \frac{0.8853 a_0}{ \left( \sqrt{Z_i} + \sqrt{Z_j} \right)^{2/3}} \f]

    and \a a_0 is the Bohr radius and \a Z_x denotes the atomic number of species \a x.

*/

class EvaluatorPairMoliere
    {
    public:
    //! Define the parameter type used by this pair potential evaluator
    struct param_type
        {
        Scalar qi;
        Scalar qj;
        Scalar aF;

        DEVICE void load_shared(char*& ptr, unsigned int& available_bytes) { }

        HOSTDEVICE void allocate_shared(char*& ptr, unsigned int& available_bytes) const { }

#ifdef ENABLE_HIP
        // set CUDA memory hints
        void set_memory_hint() const { }
#endif

#ifndef __HIPCC__
        param_type() : qi(0), qj(0), aF(0) { }

        param_type(pybind11::dict v, bool managed = false)
            {
            qi = v["qi"].cast<Scalar>();
            qj = v["qj"].cast<Scalar>();
            aF = v["aF"].cast<Scalar>();
            }

        pybind11::dict asDict()
            {
            pybind11::dict v;
            v["qi"] = qi;
            v["qj"] = qj;
            v["aF"] = aF;
            return v;
            }
#endif
        } __attribute__((aligned(16)));

    //! Constructs the pair potential evaluator
    /*! \param _rsq Squared distance between the particles.
        \param _rcutsq Squared distance at which the potential goes to zero.
        \param _params Per type-pair parameters of this potential
    */
    DEVICE EvaluatorPairMoliere(Scalar _rsq, Scalar _rcutsq, const param_type& _params)
        : rsq(_rsq), rcutsq(_rcutsq), Zsq(_params.qi * _params.qj), aF(_params.aF)
        {
        }

    //! Moliere potential does not use particles charges
    DEVICE static bool needsCharge()
        {
        return false;
        }
    //! Accept the optional charge values
    /*! \param qi Charge of particle i
        \param qj Charge of particle j
    */
    DEVICE void setCharge(Scalar qi, Scalar qj) { }

    //! Evaluate the force and energy.
    /*! \param force_divr Output parameter to write the computed force divided by r
        \param pair_eng Output parameter to write the computed pair energy
        \param energy_shift If true, the potential must be shifted so that V(r) is continuous at the
       cutoff

        \return True if they are evaluated or false if they are not because we are beyond the cutoff
    */
    DEVICE bool evalForceAndEnergy(Scalar& force_divr, Scalar& pair_eng, bool energy_shift)
        {
        // compute the force divided by r in force_divr
        if (rsq < rcutsq && Zsq != 0 && aF != 0)
            {
            Scalar r2inv = Scalar(1.0) / rsq;
            Scalar rinv = fast::rsqrt(rsq);

            // precalculate the exponential terms
            Scalar exp1 = Scalar(0.35) * fast::exp(Scalar(-0.3) / aF / rinv);
            Scalar exp2 = Scalar(0.55) * fast::exp(Scalar(-1.2) / aF / rinv);
            Scalar exp3 = Scalar(0.1) * fast::exp(Scalar(-6.0) / aF / rinv);

            // evaluate the force
            force_divr = rinv * (exp1 + exp2 + exp3);
            force_divr += Scalar(1.0) / aF
                          * (Scalar(0.3) * exp1 + Scalar(1.2) * exp2 + Scalar(6.0) * exp3);
            force_divr *= Zsq * r2inv;

            // evaluate the pair energy
            pair_eng = Zsq * rinv * (exp1 + exp2 + exp3);
            if (energy_shift)
                {
                Scalar rcutinv = fast::rsqrt(rcutsq);

                Scalar expcut1 = Scalar(0.35) * fast::exp(Scalar(-0.3) / aF / rcutinv);
                Scalar expcut2 = Scalar(0.55) * fast::exp(Scalar(-1.2) / aF / rcutinv);
                Scalar expcut3 = Scalar(0.1) * fast::exp(Scalar(-6.0) / aF / rcutinv);

                pair_eng -= Zsq * rcutinv * (expcut1 + expcut2 + expcut3);
                }

            return true;
            }
        else
            return false;
        }

    DEVICE Scalar evalPressureLRCIntegral()
        {
        return 0;
        }

    DEVICE Scalar evalEnergyLRCIntegral()
        {
        return 0;
        }

#ifndef __HIPCC__
    //! Get the name of this potential
    /*! \returns The potential name.
     */
    static std::string getName()
        {
        return std::string("moliere");
        }

    std::string getShapeSpec() const
        {
        throw std::runtime_error("Shape definition not supported for this pair potential.");
        }
#endif

    protected:
    Scalar rsq;    //!< Stored rsq from the constructor
    Scalar rcutsq; //!< Stored rcutsq from the constructor
    Scalar Zsq;    //!< Zsq parameter extracted from the params passed to the constructor
    Scalar aF;     //!< aF parameter extracted from the params passed to the constructor
    };

    } // end namespace md
    } // end namespace hoomd

#endif // __PAIR_EVALUATOR_MOLIERE__
