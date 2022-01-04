// Copyright (c) 2009-2022 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

#ifndef __PAIR_EVALUATOR_ExpandedMie_H__
#define __PAIR_EVALUATOR_ExpandedMie_H__

#ifndef __HIPCC__
#include <string>
#endif

#include "EvaluatorPairLJ.h"
#include "hoomd/HOOMDMath.h"

/*! \file EvaluatorPairExpandedMie.h
    \brief Defines the pair evaluator class for Expanded Mie potentials
*/

// need to declare these class methods with __device__ qualifiers when building in nvcc
// DEVICE is __host__ __device__ when included in nvcc and blank when included into the host
// compiler
#ifdef __HIPCC__
#define DEVICE __device__
#else
#define DEVICE
#endif

namespace hoomd
    {
namespace md
    {
//! Class for evaluating the Expanded Mie pair potential
/*! <b>General Overview</b>

    See EvaluatorPairSLJ and EvaluatorPairMie

    <b>ExpandedMie specifics</b>

    EvaluatorPairExpandedMie evaluates the function:
    \f{eqnarray*}
    V_{\mathrm{ExpMie}}(r)  = \left( \frac{n}{n-m} \right) {\left( \frac{n}{m}
   \right)}^{\frac{m}{n-m}} \varepsilon \left[ \left( \frac{\sigma}{r-\Delta} \right)^{n} -
                \left( \frac{\sigma}{r-\Delta} \right)^{m} \right] r \l (r_{\mathrm{cut}} +
   \Delta)\\
                = & 0 & r \ge (r_{\mathrm{cut}} + \Delta) \\
    \f}
    where  \f$ \Delta is specified by the user, conventionally calcualted as, \f$ \Delta = (d_i +
   d_j)/2 - \sigma \f$ and \f$ d_i \f$ is the diameter of particle \f$ i \f$.

   The ExpandedMie potential needs neither charge nor diameter. Five parameters are specified and
   stored in a Scalar5. \a repulsive \a attractive \a n_pow \a m_pow \a delta are stored in \a
   params


    These are related to the standard lj parameters sigma and epsilon and the variable exponents n
   and m by:
    - \a repulsive = epsilon * pow(sigma,n) * (n/(n-m)) * power(n/m,m/(n-m))
    - \a attractive = epsilon * pow(sigma,m) * (n/(n-m)) * power(n/m,m/(n-m))
    - \a n_pow = n
    - \a m_pow = m
    - \a delta = delta

    Due to the way that ExpandedMie modifies the cutoff condition, it will not function properly
   with the xplor shifting mode.
*/
class EvaluatorPairExpandedMie
    {
    public:
    //! Define the parameter type used by this pair potential evaluator
    struct param_type
        {
        Scalar repulsive;  //!< Lumped repulsive term to simplify/speed up computation
        Scalar attractive; //!< Lumped attractive term to simplify/speed up computation
        Scalar n_pow;      //!< Higher exponent for potential
        Scalar m_pow;      //!< Lower exponent for potential
        Scalar delta;      //!< shift in radial distance for use in Mie potential
        Scalar sigma;
        Scalar epsilon;

        DEVICE void load_shared(char*& ptr, unsigned int& available_bytes) { }

        HOSTDEVICE void allocate_shared(char*& ptr, unsigned int& available_bytes) const { }

#ifndef ENABLE_HIP
        //! set CUDA memory hints
        void set_memory_hint() const { }
#endif

#ifndef __HIPCC__
        param_type()
            : repulsive(0), attractive(0), n_pow(0), m_pow(0), delta(0), sigma(0), epsilon(0)
            {
            }

        param_type(const pybind11::dict v, bool managed = false)
            {
            n_pow = v["n"].cast<Scalar>();
            m_pow = v["m"].cast<Scalar>();

            sigma = v["sigma"].cast<Scalar>();
            epsilon = v["epsilon"].cast<Scalar>();

            Scalar prefactor
                = (n_pow / (n_pow - m_pow)) * fast::pow(n_pow / m_pow, m_pow / (n_pow - m_pow));
            repulsive = prefactor * epsilon * fast::pow(sigma, n_pow);
            attractive = prefactor * epsilon * fast::pow(sigma, m_pow);

            delta = v["delta"].cast<Scalar>();
            }

        pybind11::dict asDict() const
            {
            pybind11::dict v;
            v["n"] = n_pow;
            v["m"] = m_pow;

            v["epsilon"] = epsilon;
            v["sigma"] = sigma;

            v["delta"] = delta;
            return v;
            }
#endif
        } __attribute__((aligned(16)));

    //! Constructs the pair potential evaluator
    /*! \param _rsq Squared distance between the particles
        \param _rcutsq Squared distance at which the potential goes to 0
        \param _n First, larger exponent that captures hard-core repulsion
        \param _m Second, smaller exponent that captures attraction
        \param _params Per type pair parameters of this potential
        \param _delta Horizontal shift in r
    */
    DEVICE
    EvaluatorPairExpandedMie(const Scalar _rsq, const Scalar _rcutsq, const param_type& _params)
        : rsq(_rsq), rcutsq(_rcutsq), repulsive(_params.repulsive), attractive(_params.attractive),
          n_pow(_params.n_pow), m_pow(_params.m_pow), delta(_params.delta)
        {
        }

    //! ExpandedMie doesn't use diameter
    DEVICE static bool needsDiameter()
        {
        return false;
        }
    //! Accept the optional diameter values
    /*! \param di Diameter of particle i
        \param dj Diameter of particle j
    */
    DEVICE void setDiameter(Scalar di, Scalar dj) const { }

    //! ExpandedMie doesn't use charge
    DEVICE static bool needsCharge()
        {
        return false;
        }
    //! Accept the optional diameter values
    /*! \param qi Charge of particle i
        \param qj Charge of particle j
    */
    DEVICE void setCharge(Scalar qi, Scalar qj) const { }

    //! Evaluate the force and energy
    /*! \param force_divr Output parameter to write the computed force divided by r.
        \param pair_eng Output parameter to write the computed pair energy
        \param energy_shift If true, the potential must be shifted so that V(r) is continuous at the
       cutoff \note There is no need to check if rsq < rcutsq in this method. Cutoff tests are
       performed in PotentialPair.

        \return True if they are evaluated or false if they are not because we are beyond the cutoff
    */
    DEVICE bool evalForceAndEnergy(Scalar& force_divr, Scalar& pair_eng, bool energy_shift) const
        {
        // precompute some quantities
        const Scalar r = fast::sqrt(rsq);
        const Scalar rinv = fast::rsqrt(rsq);

        // compute the force divided by r in force_divr
        if (rsq < rcutsq && repulsive != 0)
            {
            Scalar rmd = r - delta;
            Scalar rmdinv = Scalar(1.0) / rmd;
            Scalar rmd2inv = rmdinv * rmdinv;
            Scalar rmdninv = fast::pow(rmd2inv, n_pow / Scalar(2.0));
            Scalar rmdminv = fast::pow(rmd2inv, m_pow / Scalar(2.0));
            force_divr
                = rinv * rmdinv * (n_pow * repulsive * rmdninv - m_pow * attractive * rmdminv);

            pair_eng = repulsive * rmdninv - attractive * rmdminv;

            if (energy_shift)
                {
                Scalar rcutninv = fast::pow(rcutsq, -n_pow / Scalar(2.0));
                Scalar rcutminv = fast::pow(rcutsq, -m_pow / Scalar(2.0));
                pair_eng -= repulsive * rcutninv - attractive * rcutminv;
                }
            return true;
            }
        else
            return false;
        }

#ifndef __HIPCC__
    //! Get the name of this potential
    /*! \returns The potential name. Must be short and all lowercase, as this is the name energies
       will be logged as via analyze.log.
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
    Scalar rsq;        //!< distance between particles squared
    Scalar rcutsq;     //!< the cutoff radius of the potential squared
    Scalar repulsive;  //!< Lumped repulsive term to simplify/speed up computation
    Scalar attractive; //!< Lumped attractive term to simplify/speed up computation
    Scalar n_pow;      //!< Higher exponent for potential
    Scalar m_pow;      //!< Lower exponent for potential
    Scalar delta;      //!< shift in radial distance for use in Mie potential
    };

    } // end namespace md
    } // end namespace hoomd

#endif // __PAIR_EVALUATOR_EXPANDEDMIE_H__
