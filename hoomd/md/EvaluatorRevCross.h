// Copyright (c) 2009-2024 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

//

#ifndef __EVALUATOR_REVCROSS__
#define __EVALUATOR_REVCROSS__

#ifndef __HIPCC__
#include <string>
#endif

#include "hoomd/HOOMDMath.h"

/*! \file EvaluatorRevCross.h
    \brief Defines the evaluator class for the three-body RevCross potential
*/

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
//! Class for evaluating the RevCross three-body potential
class EvaluatorRevCross
    {
    public:
    struct param_type
        {
        Scalar sigma;
        Scalar n;
        Scalar epsilon;
        Scalar lambda3;

#ifdef ENABLE_HIP
        //! Set CUDA memory hints
        void set_memory_hint() const
            {
            // default implementation does nothing
            }
#endif

#ifndef __HIPCC__
        param_type() : sigma(0), n(0), epsilon(0), lambda3(0) { }

        param_type(pybind11::dict v)
            {
            sigma = v["sigma"].cast<Scalar>();
            n = v["n"].cast<Scalar>();
            epsilon = v["epsilon"].cast<Scalar>();
            lambda3 = v["lambda3"].cast<Scalar>();
            }

        pybind11::dict asDict()
            {
            pybind11::dict v;
            v["sigma"] = sigma;
            v["n"] = n;
            v["epsilon"] = epsilon;
            v["lambda3"] = lambda3;
            return v;
            }
#endif
        } __attribute__((aligned(16)));

    //! Constructs the evaluator
    /*! \param _rij_sq Squared distance between particles i and j
        \param _rcutsq Squared distance at which the potential goes to zero
        \param _params Per type-pair parameters for this potential
    */
    DEVICE EvaluatorRevCross(Scalar _rij_sq,
                             Scalar _rcutsq,
                             const param_type& _params) // here it receives also r cutoff
        : rij_sq(_rij_sq), rcutsq(_rcutsq), sigma_dev(_params.sigma), n_dev(_params.n),
          epsilon_dev(_params.epsilon), lambda3_dev(_params.lambda3)
        {
        }

    //! Set the square distance between particles i and j
    DEVICE void setRij(Scalar rsq)
        {
        rij_sq = rsq;
        }

    //! Set the square distance between particles i and k
    DEVICE void setRik(Scalar rsq)
        {
        rik_sq = rsq;
        }

    //! This is a pure pair potential
    DEVICE static bool hasPerParticleEnergy()
        {
        return false;
        }

    //! We do not need chi
    DEVICE static bool needsChi()
        {
        return false;
        }

    //! We have ik-forces
    DEVICE static bool hasIkForce()
        {
        return true;
        }

    //! The RevCross potential does not need the bond angle
    DEVICE static bool needsAngle()
        {
        return false;
        }

    //! No need for the bond angle value
    DEVICE void setAngle(Scalar _cos_th) { }

    //! Check whether a pair of particles is interactive
    DEVICE bool areInteractive()
        {
        return (rik_sq < rcutsq) && (epsilon_dev != Scalar(0.0));
        }

    //! Evaluate the repulsive and attractive terms of the force
    DEVICE bool evalRepulsiveAndAttractive(Scalar& invratio, Scalar& invratio2)
        {
        if ((rij_sq < rcutsq) && (epsilon_dev != Scalar(0.0)))
            {
            // compute rij
            Scalar rij = fast::sqrt(rij_sq);

            // compute the power of the ratio
            invratio = fast::pow(sigma_dev / rij, n_dev);
            invratio2 = invratio * invratio;

            return true;
            }
        else
            return false;
        }

    //! We do not have to evaluate chi
    DEVICE void evalChi(Scalar& chi) { }

    //! We don't have a scalar ij contribution
    DEVICE void evalPhi(Scalar& phi) { }

    //! Evaluate the force and potential energy due to ij interactions
    DEVICE void evalForceij(Scalar invratio,
                            Scalar invratio2,
                            Scalar chi,  // not used
                            Scalar phi,  // not used
                            Scalar& bij, // not used
                            Scalar& force_divr,
                            Scalar& potential_eng)
        {
        // compute the ij force
        // the force term includes rij_sq^-1 from the derivative over the distance and a factor 0.5
        // to compensate for double countings
        force_divr
            = Scalar(2.0) * epsilon_dev * n_dev * (Scalar(2.0) * invratio2 - invratio) / rij_sq;

        // compute the potential energy
        potential_eng = epsilon_dev * (invratio2 - invratio);
        }

    DEVICE void evalSelfEnergy(Scalar& energy, Scalar phi) { }

    //! Evaluate the forces due to ijk interactions
    DEVICE bool evalForceik(Scalar ijinvratio,
                            Scalar ijinvratio2,
                            Scalar chi, // not used
                            Scalar bij, // not used
                            Scalar3& IN_force_divr_ij,
                            Scalar3& IN_force_divr_ik)
        {
        if (rik_sq < rcutsq)
            {
            // For compatibility with Tersoff I get 3d vectors in, but I need only to calculate
            // their modulus and I store it in the x component
            Scalar force_divr_ij = IN_force_divr_ij.x;
            Scalar force_divr_ik = IN_force_divr_ik.x;

            // compute rij, rik, rcut
            Scalar rij = fast::sqrt(rij_sq);
            Scalar rik = fast::sqrt(rik_sq);
            Scalar rm = sigma_dev * fast::pow(Scalar(2.0), Scalar(1.0) / n_dev);
            Scalar ikinvratio = fast::pow(sigma_dev / rik, n_dev);
            Scalar ikinvratio2 = ikinvratio * ikinvratio;

            // In this case the three particles interact and we have to find which one of the three
            // scenarios is realized:
            // (1) both k and j closer than rm ----> no forces from the three body term
            // (2) only one closer than rm ----> two body interaction compensated by the three body
            // for i and the closer (3) both farther than rm ----> complete avaluation of the forces

            // case (1) is trivial. The following are the two realization of case (2)
            if ((rij > rm) && (rik <= rm))
                {
                force_divr_ij = Scalar(-4.0) * epsilon_dev * n_dev * lambda3_dev
                                * (Scalar(2.0) * ijinvratio2 - ijinvratio) / rij_sq;
                force_divr_ik = 0;
                }
            else if ((rij <= rm) && (rik > rm))
                {
                force_divr_ij = 0;
                // each triplets is evaluated only once
                force_divr_ik = Scalar(-4.0) * epsilon_dev * n_dev * lambda3_dev
                                * (Scalar(2.0) * ikinvratio2 - ikinvratio) / rik_sq;
                }
            //~~~~~~~~~~~~~~~~then case (3), look at S. Ciarella and W.G. Ellenbroek 2019
            // https://arxiv.org/abs/1912.08569 for details
            else if ((rij > rm) && (rik > rm))
                {
                // starting with the contribute of the particle j in the 3B term
                force_divr_ij = lambda3_dev * Scalar(16.0) * epsilon_dev * n_dev
                                * (Scalar(2.0) * ijinvratio2 - ijinvratio)
                                * (ikinvratio2 - ikinvratio) / rij_sq;

                // then the contribute of the particle k in the 3B term
                force_divr_ik = lambda3_dev * Scalar(16.0) * epsilon_dev * n_dev
                                * (Scalar(2.0) * ikinvratio2 - ikinvratio)
                                * (ijinvratio2 - ijinvratio) / rik_sq;
                }

            // Return the forces
            IN_force_divr_ij.x = force_divr_ij;
            IN_force_divr_ik.x = force_divr_ik;

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
        return std::string("revcross");
        }
#endif

    static const bool flag_for_RevCross = true;

    protected:
    Scalar rij_sq; //!< Stored rij_sq from the constructor
    Scalar rik_sq; //!< Stored rik_sq from the constructor
    Scalar rcutsq; //!< Stored rcutsq from the constructor
    Scalar sigma_dev;
    Scalar n_dev;
    Scalar epsilon_dev;
    Scalar lambda3_dev;
    };

    } // end namespace md
    } // end namespace hoomd

#endif
