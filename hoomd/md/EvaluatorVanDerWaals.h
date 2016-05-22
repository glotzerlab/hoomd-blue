// Copyright (c) 2009-2016 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.


#ifndef __EVALUATOR_VDW__
#define __EVALUATOR_VDW__

#ifndef NVCC
#include <string>
#endif

#include "hoomd/HOOMDMath.h"

/*! \file EvaluatorVanDerWaals.h
    \brief Defines the evaluator class for the three-body VanDerWaals potential

    For a derivation of the potential equations, see

    [1] I. Pagonabarraga and D. Frenkel, "Dissipative particle dynamics for interacting systems,"
    J. Chem. Phys., vol. 115, no. 11, pp. 5015-5026, 2001.

    For a discussion of the terms arising from coarse-graining, see

    [2] S. Y. Trofimov, E. L. F. Nies, and M. A. J. Michels,
    "Thermodynamic consistency in dissipative particle dynamics simulations of strongly nonideal liquids and liquid mixtures,"
    J. Chem. Phys., vol. 117, no. 20, pp. 9383-9394, 2002.
*/

#ifdef NVCC
#define DEVICE __device__
#define HOSTDEVICE __host__ __device__
#else
#define DEVICE
#define HOSTDEVICE
#endif

const Scalar eps_vdw(1e-3);

//! Parameter structure for vdW fluids
struct vdw_params
    {
    //! Constructor
    vdw_params(Scalar _a,
       Scalar _b,
       Scalar _alpha,
       Scalar _T,
       Scalar _N)
        : a(_a), b(_b), alpha(_alpha), T(_T), N(_N)
        { }

    Scalar a;     //!< energy parameter of vdW EOS
    Scalar b;     //!< covolume of vdW EOS
    Scalar alpha; //!< Coefficient of cubic term
    Scalar T;     //!< Temperature
    Scalar N;     //!< number of real particles per CG bead (default == 1)
    };

//! Class for evaluating the VanDerWaals three-body potential
class EvaluatorVanDerWaals
    {
    public:
        //! Define the parameter type used by this evaluator
        typedef vdw_params param_type;

        //! Constructs the evaluator
        /*! \param _rij_sq Squared distance between particles i and j
            \param _rcutsq Squared distance at which the potential goes to zero
            \param _params Per type-pair parameters for this potential
        */
        DEVICE EvaluatorVanDerWaals(Scalar _rij_sq, Scalar _rcutsq, const param_type& _params)
            : rij_sq(_rij_sq), rcutsq(_rcutsq), a(_params.a), b(_params.b), alpha(_params.alpha),
              T(_params.T), N(_params.N)
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

        //! We have a per-particl excess free energy
        DEVICE static bool hasPerParticleEnergy() { return true; }

        //! We don't need per-particle-pair chi
        DEVICE static bool needsChi() { return false; }

        //! We don't have ik-forces
        DEVICE static bool hasIkForce() { return false; }

        //! The VanDerWaals potential needs the bond angle
        DEVICE static bool needsAngle() { return false; }

        //! Set the bond angle value
        //! \param _cos_th Cosine of the angle between ij and ik
        DEVICE void setAngle(Scalar _cos_th)
            { }

        //! Check whether a pair of particles is interactive
        DEVICE bool areInteractive()
            {
            return true;
            }

        //! Evaluate the repulsive and attractive terms of the force
        DEVICE bool evalRepulsiveAndAttractive(Scalar& fR, Scalar& fA)
            {
            // this method does nothing except checking if we're inside the cut-off
            return (rij_sq < rcutsq);
            }

        //! Evaluate chi (the scalar ik contribution) for this triplet
        DEVICE void evalChi(Scalar& chi)
            {
            }

        //! Evaluate chi (the scalar ij contribution) for this triplet
        DEVICE void evalPhi(Scalar& phi)
            {
            // add up density n_i
            if (rij_sq < rcutsq)
                {
                Scalar norm(15.0/(2.0*M_PI));
                Scalar rcut = fast::sqrt(rcutsq);
                norm /= rcutsq*rcut;

                Scalar rij = fast::sqrt(rij_sq);
                Scalar fac = Scalar(1.0)-rij/rcut;
                phi += fac*fac*norm;
                }
            }

        //! Evaluate the force and potential energy due to ij interactions
        DEVICE void evalForceij(Scalar fR,
                                Scalar fA,
                                Scalar chi,
                                Scalar phi,
                                Scalar& bij,
                                Scalar& force_divr,
                                Scalar& potential_eng)
            {
            if (rij_sq < rcutsq)
                {
                Scalar rho_i = phi;
                Scalar norm(15.0/(2.0*M_PI));
                Scalar rcut = fast::sqrt(rcutsq);
                norm /= rcutsq*rcut;

                Scalar rij = fast::sqrt(rij_sq);
                Scalar fac = Scalar(1.0)-rij/rcut;

                // add self-weight
 //               rho_i += norm; // see discussion in [2]

                // compute the ij force
                Scalar w_prime = Scalar(2.0)*norm*fac/rcut/rij;
                force_divr = (T*b/(Scalar(1.0)-b*rho_i)-a-alpha*a*b*rho_i)*w_prime;

                // guard against numerical errors at low density
                if (fabs(N-Scalar(1.0)) > eps_vdw)
                    {
                    // CG correction
                    force_divr += (N-Scalar(1.0))*T/rho_i/(Scalar(1.0)-b*rho_i)*w_prime;
                    }
                }
            }

        DEVICE void evalSelfEnergy(Scalar& energy, Scalar phi)
            {
            Scalar rho_i = phi;

            // add self-interaction
            Scalar norm(15.0/(2.0*M_PI));
            Scalar rcut = fast::sqrt(rcutsq);
            norm /= rcutsq*rcut;

            // add self-weight
//            rho_i += norm; // see discussion in [2]

            // *excess* free energy of a vdW fluid (subtract ideal gas contribution)
            energy = -T*logf(Scalar(1.0)-b*rho_i)-a*rho_i-Scalar(0.5)*alpha*a*b*rho_i*rho_i;


            // guard against numerical errors at low density
            if (fabs(N-Scalar(1.0)) > eps_vdw)
                {
                // CG correction
                energy += (N-Scalar(1.0))*T*logf(rho_i/(Scalar(1.0)-b*rho_i));
                }
            }

        //! Evaluate the forces due to ijk interactions
        DEVICE bool evalForceik(Scalar fR,
                                Scalar fA,
                                Scalar chi,
                                Scalar bij,
                                Scalar3& force_divr_ij,
                                Scalar3& force_divr_ik)
            {
            return false;
            }

        #ifndef NVCC
        //! Get the name of this potential
        /*! \returns The potential name.  Must be short and all lowercase, as this is the name
            energies will be logged as via analyze.log.
        */
        static std::string getName()
            {
            return std::string("van_der_waals");
            }
        #endif

    protected:
        Scalar rij_sq; //!< Stored rij_sq from the constructor
        Scalar rik_sq; //!< Stored rik_sq
        Scalar rcutsq; //!< Stored rcutsq from the constructor
        Scalar a;      //!< a parameter in vdW EOS
        Scalar b;      //!< b parameter in vdW EOS
        Scalar alpha;  //!< Coefficient of cubic term
        Scalar T;      //!< temperature scaling factor for ideal gas
        Scalar N;      //!< coarse-graining multiplier
    };

#endif
