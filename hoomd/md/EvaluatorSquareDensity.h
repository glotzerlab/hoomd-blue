// Copyright (c) 2009-2019 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.


#ifndef __EVALUATOR_SQUARE_DENSITY__
#define __EVALUATOR_SQUARE_DENSITY__

#ifndef NVCC
#include <string>
#endif

#include "hoomd/HOOMDMath.h"

/*! \file EvaluatorSquareDensity.h
    \brief Defines the evaluator class for a three-body square density potential

   see P. Warren, "Vapor-liquid coexistence in many-bod dissipative particle dynamics"
   Phys. Rev. E 68, p. 066702 (2003)
*/

#ifdef NVCC
#define DEVICE __device__
#define HOSTDEVICE __host__ __device__
#else
#define DEVICE
#define HOSTDEVICE
#endif

//! Class for evaluating the SquareDensity three-body potential
class EvaluatorSquareDensity
    {
    public:
        //! Define the parameter type used by this evaluator
        typedef Scalar2 param_type;

        //! Constructs the evaluator
        /*! \param _rij_sq Squared distance between particles i and j
            \param _rcutsq Squared distance at which the potential goes to zero
            \param _params Per type-pair parameters for this potential
        */
        DEVICE EvaluatorSquareDensity(Scalar _rij_sq, Scalar _rcutsq, const param_type& _params)
            : rij_sq(_rij_sq), rcutsq(_rcutsq), A(_params.x), B(_params.y)
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

        //! The SquareDensity potential needs the bond angle
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

                // compute the ij force
                Scalar w_prime = Scalar(2.0)*norm*fac/rcut/rij;
                force_divr = B*(rho_i-A)*w_prime;
                }
            }

        DEVICE void evalSelfEnergy(Scalar& energy, Scalar phi)
            {
            Scalar rho_i = phi;

            // *excess* free energy of a vdW fluid (subtract ideal gas contribution)
            energy = Scalar(0.5)*B*(rho_i-A)*(rho_i-A);
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
            return std::string("squared_density");
            }

        std::string getShapeSpec() const
            {
            throw std::runtime_error("Shape definition not supported for this pair potential.");
            }
        #endif

    protected:
        Scalar rij_sq; //!< Stored rij_sq from the constructor
        Scalar rik_sq; //!< Stored rik_sq
        Scalar rcutsq; //!< Stored rcutsq from the constructor
        Scalar A;      //!< center of harmonic potential
        Scalar B;      //!< repulsion parameter
    };

#endif
