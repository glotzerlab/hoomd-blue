// Copyright (c) 2009-2019 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.
//
// Maintainer: SCiarella

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

//! Parameter type for this potential
struct revcross_params
    {
    Scalar sigma; //!< hard body of the particle
    Scalar n; //!< exponent
    Scalar epsilon; //!< unit energy
    Scalar lambda3; //!< three body parameter
    
};

//! Function to make the parameter type
HOSTDEVICE inline revcross_params make_revcross_params(Scalar sigma,
                                                     Scalar n,
                                                     Scalar epsilon,
                                                     Scalar lambda3
                                                     )
    {
    revcross_params retval;

    retval.sigma = sigma;
    retval.n = n;
    retval.epsilon = epsilon;
    retval.lambda3 = lambda3;
    return retval;
    }

//! Class for evaluating the RevCross three-body potential
class EvaluatorRevCross
    {
    public:
        //! Define the parameter type used by this evaluator
        typedef revcross_params param_type;

        //! Constructs the evaluator
        /*! \param _rij_sq Squared distance between particles i and j
            \param _rcutsq Squared distance at which the potential goes to zero
            \param _params Per type-pair parameters for this potential
        */
        DEVICE EvaluatorRevCross(Scalar _rij_sq, Scalar _rcutsq, const param_type& _params)          //here it receives also r cutoff
            : rij_sq(_rij_sq), rcutsq(_rcutsq),
              sigma_dev(_params.sigma), n_dev(_params.n),
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

        //! Check whether a pair of particles is interactive
        DEVICE bool areInteractive()
            {
            if ((rik_sq < rcutsq )&&(epsilon_dev!=Scalar(0.0)))                 
            //if rik_sq < rcutsq                 
                return true;
            else return false;
            }

        //! Evaluate the repulsive and attractive terms of the force
        DEVICE bool evalRepulsiveAndAttractive(Scalar& invratio, Scalar& invratio2)
            {
            if ((rij_sq < rcutsq )&&(epsilon_dev!=Scalar(0.0)))
                {
                // compute rij
                Scalar rij = fast::sqrt(rij_sq);

                // compute the power of the ratio
                invratio = fast::pow( sigma_dev/rij,n_dev );
                invratio2 = invratio*invratio;

                return true;
                }
            else return false;
            }

        //! Evaluate the force and potential energy due to ij interactions
        DEVICE void evalForceij(Scalar invratio,
                                Scalar invratio2,
                                Scalar& force_divr,
                                Scalar& potential_eng)
            {
            // compute the ij force
	    // the force term includes rij_sq^-1 from the derivative over the distance and a factor 0.5 to compensate for double countings
            force_divr = Scalar(2.0) *epsilon_dev * n_dev * (Scalar(2.0)*invratio2-invratio) / rij_sq;    

	    // compute the potential energy
            potential_eng = epsilon_dev*( invratio2 - invratio);   
            }                                                                     
                                                                           
        //! Evaluate the forces due to ijk interactions
        DEVICE bool evalForceik(Scalar ijinvratio,
                                Scalar ijinvratio2,
                                Scalar& force_divr_ij,
                                Scalar& force_divr_ik)
            {
            if (rik_sq < rcutsq)
                {
                // compute rij, rik, rcut
                Scalar rij = fast::sqrt(rij_sq);
                Scalar rik = fast::sqrt(rik_sq);
                Scalar rm = sigma_dev * fast::pow(Scalar(2.0),Scalar(1.0)/n_dev);    
                Scalar ikinvratio = fast::pow(sigma_dev/rik, n_dev);
                Scalar ikinvratio2 = ikinvratio*ikinvratio;

		//In this case the three particles interact and we have to find which one of the three scenarios is realized:
		// (1) both k and j closer than rm ----> no forces from the three body term
		// (2) only one closer than rm ----> two body interaction compensated by the three body for i and the closer 
		// (3) both farther than rm ----> complete avaluation of the forces
		
		//case (1) is trivial. The following are the two realization of case (2) 
		if ((rij > rm) && (rik <= rm) ){	
		       	
		       force_divr_ij = Scalar(-4.0) * epsilon_dev * n_dev * lambda3_dev * (Scalar(2.0)*ijinvratio2-ijinvratio) / rij_sq;		
		       force_divr_ik = 0;		
		       				
		}
		else if ((rij <= rm ) && (rik > rm) ){
		
		       force_divr_ij = 0;		
		       //each triplets is evaluated only once
		       force_divr_ik = Scalar(-4.0) * epsilon_dev * n_dev * lambda3_dev * (Scalar(2.0)*ikinvratio2-ikinvratio) / rik_sq;	
		       				
		}
		       	
		//~~~~~~~~~~~~~~~~then case (3), look at S. Ciarella and W.G. Ellenbroek 2019 https://arxiv.org/abs/1912.08569 for details
		else if ((rij > rm) && (rik > rm) ){
		
		       //starting with the contribute of the particle j in the 3B term	
		       force_divr_ij = lambda3_dev * Scalar(16.0) * epsilon_dev * n_dev * ( Scalar(2.0)*ijinvratio2 - ijinvratio )*( ikinvratio2 - ikinvratio ) / rij_sq ;							
		
		       //then the contribute of the particle k in the 3B term	
		       force_divr_ik = lambda3_dev * Scalar(16.0) * epsilon_dev * n_dev * ( Scalar(2.0)*ikinvratio2 - ikinvratio )*( ijinvratio2 - ijinvratio ) / rik_sq ;		
		       	
		}
	
                return true;
                }
            else return false;
            }

        #ifndef __HIPCC__
        //! Get the name of this potential
        /*! \returns The potential name.  Must be short and all lowercase, as this is the name
            energies will be logged as via analyze.log.
        */
        static std::string getName()
            {
            return std::string("revcross");
            }
        #endif

    protected:
        Scalar rij_sq; //!< Stored rij_sq from the constructor
        Scalar rik_sq; //!< Stored rik_sq from the constructor
        Scalar rcutsq; //!< Stored rcutsq from the constructor
        Scalar sigma_dev;
        Scalar n_dev;
        Scalar epsilon_dev;
        Scalar lambda3_dev;
    };

#endif
