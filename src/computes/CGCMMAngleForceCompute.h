#include <boost/shared_ptr.hpp>

#include "ForceCompute.h"
#include "AngleData.h"

#include <vector>

/*! \file HarmonicAngleForceCompute.h
	\brief Declares a class for computing harmonic bonds
*/

#ifndef __CGCMMANGLEFORCECOMPUTE_H__
#define __CGCMMANGLEFORCECOMPUTE_H__

//! Computes course grained harmonic angle forces on each particle
/*! Harmonic angle forces are computed on every particle in the simulation.

	The angles which forces are computed on are accessed from ParticleData::getAngleData
	\ingroup computes
*/
class CGCMMAngleForceCompute : public ForceCompute
	{
	public:
		//! Constructs the compute
		CGCMMAngleForceCompute(boost::shared_ptr<ParticleData> pdata);
		
		//! Destructor
		~CGCMMAngleForceCompute();
		
		//! Set the parameters
		virtual void setParams(unsigned int type, Scalar K, Scalar t_0, unsigned int cg_type, Scalar eps, Scalar sigma);
		
		//! Returns a list of log quantities this compute calculates
		virtual std::vector< std::string > getProvidedLogQuantities(); 
		
		//! Calculates the requested log value and returns it
		virtual Scalar getLogValue(const std::string& quantity, unsigned int timestep);

	protected:
		Scalar *m_K;	//!< K parameter for multiple angle tyes
		Scalar *m_t_0;	//!< t_0 parameter for multiple angle types

                // THESE ARE NEW FOR GC ANGLES
		Scalar *m_eps;	//!< epsilon parameter for 1-3 repulsion of multiple angle tyes
		Scalar *m_sigma;//!< sigma parameter for 1-3 repulsion of multiple angle types
		Scalar *m_rcut;//!< cutoff parameter for 1-3 repulsion of multiple angle types
                unsigned int *m_cg_type;//!< course grain angle type (0-3)

                float prefact[4];//!< prefact precomputed prefactors for CG-CMM angles
                float cgPow1[4];//!< cgPow1 1st powers for CG-CMM angles
                float cgPow2[4];//!< cgPow2 2nd powers for CG-CMM angles
		
		boost::shared_ptr<AngleData> m_CGCMMangle_data; //!< Angle data to use in computing angles
		
		//! Actually compute the forces
		virtual void computeForces(unsigned int timestep);
	};
	
//! Exports the BondForceCompute class to python
void export_CGCMMAngleForceCompute();

#endif
