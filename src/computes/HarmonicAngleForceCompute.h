#include <boost/shared_ptr.hpp>

#include "ForceCompute.h"
#include "AngleData.h"

#include <vector>

/*! \file HarmonicAngleForceCompute.h
	\brief Declares a class for computing harmonic angles
*/

#ifndef __HARMONICANGLEFORCECOMPUTE_H__
#define __HARMONICANGLEFORCECOMPUTE_H__

//! Computes harmonic angle forces on each particle
/*! Harmonic angle forces are computed on every particle in the simulation.

	The angles which forces are computed on are accessed from ParticleData::getAngleData
	\ingroup computes
*/
class HarmonicAngleForceCompute : public ForceCompute
	{
	public:
		//! Constructs the compute
		HarmonicAngleForceCompute(boost::shared_ptr<ParticleData> pdata);
		
		//! Destructor
		~HarmonicAngleForceCompute();
		
		//! Set the parameters
		virtual void setParams(unsigned int type, Scalar K, Scalar t_0);
		
		//! Returns a list of log quantities this compute calculates
		virtual std::vector< std::string > getProvidedLogQuantities(); 
		
		//! Calculates the requested log value and returns it
		virtual Scalar getLogValue(const std::string& quantity, unsigned int timestep);

	protected:
		Scalar *m_K;	//!< K parameter for multiple angle tyes
		Scalar *m_t_0;	//!< r_0 parameter for multiple angle types
		
		boost::shared_ptr<AngleData> m_angle_data;	//!< Angle data to use in computing angles
		
		//! Actually compute the forces
		virtual void computeForces(unsigned int timestep);
	};
	
//! Exports the AngleForceCompute class to python
void export_HarmonicAngleForceCompute();

#endif
