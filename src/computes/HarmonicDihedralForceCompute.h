#include <boost/shared_ptr.hpp>

#include "ForceCompute.h"
#include "DihedralData.h"

#include <vector>

/*! \file HarmonicDihedralForceCompute.h
	\brief Declares a class for computing harmonic dihedrals
*/

#ifndef __HARMONICDIHEDRALFORCECOMPUTE_H__
#define __HARMONICDIHEDRALFORCECOMPUTE_H__

//! Computes harmonic dihedral forces on each particle
/*! Harmonic dihedral forces are computed on every particle in the simulation.

	The dihedrals which forces are computed on are accessed from ParticleData::getDihedralData
	\ingroup computes
*/
class HarmonicDihedralForceCompute : public ForceCompute
	{
	public:
		//! Constructs the compute
		HarmonicDihedralForceCompute(boost::shared_ptr<ParticleData> pdata);
		
		//! Destructor
		~HarmonicDihedralForceCompute();
		
		//! Set the parameters
		virtual void setParams(unsigned int type, Scalar K, int sign, unsigned int multiplicity);
		
		//! Returns a list of log quantities this compute calculates
		virtual std::vector< std::string > getProvidedLogQuantities(); 
		
		//! Calculates the requested log value and returns it
		virtual Scalar getLogValue(const std::string& quantity, unsigned int timestep);

	protected:
		Scalar *m_K;	//!< K parameter for multiple dihedral tyes
		Scalar *m_sign;	//!< sign parameter for multiple dihedral types
		Scalar *m_multi;//!< multiplicity parameter for multiple dihedral types
		
		boost::shared_ptr<DihedralData> m_dihedral_data;	//!< Dihedral data to use in computing dihedrals
		
		//! Actually compute the forces
		virtual void computeForces(unsigned int timestep);
	};
	
//! Exports the DihedralForceCompute class to python
void export_HarmonicDihedralForceCompute();

#endif
