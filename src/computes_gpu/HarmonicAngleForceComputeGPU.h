#ifdef WIN32
#pragma warning( push )
#pragma warning( disable : 4103 )
#endif

#include "HarmonicAngleForceCompute.h"
#include "HarmonicAngleForceGPU.cuh"

#include <boost/shared_ptr.hpp>
#include <boost/signals.hpp>

/*! \file HarmonicAngleForceComputeGPU.h
	\brief Declares the HarmonicAngleForceGPU class
*/

#ifndef __HARMONICANGLEFORCECOMPUTEGPU_H__
#define __HARMONICANGLEFORCECOMPUTEGPU_H__

//! Implements the harmonic angle force calculation on the GPU
/*!	HarmonicAngleForceComputeGPU implements the same calculations as HarmonicAngleForceCompute,
	but executing on the GPU.
	
	Per-type parameters are stored in a simple global memory area pointed to by
	\a m_gpu_params. They are stored as float2's with the \a x component being K and the
	\a y component being t_0.
	
	The GPU kernel can be found in angleforce_kernel.cu.

	\ingroup computes
*/
class HarmonicAngleForceComputeGPU : public HarmonicAngleForceCompute
	{
	public:
		//! Constructs the compute
		HarmonicAngleForceComputeGPU(boost::shared_ptr<ParticleData> pdata);
		//! Destructor
		~HarmonicAngleForceComputeGPU();
		
		//! Sets the block size to run on the device
		/*! \param block_size Block size to set
		*/
		void setBlockSize(int block_size) { m_block_size = block_size; }
		
		//! Set the parameters
		virtual void setParams(unsigned int type, Scalar K, Scalar t_0);
		
	protected:
		int m_block_size;		//!< Block size to run calculation on
		vector<float2 *> m_gpu_params;	//!< Parameters stored on the GPU
		float2 *m_host_params;	//!< Host parameters
		
		//! Actually compute the forces
		virtual void computeForces(unsigned int timestep);
	};
	
//! Export the AngleForceComputeGPU class to python
void export_HarmonicAngleForceComputeGPU();

#endif

#ifdef WIN32
#pragma warning( pop )
#endif

