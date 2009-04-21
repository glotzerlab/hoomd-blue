#ifdef WIN32
#pragma warning( push )
#pragma warning( disable : 4103 )
#endif

#include "HarmonicDihedralForceCompute.h"
#include "HarmonicDihedralForceGPU.cuh"

#include <boost/shared_ptr.hpp>
#include <boost/signals.hpp>

/*! \file HarmonicDihedralForceComputeGPU.h
	\brief Declares the HarmonicDihedralForceGPU class
*/

#ifndef __HARMONICDIHEDRALFORCECOMPUTEGPU_H__
#define __HARMONICDIHEDRALFORCECOMPUTEGPU_H__

//! Implements the harmonic dihedral force calculation on the GPU
/*!	HarmonicDihedralForceComputeGPU implements the same calculations as HarmonicDihedralForceCompute,
	but executing on the GPU.
	
	Per-type parameters are stored in a simple global memory area pointed to by
	\a m_gpu_params. They are stored as float2's with the \a x component being K and the
	\a y component being t_0.
	
	The GPU kernel can be found in dihedralforce_kernel.cu.

	\ingroup computes
*/
class HarmonicDihedralForceComputeGPU : public HarmonicDihedralForceCompute
	{
	public:
		//! Constructs the compute
		HarmonicDihedralForceComputeGPU(boost::shared_ptr<ParticleData> pdata);
		//! Destructor
		~HarmonicDihedralForceComputeGPU();
		
		//! Sets the block size to run on the device
		/*! \param block_size Block size to set
		*/
		void setBlockSize(int block_size) { m_block_size = block_size; }
		
		//! Set the parameters
		virtual void setParams(unsigned int type, Scalar K, int sign, unsigned int multiplicity);
		
	protected:
		int m_block_size;		//!< Block size to run calculation on
		//vector<float2 *> m_gpu_params;	//!< Parameters stored on the GPU
		//float2 *m_host_params;	//!< Host parameters
		vector<float4 *> m_gpu_params;	//!< Parameters stored on the GPU (k,sign,m)
		float4 *m_host_params;	//!< Host parameters -- padded to float4 due to a problem with reading float 3s from texture cashe

		//! Actually compute the forces
		virtual void computeForces(unsigned int timestep);
	};
	
//! Export the DihedralForceComputeGPU class to python
void export_HarmonicDihedralForceComputeGPU();

#endif

#ifdef WIN32
#pragma warning( pop )
#endif

