/**
powered by:
Moscow group.
*/


#include "EAMForceCompute.h"
#include "NeighborList.h"
#include "EAMTexForceGPU.cuh"

#include <boost/shared_ptr.hpp>

/*! \file EAMTexForceComputeGPU.h
	\brief Declares the class EAMTexForceComputeGPU
*/

#ifndef __EAMTexForceComputeGPU_H__
#define __EAMTexForceComputeGPU_H__

//! Computes EAM forces on each particle using the GPU
/*! Calculates the same forces as EAMForceCompute, but on the GPU by using texture memory(Linear).
	
	The GPU kernel for calculating the forces is in ljforcesum_kernel.cu.
	\ingroup computes
*/
class EAMTexForceComputeGPU : public EAMForceCompute
	{
	public:
		//! Constructs the compute
		EAMTexForceComputeGPU(boost::shared_ptr<ParticleData> pdata, boost::shared_ptr<NeighborList> nlist, Scalar r_cut, char *filename);
		
		//! Destructor
		virtual ~EAMTexForceComputeGPU();
		
		
		//! Sets the block size to run at
		void setBlockSize(int block_size);

	protected:
		EAMTexData eam_data;
		vector<EAMLinear> eam_linear_data;
		vector<Scalar *> d_atomDerivativeEmbeddingFunction; //!<array F'(rho) for each particle
		
		vector<float2 *> d_coeffs;		//!< Pointer to the coefficients on the GPU
		float2 * h_coeffs;				//!< Pointer to the coefficients on the host
		int m_block_size;				//!< The block size to run on the GPU
		bool m_ulf_workaround;			//!< Stores decision made by the constructor whether to enable the ULF workaround

		//! Actually compute the forces
		virtual void computeForces(unsigned int timestep);
	};
	
//! Exports the EAMTexForceComputeGPU class to python
void export_EAMTexForceComputeGPU();

#endif
