

/**
powered by:
Moscow group.
*/

#include "EAMForceCompute.h"
#include "NeighborList.h"
#include "EAMTexInterForceGPU.cuh"

#include <boost/shared_ptr.hpp>

/*! \file EAMTexInterForceComputeGPU.h
	\brief Declares the class EAMTexInterForceComputeGPU
*/

#ifndef __EAMTexInterForceComputeGPU_H__
#define __EAMTexInterForceComputeGPU_H__

//! Computes Lennard-Jones forces on each particle using the GPU
/*! Calculates the same forces as EAMForceCompute, but on the GPU by using texture memory(cudaArray) with hardware interpolation.
*/
class EAMTexInterForceComputeGPU : public EAMForceCompute
	{
	public:
		//! Constructs the compute
		EAMTexInterForceComputeGPU(boost::shared_ptr<SystemDefinition> sysdef, char *filename, int type_of_file);
		
		//! Destructor
		virtual ~EAMTexInterForceComputeGPU();
		
		
		//! Sets the block size to run at
		void setBlockSize(int block_size);

	protected:
		EAMTexInterData eam_data;
		vector<EAMtex> eam_tex_data;
		vector<Scalar *> d_atomDerivativeEmbeddingFunction; //!<array F'(rho) for each particle
		vector<float> a;
		vector<float *> b;
		
		vector<float2 *> d_coeffs;		//!< Pointer to the coefficients on the GPU
		float2 * h_coeffs;				//!< Pointer to the coefficients on the host
		int m_block_size;				//!< The block size to run on the GPU
		bool m_ulf_workaround;			//!< Stores decision made by the constructor whether to enable the ULF workaround

		//! Actually compute the forces
		virtual void computeForces(unsigned int timestep);
	};
	
//! Exports the EAMTexInterForceComputeGPU class to python
void export_EAMTexInterForceComputeGPU();

#endif
