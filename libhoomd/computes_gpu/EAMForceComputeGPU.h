/**
powered by:
Moscow group.
*/

#include "EAMForceCompute.h"
#include "NeighborList.h"
#include "EAMForceGPU.cuh"

#include <boost/shared_ptr.hpp>

/*! \file EAMForceComputeGPU.h
	\brief Declares the class EAMForceComputeGPU
*/

#ifndef __EAMFORCECOMPUTEGPU_H__
#define __EAMFORCECOMPUTEGPU_H__

//! Computes EAM forces on each particle using the GPU
/*! Calculates the same forces as EAMForceCompute, but on the GPU by using global memory.

*/
class EAMForceComputeGPU : public EAMForceCompute
	{
	public:
		//! Constructs the compute
		EAMForceComputeGPU(boost::shared_ptr<SystemDefinition> sysdef, char *filename, int type_of_file);

		//! Destructor
		virtual ~EAMForceComputeGPU();

		//! Set the parameters for a single type pair

		//! Sets the block size to run at
		void setBlockSize(int block_size);

	protected:
		EAMData eam_data;

		vector<Scalar *> d_electronDensity; //!<array rho(r)
		vector<Scalar *> d_pairPotential; //!<array Z(r)
		vector<Scalar *> d_embeddingFunction; //!<array F(rho)

		vector<Scalar *> d_derivativeElectronDensity; //!<array rho'(r)
		vector<Scalar *> d_derivativePairPotential; //!<array Z'(r)
		vector<Scalar *> d_derivativeEmbeddingFunction; //!<array F'(rho)
		vector<Scalar *> d_atomDerivativeEmbeddingFunction; //!<array F'(rho)
		vector<float2 *> d_coeffs;		//!< Pointer to the coefficients on the GPU
		float2 * h_coeffs;				//!< Pointer to the coefficients on the host
		int m_block_size;				//!< The block size to run on the GPU
		bool m_ulf_workaround;			//!< Stores decision made by the constructor whether to enable the ULF workaround

		//! Actually compute the forces
		virtual void computeForces(unsigned int timestep);
	};

//! Exports the EAMForceComputeGPU class to python
void export_EAMForceComputeGPU();

#endif
