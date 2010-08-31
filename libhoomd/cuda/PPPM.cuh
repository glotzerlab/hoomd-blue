#include <cufft.h>

#include "HOOMDMath.h"
#include "ForceCompute.cuh"
#include "NeighborList.cuh"
#include "ParticleData.cuh"
#include "Index1D.h"

#ifndef __ELECTROSTATICS_H__
#define __ELECTROSTATICS_H__


struct electrostatics_data
{
 	//electrostatics stuff:	
	int electrostatics_allocation_bool; 
	cufftHandle plan; 
	cufftComplex *GPU_rho_real_space; 	//also used for transformed rho values
// 	cufftComplex *GPU_rho_k_space; 
	float* GPU_green_hat;
	float3* GPU_k_vec;
	cufftComplex* GPU_E_x; 
	cufftComplex* GPU_E_y;
	cufftComplex* GPU_E_z;
	float3* GPU_field; 			//put field components in one array after back-transform
// 	float4* Particle_forces; 		
// 	cufftComplex *CPU_rho_real_space;
	int Nx, Ny, Nz, interpolation_order;
	float r_cutoff;
	float kappa;
	float* CPU_rho_coeff;
	float3* cuda_thermo_quantities;
	float3* vg; //for k-space virials
	float CPU_energy_virial_factor;
	int show_virial_flag;
	
// 	//just for CPU charge assignment:
// 	float4* CPU_particles;
// 	float* CPU_charges;
// 	cufftComplex* CPU_rho;
};

int get_virial_flag_value();
void print_bool_value();
float3 calculate_thermo_quantities(const gpu_pdata_arrays &pdata, const gpu_boxsize &box);
void electrostatics_calculation(const gpu_force_data_arrays& force_data, const gpu_pdata_arrays &pdata, const gpu_boxsize &box, float3 *d_params, float *d_rcutsq);

// void charge_density_assignment(const gpu_pdata_arrays &pdata, const gpu_boxsize &box);

#endif