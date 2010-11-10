#include <cufft.h>

#include "HOOMDMath.h"
#include "ForceCompute.cuh"
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
	float* GPU_green_hat;
	float3* GPU_k_vec;
	cufftComplex* GPU_E_x; 
	cufftComplex* GPU_E_y;
	cufftComplex* GPU_E_z;
	float3* GPU_field; 			//put field components in one array after back-transform
	int Nx, Ny, Nz, interpolation_order;
	float r_cutoff;
	float kappa;
	float* CPU_rho_coeff;
	float2* cuda_thermo_quantities;
        float *gf_b;
	float3* vg; //for k-space virials
	float CPU_energy_virial_factor;
        float q2;
	int show_virial_flag;
	float2* o_data;
	float2* i_data;
};

int get_virial_flag_value();
void print_bool_value();
float2 calculate_thermo_quantities(const gpu_pdata_arrays &pdata, const gpu_boxsize &box);
void electrostatics_calculation(const gpu_force_data_arrays& force_data, const gpu_pdata_arrays &pdata, const gpu_boxsize &box, const float3 *d_params, const float *d_rcutsq);

// void charge_density_assignment(const gpu_pdata_arrays &pdata, const gpu_boxsize &box);

#endif
