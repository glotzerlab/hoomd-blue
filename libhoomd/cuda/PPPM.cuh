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
    int electrostatics_allocation_bool;  //!< This is used to see if we need to allocate arrays for electrostatics, is set to false after allocation has been done
    cufftHandle plan;                    //!< Used for the Fast Fourier Transformations performed on the GPU
    cufftComplex *GPU_rho_real_space; 	 //!< Used to transform the grid based charge density
    float* GPU_green_hat;                //!< Modified Hockney-Eastwood Green's function
    float3* GPU_k_vec;                   //!< k-vectors for each grid point
    cufftComplex* GPU_E_x;               //!< x component of the grid based electric field
    cufftComplex* GPU_E_y;               //!< y component of the grid based electric field
    cufftComplex* GPU_E_z;               //!< z component of the grid based electric field
    float3* GPU_field; 			 //!< Put field components in one array after back-transform
    int Nx;                              //!< Number of grid points in the x direction
    int Ny;                              //!< Number of grid points in the y direction 
    int Nz;                              //!< Number of grid points in the z direction 
    int interpolation_order;             //!< Number of grid points in each direction to interpolate charge over
    float r_cutoff;                      //!< Real-Space cutoff for the electrostatics calculation
    float kappa;                         //!< Screening parameter for the erfc in the real-space potential
    float* CPU_rho_coeff;                //!< Coefficients for computing the grid based charge density
    float2* cuda_thermo_quantities;      //!< Store the Fourier space contribution to the pressure and energy 
    float *gf_b;                         //!< Used to compute the grid based Green's function
    float3* vg;                          //!< Coefficients for Fourier space virial calculation
    float CPU_energy_virial_factor;      //!< Scale factor for Fourier space energy and pressure calculation
    float q2;                            //!< Sum(q_i*q_i), where q_i is the charge of each particle
    float2* o_data;                      //!< Used to quickly sum grid points for pressure and energy calcuation (output)
    float2* i_data;                      //!< Used to quickly sum grid points for pressure and energy calcuation (input)
};

int get_virial_flag_value();
void print_bool_value();
float2 calculate_thermo_quantities(const gpu_pdata_arrays &pdata, const gpu_boxsize &box);
void electrostatics_calculation(const gpu_force_data_arrays& force_data, const gpu_pdata_arrays &pdata, const gpu_boxsize &box, const float3 *d_params, const float *d_rcutsq);

// void charge_density_assignment(const gpu_pdata_arrays &pdata, const gpu_boxsize &box);

#endif
