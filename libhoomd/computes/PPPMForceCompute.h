/*
Highly Optimized Object-oriented Many-particle Dynamics -- Blue Edition
(HOOMD-blue) Open Source Software License Copyright 2008, 2009 Ames Laboratory
Iowa State University and The Regents of the University of Michigan All rights
reserved.

HOOMD-blue may contain modifications ("Contributions") provided, and to which
copyright is held, by various Contributors who have granted The Regents of the
University of Michigan the right to modify and/or distribute such Contributions.

Redistribution and use of HOOMD-blue, in source and binary forms, with or
without modification, are permitted, provided that the following conditions are
met:

* Redistributions of source code must retain the above copyright notice, this
list of conditions, and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright notice, this
list of conditions, and the following disclaimer in the documentation and/or
other materials provided with the distribution.

* Neither the name of the copyright holder nor the names of HOOMD-blue's
contributors may be used to endorse or promote products derived from this
software without specific prior written permission.

Disclaimer

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDER AND CONTRIBUTORS ``AS IS''
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, AND/OR
ANY WARRANTIES THAT THIS SOFTWARE IS FREE OF INFRINGEMENT ARE DISCLAIMED.

IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT,
INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE
OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF
ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

// Maintainer: sbarr

#include <boost/shared_ptr.hpp>
#include <boost/signals.hpp>

#include "ForceCompute.h"
#include "NeighborList.h"
#include "ParticleGroup.h"
#include "kiss_fftnd.h"

#include <vector>

#ifdef ENABLE_CUDA
#include <cufft.h>
#endif

/*! \file PPPMForceCompute.h
    \brief Declares a class for computing harmonic bonds
*/

#ifndef __PPPMFORCECOMPUTE_H__
#define __PPPMFORCECOMPUTE_H__

//! MAX gives the larger of two values
#define MAX(a,b) ((a) > (b) ? (a) : (b))
//! MIN gives the lesser of two values
#define MIN(a,b) ((a) < (b) ? (a) : (b))
//! MaxOrder is the largest allowed value of the interpolation order
#define MaxOrder 7
//! ConstSize is used to make sure the rho_coeff will fit into memory on the GPU
#define CONSTANT_SIZE 2048
//! EPS_HOC is used to calculate the Green's function
#define EPS_HOC 1.0e-7

//! Computes the long ranged part of the electrostatic forces on each particle
/*! PPPM forces are computed on every particle in the simulation.

*/
class PPPMForceCompute : public ForceCompute
    {
    public:
        //! Constructs the compute
        PPPMForceCompute(boost::shared_ptr<SystemDefinition> sysdef,
                         boost::shared_ptr<NeighborList> nlist,
                         boost::shared_ptr<ParticleGroup> group);
       
        //! Destructor
        ~PPPMForceCompute();
        
        //! Set the parameters
        virtual void setParams(int Nx, int Ny, int Nz, int order, Scalar kappa, Scalar rcut);
        
        //! Returns a list of log quantities this compute calculates
        virtual std::vector< std::string > getProvidedLogQuantities();
        
        //! Calculates the requested log value and returns it
        virtual Scalar getLogValue(const std::string& quantity, unsigned int timestep);
        
        //! Notification of a box size change
        void slotBoxChanged()
            {
            m_box_changed = true;
            }

        //! root mean square error in force calculation
        Scalar rms(Scalar h, Scalar prd, Scalar natoms);
        //! computes coefficients for assigning charges to grid points
        void compute_rho_coeff();
        //! computes coefficients for the Green's function
        void compute_gf_denom();
        //! computes coefficients for the Green's function
        Scalar gf_denom(Scalar x, Scalar y, Scalar z);
        //! resets kvec, Green's function and virial coefficients if the box size changes
        void reset_kvec_green_hat_cpu();
        //! assigns charges to grid points
        void assign_charges_to_grid();
        //! multiply Green's function by charge density to get electric field
        void combined_green_e();
        //! Do the final force calculation
        void calculate_forces();
        //! fix the force due to excluded particles
        void fix_exclusions_cpu();
    protected:
        bool m_params_set;                       //!< Set to true when the parameters are set
        int m_Nx;                                //!< Number of grid points in x direction
        int m_Ny;                                //!< Number of grid points in y direction
        int m_Nz;                                //!< Number of grid points in z direction
        int m_order;                             //!< Interpolation order
        Scalar m_kappa;                          //!< screening parameter for erfc(kappa*r)
        Scalar m_rcut;                           //!< Real space cutoff
        Scalar2 thermo_quantites;                //!< Store the Fourier space contribution to the pressure and energy 
        Scalar m_q;                              //!< Total system charge
        Scalar m_q2;                             //!< Sum(q_i*q_i), where q_i is the charge of each particle
        Scalar m_energy_virial_factor;           //!< Multiplication factor for energy and virial
        bool m_box_changed;                      //!< Set to ttrue when the box size has changed
        GPUArray<Scalar3> m_kvec;                //!< k-vectors for each grid point
        GPUArray<cufftComplex> m_Ex;             //!< x component of the grid based electric field
        GPUArray<cufftComplex> m_Ey;             //!< y component of the grid based electric field
        GPUArray<cufftComplex> m_Ez;             //!< z component of the grid based electric field
        GPUArray<Scalar3>m_field;                //!< grid based Electric field, combined
        GPUArray<Scalar> m_rho_coeff;            //!< Coefficients for computing the grid based charge density
        GPUArray<Scalar> m_gf_b;                 //!< Used to compute the grid based Green's function
        boost::signals::connection m_boxchange_connection;   //!< Connection to the ParticleData box size change signal
        boost::shared_ptr<NeighborList> m_nlist;  //!< The neighborlist to use for the computation
        boost::shared_ptr<ParticleGroup> m_group; //!< Group to compute properties for

        kiss_fft_cpx *fft_in;                     //!< For FFTs on CPU rho_real_space
        kiss_fft_cpx *fft_ex;                     //!< For FFTs on CPU E-field x component
        kiss_fft_cpx *fft_ey;                     //!< For FFTs on CPU E-field y component
        kiss_fft_cpx *fft_ez;                     //!< For FFTs on CPU E-field z component
        kiss_fftnd_cfg fft_forward;               //!< Forward FFT on CPU
        kiss_fftnd_cfg fft_inverse;               //!< Inverse FFT on CPU

        int first_run;                            //!< flag for allocating arrays

        //! Actually compute the forces
        virtual void computeForces(unsigned int timestep);
    };


//! Exports the PPPMForceCompute class to python
void export_PPPMForceCompute();

//! Holds some data for computing the thermodynamic output
class PPPMData
    {
    public:
        static int compute_pppm_flag;                   //!< Flag to see if we should compute the PPPM thermodynamic properties
        static int Nx;                                  //!< Number of grid points in x direction
        static int Ny;                                  //!< Number of grid points in y direction
        static int Nz;                                  //!< Number of grid points in z direction
        static Scalar q2;                               //!< Sum(q_i*q_i), where q_i is the charge of each particle
        static Scalar q;                                //!< Sum(q_i*q_i), where q_i is the charge of each particle
        static Scalar kappa;                            //!< screening parameter for erfc(kappa*r)
        static Scalar energy_virial_factor;             //!< Multiplication factor for energy and virial
        static Scalar pppm_energy;                      //!< Used for logging the energy
        static GPUArray<cufftComplex> m_rho_real_space; //!< x component of the grid based electric field
        static GPUArray<Scalar> m_green_hat;            //!< Modified Hockney-Eastwood Green's function
        static GPUArray<Scalar3> m_vg;                  //!< Coefficients for Fourier space virial calculation
        static GPUArray<Scalar2> o_data;                //!< Used to quickly sum grid points for pressure and energy calcuation (output)
        static GPUArray<Scalar2> i_data;                //!< Used to quickly sum grid points for pressure and energy calcuation (input)
    };


#endif

