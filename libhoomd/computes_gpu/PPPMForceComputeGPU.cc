/*
Highly Optimized Object-oriented Many-particle Dynamics -- Blue Edition
(HOOMD-blue) Open Source Software License Copyright 2008-2011 Ames Laboratory
Iowa State University and The Regents of the University of Michigan All rights
reserved.

HOOMD-blue may contain modifications ("Contributions") provided, and to which
copyright is held, by various Contributors who have granted The Regents of the
University of Michigan the right to modify and/or distribute such Contributions.

You may redistribute, use, and create derivate works of HOOMD-blue, in source
and binary forms, provided you abide by the following conditions:

* Redistributions of source code must retain the above copyright notice, this
list of conditions, and the following disclaimer both in the code and
prominently in any materials provided with the distribution.

* Redistributions in binary form must reproduce the above copyright notice, this
list of conditions, and the following disclaimer in the documentation and/or
other materials provided with the distribution.

* All publications and presentations based on HOOMD-blue, including any reports
or published results obtained, in whole or in part, with HOOMD-blue, will
acknowledge its use according to the terms posted at the time of submission on:
http://codeblue.umich.edu/hoomd-blue/citations.html

* Any electronic documents citing HOOMD-Blue will link to the HOOMD-Blue website:
http://codeblue.umich.edu/hoomd-blue/

* Apart from the above required attributions, neither the name of the copyright
holder nor the names of HOOMD-blue's contributors may be used to endorse or
promote products derived from this software without specific prior written
permission.

Disclaimer

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDER AND CONTRIBUTORS ``AS IS'' AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, AND/OR ANY
WARRANTIES THAT THIS SOFTWARE IS FREE OF INFRINGEMENT ARE DISCLAIMED.

IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT,
INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE
OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF
ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

// Maintainer: sbarr

/*! \file PPPMForceComputeGPU.cc
    \brief Defines the PPPMForceComputeGPU class
*/

#include "PotentialPair.h"
#include "PPPMForceComputeGPU.h"

#include <boost/python.hpp>
using namespace boost::python;

#include <boost/bind.hpp>
using namespace boost;

using namespace std;

/*! \param sysdef System to compute bond forces on
    \param nlist Neighbor list
    \param group Particle group

*/
PPPMForceComputeGPU::PPPMForceComputeGPU(boost::shared_ptr<SystemDefinition> sysdef,
                                         boost::shared_ptr<NeighborList> nlist,
                                         boost::shared_ptr<ParticleGroup> group)
    : PPPMForceCompute(sysdef, nlist, group), m_block_size(256)
    {

    // can't run on the GPU if there aren't any GPUs in the execution configuration
    if (!exec_conf->isCUDAEnabled())
        {
        m_exec_conf->msg->error() << "Creating a PPMForceComputeGPU with no GPU in the execution configuration" << endl;
        throw std::runtime_error("Error initializing PPMForceComputeGPU");
        }
    CHECK_CUDA_ERROR();
    }

PPPMForceComputeGPU::~PPPMForceComputeGPU()
    {
    }

/*! 
  \param Nx Number of grid points in x direction
  \param Ny Number of grid points in y direction
  \param Nz Number of grid points in z direction
  \param order Number of grid points in each direction to assign charges to
  \param kappa Screening parameter in erfc
  \param rcut Short-ranged cutoff, used for computing the relative force error

  Sets parameters for the long-ranged part of the electrostatics calculation and updates the
  parameters on the GPU.
*/
void PPPMForceComputeGPU::setParams(int Nx, int Ny, int Nz, int order, Scalar kappa, Scalar rcut)
    {
    PPPMForceCompute::setParams(Nx, Ny, Nz, order, kappa, rcut);
    cufftPlan3d(&plan, Nx, Ny, Nz, CUFFT_C2C);
    GPUArray<Scalar2> n_i_data(Nx*Ny*Nz, exec_conf);
    m_pppm_data.i_data.swap(n_i_data);
    GPUArray<Scalar2> n_o_data(Nx*Ny*Nz, exec_conf);
    m_pppm_data.o_data.swap(n_o_data);
    }



/*! Internal method for computing the forces on the GPU.
  \post The force data on the GPU is written with the calculated forces

  \param timestep Current time step of the simulation

  Calls gpu_compute_harmonic_bond_forces to do the dirty work.
*/
void PPPMForceComputeGPU::computeForces(unsigned int timestep)
    {
    if (!m_params_set)
        {
        m_exec_conf->msg->error() << "charge.pppm: setParams must be called prior to computeForces()" << endl;
        throw std::runtime_error("Error computing forces in PPPMForceComputeGPU");
        }
    
    unsigned int group_size = m_group->getNumMembers();
    // just drop out if the group is an empty group
    if (group_size == 0)
        return;

    // start the profile
    if (m_prof) m_prof->push(exec_conf, "PPPM");
    
    assert(m_pdata);

    ArrayHandle<Scalar4> d_pos(m_pdata->getPositions(), access_location::device, access_mode::read);
    ArrayHandle<Scalar> d_charge(m_pdata->getCharges(), access_location::device, access_mode::read);

    BoxDim box = m_pdata->getBox();
    ArrayHandle<cufftComplex> d_rho_real_space(m_pppm_data.m_rho_real_space, access_location::device, access_mode::readwrite);
    ArrayHandle<cufftComplex> d_Ex(m_Ex, access_location::device, access_mode::readwrite);
    ArrayHandle<cufftComplex> d_Ey(m_Ey, access_location::device, access_mode::readwrite);
    ArrayHandle<cufftComplex> d_Ez(m_Ez, access_location::device, access_mode::readwrite);
    ArrayHandle<Scalar3> d_kvec(m_kvec, access_location::device, access_mode::readwrite);
    ArrayHandle<Scalar> d_green_hat(m_pppm_data.m_green_hat, access_location::device, access_mode::readwrite);
    ArrayHandle<Scalar> h_rho_coeff(m_rho_coeff, access_location::host, access_mode::readwrite);
    ArrayHandle<Scalar3> d_field(m_field, access_location::device, access_mode::readwrite);

    ArrayHandle<Scalar4> d_force(m_force,access_location::device,access_mode::overwrite);
    ArrayHandle<Scalar> d_virial(m_virial,access_location::device,access_mode::overwrite);

    // access the group
    ArrayHandle< unsigned int > d_index_array(m_group->getIndexArray(), access_location::device, access_mode::read);

    if(m_box_changed) {
        Scalar3 L = box.getL();
        Scalar temp = floor(((m_kappa*L.x/(M_PI*m_Nx)) *  pow(-log(EPS_HOC),0.25)));
        int nbx = (int)temp;
        temp = floor(((m_kappa*L.y/(M_PI*m_Ny)) * pow(-log(EPS_HOC),0.25)));
        int nby = (int)temp;
        temp =  floor(((m_kappa*L.z/(M_PI*m_Nz)) *  pow(-log(EPS_HOC),0.25)));
        int nbz = (int)temp;

        ArrayHandle<Scalar3> d_vg(m_pppm_data.m_vg, access_location::device, access_mode::readwrite);;
        ArrayHandle<Scalar> d_gf_b(m_gf_b, access_location::device, access_mode::readwrite);

        reset_kvec_green_hat(box,
                             m_Nx,
                             m_Ny,
                             m_Nz,
                             nbx,
                             nby,
                             nbz,
                             m_order,
                             m_kappa,
                             d_kvec.data,
                             d_green_hat.data,
                             d_vg.data,
                             d_gf_b.data,
                             m_block_size);


        Scalar scale = 1.0f/((Scalar)(m_Nx * m_Ny * m_Nz));
        m_energy_virial_factor = 0.5 * L.x * L.y * L.z * scale * scale;
        m_box_changed = false;
        }

    // run the kernel in parallel on all GPUs

    gpu_compute_pppm_forces(d_force.data,
                            d_virial.data,
                            m_virial.getPitch(),
                            m_pdata->getN(),
                            d_pos.data,
                            d_charge.data,
                            box,
                            m_Nx,
                            m_Ny,
                            m_Nz,
                            m_order,
                            h_rho_coeff.data,
                            d_rho_real_space.data,
                            plan,
                            d_Ex.data,
                            d_Ey.data,
                            d_Ez.data,
                            d_kvec.data,
                            d_green_hat.data,
                            d_field.data,            
                            d_index_array.data,
                            group_size,
                            m_block_size);

    if (exec_conf->isCUDAErrorCheckingEnabled())
        CHECK_CUDA_ERROR();

    // If there are exclusions, correct for the long-range part of the potential
    if( m_nlist->getExclusionsSet()) 
        {
        ArrayHandle<unsigned int> d_exlist(m_nlist->getExListArray(), access_location::device, access_mode::read);
        ArrayHandle<unsigned int> d_n_ex(m_nlist->getNExArray(), access_location::device, access_mode::read);
        Index2D nex = m_nlist->getExListIndexer();

        fix_exclusions(d_force.data,
                       d_virial.data,
                       m_virial.getPitch(),
                       m_pdata->getN(),
                       d_pos.data,
                       d_charge.data,
                       box,
                       d_n_ex.data,
                       d_exlist.data,
                       nex,
                       m_kappa,
                       d_index_array.data,
                       group_size,
                       m_block_size);
        if (exec_conf->isCUDAErrorCheckingEnabled())
            CHECK_CUDA_ERROR();
        }



    //   int64_t mem_transfer = m_pdata->getN() * 4+16+20 + m_bond_data->getNumBonds() * 2 * (8+16+8);
    //    int64_t flops = m_bond_data->getNumBonds() * 2 * (3+12+16+3+7);
    if (m_prof) m_prof->pop(exec_conf, 1, 1);
    }


/*! Computes the additional energy and virial contributed by PPPM
    \note The additional terms are simply added onto particle 0 so that they will be accounted for by
    ComputeThermo
*/
void PPPMForceComputeGPU::fix_thermo_quantities()
    {
    BoxDim box = m_pdata->getBox();

    // access data arrays
    ArrayHandle<cufftComplex> d_rho_real_space(m_pppm_data.m_rho_real_space, access_location::device, access_mode::readwrite);
    ArrayHandle<Scalar> d_green_hat(m_pppm_data.m_green_hat, access_location::device, access_mode::readwrite);
    ArrayHandle<Scalar3> d_vg(m_pppm_data.m_vg, access_location::device, access_mode::readwrite);
    ArrayHandle<Scalar2> d_i_data(m_pppm_data.i_data, access_location::device, access_mode::readwrite);
    ArrayHandle<Scalar2> d_o_data(m_pppm_data.o_data, access_location::device, access_mode::readwrite);

    // compute correction
    Scalar2 pppm_virial_energy =  gpu_compute_pppm_thermo(m_Nx,
                                                          m_Ny,
                                                          m_Nz,
                                                          d_rho_real_space.data,
                                                          d_vg.data,
                                                          d_green_hat.data,
                                                          d_o_data.data,
                                                          d_i_data.data,
                                                          256);

    Scalar3 L = box.getL();
    pppm_virial_energy.x *= m_energy_virial_factor/ (3.0f * L.x * L.y * L.z);
    pppm_virial_energy.y *= m_energy_virial_factor;
    pppm_virial_energy.y -= m_q2 * m_kappa / 1.772453850905516027298168f;
    pppm_virial_energy.y -= 0.5*M_PI*m_q*m_q / (m_kappa*m_kappa* L.x * L.y * L.z);

    // apply the correction to particle 0
    ArrayHandle<Scalar4> h_force(m_force,access_location::host, access_mode::readwrite);
    ArrayHandle<Scalar> h_virial(m_virial,access_location::host, access_mode::readwrite);
    h_force.data[0].w += pppm_virial_energy.y;

    // TODO: This adds a pressure. It should add a correction to the virial instead
    h_virial.data[0] += pppm_virial_energy.x;

    // TODO: Compute full virial tensor
    // unsigned int virial_pitch = net_virial.getPitch();
    //h_virial.data[0*virial_pitch+0] += xx virial;
    //h_virial.data[1*virial_pitch+0] += xy virial;
    //h_virial.data[2*virial_pitch+0] += xz virial;
    //h_virial.data[3*virial_pitch+0] += yy virial;
    //h_virial.data[4*virial_pitch+0] += yz virial;
    //h_virial.data[5*virial_pitch+0] += zz virial;
    }

void export_PPPMForceComputeGPU()
    {
    class_<PPPMForceComputeGPU, boost::shared_ptr<PPPMForceComputeGPU>, bases<PPPMForceCompute>, boost::noncopyable >
        ("PPPMForceComputeGPU", init< boost::shared_ptr<SystemDefinition>, 
         boost::shared_ptr<NeighborList>,
         boost::shared_ptr<ParticleGroup> >())
        .def("setBlockSize", &PPPMForceComputeGPU::setBlockSize)
        ;
    }
