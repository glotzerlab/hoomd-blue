// Copyright (c) 2009-2017 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.


// Maintainer: akohlmey

/*! \file PCNDForceComputeGPU.cc
    \brief Defines the PCNDForceComputeGPU class
*/


#include "PCNDForceComputeGPU.h"

#include <stdexcept>

using namespace std;

namespace hoomd
    {
namespace md
    {
/*! \param sysdef System to compute forces on
    \param nlist Neighborlist to use for computing the forces
    \param r_cut Cuttoff radius beyond which the force is 0

    \post memory is allocated and all parameters ljX are set to 0.0

    \note The PCNDForceComputeGPU does not own the Neighborlist, the caller should
    delete the neighborlist when done.
*/
PCNDForceComputeGPU::PCNDForceComputeGPU(std::shared_ptr<SystemDefinition> sysdef,
                                         std::shared_ptr<ParticleGroup> group,  
		                         std::shared_ptr<NeighborList> nlist,
                                         Scalar r_cut)
    : PCNDForceCompute(sysdef, group, nlist, r_cut)
    {
    // can't run on the GPU if there aren't any GPUs in the execution configuration
    if (!m_exec_conf->isCUDAEnabled())
        {
        m_exec_conf->msg->error() << "Creating a PCNDForceComputeGPU with no GPU in the execution configuration" << endl;
        throw std::runtime_error("Error initializing PCNDForceComputeGPU");
        }

    // allocate the coeff data on the CPU
    GPUArray<Scalar4> coeffs(m_pdata->getNTypes()*m_pdata->getNTypes(),m_exec_conf);
    m_coeffs.swap(coeffs);
    }


PCNDForceComputeGPU::~PCNDForceComputeGPU()
    {
    }

/*! \param block_size Size of the block to run on the device
    Performance of the code may be dependant on the block size run
    on the GPU. \a block_size should be set to be a multiple of 32.
*/
void PCNDForceComputeGPU::setBlockSize(int block_size)
    {
    m_block_size = block_size;
    }

/*! \post The parameters \a lj12 through \a lj4 are set for the pairs \a typ1, \a typ2 and \a typ2, \a typ1.
    \note \a lj? are low level parameters used in the calculation. In order to specify
    these for a 12-4 and 9-6 lennard jones formula (with alpha), they should be set to the following.

        12-4
    - \a lj12 = 2.598076 * epsilon * pow(sigma,12.0)
    - \a lj9 = 0.0
    - \a lj6 = 0.0
    - \a lj4 = -alpha * 2.598076 * epsilon * pow(sigma,4.0)

        9-6
    - \a lj12 = 0.0
    - \a lj9 = 6.75 * epsilon * pow(sigma,9.0);
    - \a lj6 = -alpha * 6.75 * epsilon * pow(sigma,6.0)
    - \a lj4 = 0.0

       12-6
    - \a lj12 = 4.0 * epsilon * pow(sigma,12.0)
    - \a lj9 = 0.0
    - \a lj6 = -alpha * 4.0 * epsilon * pow(sigma,4.0)
    - \a lj4 = 0.0

    Setting the parameters for typ1,typ2 automatically sets the same parameters for typ2,typ1: there
    is no need to call this funciton for symmetric pairs. Any pairs that this function is not called
    for will have lj12 through lj4 set to 0.0.

    \param typ1 Specifies one type of the pair
    \param typ2 Specifies the second type of the pair
    \param lj12 1/r^12 term
    \param lj9  1/r^9 term
    \param lj6  1/r^6 term
    \param lj4  1/r^4 term
*/
void PCNDForceComputeGPU::setParams(unsigned int typ1, unsigned int typ2, Scalar lj12, Scalar lj9, Scalar lj6, Scalar lj4)
    {
    if (typ1 >= m_ntypes || typ2 >= m_ntypes)
        {
        m_exec_conf->msg->error() << "pair.pcnd: Trying to set params for a non existent type! " << typ1 << "," << typ2 << endl;
        throw runtime_error("PCNDForceComputeGpu::setParams argument error");
        }

    ArrayHandle<Scalar4> h_coeffs(m_coeffs, access_location::host, access_mode::readwrite);
    // set coeffs in both symmetric positions in the matrix
    h_coeffs.data[typ1*m_pdata->getNTypes() + typ2] = make_scalar4(lj12, lj9, lj6, lj4);
    h_coeffs.data[typ2*m_pdata->getNTypes() + typ1] = make_scalar4(lj12, lj9, lj6, lj4);
    }

/*! \post The PCND forces are computed for the given timestep on the GPU.
    The neighborlist's compute method is called to ensure that it is up to date
    before forces are computed.
    \param timestep Current time step of the simulation

    Calls gpu_compute_pcnd_forces to do the dirty work.
*/
void PCNDForceComputeGPU::computeForces(uint64_t timestep)
    {
    // start by updating the neighborlist
    m_nlist->compute(timestep);

    // start the profile
    if (m_prof) m_prof->push(m_exec_conf, "PCND pair");

    // The GPU implementation CANNOT handle a half neighborlist, error out now
    bool third_law = m_nlist->getStorageMode() == NeighborList::half;
    if (third_law)
        {
        m_exec_conf->msg->error() << "PCNDForceComputeGPU cannot handle a half neighborlist" << endl;
        throw runtime_error("Error computing forces in PCNDForceComputeGPU");
        }

    // access the neighbor list, which just selects the neighborlist into the device's memory, copying
    // it there if needed
    ArrayHandle<unsigned int> d_n_neigh(this->m_nlist->getNNeighArray(), access_location::device, access_mode::read);
    ArrayHandle<unsigned int> d_nlist(this->m_nlist->getNListArray(), access_location::device, access_mode::read);
    ArrayHandle<size_t> d_head_list(this->m_nlist->getHeadList(), access_location::device, access_mode::read);

    ArrayHandle<Scalar4> d_coeffs(m_coeffs, access_location::device, access_mode::read);

    // access the particle data
    ArrayHandle<Scalar4> d_pos(m_pdata->getPositions(), access_location::device, access_mode::read);
    BoxDim box = m_pdata->getBox();
    unsigned int group_size = m_group->getNumMembers();
    ArrayHandle<Scalar4> d_force(m_force,access_location::device,access_mode::overwrite);
    ArrayHandle<Scalar> d_virial(m_virial,access_location::device,access_mode::overwrite);

    // run the kernel on all GPUs in parallel
    kernel::gpu_compute_pcnd_forces(group_size,
		             d_force.data,
                             d_virial.data,
                             m_virial.getPitch(),
                             d_pos.data,
                             box,
                             d_n_neigh.data,
                             d_nlist.data,
                             d_head_list.data,
                             d_coeffs.data,
                             //this->m_nlist->getNListArray().getPitch(),
                             //m_pdata->getNTypes(),
                             m_r_cut * m_r_cut,
                             m_pdata-> getNTypes(),
			     this->m_nlist->getNListArray().getPitch(),
			     m_block_size,
                             m_exec_conf->dev_prop.warpSize);
    if (m_exec_conf->isCUDAErrorCheckingEnabled())
        CHECK_CUDA_ERROR();

    Scalar avg_neigh = m_nlist->estimateNNeigh();
    int64_t n_calc = int64_t(avg_neigh * m_pdata->getN());
    int64_t mem_transfer = m_pdata->getN() * (4 + 16 + 20) + n_calc * (4 + 16);
    int64_t flops = n_calc * (3+12+5+2+3+11+3+8+7);
    if (m_prof) m_prof->pop(m_exec_conf, flops, mem_transfer);
    }

namespace detail
    {
void export_PCNDForceComputeGPU(pybind11::module& m)
    {
    pybind11::class_<PCNDForceComputeGPU, std::shared_ptr<PCNDForceComputeGPU> >(m, "PCNDForceComputeGPU", pybind11::base<PCNDForceCompute>())
    .def(pybind11::init< std::shared_ptr<SystemDefinition>, std::shared_ptr<ParticleGroup>, std::shared_ptr<NeighborList>, Scalar >())
    .def("setBlockSize", &PCNDForceComputeGPU::setBlockSize)
    ;
    }

    } // end namespace detail
    } // end namespace md
    } // end namespace hoomd
