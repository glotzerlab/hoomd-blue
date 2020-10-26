// Copyright (c) 2009-2019 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.


// Maintainer: jglaser

#include "ForceCompositeGPU.h"
#include "hoomd/VectorMath.h"

#include "ForceCompositeGPU.cuh"

namespace py = pybind11;

/*! \file ForceCompositeGPU.cc
    \brief Contains code for the ForceCompositeGPU class
*/

/*! \param sysdef SystemDefinition containing the ParticleData to compute forces on
*/
ForceCompositeGPU::ForceCompositeGPU(std::shared_ptr<SystemDefinition> sysdef)
        : ForceComposite(sysdef)
    {

    // power of two block sizes
    const cudaDeviceProp& dev_prop = m_exec_conf->dev_prop;
    std::vector<unsigned int> valid_params;
    unsigned int bodies_per_block = 1;
    for (unsigned int i = 0; i < 5; ++i)
        {
        bodies_per_block = 1 << i;
        unsigned int cur_block_size = 32;
        while (cur_block_size <= (unsigned int) dev_prop.maxThreadsPerBlock)
            {
            if (cur_block_size >= bodies_per_block)
                {
                valid_params.push_back(cur_block_size + bodies_per_block*10000);
                }
            cur_block_size *=2;
            }
        }

    m_tuner_force.reset(new Autotuner(valid_params, 5, 100000, "force_composite", this->m_exec_conf));
    m_tuner_virial.reset(new Autotuner(valid_params, 5, 100000, "virial_composite", this->m_exec_conf));

    // initialize autotuner
    std::vector<unsigned int> valid_params_update;
    for (unsigned int block_size = 32; block_size <= 1024; block_size += 32)
        valid_params_update.push_back(block_size);

    m_tuner_update.reset(new Autotuner(valid_params_update, 5, 100000, "update_composite", this->m_exec_conf));

    GlobalArray<uint2> flag(1, m_exec_conf);
    std::swap(m_flag, flag);

        {
        ArrayHandle<uint2> h_flag(m_flag, access_location::host, access_mode::overwrite);
        *h_flag.data = make_uint2(0,0);
        }
    }

ForceCompositeGPU::~ForceCompositeGPU()
    {}


//! Compute the forces and torques on the central particle
void ForceCompositeGPU::computeForces(unsigned int timestep)
    {
    if (m_prof)
        m_prof->push(m_exec_conf, "constrain_rigid");

    if (m_prof)
        m_prof->push(m_exec_conf, "sum force and torque");

    // access local molecule data (need to move this on top because of GPUArray scoping issues)
    const Index2D& molecule_indexer = getMoleculeIndexer();
    unsigned int nmol = molecule_indexer.getH();

    const GlobalVector<unsigned int>& molecule_list = getMoleculeList();
    const GlobalVector<unsigned int>& molecule_length = getMoleculeLengths();

    ArrayHandle<unsigned int> d_molecule_length(molecule_length, access_location::device, access_mode::read);
    ArrayHandle<unsigned int> d_molecule_list(molecule_list, access_location::device, access_mode::read);
    ArrayHandle<unsigned int> d_molecule_idx(getMoleculeIndex(), access_location::device, access_mode::read);

    // access particle data
    ArrayHandle<unsigned int> d_body(m_pdata->getBodies(), access_location::device, access_mode::read);
    ArrayHandle<Scalar4> d_postype(m_pdata->getPositions(), access_location::device, access_mode::read);
    ArrayHandle<Scalar4> d_orientation(m_pdata->getOrientationArray(), access_location::device, access_mode::read);
    ArrayHandle<unsigned int> d_tag(m_pdata->getTags(), access_location::device, access_mode::read);

    // access net force and torque acting on constituent particles
    ArrayHandle<Scalar4> d_net_force(m_pdata->getNetForce(), access_location::device, access_mode::readwrite);
    ArrayHandle<Scalar4> d_net_torque(m_pdata->getNetTorqueArray(), access_location::device, access_mode::readwrite);
    ArrayHandle<Scalar> d_net_virial(m_pdata->getNetVirial(), access_location::device, access_mode::readwrite);

    // access the force and torque array for the central ptl
    ArrayHandle<Scalar4> d_force(m_force, access_location::device, access_mode::overwrite);
    ArrayHandle<Scalar4> d_torque(m_torque, access_location::device, access_mode::overwrite);
    ArrayHandle<Scalar> d_virial(m_virial, access_location::device, access_mode::overwrite);

    // access rigid body definition
    ArrayHandle<Scalar3> d_body_pos(m_body_pos, access_location::device, access_mode::read);
    ArrayHandle<Scalar4> d_body_orientation(m_body_orientation, access_location::device, access_mode::read);
    ArrayHandle<unsigned int> d_body_len(m_body_len, access_location::device, access_mode::read);
    ArrayHandle<unsigned int> d_rigid_center(m_rigid_center, access_location::device, access_mode::read);

    PDataFlags flags = m_pdata->getFlags();
    bool compute_virial = false;
    if (flags[pdata_flag::isotropic_virial] || flags[pdata_flag::pressure_tensor])
        {
        compute_virial = true;
        }

        {
        ArrayHandle<uint2> d_flag(m_flag, access_location::device, access_mode::overwrite);

        // reset force and torque
        m_exec_conf->beginMultiGPU();

        for (int idev = m_exec_conf->getNumActiveGPUs() - 1; idev >= 0; idev--)
            {
            std::pair<unsigned int, unsigned int> range = m_pdata->getGPUPartition().getRangeAndSetGPU(idev);
            unsigned int nelem = range.second - range.first;

            if (nelem == 0)
                continue;

            cudaMemsetAsync(d_force.data+range.first, 0, sizeof(Scalar4)*nelem);
            cudaMemsetAsync(d_torque.data+range.first, 0, sizeof(Scalar4)*nelem);

            if (m_exec_conf->isCUDAErrorCheckingEnabled())
                CHECK_CUDA_ERROR();
            }
        m_exec_conf->endMultiGPU();

        m_exec_conf->beginMultiGPU();

        m_tuner_force->begin();
        unsigned int param = m_tuner_force->getParam();
        unsigned int block_size = param % 10000;
        unsigned int n_bodies_per_block = param/10000;

        // launch GPU kernel
        gpu_rigid_force(d_force.data,
                        d_torque.data,
                        d_molecule_length.data,
                        d_molecule_list.data,
                        d_molecule_idx.data,
                        d_rigid_center.data,
                        molecule_indexer,
                        d_postype.data,
                        d_orientation.data,
                        m_body_idx,
                        d_body_pos.data,
                        d_body_orientation.data,
                        d_body_len.data,
                        d_body.data,
                        d_tag.data,
                        d_flag.data,
                        d_net_force.data,
                        d_net_torque.data,
                        nmol,
                        m_pdata->getN(),
                        n_bodies_per_block,
                        block_size,
                        m_exec_conf->dev_prop,
                        !compute_virial,
                        m_gpu_partition);

        if (m_exec_conf->isCUDAErrorCheckingEnabled())
            CHECK_CUDA_ERROR();

        m_exec_conf->endMultiGPU();
        }

    uint2 flag;
        {
        ArrayHandle<uint2> h_flag(m_flag, access_location::host, access_mode::read);
        flag = *h_flag.data;
        }

    if (flag.x)
        {
        m_exec_conf->msg->errorAllRanks() << "constrain.rigid(): Composite particle with body tag " << flag.x-1
                                          << " incomplete" << std::endl << std::endl;
        throw std::runtime_error("Error computing composite particle forces.\n");
        }

    m_tuner_force->end();

    if (compute_virial)
        {
        // reset virial
        m_exec_conf->beginMultiGPU();

        for (int idev = m_exec_conf->getNumActiveGPUs() - 1; idev >= 0; idev--)
            {
            std::pair<unsigned int, unsigned int> range = m_pdata->getGPUPartition().getRangeAndSetGPU(idev);
            unsigned int nelem = range.second - range.first;

            if (nelem == 0)
                continue;

            for (unsigned int i = 0; i < 6; i++)
                {
                cudaMemsetAsync(d_virial.data+i*m_virial_pitch+range.first, 0, sizeof(Scalar)*nelem);
                }

            if (m_exec_conf->isCUDAErrorCheckingEnabled())
                CHECK_CUDA_ERROR();
            }
        m_exec_conf->endMultiGPU();

        m_exec_conf->beginMultiGPU();
        m_tuner_virial->begin();
        unsigned int param = m_tuner_virial->getParam();
        unsigned int block_size = param % 10000;
        unsigned int n_bodies_per_block = param/10000;

        // launch GPU kernel
        gpu_rigid_virial(d_virial.data,
                        d_molecule_length.data,
                        d_molecule_list.data,
                        d_molecule_idx.data,
                        d_rigid_center.data,
                        molecule_indexer,
                        d_postype.data,
                        d_orientation.data,
                        m_body_idx,
                        d_body_pos.data,
                        d_body_orientation.data,
                        d_net_force.data,
                        d_net_virial.data,
                        d_body.data,
                        d_tag.data,
                        nmol,
                        m_pdata->getN(),
                        n_bodies_per_block,
                        m_pdata->getNetVirial().getPitch(),
                        m_virial_pitch,
                        block_size,
                        m_exec_conf->dev_prop,
                        m_gpu_partition);

        if (m_exec_conf->isCUDAErrorCheckingEnabled())
            CHECK_CUDA_ERROR();

        m_tuner_virial->end();
        m_exec_conf->endMultiGPU();
        }

    if (m_prof) m_prof->pop(m_exec_conf);
    if (m_prof) m_prof->pop(m_exec_conf);
    }

void ForceCompositeGPU::updateCompositeParticles(unsigned int timestep)
    {
    if (m_prof)
        m_prof->push(m_exec_conf, "constrain_rigid");

    if (m_prof)
        m_prof->push(m_exec_conf, "update");

    // access molecule order
    const GlobalArray<unsigned int>& molecule_length = getMoleculeLengths();

    ArrayHandle<unsigned int> d_molecule_order(getMoleculeOrder(), access_location::device, access_mode::read);
    ArrayHandle<unsigned int> d_molecule_len(molecule_length, access_location::device, access_mode::read);
    ArrayHandle<unsigned int> d_molecule_idx(getMoleculeIndex(), access_location::device, access_mode::read);

    // access the particle data arrays
    ArrayHandle<Scalar4> d_postype(m_pdata->getPositions(), access_location::device, access_mode::readwrite);
    ArrayHandle<Scalar4> d_orientation(m_pdata->getOrientationArray(), access_location::device, access_mode::readwrite);
    ArrayHandle<int3> d_image(m_pdata->getImages(), access_location::device, access_mode::readwrite);

    // access body positions and orientations
    ArrayHandle<Scalar3> d_body_pos(m_body_pos, access_location::device, access_mode::read);
    ArrayHandle<Scalar4> d_body_orientation(m_body_orientation, access_location::device, access_mode::read);
    ArrayHandle<unsigned int> d_body_len(m_body_len, access_location::device, access_mode::read);

    // lookup table
    ArrayHandle<unsigned int> d_lookup_center(m_lookup_center, access_location::device, access_mode::read);

        {
        ArrayHandle<uint2> d_flag(m_flag, access_location::device, access_mode::overwrite);

        m_exec_conf->beginMultiGPU();

        m_tuner_update->begin();
        unsigned int block_size = m_tuner_update->getParam();

        gpu_update_composite(m_pdata->getN(),
            m_pdata->getNGhosts(),
            d_postype.data,
            d_orientation.data,
            m_body_idx,
            d_lookup_center.data,
            d_body_pos.data,
            d_body_orientation.data,
            d_body_len.data,
            d_molecule_order.data,
            d_molecule_len.data,
            d_molecule_idx.data,
            d_image.data,
            m_pdata->getBox(),
            m_pdata->getGlobalBox(),
            block_size,
            d_flag.data,
            m_pdata->getGPUPartition());

        if (m_exec_conf->isCUDAErrorCheckingEnabled())
            CHECK_CUDA_ERROR();

        m_tuner_update->end();

        m_exec_conf->endMultiGPU();
        }

    uint2 flag;
        {
        ArrayHandle<uint2> h_flag(m_flag, access_location::host, access_mode::read);
        flag = *h_flag.data;
        }

    if (flag.x)
        {
        ArrayHandle<unsigned int> h_body(m_pdata->getBodies(), access_location::host, access_mode::read);
        ArrayHandle<unsigned int> h_tag(m_pdata->getTags(), access_location::host, access_mode::read);

        unsigned int idx = flag.x - 1;
        unsigned int body_id = h_body.data[idx];
        unsigned int tag = h_tag.data[idx];

        m_exec_conf->msg->errorAllRanks() << "constrain.rigid(): Particle " << tag << " part of composite body "
                                          << body_id << " is missing central particle" << std::endl << std::endl;
        throw std::runtime_error("Error while updating constituent particles");
        }

    if (flag.y)
        {
        ArrayHandle<unsigned int> h_body(m_pdata->getBodies(), access_location::host, access_mode::read);

        unsigned int idx = flag.y - 1;
        unsigned int body_id = h_body.data[idx];

        m_exec_conf->msg->errorAllRanks() << "constrain.rigid(): Composite particle with body id " << body_id
                                          << " incomplete" << std::endl << std::endl;
        throw std::runtime_error("Error while updating constituent particles");
        }

    if (m_prof)
        m_prof->pop(m_exec_conf);

    if (m_prof)
        m_prof->pop(m_exec_conf);
    }

void ForceCompositeGPU::findRigidCenters()
    {
    ArrayHandle<unsigned int> d_tag(m_pdata->getTags(), access_location::device, access_mode::read);
    ArrayHandle<unsigned int> d_rtag(m_pdata->getRTags(), access_location::device, access_mode::read);
    ArrayHandle<unsigned int> d_body(m_pdata->getBodies(), access_location::device, access_mode::read);

    m_rigid_center.resize(m_pdata->getN()+m_pdata->getNGhosts());

    unsigned int old_size = m_lookup_center.getNumElements();
    m_lookup_center.resize(m_pdata->getN()+m_pdata->getNGhosts());

    if (m_exec_conf->allConcurrentManagedAccess() && m_lookup_center.getNumElements() != old_size)
        {
        // set memory hints
        cudaMemAdvise(m_lookup_center.get(), sizeof(unsigned int)*m_lookup_center.getNumElements(), cudaMemAdviseSetReadMostly, 0);
        CHECK_CUDA_ERROR();
        }

    ArrayHandle<unsigned int> d_rigid_center(m_rigid_center, access_location::device, access_mode::overwrite);
    ArrayHandle<unsigned int> d_lookup_center(m_lookup_center, access_location::device, access_mode::overwrite);

    unsigned int n_rigid = 0;
    gpu_find_rigid_centers(d_body.data,
                        d_tag.data,
                        d_rtag.data,
                        m_pdata->getN(),
                        m_pdata->getNGhosts(),
                        d_rigid_center.data,
                        d_lookup_center.data,
                        n_rigid);

    // distribute rigid body centers over GPUs
    m_gpu_partition = GPUPartition(m_exec_conf->getGPUIds());
    m_gpu_partition.setN(n_rigid);
    }

void ForceCompositeGPU::lazyInitMem()
    {
    bool initialized = m_memory_initialized;

    // call base class method
    ForceComposite::lazyInitMem();

    if (!initialized)
        {
        GlobalVector<unsigned int> rigid_center(m_exec_conf);
        m_rigid_center.swap(rigid_center);
        TAG_ALLOCATION(m_rigid_center);

        GlobalVector<unsigned int> lookup_center(m_exec_conf);
        m_lookup_center.swap(lookup_center);
        TAG_ALLOCATION(m_lookup_center);
        }

    if (m_exec_conf->allConcurrentManagedAccess())
        {
        cudaMemAdvise(m_body_len.get(), sizeof(unsigned int)*m_body_len.getNumElements(), cudaMemAdviseSetReadMostly, 0);
        cudaMemAdvise(m_body_orientation.get(), sizeof(Scalar4)*m_body_orientation.getNumElements(), cudaMemAdviseSetReadMostly, 0);
        cudaMemAdvise(m_body_pos.get(), sizeof(Scalar3)*m_body_pos.getNumElements(), cudaMemAdviseSetReadMostly, 0);
        cudaMemAdvise(m_body_types.get(), sizeof(unsigned int)*m_body_types.getNumElements(), cudaMemAdviseSetReadMostly, 0);
        CHECK_CUDA_ERROR();
        }
    }

void export_ForceCompositeGPU(py::module& m)
    {
    py::class_< ForceCompositeGPU, std::shared_ptr<ForceCompositeGPU> >(m, "ForceCompositeGPU", py::base<ForceComposite>())
        .def(py::init< std::shared_ptr<SystemDefinition> >())
    ;
    }
