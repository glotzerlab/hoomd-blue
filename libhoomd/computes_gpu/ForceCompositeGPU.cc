/*
Highly Optimized Object-oriented Many-particle Dynamics -- Blue Edition
(HOOMD-blue) Open Source Software License Copyright 2009-2015 The Regents of
the University of Michigan All rights reserved.

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

// Maintainer: jglaser

#include "ForceCompositeGPU.h"
#include "VectorMath.h"

#include "ForceCompositeGPU.cuh"

#include <boost/python.hpp>

/*! \file ForceCompositeGPU.cc
    \brief Contains code for the ForceCompositeGPU class
*/

/*! \param sysdef SystemDefinition containing the ParticleData to compute forces on
*/
ForceCompositeGPU::ForceCompositeGPU(boost::shared_ptr<SystemDefinition> sysdef)
        : ForceComposite(sysdef)
    {

    // power of two block sizes
    const cudaDeviceProp& dev_prop = m_exec_conf->dev_prop;
    std::vector<unsigned int> valid_params;
    unsigned int bodies_per_block = 1;
    for (unsigned int i = 0; i < 5; ++i)
        {
        bodies_per_block = 1 << i;
        unsigned int cur_block_size = 1;
        while (cur_block_size <= dev_prop.maxThreadsPerBlock)
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
    }

ForceCompositeGPU::~ForceCompositeGPU()
    {}


//! Compute the forces and torques on the central particle
void ForceCompositeGPU::computeForces(unsigned int timestep)
    {
    // at this point, all constituent particles of a local rigid body (i.e. one for which the central particle
    // is local) need to be present to correctly compute forces
    if (m_particles_sorted)
        {
        // initialize molecule table
        initMolecules();
        }

    if (m_prof)
        m_prof->push(m_exec_conf, "constrain_rigid");

    if (m_prof)
        m_prof->push(m_exec_conf, "sum force and torque");

    // access particle data
    ArrayHandle<unsigned int> d_body(m_pdata->getBodies(), access_location::device, access_mode::read);
    ArrayHandle<unsigned int> d_rtag(m_pdata->getRTags(), access_location::device, access_mode::read);
    ArrayHandle<unsigned int> d_tag(m_pdata->getTags(), access_location::device, access_mode::read);
    ArrayHandle<Scalar4> d_postype(m_pdata->getPositions(), access_location::device, access_mode::read);
    ArrayHandle<Scalar4> d_orientation(m_pdata->getOrientationArray(), access_location::device, access_mode::read);

    // access net force and torque acting on constituent particles
    ArrayHandle<Scalar4> d_net_force(m_pdata->getNetForce(), access_location::device, access_mode::read);
    ArrayHandle<Scalar4> d_net_torque(m_pdata->getNetTorqueArray(), access_location::device, access_mode::read);
    ArrayHandle<Scalar> d_net_virial(m_pdata->getNetVirial(), access_location::device, access_mode::read);

    // access the force and torque array for the central ptl
    ArrayHandle<Scalar4> d_force(m_force, access_location::device, access_mode::overwrite);
    ArrayHandle<Scalar4> d_torque(m_torque, access_location::device, access_mode::overwrite);
    ArrayHandle<Scalar> d_virial(m_virial, access_location::device, access_mode::overwrite);

    // for each local body
    unsigned int nmol = m_molecule_indexer.getW();

    ArrayHandle<unsigned int> d_molecule_length(m_molecule_length, access_location::device, access_mode::read);
    ArrayHandle<unsigned int> d_molecule_list(m_molecule_list, access_location::device, access_mode::read);

    ArrayHandle<Scalar3> d_body_pos(m_body_pos, access_location::device, access_mode::read);
    ArrayHandle<Scalar4> d_body_orientation(m_body_orientation, access_location::device, access_mode::read);

    m_tuner_force->begin();
    unsigned int param = m_tuner_force->getParam();
    unsigned int block_size = param % 10000;
    unsigned int n_bodies_per_block = param/10000;

    // launch GPU kernel
    gpu_rigid_force(d_force.data,
                    d_torque.data,
                    d_molecule_length.data,
                    d_molecule_list.data,
                    d_tag.data,
                    d_rtag.data,
                    m_molecule_indexer,
                    d_postype.data,
                    d_orientation.data,
                    m_body_idx,
                    d_body_pos.data,
                    d_body_orientation.data,
                    d_net_force.data,
                    d_net_torque.data,
                    nmol,
                    m_pdata->getN(),
                    n_bodies_per_block,
                    block_size,
                    m_exec_conf->dev_prop);

    if (m_exec_conf->isCUDAErrorCheckingEnabled())
        CHECK_CUDA_ERROR();

    m_tuner_force->end();

    PDataFlags flags = m_pdata->getFlags();
    bool compute_virial = false;
    if (flags[pdata_flag::isotropic_virial] || flags[pdata_flag::pressure_tensor])
        {
        compute_virial = true;
        }

    if (compute_virial)
        {
        m_tuner_virial->begin();
        param = m_tuner_virial->getParam();
        block_size = param % 10000;
        n_bodies_per_block = param/10000;

        // launch GPU kernel
        gpu_rigid_virial(d_virial.data,
                        d_molecule_length.data,
                        d_molecule_list.data,
                        d_tag.data,
                        d_rtag.data,
                        m_molecule_indexer,
                        d_postype.data,
                        d_orientation.data,
                        m_body_idx,
                        d_body_pos.data,
                        d_body_orientation.data,
                        d_net_force.data,
                        d_net_virial.data,
                        nmol,
                        m_pdata->getN(),
                        n_bodies_per_block,
                        m_pdata->getNetVirial().getPitch(),
                        m_virial_pitch,
                        block_size,
                        m_exec_conf->dev_prop);

        if (m_exec_conf->isCUDAErrorCheckingEnabled())
            CHECK_CUDA_ERROR();

        m_tuner_virial->end();
        }


    if (m_prof) m_prof->pop(m_exec_conf);
    if (m_prof) m_prof->pop(m_exec_conf);
    }

void ForceCompositeGPU::updateCompositeParticles(unsigned int timestep, bool remote)
    {
    if (m_prof)
        m_prof->push(m_exec_conf, "constrain_rigid");

    if (m_prof)
        m_prof->push(m_exec_conf, "update");

    // access the particle data arrays
    ArrayHandle<Scalar4> d_postype(m_pdata->getPositions(), access_location::device, access_mode::readwrite);
    ArrayHandle<Scalar4> d_orientation(m_pdata->getOrientationArray(), access_location::device, access_mode::readwrite);
    ArrayHandle<int3> d_image(m_pdata->getImages(), access_location::device, access_mode::readwrite);

    ArrayHandle<unsigned int> d_body(m_pdata->getBodies(), access_location::device, access_mode::read);
    ArrayHandle<unsigned int> d_rtag(m_pdata->getRTags(), access_location::device, access_mode::read);
    ArrayHandle<unsigned int> d_tag(m_pdata->getTags(), access_location::device, access_mode::read);

    // access body positions and orientations
    ArrayHandle<Scalar3> d_body_pos(m_body_pos, access_location::device, access_mode::read);
    ArrayHandle<Scalar4> d_body_orientation(m_body_orientation, access_location::device, access_mode::read);

    m_tuner_update->begin();
    unsigned int block_size = m_tuner_update->getParam();

    gpu_update_composite(m_pdata->getN(),
        m_pdata->getNGhosts(),
        d_body.data,
        d_rtag.data,
        d_tag.data,
        d_postype.data,
        d_orientation.data,
        m_body_idx,
        d_body_pos.data,
        d_body_orientation.data,
        d_image.data,
        m_pdata->getBox(),
        remote,
        block_size);

    if (m_exec_conf->isCUDAErrorCheckingEnabled())
        CHECK_CUDA_ERROR();

    m_tuner_update->end();

    if (m_prof)
        m_prof->pop(m_exec_conf);

    if (m_prof)
        m_prof->pop(m_exec_conf);
    }

void export_ForceCompositeGPU()
    {
    class_< ForceCompositeGPU, boost::shared_ptr<ForceCompositeGPU>, bases<ForceComposite>, boost::noncopyable >
    ("ForceCompositeGPU", init< boost::shared_ptr<SystemDefinition> >())
    ;
    }
