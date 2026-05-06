// Copyright (c) 2009-2026 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

/*! \file md/BoxDeformer.cc
    \brief Definition of box deformers
*/

#include "BoxDeformer.h"

#ifdef ENABLE_HIP
#include "BoxDeformerGPU.cuh"
#endif

namespace hoomd
    {
namespace md
    {
/*!
    \param sysdef System definition containing the particle data this method acts on
 */
BoxDeformer::BoxDeformer(std::shared_ptr<SystemDefinition> sysdef)
    : m_sysdef(sysdef), m_pdata(sysdef->getParticleData()), m_exec_conf(m_pdata->getExecConf())
    {
    m_exec_conf->msg->notice(5) << "Constructing BoxDeformer" << std::endl;

#ifdef ENABLE_HIP
    if (m_exec_conf->isCUDAEnabled())
        {
        m_tuner_wrap.reset(new Autotuner<1>({AutotunerBase::makeBlockSizeRange(m_exec_conf)},
                                            m_exec_conf,
                                            "box_deformer_remap"));
        m_autotuners.push_back(m_tuner_wrap);
        }
#endif
    }

BoxDeformer::~BoxDeformer()
    {
    m_exec_conf->msg->notice(5) << "Destroying BoxDeformer" << std::endl;
    }

void BoxDeformer::setDeltaT(Scalar deltaT)
    {
    m_deltaT = deltaT;
    }

void BoxDeformer::update(uint64_t timestep)
    {
    // Get the current box
    const BoxDim old_box = m_pdata->getGlobalBox();

    // Compute new box (child class will determine the new box geometry)
    BoxDim new_box = computeNewBox(timestep, old_box);

    if (new_box != old_box)
        {
        // Set the new global box
        m_pdata->setGlobalBox(new_box);

        // Apply any post-deformation processing
        processAfterDeformation(old_box, new_box);
        }
    }

BoxDim BoxDeformer::computeNewBox(uint64_t timestep, const BoxDim& old_box)
    {
    // By default, just return the current box (no deformation)
    // Derived classes should override this to apply actual deformation.
    return old_box;
    }

// Post deformation particle processing: PBC wrapping by default but child classes can add up
void BoxDeformer::processAfterDeformation(const BoxDim& old_box, const BoxDim& new_box)
    {
#ifdef ENABLE_HIP
    if (m_exec_conf->isCUDAEnabled())
        {
        // GPU path
        ArrayHandle<Scalar4> d_pos(m_pdata->getPositions(),
                                   access_location::device,
                                   access_mode::readwrite);

        ArrayHandle<Scalar4> d_vel(m_pdata->getVelocities(),
                                   access_location::device,
                                   access_mode::readwrite);

        ArrayHandle<int3> d_image(m_pdata->getImages(),
                                  access_location::device,
                                  access_mode::readwrite);

        m_tuner_wrap->begin();
        kernel::gpu_boxdeformer_wrap(m_pdata->getN(),
                                     d_pos.data,
                                     d_vel.data,
                                     d_image.data,
                                     new_box,
                                     m_tuner_wrap->getParam()[0]);
        if (m_exec_conf->isCUDAErrorCheckingEnabled())
            CHECK_CUDA_ERROR();
        m_tuner_wrap->end();
        }
    else
#endif
        {
        // CPU path
        ArrayHandle<Scalar4> h_pos(m_pdata->getPositions(),
                                   access_location::host,
                                   access_mode::readwrite);
        ArrayHandle<Scalar4> h_vel(m_pdata->getVelocities(),
                                   access_location::host,
                                   access_mode::readwrite);
        ArrayHandle<int3> h_image(m_pdata->getImages(),
                                  access_location::host,
                                  access_mode::readwrite);

        for (unsigned int i = 0; i < m_pdata->getN(); i++)
            {
            new_box.wrap(h_pos.data[i], h_vel.data[i], h_image.data[i]);
            }
        }
    }
namespace detail
    {
void export_BoxDeformer(pybind11::module& m)
    {
    pybind11::class_<BoxDeformer, std::shared_ptr<BoxDeformer>>(m, "BoxDeformer")
        .def(pybind11::init<std::shared_ptr<SystemDefinition>>());
    }

    } // end namespace detail
    } // end namespace md
    } // end namespace hoomd
