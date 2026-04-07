// Copyright (c) 2009-2025 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

/*! \file md/LeesEdwardsBoxDeformer.cc
    \brief Definition of Lees–Edwards box deformer for triclinic boxes
*/

#include "LeesEdwardsBoxDeformer.h"

#ifdef ENABLE_HIP
#include "LeesEdwardsBoxDeformerGPU.cuh"
#endif

namespace hoomd
    {
namespace md
    {
/*! @param sysdef System definition containing the particle data this method acts on
    @param xy_rate Rate at which the box xy tilt deforms
    @param max_xy_tilt Maximum allowed xy tilt before the box flips; must be non-negative
 */
LeesEdwardsBoxDeformer::LeesEdwardsBoxDeformer(std::shared_ptr<SystemDefinition> sysdef,
                                               Scalar xy_rate,
                                               Scalar max_xy_tilt)
    : BoxDeformer(sysdef), m_xy_rate(xy_rate), m_max_xy_tilt(max_xy_tilt)
    {
    m_exec_conf->msg->notice(5) << "Constructing LeesEdwardsBoxDeformer" << std::endl;

#ifdef ENABLE_HIP
    if (m_exec_conf->isCUDAEnabled())
        {
        m_tuner_remap.reset(new Autotuner<1>({AutotunerBase::makeBlockSizeRange(m_exec_conf)},
                                             m_exec_conf,
                                             "box_deformer_remap"));
        m_autotuners.push_back(m_tuner_remap);
        }
#endif
    }

LeesEdwardsBoxDeformer::~LeesEdwardsBoxDeformer()
    {
    m_exec_conf->msg->notice(5) << "Destroying LeesEdwardsBoxDeformer" << std::endl;
    }

BoxDim LeesEdwardsBoxDeformer::computeNewBox(uint64_t timestep, const BoxDim& old_box)
    {
    // Get the tilt factors (xz, yz are unchanged but needed to reset the box)
    Scalar xy = old_box.getTiltFactorXY();
    const Scalar xz = old_box.getTiltFactorXZ();
    const Scalar yz = old_box.getTiltFactorYZ();

    // Update xy tilt factor using stored rate and deltaT
    xy += m_xy_rate * m_deltaT;

    // Return updated box
    BoxDim new_box(old_box);
    new_box.setTiltFactors(xy, xz, yz);
    new_box.setTiltDeformationRates(m_xy_rate, 0.0, 0.0);

    return new_box;
    }

void LeesEdwardsBoxDeformer::processAfterDeformation(const BoxDim& old_box, const BoxDim& new_box)
    {
    // Get xy tilt, and flip the box if tilt is outside the range [-max_xy_tilt, +max_xy_tilt]
    Scalar xy = new_box.getTiltFactorXY();

    // Shift the range to [0, 2*max_xy_tilt], then divide by (2*max_xy_tilt) to normalize,
    // and floor to get a discrete flip value
    int flip = static_cast<int>(std::floor((xy + m_max_xy_tilt) / (2.0 * m_max_xy_tilt)));

    if (flip != 0)
        {
            // Flip box and remap particles
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
            m_tuner_remap->begin();
            kernel::gpu_lees_edwards_remap(m_pdata->getN(),
                                           d_pos.data,
                                           d_vel.data,
                                           d_image.data,
                                           new_box,
                                           flip,
                                           m_tuner_remap->getParam()[0]);
            if (m_exec_conf->isCUDAErrorCheckingEnabled())
                CHECK_CUDA_ERROR();
            m_tuner_remap->end();
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

            const unsigned int N = m_pdata->getN();
            Scalar Ly = new_box.getL().y;

            for (unsigned int i = 0; i < N; i++)
                {
                h_pos.data[i].x -= Scalar(flip) * Ly;
                h_image.data[i].x -= flip;
                new_box.wrap(h_pos.data[i], h_vel.data[i], h_image.data[i]);
                }
            }

        // Reset global box with updated xy tilt to reflect flip
        BoxDim flipped_box(new_box);
        xy -= Scalar(2.0 * flip) * m_max_xy_tilt;
        const Scalar xz = new_box.getTiltFactorXZ();
        const Scalar yz = new_box.getTiltFactorYZ();
        flipped_box.setTiltFactors(xy, xz, yz);
        m_pdata->setGlobalBox(flipped_box);

#ifdef ENABLE_MPI
        // On MPI, take a snapshot to pull particle data to rank 0
        // Initializing will broadcast particle across MPI ranks, ensuring
        // consistency in particle data across all domains
        if (m_sysdef->isDomainDecomposed())
            {
            SnapshotParticleData<Scalar> snap;
            m_pdata->takeSnapshot(snap);
            m_pdata->initializeFromSnapshot(snap);
            }
#endif
        }
    else
        {
        // No flip on this step, call base class to perform default PBC wrapping
        BoxDeformer::processAfterDeformation(old_box, new_box);
        }
    }

namespace detail
    {
void export_LeesEdwardsBoxDeformer(pybind11::module& m)
    {
    pybind11::class_<LeesEdwardsBoxDeformer, BoxDeformer, std::shared_ptr<LeesEdwardsBoxDeformer>>(
        m,
        "LeesEdwardsBoxDeformer")
        .def(pybind11::init<std::shared_ptr<SystemDefinition>, Scalar, Scalar>())
        .def_property("shear_rate",
                      &LeesEdwardsBoxDeformer::getShearRate,
                      &LeesEdwardsBoxDeformer::setShearRate)
        .def_property("max_tilt",
                      &LeesEdwardsBoxDeformer::getMaxXYTilt,
                      &LeesEdwardsBoxDeformer::setMaxXYTilt);
    }

    } // end namespace detail
    } // end namespace md
    } // end namespace hoomd
