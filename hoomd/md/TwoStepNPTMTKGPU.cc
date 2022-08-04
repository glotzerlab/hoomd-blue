// Copyright (c) 2009-2022 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

#include "TwoStepNPTMTKGPU.h"
#include "TwoStepNPTMTKGPU.cuh"

#include "TwoStepNVEGPU.cuh"

#ifdef ENABLE_MPI
#include "hoomd/Communicator.h"
#include "hoomd/HOOMDMPI.h"
#endif

using namespace std;

/*! \file TwoStepNPTMTKGPU.h
    \brief Contains code for the TwoStepNPTMTKGPU class
*/

namespace hoomd
    {
namespace md
    {
/*! \param sysdef SystemDefinition this method will act on. Must not be NULL.
    \param group The group of particles this integration method is to work on
    \param thermo_group ComputeThermo to compute thermo properties of the integrated \a group
    \param thermo_group ComputeThermo to compute thermo properties of the integrated \a group at
   full time step \param tau NPT temperature period \param tauS NPT pressure period \param T
   Temperature set point \param S Pressure or Stress set point. Pressure if one value, Stress if a
   list of 6. Stress should be ordered as [xx, yy, zz, yz, xz, xy] \param couple Coupling mode
    \param flags Barostatted simulation box degrees of freedom
*/
TwoStepNPTMTKGPU::TwoStepNPTMTKGPU(std::shared_ptr<SystemDefinition> sysdef,
                                   std::shared_ptr<ParticleGroup> group,
                                   std::shared_ptr<ComputeThermo> thermo_group,
                                   std::shared_ptr<ComputeThermo> thermo_group_t,
                                   Scalar tau,
                                   Scalar tauS,
                                   std::shared_ptr<Variant> T,
                                   const std::vector<std::shared_ptr<Variant>>& S,
                                   const std::string& couple,
                                   const std::vector<bool>& flags,
                                   const bool nph)

    : TwoStepNPTMTK(sysdef,
                    group,
                    thermo_group,
                    thermo_group_t,
                    tau,
                    tauS,
                    T,
                    S,
                    couple,
                    flags,
                    nph)
    {
    if (!m_exec_conf->isCUDAEnabled())
        {
        throw std::runtime_error("Cannot create TwoStepNPTMTKGPU on a CPU device.");
        }

    m_exec_conf->msg->notice(5) << "Constructing TwoStepNPTMTKGPU" << endl;

    m_tuner_one.reset(new Autotuner<1>({AutotunerBase::makeBlockSizeRange(m_exec_conf)},
                                       m_exec_conf,
                                       "npt_mtk_step_one"));
    m_tuner_two.reset(new Autotuner<1>({AutotunerBase::makeBlockSizeRange(m_exec_conf)},
                                       m_exec_conf,
                                       "npt_mtk_step_two"));
    m_tuner_wrap.reset(new Autotuner<1>({AutotunerBase::makeBlockSizeRange(m_exec_conf)},
                                        m_exec_conf,
                                        "npt_mtk_wrap"));
    m_tuner_rescale.reset(new Autotuner<1>({AutotunerBase::makeBlockSizeRange(m_exec_conf)},
                                           m_exec_conf,
                                           "npt_mtk_rescale"));
    m_tuner_angular_one.reset(new Autotuner<1>({AutotunerBase::makeBlockSizeRange(m_exec_conf)},
                                               m_exec_conf,
                                               "npt_mtk_angular_one"));
    m_tuner_angular_two.reset(new Autotuner<1>({AutotunerBase::makeBlockSizeRange(m_exec_conf)},
                                               m_exec_conf,
                                               "npt_mtk_angular_two"));

    m_autotuners.insert(m_autotuners.end(),
                        {m_tuner_one,
                         m_tuner_two,
                         m_tuner_wrap,
                         m_tuner_rescale,
                         m_tuner_angular_one,
                         m_tuner_angular_two});
    }

TwoStepNPTMTKGPU::~TwoStepNPTMTKGPU()
    {
    m_exec_conf->msg->notice(5) << "Destroying TwoStepNPTMTKGPU" << endl;
    }

/*! \param timestep Current time step
    \post Particle positions are moved forward to timestep+1 and velocities to timestep+1/2 per the
   Nose-Hoover thermostat and Anderson barostat
*/
void TwoStepNPTMTKGPU::integrateStepOne(uint64_t timestep)
    {
    if (m_group->getNumMembersGlobal() == 0)
        {
        throw std::runtime_error("Integration group empty.");
        }

    // update degrees of freedom for MTK term
    m_ndof = m_group->getTranslationalDOF();

    // advance barostat (m_barostat.nu_xx, m_barostat.nu_yy, m_barostat.nu_zz) half a time step
    advanceBarostat(timestep);

    // Martyna-Tobias-Klein correction
    Scalar mtk = (m_barostat.nu_xx + m_barostat.nu_yy + m_barostat.nu_zz) / (Scalar)m_ndof;

    // update the propagator matrix using current barostat momenta
    updatePropagator();

    // advance box lengths
    BoxDim global_box = m_pdata->getGlobalBox();
    Scalar3 a = global_box.getLatticeVector(0);
    Scalar3 b = global_box.getLatticeVector(1);
    Scalar3 c = global_box.getLatticeVector(2);

    // (a,b,c) are the columns of the (upper triangular) cell parameter matrix
    // multiply with upper triangular matrix
    a.x = m_mat_exp_r[0] * a.x;
    b.x = m_mat_exp_r[0] * b.x + m_mat_exp_r[1] * b.y;
    b.y = m_mat_exp_r[3] * b.y;
    c.x = m_mat_exp_r[0] * c.x + m_mat_exp_r[1] * c.y + m_mat_exp_r[2] * c.z;
    c.y = m_mat_exp_r[3] * c.y + m_mat_exp_r[4] * c.z;
    c.z = m_mat_exp_r[5] * c.z;

    // update box dimensions
    bool twod = m_sysdef->getNDimensions() == 2;

    global_box.setL(make_scalar3(a.x, b.y, c.z));
    Scalar xy = b.x / b.y;

    Scalar xz(0.0);
    Scalar yz(0.0);

    if (!twod)
        {
        xz = c.x / c.z;
        yz = c.y / c.z;
        }

    global_box.setTiltFactors(xy, xz, yz);

    // set global box
    m_pdata->setGlobalBox(global_box);
    m_V = global_box.getVolume(twod); // volume

    // update the propagator matrix
    updatePropagator();

    if (m_rescale_all)
        {
        ArrayHandle<Scalar4> d_pos(m_pdata->getPositions(),
                                   access_location::device,
                                   access_mode::readwrite);

        // perform the update on the GPU
        m_exec_conf->beginMultiGPU();
        m_tuner_rescale->begin();

        // perform the particle update on the GPU
        kernel::gpu_npt_mtk_rescale(m_pdata->getGPUPartition(),
                                    d_pos.data,
                                    m_mat_exp_r[0],
                                    m_mat_exp_r[1],
                                    m_mat_exp_r[2],
                                    m_mat_exp_r[3],
                                    m_mat_exp_r[4],
                                    m_mat_exp_r[5],
                                    m_tuner_rescale->getParam()[0]);

        if (m_exec_conf->isCUDAErrorCheckingEnabled())
            CHECK_CUDA_ERROR();

        m_tuner_rescale->end();
        m_exec_conf->endMultiGPU();
        }

        {
        ArrayHandle<Scalar4> d_vel(m_pdata->getVelocities(),
                                   access_location::device,
                                   access_mode::readwrite);
        ArrayHandle<Scalar3> d_accel(m_pdata->getAccelerations(),
                                     access_location::device,
                                     access_mode::read);
        ArrayHandle<Scalar4> d_pos(m_pdata->getPositions(),
                                   access_location::device,
                                   access_mode::readwrite);

        ArrayHandle<unsigned int> d_index_array(m_group->getIndexArray(),
                                                access_location::device,
                                                access_mode::read);

        // precompute loop invariant quantity
        Scalar exp_thermo_fac = exp(-Scalar(1.0 / 2.0) * (m_thermostat.xi + mtk) * m_deltaT);

        // perform the particle update on the GPU
        m_exec_conf->beginMultiGPU();
        m_tuner_one->begin();

        kernel::gpu_npt_mtk_step_one(d_pos.data,
                                     d_vel.data,
                                     d_accel.data,
                                     d_index_array.data,
                                     m_group->getGPUPartition(),
                                     exp_thermo_fac,
                                     m_mat_exp_v,
                                     m_mat_exp_r,
                                     m_mat_exp_r_int,
                                     m_deltaT,
                                     m_rescale_all,
                                     m_tuner_one->getParam()[0]);

        if (m_exec_conf->isCUDAErrorCheckingEnabled())
            CHECK_CUDA_ERROR();

        m_tuner_one->end();
        m_exec_conf->endMultiGPU();
        } // end of GPUArray scope

    // Get new (local) box lengths
    BoxDim box = m_pdata->getBox();

        {
        ArrayHandle<Scalar4> d_pos(m_pdata->getPositions(),
                                   access_location::device,
                                   access_mode::readwrite);
        ArrayHandle<int3> d_image(m_pdata->getImages(),
                                  access_location::device,
                                  access_mode::readwrite);

        // Wrap particles
        m_exec_conf->beginMultiGPU();
        m_tuner_wrap->begin();

        kernel::gpu_npt_mtk_wrap(m_pdata->getGPUPartition(),
                                 d_pos.data,
                                 d_image.data,
                                 box,
                                 m_tuner_wrap->getParam()[0]);

        m_tuner_wrap->end();
        m_exec_conf->endMultiGPU();
        }

    if (m_aniso)
        {
        // first part of angular update
        ArrayHandle<Scalar4> d_orientation(m_pdata->getOrientationArray(),
                                           access_location::device,
                                           access_mode::readwrite);
        ArrayHandle<Scalar4> d_angmom(m_pdata->getAngularMomentumArray(),
                                      access_location::device,
                                      access_mode::readwrite);
        ArrayHandle<Scalar4> d_net_torque(m_pdata->getNetTorqueArray(),
                                          access_location::device,
                                          access_mode::read);
        ArrayHandle<Scalar3> d_inertia(m_pdata->getMomentsOfInertiaArray(),
                                       access_location::device,
                                       access_mode::read);
        ArrayHandle<unsigned int> d_index_array(m_group->getIndexArray(),
                                                access_location::device,
                                                access_mode::read);

        // precompute loop invariant quantity
        Scalar exp_thermo_fac_rot = exp(-(m_thermostat.xi_rot + mtk) * m_deltaT / Scalar(2.0));

        m_exec_conf->beginMultiGPU();
        m_tuner_angular_one->begin();

        kernel::gpu_nve_angular_step_one(d_orientation.data,
                                         d_angmom.data,
                                         d_inertia.data,
                                         d_net_torque.data,
                                         d_index_array.data,
                                         m_group->getGPUPartition(),
                                         m_deltaT,
                                         exp_thermo_fac_rot,
                                         m_tuner_angular_one->getParam()[0]);

        if (m_exec_conf->isCUDAErrorCheckingEnabled())
            CHECK_CUDA_ERROR();
        m_tuner_angular_one->end();
        m_exec_conf->endMultiGPU();
        }

    if (!m_nph)
        {
        // propagate thermostat variables forward
        advanceThermostat(timestep);
        }

#ifdef ENABLE_MPI
    if (m_sysdef->isDomainDecomposed())
        {
        // broadcast integrator variables from rank 0 to other processors
        MPI_Bcast(&m_thermostat, 4, MPI_HOOMD_SCALAR, 0, m_exec_conf->getMPICommunicator());
        MPI_Bcast(&m_barostat, 6, MPI_HOOMD_SCALAR, 0, m_exec_conf->getMPICommunicator());
        }
#endif
    }

/*! \param timestep Current time step
    \post particle velocities are moved forward to timestep+1
*/
void TwoStepNPTMTKGPU::integrateStepTwo(uint64_t timestep)
    {
    const GlobalArray<Scalar4>& net_force = m_pdata->getNetForce();

    // Martyna-Tobias-Klein correction
    Scalar mtk = (m_barostat.nu_xx + m_barostat.nu_yy + m_barostat.nu_zz) / (Scalar)m_ndof;

        {
        ArrayHandle<Scalar4> d_vel(m_pdata->getVelocities(),
                                   access_location::device,
                                   access_mode::readwrite);
        ArrayHandle<Scalar3> d_accel(m_pdata->getAccelerations(),
                                     access_location::device,
                                     access_mode::overwrite);

        ArrayHandle<Scalar4> d_net_force(net_force, access_location::device, access_mode::read);
        ArrayHandle<unsigned int> d_index_array(m_group->getIndexArray(),
                                                access_location::device,
                                                access_mode::read);

        // precompute loop invariant quantity
        Scalar exp_thermo_fac = exp(-Scalar(1.0 / 2.0) * (m_thermostat.xi + mtk) * m_deltaT);

        // perform second half step of NPT integration (update velocities and accelerations)
        m_exec_conf->beginMultiGPU();
        m_tuner_two->begin();

        kernel::gpu_npt_mtk_step_two(d_vel.data,
                                     d_accel.data,
                                     d_index_array.data,
                                     m_group->getGPUPartition(),
                                     d_net_force.data,
                                     m_mat_exp_v,
                                     m_deltaT,
                                     exp_thermo_fac,
                                     m_tuner_two->getParam()[0]);

        if (m_exec_conf->isCUDAErrorCheckingEnabled())
            CHECK_CUDA_ERROR();
        m_tuner_two->end();
        m_exec_conf->endMultiGPU();

        } // end GPUArray scope

    if (m_aniso)
        {
        // apply angular (NO_SQUISH) equations of motion
        ArrayHandle<Scalar4> d_orientation(m_pdata->getOrientationArray(),
                                           access_location::device,
                                           access_mode::read);
        ArrayHandle<Scalar4> d_angmom(m_pdata->getAngularMomentumArray(),
                                      access_location::device,
                                      access_mode::readwrite);
        ArrayHandle<Scalar4> d_net_torque(m_pdata->getNetTorqueArray(),
                                          access_location::device,
                                          access_mode::read);
        ArrayHandle<Scalar3> d_inertia(m_pdata->getMomentsOfInertiaArray(),
                                       access_location::device,
                                       access_mode::read);
        ArrayHandle<unsigned int> d_index_array(m_group->getIndexArray(),
                                                access_location::device,
                                                access_mode::read);

        // precompute loop invariant quantity
        Scalar exp_thermo_fac_rot = exp(-(m_thermostat.xi_rot + mtk) * m_deltaT / Scalar(2.0));

        m_exec_conf->beginMultiGPU();
        m_tuner_angular_two->begin();

        kernel::gpu_nve_angular_step_two(d_orientation.data,
                                         d_angmom.data,
                                         d_inertia.data,
                                         d_net_torque.data,
                                         d_index_array.data,
                                         m_group->getGPUPartition(),
                                         m_deltaT,
                                         exp_thermo_fac_rot,
                                         m_tuner_angular_two->getParam()[0]);

        if (m_exec_conf->isCUDAErrorCheckingEnabled())
            CHECK_CUDA_ERROR();

        m_tuner_angular_two->end();
        m_exec_conf->endMultiGPU();
        }

    // advance barostat (m_barostat.nu_xx, m_barostat.nu_yy, m_barostat.nu_zz) half a time step
    advanceBarostat(timestep + 1);
    }

namespace detail
    {
void export_TwoStepNPTMTKGPU(pybind11::module& m)
    {
    pybind11::class_<TwoStepNPTMTKGPU, TwoStepNPTMTK, std::shared_ptr<TwoStepNPTMTKGPU>>(
        m,
        "TwoStepNPTMTKGPU")
        .def(pybind11::init<std::shared_ptr<SystemDefinition>,
                            std::shared_ptr<ParticleGroup>,
                            std::shared_ptr<ComputeThermo>,
                            std::shared_ptr<ComputeThermo>,
                            Scalar,
                            Scalar,
                            std::shared_ptr<Variant>,
                            const std::vector<std::shared_ptr<Variant>>&,
                            const std::string&,
                            const std::vector<bool>&,
                            const bool>());
    }
    } // end namespace detail
    } // end namespace md
    } // end namespace hoomd
