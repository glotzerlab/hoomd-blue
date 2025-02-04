// Copyright (c) 2009-2024 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

#include "Integrator.h"

#ifdef ENABLE_HIP
#include "Integrator.cuh"
#endif

#ifdef ENABLE_MPI
#include "Communicator.h"
#endif

#include <pybind11/stl_bind.h>
PYBIND11_MAKE_OPAQUE(std::vector<std::shared_ptr<hoomd::ForceConstraint>>);
PYBIND11_MAKE_OPAQUE(std::vector<std::shared_ptr<hoomd::ForceCompute>>);

using namespace std;

namespace hoomd
    {
/** @param sysdef System to update
    @param deltaT Time step to use
*/
Integrator::Integrator(std::shared_ptr<SystemDefinition> sysdef, Scalar deltaT)
    : Updater(sysdef, std::make_shared<PeriodicTrigger>(1)), m_deltaT(deltaT)
    {
#ifdef ENABLE_MPI
    if (m_sysdef->isDomainDecomposed())
        {
        auto comm_weak = m_sysdef->getCommunicator();
        assert(comm_weak.lock());
        m_comm = comm_weak.lock();

        // connect to ghost communication flags request
        m_comm->getCommFlagsRequestSignal().connect<Integrator, &Integrator::determineFlags>(this);

        m_comm->getComputeCallbackSignal().connect<Integrator, &Integrator::computeCallback>(this);
        }
#endif

    if (m_deltaT < 0)
        m_exec_conf->msg->warning() << "A step size dt of less than 0 was specified." << endl;
    }

Integrator::~Integrator()
    {
#ifdef ENABLE_MPI
    // disconnect
    if (m_sysdef->isDomainDecomposed())
        {
        m_comm->getCommFlagsRequestSignal().disconnect<Integrator, &Integrator::determineFlags>(
            this);

        m_comm->getComputeCallbackSignal().disconnect<Integrator, &Integrator::computeCallback>(
            this);
        }
#endif
    }

/** @param deltaT New time step to set
 */
void Integrator::setDeltaT(Scalar deltaT)
    {
    if (m_deltaT < 0.0)
        throw std::domain_error("delta_t must be positive");

    for (auto& force : m_forces)
        {
        force->setDeltaT(deltaT);
        }

    for (auto& constraint_force : m_constraint_forces)
        {
        constraint_force->setDeltaT(deltaT);
        }

    m_deltaT = deltaT;
    }

/** \return the timestep deltaT
 */
Scalar Integrator::getDeltaT()
    {
    return m_deltaT;
    }

/** Loops over all constraint forces in the Integrator and sums up the number of DOF removed
    @param query The group over which to compute the removed degrees of freedom
*/
Scalar Integrator::getNDOFRemoved(std::shared_ptr<ParticleGroup> query)
    {
    // start counting at 0
    Scalar n = 0;

    for (const auto& constraint_force : m_constraint_forces)
        {
        n += constraint_force->getNDOFRemoved(query);
        }
    return n;
    }

/** @param timestep Current timestep
    \post \c h_accel.data[i] is set based on the forces computed by the ForceComputes
*/
void Integrator::computeAccelerations(uint64_t timestep)
    {
    m_exec_conf->msg->notice(5) << "integrate.*: pre-computing missing acceleration data" << endl;

    // now, get our own access to the arrays and calculate the accelerations
    ArrayHandle<Scalar3> h_accel(m_pdata->getAccelerations(),
                                 access_location::host,
                                 access_mode::readwrite);
    ArrayHandle<Scalar4> h_vel(m_pdata->getVelocities(), access_location::host, access_mode::read);
    ArrayHandle<Scalar4> h_net_force(m_pdata->getNetForce(),
                                     access_location::host,
                                     access_mode::read);

    // now, add up the accelerations
    for (unsigned int j = 0; j < m_pdata->getN(); j++)
        {
        Scalar minv = Scalar(1.0) / h_vel.data[j].w;
        h_accel.data[j].x = h_net_force.data[j].x * minv;
        h_accel.data[j].y = h_net_force.data[j].y * minv;
        h_accel.data[j].z = h_net_force.data[j].z * minv;
        }
    }

/** @param timestep Current time step of the simulation
 */
vec3<double> Integrator::computeLinearMomentum()
    {
    // grab access to the particle data
    ArrayHandle<Scalar4> h_vel(m_pdata->getVelocities(), access_location::host, access_mode::read);

    // sum the linear momentum in the system
    vec3<double> p_total;
    for (unsigned int i = 0; i < m_pdata->getN(); i++)
        {
        double mass = h_vel.data[i].w;
        vec3<double> velocity(h_vel.data[i]);
        p_total += mass * velocity;
        }

#ifdef ENABLE_MPI
    if (m_pdata->getDomainDecomposition())
        {
        MPI_Allreduce(MPI_IN_PLACE,
                      &p_total,
                      3,
                      MPI_DOUBLE,
                      MPI_SUM,
                      m_exec_conf->getMPICommunicator());
        }
#endif

    return p_total;
    }

/** @param timestep Current time step of the simulation
    \post All added force computes in \a m_forces are computed and totaled up in \a m_net_force and
   \a m_net_virial \note The summation step is performed <b>on the CPU</b> and will result in a lot
   of data traffic back and forth if the forces and/or integrator are on the GPU. Call
   computeNetForcesGPU() to sum the forces on the GPU
*/
void Integrator::computeNetForce(uint64_t timestep)
    {
    for (auto& force : m_forces)
        {
        force->compute(timestep);
        }

    Scalar external_virial[6];
    Scalar external_energy;
        {
        // access the net force and virial arrays
        const GlobalArray<Scalar4>& net_force = m_pdata->getNetForce();
        const GlobalArray<Scalar>& net_virial = m_pdata->getNetVirial();
        const GlobalArray<Scalar4>& net_torque = m_pdata->getNetTorqueArray();
        ArrayHandle<Scalar4> h_net_force(net_force, access_location::host, access_mode::overwrite);
        ArrayHandle<Scalar> h_net_virial(net_virial, access_location::host, access_mode::overwrite);
        ArrayHandle<Scalar4> h_net_torque(net_torque,
                                          access_location::host,
                                          access_mode::overwrite);

        // start by zeroing the net force and virial arrays
        memset((void*)h_net_force.data, 0, sizeof(Scalar4) * net_force.getNumElements());
        memset((void*)h_net_virial.data, 0, sizeof(Scalar) * net_virial.getNumElements());
        memset((void*)h_net_torque.data, 0, sizeof(Scalar4) * net_torque.getNumElements());

        for (unsigned int i = 0; i < 6; ++i)
            external_virial[i] = Scalar(0.0);

        external_energy = Scalar(0.0);

        // now, add up the net forces
        // also sum up forces for ghosts, in case they are needed by the communicator
        unsigned int nparticles = m_pdata->getN() + m_pdata->getNGhosts();
        size_t net_virial_pitch = net_virial.getPitch();

        assert(nparticles <= net_force.getNumElements());
        assert(6 * nparticles <= net_virial.getNumElements());
        assert(nparticles <= net_torque.getNumElements());

        for (const auto& force : m_forces)
            {
            const GlobalArray<Scalar4>& h_force_array = force->getForceArray();
            const GlobalArray<Scalar>& h_virial_array = force->getVirialArray();
            const GlobalArray<Scalar4>& h_torque_array = force->getTorqueArray();

            assert(nparticles <= h_force_array.getNumElements());
            assert(6 * nparticles <= h_virial_array.getNumElements());
            assert(nparticles <= h_torque_array.getNumElements());

            ArrayHandle<Scalar4> h_force(h_force_array, access_location::host, access_mode::read);
            ArrayHandle<Scalar> h_virial(h_virial_array, access_location::host, access_mode::read);
            ArrayHandle<Scalar4> h_torque(h_torque_array, access_location::host, access_mode::read);

            size_t virial_pitch = h_virial_array.getPitch();
            for (unsigned int j = 0; j < nparticles; j++)
                {
                h_net_force.data[j].x += h_force.data[j].x;
                h_net_force.data[j].y += h_force.data[j].y;
                h_net_force.data[j].z += h_force.data[j].z;
                h_net_force.data[j].w += h_force.data[j].w;

                h_net_torque.data[j].x += h_torque.data[j].x;
                h_net_torque.data[j].y += h_torque.data[j].y;
                h_net_torque.data[j].z += h_torque.data[j].z;
                h_net_torque.data[j].w += h_torque.data[j].w;

                for (unsigned int k = 0; k < 6; k++)
                    {
                    h_net_virial.data[k * net_virial_pitch + j]
                        += h_virial.data[k * virial_pitch + j];
                    }
                }

            for (unsigned int k = 0; k < 6; k++)
                {
                external_virial[k] += force->getExternalVirial(k);
                }

            external_energy += force->getExternalEnergy();
            }
        }

    for (unsigned int k = 0; k < 6; k++)
        {
        m_pdata->setExternalVirial(k, external_virial[k]);
        }

    m_pdata->setExternalEnergy(external_energy);

    // return early if there are no constraint forces or no HalfStepHook set
    if (m_constraint_forces.size() == 0)
        return;

#ifdef ENABLE_MPI
    if (m_sysdef->isDomainDecomposed())
        {
        // communicate the net force
        m_comm->updateNetForce(timestep);
        }
#endif

    // compute all the constraint forces next
    // constraint forces only apply a force, not a torque
    for (auto& constraint_force : m_constraint_forces)
        {
        constraint_force->compute(timestep);
        }

        {
        // access the net force and virial arrays
        const GlobalArray<Scalar4>& net_force = m_pdata->getNetForce();
        const GlobalArray<Scalar>& net_virial = m_pdata->getNetVirial();
        const GlobalArray<Scalar4>& net_torque = m_pdata->getNetTorqueArray();
        ArrayHandle<Scalar4> h_net_force(net_force, access_location::host, access_mode::readwrite);
        ArrayHandle<Scalar> h_net_virial(net_virial, access_location::host, access_mode::readwrite);
        ArrayHandle<Scalar4> h_net_torque(net_torque,
                                          access_location::host,
                                          access_mode::readwrite);
        size_t net_virial_pitch = net_virial.getPitch();

        // now, add up the net forces
        unsigned int nparticles = m_pdata->getN();
        assert(nparticles <= net_force.getNumElements());
        assert(6 * nparticles <= net_virial.getNumElements());
        for (const auto& constraint_force : m_constraint_forces)
            {
            const GlobalArray<Scalar4>& h_force_array = constraint_force->getForceArray();
            const GlobalArray<Scalar>& h_virial_array = constraint_force->getVirialArray();
            const GlobalArray<Scalar4>& h_torque_array = constraint_force->getTorqueArray();
            ArrayHandle<Scalar4> h_force(h_force_array, access_location::host, access_mode::read);
            ArrayHandle<Scalar> h_virial(h_virial_array, access_location::host, access_mode::read);
            ArrayHandle<Scalar4> h_torque(h_torque_array, access_location::host, access_mode::read);
            size_t virial_pitch = h_virial_array.getPitch();

            assert(nparticles <= h_force_array.getNumElements());
            assert(6 * nparticles <= h_virial_array.getNumElements());
            assert(nparticles <= h_torque_array.getNumElements());

            for (unsigned int j = 0; j < nparticles; j++)
                {
                h_net_force.data[j].x += h_force.data[j].x;
                h_net_force.data[j].y += h_force.data[j].y;
                h_net_force.data[j].z += h_force.data[j].z;
                h_net_force.data[j].w += h_force.data[j].w;

                h_net_torque.data[j].x += h_torque.data[j].x;
                h_net_torque.data[j].y += h_torque.data[j].y;
                h_net_torque.data[j].z += h_torque.data[j].z;
                h_net_torque.data[j].w += h_torque.data[j].w;

                for (unsigned int k = 0; k < 6; k++)
                    {
                    h_net_virial.data[k * net_virial_pitch + j]
                        += h_virial.data[k * virial_pitch + j];
                    }
                }
            for (unsigned int k = 0; k < 6; k++)
                {
                external_virial[k] += constraint_force->getExternalVirial(k);
                }

            external_energy += constraint_force->getExternalEnergy();
            }
        }

    for (unsigned int k = 0; k < 6; k++)
        {
        m_pdata->setExternalVirial(k, external_virial[k]);
        }

    m_pdata->setExternalEnergy(external_energy);
    }

#ifdef ENABLE_HIP
/** @param timestep Current time step of the simulation
    \post All added force computes in \a m_forces are computed and totaled up in \a m_net_force and
   \a m_net_virial \note The summation step is performed <b>on the GPU</b>.
*/
void Integrator::computeNetForceGPU(uint64_t timestep)
    {
    if (!m_exec_conf->isCUDAEnabled())
        {
        throw runtime_error("Cannot compute net force on the GPU if CUDA is disabled.");
        }

    // compute all the normal forces first

    for (auto& force : m_forces)
        {
        force->compute(timestep);
        }

    Scalar external_virial[6];
    Scalar external_energy;

        {
        // access the net force and virial arrays
        const GlobalArray<Scalar4>& net_force = m_pdata->getNetForce();
        const GlobalArray<Scalar4>& net_torque = m_pdata->getNetTorqueArray();
        const GlobalArray<Scalar>& net_virial = m_pdata->getNetVirial();
        size_t net_virial_pitch = net_virial.getPitch();

        ArrayHandle<Scalar4> d_net_force(net_force,
                                         access_location::device,
                                         access_mode::overwrite);
        ArrayHandle<Scalar> d_net_virial(net_virial,
                                         access_location::device,
                                         access_mode::overwrite);
        ArrayHandle<Scalar4> d_net_torque(net_torque,
                                          access_location::device,
                                          access_mode::overwrite);

        // also sum up forces for ghosts, in case they are needed by the communicator
        unsigned int nparticles = m_pdata->getN() + m_pdata->getNGhosts();
        assert(nparticles <= net_force.getNumElements());
        assert(nparticles * 6 <= net_virial.getNumElements());
        assert(nparticles <= net_torque.getNumElements());

        // zero external virial
        for (unsigned int i = 0; i < 6; ++i)
            external_virial[i] = Scalar(0.0);

        external_energy = Scalar(0.0);

        // there is no need to zero out the initial net force and virial here, the first call to the
        // addition kernel will do that ahh!, but we do need to zer out the net force and virial if
        // there are 0 forces!
        if (m_forces.size() == 0)
            {
            // start by zeroing the net force and virial arrays
            hipMemset(d_net_force.data, 0, sizeof(Scalar4) * net_force.getNumElements());
            hipMemset(d_net_torque.data, 0, sizeof(Scalar4) * net_torque.getNumElements());
            hipMemset(d_net_virial.data, 0, 6 * sizeof(Scalar) * net_virial_pitch);
            if (m_exec_conf->isCUDAErrorCheckingEnabled())
                CHECK_CUDA_ERROR();
            }

        // now, add up the accelerations
        // sum all the forces into the net force
        // perform the sum in groups of 6 to avoid kernel launch and memory access overheads
        for (unsigned int cur_force = 0; cur_force < m_forces.size(); cur_force += 6)
            {
            // grab the device pointers for the current set
            kernel::gpu_force_list force_list;

            const GlobalArray<Scalar4>& d_force_array0 = m_forces[cur_force]->getForceArray();
            ArrayHandle<Scalar4> d_force0(d_force_array0,
                                          access_location::device,
                                          access_mode::read);
            const GlobalArray<Scalar>& d_virial_array0 = m_forces[cur_force]->getVirialArray();
            ArrayHandle<Scalar> d_virial0(d_virial_array0,
                                          access_location::device,
                                          access_mode::read);
            const GlobalArray<Scalar4>& d_torque_array0 = m_forces[cur_force]->getTorqueArray();
            ArrayHandle<Scalar4> d_torque0(d_torque_array0,
                                           access_location::device,
                                           access_mode::read);
            force_list.f0 = d_force0.data;
            force_list.v0 = d_virial0.data;
            force_list.vpitch0 = d_virial_array0.getPitch();
            force_list.t0 = d_torque0.data;

            if (cur_force + 1 < m_forces.size())
                {
                const GlobalArray<Scalar4>& d_force_array1
                    = m_forces[cur_force + 1]->getForceArray();
                ArrayHandle<Scalar4> d_force1(d_force_array1,
                                              access_location::device,
                                              access_mode::read);
                const GlobalArray<Scalar>& d_virial_array1
                    = m_forces[cur_force + 1]->getVirialArray();
                ArrayHandle<Scalar> d_virial1(d_virial_array1,
                                              access_location::device,
                                              access_mode::read);
                const GlobalArray<Scalar4>& d_torque_array1
                    = m_forces[cur_force + 1]->getTorqueArray();
                ArrayHandle<Scalar4> d_torque1(d_torque_array1,
                                               access_location::device,
                                               access_mode::read);
                force_list.f1 = d_force1.data;
                force_list.v1 = d_virial1.data;
                force_list.vpitch1 = d_virial_array1.getPitch();
                force_list.t1 = d_torque1.data;
                }
            if (cur_force + 2 < m_forces.size())
                {
                const GlobalArray<Scalar4>& d_force_array2
                    = m_forces[cur_force + 2]->getForceArray();
                ArrayHandle<Scalar4> d_force2(d_force_array2,
                                              access_location::device,
                                              access_mode::read);
                const GlobalArray<Scalar>& d_virial_array2
                    = m_forces[cur_force + 2]->getVirialArray();
                ArrayHandle<Scalar> d_virial2(d_virial_array2,
                                              access_location::device,
                                              access_mode::read);
                const GlobalArray<Scalar4>& d_torque_array2
                    = m_forces[cur_force + 2]->getTorqueArray();
                ArrayHandle<Scalar4> d_torque2(d_torque_array2,
                                               access_location::device,
                                               access_mode::read);
                force_list.f2 = d_force2.data;
                force_list.v2 = d_virial2.data;
                force_list.vpitch2 = d_virial_array2.getPitch();
                force_list.t2 = d_torque2.data;
                }
            if (cur_force + 3 < m_forces.size())
                {
                const GlobalArray<Scalar4>& d_force_array3
                    = m_forces[cur_force + 3]->getForceArray();
                ArrayHandle<Scalar4> d_force3(d_force_array3,
                                              access_location::device,
                                              access_mode::read);
                const GlobalArray<Scalar>& d_virial_array3
                    = m_forces[cur_force + 3]->getVirialArray();
                ArrayHandle<Scalar> d_virial3(d_virial_array3,
                                              access_location::device,
                                              access_mode::read);
                const GlobalArray<Scalar4>& d_torque_array3
                    = m_forces[cur_force + 3]->getTorqueArray();
                ArrayHandle<Scalar4> d_torque3(d_torque_array3,
                                               access_location::device,
                                               access_mode::read);
                force_list.f3 = d_force3.data;
                force_list.v3 = d_virial3.data;
                force_list.vpitch3 = d_virial_array3.getPitch();
                force_list.t3 = d_torque3.data;
                }
            if (cur_force + 4 < m_forces.size())
                {
                const GlobalArray<Scalar4>& d_force_array4
                    = m_forces[cur_force + 4]->getForceArray();
                ArrayHandle<Scalar4> d_force4(d_force_array4,
                                              access_location::device,
                                              access_mode::read);
                const GlobalArray<Scalar>& d_virial_array4
                    = m_forces[cur_force + 4]->getVirialArray();
                ArrayHandle<Scalar> d_virial4(d_virial_array4,
                                              access_location::device,
                                              access_mode::read);
                const GlobalArray<Scalar4>& d_torque_array4
                    = m_forces[cur_force + 4]->getTorqueArray();
                ArrayHandle<Scalar4> d_torque4(d_torque_array4,
                                               access_location::device,
                                               access_mode::read);
                force_list.f4 = d_force4.data;
                force_list.v4 = d_virial4.data;
                force_list.vpitch4 = d_virial_array4.getPitch();
                force_list.t4 = d_torque4.data;
                }
            if (cur_force + 5 < m_forces.size())
                {
                const GlobalArray<Scalar4>& d_force_array5
                    = m_forces[cur_force + 5]->getForceArray();
                ArrayHandle<Scalar4> d_force5(d_force_array5,
                                              access_location::device,
                                              access_mode::read);
                const GlobalArray<Scalar>& d_virial_array5
                    = m_forces[cur_force + 5]->getVirialArray();
                ArrayHandle<Scalar> d_virial5(d_virial_array5,
                                              access_location::device,
                                              access_mode::read);
                const GlobalArray<Scalar4>& d_torque_array5
                    = m_forces[cur_force + 5]->getTorqueArray();
                ArrayHandle<Scalar4> d_torque5(d_torque_array5,
                                               access_location::device,
                                               access_mode::read);
                force_list.f5 = d_force5.data;
                force_list.v5 = d_virial5.data;
                force_list.vpitch5 = d_virial_array5.getPitch();
                force_list.t5 = d_torque5.data;
                }

            // clear on the first iteration only
            bool clear = (cur_force == 0);

            // access flags
            PDataFlags flags = this->m_pdata->getFlags();

            m_exec_conf->beginMultiGPU();

            gpu_integrator_sum_net_force(d_net_force.data,
                                         d_net_virial.data,
                                         net_virial_pitch,
                                         d_net_torque.data,
                                         force_list,
                                         nparticles,
                                         clear,
                                         flags[pdata_flag::pressure_tensor],
                                         m_pdata->getGPUPartition());

            if (m_exec_conf->isCUDAErrorCheckingEnabled())
                CHECK_CUDA_ERROR();

            m_exec_conf->endMultiGPU();
            }
        }

    // add up external virials and energies
    for (const auto& force : m_forces)
        {
        for (unsigned int k = 0; k < 6; k++)
            external_virial[k] += force->getExternalVirial(k);
        external_energy += force->getExternalEnergy();
        }

    for (unsigned int k = 0; k < 6; k++)
        m_pdata->setExternalVirial(k, external_virial[k]);

    m_pdata->setExternalEnergy(external_energy);

    // return early if there are no constraint forces or no HalfStepHook set
    if (m_constraint_forces.size() == 0)
        return;

#ifdef ENABLE_MPI
    if (m_sysdef->isDomainDecomposed())
        {
        // communicate the net force
        m_comm->updateNetForce(timestep);
        }
#endif

    // compute all the constraint forces next
    for (auto& constraint_force : m_constraint_forces)
        {
        constraint_force->compute(timestep);
        }

        {
        // access the net force and virial arrays
        const GlobalArray<Scalar4>& net_force = m_pdata->getNetForce();
        const GlobalArray<Scalar>& net_virial = m_pdata->getNetVirial();
        const GlobalArray<Scalar4>& net_torque = m_pdata->getNetTorqueArray();
        ArrayHandle<Scalar4> d_net_force(net_force,
                                         access_location::device,
                                         access_mode::readwrite);
        ArrayHandle<Scalar> d_net_virial(net_virial,
                                         access_location::device,
                                         access_mode::readwrite);
        ArrayHandle<Scalar4> d_net_torque(net_torque,
                                          access_location::device,
                                          access_mode::readwrite);

        unsigned int nparticles = m_pdata->getN();
        assert(nparticles <= net_force.getNumElements());
        assert(6 * nparticles <= net_virial.getNumElements());
        assert(nparticles <= net_torque.getNumElements());

        // now, add up the accelerations
        // sum all the forces into the net force
        // perform the sum in groups of 6 to avoid kernel launch and memory access overheads
        for (unsigned int cur_force = 0; cur_force < m_constraint_forces.size(); cur_force += 6)
            {
            // grab the device pointers for the current set
            kernel::gpu_force_list force_list;
            const GlobalArray<Scalar4>& d_force_array0
                = m_constraint_forces[cur_force]->getForceArray();
            ArrayHandle<Scalar4> d_force0(d_force_array0,
                                          access_location::device,
                                          access_mode::read);
            const GlobalArray<Scalar>& d_virial_array0
                = m_constraint_forces[cur_force]->getVirialArray();
            ArrayHandle<Scalar> d_virial0(d_virial_array0,
                                          access_location::device,
                                          access_mode::read);
            const GlobalArray<Scalar4>& d_torque_array0
                = m_constraint_forces[cur_force]->getTorqueArray();
            ArrayHandle<Scalar4> d_torque0(d_torque_array0,
                                           access_location::device,
                                           access_mode::read);
            force_list.f0 = d_force0.data;
            force_list.t0 = d_torque0.data;
            force_list.v0 = d_virial0.data;
            force_list.vpitch0 = d_virial_array0.getPitch();

            if (cur_force + 1 < m_constraint_forces.size())
                {
                const GlobalArray<Scalar4>& d_force_array1
                    = m_constraint_forces[cur_force + 1]->getForceArray();
                ArrayHandle<Scalar4> d_force1(d_force_array1,
                                              access_location::device,
                                              access_mode::read);
                const GlobalArray<Scalar>& d_virial_array1
                    = m_constraint_forces[cur_force + 1]->getVirialArray();
                ArrayHandle<Scalar> d_virial1(d_virial_array1,
                                              access_location::device,
                                              access_mode::read);
                const GlobalArray<Scalar4>& d_torque_array1
                    = m_constraint_forces[cur_force + 1]->getTorqueArray();
                ArrayHandle<Scalar4> d_torque1(d_torque_array1,
                                               access_location::device,
                                               access_mode::read);
                force_list.f1 = d_force1.data;
                force_list.t1 = d_torque1.data;
                force_list.v1 = d_virial1.data;
                force_list.vpitch1 = d_virial_array1.getPitch();
                }
            if (cur_force + 2 < m_constraint_forces.size())
                {
                const GlobalArray<Scalar4>& d_force_array2
                    = m_constraint_forces[cur_force + 2]->getForceArray();
                ArrayHandle<Scalar4> d_force2(d_force_array2,
                                              access_location::device,
                                              access_mode::read);
                const GlobalArray<Scalar>& d_virial_array2
                    = m_constraint_forces[cur_force + 2]->getVirialArray();
                ArrayHandle<Scalar> d_virial2(d_virial_array2,
                                              access_location::device,
                                              access_mode::read);
                const GlobalArray<Scalar4>& d_torque_array2
                    = m_constraint_forces[cur_force + 2]->getTorqueArray();
                ArrayHandle<Scalar4> d_torque2(d_torque_array2,
                                               access_location::device,
                                               access_mode::read);
                force_list.f2 = d_force2.data;
                force_list.t2 = d_torque2.data;
                force_list.v2 = d_virial2.data;
                force_list.vpitch2 = d_virial_array2.getPitch();
                }
            if (cur_force + 3 < m_constraint_forces.size())
                {
                const GlobalArray<Scalar4>& d_force_array3
                    = m_constraint_forces[cur_force + 3]->getForceArray();
                ArrayHandle<Scalar4> d_force3(d_force_array3,
                                              access_location::device,
                                              access_mode::read);
                const GlobalArray<Scalar>& d_virial_array3
                    = m_constraint_forces[cur_force + 3]->getVirialArray();
                ArrayHandle<Scalar> d_virial3(d_virial_array3,
                                              access_location::device,
                                              access_mode::read);
                const GlobalArray<Scalar4>& d_torque_array3
                    = m_constraint_forces[cur_force + 3]->getTorqueArray();
                ArrayHandle<Scalar4> d_torque3(d_torque_array3,
                                               access_location::device,
                                               access_mode::read);
                force_list.f3 = d_force3.data;
                force_list.t3 = d_torque3.data;
                force_list.v3 = d_virial3.data;
                force_list.vpitch3 = d_virial_array3.getPitch();
                }
            if (cur_force + 4 < m_constraint_forces.size())
                {
                const GlobalArray<Scalar4>& d_force_array4
                    = m_constraint_forces[cur_force + 4]->getForceArray();
                ArrayHandle<Scalar4> d_force4(d_force_array4,
                                              access_location::device,
                                              access_mode::read);
                const GlobalArray<Scalar>& d_virial_array4
                    = m_constraint_forces[cur_force + 4]->getVirialArray();
                ArrayHandle<Scalar> d_virial4(d_virial_array4,
                                              access_location::device,
                                              access_mode::read);
                const GlobalArray<Scalar4>& d_torque_array4
                    = m_constraint_forces[cur_force + 4]->getTorqueArray();
                ArrayHandle<Scalar4> d_torque4(d_torque_array4,
                                               access_location::device,
                                               access_mode::read);
                force_list.f4 = d_force4.data;
                force_list.t4 = d_torque4.data;
                force_list.v4 = d_virial4.data;
                force_list.vpitch4 = d_virial_array4.getPitch();
                }
            if (cur_force + 5 < m_constraint_forces.size())
                {
                const GlobalArray<Scalar4>& d_force_array5
                    = m_constraint_forces[cur_force + 5]->getForceArray();
                ArrayHandle<Scalar4> d_force5(d_force_array5,
                                              access_location::device,
                                              access_mode::read);
                const GlobalArray<Scalar>& d_virial_array5
                    = m_constraint_forces[cur_force + 5]->getVirialArray();
                ArrayHandle<Scalar> d_virial5(d_virial_array5,
                                              access_location::device,
                                              access_mode::read);
                const GlobalArray<Scalar4>& d_torque_array5
                    = m_constraint_forces[cur_force + 5]->getTorqueArray();
                ArrayHandle<Scalar4> d_torque5(d_torque_array5,
                                               access_location::device,
                                               access_mode::read);
                force_list.f5 = d_force5.data;
                force_list.t5 = d_torque5.data;
                force_list.v5 = d_virial5.data;
                force_list.vpitch5 = d_virial_array5.getPitch();
                }

            // clear only on the first iteration AND if there are zero forces
            bool clear = (cur_force == 0) && (m_forces.size() == 0);

            // access flags
            PDataFlags flags = this->m_pdata->getFlags();

            m_exec_conf->beginMultiGPU();

            gpu_integrator_sum_net_force(d_net_force.data,
                                         d_net_virial.data,
                                         net_virial.getPitch(),
                                         d_net_torque.data,
                                         force_list,
                                         nparticles,
                                         clear,
                                         flags[pdata_flag::pressure_tensor],
                                         m_pdata->getGPUPartition());

            if (m_exec_conf->isCUDAErrorCheckingEnabled())
                CHECK_CUDA_ERROR();

            m_exec_conf->endMultiGPU();
            }
        }

    // add up external virials
    for (const auto& constraint_force : m_constraint_forces)
        {
        for (unsigned int k = 0; k < 6; k++)
            {
            external_virial[k] += constraint_force->getExternalVirial(k);
            }
        external_energy += constraint_force->getExternalEnergy();
        }

    for (unsigned int k = 0; k < 6; k++)
        m_pdata->setExternalVirial(k, external_virial[k]);

    m_pdata->setExternalEnergy(external_energy);
    }
#endif

/** The base class integrator actually does nothing in update()
    @param timestep Current time step of the simulation
*/
void Integrator::update(uint64_t timestep)
    {
    Updater::update(timestep);

    // ensure that the force computes know the current step size
    for (auto& force : m_forces)
        {
        force->setDeltaT(m_deltaT);
        }

    for (auto& constraint_force : m_constraint_forces)
        {
        constraint_force->setDeltaT(m_deltaT);
        }
    }

/** prepRun() is to be called at the very beginning of each run, before any analyzers are called,
   but after the full simulation is defined. It allows the integrator to perform any one-off setup
   tasks and update net_force and net_virial, if needed.

    Specifically, updated net_force and net_virial in this call is a must to properly carry
    over in restarted jobs.

    The base class does nothing, it is up to derived classes to implement the correct behavior.
*/
void Integrator::prepRun(uint64_t timestep)
    {
    // ensure that all forces have updated delta t values at the start of step 0

    for (auto& force : m_forces)
        {
        force->setDeltaT(m_deltaT);
        }

    for (auto& constraint_force : m_constraint_forces)
        {
        constraint_force->setDeltaT(m_deltaT);
        }
    }

#ifdef ENABLE_MPI
/** @param tstep Time step for which to determine the flags

    The flags needed are determined by peeking to \a tstep and then using bitwise or
    to combine the flags from all ForceComputes
*/
CommFlags Integrator::determineFlags(uint64_t timestep)
    {
    CommFlags flags(0);

    // query all forces
    for (const auto& force : m_forces)
        {
        flags |= force->getRequestedCommFlags(timestep);
        }

    // query all constraints
    for (const auto& constraint_force : m_constraint_forces)
        {
        flags |= constraint_force->getRequestedCommFlags(timestep);
        }

    return flags;
    }

void Integrator::computeCallback(uint64_t timestep)
    {
    // pre-compute all active forces
    for (auto& force : m_forces)
        {
        force->preCompute(timestep);
        }
    }
#endif

bool Integrator::areForcesAnisotropic()
    {
    bool aniso = false;

    for (const auto& force : m_forces)
        {
        aniso |= force->isAnisotropic();
        }

    for (const auto& constraint_force : m_constraint_forces)
        {
        aniso |= constraint_force->isAnisotropic();
        }

    return aniso;
    }

namespace detail
    {
void export_Integrator(pybind11::module& m)
    {
    pybind11::bind_vector<std::vector<std::shared_ptr<ForceCompute>>>(m, "ForceComputeList");
    pybind11::bind_vector<std::vector<std::shared_ptr<ForceConstraint>>>(m, "ForceConstraintList");
    pybind11::class_<Integrator, Updater, std::shared_ptr<Integrator>>(m, "Integrator")
        .def(pybind11::init<std::shared_ptr<SystemDefinition>, Scalar>())
        .def("updateGroupDOF", &Integrator::updateGroupDOF)
        .def_property("dt", &Integrator::getDeltaT, &Integrator::setDeltaT)
        .def_property_readonly("forces", &Integrator::getForces)
        .def_property_readonly("constraints", &Integrator::getConstraintForces)
        .def("computeLinearMomentum", &Integrator::computeLinearMomentum);
    }

    } // end namespace detail

    } // end namespace hoomd
