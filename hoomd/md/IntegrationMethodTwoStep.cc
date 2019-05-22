// Copyright (c) 2009-2019 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.


// Maintainer: joaander



#include "IntegrationMethodTwoStep.h"
#include "hoomd/VectorMath.h"
#include "QuaternionMath.h"
#include "hoomd/HOOMDMath.h"

#include "hoomd/RandomNumbers.h"
#include "hoomd/RNGIdentifiers.h"

namespace py = pybind11;

#ifdef ENABLE_MPI
#include "hoomd/Communicator.h"
#endif

using namespace std;

/*! \file IntegrationMethodTwoStep.h
    \brief Contains code for the IntegrationMethodTwoStep class
*/

/*! \param sysdef SystemDefinition this method will act on. Must not be NULL.
    \param group The group of particles this integration method is to work on
    \post The method is constructed with the given particle data and a NULL profiler.
*/
IntegrationMethodTwoStep::IntegrationMethodTwoStep(std::shared_ptr<SystemDefinition> sysdef,
                                                   std::shared_ptr<ParticleGroup> group)
    : m_sysdef(sysdef), m_group(group), m_pdata(m_sysdef->getParticleData()), m_exec_conf(m_pdata->getExecConf()),
      m_aniso(false), m_deltaT(Scalar(0.0)), m_valid_restart(false)
    {
    // sanity check
    assert(m_sysdef);
    assert(m_pdata);
    assert(m_group);

    m_integrator_id = m_sysdef->getIntegratorData()->registerIntegrator();
    }

/*! It is useful for the user to know where computation time is spent, so all integration methods
    should profile themselves. This method sets the profiler for them to use.
    This method does not need to be called, as Computes will not profile themselves
    on a NULL profiler
    \param prof Pointer to a profiler for the compute to use. Set to NULL
        (std::shared_ptr<Profiler>()) to stop the
        analyzer from profiling itself.
    \note Derived classes MUST check if m_prof is set before calling any profiler methods.
*/
void IntegrationMethodTwoStep::setProfiler(std::shared_ptr<Profiler> prof)
    {
    m_prof = prof;
    }

/*! \param deltaT New time step to set
*/
void IntegrationMethodTwoStep::setDeltaT(Scalar deltaT)
    {
    m_deltaT = deltaT;
    }


/*! \param v is the restart variables for the current integrator
    \param type is the type of expected integrator type
    \param nvariables is the expected number of variables

    If the either the integrator type or number of variables does not match the
    expected values, this function throws the appropriate warning and returns
    "false."  Otherwise, the function returns true.
*/
bool IntegrationMethodTwoStep::restartInfoTestValid(const IntegratorVariables& v, std::string type, unsigned int nvariables)
    {
    bool good = true;
    if (v.type == "")
        good = false;
    else if (v.type != type && v.type != "")
        {
        m_exec_conf->msg->warning() << "Integrator #"<<  m_integrator_id <<" type "<< type <<" does not match type ";
        m_exec_conf->msg->warning() << v.type << " found in restart file. " << endl;
        m_exec_conf->msg->warning() << "Ensure that the integrator order is consistent for restarted simulations. " << endl;
        m_exec_conf->msg->warning() << "Continuing while ignoring restart information..." << endl;
        good = false;
        }
    else if (v.type == type)
        {
        if (v.variable.size() != nvariables)
            {
            m_exec_conf->msg->warning() << "Integrator #"<< m_integrator_id <<" type "<< type << endl;
            m_exec_conf->msg->warning() << "appears to contain bad or incomplete restart information. " << endl;
            m_exec_conf->msg->warning() << "Continuing while ignoring restart information..." << endl;
            good = false;
            }
        }
    return good;
    }

/*! \param query_group Group over which to count (translational) degrees of freedom.
    A majority of the integration methods add D degrees of freedom per particle in \a query_group that is also in the
    group assigned to the method. Hence, the base class IntegrationMethodTwoStep will implement that counting.
    Derived classes can override if needed.
*/
unsigned int IntegrationMethodTwoStep::getNDOF(std::shared_ptr<ParticleGroup> query_group)
    {
    // get the size of the intersection between query_group and m_group
    unsigned int intersect_size = ParticleGroup::groupIntersection(query_group, m_group)->getNumMembersGlobal();

    return m_sysdef->getNDimensions() * intersect_size;
    }

unsigned int IntegrationMethodTwoStep::getRotationalNDOF(std::shared_ptr<ParticleGroup> query_group)
    {
    // get the size of the intersection between query_group and m_group
    std::shared_ptr<ParticleGroup> intersect = ParticleGroup::groupIntersection(query_group, m_group);

    unsigned int local_group_size = intersect->getNumMembers();

    unsigned int query_group_dof = 0;
    unsigned int dimension = m_sysdef->getNDimensions();
    unsigned int dof_one;
    ArrayHandle<Scalar3> h_moment_inertia(m_pdata->getMomentsOfInertiaArray(), access_location::host, access_mode::read);

    for (unsigned int group_idx = 0; group_idx < local_group_size; group_idx++)
        {
        unsigned int j = intersect->getMemberIndex(group_idx);
        if (dimension == 3)
            {
            dof_one = 3;
            if (fabs(h_moment_inertia.data[j].x) < EPSILON)
                dof_one--;

            if (fabs(h_moment_inertia.data[j].y) < EPSILON)
                dof_one--;

            if (fabs(h_moment_inertia.data[j].z) < EPSILON)
                dof_one--;
            }
        else
            {
            dof_one = 1;
            if (fabs(h_moment_inertia.data[j].z) < EPSILON)
                dof_one--;
            }

        query_group_dof += dof_one;
        }

    #ifdef ENABLE_MPI
    if (m_pdata->getDomainDecomposition())
        {
        MPI_Allreduce(MPI_IN_PLACE, &query_group_dof, 1, MPI_UNSIGNED, MPI_SUM, m_exec_conf->getMPICommunicator());
        }
    #endif

    return query_group_dof;
    }

/*! Checks that every particle in the group is valid. This method may be called by anyone wishing to make this
    error check.

    The base class does nothing
*/
void IntegrationMethodTwoStep::validateGroup()
    {
    ArrayHandle<unsigned int> h_body(m_pdata->getBodies(), access_location::host, access_mode::read);
    ArrayHandle<unsigned int> h_rtag(m_pdata->getRTags(), access_location::host, access_mode::read);
    ArrayHandle<unsigned int> h_tag(m_pdata->getTags(), access_location::host, access_mode::read);
    ArrayHandle<unsigned int> h_group_index(m_group->getIndexArray(), access_location::host, access_mode::read);

    for (unsigned int gidx = 0; gidx < m_group->getNumMembers(); gidx++)
        {
        unsigned int i = h_group_index.data[gidx];
        unsigned int tag = h_tag.data[i];
        unsigned int body = h_body.data[i];

        if (body < MIN_FLOPPY && body != tag)
            {
            m_exec_conf->msg->error() << "Particle " << tag << " belongs to a rigid body, but is not its center particle. "
                << std::endl << "This integration method does not operate on constituent particles."
                << std::endl << std::endl;
            throw std::runtime_error("Error initializing integration method");
            }
        }

    }

/*!
   Randomizes linear velocities and angular momenta by sampling the
   Maxwell-Boltzmann (MB) distribution.
   The user provides three arguments:
   - The particle group
   - Temperature
   - Seed for initializing the random number generator used for this purpose
   These variables are member variables of the class.
*/
void IntegrationMethodTwoStep::randomizeVelocities(unsigned int timestep)
    {
    if (m_shouldRandomize == false)
        {
        return;
        }

    /* Get the number of particles in the group */
    unsigned int group_size = m_group->getNumMembers();

    /* Grab some variables */
    const unsigned int D = Scalar(m_sysdef->getNDimensions());

    ArrayHandle<Scalar4> h_vel(m_pdata->getVelocities(),
                               access_location::host,
                               access_mode::readwrite);

    ArrayHandle<Scalar4> h_orientation(m_pdata->getOrientationArray(),
                                       access_location::host,
                                       access_mode::read);

    ArrayHandle<Scalar4> h_angmom(m_pdata->getAngularMomentumArray(),
                                  access_location::host,
                                  access_mode::readwrite);

    ArrayHandle<Scalar3> h_inertia(m_pdata->getMomentsOfInertiaArray(),
                                   access_location::host,
                                   access_mode::read);

    ArrayHandle<unsigned int> h_tag(m_pdata->getTags(),
                                    access_location::host,
                                    access_mode::read);

    /* Total momentum */
    vec3<Scalar> tot_momentum(0, 0, 0);

    /* Loop over all the particles in the group */
    for (unsigned int group_idx = 0; group_idx < group_size; group_idx++)
        {
        unsigned int j = m_group->getMemberIndex(group_idx);
        unsigned int ptag = h_tag.data[j];

        /* Initialize the random number generator */
        hoomd::RandomGenerator rng(hoomd::RNGIdentifier::IntegrationMethodTwoStep, m_seed_randomize, ptag, timestep);

        /* Generate a new random linear velocity for particle j */
        Scalar mass =  h_vel.data[j].w;
        Scalar sigma = fast::sqrt(m_T_randomize / mass);
        hoomd::NormalDistribution<Scalar> normal(sigma);
        h_vel.data[j].x = normal(rng);
        h_vel.data[j].y = normal(rng);
        if (D > 2)
            h_vel.data[j].z = normal(rng);
        else
            h_vel.data[j].z = 0; // For 2D systems

        tot_momentum += mass * vec3<Scalar>(h_vel.data[j]);

        /* Generate a new random angular momentum if the particle is a rigid
         * body and anisotropy flag gets set.
         * There may be some issues for 2D systems */
        if (m_aniso)
            {
            vec3<Scalar> p_vec(0,0,0);
            quat<Scalar> q(h_orientation.data[j]);
            vec3<Scalar> I(h_inertia.data[j]);

            /* Generate a new random angular momentum for particle j in
             * body frame */
            if (I.x >= EPSILON)
                p_vec.x = hoomd::NormalDistribution<Scalar>(fast::sqrt(m_T_randomize * I.x))(rng);
            if (I.y >= EPSILON)
                p_vec.y = hoomd::NormalDistribution<Scalar>(fast::sqrt(m_T_randomize * I.y))(rng);
            if (I.z >= EPSILON)
                p_vec.z = hoomd::NormalDistribution<Scalar>(fast::sqrt(m_T_randomize * I.z))(rng);

            /* Store the angular momentum quaternion */
            quat<Scalar> p = Scalar(2.0) * q * p_vec;
            h_angmom.data[j] = quat_to_scalar4(p);
            }
        }

    /* Remove the drift i.e. remove the center of mass velocity */

    #ifdef ENABLE_MPI
    // Reduce the total momentum from all MPI ranks
    if (m_comm)
       {
       MPI_Allreduce(MPI_IN_PLACE, &tot_momentum, 3, MPI_HOOMD_SCALAR,
                     MPI_SUM, m_exec_conf->getMPICommunicator());
       }
    #endif

    vec3<Scalar> com_momentum(tot_momentum /
                              Scalar(m_group->getNumMembersGlobal()));

    for (unsigned int group_idx = 0; group_idx < group_size; group_idx++)
        {
        unsigned int j = m_group->getMemberIndex(group_idx);
        Scalar mass =  h_vel.data[j].w;
        h_vel.data[j].x = h_vel.data[j].x - com_momentum.x / mass;
        h_vel.data[j].y = h_vel.data[j].y - com_momentum.y / mass;
        if (D > 2)
            h_vel.data[j].z = h_vel.data[j].z - com_momentum.z / mass;
        else
            h_vel.data[j].z = 0; // For 2D systems
        }

    /* Done randomizing velocities */

    /* Reset the flag */
    m_shouldRandomize = false;
    }

void export_IntegrationMethodTwoStep(py::module& m)
    {
    py::class_<IntegrationMethodTwoStep, std::shared_ptr<IntegrationMethodTwoStep> >(m, "IntegrationMethodTwoStep")
        .def(py::init< std::shared_ptr<SystemDefinition>, std::shared_ptr<ParticleGroup> >())
        .def("validateGroup", &IntegrationMethodTwoStep::validateGroup)
#ifdef ENABLE_MPI
        .def("setCommunicator", &IntegrationMethodTwoStep::setCommunicator)
#endif
        .def("setRandomizeVelocitiesParams", &IntegrationMethodTwoStep::setRandomizeVelocitiesParams)
        ;
    }
