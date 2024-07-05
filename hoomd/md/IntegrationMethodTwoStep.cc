// Copyright (c) 2009-2024 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

#include "IntegrationMethodTwoStep.h"
#include "hoomd/HOOMDMath.h"
#include "hoomd/VectorMath.h"

#ifdef ENABLE_MPI
#include "hoomd/Communicator.h"
#endif

using namespace std;

/*! \file IntegrationMethodTwoStep.h
    \brief Contains code for the IntegrationMethodTwoStep class
*/

namespace hoomd
    {
namespace md
    {
/*! \param sysdef SystemDefinition this method will act on. Must not be NULL.
    \param group The group of particles this integration method is to work on
    \post The method is constructed with the given particle.
*/
IntegrationMethodTwoStep::IntegrationMethodTwoStep(std::shared_ptr<SystemDefinition> sysdef,
                                                   std::shared_ptr<ParticleGroup> group)
    : m_sysdef(sysdef), m_group(group), m_pdata(m_sysdef->getParticleData()),
      m_exec_conf(m_pdata->getExecConf()), m_aniso(false), m_deltaT(Scalar(0.0))
    {
    // sanity check
    assert(m_sysdef);
    assert(m_pdata);
    assert(m_group);
    }

/*! \param deltaT New time step to set
 */
void IntegrationMethodTwoStep::setDeltaT(Scalar deltaT)
    {
    m_deltaT = deltaT;
    }

/*! \param query_group Group over which to count (translational) degrees of freedom.
    A majority of the integration methods add D degrees of freedom per particle in \a query_group
   that is also in the group assigned to the method. Hence, the base class IntegrationMethodTwoStep
   will implement that counting. Derived classes can override if needed.
*/
Scalar IntegrationMethodTwoStep::getTranslationalDOF(std::shared_ptr<ParticleGroup> query_group)
    {
    // get the size of the intersection between query_group and m_group
    unsigned int intersect_size = query_group->intersectionSize(m_group);

    return m_sysdef->getNDimensions() * intersect_size;
    }

Scalar IntegrationMethodTwoStep::getRotationalDOF(std::shared_ptr<ParticleGroup> query_group)
    {
    unsigned int query_group_dof = 0;
    unsigned int dimension = m_sysdef->getNDimensions();
    ArrayHandle<Scalar3> h_moment_inertia(m_pdata->getMomentsOfInertiaArray(),
                                          access_location::host,
                                          access_mode::read);

    for (unsigned int group_idx = 0; group_idx < query_group->getNumMembers(); group_idx++)
        {
        unsigned int j = query_group->getMemberIndex(group_idx);
        if (m_group->isMember(j))
            {
            if (dimension == 3)
                {
                if (fabs(h_moment_inertia.data[j].x) > 0)
                    query_group_dof++;

                if (fabs(h_moment_inertia.data[j].y) > 0)
                    query_group_dof++;

                if (fabs(h_moment_inertia.data[j].z) > 0)
                    query_group_dof++;
                }
            else
                {
                if (fabs(h_moment_inertia.data[j].z) > 0)
                    query_group_dof++;
                }
            }
        }

#ifdef ENABLE_MPI
    if (m_pdata->getDomainDecomposition())
        {
        MPI_Allreduce(MPI_IN_PLACE,
                      &query_group_dof,
                      1,
                      MPI_UNSIGNED,
                      MPI_SUM,
                      m_exec_conf->getMPICommunicator());
        }
#endif

    return query_group_dof;
    }

/*! Checks that every particle in the group is valid. This method may be called by anyone wishing to
   make this error check.

    The base class does nothing
*/
void IntegrationMethodTwoStep::validateGroup()
    {
    ArrayHandle<unsigned int> h_body(m_pdata->getBodies(),
                                     access_location::host,
                                     access_mode::read);
    ArrayHandle<unsigned int> h_rtag(m_pdata->getRTags(), access_location::host, access_mode::read);
    ArrayHandle<unsigned int> h_tag(m_pdata->getTags(), access_location::host, access_mode::read);
    ArrayHandle<unsigned int> h_group_index(m_group->getIndexArray(),
                                            access_location::host,
                                            access_mode::read);

    unsigned int error = 0;
    for (unsigned int gidx = 0; gidx < m_group->getNumMembers(); gidx++)
        {
        unsigned int i = h_group_index.data[gidx];
        unsigned int tag = h_tag.data[i];
        unsigned int body = h_body.data[i];

        if (body < MIN_FLOPPY && body != tag)
            {
            error = 1;
            }
        }

#ifdef ENABLE_MPI
    if (this->m_sysdef->isDomainDecomposed())
        {
        MPI_Allreduce(MPI_IN_PLACE,
                      &error,
                      1,
                      MPI_UNSIGNED,
                      MPI_LOR,
                      this->m_exec_conf->getMPICommunicator());
        }
#endif

    if (error)
        {
        throw std::runtime_error("Integration methods may not be applied to constituents.");
        }
    }

namespace detail
    {
void export_IntegrationMethodTwoStep(pybind11::module& m)
    {
    pybind11::class_<IntegrationMethodTwoStep,
                     Autotuned,
                     std::shared_ptr<IntegrationMethodTwoStep>>(m, "IntegrationMethodTwoStep")
        .def(pybind11::init<std::shared_ptr<SystemDefinition>, std::shared_ptr<ParticleGroup>>())
        .def("validateGroup", &IntegrationMethodTwoStep::validateGroup)
        .def_property_readonly("filter",
                               [](const std::shared_ptr<IntegrationMethodTwoStep> method)
                               { return method->getGroup()->getFilter(); });
    }

    } // end namespace detail
    } // end namespace md
    } // end namespace hoomd
