// Copyright (c) 2009-2024 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

#include "Compute.h"
#include "Communicator.h"

#include <iostream>
#include <stdexcept>
using namespace std;

/*! \file Compute.cc
    \brief Contains code for the Compute class
*/

namespace hoomd
    {
/*! \param sysdef SystemDefinition this compute will act on. Must not be NULL.
    \post The Compute is constructed with the given particle data.
*/
Compute::Compute(std::shared_ptr<SystemDefinition> sysdef)
    : Action(sysdef), m_force_compute(false), m_last_computed(0), m_first_compute(true)
    {
    // sanity check
    assert(m_sysdef);
    assert(m_pdata);
    assert(m_exec_conf);
    }

/*! \param timestep Current time step
    \returns true if computations should be performed, false if they have already been done
        at this \a timestep.
    \note This method is designed to only be called once per call to compute() like so:
\code
void SomeClass::compute(uint64_t timestep)
    {
    if (!shouldCompute(timestep))
        return;
    ... do compute tasks
    }
\endcode
*/
bool Compute::shouldCompute(uint64_t timestep)
    {
    // handle case where no computation has been performed yet
    if (m_first_compute)
        {
        m_first_compute = false;
        m_last_computed = timestep;
        return true;
        }

    // Update if computation is enforced, but leave m_last_computed unchanged
    // (such that after a forced compute(), still a regular compute() is possible)
    if (m_force_compute)
        {
        m_force_compute = false;
        return true;
        }

    // otherwise, we update if the last computed timestep is less than the current
    if (m_last_computed != timestep)
        {
        m_last_computed = timestep;
        return true;
        }

    // failing the above, we perform no computation
    return false;
    }

/*! \param timestep Current time step
    \returns true if computations should be performed, false if they have already been done
        at this \a timestep.

    The same logic is applied as in shouldCompute() to determine if compute() should
    be called at \a timestep. However, unlike shouldCompute(), this method does not
    modify the internal state of the Compute and is safe to be called multiple times.
*/
bool Compute::peekCompute(uint64_t timestep) const
    {
    return (m_first_compute || m_force_compute || m_last_computed != timestep);
    }

void Compute::forceCompute(uint64_t timestep)
    {
    m_force_compute = true;

    compute(timestep);
    }

namespace detail
    {
void export_Compute(pybind11::module& m)
    {
    pybind11::class_<Compute, Action, std::shared_ptr<Compute>>(m, "Compute")
        .def(pybind11::init<std::shared_ptr<SystemDefinition>>())
        .def("compute", &Compute::compute)
        .def("notifyDetach", &Compute::notifyDetach);
    }
    } // end namespace detail

    } // end namespace hoomd
