// Copyright (c) 2009-2021 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.

// Maintainer: joaander

#include "ForceCompute.h"
#include "ParticleGroup.h"

#include <map>
#include <memory>

/*! \file CustomForceCompute.h
    \brief Declares the backend for computing custom forces in python classes
*/

#ifdef __HIPCC__
#error This header cannot be compiled by nvcc
#endif

#include <pybind11/pybind11.h>

#ifndef __CUSTOMFORCECOMPUTE_H__
#define __CUSTOMFORCECOMPUTE_H__

//! Adds a custom force
/*! \ingroup computes
 */
class PYBIND11_EXPORT CustomForceCompute : public ForceCompute
    {
    public:
    //! Constructs the compute
    CustomForceCompute(std::shared_ptr<SystemDefinition> sysdef);

    //! Destructor
    ~CustomForceCompute();

    //! Set the python callback
    void setCallback(pybind11::object py_callback)
        {
        m_callback = py_callback;
        }

    protected:
    //! Function that is called on every particle sort
    void rearrangeForces();

    //! Actually compute the forces
    virtual void computeForces(uint64_t timestep);

    private:
    bool m_need_rearrange_forces; //!< True if forces need to be rearranged

    //! List of particle tags and corresponding forces
    std::map<unsigned int, vec3<Scalar>> m_forces;

    //! List of particle tags and corresponding forces
    std::map<unsigned int, vec3<Scalar>> m_torques;

    //! A python callback when the force is updated
    pybind11::object m_callback;
    };

//! Exports the CustomForceComputeClass to python
void export_CustomForceCompute(pybind11::module& m);

#endif
