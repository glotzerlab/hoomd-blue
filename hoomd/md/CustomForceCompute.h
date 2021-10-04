// Copyright (c) 2009-2021 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.

// Maintainer: joaander

#include "hoomd/ForceCompute.h"

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

    const GlobalArray<Scalar4>& getForceArray() const
        {
        return m_force;
        }

    const GlobalArray<Scalar>& getVirialArray() const
        {
        return m_virial;
        }

    const GlobalArray<Scalar4>& getTorqueArray() const
        {
        return m_torque;
        }

    protected:
    //! Function that is called on every particle sort
    void rearrangeForces();

    //! Actually compute the forces
    virtual void computeForces(uint64_t timestep);

    private:
    bool m_need_rearrange_forces; //!< True if forces need to be rearranged

    //! Lists of particle tags and corresponding forces/torques
    std::map<unsigned int, vec3<Scalar>> m_forces;
    std::map<unsigned int, vec3<Scalar>> m_torques;

    //! A python callback when the force is updated
    pybind11::object m_callback;
    };


template<class Output>
class PYBIND11_EXPORT LocalForceComputeData : public LocalDataAccess<Output, CustomForceCompute>
    {
    public:
    LocalForceComputeData(CustomForceCompute& data)
        : LocalDataAccess<Output, CustomForceCompute>(data), m_force_handle(),
        m_torque_handle(), m_virial_handle()
        {
        }

    virtual ~LocalForceComputeData() = default;

    Output getForce(GhostDataFlag flag)
        {
        return this->template getBuffer<Scalar4, Scalar>(m_force_handle,
                                                         &CustomForceCompute::getForceArray,
                                                         flag,
                                                         3);
        }

    Output getPotentialEnergy(GhostDataFlag flag)
        {
        return this->template getBuffer<Scalar4, Scalar>(m_force_handle,
                                                         &CustomForceCompute::getForceArray,
                                                         flag,
                                                         0,
                                                         3 * sizeof(Scalar));
        }

    Output getTorque(GhostDataFlag flag)
        {
        return this->template getBuffer<Scalar4, Scalar>(m_torque_handle,
                                                         &CustomForceCompute::getTorqueArray,
                                                         flag,
                                                         3);
        }

    // TODO figure out what to do with the pitch here
    Output getVirial(GhostDataFlag flag)
        {
        return this->template getBuffer<Scalar, Scalar>(m_virial_handle,
                                                        &CustomForceCompute::getVirialArray,
                                                        flag,
                                                        1);
        }

    protected:
    void clear()
        {
        m_force_handle.reset(nullptr);
        m_torque_handle.reset(nullptr);
        m_virial_handle.reset(nullptr);
        }

    private:
    std::unique_ptr<ArrayHandle<Scalar4>> m_force_handle;
    std::unique_ptr<ArrayHandle<Scalar4>> m_torque_handle;
    std::unique_ptr<ArrayHandle<Scalar>> m_virial_handle;
    };


//! Exports the CustomForceComputeClass to python
void export_CustomForceCompute(pybind11::module& m);

template<class Output> void export_LocalForceComputeData(pybind11::module& m, std::string name)
    {
    pybind11::class_<LocalForceComputeData<Output>, std::shared_ptr<LocalForceComputeData<Output>>>(
        m,
        name.c_str())
        .def(pybind11::init<ForceCompute&>())
        .def("getForce", &LocalForceComputeData<Output>::getForce)
        .def("getPotentialEnergy", &LocalForceComputeData<Output>::getPotentialEnergy)
        .def("getTorque", &LocalForceComputeData<Output>::getTorque)
        .def("getVirial", &LocalForceComputeData<Output>::getVirial)
        .def("enter", &LocalForceComputeData<Output>::enter)
        .def("exit", &LocalForceComputeData<Output>::exit);
    };

#endif
