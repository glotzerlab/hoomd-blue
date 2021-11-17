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

namespace hoomd
    {
namespace md
    {
//! Adds a custom force
/*! \ingroup computes
 */
class PYBIND11_EXPORT CustomForceCompute : public ForceCompute
    {
    public:
    //! Constructs the compute
    CustomForceCompute(std::shared_ptr<hoomd::SystemDefinition> sysdef);

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

    unsigned int getN() const
        {
        return m_pdata->getN();
        }

    unsigned int getNGhosts() const
        {
        return m_pdata->getNGhosts();
        }

    const GlobalArray<unsigned int>& getRTags() const
        {
        return m_pdata->getRTags();
        }

    protected:
    //! Actually compute the forces
    virtual void computeForces(uint64_t timestep);

    private:
    //! A python callback when the force is updated
    pybind11::object m_callback;
    };


/** Make the local particle data available to python via zero-copy access
 *
 * */
template<class Output>
class PYBIND11_EXPORT LocalForceComputeData : public LocalDataAccess<Output, CustomForceCompute>
    {
    public:
    LocalForceComputeData(CustomForceCompute& data)
        : LocalDataAccess<Output, CustomForceCompute>(data), m_force_handle(),
        m_torque_handle(), m_virial_handle(), m_virial_pitch(data.getVirialArray().getPitch())
        {
        }

    virtual ~LocalForceComputeData() = default;

    Output getRTags()
        {
        return this->template getGlobalBuffer<unsigned int>(m_rtag_handle,
                                                            &CustomForceCompute::getRTags);
        }

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

    Output getVirial(GhostDataFlag flag)
        {
        return this->template getBuffer<Scalar, Scalar>(m_virial_handle,
                                                        &CustomForceCompute::getVirialArray,
                                                        flag,
                                                        6,
                                                        0,
                                                        std::vector<ssize_t>({m_virial_pitch * sizeof(Scalar), sizeof(Scalar)}));
        }

    protected:
    void clear()
        {
        m_force_handle.reset(nullptr);
        m_torque_handle.reset(nullptr);
        m_virial_handle.reset(nullptr);
        m_rtag_handle.reset(nullptr);
        }

    private:
    std::unique_ptr<ArrayHandle<Scalar4>> m_force_handle;
    std::unique_ptr<ArrayHandle<Scalar4>> m_torque_handle;
    std::unique_ptr<ArrayHandle<Scalar>> m_virial_handle;
    std::unique_ptr<ArrayHandle<unsigned int>> m_rtag_handle;
    size_t m_virial_pitch;
    };


namespace detail
    {
//! Exports the CustomForceComputeClass to python
void export_CustomForceCompute(pybind11::module& m);

template<class Output> void export_LocalForceComputeData(pybind11::module& m, std::string name)
    {
    pybind11::class_<LocalForceComputeData<Output>, std::shared_ptr<LocalForceComputeData<Output>>>(
        m,
        name.c_str())
        .def(pybind11::init<CustomForceCompute&>())
        .def("getForce", &LocalForceComputeData<Output>::getForce)
        .def("getPotentialEnergy", &LocalForceComputeData<Output>::getPotentialEnergy)
        .def("getTorque", &LocalForceComputeData<Output>::getTorque)
        .def("getVirial", &LocalForceComputeData<Output>::getVirial)
        .def("enter", &LocalForceComputeData<Output>::enter)
        .def("exit", &LocalForceComputeData<Output>::exit);
    };

    } // end namespace detail
    } // end namespace md
    } // end namespace hoomd

#endif
