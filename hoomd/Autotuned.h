// Copyright (c) 2009-2024 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

#pragma once

#ifdef __HIPCC__
#error This header cannot be compiled by nvcc
#endif

#include <memory>
#include <pybind11/pybind11.h>
#include <sstream>
#include <vector>

#include "Autotuner.h"

namespace hoomd
    {

/// Base class for autotuned classes.
/*! Some, but not all classes in HOOMD provide autotuners. To give the user a unified API to query
    and interact with these autotuners, Autotuned provides a pybind11 interface to get and set
    autotuner parameters for all child classes. Derived classes must add all autotuners to
    m_autotuners for the base class API to be effective.
*/
class PYBIND11_EXPORT Autotuned
    {
    public:
    Autotuned() { }

    virtual ~Autotuned() { }

    /// Get autotuner parameters.
    pybind11::dict getAutotunerParameters()
        {
        pybind11::dict params;

        for (const auto& tuner : m_autotuners)
            {
            params[tuner->getName().c_str()] = tuner->getParameterPython();
            }
        return params;
        }

    /// Set autotuner parameters.
    void setAutotunerParameters(pybind11::dict params)
        {
        for (auto item : params)
            {
            auto name_match = [item](const std::shared_ptr<AutotunerBase> tuner)
            { return tuner->getName() == pybind11::cast<std::string>(item.first); };
            auto tuner = std::find_if(m_autotuners.begin(), m_autotuners.end(), name_match);

            if (tuner == m_autotuners.end())
                {
                std::ostringstream s;
                s << "Error setting autotuner parameters. Unexpected key: "
                  << pybind11::cast<std::string>(item.first);
                throw std::runtime_error(s.str());
                }

            (*tuner)->setParameterPython(pybind11::cast<pybind11::tuple>(item.second));
            }
        }

    /// Start an autotuning sequence.
    virtual void startAutotuning()
        {
        for (const auto& tuner : m_autotuners)
            {
            tuner->startScan();
            }
        }

    /// Check if autotuning is complete.
    virtual bool isAutotuningComplete()
        {
        bool result = true;
        for (const auto& tuner : m_autotuners)
            {
            result = result && tuner->isComplete();
            }
        return result;
        }

    protected:
    /// All autotuners used by this class instance.
    std::vector<std::shared_ptr<AutotunerBase>> m_autotuners;
    };

namespace detail
    {
/// Exports the Autotuned class to python.
void export_Autotuned(pybind11::module& m);
    } // end namespace detail

    } // end namespace hoomd
