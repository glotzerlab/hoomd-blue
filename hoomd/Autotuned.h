// Copyright (c) 2009-2022 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

#pragma once

#ifdef __HIPCC__
#error This header cannot be compiled by nvcc
#endif

#include <vector>
#include <memory>
#include <pybind11/pybind11.h>

#include "Autotuner.h"

namespace hoomd {

/// Base class for autotuned classes.
/*! Some, but not all classes in HOOMD provide autotuners. To give the user a unified API to query
    and interact with these autotuners, Autotuned provides a pybind11 interface to get and set
    autotuner parameters for all child classes. Derived classes must add all autotuners to
    m_autotuners for the base class API to be effective.
*/
class Autotuned
    {
    public:
        Autotuned()
            {
            }

        /// Get autotuner parameters.
        pybind11::tuple getAutotunerParameters()
            {
            pybind11::list params;

            for (const auto& tuner : m_autotuners)
                {
                params.append(tuner->getParameterPython());
                }
            return pybind11::tuple(params);
            }

        /// Set autotuner parameters.
        void setAutotunerParameters(pybind11::tuple params)
            {
            size_t n_params = pybind11::len(params);
            if (n_params != m_autotuners.size())
                {
                std::ostringstream s;
                s << "Error setting autotuner parameters. Got "
                  << n_params << " parameters, but expected "
                  << m_autotuners.size() << "." << std::endl;
                throw std::runtime_error(s.str());
                }

            for (unsigned int i=0; i < n_params; i++)
                {
                m_autotuners[i]->setParameterPython(params[i]);
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
        std::vector<std::shared_ptr<AutotunerInterface>> m_autotuners;
    };

namespace detail
    {
/// Exports the Autotuned class to python.
void export_Autotuned(pybind11::module& m);
    } // end namespace detail

} // end namespace hoomd
