// Copyright (c) 2009-2019 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.

#pragma once

#include <cstdint>
#include <pybind11/pybind11.h>

/** Defines on what time steps operations should be performed
 *
 *  System schedules Analyzer and Updater instances to be executed only on specific time steps.
 *  Trigger provides an abstract mechanism to implement this. Users can derive subclasses of Trigger
 *  (in python) to implement custom behavior.
 *
 *  A Trigger may store internal staten and perform complex calculations to determine when it
*/
class PYBIND11_EXPORT Trigger
    {
    public:
        /// Construct a Trigger
        Trigger() { }

        virtual ~Trigger() { }

        /** Determine if an operation should be performed on the given timestep
         *
         *  @param timestep Time step to query
         *  @returns `true` if the operation should occur, `false` if not
        */
        virtual bool operator()(uint64_t timestep)
            {
            return false;
            }
    };

/** Periodic trigger
 *
 *  Trigger every `period` time steps offset by `phase`.
*/
class PYBIND11_EXPORT PeriodicTrigger : public Trigger
    {
    public:

        /** Construct a PeriodicTrigger
         *
         *  @param period The period
         *  @param phase The phase
        */
        PeriodicTrigger(uint64_t period, uint64_t phase=0)
            : m_period(period), m_phase(phase)
            {
            }

        bool operator()(uint64_t timestep)
            {
            return (timestep - m_phase) % m_period == 0;
            }

        /// Set the period
        void setPeriod(uint64_t period)
            {
            m_period = period;
            }

        /// Get the period
        uint64_t getPeriod()
            {
            return m_period;
            }

        /// Set the phase
        void setPhase(uint64_t phase)
            {
            m_phase = phase;
            }

        /// Get the phase
        uint64_t getPhase()
            {
            return m_phase;
            }

    protected:
        /// The period
        uint64_t m_period;
        /// The phase
        uint64_t m_phase;
    };

/// Export Trigger classes to Python
void export_Trigger(pybind11::module& m);
