#pragma once

#include <cstdint>
#include <pybind11/pybind11.h>

/** Defines on what time steps operations should be performed

    System schedules Analyzer and Updater instances to be executed only on specific time steps. Trigger provides an
    abstract mechanism to implement this. Users can derive subclasses of Trigger (in python) to implement custom
    behavior.
*/
class PYBIND11_EXPORT Trigger
    {
    public:
        Trigger() { }

        /** Determine if an operation should be performed on the given timestep

            Args:
                timestep: Time step to query

            Returns:
                true if the operation should occur, false if not
        */
        virtual bool operator()(uint64_t timestep)
            {
            return false;
            }
    };

/** Periodic trigger

    Trigger every ``period`` time steps offset by ``phase``
*/
class PYBIND11_EXPORT PeriodicTrigger : public Trigger
    {
    public:
        /** Construct a periodic trigger

            Args:
                period: The period
                phase: The phase
        */
        PeriodicTrigger(uint64_t period, uint64_t phase=0)
            : m_period(period), m_phase(phase)
            {
            }

        virtual bool operator()(uint64_t timestep)
            {
            return (timestep - m_phase) % m_period == 0;
            }

        void setPeriod(uint64_t period)
            {
            m_period = period;
            }

        uint64_t getPeriod()
            {
            return m_period;
            }

        void setPhase(uint64_t phase)
            {
            m_phase = phase;
            }

        uint64_t getPhase()
            {
            return m_phase;
            }

    protected:
        uint64_t m_period;
        uint64_t m_phase;
    };

void export_Trigger(pybind11::module& m);
