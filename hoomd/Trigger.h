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
        Trigger(): m_last_timestep(-1), m_last_trigger(false) { }

        virtual ~Trigger() { }

        /** Determine if an operation should be performed on the given timestep
         *
         *  @param timestep Time step to query
         *  @returns `true` if the operation should occur, `false` if not
        */
        bool operator()(uint64_t timestep)
            {
            if (m_last_timestep == timestep)
                {
                return m_last_trigger;
                }
            else
                {
                auto triggered = compute(timestep);
                m_last_timestep = timestep;
                m_last_trigger = triggered;
                return triggered;
                }
            }

        virtual bool compute(uint64_t timestep) = 0;

    private:
            uint64_t m_last_timestep;
            bool m_last_trigger;
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
            : Trigger(), m_period(period), m_phase(phase)
            {
            }

        bool compute(uint64_t timestep)
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

/** Until trigger
 *
 *  Trigger every time step until `timestep >= m_until`.
*/
class PYBIND11_EXPORT UntilTrigger : public Trigger
    {
    public:

    UntilTrigger(uint64_t until) : Trigger(), m_until(until) {}

    bool compute(uint64_t timestep)
        {
        if (timestep < m_until)
            {
            return true;
            }
        else
            {
            return false;
            }
        }

    /// Get the point where triggering will stop
    uint64_t getUntil() {return m_until;}

    /// Set the point where triggering will stop
    void setUntil(uint64_t until) {m_until = until;}

    protected:
        /// Always trigger until this timestep
        uint64_t m_until;
    };

/** After trigger
 *
 *  Trigger every time step after and including `timestep > m_after`.
*/
class PYBIND11_EXPORT AfterTrigger : public Trigger
    {
    public:

    AfterTrigger(uint64_t after) : Trigger(), m_after(after) {}

    bool compute(uint64_t timestep)
        {
        if (timestep > m_after)
            {
            return true;
            }
        else
            {
            return false;
            }
        }

    /// Get the point where triggering will stop
    uint64_t getAfter() {return m_after;}

    /// Set the point where triggering will stop
    void setAfter(uint64_t after) {m_after = after;}

    protected:
        /// Always trigger after this timestep
        uint64_t m_after;
    };

/** Not trigger
 *
 *  Negates any given trigger.
*/
class PYBIND11_EXPORT NotTrigger : public Trigger
    {
    public:
        NotTrigger(std::shared_ptr<Trigger> trigger) :
            Trigger(), m_trigger(trigger) {}

        bool compute(uint64_t timestep)
            {
            return !(m_trigger->compute(timestep));
            }

        /// Get the trigger thats negated
        std::shared_ptr<Trigger> getTrigger() {return m_trigger;}

        /// Set the trigger to negate
        void setTrigger(std::shared_ptr<Trigger> trigger) {m_trigger = trigger;}

    protected:
        std::shared_ptr<Trigger> m_trigger;
    };

/** And trigger
 *
 *  The logical AND between two triggers.
*/
class PYBIND11_EXPORT AndTrigger : public Trigger
    {
    public:
        AndTrigger(std::shared_ptr<Trigger> trigger1,
                   std::shared_ptr<Trigger> trigger2) :
            Trigger(), m_trigger1(trigger1), m_trigger2(trigger2) {}

        bool compute(uint64_t timestep)
            {
            return m_trigger1->compute(timestep) &&
                   m_trigger2->compute(timestep);
            }

        /// Get the first trigger
        std::shared_ptr<Trigger> getTrigger1() {return m_trigger1;}

        /// Set the second trigger
        void setTrigger1(std::shared_ptr<Trigger> trigger)
            {
            m_trigger1 = trigger;
            }

        /// Get the second trigger
        std::shared_ptr<Trigger> getTrigger2() {return m_trigger2;}

        /// Set the second trigger
        void setTrigger2(std::shared_ptr<Trigger> trigger)
            {
            m_trigger2 = trigger;
            }

    protected:
        std::shared_ptr<Trigger> m_trigger1;
        std::shared_ptr<Trigger> m_trigger2;
    };

/** Or trigger
 *
 *  The logical OR between two triggers.
*/
class PYBIND11_EXPORT OrTrigger : public Trigger
    {
    public:
        OrTrigger(std::shared_ptr<Trigger> trigger1,
                   std::shared_ptr<Trigger> trigger2) :
            Trigger(), m_trigger1(trigger1), m_trigger2(trigger2) {}

        bool compute(uint64_t timestep)
            {
            return m_trigger1->compute(timestep) ||
                   m_trigger2->compute(timestep);
            }

        /// Get the first trigger
        std::shared_ptr<Trigger> getTrigger1() {return m_trigger1;}

        /// Set the second trigger
        void setTrigger1(std::shared_ptr<Trigger> trigger)
            {
            m_trigger1 = trigger;
            }

        /// Get the second trigger
        std::shared_ptr<Trigger> getTrigger2() {return m_trigger2;}

        /// Set the second trigger
        void setTrigger2(std::shared_ptr<Trigger> trigger)
            {
            m_trigger2 = trigger;
            }

    protected:
        std::shared_ptr<Trigger> m_trigger1;
        std::shared_ptr<Trigger> m_trigger2;
    };

/// Export Trigger classes to Python
void export_Trigger(pybind11::module& m);
