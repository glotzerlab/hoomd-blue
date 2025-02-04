// Copyright (c) 2009-2024 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

#pragma once

#include <algorithm>
#include <cstdint>
#include <memory>
#include <pybind11/iostream.h>
#include <pybind11/pybind11.h>
#include <stdexcept>
#include <vector>

namespace hoomd
    {
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
    Trigger() : m_last_timestep(-1), m_last_trigger(false) { }

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
            m_last_timestep = timestep;
            m_last_trigger = compute(timestep);
            return m_last_trigger;
            }
        }

    virtual bool compute(uint64_t timestep) = 0;

    private:
    /// Caches the last time step at which the trigger was computed
    uint64_t m_last_timestep;
    /// Caches whether the trigger was activated on m_last_timestep
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
    PeriodicTrigger(uint64_t period, uint64_t phase = 0)
        : Trigger(), m_period(period), m_phase(phase)
        {
        if (m_period == 0)
            {
            throw std::runtime_error("Period cannot be set to 0");
            }
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
    uint64_t getPeriod() const
        {
        return m_period;
        }

    /// Set the phase
    void setPhase(uint64_t phase)
        {
        m_phase = phase;
        }

    /// Get the phase
    uint64_t getPhase() const
        {
        return m_phase;
        }

    protected:
    /// The period
    uint64_t m_period;
    /// The phase
    uint64_t m_phase;
    };

/** Before trigger
 *
 *  Trigger every time step before `m_timestep`.
 */
class PYBIND11_EXPORT BeforeTrigger : public Trigger
    {
    public:
    BeforeTrigger(uint64_t timestep) : Trigger(), m_timestep(timestep) { }

    bool compute(uint64_t timestep)
        {
        return timestep < m_timestep;
        }

    /// Get the timestep before which the trigger is active.
    uint64_t getTimestep() const
        {
        return m_timestep;
        }
    const

        /// Set the timestep before which the trigger is active.
        void
        setTimestep(uint64_t timestep)
        {
        m_timestep = timestep;
        }

    protected:
    uint64_t m_timestep; /// trigger timestep < m_timestep
    };

/** On trigger
 *
 *  Trigger on timestep `m_timestep`.
 */
class PYBIND11_EXPORT OnTrigger : public Trigger
    {
    public:
    OnTrigger(uint64_t timestep) : Trigger(), m_timestep(timestep) { }

    bool compute(uint64_t timestep)
        {
        return timestep == m_timestep;
        }

    /// Get the timestep when the trigger is active.
    uint64_t getTimestep() const
        {
        return m_timestep;
        }
    const

        /// Set the timestep when the trigger is active.
        void
        setTimestep(uint64_t timestep)
        {
        m_timestep = timestep;
        }

    protected:
    uint64_t m_timestep; /// only trigger on this timestep
    };

/** After trigger
 *
 *  Trigger every time step after `m_timestep`.
 */
class PYBIND11_EXPORT AfterTrigger : public Trigger
    {
    public:
    AfterTrigger(uint64_t timestep) : Trigger(), m_timestep(timestep) { }

    bool compute(uint64_t timestep)
        {
        return timestep > m_timestep;
        }

    /// Get the timestep after which the trigger is active.
    uint64_t getTimestep() const
        {
        return m_timestep;
        }
    const

        /// Set the timestep after which the trigger is active.
        void
        setTimestep(uint64_t timestep)
        {
        m_timestep = timestep;
        }

    protected:
    uint64_t m_timestep; /// trigger timestep > m_timestep
    };

/** Not trigger
 *
 *  Negates any given trigger.
 */
class PYBIND11_EXPORT NotTrigger : public Trigger
    {
    public:
    NotTrigger(std::shared_ptr<Trigger> trigger) : Trigger(), m_trigger(trigger) { }

    bool compute(uint64_t timestep)
        {
        return !(m_trigger->operator()(timestep));
        }

    /// Get the trigger that is negated
    std::shared_ptr<Trigger> getTrigger() const
        {
        return m_trigger;
        }

    /// Set the trigger to negate
    void setTrigger(std::shared_ptr<Trigger> trigger)
        {
        m_trigger = trigger;
        }

    protected:
    std::shared_ptr<Trigger> m_trigger; ///  trigger to be negated
    };

/** And trigger
 *
 *  The logical AND between multiple triggers.
 */
class PYBIND11_EXPORT AndTrigger : public Trigger
    {
    public:
    AndTrigger(std::vector<std::shared_ptr<Trigger>> triggers) : Trigger(), m_triggers(triggers) { }

    AndTrigger(pybind11::object triggers) : Trigger()
        {
        m_triggers = std::vector<std::shared_ptr<Trigger>>();
        for (auto t : triggers)
            {
            m_triggers.push_back(t.cast<std::shared_ptr<Trigger>>());
            }
        }

    bool compute(uint64_t timestep)
        {
        return std::all_of(m_triggers.begin(),
                           m_triggers.end(),
                           [timestep](std::shared_ptr<Trigger> t)
                           { return t->operator()(timestep); });
        }

    const std::vector<std::shared_ptr<Trigger>>& getTriggers() const
        {
        return m_triggers;
        }

    protected:
    /// Vector of triggers to do a n-way AND
    std::vector<std::shared_ptr<Trigger>> m_triggers;
    };

/** Or trigger
 *
 *  The logical OR between multiple triggers.
 */
class PYBIND11_EXPORT OrTrigger : public Trigger
    {
    public:
    OrTrigger(std::vector<std::shared_ptr<Trigger>> triggers) : Trigger(), m_triggers(triggers) { }

    OrTrigger(pybind11::object triggers) : Trigger()
        {
        m_triggers = std::vector<std::shared_ptr<Trigger>>();
        for (auto t : triggers)
            {
            m_triggers.push_back(t.cast<std::shared_ptr<Trigger>>());
            }
        }

    bool compute(uint64_t timestep)
        {
        return std::any_of(m_triggers.begin(),
                           m_triggers.end(),
                           [timestep](std::shared_ptr<Trigger> t)
                           { return t->operator()(timestep); });
        }

    const std::vector<std::shared_ptr<Trigger>>& getTriggers() const
        {
        return m_triggers;
        }

    protected:
    /// Vector of triggers to do a n-way OR
    std::vector<std::shared_ptr<Trigger>> m_triggers;
    };

namespace detail
    {
/// Export Trigger classes to Python
void export_Trigger(pybind11::module& m);

    } // end namespace detail

    } // end namespace hoomd
