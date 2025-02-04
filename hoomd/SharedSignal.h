// Copyright (c) 2009-2024 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

#ifndef SYSTEM_SIGNAL
#define SYSTEM_SIGNAL
#include <functional>
#include <hoomd/extern/nano-signal-slot/nano_signal_slot.hpp>

namespace hoomd
    {
namespace detail
    {
/*! \ingroup hoomd_lib
    @{
*/

/*! \defgroup SharedSignal
    \brief Classes that guarantee object lifetimes with signal slots
*/

/*! @}
 */

template<typename R> class SharedSignalSlot;

//! Manages signal lifetime and slot lifetime
/*! The SharedSignal is a class that wraps the Nano::Signal class that allows for

    The SharedSignal has two types of signals: 1) the signal that we want to manage
    and 2) the disconnect signal that is emitted on destruction that tells all
    SharedSignalSlots to disconnect.

    See \ref page_dev_info for more information

    \ingroup SharedSignal
*/
template<class SignalType> class SharedSignal : public Nano::Signal<SignalType>
    {
    public:
    SharedSignal() { }
    virtual ~SharedSignal()
        {
        // The shared signal is being destroyed so we need to clean up any
        // references to the signal before it is freed.
        disconnect_signal.emit();
        }
    friend class SharedSignalSlot<SignalType>;

    private:
    Nano::Signal<void()> disconnect_signal; //!< Disconnect Signal
    };

//! Manages signal lifetime and slot lifetime
/*! The SignalSlot defines the interface for the SharedSignalSlot classes. The base
    class allows for multiple SharedSignalSlot's to be stored in a common container
    class. The connected() method returns true if the slot is connected to a signal.
    The disconnect() method must disconnect from both the signal and disconnect signal
    to be memory safe and should set m_connected to be false. Once disconnected from
    a signal the slot cannot reconnect.  The constructor of the derived class will
    connect to the signals.

    See \ref page_dev_info for more information

    \ingroup SharedSignal
*/
class SignalSlot
    {
    public:
    SignalSlot() : m_connected(false) { }
    virtual ~SignalSlot() { }
    virtual void disconnect() = 0;
    bool connected()
        {
        return m_connected;
        }
    SignalSlot(const SignalSlot&) = delete;            // non-copyable
    SignalSlot& operator=(const SignalSlot&) = delete; // non-assignable
    protected:
    bool m_connected;
    };

//! Manages signal lifetime and slot lifetime
/*! The SharedSignalSlot class manages the connections between the functions and the
    SharedSignal. On construction the slot is connected to the signal. On destruction
    the slot is disconnected. The slot can be disconnected by calling disconnect()
    and once it is disconnected it can not be reconnected.

    See \ref page_dev_info for more information

    \ingroup SharedSignal
*/
template<typename R, typename... Args> class SharedSignalSlot<R(Args...)> : public SignalSlot
    {
    public:
    // bind to object member function (const)
    template<typename T>
    SharedSignalSlot(SharedSignal<R(Args...)>& signal, T* obj, R (T::*f)(Args...) const)
        : m_signal(signal)
        {
        m_func = [obj, f](Args... args) -> R { return (obj->*f)(std::forward<Args>(args)...); };
        connect();
        }
    // bind to object member function (non-const)
    template<typename T>
    SharedSignalSlot(SharedSignal<R(Args...)>& signal, T* obj, R (T::*f)(Args...))
        : m_signal(signal)
        {
        m_func = [obj, f](Args... args) -> R { return (obj->*f)(std::forward<Args>(args)...); };
        connect();
        }
    // bind to general function
    template<typename Func>
    SharedSignalSlot(SharedSignal<R(Args...)>& signal, Func&& f) : m_signal(signal)
        {
        m_func = [f](Args... args) -> R { return f(std::forward<Args>(args)...); };
        connect();
        }

    ~SharedSignalSlot()
        {
        disconnect();
        }

    void disconnect()
        {
        if (!m_connected) // never disconnect more than once
            return;
        m_signal.disconnect(m_func);
        m_signal.disconnect_signal.template disconnect<SharedSignalSlot<R(Args...)>,
                                                       &SharedSignalSlot<R(Args...)>::disconnect>(
            this);
        m_connected = false;
        }

    private:
    void connect()
        {
        m_signal.disconnect_signal.template connect<SharedSignalSlot<R(Args...)>,
                                                    &SharedSignalSlot<R(Args...)>::disconnect>(
            this);
        m_signal.connect(m_func);
        m_connected = true;
        }

    protected:
    SharedSignal<R(Args...)>& m_signal;
    std::function<R(Args...)> m_func;
    };
    } // end namespace detail
    } // end namespace hoomd
#endif
