// Copyright (c) 2009-2024 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

/*! \file pdata_test.cc
    \brief Unit tests for BoxDim, ParticleData, SimpleCubicInitializer, and RandomInitializer
   classes. \ingroup unit_tests
*/

// this include is necessary to get MPI included before anything else to support intel MPI
#include "hoomd/ExecutionConfiguration.h"

#include <iostream>

#include "../SharedSignal.h"
using namespace std;
#include "upp11_config.h"

// A few helper classes
class SignalHolder
    {
    public:
    hoomd::detail::SharedSignal<void()> signal;
    };

class OtherSignalHolder
    {
    public:
    hoomd::detail::SharedSignal<bool(int)> signal;
    };

class SlotHolder
    {
    public:
    SlotHolder(hoomd::detail::SharedSignal<void()>& signal)
        : slot(signal, this, &SlotHolder::callback)
        {
        }
    void callback()
        {
        std::cout << "SlotHolder::callback()" << std::endl;
        }
    hoomd::detail::SharedSignalSlot<void()> slot;
    };

class SlotContainer
    {
    public:
    SlotContainer(int val) : pint(new int)
        {
        *pint = val;
        }
    ~SlotContainer()
        {
        std::cout << "SlotContainer destroyed" << std::endl;
        }
    void callback()
        {
        std::cout << "SlotContainer::callback() object @ " << hex << this << dec
                  << " with value = " << *pint << std::endl;
        }
    bool otherCallback(int i)
        {
        std::cout << "SlotContainer::otherCallback(" << i << ")@ " << hex << this << dec
                  << " with value = " << *pint << std::endl;
        return i == 0;
        }
    vector<std::unique_ptr<hoomd::detail::SignalSlot>> slots;
    std::unique_ptr<int> pint;
    };

void func()
    {
    std::cout << "func" << std::endl;
    }
bool otherFunc(int i)
    {
    std::cout << "otherFunc(" << i << ")" << std::endl;
    return i == 0;
    }

HOOMD_UP_MAIN();

//! Perform some basic tests on the boxdim structure
UP_TEST(SharedSignal_basic_test)
    {
        // Scalar tol = Scalar(1e-6);

        { // slot goes out of scope first.
        SignalHolder sig;
        OtherSignalHolder osig;
            {
            std::unique_ptr<SlotContainer> s(new SlotContainer(2));
            // nano.connect<SlotContainer, &SlotContainer::callback>(s.get());
            sig.signal.emit();
            s->slots.push_back(std::unique_ptr<hoomd::detail::SignalSlot>(
                new hoomd::detail::SharedSignalSlot<void()>(sig.signal,
                                                            s.get(),
                                                            &SlotContainer::callback)));
            s->slots.push_back(std::unique_ptr<hoomd::detail::SignalSlot>(
                new hoomd::detail::SharedSignalSlot<bool(int)>(osig.signal,
                                                               s.get(),
                                                               &SlotContainer::otherCallback)));
            sig.signal.emit();
            osig.signal.emit(0);
            }
        sig.signal.emit();
        osig.signal.emit(1);
        }
        { // signal goes out of scope first.
        std::unique_ptr<SlotContainer> s(new SlotContainer(2));
            {
            SignalHolder sig;
            OtherSignalHolder osig;
            // nano.connect<SlotContainer, &SlotContainer::callback>(s.get());
            sig.signal.emit();
            s->slots.push_back(std::unique_ptr<hoomd::detail::SignalSlot>(
                new hoomd::detail::SharedSignalSlot<void()>(sig.signal,
                                                            s.get(),
                                                            &SlotContainer::callback)));
            s->slots.push_back(std::unique_ptr<hoomd::detail::SignalSlot>(
                new hoomd::detail::SharedSignalSlot<bool(int)>(osig.signal,
                                                               s.get(),
                                                               &SlotContainer::otherCallback)));
            sig.signal.emit();
            osig.signal.emit(0);
            }
        }
    // TODO: add more extensive tests.

    // test default constructor
    // BoxDim a;
    // MY_CHECK_CLOSE(a.getLo().x,0.0, tol);
    // UP_ASSERT_EQUAL(b.getPeriodic().x, 1);
    // UP_ASSERT(a.getPositions().getNumElements() == 1);
    }
