// Copyright (c) 2009-2019 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.


// Maintainer: joaander

#include <signal.h>
#include "SignalHandler.h"
#include <iostream>

using namespace std;

/*! \file SignalHandler.cc
    \brief Defines variables and functions related to handling signals
*/

//! Tracks the previous signal handler that was set to make a chain
void (*prev_sigint_handler)(int) = NULL;

volatile sig_atomic_t g_sigint_recvd = 0;

//! The actual signal handler
extern "C" void sigint_handler(int sig)
    {
    // ignore if we didn't get SIGINT
    if (sig != SIGINT)
        return;

    // call the previous signal handler, but only if it is well defined
    if (prev_sigint_handler && prev_sigint_handler != SIG_ERR && prev_sigint_handler != SIG_DFL && prev_sigint_handler != SIG_IGN)
        prev_sigint_handler(sig);

    // set the global
    g_sigint_recvd = 1;
    }

/*! Call only once at the start of program execution. This method
    installs a signal handler for SIGINT that will set \c g_sigint_recvd
    to 1. It will also call the previously set signal handler.
*/
void InstallSIGINTHandler()
    {
    void (*retval)(int) = NULL;
    retval = signal(SIGINT, sigint_handler);

    if (retval == SIG_ERR)
        {
        cerr << "Error setting signal handler" << endl;
        return;
        }

    // set the previous signal handler, but only if it is not the same as the
    // one we just set. That would make for a fun infinite loop!
    if (retval != sigint_handler)
        prev_sigint_handler = retval;
    else
        prev_sigint_handler = NULL;
    }
