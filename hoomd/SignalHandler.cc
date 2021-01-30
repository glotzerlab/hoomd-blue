// Copyright (c) 2009-2021 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.


// Maintainer: joaander

#include <signal.h>
#include "SignalHandler.h"
#include <iostream>
#include <cstring>

using namespace std;

/*! \file SignalHandler.cc
    \brief Defines variables and functions related to handling signals
*/

volatile sig_atomic_t g_sigint_recvd = 0;

//! The actual signal handler
extern "C" void sigint_handler(int sig)
    {
    // ignore if we didn't get SIGINT
    if (sig != SIGINT)
        return;

    // set the global
    g_sigint_recvd = 1;
    }

ScopedSignalHandler::ScopedSignalHandler()
    {
    struct sigaction newact;
    newact.sa_handler = sigint_handler;
    sigemptyset(&newact.sa_mask);
    newact.sa_flags = 0;

    int retval = sigaction(SIGINT, &newact, &m_old_action);

    if (retval != 0)
        {
        cerr << "Error setting signal handler: " << strerror(errno) << endl;
        }
    }

ScopedSignalHandler::~ScopedSignalHandler()
    {
    struct sigaction dummy_action;
    int retval = sigaction(SIGINT, &m_old_action, &dummy_action);

    if (retval != 0)
        {
        cerr << "Error setting signal handler: " << strerror(errno) << endl;
        }
    }
