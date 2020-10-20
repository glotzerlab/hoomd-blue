// Copyright (c) 2009-2019 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.


// Maintainer: joaander

#include <signal.h>
#include "SignalHandler.h"
#include <iostream>
#include "string.h"

using namespace std;

/*! \file SignalHandler.cc
    \brief Defines variables and functions related to handling signals
*/

//! Tracks the previous signal handler
static struct sigaction g_oldact;

volatile sig_atomic_t g_sigint_recvd = 0;

//! The actual signal handler
extern "C" void sigint_handler(int sig)
    {
    // ignore if we didn't get SIGINT
    if (sig != SIGINT)
        return;

    std::cout << "HOOMD caught SIGINT" << std::endl;

    // set the global
    g_sigint_recvd = 1;
    }

/*! This method installs a signal handler for SIGINT that will set \c g_sigint_recvd to 1.
*/
void InstallSIGINTHandler()
    {
    struct sigaction newact;
    newact.sa_handler = sigint_handler;
    sigemptyset(&newact.sa_mask);
    newact.sa_flags = 0;

    int retval = sigaction(SIGINT, &newact, &g_oldact);

    if (retval != 0)
        {
        cerr << "Error setting signal handler: " << strerror(errno) << endl;
        return;
        }
    }

void RemoveSIGINTHandler()
    {
    struct sigaction dummy_action;
    int retval = sigaction(SIGINT, &g_oldact, &dummy_action);

    if (retval != 0)
        {
        cerr << "Error setting signal handler: " << strerror(errno) << endl;
        return;
        }
    }
