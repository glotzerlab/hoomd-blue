// Copyright (c) 2009-2019 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.


// Maintainer: joaander

/*! \file SignalHandler.h
    \brief Declares variables and functions related to handling signals
*/

#ifdef NVCC
#error This header cannot be compiled by nvcc
#endif

#ifndef __SIGNALHANDLER_H__
#define __SIGNALHANDLER_H__

#include <signal.h>

//! Value set to non-zero if SIGINT has occurred
/*! Any method that reads this value as non-zero should immediately reset it to 0
    and return.
*/
extern volatile sig_atomic_t g_sigint_recvd;

//! Installs the signal handler
void InstallSIGINTHandler();

#endif
