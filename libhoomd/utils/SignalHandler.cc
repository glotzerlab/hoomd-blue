/*
Highly Optimized Object-oriented Many-particle Dynamics -- Blue Edition
(HOOMD-blue) Open Source Software License Copyright 2008-2011 Ames Laboratory
Iowa State University and The Regents of the University of Michigan All rights
reserved.

HOOMD-blue may contain modifications ("Contributions") provided, and to which
copyright is held, by various Contributors who have granted The Regents of the
University of Michigan the right to modify and/or distribute such Contributions.

You may redistribute, use, and create derivate works of HOOMD-blue, in source
and binary forms, provided you abide by the following conditions:

* Redistributions of source code must retain the above copyright notice, this
list of conditions, and the following disclaimer both in the code and
prominently in any materials provided with the distribution.

* Redistributions in binary form must reproduce the above copyright notice, this
list of conditions, and the following disclaimer in the documentation and/or
other materials provided with the distribution.

* All publications and presentations based on HOOMD-blue, including any reports
or published results obtained, in whole or in part, with HOOMD-blue, will
acknowledge its use according to the terms posted at the time of submission on:
http://codeblue.umich.edu/hoomd-blue/citations.html

* Any electronic documents citing HOOMD-Blue will link to the HOOMD-Blue website:
http://codeblue.umich.edu/hoomd-blue/

* Apart from the above required attributions, neither the name of the copyright
holder nor the names of HOOMD-blue's contributors may be used to endorse or
promote products derived from this software without specific prior written
permission.

Disclaimer

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDER AND CONTRIBUTORS ``AS IS'' AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, AND/OR ANY
WARRANTIES THAT THIS SOFTWARE IS FREE OF INFRINGEMENT ARE DISCLAIMED.

IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT,
INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE
OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF
ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

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
    installs a signal handler for SIGING that will set \c g_sigint_recvd
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

