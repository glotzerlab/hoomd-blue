# -- start license --
# Highly Optimized Object-oriented Many-particle Dynamics -- Blue Edition
# (HOOMD-blue) Open Source Software License Copyright 2008-2011 Ames Laboratory
# Iowa State University and The Regents of the University of Michigan All rights
# reserved.

# HOOMD-blue may contain modifications ("Contributions") provided, and to which
# copyright is held, by various Contributors who have granted The Regents of the
# University of Michigan the right to modify and/or distribute such Contributions.

# You may redistribute, use, and create derivate works of HOOMD-blue, in source
# and binary forms, provided you abide by the following conditions:

# * Redistributions of source code must retain the above copyright notice, this
# list of conditions, and the following disclaimer both in the code and
# prominently in any materials provided with the distribution.

# * Redistributions in binary form must reproduce the above copyright notice, this
# list of conditions, and the following disclaimer in the documentation and/or
# other materials provided with the distribution.

# * All publications and presentations based on HOOMD-blue, including any reports
# or published results obtained, in whole or in part, with HOOMD-blue, will
# acknowledge its use according to the terms posted at the time of submission on:
# http://codeblue.umich.edu/hoomd-blue/citations.html

# * Any electronic documents citing HOOMD-Blue will link to the HOOMD-Blue website:
# http://codeblue.umich.edu/hoomd-blue/

# * Apart from the above required attributions, neither the name of the copyright
# holder nor the names of HOOMD-blue's contributors may be used to endorse or
# promote products derived from this software without specific prior written
# permission.

# Disclaimer

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDER AND CONTRIBUTORS ``AS IS'' AND
# ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
# WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, AND/OR ANY
# WARRANTIES THAT THIS SOFTWARE IS FREE OF INFRINGEMENT ARE DISCLAIMED.

# IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT,
# INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
# BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
# DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
# LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE
# OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF
# ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
# -- end license --

# Maintainer: joaander

import sys;
import traceback;
import os.path;
import linecache;
import re;
import globals;

## \internal
# \package hoomd_script.util
# \brief Internal utility functions used by hoomd_script

## \internal
# \brief Internal flag tracking if 
_disable_status_lines = False;

## Prints a status line tracking the execution of the current hoomd script
def print_status_line():
    if _disable_status_lines:
        return;
    
    # get the traceback info first
    stack = traceback.extract_stack();
    if len(stack) < 3:
        print "hoomd_script executing unknown command";
    file_name, line, module, code = stack[-3];
    
    # if we are in interactive mode, there is no need to print anything: the
    # interpreter loop does it for us. We can make that check by testing if
    # sys.ps1 is defined (this is not a hack, the python documentation states 
    # that ps1 is _only_ defined in interactive mode
    if 'ps1' in sys.__dict__:
        return

    # piped input from stdin doesn't provide a code line, handle the situation 
    if not code:
        message = os.path.basename(file_name) + ":" + str(line).zfill(3) + "  |  <unknown code>";
        print message;
    else:
        # build and print the message line
        # Go upwards in the source until you match the closing paren
        # dequote ensures we ignore literal parens
        dequote = lambda x: re.sub(r'[\'"].*?[\'"]','',x)
        balance = lambda y: y.count('(') - y.count(')')
        message = []
        while True:
            message.insert(0,linecache.getline(file_name,line))
            if sum(balance(dequote(x)) for x in message) == 0 or line == 0:
                break
            line = line - 1
        
        message.insert(0,os.path.basename(file_name) + ":" + str(line).zfill(3) + "  |  ")
        print ''.join(message).rstrip('\n')
        linecache.clearcache()

