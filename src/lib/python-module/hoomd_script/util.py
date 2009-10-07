# -*- coding: iso-8859-1 -*-
# Highly Optimized Object-Oriented Molecular Dynamics (HOOMD) Open
# Source Software License
# Copyright (c) 2008 Ames Laboratory Iowa State University
# All rights reserved.

# Redistribution and use of HOOMD, in source and binary forms, with or
# without modification, are permitted, provided that the following
# conditions are met:

# * Redistributions of source code must retain the above copyright notice,
# this list of conditions and the following disclaimer.

# * Redistributions in binary form must reproduce the above copyright
# notice, this list of conditions and the following disclaimer in the
# documentation and/or other materials provided with the distribution.

# * Neither the name of the copyright holder nor the names HOOMD's
# contributors may be used to endorse or promote products derived from this
# software without specific prior written permission.

# Disclaimer

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDER AND
# CONTRIBUTORS ``AS IS''  AND ANY EXPRESS OR IMPLIED WARRANTIES,
# INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY
# AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. 

# IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS  BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF
# THE POSSIBILITY OF SUCH DAMAGE.

# $Id$
# $URL$
# Maintainer: joaander

import sys;
import traceback;

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
    # gracefully
    if not code:
        code = "<unknown code>";
    
    # build and print the message line
    message = file_name + ":" + str(line).zfill(3) + "  |  " + code;
    print message;

