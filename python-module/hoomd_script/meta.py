# -- start license --
# Highly Optimized Object-oriented Many-particle Dynamics -- Blue Edition
# (HOOMD-blue) Open Source Software License Copyright 2009-2016 The Regents of
# the University of Michigan All rights reserved.

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

# Maintainer: joaander / All Developers are free to add commands for new features

## \package hoomd_script.meta
# \brief Write out simulation and environment context metadata
#
# Metadata is stored in form of key-value pairs in a JSON file and used
# to summarize the per-run simulation parameters so that they can be easily
# taken up by other scripts and stored in a database.
#
# The metadata is returned by dump_metadata() and can optionally be written to a file.
# Provide user defined metadata as mapping type.
#
# \code
# metadata = meta.dump_metadata()
# meta.dump_metadata(filename = "metadata.json", overwrite = False, user = {'debug': True}, indent=2)
# \endcode

import hoomd;
import hoomd_script;
import json, collections;
import time
import datetime

from collections import OrderedDict

## \internal
# \brief A Mixin to facilitate storage of simulation metadata
class _metadata:
    def __init__(self):
        # No metadata provided per default
        self.metadata_fields = []

    ## \internal
    # \brief Return the metadata
    def get_metadata(self):
        data = OrderedDict()

        for m in self.metadata_fields:
            data[m] = getattr(self, m)

        return data

class _metadata_from_dict:
    def __init__(self, d):
        self.d = d;

    def get_metadata(self):
        data = OrderedDict()

        for m in self.d.keys():
            data[m] = self.d[m];

        return data

## Writes simulation metadata into a file
#
# When called, this function will query all registered forces, updaters etc.
# and ask them to provide metadata. E.g. a pair potential will return
# information about parameters, the Logger will output the filename it is
# logging to, etc.
#
# Custom metadata can be provided as a dictionary.
#
# The output is aggregated into a dictionary and written to a
# JSON file, together with a timestamp.
#
# \param filename The name of the file to write JSON metadata to (optional)
# \param user Additional metadata, needs to be a mapping type, such as a dict()
# \param overwrite If true, overwrite output file if it already exists
# \param indent The json indentation size
#
# \returns metadata as a dictionary
def dump_metadata(filename=None,user=None,overwrite=False,indent=4):
    hoomd_script.util.print_status_line();

    if not hoomd_script.init.is_initialized():
        hoomd_script.context.msg.error("Need to initialize system first.\n")
        raise RuntimeError("Error writing out metadata.")

    metadata = []
    obj = OrderedDict()

    if user is not None:
        if not isinstance(user, collections.Mapping):
            hoomd_script.context.msg.warning("Extra meta data needs to be a mapping type. Ignoring.\n")
        else:
            obj['user'] = _metadata_from_dict(user);

    if not overwrite and filename is not None:
        try:
            with open(filename) as f:
                metadata = json.load(f)
                hoomd_script.context.msg.notice(2,"Appending to file {1}." % filename)
        except Exception:
            pass

    # Generate time stamp
    ts = time.time()
    st = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S')
    obj['timestamp'] = st
    obj['context'] = hoomd_script.context.ExecutionContext()
    obj['hoomd'] = hoomd_script.context.HOOMDContext()

    global_objs = [hoomd_script.data.system_data(hoomd_script.globals.system_definition)];
    global_objs += hoomd_script.globals.forces;
    global_objs += hoomd_script.globals.constraint_forces;
    global_objs += [hoomd_script.globals.integrator];
    global_objs += hoomd_script.globals.integration_methods;
    global_objs += hoomd_script.globals.forces
    global_objs += hoomd_script.globals.analyzers;
    global_objs += hoomd_script.globals.updaters;

    # add list of objects to JSON
    for o in global_objs:
        if o is not None:
            name = o.__module__+'.'+o.__class__.__name__;
            if len(name) > 13 and name[:13] == 'hoomd_script.':
                name = name[13:];
            obj[name] = o

    metadata.append(obj)

    # handler for unknown objects
    def default_handler(obj):
        try:
            return obj.get_metadata()
        except (AttributeError, NotImplementedError):
            return None

    # dump to JSON
    meta_str = json.dumps(metadata,default=default_handler,indent=indent)

    # only write files on rank 0
    if filename is not None and hoomd_script.comm.get_rank() == 0:
        with open(filename, 'w') as file:
            file.write(meta_str)
    return json.loads(meta_str)
