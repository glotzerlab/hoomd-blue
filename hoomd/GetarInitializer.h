/*
  Highly Optimized Object-oriented Many-particle Dynamics -- Blue Edition
  (HOOMD-blue) Open Source Software License Copyright 2009-2014 The Regents of
  the University of Michigan All rights reserved.

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

#ifndef __GETARINITIALIZER_H_
#define __GETARINITIALIZER_H_

#include "SnapshotSystemData.h"
#include "hoomd/extern/libgetar/src/GTAR.hpp"
#include "hoomd/extern/libgetar/src/Record.hpp"
#include "GetarDumpWriter.h"
#include "hoomd/GetarDumpIterators.h"
#include <boost/shared_ptr.hpp>

#include <map>
#include <string>
#include <vector>

using boost::shared_ptr;
using std::map;
using std::set;
using std::string;
using std::vector;
using namespace boost::python;
using namespace gtar;

namespace getardump{

    /// Object to use to restore HOOMD system snapshots
    class GetarInitializer
        {
        public:
            /// Constructor
            ///
            /// :param exec_conf: Execution configuration to use
            /// :param filename: Filename to restore from
            GetarInitializer(boost::shared_ptr<const ExecutionConfiguration> exec_conf,
                const string &filename);

            /// Python binding to initialize the system from a set of
            /// restoration properties
            shared_ptr<SystemSnapshot> initializePy(dict &pyModes);

            /// Python binding to restore part of the system from a set of
            /// restoration properties. Values are first taken from the
            /// given system definition.
            void restorePy(dict &pyModes, shared_ptr<SystemDefinition> sysdef);

            /// Grab the greatest timestep from the most recent
            /// restoration or initialization stage
            unsigned int getTimestep() const;

        private:
            /// Return true if the Record indicates a property we know how
            /// to restore
            bool knownProperty(const Record &rec) const;

            /// Insert one or more known records to restore into the given
            /// set if the records match the given name
            bool insertRecord(const string &name, std::set<Record> &rec) const;

            /// Convert a particular python dict into a std::map
            map<set<Record>, string> parseModes(dict &pyModes);

            /// Initialize the system given a set of modes
            shared_ptr<SystemSnapshot> initialize(const map<set<Record>, string> &modes);

            /// Restore part of a system given a system definition and a
            /// set of modes
            void restore(shared_ptr<SystemDefinition> &sysdef, const map<set<Record>, string> &modes);

            /// Fill in any missing data in the given snapshot and perform
            /// basic consistency checks
            void fillSnapshot(shared_ptr<SystemSnapshot> snapshot);

            /// Restore a system from bits of the given snapshot and the
            /// given restoration modes
            shared_ptr<SystemSnapshot> restoreSnapshot(
                shared_ptr<SystemSnapshot> &systemSnap, const map<set<Record>, string> &modes);

            /// Restore a set of records for the same frame
            void restoreSimultaneous(shared_ptr<SystemSnapshot> snapshot,
                const set<Record> &records, string frame);

            /// Restore a single property
            void restoreSingle(shared_ptr<SystemSnapshot> snap,
                const Record &rec);

            /// Parse a type_names.json file
            vector<string> parseTypeNames(const string &json);

            /// Saved execution configuration
            boost::shared_ptr<const ExecutionConfiguration> m_exec_conf;
            /// Saved trajectory archive object
            GTAR m_traj;
            /// Set of known records we found in the current trajectory archive
            vector<Record> m_knownRecords;
            /// Cached timestep
            unsigned int m_timestep;
        };


void export_GetarInitializer();

}

#endif
