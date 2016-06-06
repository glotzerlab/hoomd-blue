// Copyright (c) 2009-2016 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.

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
                const std::string &filename);

            /// Python binding to initialize the system from a set of
            /// restoration properties
            boost::shared_ptr<SystemSnapshot> initializePy(boost::python::dict &pyModes);

            /// Python binding to restore part of the system from a set of
            /// restoration properties. Values are first taken from the
            /// given system definition.
            void restorePy(boost::python::dict &pyModes, boost::shared_ptr<SystemDefinition> sysdef);

            /// Grab the greatest timestep from the most recent
            /// restoration or initialization stage
            unsigned int getTimestep() const;

        private:
            /// Return true if the Record indicates a property we know how
            /// to restore
            bool knownProperty(const gtar::Record &rec) const;

            /// Insert one or more known records to restore into the given
            /// set if the records match the given name
            bool insertRecord(const std::string &name, std::set<gtar::Record> &rec) const;

            /// Convert a particular python dict into a std::map
            std::map<std::set<gtar::Record>, std::string> parseModes(boost::python::dict &pyModes);

            /// Initialize the system given a set of modes
            boost::shared_ptr<SystemSnapshot> initialize(const std::map<std::set<gtar::Record>, std::string> &modes);

            /// Restore part of a system given a system definition and a
            /// set of modes
            void restore(boost::shared_ptr<SystemDefinition> &sysdef, const std::map<std::set<gtar::Record>, std::string> &modes);

            /// Fill in any missing data in the given snapshot and perform
            /// basic consistency checks
            void fillSnapshot(boost::shared_ptr<SystemSnapshot> snapshot);

            /// Restore a system from bits of the given snapshot and the
            /// given restoration modes
            boost::shared_ptr<SystemSnapshot> restoreSnapshot(
                boost::shared_ptr<SystemSnapshot> &systemSnap, const std::map<std::set<gtar::Record>, std::string> &modes);

            /// Restore a set of records for the same frame
            void restoreSimultaneous(boost::shared_ptr<SystemSnapshot> snapshot,
                                     const std::set<gtar::Record> &records, std::string frame);

            /// Restore a single property
            void restoreSingle(boost::shared_ptr<SystemSnapshot> snap,
                               const gtar::Record &rec);

            /// Parse a type_names.json file
            std::vector<std::string> parseTypeNames(const std::string &json);

            /// Saved execution configuration
            boost::shared_ptr<const ExecutionConfiguration> m_exec_conf;
            /// Saved trajectory archive object
            gtar::GTAR m_traj;
            /// Set of known records we found in the current trajectory archive
            std::vector<gtar::Record> m_knownRecords;
            /// Cached timestep
            unsigned int m_timestep;
        };


void export_GetarInitializer();

}

#endif
