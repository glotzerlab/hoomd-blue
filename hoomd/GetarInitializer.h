// Copyright (c) 2009-2019 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.

#ifndef __GETARINITIALIZER_H_
#define __GETARINITIALIZER_H_

#include "SnapshotSystemData.h"
#include "hoomd/extern/libgetar/src/GTAR.hpp"
#include "hoomd/extern/libgetar/src/Record.hpp"
#include "GetarDumpWriter.h"
#include "hoomd/GetarDumpIterators.h"
#include <memory>

#include <map>
#include <string>
#include <vector>

#ifndef NVCC
#include <hoomd/extern/pybind/include/pybind11/pybind11.h>
#endif

namespace getardump{

    /// Object to use to restore HOOMD system snapshots
    class PYBIND11_EXPORT GetarInitializer
        {
        public:
            /// Constructor
            ///
            /// :param exec_conf: Execution configuration to use
            /// :param filename: Filename to restore from
            GetarInitializer(std::shared_ptr<const ExecutionConfiguration> exec_conf,
                const std::string &filename);

            /// Python binding to initialize the system from a set of
            /// restoration properties
            std::shared_ptr<SystemSnapshot> initializePy(pybind11::dict &pyModes);

            /// Python binding to restore part of the system from a set of
            /// restoration properties. Values are first taken from the
            /// given system definition.
            void restorePy(pybind11::dict &pyModes, std::shared_ptr<SystemDefinition> sysdef);

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
            std::map<std::set<gtar::Record>, std::string> parseModes(pybind11::dict &pyModes);

            /// Initialize the system given a set of modes
            std::shared_ptr<SystemSnapshot> initialize(const std::map<std::set<gtar::Record>, std::string> &modes);

            /// Restore part of a system given a system definition and a
            /// set of modes
            void restore(std::shared_ptr<SystemDefinition> &sysdef, const std::map<std::set<gtar::Record>, std::string> &modes);

            /// Fill in any missing data in the given snapshot and perform
            /// basic consistency checks
            void fillSnapshot(std::shared_ptr<SystemSnapshot> snapshot);

            /// Restore a system from bits of the given snapshot and the
            /// given restoration modes
            std::shared_ptr<SystemSnapshot> restoreSnapshot(
                std::shared_ptr<SystemSnapshot> &systemSnap, const std::map<std::set<gtar::Record>, std::string> &modes);

            /// Restore a set of records for the same frame
            void restoreSimultaneous(std::shared_ptr<SystemSnapshot> snapshot,
                                     const std::set<gtar::Record> &records, std::string frame);

            /// Restore a single property
            void restoreSingle(std::shared_ptr<SystemSnapshot> snap,
                               const gtar::Record &rec);

            /// Parse a type_names.json file
            std::vector<std::string> parseTypeNames(const std::string &json);

            /// Saved execution configuration
            std::shared_ptr<const ExecutionConfiguration> m_exec_conf;
            /// Saved trajectory archive object
            std::auto_ptr<gtar::GTAR> m_traj;
            /// Set of known records we found in the current trajectory archive
            std::vector<gtar::Record> m_knownRecords;
            /// Cached timestep
            unsigned int m_timestep;
        };

#ifndef NVCC
void export_GetarInitializer(pybind11::module& m);
#endif

}

#endif
