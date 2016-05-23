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

#ifndef __GETARDUMPER_H_
#define __GETARDUMPER_H_

#include "hoomd/Analyzer.h"
#include "hoomd/SnapshotSystemData.h"
#include "hoomd/extern/libgetar/src/GTAR.hpp"
#include "hoomd/extern/libgetar/src/Record.hpp"
#include "hoomd/GetarDumpIterators.h"
#include <boost/shared_ptr.hpp>

#include <map>
#include <string>
#include <vector>

namespace getardump{

    typedef SnapshotSystemData<Scalar> SystemSnapshot;
    boost::shared_ptr<SystemSnapshot> takeSystemSnapshot(
        boost::shared_ptr<SystemDefinition>, bool, bool, bool, bool, bool, bool, bool);

    /// Known operation modes
    enum GetarDumpMode {
        Overwrite, /// Overwrite the file if it exists already
        Append,    /// Add records to the file if it exists already
        OneShot};  /// Overwrite the file each time we dump a frame

    /// List of properties we know how to [calculate and] dump
    enum Property {
        AngleNames,
        AngleTags,
        AngleTypes,
        AngularMomentum,
        Body,  // particle rigid body index
        BodyAngularMomentum,
        BodyCOM,
        BodyImage,
        BodyMomentInertia,
        BodyOrientation,
        BodyVelocity,
        BondNames,
        BondTags,
        BondTypes,
        Box,
        Charge,
        Diameter,
        DihedralNames,
        DihedralTags,
        DihedralTypes,
        Dimensions,
        Image,
        ImproperNames,
        ImproperTags,
        ImproperTypes,
        Mass,
        MomentInertia,
        Orientation,
        Position,
        PotentialEnergy,
        Type,
        TypeNames,
        Velocity,
        Virial};

    /// ways to index values of a GetarDumpWriter::NeedSnapshotMap
    enum NeedSnapshotIdx {
        NeedSystem = 0,
        NeedPData,
        NeedBond,
        NeedAngle,
        NeedDihedral,
        NeedImproper,
        NeedRigid,
        NeedIntegrator};

    /// Helper class to keep track of which snapshots are needed for
    /// each period
    class NeedSnapshots
    {
    public:
        /// Ctor. Initialize all needs to false.
        NeedSnapshots();

        /// Copy ctor
        NeedSnapshots(const NeedSnapshots &rhs);

        /// Assignment
        void operator=(const NeedSnapshots &rhs);

        /// Index with int
        bool &operator[](unsigned int index);
        /// Index with enum
        bool &operator[](NeedSnapshotIdx index);

        /// Index with int
        const bool &operator[](unsigned int index) const;
        /// Index with enum
        const bool &operator[](NeedSnapshotIdx index) const;

    private:
        /// Array of needed snapshot values corresponding to `NeedSnapshotIdx`
        bool needs[9];
    };

    /// Return the filename within the archive of a getar dump property
    std::string getPropertyName(Property prop, bool highPrecision=false);

    /// A self-contained description of a property to dump and how it
    /// should be stored.
    class GetarDumpDescription
    {
    public:
        /// Constructor.
        ///
        /// :param prop: Property to dump
        /// :param res: Resolution to dump the property at
        /// :param behavior: Time behavior of the property
        /// :param highPrecision: If true, try to dump a high-precision version
        /// :param compression: Compress with the given compression
        GetarDumpDescription(Property prop, gtar::Resolution res, gtar::Behavior behavior,
                           bool highPrecision, gtar::CompressMode compression):
            m_prop(prop), m_res(res), m_behavior(behavior),
            m_highPrecision(highPrecision), m_compression(compression)
        {
            if(prop == BodyAngularMomentum || prop == BodyMomentInertia ||
               prop == BodyCOM || prop == BodyImage || prop == BodyOrientation ||
               prop == BodyVelocity)
                m_prefix += "rigid_body/";

            if(prop == AngleNames || prop == AngleTags || prop == AngleTypes)
                m_prefix += "angle/";

            if(prop == BondNames || prop == BondTags || prop == BondTypes)
                m_prefix += "bond/";

            if(prop == DihedralNames || prop == DihedralTags || prop == DihedralTypes)
                m_prefix += "dihedral/";

            if(prop == ImproperNames || prop == ImproperTags || prop == ImproperTypes)
                m_prefix += "improper/";

            if(behavior == gtar::Discrete)
            {
                m_prefix += "frames/";
                m_suffix += "/";
            }

            m_suffix += getPropertyName(prop, m_highPrecision);

            if(m_res == gtar::Uniform)
                m_suffix += ".uni";
            else if(m_res == gtar::Individual)
                m_suffix += ".ind";
        }

        /// Copy constructor
        GetarDumpDescription(const GetarDumpDescription &rhs):
            m_prop(rhs.m_prop), m_res(rhs.m_res), m_behavior(rhs.m_behavior),
            m_highPrecision(rhs.m_highPrecision), m_compression(rhs.m_compression),
            m_prefix(rhs.m_prefix), m_suffix(rhs.m_suffix)
        {}

        /// Equality
        bool operator==(const GetarDumpDescription &rhs) const
        {
            return m_prop == rhs.m_prop && m_res == rhs.m_res &&
                m_behavior == rhs.m_behavior && m_highPrecision == rhs.m_highPrecision;
        }

        /// Returns the path within the archive where this property
        /// should be stored
        std::string getFormattedPath(unsigned int timestep) const
        {
            if(m_behavior == gtar::Constant)
                return m_prefix + m_suffix;
            else
            {
                std::ostringstream conv;
                conv << timestep;
                return m_prefix + conv.str() + m_suffix;
            }
        }

        /// Property to dump
        Property m_prop;
        /// Resolution to dump at
        gtar::Resolution m_res;
        /// Time behavior to dump at
        gtar::Behavior m_behavior;
        /// true if we want to save a high-precision version
        bool m_highPrecision;
        /// Compression which should be used when dumping
        gtar::CompressMode m_compression;
        /// Path prefix
        std::string m_prefix;
        /// Path suffix
        std::string m_suffix;
    };

    /// HOOMD analyzer which periodically dumps a set of properties
    class GetarDumpWriter: public Analyzer
    {
    public:
        typedef std::map<unsigned int, std::vector<GetarDumpDescription> > PeriodMap;
        typedef std::map<unsigned int, NeedSnapshots> NeedSnapshotMap;

        /// Constructor
        ///
        /// :param sysdef: System definition to grab snapshots from
        /// :param filename: File name to dump to
        /// :param operationMode: Operation mode
        /// :param offset: Timestep offset
        GetarDumpWriter(boost::shared_ptr<SystemDefinition> sysdef,
                  const std::string &filename, GetarDumpMode operationMode, unsigned int offset=0);

        /// Destructor: closes the file and finalizes any IO
        ~GetarDumpWriter();

        /// Close the getar file manually after finalizing any IO
        void close();

        /// Get needed pdata flags
        virtual PDataFlags getRequestedPDataFlags()
        {
            PDataFlags flags;

            for(PeriodMap::iterator iteri(m_periods.begin());
                iteri != m_periods.end(); ++iteri)
                for(std::vector<GetarDumpDescription>::iterator iterj(iteri->second.begin());
                    iterj != iteri->second.end(); ++iterj)
                {
                    const Property prop(iterj->m_prop);
                    flags[pdata_flag::potential_energy] = (flags[pdata_flag::potential_energy] |
                                                           (prop == PotentialEnergy));
                    flags[pdata_flag::pressure_tensor] = (flags[pdata_flag::pressure_tensor] |
                                                           (unsigned int)(prop == Virial));
                }
            return flags;
        }

        /// Called every timestep
        void analyze(unsigned int timestep);

        /// Write any GetarDumpDescription for the given timestep
        void write(gtar::GTAR::BulkWriter &writer, const GetarDumpDescription &desc, unsigned int timestep);
        /// Write an individual GetarDumpDescription for the given timestep
        void writeIndividual(gtar::GTAR::BulkWriter &writer, const GetarDumpDescription &desc, unsigned int timestep);
        /// Write a uniform GetarDumpDescription for the given timestep
        void writeUniform(gtar::GTAR::BulkWriter &writer, const GetarDumpDescription &desc, unsigned int timestep);
        /// Write a text GetarDumpDescription for the given timestep
        void writeText(gtar::GTAR::BulkWriter &writer, const GetarDumpDescription &desc, unsigned int timestep);

        /// Calculate the correct period for all of the properties
        /// activated on this analyzer
        unsigned int getPeriod() const;

        /// Set a property to be dumped at the given period
        void setPeriod(Property prop, gtar::Resolution res, gtar::Behavior behavior,
                       bool highPrecision, gtar::CompressMode compression, unsigned int period);

        /// Remove all instances of a dump triplet from being dumped
        void removeDump(Property prop, gtar::Resolution res, gtar::Behavior behavior,
                        bool highPrecision);

    private:
        /// File archive interface
        boost::shared_ptr<gtar::GTAR> m_archive;
        /// Stored properties to dump
        PeriodMap m_periods;
        /// Timestep offset
        unsigned int m_offset;
        /// Saved static records for one-shot mode
        std::vector<GetarDumpDescription> m_staticRecords;
        /// Saved operation mode
        GetarDumpMode m_operationMode;
        /// Saved dump filename
        std::string m_filename;
        /// Temporary name to write to in one-shot mode
        std::string m_tempName;

        /// System snapshot to manipulate
        boost::shared_ptr<SystemSnapshot> m_systemSnap;
        /// Map detailing when we need which snapshots
        NeedSnapshotMap m_neededSnapshots;
    };

void export_GetarDumpWriter();

}

#endif
