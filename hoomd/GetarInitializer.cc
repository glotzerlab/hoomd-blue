// Copyright (c) 2009-2019 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.

#include "GetarInitializer.h"
#include <iostream>

namespace py = pybind11;

namespace getardump{

    using std::shared_ptr;
    using std::auto_ptr;
    using std::endl;
    using std::vector;
    using std::map;
    using std::max;
    using std::runtime_error;
    using std::set;
    using std::string;
    using std::stringstream;
    using namespace gtar;

    GetarInitializer::GetarInitializer(shared_ptr<const ExecutionConfiguration> exec_conf,
        const string &filename):
        m_exec_conf(exec_conf), m_traj(), m_knownRecords(), m_timestep(0)
        {
        try
            {
#ifdef ENABLE_MPI
        // don't open on non-0 MPI ranks
            if (m_exec_conf->isRoot())
#endif
            m_traj.reset(new GTAR(filename, Read));
            }
        catch(runtime_error &error)
            {
            m_exec_conf->msg->error() << error.what() << endl;
            throw;
            }

        if(m_traj.get())
            {
            const vector<Record> allRecords(m_traj->getRecordTypes());

            for(vector<Record>::const_iterator iter(allRecords.begin());
                iter != allRecords.end(); ++iter)
                {
                if(knownProperty(*iter))
                    m_knownRecords.push_back(*iter);
                }
            }
        }

    map<set<Record>, string> GetarInitializer::parseModes(py::dict &pyModes)
        {
        map<set<Record>, string> modes;

        if(!m_traj.get())
            return modes;

        for (auto item : pyModes)
            {
            py::tuple pyKey = py::cast<py::tuple>(item.first);
            string value = py::cast<string>(item.second);
            set<Record> key;
            for(unsigned int j(0); j < py::len(pyKey); ++j)
                {
                string name = pyKey[j].cast<string>();
                if(!insertRecord(name, key))
                    {
                    const string error_message(
                        string("Can't find the requested property ") + name);
                    m_exec_conf->msg->error() << error_message << endl;
                    throw runtime_error(error_message);
                    }
                else if(name == "any")
                    {
                    for(set<Record>::const_iterator iter(key.begin());
                        iter != key.end(); ++iter)
                        {
                        set<Record> single;
                        single.insert(*iter);
                        modes[single] = value;
                        }
                    }
                }

            modes[key] = value;
            }

        return modes;
        }

    shared_ptr<SystemSnapshot> GetarInitializer::initializePy(py::dict &pyModes)
        {
        map<set<Record>, string> modes(parseModes(pyModes));
        return initialize(modes);
        }

    void GetarInitializer::restorePy(
        py::dict &pyModes, shared_ptr<SystemDefinition> sysdef)
        {
        map<set<Record>, string> modes(parseModes(pyModes));
        restore(sysdef, modes);
        }

    unsigned int GetarInitializer::getTimestep() const
        {
        return m_timestep;
        }

    shared_ptr<SystemSnapshot> GetarInitializer::initialize(const map<set<Record>, string> &modes)
        {
        shared_ptr<SystemSnapshot> snap(new SystemSnapshot());
        return restoreSnapshot(snap, modes);
        }

    void GetarInitializer::restore(shared_ptr<SystemDefinition> &sysdef,
        const map<set<Record>, string> &modes)
        {
        shared_ptr<SystemSnapshot> snap(
            takeSystemSnapshot(sysdef, true, true, true, true, true, true, true, true));
        restoreSnapshot(snap, modes);

        sysdef->initializeFromSnapshot(snap);
        }

    shared_ptr<SystemSnapshot> GetarInitializer::restoreSnapshot(shared_ptr<SystemSnapshot> &systemSnap,
        const map<set<Record>, string> &modes)
        {
        if(!m_traj.get())
            return systemSnap;

        for(map<set<Record>, string>::const_iterator iter(modes.begin());
            iter != modes.end(); ++iter)
            {
            if("any" == iter->second)
                {
                for(set<Record>::const_iterator setIter(iter->first.begin());
                    setIter != iter->first.end(); ++setIter)
                    {
                    set<Record> singleItem;
                    singleItem.insert(*setIter);
                    restoreSimultaneous(systemSnap, singleItem, "latest");
                    }
                }
            else
                restoreSimultaneous(systemSnap, iter->first, iter->second);
            }

        fillSnapshot(systemSnap);

        return systemSnap;
        }

    void GetarInitializer::fillSnapshot(shared_ptr<SystemSnapshot> snapshot)
        {
        unsigned int pdata_N(snapshot->particle_data.size);

        if(!pdata_N)
            {
            const string error_message("No per-particle properties found to restore");
            m_exec_conf->msg->error() << error_message << endl;
            throw runtime_error(error_message);
            }

        if(snapshot->particle_data.pos.size() != pdata_N)
            {
            if(snapshot->particle_data.pos.size())
                {
                stringstream message;
                message << "Expected to find " << pdata_N << " particles, but found " <<
                    snapshot->particle_data.pos.size() << " positions" << endl;
                m_exec_conf->msg->error() << message.str() << endl;
                throw runtime_error(message.str());
                }
            m_exec_conf->msg->notice(3) << "Filling particle positions with the value (0, 0, 0)" << endl;
            snapshot->particle_data.pos = vector<vec3<Scalar> >(pdata_N, vec3<Scalar>(0, 0, 0));
            }

        if(snapshot->particle_data.vel.size() != pdata_N)
            {
            if(snapshot->particle_data.vel.size())
                {
                stringstream message;
                message << "Expected to find " << pdata_N << " particles, but found " <<
                    snapshot->particle_data.vel.size() << " velocities" << endl;
                m_exec_conf->msg->error() << message.str() << endl;
                throw runtime_error(message.str());
                }
            m_exec_conf->msg->notice(3) << "Filling particle velocities with the value (0, 0, 0)" << endl;
            snapshot->particle_data.vel = vector<vec3<Scalar> >(pdata_N, vec3<Scalar>(0, 0, 0));
            }

        if(snapshot->particle_data.accel.size() != pdata_N)
            {
            if(snapshot->particle_data.accel.size())
                {
                stringstream message;
                message << "Expected to find " << pdata_N << " particles, but found " <<
                    snapshot->particle_data.accel.size() << " accelerations" << endl;
                m_exec_conf->msg->error() << message.str() << endl;
                throw runtime_error(message.str());
                }
            m_exec_conf->msg->notice(3) << "Filling particle accelerations with the value (0, 0, 0)" << endl;
            snapshot->particle_data.accel = vector<vec3<Scalar> >(pdata_N, vec3<Scalar>(0, 0, 0));
            }

        if(snapshot->particle_data.type.size() != pdata_N)
            {
            if(snapshot->particle_data.type.size())
                {
                stringstream message;
                message << "Expected to find " << pdata_N << " particles, but found " <<
                    snapshot->particle_data.type.size() << " types" << endl;
                m_exec_conf->msg->error() << message.str() << endl;
                throw runtime_error(message.str());
                }
            m_exec_conf->msg->notice(3) << "Filling particle types with the value 0" << endl;
            snapshot->particle_data.type = vector<unsigned int>(pdata_N, 0);
            }

        if(snapshot->particle_data.mass.size() != pdata_N)
            {
            if(snapshot->particle_data.mass.size())
                {
                stringstream message;
                message << "Expected to find " << pdata_N << " particles, but found " <<
                    snapshot->particle_data.mass.size() << " masses" << endl;
                m_exec_conf->msg->error() << message.str() << endl;
                throw runtime_error(message.str());
                }
            m_exec_conf->msg->notice(3) << "Filling particle masses with the value 1" << endl;
            snapshot->particle_data.mass = vector<Scalar>(pdata_N, 1);
            }

        if(snapshot->particle_data.charge.size() != pdata_N)
            {
            if(snapshot->particle_data.charge.size())
                {
                stringstream message;
                message << "Expected to find " << pdata_N << " particles, but found " <<
                    snapshot->particle_data.charge.size() << " charges" << endl;
                m_exec_conf->msg->error() << message.str() << endl;
                throw runtime_error(message.str());
                }
            m_exec_conf->msg->notice(3) << "Filling particle charges with the value 0" << endl;
            snapshot->particle_data.charge = vector<Scalar>(pdata_N, 0);
            }

        if(snapshot->particle_data.diameter.size() != pdata_N)
            {
            if(snapshot->particle_data.diameter.size())
                {
                stringstream message;
                message << "Expected to find " << pdata_N << " particles, but found " <<
                    snapshot->particle_data.diameter.size() << " diameters" << endl;
                m_exec_conf->msg->error() << message.str() << endl;
                throw runtime_error(message.str());
                }
            m_exec_conf->msg->notice(3) << "Filling particle diameters with the value 1" << endl;
            snapshot->particle_data.diameter = vector<Scalar>(pdata_N, 1);
            }

        if(snapshot->particle_data.image.size() != pdata_N)
            {
            if(snapshot->particle_data.image.size())
                {
                stringstream message;
                message << "Expected to find " << pdata_N << " particles, but found " <<
                    snapshot->particle_data.image.size() << " images" << endl;
                m_exec_conf->msg->error() << message.str() << endl;
                throw runtime_error(message.str());
                }
            m_exec_conf->msg->notice(3) << "Filling particle images with the value (0, 0, 0)" << endl;
            snapshot->particle_data.image = vector<int3>(pdata_N, make_int3(0, 0, 0));
            }

        if(snapshot->particle_data.body.size() != pdata_N)
            {
            if(snapshot->particle_data.body.size())
                {
                stringstream message;
                message << "Expected to find " << pdata_N << " particles, but found " <<
                    snapshot->particle_data.body.size() << " bodies" << endl;
                m_exec_conf->msg->error() << message.str() << endl;
                throw runtime_error(message.str());
                }
            m_exec_conf->msg->notice(3) << "Filling particle bodies with the value -1" << endl;
            snapshot->particle_data.body = vector<unsigned int>(pdata_N, (unsigned int) -1);
            }

        if(snapshot->particle_data.orientation.size() != pdata_N)
            {
            if(snapshot->particle_data.orientation.size())
                {
                stringstream message;
                message << "Expected to find " << pdata_N << " particles, but found " <<
                    snapshot->particle_data.orientation.size() << " orientations" << endl;
                m_exec_conf->msg->error() << message.str() << endl;
                throw runtime_error(message.str());
                }
            m_exec_conf->msg->notice(3) << "Filling particle orientations with the value (1, 0, 0, 0)" << endl;
            snapshot->particle_data.orientation = vector<quat<Scalar> >(pdata_N, quat<Scalar>(1, vec3<Scalar>(0, 0, 0)));
            }

        if(snapshot->particle_data.angmom.size() != pdata_N)
            {
            if(snapshot->particle_data.angmom.size())
                {
                stringstream message;
                message << "Expected to find " << pdata_N << " particles, but found " <<
                    snapshot->particle_data.angmom.size() << " angular momenta" << endl;
                m_exec_conf->msg->error() << message.str() << endl;
                throw runtime_error(message.str());
                }
            m_exec_conf->msg->notice(3) << "Filling particle angular momenta with the value (0, 0, 0, 0)" << endl;
            snapshot->particle_data.angmom = vector<quat<Scalar> >(pdata_N, quat<Scalar>(0, vec3<Scalar>(0, 0, 0)));
            }

        if(snapshot->particle_data.inertia.size() != pdata_N)
            {
            if(snapshot->particle_data.inertia.size())
                {
                stringstream message;
                message << "Expected to find " << pdata_N << " particles, but found " <<
                    snapshot->particle_data.inertia.size() << " moments of inertia" << endl;
                m_exec_conf->msg->error() << message.str() << endl;
                throw runtime_error(message.str());
                }
            m_exec_conf->msg->notice(3) << "Filling particle moment of inertia with the value (1, 1, 1)" << endl;
            snapshot->particle_data.inertia = vector<vec3<Scalar> >(pdata_N, vec3<Scalar>(1, 1, 1));
            }

        unsigned int maxtype(*std::max_element(snapshot->particle_data.type.begin(),
                snapshot->particle_data.type.end()));
        for(unsigned int i(snapshot->particle_data.type_mapping.size()); i < maxtype + 1; ++i)
            {
            snapshot->particle_data.type_mapping.push_back(string(1, 'A' + (char) i));
            }

        unsigned int bond_N(snapshot->bond_data.type_id.size());

        if(snapshot->bond_data.groups.size() != bond_N)
            {
            stringstream message;
            message << "Expected to find " << bond_N << " bonds, but found " <<
                snapshot->bond_data.groups.size() << " (i, j) pairs" << endl;
            m_exec_conf->msg->error() << message.str() << endl;
            throw runtime_error(message.str());
            }

        if(bond_N)
            {
            unsigned int maxbondtype(*std::max_element(snapshot->bond_data.type_id.begin(),
                    snapshot->bond_data.type_id.end()));
            for(unsigned int i(snapshot->bond_data.type_mapping.size()); i < maxbondtype + 1; ++i)
                {
                snapshot->bond_data.type_mapping.push_back(string(1, 'A' + (char) i));
                }
            }

        unsigned int pair_N(snapshot->pair_data.type_id.size());

        if(snapshot->pair_data.groups.size() != pair_N)
            {
            stringstream message;
            message << "Expected to find " << pair_N << " pairs, but found " <<
                snapshot->pair_data.groups.size() << " (i, j) pairs" << endl;
            m_exec_conf->msg->error() << message.str() << endl;
            throw runtime_error(message.str());
            }

        if(pair_N)
            {
            unsigned int maxpairtype(*std::max_element(snapshot->pair_data.type_id.begin(),
                    snapshot->pair_data.type_id.end()));
            for(unsigned int i(snapshot->pair_data.type_mapping.size()); i < maxpairtype + 1; ++i)
                {
                snapshot->pair_data.type_mapping.push_back(string(1, 'A' + (char) i));
                }
            }


        unsigned int angle_N(snapshot->angle_data.type_id.size());

        if(snapshot->angle_data.groups.size() != angle_N)
            {
            stringstream message;
            message << "Expected to find " << angle_N << " angles, but found " <<
                snapshot->angle_data.groups.size() << " (i, j, k) triplets" << endl;
            m_exec_conf->msg->error() << message.str() << endl;
            throw runtime_error(message.str());
            }

        if(angle_N)
            {
            unsigned int maxangletype(*std::max_element(snapshot->angle_data.type_id.begin(),
                    snapshot->angle_data.type_id.end()));
            for(unsigned int i(snapshot->angle_data.type_mapping.size()); i < maxangletype + 1; ++i)
                {
                snapshot->angle_data.type_mapping.push_back(string(1, 'A' + (char) i));
                }
            }

        unsigned int dihedral_N(snapshot->dihedral_data.type_id.size());

        if(snapshot->dihedral_data.groups.size() != dihedral_N)
            {
            stringstream message;
            message << "Expected to find " << dihedral_N << " dihedrals, but found " <<
                snapshot->dihedral_data.groups.size() << " (i, j, k, l) quartets" << endl;
            m_exec_conf->msg->error() << message.str() << endl;
            throw runtime_error(message.str());
            }

        if(dihedral_N)
            {
            unsigned int maxdihedraltype(*std::max_element(snapshot->dihedral_data.type_id.begin(),
                    snapshot->dihedral_data.type_id.end()));
            for(unsigned int i(snapshot->dihedral_data.type_mapping.size()); i < maxdihedraltype + 1; ++i)
                {
                snapshot->dihedral_data.type_mapping.push_back(string(1, 'A' + (char) i));
                }
            }

        unsigned int improper_N(snapshot->improper_data.type_id.size());

        if(snapshot->improper_data.groups.size() != improper_N)
            {
            stringstream message;
            message << "Expected to find " << improper_N << " impropers, but found " <<
                snapshot->improper_data.groups.size() << " (i, j, k, l) quartets" << endl;
            m_exec_conf->msg->error() << message.str() << endl;
            throw runtime_error(message.str());
            }

        if(improper_N)
            {
            unsigned int maximpropertype(*std::max_element(snapshot->improper_data.type_id.begin(),
                    snapshot->improper_data.type_id.end()));
            for(unsigned int i(snapshot->improper_data.type_mapping.size()); i < maximpropertype + 1; ++i)
                {
                snapshot->improper_data.type_mapping.push_back(string(1, 'A' + (char) i));
                }
            }
        }

    void GetarInitializer::restoreSimultaneous(shared_ptr<SystemSnapshot> snapshot,
        const set<Record> &records, string frame)
        {
        set<string, IndexCompare> availableFrames;

        for(set<Record>::const_iterator iter(records.begin());
            iter != records.end(); ++iter)
            {
            if(iter->getBehavior() == Discrete)
                {
                vector<string> recordFrames(m_traj->queryFrames(*iter));

                if(iter == records.begin())
                    {
                    for(vector<string>::const_iterator frameIter(recordFrames.begin());
                        frameIter != recordFrames.end(); ++frameIter)
                        availableFrames.insert(*frameIter);
                    }
                else
                    {
                    set<string, IndexCompare> intersectFrames;
                    for(vector<string>::const_iterator frameIter(recordFrames.begin());
                        frameIter != recordFrames.end(); ++frameIter)
                        {
                        if(availableFrames.find(*frameIter) != availableFrames.end())
                            intersectFrames.insert(*frameIter);
                        }
                    availableFrames = intersectFrames;
                    }
                }
            else if(iter->getBehavior() == Constant)
                availableFrames.insert("");
            }

        vector<string> foundFrames(availableFrames.begin(), availableFrames.end());
        string selectedFrame;
        if(foundFrames.size() && ("latest" == frame || "any" == frame || "" == foundFrames.back()))
            selectedFrame = foundFrames.back();
        else if(foundFrames.size() && "earliest" == frame)
            selectedFrame = foundFrames.front();
        else if(find(foundFrames.begin(), foundFrames.end(), frame) != foundFrames.end())
            selectedFrame = frame;
        else
            {
            if(!foundFrames.size())
                {
                stringstream msg;
                msg << "Couldn't find a consistent frame for the following properties: ";
                for(set<Record>::const_iterator iter(records.begin());
                    iter != records.end();)
                    {
                    msg << iter->getPath();
                    if(++iter != records.end())
                        msg << ", ";
                    }

                m_exec_conf->msg->error() << msg.str() << endl;
                throw runtime_error(msg.str());
                }
            else
                {
                stringstream msg;
                msg << "Couldn't find data at frame " << frame << " for the following properties: ";
                for(set<Record>::const_iterator iter(records.begin());
                    iter != records.end();)
                    {
                    msg << iter->getPath();
                    if(++iter != records.end())
                        msg << ", ";
                    }

                m_exec_conf->msg->error() << msg.str() << endl;
                throw runtime_error(msg.str());
                }
            }

        m_timestep = max(m_timestep, (unsigned int) atoi(selectedFrame.c_str()));

        for(set<Record>::const_iterator iter(records.begin());
            iter != records.end(); ++iter)
            {
            Record rec(*iter);
            if(iter->getBehavior() == Discrete)
                rec.setIndex(selectedFrame);
            else
                rec.setIndex(string(""));

            restoreSingle(snapshot, rec);
            }
        }

    void GetarInitializer::restoreSingle(shared_ptr<SystemSnapshot> snap,
        const Record &rec)
        {
        bool found(false);

        // uint quantities
        if(rec.getName() == "dimensions" || rec.getName() == "type" ||
            rec.getName() == "tag")
            {
            vector<unsigned int> data;
            if(rec.getFormat() == UInt32)
                {
                SharedArray<uint32_t> rawData(m_traj->readIndividual<uint32_t>(rec.getPath()));
                found = rawData.get();
                TypecastIterator<uint32_t, unsigned int> begin(rawData.begin());
                TypecastIterator<uint32_t, unsigned int> end(rawData.end());

                data = vector<unsigned int>(begin, end);
                // TODO make sure body -1's here still turn into -1's?
                // maybe should just store body ID's as int's?
                }
            else if(rec.getFormat() == UInt64)
                {
                SharedArray<uint64_t> rawData(m_traj->readIndividual<uint64_t>(rec.getPath()));
                found = rawData.get();
                TypecastIterator<uint64_t, unsigned int> begin(rawData.begin());
                TypecastIterator<uint64_t, unsigned int> end(rawData.end());

                data = vector<unsigned int>(begin, end);
                }
            else
                {
                const string error_message("Can't understand the format of the data at " + rec.getPath());
                m_exec_conf->msg->error() << error_message << endl;
                throw runtime_error(error_message);
                }

            if(!found)
                {
                const string error_message("Found no data at " + rec.getPath());
                m_exec_conf->msg->error() << error_message << endl;
                throw runtime_error(error_message);
                }

            if(rec.getName() == "dimensions")
                {
                snap->dimensions = data[0];
                }
            else if(rec.getName() == "type")
                {
                if(rec.getGroup() == "bond")
                    {
                    snap->bond_data.type_id = data;
                    }
                else if(rec.getGroup() == "angle")
                    {
                    snap->angle_data.type_id = data;
                    }
                else if(rec.getGroup() == "dihedral")
                    {
                    snap->dihedral_data.type_id = data;
                    }
                else if(rec.getGroup() == "improper")
                    {
                    snap->improper_data.type_id = data;
                    }
                else if(rec.getGroup() == "pair")
                    {
                    snap->pair_data.type_id = data;
                    }
                else
                    {
                    snap->particle_data.type = data;
                    snap->particle_data.size = data.size();
                    }
                }
            else if(rec.getName() == "tag")
                {
                if(rec.getGroup() == "bond")
                    {
                    vector<group_storage<2> > groupData(InvGroupTagIterator<2>(data.begin()),
                        InvGroupTagIterator<2>(data.end()));
                    snap->bond_data.groups = groupData;
                    }
                else if(rec.getGroup() == "angle")
                    {
                    vector<group_storage<3> > groupData(InvGroupTagIterator<3>(data.begin()),
                        InvGroupTagIterator<3>(data.end()));
                    snap->angle_data.groups = groupData;
                    }
                else if(rec.getGroup() == "dihedral")
                    {
                    vector<group_storage<4> > groupData(InvGroupTagIterator<4>(data.begin()),
                        InvGroupTagIterator<4>(data.end()));
                    snap->dihedral_data.groups = groupData;
                    }
                else if(rec.getGroup() == "improper")
                    {
                    vector<group_storage<4> > groupData(InvGroupTagIterator<4>(data.begin()),
                        InvGroupTagIterator<4>(data.end()));
                    snap->improper_data.groups = groupData;
                    }
                if(rec.getGroup() == "pair")
                    {
                    vector<group_storage<2> > groupData(InvGroupTagIterator<2>(data.begin()),
                        InvGroupTagIterator<2>(data.end()));
                    snap->pair_data.groups = groupData;
                    }
                }
            else
                {
                string msg("Failed to locate where to set the property at ");
                msg += rec.getPath();
                msg += " (programmer error)";
                m_exec_conf->msg->error() << msg << endl;
                throw runtime_error(msg);
                }
            }
        // int quantities
        else if(rec.getName() == "body")
            {
            vector<unsigned int> data;
            if(rec.getFormat() == Int32)
                {
                SharedArray<int32_t> rawData(m_traj->readIndividual<int32_t>(rec.getPath()));
                found = rawData.get();
                TypecastIterator<int32_t, unsigned int> begin(rawData.begin());
                TypecastIterator<int32_t, unsigned int> end(rawData.end());

                data = vector<unsigned int>(begin, end);
                }
            else if(rec.getFormat() == Int64)
                {
                SharedArray<int64_t> rawData(m_traj->readIndividual<int64_t>(rec.getPath()));
                found = rawData.get();
                TypecastIterator<int64_t, unsigned int> begin(rawData.begin());
                TypecastIterator<int64_t, unsigned int> end(rawData.end());

                data = vector<unsigned int>(begin, end);
                }
            else
                {
                const string error_message("Can't understand the format of the data at " + rec.getPath());
                m_exec_conf->msg->error() << error_message << endl;
                throw runtime_error(error_message);
                }

            if(!found)
                {
                const string error_message("Found no data at " + rec.getPath());
                m_exec_conf->msg->error() << error_message << endl;
                throw runtime_error(error_message);
                }

            if(rec.getName() ==  "body")
                {
                snap->particle_data.body = data;
                snap->particle_data.size = data.size();
                }
            else
                {
                string msg("Failed to locate where to set the property at ");
                msg += rec.getPath();
                msg += " (programmer error)";
                m_exec_conf->msg->error() << msg << endl;
                throw runtime_error(msg);
                }
            }
        // int3 quantities
        else if(rec.getName() == "image")
            {
            vector<int3> data;
            if(rec.getFormat() == Int32)
                {
                SharedArray<int32_t> rawData(m_traj->readIndividual<int32_t>(rec.getPath()));
                found = rawData.get();

                if(rawData.size() % 3)
                    {
                    stringstream msg;
                    msg << "Data at " << rec.getPath() << " are not evenly divisible by 3";
                    m_exec_conf->msg->error() << msg.str() << endl;
                    throw runtime_error(msg.str());
                    }

                int3Iterator<int32_t> begin(rawData.begin());
                int3Iterator<int32_t> end(rawData.end());

                data = vector<int3>(begin, end);
                }
            else if(rec.getFormat() == Int64)
                {
                SharedArray<int64_t> rawData(m_traj->readIndividual<int64_t>(rec.getPath()));
                found = rawData.get();

                if(rawData.size() % 3)
                    {
                    stringstream msg;
                    msg << "Data at " << rec.getPath() << " are not evenly divisible by 3";
                    m_exec_conf->msg->error() << msg.str() << endl;
                    throw runtime_error(msg.str());
                    }

                int3Iterator<int64_t> begin(rawData.begin());
                int3Iterator<int64_t> end(rawData.end());

                data = vector<int3>(begin, end);
                }
            else
                {
                const string error_message("Can't understand the format of the data at " + rec.getPath());
                m_exec_conf->msg->error() << error_message << endl;
                throw runtime_error(error_message);
                }

            if(!found)
                {
                const string error_message("Found no data at " + rec.getPath());
                m_exec_conf->msg->error() << error_message << endl;
                throw runtime_error(error_message);
                }

            if(rec.getName() ==  "image")
                {
                snap->particle_data.image = data;
                snap->particle_data.size = data.size();
                }
            else
                {
                string msg("Failed to locate where to set the property at ");
                msg += rec.getPath();
                msg += " (programmer error)";
                m_exec_conf->msg->error() << msg << endl;
                throw runtime_error(msg);
                }
            }
        // Scalar quantities
        else if(rec.getName() == "box" || rec.getName() == "mass" ||
            rec.getName() == "charge" || rec.getName() == "diameter" ||
            rec.getName() == "moment_inertia_tensor")
            {
            vector<Scalar> data;
            if(rec.getFormat() == Float32)
                {
                SharedArray<float> rawData(m_traj->readIndividual<float>(rec.getPath()));
                found = rawData.get();
                TypecastIterator<float, Scalar> begin(rawData.begin());
                TypecastIterator<float, Scalar> end(rawData.end());

                data = vector<Scalar>(begin, end);
                }
            else if(rec.getFormat() == Float64)
                {
                SharedArray<double> rawData(m_traj->readIndividual<double>(rec.getPath()));
                found = rawData.get();
                TypecastIterator<double, Scalar> begin(rawData.begin());
                TypecastIterator<double, Scalar> end(rawData.end());

                data = vector<Scalar>(begin, end);
                }
            else
                throw runtime_error("Can't understand the format of the data at " + rec.getPath());

            if(!found)
                {
                throw runtime_error("Found no data at " + rec.getPath());
                }

            if(rec.getName() == "box")
                {
                BoxDim box(data[0], data[1], data[2]);
                if(data.size() == 6)
                    box.setTiltFactors(data[3], data[4], data[5]);
                snap->global_box = box;
                }
            else if(rec.getName() == "mass")
                {
                snap->particle_data.mass = data;
                snap->particle_data.size = data.size();
                }
            else if(rec.getName() == "charge")
                {
                snap->particle_data.charge = data;
                snap->particle_data.size = data.size();
                }
            else if(rec.getName() == "diameter")
                {
                snap->particle_data.diameter = data;
                snap->particle_data.size = data.size();
                }
            else if(rec.getName() == "moment_inertia_tensor")
                {
                vector<vec3<Scalar> > newData(InvInertiaTensorIterator<Scalar>(data.begin()),
                    InvInertiaTensorIterator<Scalar>(data.end()));
                snap->particle_data.inertia = newData;
                }
            else
                {
                string msg("Failed to locate where to set the property at ");
                msg += rec.getPath();
                msg += " (programmer error)";
                m_exec_conf->msg->error() << msg;
                throw runtime_error(msg);
                }
            }
        // Scalar3/vec3<Scalar> properties, depending on hoomd
        // version. Note that velocity is for particles, not rigid
        // bodies here
        else if(rec.getName() == "position" || rec.getName() == "velocity" ||
            rec.getName() == "acceleration" || rec.getName() == "moment_inertia")
            {
            vector<vec3<Scalar> > data;
            if(rec.getFormat() == Float32)
                {
                SharedArray<float> rawData(m_traj->readIndividual<float>(rec.getPath()));
                found = rawData.get();

                if(rawData.size() % 3)
                    {
                    stringstream msg;
                    msg << "Data at " << rec.getPath() << " are not evenly divisible by 3";
                    m_exec_conf->msg->error() << msg.str() << endl;
                    throw runtime_error(msg.str());
                    }

                Vec3Iterator<float> begin(rawData.begin());
                Vec3Iterator<float> end(rawData.end());

                data = vector<vec3<Scalar> >(begin, end);
                }
            else if(rec.getFormat() == Float64)
                {
                SharedArray<double> rawData(m_traj->readIndividual<double>(rec.getPath()));
                found = rawData.get();

                if(rawData.size() % 3)
                    {
                    stringstream msg;
                    msg << "Data at " << rec.getPath() << " are not evenly divisible by 3";
                    m_exec_conf->msg->error() << msg.str() << endl;
                    throw runtime_error(msg.str());
                    }

                Vec3Iterator<double> begin(rawData.begin());
                Vec3Iterator<double> end(rawData.end());

                data = vector<vec3<Scalar> >(begin, end);
                }
            else
                {
                const string error_message("Can't understand the format of the data at " + rec.getPath());
                m_exec_conf->msg->error() << error_message << endl;
                throw runtime_error(error_message);
                }

            if(!found)
                {
                const string error_message("Found no data at " + rec.getPath());
                m_exec_conf->msg->error() << error_message << endl;
                throw runtime_error(error_message);
                }

            if(rec.getName() == "position")
                {
                snap->particle_data.pos = data;
                snap->particle_data.size = data.size();
                }
            else if(rec.getName() == "velocity")
                {
                snap->particle_data.vel = data;
                snap->particle_data.size = data.size();
                }
            else if(rec.getName() == "acceleration")
                {
                snap->particle_data.accel = data;
                snap->particle_data.size = data.size();
                }
            else if(rec.getName() == "moment_inertia")
                {
                snap->particle_data.inertia = data;
                snap->particle_data.size = data.size();
                }
            else
                {
                string msg("Failed to locate where to set the property at ");
                msg += rec.getPath();
                msg += " (programmer error)";
                m_exec_conf->msg->error() << msg << endl;
                throw runtime_error(msg);
                }
            }
        // quat<Scalar> quantities
        else if(rec.getName() == "angular_momentum_quat")
            {
            vector<quat<Scalar> > data;
            if(rec.getFormat() == Float32)
                {
                SharedArray<float> rawData(m_traj->readIndividual<float>(rec.getPath()));
                found = rawData.get();

                if(rawData.size() % 4)
                    {
                    stringstream msg;
                    msg << "Data at " << rec.getPath() << " are not evenly divisible by 4";
                    m_exec_conf->msg->error() << msg.str() << endl;
                    throw runtime_error(msg.str());
                    }

                QuatIterator<float> begin(rawData.begin());
                QuatIterator<float> end(rawData.end());

                data = vector<quat<Scalar> >(begin, end);
                }
            else if(rec.getFormat() == Float64)
                {
                SharedArray<double> rawData(m_traj->readIndividual<double>(rec.getPath()));
                found = rawData.get();

                if(rawData.size() % 4)
                    {
                    stringstream msg;
                    msg << "Data at " << rec.getPath() << " are not evenly divisible by 4";
                    m_exec_conf->msg->error() << msg.str() << endl;
                    throw runtime_error(msg.str());
                    }

                QuatIterator<double> begin(rawData.begin());
                QuatIterator<double> end(rawData.end());

                data = vector<quat<Scalar> >(begin, end);
                }
            else
                {
                const string error_message("Can't understand the format of the data at " + rec.getPath());
                m_exec_conf->msg->error() << error_message << endl;
                throw runtime_error(error_message);
                }

            if(!found)
                {
                const string error_message("Found no data at " + rec.getPath());
                m_exec_conf->msg->error() << error_message << endl;
                throw runtime_error(error_message);
                }

            if(rec.getName() == "angular_momentum_quat")
                {
                snap->particle_data.angmom = data;
                snap->particle_data.size = data.size();
                }
            else
                {
                string msg("Failed to locate where to set the property at ");
                msg += rec.getPath();
                msg += " (programmer error)";
                m_exec_conf->msg->error() << msg << endl;
                throw runtime_error(msg);
                }
            }
        // Scalar4 or quat<Scalar> property
        else if(rec.getName() == "orientation")
            {
            vector<quat<Scalar> > data;
            if(rec.getFormat() == Float32)
                {
                SharedArray<float> rawData(m_traj->readIndividual<float>(rec.getPath()));
                found = rawData.get();

                if(rawData.size() % 4)
                    {
                    stringstream msg;
                    msg << "Data at " << rec.getPath() << " are not evenly divisible by 4";
                    m_exec_conf->msg->error() << msg.str() << endl;
                    throw runtime_error(msg.str());
                    }

                QuatIterator<float> begin(rawData.begin());
                QuatIterator<float> end(rawData.end());

                data = vector<quat<Scalar> >(begin, end);
                }
            else if(rec.getFormat() == Float64)
                {
                SharedArray<double> rawData(m_traj->readIndividual<double>(rec.getPath()));
                found = rawData.get();

                if(rawData.size() % 4)
                    {
                    stringstream msg;
                    msg << "Data at " << rec.getPath() << " are not evenly divisible by 4";
                    m_exec_conf->msg->error() << msg.str() << endl;
                    throw runtime_error(msg.str());
                    }

                QuatIterator<double> begin(rawData.begin());
                QuatIterator<double> end(rawData.end());

                data = vector<quat<Scalar> >(begin, end);
                }
            else
                {
                const string error_message("Can't understand the format of the data at " + rec.getPath());
                m_exec_conf->msg->error() << error_message << endl;
                throw runtime_error(error_message);
                }

            if(!found)
                {
                const string error_message("Found no data at " + rec.getPath());
                m_exec_conf->msg->error() << error_message << endl;
                throw runtime_error(error_message);
                }

            if(rec.getName() == "orientation")
                {
                snap->particle_data.orientation = data;
                snap->particle_data.size = data.size();
                }
            else
                {
                string msg("Failed to locate where to set the property at ");
                msg += rec.getPath();
                msg += " (programmer error)";
                m_exec_conf->msg->error() << msg << endl;
                throw runtime_error(msg);
                }
            }
        else if(rec.getName() == "type_names.json")
            {
            vector<string> names;

            SharedArray<char> rawData(m_traj->readBytes(rec.getPath()));
            found = rawData.get();
            string data(rawData.get(), rawData.size());

            if(!found)
                {
                const string error_message("Found no data at " + rec.getPath());
                m_exec_conf->msg->error() << error_message << endl;
                throw runtime_error(error_message);
                }

            names = parseTypeNames(data);

            if(rec.getGroup() == "bond")
                {
                snap->bond_data.type_mapping = names;
                }
            else if(rec.getGroup() == "angle")
                {
                snap->angle_data.type_mapping = names;
                }
            else if(rec.getGroup() == "dihedral")
                {
                snap->dihedral_data.type_mapping = names;
                }
            else if(rec.getGroup() == "improper")
                {
                snap->improper_data.type_mapping = names;
                }
            else if(rec.getGroup() == "pair")
                {
                snap->pair_data.type_mapping = names;
                }
            else
                {
                snap->particle_data.type_mapping = names;
                }
            }
        else
            {
            const string error_message("Unknown record with path " + rec.getPath());
            m_exec_conf->msg->error() << error_message << endl;
            throw runtime_error(error_message);
            }
        }

    /// Parse the contents of a "type_names.json" file. What it
    /// actually does is grabs every other string in between pairs of
    /// quotation marks, starting at the second one.
    vector<string> GetarInitializer::parseTypeNames(const string &json)
        {
        bool save(false);
        vector<string> result;

        stringstream current;
        for(string::const_iterator iter(json.begin());
            iter != json.end(); ++iter)
            {
            if(*iter == '"')
                {
                if(save)
                    {
                    result.push_back(current.str());
                    current.clear();
                    current.str(string());
                    }
                save = !save;
                }
            else if(save)
                current << *iter;
            }

        return result;
        }

    bool GetarInitializer::knownProperty(const Record &rec) const
        {
        if(rec.getBehavior() == Continuous)
            return false;

        return (rec.getName() == "acceleration" ||
            rec.getName() == "angular_momentum" ||
            rec.getName() == "angular_momentum_quat" ||
            rec.getName() == "body" ||
            rec.getName() == "box" ||
            rec.getName() == "center_of_mass" ||
            rec.getName() == "charge" ||
            rec.getName() == "diameter" ||
            rec.getName() == "dimensions" ||
            rec.getName() == "image" ||
            rec.getName() == "mass" ||
            rec.getName() == "moment_inertia" ||
            rec.getName() == "moment_inertia_tensor" ||
            rec.getName() == "orientation" ||
            rec.getName() == "position" ||
            rec.getName() == "tag" ||
            rec.getName() == "type" ||
            rec.getName() == "type_names.json" ||
            rec.getName() == "velocity" ||
            rec.getName() == "virial");
        }

    bool GetarInitializer::insertRecord(const string &name, set<Record> &recs) const
        {
        bool result(false);
        string recGroup, recName(name);

        if(name.substr(0, 5) == "body_")
            {
            recGroup = "rigid_body";
            recName = name.substr(5);
            }
        else if(name.substr(0, 5) == "bond_")
            {
            recGroup = "bond";
            recName = name.substr(5);
            }
        else if(name.substr(0, 6) == "angle_")
            {
            recGroup = "angle";
            recName = name.substr(6);
            }
        else if(name.substr(0, 9) == "dihedral_")
            {
            recGroup = "dihedral";
            recName = name.substr(9);
            }
        else if(name.substr(0, 9) == "improper_")
            {
            recGroup = "improper";
            recName = name.substr(9);
            }
        else if(name.substr(0, 5) == "pair_")
            {
            recGroup = "pair";
            recName = name.substr(5);
            }
        if(recName == "type_names")
            recName = "type_names.json";

        for(vector<Record>::const_iterator iter(m_knownRecords.begin());
            iter != m_knownRecords.end(); ++iter)
            {
            const bool takeAny("" == recGroup && "any" == recName);
            const bool anyWithGroup("any" == recName && iter->getGroup() == recGroup);
            const bool match(iter->getGroup() == recGroup && iter->getName() == recName);

            if(takeAny || anyWithGroup || match)
                {
                result = true;
                recs.insert(*iter);
                }
            }

        return result;
        }

    void export_GetarInitializer(py::module& m)
        {
        py::class_<GetarInitializer, shared_ptr<GetarInitializer> >(m,"GetarInitializer")
            .def(py::init< shared_ptr<const ExecutionConfiguration>, string>())
            .def("initialize", &GetarInitializer::initializePy)
            .def("restore", &GetarInitializer::restorePy)
            .def("getTimestep", &GetarInitializer::getTimestep)
            ;
        }

}
