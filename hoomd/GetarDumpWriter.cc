// Copyright (c) 2009-2019 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.

#include "ParticleData.h"
#include "GetarDumpWriter.h"
#include "GetarDumpIterators.h"

#include <cstdio>
#include <iostream>

namespace py = pybind11;

namespace getardump{

    using namespace gtar;
    using std::shared_ptr;
    using std::endl;
    using std::map;
    using std::max;
    using std::runtime_error;
    using std::string;
    using std::stringstream;
    using std::vector;

    // Wrapper function
    shared_ptr<SystemSnapshot> takeSystemSnapshot(
        shared_ptr<SystemDefinition> sysdef, bool particles, bool bonds,
        bool angles, bool dihedrals, bool impropers, bool pairs, bool rigid, bool integrator)
        {
        return sysdef->takeSnapshot<Scalar>(particles, bonds, angles, dihedrals,
            impropers, rigid, integrator, pairs);
        }

// greatest common denominator, using Euclid's algorithm
    template<typename T>
    T gcd(T x, T y)
        {
        while(y != 0)
            {
            const T temp(y);
            y = x % y;
            x = temp;
            }
        return x;
        }

// least common multiple
    template<typename T>
    T lcm(T x, T y)
        {
        return x*y/gcd<T>(x, y);
        }

// return true if a prop needs particle data snapshots
    bool needSnapshot(NeedSnapshotIdx idx, Property prop)
        {
        switch(idx)
            {
            case NeedSystem:
                switch(prop)
                    {
                    case Box:
                    case Dimensions:
                        return true;
                    default:
                        break;
                    }
            case NeedPData:
                switch(prop)
                    {
                    case AngularMomentum:
                    case Body:
                    case Charge:
                    case Diameter:
                    case Image:
                    case Mass:
                    case MomentInertia:
                    case Orientation:
                    case Position:
                    case Type:
                    case TypeNames:
                    case Velocity:
                        return true;
                    default:
                        break;
                    }
            case NeedBond:
                switch(prop)
                    {
                    case BondNames:
                    case BondTags:
                    case BondTypes:
                        return true;
                    default:
                        break;
                    }
            case NeedAngle:
                switch(prop)
                    {
                    case AngleNames:
                    case AngleTags:
                    case AngleTypes:
                        return true;
                    default:
                        break;
                    }
            case NeedDihedral:
                switch(prop)
                    {
                    case DihedralNames:
                    case DihedralTags:
                    case DihedralTypes:
                        return true;
                    default:
                        break;
                    }
            case NeedImproper:
                switch(prop)
                    {
                    case ImproperNames:
                    case ImproperTags:
                    case ImproperTypes:
                        return true;
                    default:
                        break;
                    }
            case NeedPair:
                switch(prop)
                    {
                    case PairNames:
                    case PairTags:
                    case PairTypes:
                        return true;
                    default:
                        break;
                    }
            case NeedRigid:
                switch(prop)
                    {
                    case BodyAngularMomentum:
                    case BodyCOM:
                        // case BodyMomentInertia:
                    case BodyImage:
                        // case BodyOrientation:
                    case BodyVelocity:
                        return true;
                    default:
                        break;
                    }
            case NeedIntegrator:
                switch(prop)
                    {
                    default:
                        break;
                    }
            }
        return false;
        }

    string getPropertyName(Property prop, bool highPrecision)
        {
        switch(prop)
            {
            case AngularMomentum:
                return string("angular_momentum_quat.f32");
            case AngleNames:
                return string("type_names.json");
            case AngleTags:
                return string("tag.u32");
            case AngleTypes:
                return string("type.u32");
            case Body:
                return string("body.i32");
            case BodyAngularMomentum:
                return string("angular_momentum.f32");
            case BodyCOM:
                return string("center_of_mass.f32");
            case BodyImage:
                return string("image.i32");
            case BodyMomentInertia:
                return string("moment_inertia.f32");
            case BodyOrientation:
                return string("orientation.f32");
            case BodyVelocity:
                return string("velocity.f32");
            case BondNames:
                return string("type_names.json");
            case BondTags:
                return string("tag.u32");
            case BondTypes:
                return string("type.u32");
            case Box:
                return string("box.") + string(highPrecision? "f64": "f32");
            case Charge:
                return string("charge.f32");
            case Diameter:
                return string("diameter.f32");
            case DihedralNames:
                return string("type_names.json");
            case DihedralTags:
                return string("tag.u32");
            case DihedralTypes:
                return string("type.u32");
            case Dimensions:
                return string("dimensions.u32");
            case Image:
                return string("image.i32");
            case ImproperNames:
                return string("type_names.json");
            case ImproperTags:
                return string("tag.u32");
            case ImproperTypes:
                return string("type.u32");
            case PairNames:
                return string("type_names.json");
            case PairTags:
                return string("tag.u32");
            case PairTypes:
                return string("type.u32");
            case Mass:
                return string("mass.f32");
            case MomentInertia:
                return string("moment_inertia.f32");
            case Orientation:
                return string("orientation.") + string(highPrecision? "f64": "f32");
            case Position:
                return string("position.") + string(highPrecision? "f64": "f32");
            case PotentialEnergy:
                return string("potential_energy.f32");
            case Type:
                return string("type.u32");
            case TypeNames:
                return string("type_names.json");
            case Velocity:
                return string("velocity.") + string(highPrecision? "f64": "f32");
            case Virial:
                return string("virial.f32");
            default:
                return string("unknown.u8");
            }
        }

    // Make the text of a json file encoding a list of type names
    string makeTypeList(const vector<string> &names)
        {
        stringstream result;

        result << '[';

        for(vector<string>::const_iterator iter(names.begin());
            iter != names.end(); ++iter)
            {
            result << '"' << *iter << '"';
            if(iter + 1 != names.end())
                result << ',';
            }

        result << ']';

        return result.str();
        }

    NeedSnapshots::NeedSnapshots()
        {
        for(unsigned int i(0); i < 9; ++i)
            needs[i] = false;
        }

    NeedSnapshots::NeedSnapshots(const NeedSnapshots &rhs)
        {
        this->operator=(rhs);
        }

    void NeedSnapshots::operator=(const NeedSnapshots &rhs)
        {
        for(unsigned int i(0); i < 9; ++i)
            needs[i] = rhs.needs[i];
        }

    bool &NeedSnapshots::operator[](unsigned int index)
        {
        return needs[index];
        }

    bool &NeedSnapshots::operator[](NeedSnapshotIdx index)
        {
        return needs[(unsigned int) index];
        }

    const bool &NeedSnapshots::operator[](unsigned int index) const
        {
        return needs[index];
        }

    const bool &NeedSnapshots::operator[](NeedSnapshotIdx index) const
        {
        return needs[(unsigned int) index];
        }

    GetarDumpWriter::GetarDumpWriter(std::shared_ptr<SystemDefinition> sysdef,
        const std::string &filename, GetarDumpMode operationMode, unsigned int offset):
        Analyzer(sysdef), m_archive(), m_periods(), m_offset(offset),
        m_staticRecords(), m_operationMode(operationMode), m_filename(filename),
        m_tempName(), m_systemSnap(), m_neededSnapshots()
        {
        if(m_operationMode == getardump::OneShot)
            {
            const size_t dot(m_filename.find_last_of("."));
            if(dot != std::string::npos)
                {
                const string base(m_filename.substr(0,dot));
                const string suffix(m_filename.substr(dot));
                m_tempName = base + ".getardump_temp" + suffix;
                }
            else
                {
                m_tempName = m_filename + ".getardump_temp";
                }
            }
        else
            {
            OpenMode openMode(operationMode == getardump::Append? gtar::Append: gtar::Write);
#ifdef ENABLE_MPI
            // only open archive on root processor
            if (m_exec_conf->isRoot())
#endif
                m_archive.reset(new GTAR(filename, openMode));
            }

        m_systemSnap = takeSystemSnapshot(m_sysdef, true, true, true, true, true, true, true, true);
        }

    GetarDumpWriter::~GetarDumpWriter()
        {}

    void GetarDumpWriter::close()
        {
        if(m_archive)
            m_archive->close();
        }

    void GetarDumpWriter::analyze(unsigned int timestep)
        {
        const unsigned int shiftedTimestep(timestep - m_offset);
        bool neededSnapshots[9] = {false, false, false, false, false,
                                   false, false, false, false};
        bool ranThisStep(false);

        for(NeedSnapshotMap::iterator pIter(m_neededSnapshots.begin());
            pIter != m_neededSnapshots.end(); ++pIter)
            {
            if(!(shiftedTimestep%pIter->first))
                {
                for(unsigned int i(0); i < 9; ++i)
                    neededSnapshots[i] |= pIter->second[i];
                }
            }

        if(neededSnapshots[NeedSystem])
            m_systemSnap = takeSystemSnapshot(m_sysdef,
                neededSnapshots[NeedPData], neededSnapshots[NeedBond],
                neededSnapshots[NeedAngle], neededSnapshots[NeedDihedral],
                neededSnapshots[NeedImproper], neededSnapshots[NeedPair], neededSnapshots[NeedRigid],
                neededSnapshots[NeedIntegrator]);

#ifdef ENABLE_MPI
        // only open archive on root processor
        if (!m_exec_conf->isRoot())
            return;
#endif

        if(m_operationMode == OneShot)
            {
            for(PeriodMap::iterator pIter(m_periods.begin());
                pIter != m_periods.end(); ++pIter)
                if(!(shiftedTimestep%pIter->first))
                    ranThisStep = true;

            if(ranThisStep)
                {
                m_archive.reset(new GTAR(m_tempName, gtar::Write));
                    {
                    GTAR::BulkWriter writer(*m_archive);

                    for(PeriodMap::iterator pIter(m_periods.begin());
                        pIter != m_periods.end(); ++pIter)
                        {
                        if(!(shiftedTimestep%pIter->first))
                            {
                            for(vector<GetarDumpDescription>::iterator dIter(pIter->second.begin());
                                dIter != pIter->second.end(); ++dIter)
                                {
                                write(writer, *dIter, timestep);
                                // const Property prop(dIter->m_prop);
                                // const Resolution res(dIter->m_res);

                                // string path(dIter->getFormattedPath(timestep));
                                }
                            }
                        }

                    for(vector<GetarDumpDescription>::const_iterator iter(m_staticRecords.begin());
                        iter != m_staticRecords.end(); ++iter)
                        write(writer, *iter, 0);

                    }

                m_archive.reset();
                int result(rename(m_tempName.c_str(), m_filename.c_str()));

                if(result)
                    {
                    stringstream msg;
                    msg << "Error " << result << " in one-shot file: " << strerror(result);
                    m_exec_conf->msg->error() << msg.str() << endl;
                    throw runtime_error(msg.str());
                    }
                }
            }
        else if(m_archive)
            {
            GTAR::BulkWriter writer(*m_archive);

            for(PeriodMap::iterator pIter(m_periods.begin());
                pIter != m_periods.end(); ++pIter)
                {
                if(!(shiftedTimestep%pIter->first))
                    {
                    ranThisStep = true;

                    for(vector<GetarDumpDescription>::iterator dIter(pIter->second.begin());
                        dIter != pIter->second.end(); ++dIter)
                        {
                        write(writer, *dIter, timestep);
                        // const Property prop(dIter->m_prop);
                        // const Resolution res(dIter->m_res);

                        // string path(dIter->getFormattedPath(timestep));
                        }
                    }
                }
            }
        }

    void GetarDumpWriter::write(GTAR::BulkWriter &writer, const GetarDumpDescription &desc, unsigned int timestep)
        {
        if(!m_archive)
            return;

        if(desc.m_res == Individual)
            writeIndividual(writer, desc, timestep);
        else if(desc.m_res == Text)
            writeText(writer, desc, timestep);
        else if(desc.m_res == Uniform)
            writeUniform(writer, desc, timestep);
        }

    void GetarDumpWriter::writeIndividual(GTAR::BulkWriter &writer, const GetarDumpDescription &desc, unsigned int timestep)
        {
        if(desc.m_prop == AngularMomentum)
            {
            typedef QuatsxyzIterator<float, vector<quat<Scalar> >::iterator> iter_t;
            iter_t begin(m_systemSnap->particle_data.angmom.begin());
            iter_t end(m_systemSnap->particle_data.angmom.end());
            writer.writeIndividual<iter_t, float>(
                desc.getFormattedPath(timestep), begin, end, desc.m_compression);
            }
        else if(desc.m_prop == AngleNames)
            {
            string json(makeTypeList(m_systemSnap->angle_data.type_mapping));
            writer.writeString(desc.getFormattedPath(timestep), json, desc.m_compression);
            }
        else if(desc.m_prop == AngleTags)
            {
            GroupTagIterator<3> begin(m_systemSnap->angle_data.groups.begin());
            GroupTagIterator<3> end(m_systemSnap->angle_data.groups.end());
            writer.writeIndividual<GroupTagIterator<3>, uint32_t>(
                desc.getFormattedPath(timestep), begin, end, desc.m_compression);
            }
        else if(desc.m_prop == AngleTypes)
            {
            vector<unsigned int>::iterator begin(m_systemSnap->angle_data.type_id.begin());
            vector<unsigned int>::iterator end(m_systemSnap->angle_data.type_id.end());
            writer.writeIndividual<vector<unsigned int>::iterator, uint32_t>(
                desc.getFormattedPath(timestep), begin, end, desc.m_compression);
            }
        else if(desc.m_prop == Body)
            {
            vector<unsigned int>::iterator begin(m_systemSnap->particle_data.body.begin());
            vector<unsigned int>::iterator end(m_systemSnap->particle_data.body.end());
            writer.writeIndividual<vector<unsigned int>::iterator, int32_t>(
                desc.getFormattedPath(timestep), begin, end, desc.m_compression);
            }
        else if(desc.m_prop == BondNames)
            {
            string json(makeTypeList(m_systemSnap->bond_data.type_mapping));
            writer.writeString(desc.getFormattedPath(timestep), json, desc.m_compression);
            }
        else if(desc.m_prop == BondTags)
            {
            GroupTagIterator<2> begin(m_systemSnap->bond_data.groups.begin());
            GroupTagIterator<2> end(m_systemSnap->bond_data.groups.end());
            writer.writeIndividual<GroupTagIterator<2>, uint32_t>(
                desc.getFormattedPath(timestep), begin, end, desc.m_compression);
            }
        else if(desc.m_prop == BondTypes)
            {
            vector<unsigned int>::iterator begin(m_systemSnap->bond_data.type_id.begin());
            vector<unsigned int>::iterator end(m_systemSnap->bond_data.type_id.end());
            writer.writeIndividual<vector<unsigned int>::iterator, uint32_t>(
                desc.getFormattedPath(timestep), begin, end, desc.m_compression);
            }
        else if(desc.m_prop == Charge)
            {
            vector<Scalar>::iterator begin(m_systemSnap->particle_data.charge.begin());
            vector<Scalar>::iterator end(m_systemSnap->particle_data.charge.end());
            writer.writeIndividual<vector<Scalar>::iterator, float>(
                desc.getFormattedPath(timestep), begin, end, desc.m_compression);
            }
        else if(desc.m_prop == Diameter)
            {
            vector<Scalar>::iterator begin(m_systemSnap->particle_data.diameter.begin());
            vector<Scalar>::iterator end(m_systemSnap->particle_data.diameter.end());
            writer.writeIndividual<vector<Scalar>::iterator, float>(
                desc.getFormattedPath(timestep), begin, end, desc.m_compression);
            }
        else if(desc.m_prop == DihedralNames)
            {
            string json(makeTypeList(m_systemSnap->dihedral_data.type_mapping));
            writer.writeString(desc.getFormattedPath(timestep), json, desc.m_compression);
            }
        else if(desc.m_prop == DihedralTags)
            {
            GroupTagIterator<4> begin(m_systemSnap->dihedral_data.groups.begin());
            GroupTagIterator<4> end(m_systemSnap->dihedral_data.groups.end());
            writer.writeIndividual<GroupTagIterator<4>, uint32_t>(
                desc.getFormattedPath(timestep), begin, end, desc.m_compression);
            }
        else if(desc.m_prop == DihedralTypes)
            {
            vector<unsigned int>::iterator begin(m_systemSnap->dihedral_data.type_id.begin());
            vector<unsigned int>::iterator end(m_systemSnap->dihedral_data.type_id.end());
            writer.writeIndividual<vector<unsigned int>::iterator, uint32_t>(
                desc.getFormattedPath(timestep), begin, end, desc.m_compression);
            }
        else if(desc.m_prop == Image)
            {
            typedef Int3xyzIterator<vector<int3>::iterator> iter_t;
            iter_t begin(m_systemSnap->particle_data.image.begin());
            iter_t end(m_systemSnap->particle_data.image.end());
            writer.writeIndividual<iter_t, int32_t>(
                desc.getFormattedPath(timestep), begin, end, desc.m_compression);
            }
        else if(desc.m_prop == ImproperNames)
            {
            string json(makeTypeList(m_systemSnap->improper_data.type_mapping));
            writer.writeString(desc.getFormattedPath(timestep), json, desc.m_compression);
            }
        else if(desc.m_prop == ImproperTags)
            {
            GroupTagIterator<4> begin(m_systemSnap->improper_data.groups.begin());
            GroupTagIterator<4> end(m_systemSnap->improper_data.groups.end());
            writer.writeIndividual<GroupTagIterator<4>, uint32_t>(
                desc.getFormattedPath(timestep), begin, end, desc.m_compression);
            }
        else if(desc.m_prop == ImproperTypes)
            {
            vector<unsigned int>::iterator begin(m_systemSnap->improper_data.type_id.begin());
            vector<unsigned int>::iterator end(m_systemSnap->improper_data.type_id.end());
            writer.writeIndividual<vector<unsigned int>::iterator, uint32_t>(
                desc.getFormattedPath(timestep), begin, end, desc.m_compression);
            }
        else if(desc.m_prop == PairNames)
            {
            string json(makeTypeList(m_systemSnap->pair_data.type_mapping));
            writer.writeString(desc.getFormattedPath(timestep), json, desc.m_compression);
            }
        else if(desc.m_prop == PairTags)
            {
            GroupTagIterator<2> begin(m_systemSnap->pair_data.groups.begin());
            GroupTagIterator<2> end(m_systemSnap->pair_data.groups.end());
            writer.writeIndividual<GroupTagIterator<2>, uint32_t>(
                desc.getFormattedPath(timestep), begin, end, desc.m_compression);
            }
        else if(desc.m_prop == PairTypes)
            {
            vector<unsigned int>::iterator begin(m_systemSnap->pair_data.type_id.begin());
            vector<unsigned int>::iterator end(m_systemSnap->pair_data.type_id.end());
            writer.writeIndividual<vector<unsigned int>::iterator, uint32_t>(
                desc.getFormattedPath(timestep), begin, end, desc.m_compression);
            }
        else if(desc.m_prop == Mass)
            {
            vector<Scalar>::iterator begin(m_systemSnap->particle_data.mass.begin());
            vector<Scalar>::iterator end(m_systemSnap->particle_data.mass.end());
            writer.writeIndividual<vector<Scalar>::iterator, float>(
                desc.getFormattedPath(timestep), begin, end, desc.m_compression);
            }
        else if(desc.m_prop == MomentInertia)
            {
            typedef Scalar3xyzIterator<float, vector<vec3<Scalar> >::iterator> iter_t;
            iter_t begin(m_systemSnap->particle_data.inertia.begin());
            iter_t end(m_systemSnap->particle_data.inertia.end());
            writer.writeIndividual<iter_t, float>(
                desc.getFormattedPath(timestep), begin, end, desc.m_compression);
            }
        else if(desc.m_prop == Orientation)
            {
            if(desc.m_highPrecision == false)
                {
                typedef QuatsxyzIterator<float, vector<quat<Scalar> >::iterator> iter_t;
                iter_t begin(m_systemSnap->particle_data.orientation.begin());
                iter_t end(m_systemSnap->particle_data.orientation.end());
                writer.writeIndividual<iter_t, float>(
                    desc.getFormattedPath(timestep), begin, end, desc.m_compression);
                }
            else
                {
                typedef QuatsxyzIterator<double, vector<quat<Scalar> >::iterator> iter_t;
                iter_t begin(m_systemSnap->particle_data.orientation.begin());
                iter_t end(m_systemSnap->particle_data.orientation.end());
                writer.writeIndividual<iter_t, double>(
                    desc.getFormattedPath(timestep), begin, end, desc.m_compression);
                }
            }
        else if(desc.m_prop == Position)
            {
            if(desc.m_highPrecision == false)
                {
                typedef Scalar3xyzIterator<float, vector<vec3<Scalar> >::iterator> iter_t;
                iter_t begin(m_systemSnap->particle_data.pos.begin());
                iter_t end(m_systemSnap->particle_data.pos.end());
                writer.writeIndividual<iter_t, float>(
                    desc.getFormattedPath(timestep), begin, end, desc.m_compression);
                }
            else
                {
                typedef Scalar3xyzIterator<double, vector<vec3<Scalar> >::iterator> iter_t;
                iter_t begin(m_systemSnap->particle_data.pos.begin());
                iter_t end(m_systemSnap->particle_data.pos.end());
                writer.writeIndividual<iter_t, double>(
                    desc.getFormattedPath(timestep), begin, end, desc.m_compression);
                }
            }
        else if(desc.m_prop == PotentialEnergy)
            {
            ArrayHandle<unsigned int> tags(m_pdata->getTags(),
                access_location::host, access_mode::read);
            ArrayHandle<Scalar4> handle(m_pdata->getNetForce(),
                access_location::host, access_mode::read);
            const unsigned int N(m_pdata->getN());

            map<unsigned int, Scalar4> sorted;
            for(unsigned int i(0); i < N; ++i)
                sorted[tags.data[i]] = handle.data[i];

            typedef Scalar4wIterator<MapValueIterator<unsigned int, Scalar4> > iter_t;
            iter_t begin(sorted.begin());
            iter_t end(sorted.end());
            writer.writeIndividual<iter_t, float>(
                desc.getFormattedPath(timestep), begin, end, desc.m_compression);
            }
        else if(desc.m_prop == Type)
            {
            vector<unsigned int>::iterator begin(m_systemSnap->particle_data.type.begin());
            vector<unsigned int>::iterator end(m_systemSnap->particle_data.type.end());
            writer.writeIndividual<vector<unsigned int>::iterator, uint32_t>(
                desc.getFormattedPath(timestep), begin, end, desc.m_compression);
            }
        else if(desc.m_prop == Velocity)
            {
            if(desc.m_highPrecision == false)
                {
                typedef Scalar3xyzIterator<float, vector<vec3<Scalar> >::iterator> iter_t;
                iter_t begin(m_systemSnap->particle_data.vel.begin());
                iter_t end(m_systemSnap->particle_data.vel.end());
                writer.writeIndividual<iter_t, float>(
                    desc.getFormattedPath(timestep), begin, end, desc.m_compression);
                }
            else
                {
                typedef Scalar3xyzIterator<double, vector<vec3<Scalar> >::iterator> iter_t;
                iter_t begin(m_systemSnap->particle_data.vel.begin());
                iter_t end(m_systemSnap->particle_data.vel.end());
                writer.writeIndividual<iter_t, double>(
                    desc.getFormattedPath(timestep), begin, end, desc.m_compression);
                }
            }
        else if(desc.m_prop == Virial)
            {
            ArrayHandle<unsigned int> tags(m_pdata->getTags(),
                access_location::host, access_mode::read);
            ArrayHandle<Scalar> handle(m_pdata->getNetVirial(),
                access_location::host, access_mode::read);

            const unsigned int N(m_pdata->getN());
            const unsigned int virialPitch(m_pdata->getNetVirial().getPitch());

            map<unsigned int, Scalar> sorted[6];
            for(unsigned int i(0); i < N; ++i)
                for(unsigned int j(0); j < 6; ++j)
                    sorted[j][tags.data[i]] = handle.data[j*virialPitch + i];

            typedef VirialIterator<float, MapValueIterator<unsigned int, Scalar> > iter_t;
            typedef MapValueIterator<unsigned int, Scalar> map_iter_t;

            // Produce elements in the following order: xx, xy, xz, yy, yz, zz
            map_iter_t begins[6];
            map_iter_t ends[6];
            for(unsigned int i(0); i < 6; ++i)
                {
                begins[i] = map_iter_t(sorted[i].begin());
                ends[i] = map_iter_t(sorted[i].end());
                }
            iter_t begin(begins);
            iter_t end(ends);
            writer.writeIndividual<iter_t, float>(
                desc.getFormattedPath(timestep), begin, end, desc.m_compression);
            }
        else
            {
            string msg("Asked to write an individual property we don't know: ");
            msg += desc.getFormattedPath(timestep);
            m_exec_conf->msg->error() << msg << endl;
            throw runtime_error(msg);
            }
        }

    void GetarDumpWriter::writeUniform(GTAR::BulkWriter &writer, const GetarDumpDescription &desc, unsigned int timestep)
        {
        if(desc.m_prop == Box)
            {
            Scalar3 box(m_pdata->getGlobalBox().getL());

            if(desc.m_highPrecision == false)
                {
                float arr[] = {float(box.x), float(box.y), float(box.z),
                               float(m_pdata->getGlobalBox().getTiltFactorXY()),
                               float(m_pdata->getGlobalBox().getTiltFactorXZ()),
                               float(m_pdata->getGlobalBox().getTiltFactorYZ())};
                writer.writeIndividual<float*, float>(
                    desc.getFormattedPath(timestep), arr, &arr[6], desc.m_compression);
                }
            else
                {
                double arr[] = {box.x, box.y, box.z,
                                m_pdata->getGlobalBox().getTiltFactorXY(),
                                m_pdata->getGlobalBox().getTiltFactorXZ(),
                                m_pdata->getGlobalBox().getTiltFactorYZ()};
                writer.writeIndividual<double*, double>(
                    desc.getFormattedPath(timestep), arr, &arr[6], desc.m_compression);
                }
            }
        else if(desc.m_prop == Dimensions)
            {
            writer.writeUniform<unsigned int>(desc.getFormattedPath(timestep), m_systemSnap->dimensions);
            }
        else
            {
            string msg("Unable to write the requested uniform");
            m_exec_conf->msg->error() << msg << endl;
            throw runtime_error(msg);
            }
        }

    void GetarDumpWriter::writeText(GTAR::BulkWriter &writer, const GetarDumpDescription &desc, unsigned int timestep)
        {
        if(desc.m_prop == TypeNames)
            {
            string json(makeTypeList(m_systemSnap->particle_data.type_mapping));
            writer.writeString(desc.getFormattedPath(timestep), json, desc.m_compression);
            }
        else if(desc.m_prop == AngleNames)
            {
            string json(makeTypeList(m_systemSnap->angle_data.type_mapping));
            writer.writeString(desc.getFormattedPath(timestep), json, desc.m_compression);
            }
        else if(desc.m_prop == BondNames)
            {
            string json(makeTypeList(m_systemSnap->bond_data.type_mapping));
            writer.writeString(desc.getFormattedPath(timestep), json, desc.m_compression);
            }
        else if(desc.m_prop == DihedralNames)
            {
            string json(makeTypeList(m_systemSnap->dihedral_data.type_mapping));
            writer.writeString(desc.getFormattedPath(timestep), json, desc.m_compression);
            }
        else if(desc.m_prop == ImproperNames)
            {
            string json(makeTypeList(m_systemSnap->improper_data.type_mapping));
            writer.writeString(desc.getFormattedPath(timestep), json, desc.m_compression);
            }
        else if(desc.m_prop == PairNames)
            {
            string json(makeTypeList(m_systemSnap->pair_data.type_mapping));
            writer.writeString(desc.getFormattedPath(timestep), json, desc.m_compression);
            }
        else
            {
            string msg("Unable to write the requested text property");
            m_exec_conf->msg->error() << msg << endl;
            throw runtime_error(msg);
            }
        }

    unsigned int GetarDumpWriter::getPeriod() const
        {
        unsigned int result(1);
        bool found;

        for(PeriodMap::const_iterator iter(m_periods.begin()); iter != m_periods.end(); ++iter)
            {
            if(!found)
                result = iter->first;
            else if(iter->first)
                {
                result = gcd(result, iter->first);
                found = true;
                }
            }

        return found? result: 1;
        }

    void GetarDumpWriter::setPeriod(Property prop, Resolution res, Behavior behavior,
        bool highPrecision, CompressMode compression, unsigned int period)
        {
        GetarDumpDescription desc(prop, res, behavior, highPrecision, compression);

        if(behavior == Constant)
            {
            m_staticRecords.push_back(desc);
            if(m_archive)
                {
                GTAR::BulkWriter writer(*m_archive);
                write(writer, desc, 0);
                }
            }
        else if(behavior == Discrete)
            {
            period = max(period, (unsigned int) 1);
            if(m_periods.find(period) != m_periods.end())
                {
                m_periods[period].push_back(desc);
                for(unsigned int i(1); i < 9; ++i)
                    {
                    m_neededSnapshots[period][i] |= needSnapshot((NeedSnapshotIdx) i, prop);
                    m_neededSnapshots[period][0] |= m_neededSnapshots[period][i];
                    }
                }
            else
                {
                m_periods[period] = vector<GetarDumpDescription>(1, desc);
                m_neededSnapshots[period] = NeedSnapshots();
                for(unsigned int i(1); i < 9; ++i)
                    {
                    m_neededSnapshots[period][i] = needSnapshot((NeedSnapshotIdx) i, prop);
                    m_neededSnapshots[period][0] |= m_neededSnapshots[period][i];
                    }
                }
            }
        }

    void GetarDumpWriter::removeDump(Property prop, Resolution res, Behavior behavior,
        bool highPrecision)
        {
        GetarDumpDescription desc(prop, res, behavior, highPrecision, NoCompress);
        for(PeriodMap::iterator pIter(m_periods.begin()); pIter != m_periods.end(); ++pIter)
            {
            for(vector<GetarDumpDescription>::iterator dIter(pIter->second.begin());
                dIter != pIter->second.end(); ++dIter)
                {
                if(*dIter == desc)
                    {
                    pIter->second.erase(dIter);
                    break;
                    }
                }
            }
        }

    void GetarDumpWriter::writeStr(const std::string &name, const std::string &contents, int timestep)
        {
        bool dynamic(timestep >= 0);
        gtar::Record rec("", name, "", gtar::Constant, gtar::UInt8, gtar::Text);

        if(dynamic)
            {
            std::ostringstream conv;
            conv << timestep;

            rec = gtar::Record("", name, conv.str(), gtar::Discrete, gtar::UInt8, gtar::Text);
            }

#ifdef ENABLE_MPI
        // only write on root rank
        if (m_exec_conf->isRoot())
#endif
        m_archive->writeString(rec.getPath(), contents, gtar::FastCompress);
        }

    void export_GetarDumpWriter(py::module& m)
        {
        py::class_<GetarDumpWriter, std::shared_ptr<GetarDumpWriter> >(m,"GetarDumpWriter", py::base<Analyzer>())
            .def(py::init< std::shared_ptr<SystemDefinition>, std::string, getardump::GetarDumpMode, unsigned int>())
            .def("close", &GetarDumpWriter::close)
            .def("getPeriod", &GetarDumpWriter::getPeriod)
            .def("setPeriod", &GetarDumpWriter::setPeriod)
            .def("removeDump", &GetarDumpWriter::removeDump)
            .def("writeStr", &GetarDumpWriter::writeStr)
        ;


        py::enum_<getardump::GetarDumpMode>(m,"GetarDumpMode")
            .value("Overwrite", getardump::Overwrite)
            .value("Append", getardump::Append)
            .value("OneShot", getardump::OneShot)
            .export_values()
        ;

        py::enum_<getardump::Property>(m,"GetarProperty")
            .value("AngleNames", AngleNames)
            .value("AngleTags", AngleTags)
            .value("AngleTypes", AngleTypes)
            .value("AngularMomentum", AngularMomentum)
            .value("Body", Body)
            .value("BodyAngularMomentum", BodyAngularMomentum)
            .value("BodyCOM", BodyCOM)
            .value("BodyImage", BodyImage)
            .value("BodyMomentInertia", BodyMomentInertia)
            .value("BodyOrientation", BodyOrientation)
            .value("BodyVelocity", BodyVelocity)
            .value("BondNames", BondNames)
            .value("BondTags", BondTags)
            .value("BondTypes", BondTypes)
            .value("Box", Box)
            .value("Charge", Charge)
            .value("Diameter", Diameter)
            .value("DihedralNames", DihedralNames)
            .value("DihedralTags", DihedralTags)
            .value("DihedralTypes", DihedralTypes)
            .value("Dimensions", Dimensions)
            .value("Image", Image)
            .value("ImproperNames", ImproperNames)
            .value("ImproperTags", ImproperTags)
            .value("ImproperTypes", ImproperTypes)
            .value("Mass", Mass)
            .value("MomentInertia", MomentInertia)
            .value("Orientation", Orientation)
            .value("Position", Position)
            .value("PotentialEnergy", PotentialEnergy)
            .value("Type", Type)
            .value("TypeNames", TypeNames)
            .value("Velocity", Velocity)
            .value("Virial", Virial)
            .export_values()
        ;

        py::enum_<getardump::Resolution>(m,"GetarResolution")
            .value("Text", Text)
            .value("Uniform", Uniform)
            .value("Individual", Individual)
            .export_values()
        ;

        py::enum_<getardump::Behavior>(m,"GetarBehavior")
            .value("Constant", Constant)
            .value("Discrete", Discrete)
            .value("Continuous", Continuous)
            .export_values()
        ;

        py::enum_<getardump::CompressMode>(m,"GetarCompression")
            .value("NoCompress", NoCompress)
            .value("FastCompress", FastCompress)
            .value("MediumCompress", MediumCompress)
            .value("SlowCompress", SlowCompress)
            .export_values()
        ;
        }
}
