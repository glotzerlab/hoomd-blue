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

/*! \file HOOMDDumpWriter.cc
    \brief Defines the HOOMDDumpWriter class
*/

#ifdef WIN32
#pragma warning( push )
#pragma warning( disable : 4244 )
#endif

#include <boost/python.hpp>
using namespace boost::python;

#include <sstream>
#include <fstream>
#include <stdexcept>
#include <iomanip>
#include <boost/shared_ptr.hpp>

#include "HOOMDDumpWriter.h"
#include "BondData.h"
#include "AngleData.h"
#include "DihedralData.h"
#include "WallData.h"

#ifdef ENABLE_MPI
#include "Communicator.h"
#endif

using namespace std;
using namespace boost;

/*! \param sysdef SystemDefinition containing the ParticleData to dump
    \param base_fname The base name of the file xml file to output the information

    \note .timestep.xml will be apended to the end of \a base_fname when analyze() is called.
*/
HOOMDDumpWriter::HOOMDDumpWriter(boost::shared_ptr<SystemDefinition> sysdef, std::string base_fname)
        : Analyzer(sysdef), m_base_fname(base_fname), m_output_position(true),
        m_output_image(false), m_output_velocity(false), m_output_mass(false), m_output_diameter(false),
        m_output_type(false), m_output_bond(false), m_output_angle(false), m_output_wall(false),
        m_output_dihedral(false), m_output_improper(false), m_output_accel(false), m_output_body(false),
        m_output_charge(false), m_output_orientation(false), m_output_moment_inertia(false), m_vizsigma_set(false)
    {
    m_exec_conf->msg->notice(5) << "Constructing HOOMDDumpWriter: " << base_fname << endl;
    }

HOOMDDumpWriter::~HOOMDDumpWriter()
    {
    m_exec_conf->msg->notice(5) << "Destroying HOOMDDumpWriter" << endl;
    }

/*! \param enable Set to true to enable the writing of particle positions to the files in analyze()
*/
void HOOMDDumpWriter::setOutputPosition(bool enable)
    {
    m_output_position = enable;
    }

/*! \param enable Set to true to enable the writing of particle images to the files in analyze()
*/
void HOOMDDumpWriter::setOutputImage(bool enable)
    {
    m_output_image = enable;
    }

/*!\param enable Set to true to output particle velocities to the XML file on the next call to analyze()
*/
void HOOMDDumpWriter::setOutputVelocity(bool enable)
    {
    m_output_velocity = enable;
    }

/*!\param enable Set to true to output particle masses to the XML file on the next call to analyze()
*/
void HOOMDDumpWriter::setOutputMass(bool enable)
    {
    m_output_mass = enable;
    }

/*!\param enable Set to true to output particle diameters to the XML file on the next call to analyze()
*/
void HOOMDDumpWriter::setOutputDiameter(bool enable)
    {
    m_output_diameter = enable;
    }

/*! \param enable Set to true to output particle types to the XML file on the next call to analyze()
*/
void HOOMDDumpWriter::setOutputType(bool enable)
    {
    m_output_type = enable;
    }
/*! \param enable Set to true to output bonds to the XML file on the next call to analyze()
*/
void HOOMDDumpWriter::setOutputBond(bool enable)
    {
    m_output_bond = enable;
    }
/*! \param enable Set to true to output angles to the XML file on the next call to analyze()
*/
void HOOMDDumpWriter::setOutputAngle(bool enable)
    {
    m_output_angle = enable;
    }
/*! \param enable Set to true to output walls to the XML file on the next call to analyze()
*/
void HOOMDDumpWriter::setOutputWall(bool enable)
    {
    m_output_wall = enable;
    }
/*! \param enable Set to true to output dihedrals to the XML file on the next call to analyze()
*/
void HOOMDDumpWriter::setOutputDihedral(bool enable)
    {
    m_output_dihedral = enable;
    }
/*! \param enable Set to true to output impropers to the XML file on the next call to analyze()
*/
void HOOMDDumpWriter::setOutputImproper(bool enable)
    {
    m_output_improper = enable;
    }
/*! \param enable Set to true to output acceleration to the XML file on the next call to analyze()
*/
void HOOMDDumpWriter::setOutputAccel(bool enable)
    {
    m_output_accel = enable;
    }
/*! \param enable Set to true to output body to the XML file on the next call to analyze()
*/
void HOOMDDumpWriter::setOutputBody(bool enable)
    {
    m_output_body = enable;
    }

/*! \param enable Set to true to output body to the XML file on the next call to analyze()
*/
void HOOMDDumpWriter::setOutputCharge(bool enable)
    {
    m_output_charge = enable;
    }

/*! \param enable Set to true to output orientation to the XML file on the next call to analyze()
*/
void HOOMDDumpWriter::setOutputOrientation(bool enable)
    {
    m_output_orientation = enable;
    }

/*! \param enable Set to true to output moment_inertia to the XML file on the next call to analyze()
*/
void HOOMDDumpWriter::setOutputMomentInertia(bool enable)
    {
    m_output_moment_inertia = enable;
    }

/*! \param fname File name to write
    \param timestep Current time step of the simulation
*/
void HOOMDDumpWriter::writeFile(std::string fname, unsigned int timestep)
    {
    // acquire the particle data
    SnapshotParticleData snapshot(m_pdata->getNGlobal());

    m_pdata->takeSnapshot(snapshot);

    SnapshotBondData bdata_snapshot(m_sysdef->getBondData()->getNumBondsGlobal());

    if (m_output_bond)
        {
        // take a bond data snapshot
        boost::shared_ptr<BondData> bond_data = m_sysdef->getBondData();

        bond_data->takeSnapshot(bdata_snapshot);
        }

#ifdef ENABLE_MPI
    // only the root processor writes the output file
    if (m_pdata->getDomainDecomposition() && ! m_exec_conf->isRoot())
        return;
#endif

    // open the file for writing
    ofstream f(fname.c_str());

    if (!f.good())
        {
        m_exec_conf->msg->error() << "dump.xml: Unable to open dump file for writing: " << fname << endl;
        throw runtime_error("Error writting hoomd_xml dump file");
        }

    BoxDim box = m_pdata->getGlobalBox();
    Scalar3 L = box.getL();
    Scalar xy = box.getTiltFactorXY();
    Scalar xz = box.getTiltFactorXZ();
    Scalar yz = box.getTiltFactorYZ();

    f.precision(13);
    f << "<?xml version=\"1.0\" encoding=\"UTF-8\"?>" << "\n";
    f << "<hoomd_xml version=\"1.5\">" << "\n";
    f << "<configuration time_step=\"" << timestep << "\" "
      << "dimensions=\"" << m_sysdef->getNDimensions() << "\" "
      << "natoms=\"" << m_pdata->getNGlobal() << "\" ";
    if (m_vizsigma_set)
        f << "vizsigma=\"" << m_vizsigma << "\" ";
    f << ">" << "\n";
    f << "<box " << "lx=\"" << L.x << "\" ly=\""<< L.y << "\" lz=\""<< L.z
      << "\" xy=\"" << xy << "\" xz=\"" << xz << "\" yz=\"" << yz << "\"/>" << "\n";

    f.precision(12);

    // If the position flag is true output the position of all particles to the file
    if (m_output_position)
        {
        f << "<position num=\"" << m_pdata->getNGlobal() << "\">" << "\n";
        for (unsigned int j = 0; j < m_pdata->getNGlobal(); j++)
            {
            Scalar3 pos = snapshot.pos[j];

            f << pos.x << " " << pos.y << " "<< pos.z << "\n";

            if (!f.good())
                {
                m_exec_conf->msg->error() << "dump.xml: I/O error while writing HOOMD dump file" << endl;
                throw runtime_error("Error writting HOOMD dump file");
                }
            }
        f <<"</position>" << "\n";
        }

    // If the image flag is true, output the image of each particle to the file
    if (m_output_image)
        {
        f << "<image num=\"" << m_pdata->getNGlobal() << "\">" << "\n";
        for (unsigned int j = 0; j < m_pdata->getNGlobal(); j++)
            {
            int3 image = snapshot.image[j];

            f << image.x << " " << image.y << " "<< image.z << "\n";

            if (!f.good())
                {
                m_exec_conf->msg->error() << "dump.xml: I/O error while writing HOOMD dump file" << endl;
                throw runtime_error("Error writting HOOMD dump file");
                }
            }
        f <<"</image>" << "\n";
        }

    // If the velocity flag is true output the velocity of all particles to the file
    if (m_output_velocity)
        {
        f <<"<velocity num=\"" << m_pdata->getNGlobal() << "\">" << "\n";

        for (unsigned int j = 0; j < m_pdata->getNGlobal(); j++)
            {
            Scalar3 vel = snapshot.vel[j];
            f << vel.x << " " << vel.y << " " << vel.z << "\n";
            if (!f.good())
                {
                m_exec_conf->msg->error() << "dump.xml: I/O error while writing HOOMD dump file" << endl;
                throw runtime_error("Error writting HOOMD dump file");
                }
            }

        f <<"</velocity>" << "\n";
        }

    // If the velocity flag is true output the velocity of all particles to the file
    if (m_output_accel)
        {
        f <<"<acceleration num=\"" << m_pdata->getNGlobal() << "\">" << "\n";

        for (unsigned int j = 0; j < m_pdata->getNGlobal(); j++)
            {
            Scalar3 accel = snapshot.accel[j];

            f << accel.x << " " << accel.y << " " << accel.z << "\n";
            if (!f.good())
                {
                m_exec_conf->msg->error() << "dump.xml: I/O error while writing HOOMD dump file" << endl;
                throw runtime_error("Error writting HOOMD dump file");
                }
            }

        f <<"</acceleration>" << "\n";
        }

    // If the mass flag is true output the mass of all particles to the file
    if (m_output_mass)
        {
        f <<"<mass num=\"" << m_pdata->getNGlobal() << "\">" << "\n";

        for (unsigned int j = 0; j < m_pdata->getNGlobal(); j++)
            {
            Scalar mass = snapshot.mass[j];

            f << mass << "\n";
            if (!f.good())
                {
                m_exec_conf->msg->error() << "dump.xml: I/O error while writing HOOMD dump file" << endl;
                throw runtime_error("Error writting HOOMD dump file");
                }
            }

        f <<"</mass>" << "\n";
        }

    // If the diameter flag is true output the mass of all particles to the file
    if (m_output_diameter)
        {
        f <<"<diameter num=\"" << m_pdata->getNGlobal() << "\">" << "\n";

        for (unsigned int j = 0; j < m_pdata->getNGlobal(); j++)
            {
            Scalar diameter = snapshot.diameter[j];
            f << diameter << "\n";
            if (!f.good())
                {
                m_exec_conf->msg->error() << "dump.xml: I/O error while writing HOOMD dump file" << endl;
                throw runtime_error("Error writting HOOMD dump file");
                }
            }

        f <<"</diameter>" << "\n";
        }

    // If the Type flag is true output the types of all particles to an xml file
    if  (m_output_type)
        {
        f <<"<type num=\"" << m_pdata->getNGlobal() << "\">" << "\n";
        for (unsigned int j = 0; j < m_pdata->getNGlobal(); j++)
            {
            unsigned int type = snapshot.type[j];
            f << m_pdata->getNameByType(type) << "\n";
            }
        f <<"</type>" << "\n";
        }

    // If the body flag is true output the bodies of all particles to an xml file
    if  (m_output_body)
        {
        f <<"<body num=\"" << m_pdata->getNGlobal() << "\">" << "\n";
        for (unsigned int j = 0; j < m_pdata->getNGlobal(); j++)
            {
            unsigned int body;
            int out;
            body = snapshot.body[j];
            if (body == NO_BODY)
                out = -1;
            else
                out = (int)body;

            f << out << "\n";
            }
        f <<"</body>" << "\n";
        }

    // if the bond flag is true, output the bonds to the xml file
    if (m_output_bond)
        {
        f << "<bond num=\"" << bdata_snapshot.bonds.size() << "\">" << "\n";
        boost::shared_ptr<BondData> bond_data = m_sysdef->getBondData();

        // loop over all bonds and write them out
        for (unsigned int i = 0; i < bdata_snapshot.bonds.size(); i++)
            {
            uint2 bond = bdata_snapshot.bonds[i];
            unsigned int bond_type = bdata_snapshot.type_id[i];
            f << bond_data->getNameByType(bond_type) << " " << bond.x << " " << bond.y << "\n";
            }

        f << "</bond>" << "\n";
        }

    // if the angle flag is true, output the angles to the xml file
    if (m_output_angle)
        {
        f << "<angle num=\"" << m_sysdef->getAngleData()->getNumAngles() << "\">" << "\n";
        boost::shared_ptr<AngleData> angle_data = m_sysdef->getAngleData();

        // loop over all angles and write them out
        for (unsigned int i = 0; i < angle_data->getNumAngles(); i++)
            {
            Angle angle = angle_data->getAngle(i);
            f << angle_data->getNameByType(angle.type) << " " << angle.a  << " " << angle.b << " " << angle.c << "\n";
            }

        f << "</angle>" << "\n";
        }

    // if dihedral is true, write out dihedrals to the xml file
    if (m_output_dihedral)
        {
        f << "<dihedral num=\"" << m_sysdef->getDihedralData()->getNumDihedrals() << "\">" << "\n";
        boost::shared_ptr<DihedralData> dihedral_data = m_sysdef->getDihedralData();

        // loop over all angles and write them out
        for (unsigned int i = 0; i < dihedral_data->getNumDihedrals(); i++)
            {
            Dihedral dihedral = dihedral_data->getDihedral(i);
            f << dihedral_data->getNameByType(dihedral.type) << " " << dihedral.a  << " " << dihedral.b << " "
            << dihedral.c << " " << dihedral.d << "\n";
            }

        f << "</dihedral>" << "\n";
        }

    // if improper is true, write out impropers to the xml file
    if (m_output_improper)
        {
        f << "<improper num=\"" << m_sysdef->getImproperData()->getNumDihedrals() << "\">" << "\n";
        boost::shared_ptr<DihedralData> improper_data = m_sysdef->getImproperData();

        // loop over all angles and write them out
        for (unsigned int i = 0; i < improper_data->getNumDihedrals(); i++)
            {
            Dihedral dihedral = improper_data->getDihedral(i);
            f << improper_data->getNameByType(dihedral.type) << " " << dihedral.a  << " " << dihedral.b << " "
            << dihedral.c << " " << dihedral.d << "\n";
            }

        f << "</improper>" << "\n";
        }

    // if the wall flag is true, output the walls to the xml file
    if (m_output_wall)
        {
        f << "<wall>" << "\n";
        boost::shared_ptr<WallData> wall_data = m_sysdef->getWallData();

        // loop over all walls and write them out
        for (unsigned int i = 0; i < wall_data->getNumWalls(); i++)
            {
            Wall wall = wall_data->getWall(i);
            f << "<coord ox=\"" << wall.origin_x << "\" oy=\"" << wall.origin_y << "\" oz=\"" << wall.origin_z <<
            "\" nx=\"" << wall.normal_x << "\" ny=\"" << wall.normal_y << "\" nz=\"" << wall.normal_z << "\" />" << "\n";
            }
        f << "</wall>" << "\n";
        }

    // If the charge flag is true output the mass of all particles to the file
    if (m_output_charge)
        {
        f <<"<charge num=\"" << m_pdata->getNGlobal() << "\">" << "\n";

        for (unsigned int j = 0; j < m_pdata->getNGlobal(); j++)
            {
            Scalar charge = snapshot.charge[j];
            f << charge << "\n";
            if (!f.good())
                {
                m_exec_conf->msg->error() << "dump.xml: I/O error while writing HOOMD dump file" << endl;
                throw runtime_error("Error writting HOOMD dump file");
                }
            }

        f <<"</charge>" << "\n";
        }

    // if the orientation flag is set, write out the orientation quaternion to the XML file
    if (m_output_orientation)
        {
        f << "<orientation num=\"" << m_pdata->getNGlobal() << "\">" << "\n";

        for (unsigned int j = 0; j < m_pdata->getNGlobal(); j++)
            {
            // use the rtag data to output the particles in the order they were read in
            Scalar4 orientation = snapshot.orientation[j];
            f << orientation.x << " " << orientation.y << " " << orientation.z << " " << orientation.w << "\n";
            if (!f.good())
                {
                m_exec_conf->msg->error() << "dump.xml: I/O error while writing HOOMD dump file" << endl;
                throw runtime_error("Error writting HOOMD dump file");
                }
            }
        f << "</orientation>" << "\n";
        }

    // if the moment_inertia flag is set, write out the orientation quaternion to the XML file
    if (m_output_moment_inertia)
        {
        f << "<moment_inertia num=\"" << m_pdata->getNGlobal() << "\">" << "\n";

        for (unsigned int i = 0; i < m_pdata->getNGlobal(); i++)
            {
            // inertia tensors are stored by tag
            InertiaTensor I = snapshot.inertia_tensor[i];
            for (unsigned int c = 0; c < 5; c++)
                f << I.components[c] << " ";
            f << I.components[5] << "\n";

            if (!f.good())
                {
                m_exec_conf->msg->error() << "dump.xml: I/O error while writing HOOMD dump file" << endl;
                throw runtime_error("Error writting HOOMD dump file");
                }
            }
        f << "</moment_inertia>" << "\n";
        }

    f << "</configuration>" << "\n";
    f << "</hoomd_xml>" << "\n";

    if (!f.good())
        {
                m_exec_conf->msg->error() << "dump.xml: I/O error while writing HOOMD dump file" << endl;
        throw runtime_error("Error writting HOOMD dump file");
        }

    f.close();

    }

/*! \param timestep Current time step of the simulation
    Writes a snapshot of the current state of the ParticleData to a hoomd_xml file.
*/
void HOOMDDumpWriter::analyze(unsigned int timestep)
    {
    if (m_prof)
        m_prof->push("Dump XML");

    ostringstream full_fname;
    string filetype = ".xml";

    // Generate a filename with the timestep padded to ten zeros
    full_fname << m_base_fname << "." << setfill('0') << setw(10) << timestep << filetype;
    writeFile(full_fname.str(), timestep);

    if (m_prof)
        m_prof->pop();
    }

void export_HOOMDDumpWriter()
    {
    class_<HOOMDDumpWriter, boost::shared_ptr<HOOMDDumpWriter>, bases<Analyzer>, boost::noncopyable>
    ("HOOMDDumpWriter", init< boost::shared_ptr<SystemDefinition>, std::string >())
    .def("setOutputPosition", &HOOMDDumpWriter::setOutputPosition)
    .def("setOutputImage", &HOOMDDumpWriter::setOutputImage)
    .def("setOutputVelocity", &HOOMDDumpWriter::setOutputVelocity)
    .def("setOutputMass", &HOOMDDumpWriter::setOutputMass)
    .def("setOutputDiameter", &HOOMDDumpWriter::setOutputDiameter)
    .def("setOutputType", &HOOMDDumpWriter::setOutputType)
    .def("setOutputBody", &HOOMDDumpWriter::setOutputBody)
    .def("setOutputBond", &HOOMDDumpWriter::setOutputBond)
    .def("setOutputAngle", &HOOMDDumpWriter::setOutputAngle)
    .def("setOutputDihedral", &HOOMDDumpWriter::setOutputDihedral)
    .def("setOutputImproper", &HOOMDDumpWriter::setOutputImproper)
    .def("setOutputWall", &HOOMDDumpWriter::setOutputWall)
    .def("setOutputAccel", &HOOMDDumpWriter::setOutputAccel)
    .def("setOutputCharge", &HOOMDDumpWriter::setOutputCharge)
    .def("setOutputOrientation", &HOOMDDumpWriter::setOutputOrientation)
    .def("setVizSigma", &HOOMDDumpWriter::setVizSigma)
    .def("writeFile", &HOOMDDumpWriter::writeFile)
    ;
    }

#ifdef WIN32
#pragma warning( pop )
#endif
