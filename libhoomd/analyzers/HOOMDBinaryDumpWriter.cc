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

/*! \file HOOMDBinaryDumpWriter.cc
    \brief Defines the HOOMDBinaryDumpWriter class
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

#include <boost/iostreams/device/file.hpp>
#include <boost/iostreams/filtering_stream.hpp>
#ifdef ENABLE_ZLIB
#include <boost/iostreams/filter/gzip.hpp>
#endif

#include "HOOMDBinaryDumpWriter.h"
#include "BondedGroupData.h"
#include "AngleData.h"
#include "DihedralData.h"
#include "WallData.h"

using namespace std;
using namespace boost;
using namespace boost::iostreams;

//! Helper function to write a string out to a file in binary mode
static void write_string(ostream &f, const string& str)
    {
    unsigned int len = (unsigned int)str.size();
    f.write((char*)&len, sizeof(unsigned int));
    if (len != 0)
        f.write(str.c_str(), len*sizeof(char));
    }

/*! \param sysdef SystemDefinition containing the ParticleData to dump
    \param base_fname The base name of the file xml file to output the information

    \note .timestep.xml will be apended to the end of \a base_fname when analyze() is called.
*/
HOOMDBinaryDumpWriter::HOOMDBinaryDumpWriter(boost::shared_ptr<SystemDefinition> sysdef, std::string base_fname)
        : Analyzer(sysdef), m_base_fname(base_fname), m_alternating(false), m_cur_file(1), m_enable_compression(false)
    {
    m_exec_conf->msg->notice(5) << "Constructing HOOMDBinaryDumpWriter: " << base_fname << endl;
    }

HOOMDBinaryDumpWriter::~HOOMDBinaryDumpWriter()
    {
    m_exec_conf->msg->notice(5) << "Destroying HOOMDBinaryDumpWriter" << endl;
    }

/*! \param fname File name to write
    \param timestep Current time step of the simulation
*/
void HOOMDBinaryDumpWriter::writeFile(std::string fname, unsigned int timestep)
    {
    // check the file extension and warn the user
    string ext = fname.substr(fname.size()-3, fname.size());
    bool gz_ext = false;
    if (ext == string(".gz"))
         gz_ext = true;

    if (!gz_ext && m_enable_compression)
        {
        m_exec_conf->msg->warning() << "dump.bin: Writing compressed binary file without a .gz extension." << endl;
        m_exec_conf->msg->warning() << "init.read_bin will not recognize that this file is compressed" << endl;
        }
    if (gz_ext && !m_enable_compression)
        {
        m_exec_conf->msg->warning() << "dump.bin: Writing uncompressed binary file with a .gz extension." << endl;
        m_exec_conf->msg->warning() << "init.read_bin will not recognize that this file is uncompressed" << endl;
        }

    // setup the file output for compression
    filtering_ostream f;
    #ifdef ENABLE_ZLIB
    if (m_enable_compression)
        f.push(gzip_compressor());
    #endif
    f.push(file_sink(fname.c_str(), ios::out | ios::binary));

    if (!f.good())
        {
        m_exec_conf->msg->error() << "dump.bin: Unable to open dump file for writing: " << fname << endl;
        throw runtime_error("Error writing hoomd binary dump file");
        }

    // write a magic number identifying the file format
    unsigned int magic = 0x444d4f48;
    f.write((char*)&magic, sizeof(unsigned int));
    // write the version of the binary format used
    int version = 3;
    f.write((char*)&version, sizeof(int));

    // acquire the particle data
    ArrayHandle<Scalar4> h_pos(m_pdata->getPositions(), access_location::host, access_mode::read);
    ArrayHandle<Scalar4> h_vel(m_pdata->getVelocities(), access_location::host, access_mode::read);
    ArrayHandle<Scalar3> h_accel(m_pdata->getAccelerations(), access_location::host, access_mode::read);
    ArrayHandle<int3> h_image(m_pdata->getImages(), access_location::host, access_mode::read);
    ArrayHandle<unsigned int> h_body(m_pdata->getBodies(), access_location::host, access_mode::read);
    ArrayHandle<unsigned int> h_tag(m_pdata->getTags(), access_location::host, access_mode::read);
    ArrayHandle<unsigned int> h_rtag(m_pdata->getRTags(), access_location::host, access_mode::read);
    ArrayHandle<Scalar> h_charge(m_pdata->getCharges(), access_location::host, access_mode::read);
    ArrayHandle<Scalar> h_diameter(m_pdata->getDiameters(), access_location::host, access_mode::read);

    BoxDim box = m_pdata->getBox();
    Scalar3 L = box.getL();
    unsigned int dimensions = m_sysdef->getNDimensions();

    //write out the timestep, dimensions, and box
    f.write((char*)&timestep, sizeof(unsigned int));
    f.write((char*)&dimensions, sizeof(unsigned int));
    f.write((char*)&L.x, sizeof(Scalar));
    f.write((char*)&L.y, sizeof(Scalar));
    f.write((char*)&L.z, sizeof(Scalar));

    //write out particle data
    unsigned int np = m_pdata->getN();
    f.write((char*)&np, sizeof(unsigned int));
    f.write((char*)h_tag.data, np*sizeof(unsigned int));
    f.write((char*)h_rtag.data, np*sizeof(unsigned int));
    for (unsigned int i = 0; i < np; i++)
       f.write((char*)&h_pos.data[i].x, sizeof(Scalar));
    for (unsigned int i = 0; i < np; i++)
       f.write((char*)&h_pos.data[i].y, sizeof(Scalar));
    for (unsigned int i = 0; i < np; i++)
       f.write((char*)&h_pos.data[i].z, sizeof(Scalar));
    for (unsigned int i = 0; i < np; i++)
       f.write((char*)&h_image.data[i].x, sizeof(int));
    for (unsigned int i = 0; i < np; i++)
       f.write((char*)&h_image.data[i].y, sizeof(int));
    for (unsigned int i = 0; i < np; i++)
       f.write((char*)&h_image.data[i].z, sizeof(int));
    for (unsigned int i = 0; i < np; i++)
       f.write((char*)&h_vel.data[i].x, sizeof(Scalar));
    for (unsigned int i = 0; i < np; i++)
       f.write((char*)&h_vel.data[i].y, sizeof(Scalar));
    for (unsigned int i = 0; i < np; i++)
       f.write((char*)&h_vel.data[i].z, sizeof(Scalar));
    for (unsigned int i = 0; i < np; i++)
       f.write((char*)&h_accel.data[i].x, sizeof(Scalar));
    for (unsigned int i = 0; i < np; i++)
       f.write((char*)&h_accel.data[i].y, sizeof(Scalar));
    for (unsigned int i = 0; i < np; i++)
       f.write((char*)&h_accel.data[i].z, sizeof(Scalar));
    for (unsigned int i = 0; i < np; i++)
       f.write((char*)&h_vel.data[i].w, sizeof(Scalar));
    f.write((char*)h_diameter.data, np*sizeof(Scalar));
    f.write((char*)h_charge.data, np*sizeof(Scalar));
    f.write((char*)h_body.data, np*sizeof(unsigned int));

    //write out types and type mapping
    unsigned int ntypes = m_pdata->getNTypes();
    f.write((char*)&ntypes, sizeof(unsigned int));
    for (unsigned int i = 0; i < ntypes; i++)
        {
        std::string name = m_pdata->getNameByType(i);
        write_string(f, name);
        }
    for (unsigned int i = 0; i < np; i++)
       f.write((char*)&h_pos.data[i].w, sizeof(unsigned int));

    if (!f.good())
        {
        m_exec_conf->msg->error() << "dump.bin: I/O error writing HOOMD dump file" << endl;
        throw runtime_error("Error writing HOOMD dump file");
        }

    //Output the integrator states to the binary file
    {
    boost::shared_ptr<IntegratorData> integrator_data = m_sysdef->getIntegratorData();
    unsigned int ni = integrator_data->getNumIntegrators();
    f.write((char*)&ni, sizeof(unsigned int));
    for (unsigned int j = 0; j < ni; j++)
        {
        IntegratorVariables v = integrator_data->getIntegratorVariables(j);
        write_string(f, v.type);

        unsigned int nv = (unsigned int)v.variable.size();
        f.write((char*)&nv, sizeof(unsigned int));
        for (unsigned int k=0; k<nv; k++)
            {
            Scalar var = v.variable[k];
            f.write((char*)&var, sizeof(Scalar));
            }
        }
    }

    // Output the bonds to the binary file
    {
    //write out type mapping
    ntypes = m_sysdef->getBondData()->getNTypes();
    f.write((char*)&ntypes, sizeof(unsigned int));
    for (unsigned int i = 0; i < ntypes; i++)
        {
        std::string name = m_sysdef->getBondData()->getNameByType(i);
        write_string(f, name);
        }

    unsigned int nb = m_sysdef->getBondData()->getN();
    f.write((char*)&nb, sizeof(unsigned int));
    boost::shared_ptr<BondData> bond_data = m_sysdef->getBondData();

    // loop over all bonds and write them out
    for (unsigned int i = 0; i < bond_data->getN(); i++)
        {
        BondData::members_t bond = bond_data->getMembersByIndex(i);
        unsigned int type = bond_data->getTypeByIndex(i);
        f.write((char*)&type, sizeof(unsigned int));
        f.write((char*)&bond.tag[0], sizeof(unsigned int));
        f.write((char*)&bond.tag[1], sizeof(unsigned int));
        }
    }

    // Output the angles to the binary file
    {
    //write out type mapping
    ntypes = m_sysdef->getAngleData()->getNAngleTypes();
    f.write((char*)&ntypes, sizeof(unsigned int));
    for (unsigned int i = 0; i < ntypes; i++)
        {
        std::string name = m_sysdef->getAngleData()->getNameByType(i);
        write_string(f, name);
        }

    unsigned int na = m_sysdef->getAngleData()->getNumAngles();
    f.write((char*)&na, sizeof(unsigned int));

    boost::shared_ptr<AngleData> angle_data = m_sysdef->getAngleData();

    // loop over all angles and write them out
    for (unsigned int i = 0; i < angle_data->getNumAngles(); i++)
        {
        Angle angle = angle_data->getAngle(i);

        f.write((char*)&angle.type, sizeof(unsigned int));
        f.write((char*)&angle.a, sizeof(unsigned int));
        f.write((char*)&angle.b, sizeof(unsigned int));
        f.write((char*)&angle.c, sizeof(unsigned int));
        }
    }

    // Write out dihedrals to the binary file
    {
    //write out type mapping
    ntypes = m_sysdef->getDihedralData()->getNDihedralTypes();
    f.write((char*)&ntypes, sizeof(unsigned int));
    for (unsigned int i = 0; i < ntypes; i++)
        {
        std::string name = m_sysdef->getDihedralData()->getNameByType(i);
        write_string(f, name);
        }

    unsigned int nd = m_sysdef->getDihedralData()->getNumDihedrals();
    f.write((char*)&nd, sizeof(unsigned int));

    boost::shared_ptr<DihedralData> dihedral_data = m_sysdef->getDihedralData();

    // loop over all angles and write them out
    for (unsigned int i = 0; i < dihedral_data->getNumDihedrals(); i++)
        {
        Dihedral dihedral = dihedral_data->getDihedral(i);

        f.write((char*)&dihedral.type, sizeof(unsigned int));
        f.write((char*)&dihedral.a, sizeof(unsigned int));
        f.write((char*)&dihedral.b, sizeof(unsigned int));
        f.write((char*)&dihedral.c, sizeof(unsigned int));
        f.write((char*)&dihedral.d, sizeof(unsigned int));
        }
    }

    // Write out impropers to the binary file
    {
    ntypes = m_sysdef->getImproperData()->getNDihedralTypes();
    f.write((char*)&ntypes, sizeof(unsigned int));
    for (unsigned int i = 0; i < ntypes; i++)
        {
        std::string name = m_sysdef->getImproperData()->getNameByType(i);
        write_string(f, name);
        }

    unsigned int ni = m_sysdef->getImproperData()->getNumDihedrals();
    f.write((char*)&ni, sizeof(unsigned int));

    boost::shared_ptr<DihedralData> improper_data = m_sysdef->getImproperData();

    // loop over all angles and write them out
    for (unsigned int i = 0; i < improper_data->getNumDihedrals(); i++)
        {
        Dihedral dihedral = improper_data->getDihedral(i);

        f.write((char*)&dihedral.type, sizeof(unsigned int));
        f.write((char*)&dihedral.a, sizeof(unsigned int));
        f.write((char*)&dihedral.b, sizeof(unsigned int));
        f.write((char*)&dihedral.c, sizeof(unsigned int));
        f.write((char*)&dihedral.d, sizeof(unsigned int));
        }
    }

    // Output the walls to the binary file
    {
    boost::shared_ptr<WallData> wall_data = m_sysdef->getWallData();

    unsigned int nw = wall_data->getNumWalls();
    f.write((char*)&nw, sizeof(unsigned int));

    // loop over all walls and write them out
    for (unsigned int i = 0; i < nw; i++)
        {
        Wall wall = wall_data->getWall(i);

        f.write((char*)&(wall.origin_x), sizeof(Scalar));
        f.write((char*)&(wall.origin_y), sizeof(Scalar));
        f.write((char*)&(wall.origin_z), sizeof(Scalar));
        f.write((char*)&(wall.normal_x), sizeof(Scalar));
        f.write((char*)&(wall.normal_y), sizeof(Scalar));
        f.write((char*)&(wall.normal_z), sizeof(Scalar));
        }
    }

    // Output the rigid bodies to the binary file
    {
    boost::shared_ptr<RigidData> rigid_data = m_sysdef->getRigidData();

    unsigned int n_bodies = rigid_data->getNumBodies();
    f.write((char*)&n_bodies, sizeof(unsigned int));

    if (n_bodies <= 0)
        {
        return;
        }

    // We don't need to write forces, torques and orientation/quaternions because as the rigid bodies are constructed
    // from restart files, the orientation is recalculated for the moment of inertia- using the old one will cause mismatches in angular velocities.
    // Below are the minimal data required for a smooth restart with rigid bodies, assuming that RigidData::initializeData() already invoked.

    ArrayHandle<Scalar4> com_handle(rigid_data->getCOM(), access_location::host, access_mode::read);
    ArrayHandle<Scalar4> vel_handle(rigid_data->getVel(), access_location::host, access_mode::read);
    ArrayHandle<Scalar4> angmom_handle(rigid_data->getAngMom(), access_location::host, access_mode::read);
    ArrayHandle<int3> body_image_handle(rigid_data->getBodyImage(), access_location::host, access_mode::read);

    for (unsigned int body = 0; body < n_bodies; body++)
        {
        f.write((char*)&(com_handle.data[body].x), sizeof(Scalar));
        f.write((char*)&(com_handle.data[body].y), sizeof(Scalar));
        f.write((char*)&(com_handle.data[body].z), sizeof(Scalar));
        f.write((char*)&(com_handle.data[body].w), sizeof(Scalar));

        f.write((char*)&(vel_handle.data[body].x), sizeof(Scalar));
        f.write((char*)&(vel_handle.data[body].y), sizeof(Scalar));
        f.write((char*)&(vel_handle.data[body].z), sizeof(Scalar));
        f.write((char*)&(vel_handle.data[body].w), sizeof(Scalar));

        f.write((char*)&(angmom_handle.data[body].x), sizeof(Scalar));
        f.write((char*)&(angmom_handle.data[body].y), sizeof(Scalar));
        f.write((char*)&(angmom_handle.data[body].z), sizeof(Scalar));
        f.write((char*)&(angmom_handle.data[body].w), sizeof(Scalar));

        f.write((char*)&(body_image_handle.data[body].x), sizeof(int));
        f.write((char*)&(body_image_handle.data[body].y), sizeof(int));
        f.write((char*)&(body_image_handle.data[body].z), sizeof(int));

        }
    }

    if (!f.good())
        {
        m_exec_conf->msg->error() << "dump.bin: I/O error writing HOOMD dump file" << endl;
        throw runtime_error("Error writing HOOMD dump file");
        }

    }

/*! \param timestep Current time step of the simulation
    Writes a snapshot of the current state of the ParticleData to a hoomd_xml file.
*/
void HOOMDBinaryDumpWriter::analyze(unsigned int timestep)
    {
    if (m_prof)
        m_prof->push("Dump BIN");

    if (!m_alternating)
        {
        ostringstream full_fname;
        string filetype = ".bin";
        if (m_enable_compression)
            filetype += ".gz";

        // Generate a filename with the timestep padded to ten zeros
        full_fname << m_base_fname << "." << setfill('0') << setw(10) << timestep << filetype;
        writeFile(full_fname.str(), timestep);
        }
    else
        {
        // write out to m_fname1 and m_fname2, alternating between the two
        string fname;
        if (m_cur_file == 1)
            {
            fname = m_fname1;
            m_cur_file = 2;
            }
        else
            {
            fname = m_fname2;
            m_cur_file = 1;
            }
        writeFile(fname, timestep);
        }

    if (m_prof)
        m_prof->pop();
    }

/*! \param fname1 File name of the first file to write
    \param fname2 File nmae of the second file to write
*/
void HOOMDBinaryDumpWriter::setAlternatingWrites(const std::string& fname1, const std::string& fname2)
    {
    m_alternating = true;
    m_fname1 = fname1;
    m_fname2 = fname2;
    }

/* \param enable_compression Set to true to enable compression, falst to disable it
*/
void HOOMDBinaryDumpWriter::enableCompression(bool enable_compression)
    {
    #ifdef ENABLE_ZLIB
    m_enable_compression = enable_compression;
    #else
    m_enable_compression = false;
    if (enable_compression)
        {
        m_exec_conf->msg->warning() << "dump.bin: This build of hoomd was compiled with ENABLE_ZLIB=off.";
        m_exec_conf->msg->warning() << "binary data output will NOT be compressed" << endl;
        }
    #endif
    }

void export_HOOMDBinaryDumpWriter()
    {
    class_<HOOMDBinaryDumpWriter, boost::shared_ptr<HOOMDBinaryDumpWriter>, bases<Analyzer>, boost::noncopyable>
    ("HOOMDBinaryDumpWriter", init< boost::shared_ptr<SystemDefinition>, std::string >())
    .def("writeFile", &HOOMDBinaryDumpWriter::writeFile)
    .def("setAlternatingWrites", &HOOMDBinaryDumpWriter::setAlternatingWrites)
    .def("enableCompression", &HOOMDBinaryDumpWriter::enableCompression)
    ;
    }

#ifdef WIN32
#pragma warning( pop )
#endif
