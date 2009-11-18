/*
Highly Optimized Object-oriented Many-particle Dynamics -- Blue Edition
(HOOMD-blue) Open Source Software License Copyright 2008, 2009 Ames Laboratory
Iowa State University and The Regents of the University of Michigan All rights
reserved.

HOOMD-blue may contain modifications ("Contributions") provided, and to which
copyright is held, by various Contributors who have granted The Regents of the
University of Michigan the right to modify and/or distribute such Contributions.

Redistribution and use of HOOMD-blue, in source and binary forms, with or
without modification, are permitted, provided that the following conditions are
met:

* Redistributions of source code must retain the above copyright notice, this
list of conditions, and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright notice, this
list of conditions, and the following disclaimer in the documentation and/or
other materials provided with the distribution.

* Neither the name of the copyright holder nor the names of HOOMD-blue's
contributors may be used to endorse or promote products derived from this
software without specific prior written permission.

Disclaimer

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDER AND CONTRIBUTORS ``AS IS''
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, AND/OR
ANY WARRANTIES THAT THIS SOFTWARE IS FREE OF INFRINGEMENT ARE DISCLAIMED.

IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT,
INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE
OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF
ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

// $Id: HOOMDBinaryDumpWriter.cc 2213 2009-10-20 11:42:07Z joaander $
// $URL: https://codeblue.umich.edu/hoomd-blue/svn/trunk/src/analyzers/HOOMDBinaryDumpWriter.cc $
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
#include "BondData.h"
#include "AngleData.h"
#include "DihedralData.h"
#include "WallData.h"

using namespace std;
using namespace boost;
using namespace boost::iostreams;

//! Helper function to write a string out to a file in binary mode
static void write_string(ostream &f, const string& str)
    {
    unsigned int len = str.size();
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
         
    if (gz_ext && !m_enable_compression)
        {
        cout << endl << "***Warning! Writing compressed binary file without a .gz extension.";
        cout << "init.read_bin will not recognize that this file is compressed" << endl << endl;
        }
    if (!gz_ext && m_enable_compression)
        {
        cout << endl << "***Warning! Writing uncompressed binary file with a .gz extension.";
        cout << "init.read_bin will not recognize that this file is uncompressed" << endl << endl;
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
        cerr << endl << "***Error! Unable to open dump file for writing: " << fname << endl << endl;
        throw runtime_error("Error writing hoomd binary dump file");
        }
    
    // write a magic number identifying the file format
    unsigned int magic = 0x444d4f48;
    f.write((char*)&magic, sizeof(unsigned int));
    // write the version of the binary format used
    int version = 1;
    f.write((char*)&version, sizeof(int));

    // acquire the particle data
    ParticleDataArraysConst arrays = m_pdata->acquireReadOnly();
    BoxDim box = m_pdata->getBox();
    Scalar Lx,Ly,Lz;
    Lx=Scalar(box.xhi-box.xlo);
    Ly=Scalar(box.yhi-box.ylo);
    Lz=Scalar(box.zhi-box.zlo);
        
    //write out the timestep and box
    f.write((char*)&timestep, sizeof(int));
    f.write((char*)&Lx, sizeof(Scalar));
    f.write((char*)&Ly, sizeof(Scalar));
    f.write((char*)&Lz, sizeof(Scalar));
    
    //write out particle data
    unsigned int np = m_pdata->getN();
    f.write((char*)&np, sizeof(unsigned int));
    f.write((char*)arrays.tag, np*sizeof(unsigned int));
    f.write((char*)arrays.rtag, np*sizeof(unsigned int));
    f.write((char*)arrays.x, np*sizeof(Scalar));
    f.write((char*)arrays.y, np*sizeof(Scalar));
    f.write((char*)arrays.z, np*sizeof(Scalar));
    f.write((char*)arrays.ix, np*sizeof(int));
    f.write((char*)arrays.iy, np*sizeof(int));
    f.write((char*)arrays.iz, np*sizeof(int));
    f.write((char*)arrays.vx, np*sizeof(Scalar));
    f.write((char*)arrays.vy, np*sizeof(Scalar));
    f.write((char*)arrays.vz, np*sizeof(Scalar));
    f.write((char*)arrays.ax, np*sizeof(Scalar));
    f.write((char*)arrays.ay, np*sizeof(Scalar));
    f.write((char*)arrays.az, np*sizeof(Scalar));
    f.write((char*)arrays.mass, np*sizeof(Scalar));
    f.write((char*)arrays.diameter, np*sizeof(Scalar));
    f.write((char*)arrays.charge, np*sizeof(Scalar));

    //write out types and type mapping
    unsigned int ntypes = m_pdata->getNTypes();
    f.write((char*)&ntypes, sizeof(unsigned int));
    for (unsigned int i = 0; i < ntypes; i++)
        {
        std::string name = m_pdata->getNameByType(i);
        write_string(f, name);
        }
    f.write((char*)arrays.type, np*sizeof(unsigned int));

    if (!f.good())
        {
        cerr << endl << "***Error! Unexpected error writing HOOMD dump file" << endl << endl;
        throw runtime_error("Error writing HOOMD dump file");
        }
    
    //Output the integrator states to the binary file
    {
    shared_ptr<IntegratorData> integrator_data = m_sysdef->getIntegratorData();
    unsigned int ni = integrator_data->getNumIntegrators();
    f.write((char*)&ni, sizeof(unsigned int));
    for (unsigned int j = 0; j < ni; j++)
        {
        IntegratorVariables v = integrator_data->getIntegratorVariables(j);
        write_string(f, v.type);

        unsigned int nv = v.variable.size();
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
    ntypes = m_sysdef->getBondData()->getNBondTypes();
    f.write((char*)&ntypes, sizeof(unsigned int));
    for (unsigned int i = 0; i < ntypes; i++)
        {
        std::string name = m_sysdef->getBondData()->getNameByType(i);
        write_string(f, name);
        }

    unsigned int nb = m_sysdef->getBondData()->getNumBonds();
    f.write((char*)&nb, sizeof(unsigned int));
    shared_ptr<BondData> bond_data = m_sysdef->getBondData();
    
    // loop over all bonds and write them out
    for (unsigned int i = 0; i < bond_data->getNumBonds(); i++)
        {
        Bond bond = bond_data->getBond(i);
        f.write((char*)&bond.type, sizeof(unsigned int));
        f.write((char*)&bond.a, sizeof(unsigned int));
        f.write((char*)&bond.b, sizeof(unsigned int));
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

    shared_ptr<AngleData> angle_data = m_sysdef->getAngleData();
    
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

    shared_ptr<DihedralData> dihedral_data = m_sysdef->getDihedralData();
    
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
            
    shared_ptr<DihedralData> improper_data = m_sysdef->getImproperData();
    
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
    shared_ptr<WallData> wall_data = m_sysdef->getWallData();

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
        
    if (!f.good())
        {
        cerr << endl << "***Error! Unexpected error writing HOOMD dump file" << endl << endl;
        throw runtime_error("Error writing HOOMD dump file");
        }
        
    m_pdata->release();
    
    }

/*! \param timestep Current time step of the simulation
    Writes a snapshot of the current state of the ParticleData to a hoomd_xml file.
*/
void HOOMDBinaryDumpWriter::analyze(unsigned int timestep)
    {
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
        cout << endl << "***Warning! This build of hoomd was compiled with ENABLE_ZLIB=off.";
        cout << "binary data output will NOT be compressed" << endl << endl;
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

