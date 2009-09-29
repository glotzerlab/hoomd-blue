/*
Highly Optimized Object-Oriented Molecular Dynamics (HOOMD) Open
Source Software License
Copyright (c) 2008 Ames Laboratory Iowa State University
All rights reserved.

Redistribution and use of HOOMD, in source and binary forms, with or
without modification, are permitted, provided that the following
conditions are met:

* Redistributions of source code must retain the above copyright notice,
this list of conditions and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright
notice, this list of conditions and the following disclaimer in the
documentation and/or other materials provided with the distribution.

* Neither the name of the copyright holder nor the names HOOMD's
contributors may be used to endorse or promote products derived from this
software without specific prior written permission.

Disclaimer

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDER AND
CONTRIBUTORS ``AS IS''  AND ANY EXPRESS OR IMPLIED WARRANTIES,
INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY
AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED.

IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS  BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF
THE POSSIBILITY OF SUCH DAMAGE.
*/

// $Id$
// $URL$
// Maintainer: joaander

/*! \file PDBDumpWriter.cc
    \brief Defines the PDBDumpWriter class
*/

// this file was written from scratch, but some error checks and PDB snprintf strings were copied from VMD's molfile plugin
// it is used under the following license

// University of Illinois Open Source License
// Copyright 2003 Theoretical and Computational Biophysics Group,
// All rights reserved.

// Developed by:       Theoretical and Computational Biophysics Group
//             University of Illinois at Urbana-Champaign
//            http://www.ks.uiuc.edu/

#ifdef WIN32
#pragma warning( push )
#pragma warning( disable : 4244 )
#endif

#include <boost/python.hpp>
using namespace boost::python;

#include <string>
#include <fstream>
#include <iomanip>
#include <stdio.h>

#include "PDBDumpWriter.h"
#include "BondData.h"

using namespace std;
using namespace boost;

/*! \param sysdef System definition containing particle data to write
    \param base_fname Base filename to expand with **timestep**.pdb when writing
*/
PDBDumpWriter::PDBDumpWriter(boost::shared_ptr<SystemDefinition> sysdef, std::string base_fname)
        : Analyzer(sysdef), m_base_fname(base_fname), m_output_bond(false)
    {
    }

/*! \param timestep Current time step of the simulation

    analzye() constructs af file name m_base_fname.**timestep**.pdb and writes out the
    current state of the system to that file.
*/
void PDBDumpWriter::analyze(unsigned int timestep)
    {
    ostringstream full_fname;
    string filetype = ".pdb";
    
    // Generate a filename with the timestep padded to ten zeros
    full_fname << m_base_fname << "." << setfill('0') << setw(10) << timestep << filetype;
    // then write the file
    writeFile(full_fname.str());
    }

/*! \param fname File name to write

    Writes the current state of the system to the pdb file \a fname
*/
void PDBDumpWriter::writeFile(std::string fname)
    {
    // open the file for writing
    ofstream f(fname.c_str());
    f.exceptions ( ifstream::eofbit | ifstream::failbit | ifstream::badbit );
    
    if (!f.good())
        {
        cerr << endl << "***Error! Unable to open dump file for writing: " << fname << endl << endl;
        throw runtime_error("Error writting pdb dump file");
        }
        
    // acquire the particle data
    ParticleDataArraysConst arrays = m_pdata->acquireReadOnly();
    
    // get the box dimensions
    Scalar Lx,Ly,Lz;
    BoxDim box = m_pdata->getBox();
    Lx=Scalar(box.xhi-box.xlo);
    Ly=Scalar(box.yhi-box.ylo);
    Lz=Scalar(box.zhi-box.zlo);
    
    // start writing the heinous PDB format
    const int linesize = 82;
    char buf[linesize];
    
    // output the box dimensions
    snprintf(buf, linesize, "CRYST1%9.3f%9.3f%9.3f%7.2f%7.2f%7.2f P 1           1\n", Lx,Ly,Lz, 90.0, 90.0, 90.0);
    f << buf;
    
    // write out all the atoms
    for (unsigned int i = 0; i < m_pdata->getN(); i++)
        {
        // first check that everything will fit into the PDB output
        if (arrays.x[i] < -999.9994f || arrays.x[i] > 9999.9994f || arrays.y[i] < -999.9994f || arrays.y[i] > 9999.9994f || arrays.z[i] < -999.9994f || arrays.z[i] > 9999.9994f)
            {
            cerr << "***Error! Coordinate " << arrays.x[i] << " " << arrays.y[i] << " " << arrays.z[i] << " is out of range for PDB writing" << endl << endl;
            throw runtime_error("Error writing PDB file");
            }
        // check the length of the type name
        const string &type_name = m_pdata->getNameByType(arrays.type[i]);
        if (type_name.size() > 4)
            {
            cerr << "***Error! Type " << type_name << " is too long for PDB writing" << endl << endl;
            throw runtime_error("Error writing PDB file");
            }
            
        // start preparing the stuff to write (copied from VMD's molfile plugin)
        char indexbuf[32];
        char residbuf[32];
        char segnamebuf[5];
        char altlocchar;
        
        /* XXX                                                          */
        /* if the atom or residue indices exceed the legal PDB spec, we */
        /* start emitting asterisks or hexadecimal strings rather than  */
        /* aborting.  This is not really legal, but is an accepted hack */
        /* among various other programs that deal with large PDB files  */
        /* If we run out of hexadecimal indices, then we just print     */
        /* asterisks.                                                   */
        if (i < 100000)
            {
            sprintf(indexbuf, "%5d", i);
            }
        else if (i < 1048576)
            {
            sprintf(indexbuf, "%05x", i);
            }
        else
            {
            sprintf(indexbuf, "*****");
            }
            
        /*if (resid < 10000) {
        sprintf(residbuf, "%4d", resid);
        } else if (resid < 65536) {
        sprintf(residbuf, "%04x", resid);
        } else {
        sprintf(residbuf, "****");
        }*/
        sprintf(residbuf, "%4d", 1);
        
        //altlocchar = altloc[0];
        //if (altlocchar == '\0') {
        altlocchar = ' ';
        //}
        
        /* make sure the segname does not overflow the format */
        sprintf(segnamebuf, "SEG ");
        
        snprintf(buf, linesize, "%-6s%5s %4s%c%-4s%c%4s%c   %8.3f%8.3f%8.3f%6.2f%6.2f      %-4s%2s\n", "ATOM  ", indexbuf, type_name.c_str(), altlocchar, "RES", ' ',
                 residbuf, ' ', arrays.x[i], arrays.y[i], arrays.z[i], 0.0f, 0.0f, segnamebuf, "  ");
        f << buf;
        }
        
    if (m_output_bond)
        {
        // error check: pdb files cannot contain bonds with 100,000 or more atom records
        if (m_pdata->getN() >= 100000)
            {
            cerr << endl << "***Error! PDB files with bonds cannot hold more than 99,999 atoms!" << endl << endl;
            throw runtime_error("Error dumping PDB file");
            }
            
        // grab the bond data
        shared_ptr<BondData> bond_data = m_sysdef->getBondData();
        for (unsigned int i = 0; i < bond_data->getNumBonds(); i++)
            {
            Bond bond = bond_data->getBond(i);
            snprintf(buf, linesize, "CONECT%1$5d%2$5d\n", bond.a, bond.b);
            f << buf;
            }
        }
        
    // release the particle data
    m_pdata->release();
    }

void export_PDBDumpWriter()
    {
    class_<PDBDumpWriter, boost::shared_ptr<PDBDumpWriter>, bases<Analyzer>, boost::noncopyable>
    ("PDBDumpWriter", init< boost::shared_ptr<SystemDefinition>, std::string >())
    .def("setOutputBond", &PDBDumpWriter::setOutputBond)
    .def("writeFile", &PDBDumpWriter::writeFile)
    ;
    }

#ifdef WIN32
#pragma warning( pop )
#endif
