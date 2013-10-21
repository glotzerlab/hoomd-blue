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
    m_exec_conf->msg->notice(5) << "Constructing PDBDumpWriter: " << base_fname << endl;
    }

PDBDumpWriter::~PDBDumpWriter()
    {
    m_exec_conf->msg->notice(5) << "Destroying PDBDumpWriter" << endl;
    }

/*! \param timestep Current time step of the simulation

    analzye() constructs af file name m_base_fname.**timestep**.pdb and writes out the
    current state of the system to that file.
*/
void PDBDumpWriter::analyze(unsigned int timestep)
    {
    if (m_prof)
        m_prof->push("Dump PDB");

    ostringstream full_fname;
    string filetype = ".pdb";

    // Generate a filename with the timestep padded to ten zeros
    full_fname << m_base_fname << "." << setfill('0') << setw(10) << timestep << filetype;
    // then write the file
    writeFile(full_fname.str());

    if (m_prof)
        m_prof->pop();
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
        m_exec_conf->msg->error() << "dump.pdb: Unable to open file for writing: " << fname << endl;
        throw runtime_error("Error writting pdb dump file");
        }

    // acquire the particle data
    ArrayHandle<Scalar4> h_pos(m_pdata->getPositions(), access_location::host, access_mode::read);
    ArrayHandle<unsigned int> h_rtag(m_pdata->getRTags(), access_location::host, access_mode::read);


    // get the box dimensions
    BoxDim box = m_pdata->getBox();

    // start writing the heinous PDB format
    const int linesize = 82;
    char buf[linesize];

    // output the box dimensions
    Scalar a,b,c,alpha,beta,gamma;
    Scalar3 va = box.getLatticeVector(0);
    Scalar3 vb = box.getLatticeVector(1);
    Scalar3 vc = box.getLatticeVector(2);
    a = sqrt(dot(va,va));
    b = sqrt(dot(vb,vb));
    c = sqrt(dot(vc,vc));
    alpha = 90.0 - asin(dot(vb,vc)/(b*c)) * 90.0 / M_PI_2;
    beta = 90.0 - asin(dot(va,vc)/(a*c)) * 90.0 / M_PI_2;
    gamma = 90.0 - asin(dot(va,vb)/(a*b)) * 90.0 / M_PI_2;

    snprintf(buf, linesize, "CRYST1%9.3f%9.3f%9.3f%7.2f%7.2f%7.2f P 1           1\n", a,b,c, alpha, beta, gamma);
    f << buf;

    // write out all the atoms
    for (unsigned int j = 0; j < m_pdata->getN(); j++)
        {
        int i;
        i= h_rtag.data[j];

        // first check that everything will fit into the PDB output
        if (h_pos.data[i].x < -999.9994f || h_pos.data[i].x > 9999.9994f || h_pos.data[i].y < -999.9994f || h_pos.data[i].y > 9999.9994f || h_pos.data[i].z < -999.9994f || h_pos.data[i].z > 9999.9994f)
            {
            m_exec_conf->msg->error() << "dump.pdb: Coordinate " << h_pos.data[i].x << " " << h_pos.data[i].y << " " << h_pos.data[i].z << " is out of range for PDB writing" << endl;
            throw runtime_error("Error writing PDB file");
            }
        // check the length of the type name
        const string &type_name = m_pdata->getNameByType(__scalar_as_int(h_pos.data[i].w));
        if (type_name.size() > 4)
            {
            m_exec_conf->msg->error() << "dump.pdb: Type " << type_name << " is too long for PDB writing" << endl;
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
                 residbuf, ' ', h_pos.data[i].x, h_pos.data[i].y, h_pos.data[i].z, 0.0f, 0.0f, segnamebuf, "  ");
        f << buf;
        }

    if (m_output_bond)
        {
        // error check: pdb files cannot contain bonds with 100,000 or more atom records
        if (m_pdata->getN() >= 100000)
            {
            m_exec_conf->msg->error() << "dump.pdb: PDB files with bonds cannot hold more than 99,999 atoms!" << endl;
            throw runtime_error("Error dumping PDB file");
            }

        // grab the bond data
        boost::shared_ptr<BondData> bond_data = m_sysdef->getBondData();
        for (unsigned int i = 0; i < bond_data->getNumBonds(); i++)
            {
            Bond bond = bond_data->getBond(i);
            snprintf(buf, linesize, "CONECT%1$5d%2$5d\n", bond.a, bond.b);
            f << buf;
            }
        }

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
