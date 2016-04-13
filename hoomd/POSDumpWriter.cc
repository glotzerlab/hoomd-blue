/*
Highly Optimized Object-oriented Many-particle Dynamics -- Blue Edition
(HOOMD-blue) Open Source Software License Copyright 2009-2016 The Regents of
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

// Maintainer: harperic

/*! \file POSDumpWriter.cc
    \brief Defines the POSDumpWriter class
*/

#include "POSDumpWriter.h"

#include <boost/python.hpp>
using namespace boost::python;

#include <iomanip>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <boost/shared_ptr.hpp>

using namespace std;

/*! \param sysdef SystemDefinition containing the ParticleData to dump
    \param fname_base The base file name to write the output to
*/
POSDumpWriter::POSDumpWriter(boost::shared_ptr<SystemDefinition> sysdef, std::string fname)
        : Analyzer(sysdef), m_unwrap_rigid(false), m_write_info(false)
    {
    bool is_root = true;

    Scalar mdiam = m_pdata->getMaxDiameter();

    #ifdef ENABLE_MPI
    is_root = (m_exec_conf->getRank() == 0);
    #endif

    if (!is_root) return;

    m_file.open(fname.c_str());

    // default all shapes to spheres
    m_defs.resize(m_pdata->getNTypes());
    for (unsigned int j = 0; j < m_pdata->getNTypes(); j++)
        {
        ostringstream s;
        s << "sphere " << mdiam << " ffff0000";
        m_defs[j] = s.str();
        }
    }

/*! \param tid Type id of the def to set
    \param def Definition string to pass into the pos file
*/
void POSDumpWriter::setDef(unsigned int tid, std::string def)
    {
    #ifdef ENABLE_MPI
    // only set variable on rank 0
    if (m_exec_conf->getRank()) return;
    #endif

    if (tid >= m_pdata->getNTypes())
        {
        m_exec_conf->msg->error() << "Invalid type specified" << endl << endl;
        throw runtime_error("Error setting def string");
        }

    m_defs[tid] = def;
    }

/*! \param timestep Current time step of the simulation
    Writes a snapshot of the current state of the ParticleData to a POS file
*/
void POSDumpWriter::analyze(unsigned int timestep)
    {
    if (m_prof)
        m_prof->push("Dump pos");

    string info;
    // if there is a string to be written due to the python method addInfo, write it.
    if (m_write_info)
        {
        info = boost::python::extract<string> (m_add_info(timestep));
        }

    SnapshotParticleData<Scalar> snap(m_pdata->getNGlobal());

    // obtain particle data
    m_pdata->takeSnapshot(snap);

    bool is_root = true;

    #ifdef ENABLE_MPI
    is_root = (m_exec_conf->getRank() == 0);
    #endif

    if (!is_root)
        {
        if (m_prof) m_prof->pop();
        return;
        }

    if (!m_file.good())
        {
        m_exec_conf->msg->error() << "Unable to open dump file for writing" << endl << endl;
        throw runtime_error("Error writing pos dump file");
        }

    // Get the box information
    BoxDim box = m_pdata->getGlobalBox();
    vec3<Scalar> a1(box.getLatticeVector(0));
    vec3<Scalar> a2(box.getLatticeVector(1));
    vec3<Scalar> a3(box.getLatticeVector(2));

    unsigned int numtype;
    numtype = snap.type_mapping.size();

    m_file.precision(13);
    m_file << "boxMatrix "  << a1.x << " " << a2.x << " " << a3.x << " "
                            << a1.y << " " << a2.y << " " << a3.y << " "
                            << a1.z << " " << a2.z << " " << a3.z << "\n";

    for (unsigned int j = 0; j < numtype; j++)
        {
        string tname = snap.type_mapping[j];
        m_file << "def " << tname << " " << "\"" << m_defs[j] << "\"" << "\n";
        }

    // if there is a string to be written due to the python method addInfo, write it.
    if (m_write_info)
        {
        string info = boost::python::extract<string> (m_add_info(timestep));
        m_file << info;
        }

    for (unsigned int j = 0; j < snap.size; j++)
        {
        // get the coordinates
        vec3<Scalar> pos = snap.pos[j];
        quat<Scalar> orientation = snap.orientation[j];

        vec3<Scalar> tmp_pos = pos;

        if (m_unwrap_rigid && snap.body[j] != NO_BODY)
            {
            unsigned int central_ptl_tag = snap.body[j];
            assert(central_ptl_tag < snap.size());
            int body_ix = snap.image[central_ptl_tag].x;
            int body_iy = snap.image[central_ptl_tag].y;
            int body_iz = snap.image[central_ptl_tag].z;
            int3 particle_img = snap.image[j];
            int3 img_diff = make_int3(particle_img.x - body_ix,
                                      particle_img.y - body_iy,
                                      particle_img.z - body_iz);

            tmp_pos = box.shift(tmp_pos, img_diff);
            }

        // get the type by name
        unsigned int type_id = snap.type[j];
        string type_name = snap.type_mapping[type_id];

        m_file << type_name << " " << tmp_pos.x << " " << tmp_pos.y << " " << tmp_pos.z;
        // output quaternion only if not sphere
        if (m_defs[type_id].compare(0,7,"sphere "))
            {
            m_file << " " << orientation.s << " " << orientation.v.x << " " << orientation.v.y << " " << orientation.v.z;
            }
        m_file << "\n";

        if (!m_file.good())
            {
            m_exec_conf->msg->error() << "Unexpected error writing pos dump file" << endl << endl;
            throw runtime_error("Error writting pos dump file");
            }
        }

    if (!m_file.good())
        {
        m_exec_conf->msg->error() << "Unexpected error writing pos dump file" << endl << endl;
        throw runtime_error("Error writting pos dump file");
        }

    m_file << "eof" << endl;

    if (m_prof)
        m_prof->pop();
    }

void export_POSDumpWriter()
    {
    class_<POSDumpWriter, boost::shared_ptr<POSDumpWriter>, bases<Analyzer>, boost::noncopyable>
    ("POSDumpWriter", init< boost::shared_ptr<SystemDefinition>, std::string >())
        .def("setDef", &POSDumpWriter::setDef)
        .def("setUnwrapRigid", &POSDumpWriter::setUnwrapRigid)
        .def("setAddInfo", &POSDumpWriter::setAddInfo)
        ;
    }
