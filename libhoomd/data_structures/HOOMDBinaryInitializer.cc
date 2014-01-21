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

/*! \file HOOMDBinaryInitializer.cc
    \brief Defines the HOOMDBinaryInitializer class
*/

#ifdef WIN32
#pragma warning( push )
#pragma warning( disable : 4244 4267 )
#endif

#include "HOOMDBinaryInitializer.h"
#include "SnapshotSystemData.h"

#include <iostream>
#include <fstream>
#include <stdexcept>
#include <sstream>
#include <algorithm>

using namespace std;

#include <boost/python.hpp>
#include <boost/iostreams/device/file.hpp>
#include <boost/iostreams/filtering_stream.hpp>
#ifdef ENABLE_ZLIB
#include <boost/iostreams/filter/gzip.hpp>
#endif

using namespace boost::python;
using namespace boost;
using namespace boost::iostreams;

/*! \param ExecutionConfiguration
    \param fname File name with the data to load
    The file will be read and parsed fully during the constructor call.
*/
HOOMDBinaryInitializer::HOOMDBinaryInitializer(boost::shared_ptr<const ExecutionConfiguration> exec_conf,
                                               const std::string &fname)
    : m_exec_conf(exec_conf),
      m_timestep(0)
    {
    // execute only on rank zero
    if (m_exec_conf->getRank()) return;

    // initialize member variables
    m_num_dimensions = 3;
    // read in the file
    readFile(fname);
    }

/* XXX: shouldn't the following methods be put into
 * the header so that they get inlined? */

/*! \returns Time step parsed from the binary file
*/
unsigned int HOOMDBinaryInitializer::getTimeStep() const
    {
    return m_timestep;
    }

/* change internal timestep number. */
void HOOMDBinaryInitializer::setTimeStep(unsigned int ts)
    {
    m_timestep = ts;
    }

/*! initializes a snapshot with the internally stored copy of the particle data */
boost::shared_ptr<SnapshotSystemData> HOOMDBinaryInitializer::getSnapshot() const
    {
    boost::shared_ptr<SnapshotSystemData> snapshot(new SnapshotSystemData());

    // execute only on rank zero
    if (m_exec_conf->getRank()) return snapshot;

    // init dimensions
    snapshot->dimensions = m_num_dimensions;

    // init box
    snapshot->global_box = m_box;

    // init particle data snapshot
    SnapshotParticleData& pdata = snapshot->particle_data;

    // resize snapshot
    pdata.resize(m_x_array.size());

    // loop through all the particles and set them up
    for (unsigned int i = 0; i < pdata.size; i++)
        {
        unsigned int rtag = m_rtag_array[i];

        pdata.pos[i] = make_scalar3(m_x_array[rtag], m_y_array[rtag], m_z_array[rtag]);
        pdata.image[i] = make_int3(m_ix_array[rtag], m_iy_array[rtag], m_iz_array[rtag]);
        pdata.vel[i] = make_scalar3(m_vx_array[rtag], m_vy_array[rtag], m_vz_array[rtag]);
        pdata.accel[i] = make_scalar3(m_ax_array[rtag], m_ay_array[rtag], m_az_array[rtag]);
        pdata.mass[i] = m_mass_array[rtag];
        pdata.type[i] = m_type_array[rtag];
        pdata.diameter[i] = m_diameter_array[rtag];
        pdata.charge[i] = m_charge_array[rtag];
        pdata.body[i] = m_body_array[rtag];
        }

    pdata.type_mapping = m_type_mapping;

    /*
     * Initialize bond data
     */
    SnapshotBondData& bdata = snapshot->bond_data;

    // allocate memory in snapshot
    bdata.resize(m_bonds.size());

    // loop through all the bonds and add a bond for each
    for (unsigned int i = 0; i < m_bonds.size(); i++)
        {
        bdata.bonds[i] = make_uint2(m_bonds[i].a,m_bonds[i].b);
        bdata.type_id[i] = m_bonds[i].type;
        }

    bdata.type_mapping = m_bond_type_mapping;

    /*
     * Initialize angle data
     */
    SnapshotAngleData& adata = snapshot->angle_data;

    // allocate memory in snapshot
    adata.resize(m_angles.size());

    // loop through all the angles and add an angle for each
    for (unsigned int i = 0; i < m_angles.size(); i++)
        {
        adata.angles[i] = make_uint3(m_angles[i].a,m_angles[i].b,m_angles[i].c);
        adata.type_id[i] = m_angles[i].type;
        }

    adata.type_mapping = m_angle_type_mapping;

    /*
     * Initialize dihedral data
     */
    SnapshotDihedralData& ddata = snapshot->dihedral_data;

    // allocate memory
    ddata.resize(m_dihedrals.size());

    // loop through all the dihedrals and add an dihedral for each
    for (unsigned int i = 0; i < m_dihedrals.size(); i++)
        {
        ddata.dihedrals[i] = make_uint4(m_dihedrals[i].a,m_dihedrals[i].b,m_dihedrals[i].c, m_dihedrals[i].d);
        ddata.type_id[i] = m_dihedrals[i].type;
        }

    ddata.type_mapping = m_dihedral_type_mapping;

    /*
     * Initialize improper data
     */
    SnapshotDihedralData& idata = snapshot->improper_data;

    // allocate memory
    idata.resize(m_dihedrals.size());

    // loop through all the dihedrals and add an dihedral for each
    for (unsigned int i = 0; i < m_impropers.size(); i++)
        {
        idata.dihedrals[i] = make_uint4(m_impropers[i].a,m_impropers[i].b,m_impropers[i].c, m_impropers[i].d);
        idata.type_id[i] = m_impropers[i].type;
        }

    idata.type_mapping = m_improper_type_mapping;

    /*
     * Initialize walls
     */
    snapshot->wall_data = m_walls;

    /*
     * Initialize integrator data
     */
    snapshot->integrator_data = m_integrator_variables;

    /*
     * Initalize rigid body data
     */
    SnapshotRigidData& rdata = snapshot->rigid_data;

    unsigned int n_bodies = m_com.size();
    rdata.resize(n_bodies);

    for (unsigned int body = 0; body < n_bodies; body++)
        {
        rdata.com[body] = make_scalar3(m_com[body].x, m_com[body].y, m_com[body].z);
        rdata.vel[body] = make_scalar3(m_vel[body].x, m_vel[body].y, m_vel[body].z);
        rdata.angmom[body] = make_scalar3(m_angmom[body].x, m_angmom[body].y, m_angmom[body].z);
        rdata.body_image[body] = m_body_image[body];
        }

    return snapshot;
    }

//! Helper function to read a string from the file
static string read_string(istream &f)
    {
    unsigned int len;
    f.read((char*)&len, sizeof(unsigned int));
    if (len != 0)
        {
        char *cstr = new char[len+1];
        f.read(cstr, len*sizeof(char));
        cstr[len] = '\0';
        string str(cstr);
        delete[] cstr;
        return str;
        }
    else
        return string();
    }

/*! \param fname File name of the hoomd_binary file to read in
    \post Internal data arrays and members are filled out from which future calls
    like getSnapshot() will use to initialize the ParticleData

    This function implements the main parser loop. It reads in XML nodes from the
    file one by one and passes them of to parsers registered in \c m_parser_map.
*/
void HOOMDBinaryInitializer::readFile(const string &fname)
    {
    // check to see if the file has a .gz extension or not and enable decompression if it is
    bool enable_decompression = false;
    string ext = fname.substr(fname.size()-3, fname.size());
    if (ext == string(".gz"))
         enable_decompression = true;

    #ifndef ENABLE_ZLIB
    if (enable_decompression)
        {
        m_exec_conf->msg->error() << endl << "HOOMDBinaryInitialzier is trying to read a compressed .gz file, but ZLIB was not" << endl
            << "enabled in this build of hoomd" << endl << endl;
        throw runtime_error("Error reading binary file");
        }
    #endif

    // Open the file
    m_exec_conf->msg->notice(2) << "Reading " << fname << "..." << endl;
    // setup the file input for decompression
    filtering_istream f;
    #ifdef ENABLE_ZLIB
    if (enable_decompression)
        f.push(gzip_decompressor());
    #endif
    f.push(file_source(fname.c_str(), ios::in | ios::binary));

    // handle errors
    if (f.fail())
        {
        m_exec_conf->msg->error() << endl << "Error opening " << fname << endl << endl;
        throw runtime_error("Error reading binary file");
        }

    // read magic
    unsigned int magic = 0x444d4f48;
    unsigned int file_magic;
    f.read((char*)&file_magic, sizeof(int));
    if (magic != file_magic)
        {
        m_exec_conf->msg->error() << endl << fname << " does not appear to be a hoomd_bin file." << endl;
        if (enable_decompression)
            m_exec_conf->msg->error() << "Is it perhaps an uncompressed file with an erroneous .gz extension?" << endl << endl;
        else
            m_exec_conf->msg->error() << "Is it perhaps a compressed file without a .gz extension?" << endl << endl;

        throw runtime_error("Error reading binary file");
        }

    int version = 3;
    int file_version;
    f.read((char*)&file_version, sizeof(int));

    // right now, the version tag doesn't do anything: just warn if they don't match
    if (version != file_version)
        {
        m_exec_conf->msg->error() << endl
             << "hoomd binary file does not match the current version,"
             << endl << endl;
        throw runtime_error("Error reading binary file");
        }

    //parse timestep
    int timestep;
    f.read((char*)&timestep, sizeof(unsigned int));
    m_timestep = timestep;

    //parse dimensions
    unsigned int dimensions;
    f.read((char*)&dimensions, sizeof(unsigned int));
    m_num_dimensions = dimensions;

    //parse box
    Scalar Lx,Ly,Lz;
    f.read((char*)&Lx, sizeof(Scalar));
    f.read((char*)&Ly, sizeof(Scalar));
    f.read((char*)&Lz, sizeof(Scalar));
    m_box = BoxDim(Lx,Ly,Lz);

    //allocate memory for particle arrays
    unsigned int np = 0;
    f.read((char*)&np, sizeof(unsigned int));
    m_tag_array.resize(np); m_rtag_array.resize(np);
    m_x_array.resize(np); m_y_array.resize(np); m_z_array.resize(np);
    m_ix_array.resize(np); m_iy_array.resize(np); m_iz_array.resize(np);
    m_vx_array.resize(np); m_vy_array.resize(np); m_vz_array.resize(np);
    m_ax_array.resize(np); m_ay_array.resize(np); m_az_array.resize(np);
    m_mass_array.resize(np);
    m_diameter_array.resize(np);
    m_type_array.resize(np);
    m_charge_array.resize(np);
    m_body_array.resize(np);

    //parse particle arrays
    f.read((char*)&(m_tag_array[0]), np*sizeof(unsigned int));
    f.read((char*)&(m_rtag_array[0]), np*sizeof(unsigned int));
    f.read((char*)&(m_x_array[0]), np*sizeof(Scalar));
    f.read((char*)&(m_y_array[0]), np*sizeof(Scalar));
    f.read((char*)&(m_z_array[0]), np*sizeof(Scalar));
    f.read((char*)&(m_ix_array[0]), np*sizeof(int));
    f.read((char*)&(m_iy_array[0]), np*sizeof(int));
    f.read((char*)&(m_iz_array[0]), np*sizeof(int));
    f.read((char*)&(m_vx_array[0]), np*sizeof(Scalar));
    f.read((char*)&(m_vy_array[0]), np*sizeof(Scalar));
    f.read((char*)&(m_vz_array[0]), np*sizeof(Scalar));
    f.read((char*)&(m_ax_array[0]), np*sizeof(Scalar));
    f.read((char*)&(m_ay_array[0]), np*sizeof(Scalar));
    f.read((char*)&(m_az_array[0]), np*sizeof(Scalar));
    f.read((char*)&(m_mass_array[0]), np*sizeof(Scalar));
    f.read((char*)&(m_diameter_array[0]), np*sizeof(Scalar));
    f.read((char*)&(m_charge_array[0]), np*sizeof(Scalar));
    f.read((char*)&(m_body_array[0]), np*sizeof(unsigned int));

    //parse types
    unsigned int ntypes = 0;
    f.read((char*)&ntypes, sizeof(unsigned int));
    m_type_mapping.resize(ntypes);
    for (unsigned int i = 0; i < ntypes; i++)
        m_type_mapping[i] = read_string(f);
    f.read((char*)&(m_type_array[0]), np*sizeof(unsigned int));

    //parse integrator states
    {
    std::vector<IntegratorVariables> v;
    unsigned int ni = 0;
    f.read((char*)&ni, sizeof(unsigned int));
    v.resize(ni);
    for (unsigned int j = 0; j < ni; j++)
        {
        v[j].type = read_string(f);

        v[j].variable.clear();
        unsigned int nv = 0;
        f.read((char*)&nv, sizeof(unsigned int));
        for (unsigned int k=0; k<nv; k++)
            {
            Scalar var;
            f.read((char*)&var, sizeof(Scalar));
            v[j].variable.push_back(var);
            }
        }
        m_integrator_variables = v;
    }

    //parse bonds
    {
    ntypes = 0;
    f.read((char*)&ntypes, sizeof(unsigned int));
    m_bond_type_mapping.resize(ntypes);
    for (unsigned int i = 0; i < ntypes; i++)
        m_bond_type_mapping[i] = read_string(f);

    unsigned int nb = 0;
    f.read((char*)&nb, sizeof(unsigned int));
    for (unsigned int j = 0; j < nb; j++)
        {
        unsigned int typ, a, b;
        f.read((char*)&typ, sizeof(unsigned int));
        f.read((char*)&a, sizeof(unsigned int));
        f.read((char*)&b, sizeof(unsigned int));

        m_bonds.push_back(Bond(typ, a, b));
        }
    }

    //parse angles
    {
    ntypes = 0;
    f.read((char*)&ntypes, sizeof(unsigned int));
    m_angle_type_mapping.resize(ntypes);
    for (unsigned int i = 0; i < ntypes; i++)
        m_angle_type_mapping[i] = read_string(f);

    unsigned int na = 0;
    f.read((char*)&na, sizeof(unsigned int));
    for (unsigned int j = 0; j < na; j++)
        {
        unsigned int typ, a, b, c;
        f.read((char*)&typ, sizeof(unsigned int));
        f.read((char*)&a, sizeof(unsigned int));
        f.read((char*)&b, sizeof(unsigned int));
        f.read((char*)&c, sizeof(unsigned int));

        m_angles.push_back(Angle(typ, a, b, c));
        }
    }

    //parse dihedrals
    {
    ntypes = 0;
    f.read((char*)&ntypes, sizeof(unsigned int));
    m_dihedral_type_mapping.resize(ntypes);
    for (unsigned int i = 0; i < ntypes; i++)
        m_dihedral_type_mapping[i] = read_string(f);

    unsigned int nd = 0;
    f.read((char*)&nd, sizeof(unsigned int));
    for (unsigned int j = 0; j < nd; j++)
        {
        unsigned int typ, a, b, c, d;
        f.read((char*)&typ, sizeof(unsigned int));
        f.read((char*)&a, sizeof(unsigned int));
        f.read((char*)&b, sizeof(unsigned int));
        f.read((char*)&c, sizeof(unsigned int));
        f.read((char*)&d, sizeof(unsigned int));

        m_dihedrals.push_back(Dihedral(typ, a, b, c, d));
        }
    }

    //parse impropers
    {
    ntypes = 0;
    f.read((char*)&ntypes, sizeof(unsigned int));
    m_improper_type_mapping.resize(ntypes);
    for (unsigned int i = 0; i < ntypes; i++)
        m_improper_type_mapping[i] = read_string(f);

    unsigned int nd = 0;
    f.read((char*)&nd, sizeof(unsigned int));
    for (unsigned int j = 0; j < nd; j++)
        {
        unsigned int typ, a, b, c, d;
        f.read((char*)&typ, sizeof(unsigned int));
        f.read((char*)&a, sizeof(unsigned int));
        f.read((char*)&b, sizeof(unsigned int));
        f.read((char*)&c, sizeof(unsigned int));
        f.read((char*)&d, sizeof(unsigned int));

        m_impropers.push_back(Dihedral(typ, a, b, c, d));
        }
    }

    //parse walls
    {
    unsigned int nw = 0;
    f.read((char*)&nw, sizeof(unsigned int));
    for (unsigned int j = 0; j < nw; j++)
        {
        Scalar ox, oy, oz, nx, ny, nz;
        f.read((char*)&(ox), sizeof(Scalar));
        f.read((char*)&(oy), sizeof(Scalar));
        f.read((char*)&(oz), sizeof(Scalar));
        f.read((char*)&(nx), sizeof(Scalar));
        f.read((char*)&(ny), sizeof(Scalar));
        f.read((char*)&(nz), sizeof(Scalar));
        m_walls.push_back(Wall(ox,oy,oz,nx,ny,nz));
        }
    }

    // parse rigid bodies
    {
    unsigned int n_bodies = 0;
    f.read((char*)&n_bodies, sizeof(unsigned int));

    if (n_bodies == 0) return;

    m_com.resize(n_bodies);
    m_vel.resize(n_bodies);
    m_angmom.resize(n_bodies);
    m_body_image.resize(n_bodies);

    for (unsigned int body = 0; body < n_bodies; body++)
        {
        f.read((char*)&(m_com[body].x), sizeof(Scalar));
        f.read((char*)&(m_com[body].y), sizeof(Scalar));
        f.read((char*)&(m_com[body].z), sizeof(Scalar));
        f.read((char*)&(m_com[body].w), sizeof(Scalar));

        f.read((char*)&(m_vel[body].x), sizeof(Scalar));
        f.read((char*)&(m_vel[body].y), sizeof(Scalar));
        f.read((char*)&(m_vel[body].z), sizeof(Scalar));
        f.read((char*)&(m_vel[body].w), sizeof(Scalar));

        f.read((char*)&(m_angmom[body].x), sizeof(Scalar));
        f.read((char*)&(m_angmom[body].y), sizeof(Scalar));
        f.read((char*)&(m_angmom[body].z), sizeof(Scalar));
        f.read((char*)&(m_angmom[body].w), sizeof(Scalar));

        f.read((char*)&(m_body_image[body].x), sizeof(int));
        f.read((char*)&(m_body_image[body].y), sizeof(int));
        f.read((char*)&(m_body_image[body].z), sizeof(int));
        }

    }

    // check for required items in the file
    if (m_x_array.size() == 0)
        {
        m_exec_conf->msg->error() << endl << "No particles found in binary file" << endl << endl;
        throw runtime_error("Error extracting data from hoomd_binary file");
        }

    // notify the user of what we have accomplished
    m_exec_conf->msg->notice(2) << "--- hoomd_binary file read summary" << endl;
    m_exec_conf->msg->notice(2) << m_x_array.size() << " positions at timestep " << m_timestep << endl;
    if (m_ix_array.size() > 0)
        m_exec_conf->msg->notice(2) << m_ix_array.size() << " images" << endl;
    if (m_vx_array.size() > 0)
        m_exec_conf->msg->notice(2) << m_vx_array.size() << " velocities" << endl;
    if (m_mass_array.size() > 0)
        m_exec_conf->msg->notice(2) << m_mass_array.size() << " masses" << endl;
    if (m_diameter_array.size() > 0)
        m_exec_conf->msg->notice(2) << m_diameter_array.size() << " diameters" << endl;
    if (m_charge_array.size() > 0)
        m_exec_conf->msg->notice(2) << m_charge_array.size() << " charges" << endl;
    m_exec_conf->msg->notice(2) << m_type_mapping.size() <<  " particle types" << endl;
    if (m_integrator_variables.size() > 0)
        m_exec_conf->msg->notice(2) << m_integrator_variables.size() << " integrator states" << endl;
    if (m_bonds.size() > 0)
        m_exec_conf->msg->notice(2) << m_bonds.size() << " bonds" << endl;
    if (m_angles.size() > 0)
        m_exec_conf->msg->notice(2) << m_angles.size() << " angles" << endl;
    if (m_dihedrals.size() > 0)
        m_exec_conf->msg->notice(2) << m_dihedrals.size() << " dihedrals" << endl;
    if (m_impropers.size() > 0)
        m_exec_conf->msg->notice(2) << m_impropers.size() << " impropers" << endl;
    if (m_walls.size() > 0)
        m_exec_conf->msg->notice(2) << m_walls.size() << " walls" << endl;
    }

void export_HOOMDBinaryInitializer()
    {
    class_< HOOMDBinaryInitializer >("HOOMDBinaryInitializer",
        init<boost::shared_ptr<const ExecutionConfiguration>, const string&>())
        // virtual methods from ParticleDataInitializer are inherited
        .def("getSnapshot", &HOOMDBinaryInitializer::getSnapshot)
        .def("getTimeStep", &HOOMDBinaryInitializer::getTimeStep)
        .def("setTimeStep", &HOOMDBinaryInitializer::setTimeStep)
        ;
    }

#ifdef WIN32
#pragma warning( pop )
#endif
