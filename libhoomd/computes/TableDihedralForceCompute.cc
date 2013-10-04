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

// Maintainer: phillicl

#include <boost/python.hpp>
using namespace boost::python;

#include "TableDihedralForceCompute.h"

#include <stdexcept>

/*! \file TableDihedralForceCompute.cc
    \brief Defines the TableDihedralForceCompute class
*/

using namespace std;

// SMALL a relatively small number
#define SMALL 0.001f

/*! \param sysdef System to compute forces on
    \param table_width Width the tables will be in memory
    \param log_suffix Name given to this instance of the table potential
*/
TableDihedralForceCompute::TableDihedralForceCompute(boost::shared_ptr<SystemDefinition> sysdef,
                               unsigned int table_width,
                               const std::string& log_suffix)
        : ForceCompute(sysdef), m_table_width(table_width)
    {
    m_exec_conf->msg->notice(5) << "Constructing TableDihedralForceCompute" << endl;
    
    assert(m_pdata);

    // access the dihedral data for later use
    m_dihedral_data = m_sysdef->getDihedralData();

    if (table_width == 0)
        {
        m_exec_conf->msg->error() << "dihedral.table: Table width of 0 is invalid" << endl;
        throw runtime_error("Error initializing TableDihedralForceCompute");
        }


  
    
    // allocate storage for the tables and parameters
    GPUArray<Scalar2> tables(m_table_width, m_dihedral_data->getNDihedralTypes(), exec_conf);
    m_tables.swap(tables);
    assert(!m_tables.isNull());

    // helper to compute indices
    Index2D table_value(m_tables.getPitch(),m_dihedral_data->getNDihedralTypes());
    m_table_value = table_value;

    m_log_name = std::string("dihedral_table_energy") + log_suffix;
    }
    
TableDihedralForceCompute::~TableDihedralForceCompute()
        {
        m_exec_conf->msg->notice(5) << "Destroying TableDihedralForceCompute" << endl;
        }

/*! \param type Type of the dihedral to set parameters for
    \param V Table for the potential V
    \param T Table for the potential T (must be - dV / dphi)
    \post Values from \a V and \a T are copied into the interal storage for type pair (type)
*/
void TableDihedralForceCompute::setTable(unsigned int type,
                              const std::vector<Scalar> &V,
                              const std::vector<Scalar> &T)
    {

    // make sure the type is valid
    if (type >= m_dihedral_data->getNDihedralTypes())
        {
        cout << endl << "***Error! Invalid dihedral type specified" << endl << endl;
        throw runtime_error("Error setting parameters in PotentialDihedral");
        }


    // access the arrays
    ArrayHandle<Scalar2> h_tables(m_tables, access_location::host, access_mode::readwrite);

    if (V.size() != m_table_width || T.size() != m_table_width)
        {
        m_exec_conf->msg->error() << "dihedral.table: table provided to setTable is not of the correct size" << endl;
        throw runtime_error("Error initializing TableDihedralForceCompute");
        }

    // fill out the table
    for (unsigned int i = 0; i < m_table_width; i++)
        {
        h_tables.data[m_table_value(i, type)].x = V[i];
        h_tables.data[m_table_value(i, type)].y = T[i];
        }
    }

/*! TableDihedralForceCompute provides
    - \c dihedral_table_energy
*/
std::vector< std::string > TableDihedralForceCompute::getProvidedLogQuantities()
    {
    vector<string> list;
    list.push_back(m_log_name);
    return list;
    }

Scalar TableDihedralForceCompute::getLogValue(const std::string& quantity, unsigned int timestep)
    {
    if (quantity == m_log_name)
        {
        compute(timestep);
        return calcEnergySum();
        }
    else
        {
        m_exec_conf->msg->error() << "dihedral.table: " << quantity << " is not a valid log quantity for TableDihedralForceCompute" << endl;
        throw runtime_error("Error getting log value");
        }
    }

/*! \post The table based forces are computed for the given timestep.
\param timestep specifies the current time step of the simulation
*/
void TableDihedralForceCompute::computeForces(unsigned int timestep)
    {

    // start the profile for this compute
    if (m_prof) m_prof->push("Dihedral Table pair");


    // access the particle data
    ArrayHandle<Scalar4> h_pos(m_pdata->getPositions(), access_location::host, access_mode::read);
    ArrayHandle<Scalar4> h_force(m_force,access_location::host, access_mode::overwrite);
    ArrayHandle<Scalar> h_virial(m_virial,access_location::host, access_mode::overwrite);
    ArrayHandle<unsigned int> h_rtag(m_pdata->getRTags(), access_location::host, access_mode::read);


    // there are enough other checks on the input data: but it doesn't hurt to be safe
    assert(h_force.data);
    assert(h_virial.data);
    assert(h_pos.data);
    
    unsigned int virial_pitch = m_virial.getPitch();

    // Zero data for force calculation.
    memset((void*)h_force.data,0,sizeof(Scalar4)*m_force.getNumElements());
    memset((void*)h_virial.data,0,sizeof(Scalar)*m_virial.getNumElements());

    // get a local copy of the simulation box too
    const BoxDim& box = m_pdata->getBox();

    // access the table data
    ArrayHandle<Scalar2> h_tables(m_tables, access_location::host, access_mode::read);

    // for each of the dihedrals
    const unsigned int size = (unsigned int)m_dihedral_data->getNumDihedrals();
    for (unsigned int i = 0; i < size; i++)
        {
        // lookup the tag of each of the particles participating in the dihedral
        const Dihedral& dihedral = m_dihedral_data->getDihedral(i);
        assert(dihedral.a < m_pdata->getN());
        assert(dihedral.b < m_pdata->getN());
        assert(dihedral.c < m_pdata->getN());
        assert(dihedral.d < m_pdata->getN());        

        // transform a and b into indicies into the particle data arrays
        // (MEM TRANSFER: 4 integers)
        unsigned int idx_a = h_rtag.data[dihedral.a];
        unsigned int idx_b = h_rtag.data[dihedral.b];
        unsigned int idx_c = h_rtag.data[dihedral.c];
        unsigned int idx_d = h_rtag.data[dihedral.d];
        assert(idx_a < m_pdata->getN());
        assert(idx_b < m_pdata->getN());
        assert(idx_c < m_pdata->getN());
        assert(idx_d < m_pdata->getN());

        // calculate d\vec{r}
        Scalar3 dab;
        dab.x = h_pos.data[idx_a].x - h_pos.data[idx_b].x; //vb1x
        dab.y = h_pos.data[idx_a].y - h_pos.data[idx_b].y; //vb1y
        dab.z = h_pos.data[idx_a].z - h_pos.data[idx_b].z; //vb1z
        
        Scalar3 dcb;
        dcb.x = h_pos.data[idx_c].x - h_pos.data[idx_b].x; //vb2x
        dcb.y = h_pos.data[idx_c].y - h_pos.data[idx_b].y; //vb2y
        dcb.z = h_pos.data[idx_c].z - h_pos.data[idx_b].z; //vb2z
        
        Scalar3 dcbm;
        dcbm.x = -dcb.x;
        dcbm.y = -dcb.y;
        dcbm.z = -dcb.z;
        
        Scalar3 ddc;
        ddc.x = h_pos.data[idx_d].x - h_pos.data[idx_c].x; //vb3x
        ddc.y = h_pos.data[idx_d].y - h_pos.data[idx_c].y; //vb3y
        ddc.z = h_pos.data[idx_d].z - h_pos.data[idx_c].z; //vb3z
        
        // apply periodic boundary conditions
        dab = box.minImage(dab);
        dcb = box.minImage(dcb);
        ddc = box.minImage(ddc);
        dcbm = box.minImage(dcbm);
        
        // c0 calculation
        Scalar sb1 = 1.0 / (dab.x*dab.x + dab.y*dab.y + dab.z*dab.z);
        Scalar sb2 = 1.0 / (dcb.x*dcb.x + dcb.y*dcb.y + dcb.z*dcb.z);
        Scalar sb3 = 1.0 / (ddc.x*ddc.x + ddc.y*ddc.y + ddc.z*ddc.z);
            
        Scalar rb1 = sqrt(sb1);
        Scalar rb3 = sqrt(sb3);
            
        Scalar c0 = (dab.x*ddc.x + dab.y*ddc.y + dab.z*ddc.z) * rb1*rb3;

        // 1st and 2nd angle
            
        Scalar b1mag2 = dab.x*dab.x + dab.y*dab.y + dab.z*dab.z;
        Scalar b1mag = sqrt(b1mag2);
        Scalar b2mag2 = dcb.x*dcb.x + dcb.y*dcb.y + dcb.z*dcb.z;
        Scalar b2mag = sqrt(b2mag2);
        Scalar b3mag2 = ddc.x*ddc.x + ddc.y*ddc.y + ddc.z*ddc.z;
        Scalar b3mag = sqrt(b3mag2);

        Scalar ctmp = dab.x*dcb.x + dab.y*dcb.y + dab.z*dcb.z;
        Scalar r12c1 = 1.0 / (b1mag*b2mag);
        Scalar c1mag = ctmp * r12c1;

        ctmp = dcbm.x*ddc.x + dcbm.y*ddc.y + dcbm.z*ddc.z;
        Scalar r12c2 = 1.0 / (b2mag*b3mag);
        Scalar c2mag = ctmp * r12c2;

        // cos and sin of 2 angles and final c

        Scalar sin2 = 1.0 - c1mag*c1mag;
        if (sin2 < 0.0) sin2 = 0.0;
        Scalar sc1 = sqrt(sin2);
        if (sc1 < SMALL) sc1 = SMALL;
        sc1 = 1.0/sc1;

        sin2 = 1.0 - c2mag*c2mag;
        if (sin2 < 0.0) sin2 = 0.0;
        Scalar sc2 = sqrt(sin2);
        if (sc2 < SMALL) sc2 = SMALL;
        sc2 = 1.0/sc2;

        Scalar s1 = sc1 * sc1;
        Scalar s2 = sc2 * sc2;
        Scalar s12 = sc1 * sc2;
        Scalar c = (c0 + c1mag*c2mag) * s12;
      
        if (c > 1.0) c = 1.0;
        if (c < -1.0) c = -1.0;
        
        //phi
        Scalar phi = acos(c);
        // precomputed term
        Scalar delta_phi = M_PI/Scalar(m_table_width - 1);       
        Scalar value_f = (phi) / delta_phi;

        // compute index into the table and read in values

        /// Here we use the table!!
        unsigned int value_i = (unsigned int)floor(value_f);
        Scalar2 VT0 = h_tables.data[m_table_value(value_i, dihedral.type)];
        Scalar2 VT1 = h_tables.data[m_table_value(value_i+1, dihedral.type)];
        // unpack the data
        Scalar V0 = VT0.x;
        Scalar V1 = VT1.x;
        Scalar T0 = VT0.y;
        Scalar T1 = VT1.y;

        // compute the linear interpolation coefficient
        Scalar f = value_f - Scalar(value_i);

        // interpolate to get V and T;
        Scalar V = V0 + f * (V1 - V0);
        Scalar T = T0 + f * (T1 - T0);


        Scalar a = T; 
        c = c * a;
        s12 = s12 * a;
        Scalar a11 = c*sb1*s1;
        Scalar a22 = -sb2 * (2.0*c0*s12 - c*(s1+s2));
        Scalar a33 = c*sb3*s2;
        Scalar a12 = -r12c1*(c1mag*c*s1 + c2mag*s12);
        Scalar a13 = -rb1*rb3*s12;
        Scalar a23 = r12c2*(c2mag*c*s2 + c1mag*s12);

        Scalar sx2  = a12*dab.x + a22*dcb.x + a23*ddc.x;
        Scalar sy2  = a12*dab.y + a22*dcb.y + a23*ddc.y;
        Scalar sz2  = a12*dab.z + a22*dcb.z + a23*ddc.z;
        
        Scalar ffax = a11*dab.x + a12*dcb.x + a13*ddc.x;
        Scalar ffay = a11*dab.y + a12*dcb.y + a13*ddc.y;
        Scalar ffaz = a11*dab.z + a12*dcb.z + a13*ddc.z;
        
        Scalar ffbx = -sx2 - ffax;
        Scalar ffby = -sy2 - ffay;
        Scalar ffbz = -sz2 - ffaz;
        
        Scalar ffdx = a13*dab.x + a23*dcb.x + a33*ddc.x;
        Scalar ffdy = a13*dab.y + a23*dcb.y + a33*ddc.y;
        Scalar ffdz = a13*dab.z + a23*dcb.z + a33*ddc.z;
        
        Scalar ffcx = sx2 - ffdx;
        Scalar ffcy = sy2 - ffdy;
        Scalar ffcz = sz2 - ffdz;
        
        // Now, apply the force to each individual atom a,b,c,d
        // and accumlate the energy/virial
        // compute 1/4 of the energy, 1/4 for each atom in the dihedral
        Scalar dihedral_eng = V*Scalar(0.25);  // the .125 term is (1/2)K * 1/4

        // compute 1/4 of the virial, 1/4 for each atom in the dihedral
        // symmetrized version of virial tensor
        Scalar dihedral_virial[6];
        dihedral_virial[0] = (1./4.)*(dab.x*ffax + dcb.x*ffcx + (ddc.x+dcb.x)*ffdx);
        dihedral_virial[1] = (1./8.)*(dab.x*ffay + dcb.x*ffcy + (ddc.x+dcb.x)*ffdy
                                     +dab.y*ffax + dcb.y*ffcx + (ddc.y+dcb.y)*ffdx);
        dihedral_virial[2] = (1./8.)*(dab.x*ffaz + dcb.x*ffcz + (ddc.x+dcb.x)*ffdz
                                     +dab.z*ffax + dcb.z*ffcx + (ddc.z+dcb.z)*ffdx);
        dihedral_virial[3] = (1./4.)*(dab.y*ffay + dcb.y*ffcy + (ddc.y+dcb.y)*ffdy);
        dihedral_virial[4] = (1./8.)*(dab.y*ffaz + dcb.y*ffcz + (ddc.y+dcb.y)*ffdz
                                     +dab.z*ffay + dcb.z*ffcy + (ddc.z+dcb.z)*ffdy);
        dihedral_virial[5] = (1./4.)*(dab.z*ffaz + dcb.z*ffcz + (ddc.z+dcb.z)*ffdz);
       
        h_force.data[idx_a].x += ffax; 
        h_force.data[idx_a].y += ffay; 
        h_force.data[idx_a].z += ffaz; 
        h_force.data[idx_a].w += dihedral_eng; 
        for (int k = 0; k < 6; k++)
           h_virial.data[virial_pitch*k+idx_a]  += dihedral_virial[k];

        h_force.data[idx_b].x += ffbx; 
        h_force.data[idx_b].y += ffby; 
        h_force.data[idx_b].z += ffbz; 
        h_force.data[idx_b].w += dihedral_eng; 
        for (int k = 0; k < 6; k++)
           h_virial.data[virial_pitch*k+idx_b]  += dihedral_virial[k];

        h_force.data[idx_c].x += ffcx; 
        h_force.data[idx_c].y += ffcy; 
        h_force.data[idx_c].z += ffcz; 
        h_force.data[idx_c].w += dihedral_eng; 
        for (int k = 0; k < 6; k++)
           h_virial.data[virial_pitch*k+idx_c]  += dihedral_virial[k];

        h_force.data[idx_d].x += ffdx; 
        h_force.data[idx_d].y += ffdy; 
        h_force.data[idx_d].z += ffdz; 
        h_force.data[idx_d].w += dihedral_eng; 
        for (int k = 0; k < 6; k++)
           h_virial.data[virial_pitch*k+idx_d]  += dihedral_virial[k];
       }

    if (m_prof) m_prof->pop();
    }

//! Exports the TableDihedralForceCompute class to python
void export_TableDihedralForceCompute()
    {
    class_<TableDihedralForceCompute, boost::shared_ptr<TableDihedralForceCompute>, bases<ForceCompute>, boost::noncopyable >
    ("TableDihedralForceCompute", init< boost::shared_ptr<SystemDefinition>, unsigned int, const std::string& >())
    .def("setTable", &TableDihedralForceCompute::setTable)
    ;
    }
