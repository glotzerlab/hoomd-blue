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

#include "BondTablePotential.h"

#include <stdexcept>

/*! \file BondTablePotential.cc
    \brief Defines the BondTablePotential class
*/

using namespace std;

/*! \param sysdef System to compute forces on
    \param table_width Width the tables will be in memory
    \param log_suffix Name given to this instance of the table potential
*/
BondTablePotential::BondTablePotential(boost::shared_ptr<SystemDefinition> sysdef,
                               unsigned int table_width,
                               const std::string& log_suffix)
        : ForceCompute(sysdef), m_table_width(table_width)
    {
    m_exec_conf->msg->notice(5) << "Constructing BondTablePotential" << endl;
    
    assert(m_pdata);

    // access the bond data for later use
    m_bond_data = m_sysdef->getBondData();

    if (table_width == 0)
        {
        m_exec_conf->msg->error() << "bond.table: Table width of 0 is invalid" << endl;
        throw runtime_error("Error initializing BondTablePotential");
        }


  
    
    // allocate storage for the tables and parameters
    GPUArray<float2> tables(m_table_width, m_bond_data->getNBondTypes(), exec_conf);
    m_tables.swap(tables);
    GPUArray<Scalar4> params(m_bond_data->getNBondTypes(), exec_conf);
    m_params.swap(params);
    assert(!m_tables.isNull());

    // helper to compute indices
    Index2D table_value(m_tables->getPitch(),m_bond_data->getNBondTypes());
    m_table_value = table_value;


  
    

    m_log_name = std::string("bond_table_energy") + log_suffix;
    }
    
BondTablePotential::~BondTablePotential()
        {
        m_exec_conf->msg->notice(5) << "Destroying BondTablePotential" << endl;
        }

/*! \param type Type of the bond to set parameters for
    \param V Table for the potential V
    \param F Table for the potential F (must be - dV / dr)
    \param rmin Minimum r in the potential
    \param rmax Maximum r in the potential
    \post Values from \a V and \a F are copied into the interal storage for type pair (type)
    \note See BondTablePotential for a detailed definiton of rmin and rmax
*/
void BondTablePotential::setTable(unsigned int type,
                              const std::vector<float> &V,
                              const std::vector<float> &F,
                              Scalar rmin,
                              Scalar rmax)
    {

    // make sure the type is valid
    if (type >= m_bond_data->getNBondTypes())
        {
        cout << endl << "***Error! Invalid bond type specified" << endl << endl;
        throw runtime_error("Error setting parameters in PotentialBond");
        }


    // access the arrays
    ArrayHandle<float2> h_tables(m_tables, access_location::host, access_mode::readwrite);
    ArrayHandle<Scalar4> h_params(m_params, access_location::host, access_mode::readwrite);

    // range check on the parameters
    if (rmin < 0 || rmax < 0 || rmax <= rmin)
        {
        m_exec_conf->msg->error()<< "bond.table: rmin, rmax (" << rmin << "," << rmax
             << ") is invalid."  << endl;
        throw runtime_error("Error initializing BondTablePotential");
        }

    if (V.size() != m_table_width || F.size() != m_table_width)
        {
        m_exec_conf->msg->error() << "bond.table: table provided to setTable is not of the correct size" << endl;
        throw runtime_error("Error initializing BondTablePotential");
        }

    // fill out the parameters
    h_params.data[type].x = rmin;
    h_params.data[type].y = rmax;
    h_params.data[type].z = (rmax - rmin) / Scalar(m_table_width - 1);

    // fill out the table
    for (unsigned int i = 0; i < m_table_width; i++)
        {
        h_tables.data[m_table_value(i, type)].x = V[i];
        h_tables.data[m_table_value(i, type)].y = F[i];
        }
    }

/*! BondTablePotential provides
    - \c bond_table_energy
*/
std::vector< std::string > BondTablePotential::getProvidedLogQuantities()
    {
    vector<string> list;
    list.push_back(m_log_name);
    return list;
    }

Scalar BondTablePotential::getLogValue(const std::string& quantity, unsigned int timestep)
    {
    if (quantity == m_log_name)
        {
        compute(timestep);
        return calcEnergySum();
        }
    else
        {
        m_exec_conf->msg->error() << "bond.table: " << quantity << " is not a valid log quantity for BondTablePotential" << endl;
        throw runtime_error("Error getting log value");
        }
    }

/*! \post The table based forces are computed for the given timestep.
\param timestep specifies the current time step of the simulation
*/
void BondTablePotential::computeForces(unsigned int timestep)
    {

    // start the profile for this compute
    if (m_prof) m_prof->push("Bond Table pair");


    // access the particle data
    ArrayHandle<Scalar4> h_pos(m_pdata->getPositions(), access_location::host, access_mode::read);
    ArrayHandle<Scalar4> h_force(m_force,access_location::host, access_mode::overwrite);
    ArrayHandle<Scalar> h_virial(m_virial,access_location::host, access_mode::overwrite);
    ArrayHandle<unsigned int> h_rtag(m_pdata->getRTags(), access_location::host, access_mode::read);


    // there are enough other checks on the input data: but it doesn't hurt to be safe
    assert(h_force.data);
    assert(h_virial.data);
    assert(h_pos.data);

    // Zero data for force calculation.
    memset((void*)h_force.data,0,sizeof(Scalar4)*m_force.getNumElements());
    memset((void*)h_virial.data,0,sizeof(Scalar)*m_virial.getNumElements());

    // get a local copy of the simulation box too
    const BoxDim& box = m_pdata->getBox();

    // access the table data
    ArrayHandle<float2> h_tables(m_tables, access_location::host, access_mode::read);
    ArrayHandle<Scalar4> h_params(m_params, access_location::host, access_mode::read);

    // for each of the bonds
    const unsigned int size = (unsigned int)m_bond_data->getNumBonds();
    for (unsigned int i = 0; i < size; i++)
        {
        // lookup the tag of each of the particles participating in the bond
        const Bond& bond = m_bond_data->getBond(i);
        assert(bond.a < m_pdata->getN());
        assert(bond.b < m_pdata->getN());

        // transform a and b into indicies into the particle data arrays
        // (MEM TRANSFER: 4 integers)
        unsigned int idx_a = h_rtag.data[bond.a];
        unsigned int idx_b = h_rtag.data[bond.b];
        assert(idx_a < m_pdata->getN());
        assert(idx_b < m_pdata->getN());

        Scalar3 pa = make_scalar3(h_pos.data[idx_a].x, h_pos.data[idx_a].y, h_pos.data[idx_a].z);
        Scalar3 pb = make_scalar3(h_pos.data[idx_b].x, h_pos.data[idx_b].y, h_pos.data[idx_b].z);
        Scalar3 dx = pb-pa;
        

        // apply periodic boundary conditions
        dx = box.minImage(dx);

        // access needed parameters
        Scalar4 params = h_params.data[bond.type];
        Scalar rmin = params.x;
        Scalar rmax = params.y;
        Scalar delta_r = params.z;

        // start computing the force
        Scalar rsq = dot(dx,dx);
        Scalar r = sqrt(rsq);

        // only compute the force if the particles are within the region defined by V

        if (r < rmax && r >= rmin)
            {
            // precomputed term
            Scalar value_f = (r - rmin) / delta_r;

            // compute index into the table and read in values

            /// Here we use the table!!
            unsigned int value_i = (unsigned int)floor(value_f);
            float2 VF0 = h_tables.data[m_table_value(value_i, bond.type)];
            float2 VF1 = h_tables.data[m_table_value(value_i+1, bond.type)];
            // unpack the data
            Scalar V0 = VF0.x;
            Scalar V1 = VF1.x;
            Scalar F0 = VF0.y;
            Scalar F1 = VF1.y;

            // compute the linear interpolation coefficient
            Scalar f = value_f - float(value_i);

            // interpolate to get V and F;
            Scalar V = V0 + f * (V1 - V0);
            Scalar F = F0 + f * (F1 - F0);

            // convert to standard variables used by the other pair computes in HOOMD-blue
            Scalar force_divr = Scalar(0.0);
            if (r > Scalar(0.0))
                force_divr = F / r;
            Scalar bond_eng = Scalar(0.5) * V;

            // compute the virial
            Scalar bond_virial[6];
            Scalar force_div2r = Scalar(0.5) * force_divr;
            bond_virial[0] = dx.x * dx.x * force_div2r; // xx
            bond_virial[1] = dx.x * dx.y * force_div2r; // xy
            bond_virial[2] = dx.x * dx.z * force_div2r; // xz
            bond_virial[3] = dx.y * dx.y * force_div2r; // yy
            bond_virial[4] = dx.y * dx.z * force_div2r; // yz
            bond_virial[5] = dx.z * dx.z * force_div2r; // zz

            // add the force to the particles
            // (MEM TRANSFER: 20 Scalars / FLOPS 16)
            h_force.data[idx_b].x += force_divr * dx.x;
            h_force.data[idx_b].y += force_divr * dx.y;
            h_force.data[idx_b].z += force_divr * dx.z;
            h_force.data[idx_b].w += bond_eng;
            for (unsigned int i = 0; i < 6; i++)
                h_virial.data[i*m_virial_pitch+idx_b]  += bond_virial[i];

            h_force.data[idx_a].x -= force_divr * dx.x;
            h_force.data[idx_a].y -= force_divr * dx.y;
            h_force.data[idx_a].z -= force_divr * dx.z;
            h_force.data[idx_a].w += bond_eng;
            for (unsigned int i = 0; i < 6; i++)
                h_virial.data[i*m_virial_pitch+idx_a]  += bond_virial[i];

            }
        else
            {
            cerr << endl << "***Error! Table bond out of bounds" << endl << endl;
            throw std::runtime_error("Error in bond calculation");
            }

        }
    if (m_prof) m_prof->pop();
    }

//! Exports the BondTablePotential class to python
void export_BondTablePotential()
    {
    class_<BondTablePotential, boost::shared_ptr<BondTablePotential>, bases<ForceCompute>, boost::noncopyable >
    ("BondTablePotential", init< boost::shared_ptr<SystemDefinition>, unsigned int, const std::string& >())
    .def("setTable", &BondTablePotential::setTable)
    ;
    }
