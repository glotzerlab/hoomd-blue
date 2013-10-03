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

// Maintainer: dnlebard

#ifdef WIN32
#pragma warning( push )
#pragma warning( disable : 4244 )
#endif

#include <boost/python.hpp>
using namespace boost::python;

#include "HarmonicAngleForceCompute.h"

#include <iostream>
#include <sstream>
#include <stdexcept>
#include <math.h>

using namespace std;

// SMALL a relatively small number
#define SMALL Scalar(0.001)

/*! \file HarmonicAngleForceCompute.cc
    \brief Contains code for the HarmonicAngleForceCompute class
*/

/*! \param sysdef System to compute forces on
    \post Memory is allocated, and forces are zeroed.
*/
HarmonicAngleForceCompute::HarmonicAngleForceCompute(boost::shared_ptr<SystemDefinition> sysdef) 
    :  ForceCompute(sysdef), m_K(NULL), m_t_0(NULL)
    {
    m_exec_conf->msg->notice(5) << "Constructing HarmonicAngleForceCompute" << endl;

    // access the angle data for later use
    m_angle_data = m_sysdef->getAngleData();

    // check for some silly errors a user could make
    if (m_angle_data->getNAngleTypes() == 0)
        {
        m_exec_conf->msg->error() << "angle.harmonic: No angle types specified" << endl;
        throw runtime_error("Error initializing HarmonicAngleForceCompute");
        }

    // allocate the parameters
    m_K = new Scalar[m_angle_data->getNAngleTypes()];
    m_t_0 = new Scalar[m_angle_data->getNAngleTypes()];
    }

HarmonicAngleForceCompute::~HarmonicAngleForceCompute()
    {
    m_exec_conf->msg->notice(5) << "Destroying HarmonicAngleForceCompute" << endl;

    delete[] m_K;
    delete[] m_t_0;
    m_K = NULL;
    m_t_0 = NULL;
    }

/*! \param type Type of the angle to set parameters for
    \param K Stiffness parameter for the force computation
    \param t_0 Equilibrium angle in radians for the force computation

    Sets parameters for the potential of a particular angle type
*/
void HarmonicAngleForceCompute::setParams(unsigned int type, Scalar K, Scalar t_0)
    {
    // make sure the type is valid
    if (type >= m_angle_data->getNAngleTypes())
        {
        m_exec_conf->msg->error() << "angle.harmonic: Invalid angle type specified" << endl;
        throw runtime_error("Error setting parameters in HarmonicAngleForceCompute");
        }
        
    m_K[type] = K;
    m_t_0[type] = t_0;
    
    // check for some silly errors a user could make
    if (K <= 0)
        m_exec_conf->msg->warning() << "angle.harmonic: specified K <= 0" << endl;
    if (t_0 <= 0)
        m_exec_conf->msg->warning() << "angle.harmonic: specified t_0 <= 0" << endl;
    }

/*! AngleForceCompute provides
    - \c angle_harmonic_energy
*/
std::vector< std::string > HarmonicAngleForceCompute::getProvidedLogQuantities()
    {
    vector<string> list;
    list.push_back("angle_harmonic_energy");
    return list;
    }

/*! \param quantity Name of the quantity to get the log value of
    \param timestep Current time step of the simulation
*/
Scalar HarmonicAngleForceCompute::getLogValue(const std::string& quantity, unsigned int timestep)
    {
    if (quantity == string("angle_harmonic_energy"))
        {
        compute(timestep);
        return calcEnergySum();
        }
    else
        {
        m_exec_conf->msg->error() << "angle.harmonic: " << quantity << " is not a valid log quantity for AngleForceCompute" << endl;
        throw runtime_error("Error getting log value");
        }
    }

/*! Actually perform the force computation
    \param timestep Current time step
 */
void HarmonicAngleForceCompute::computeForces(unsigned int timestep)
    {
    if (m_prof) m_prof->push("Harmonic Angle");
    
    assert(m_pdata);
    // access the particle data arrays
    ArrayHandle<Scalar4> h_pos(m_pdata->getPositions(), access_location::host, access_mode::read);
    ArrayHandle<unsigned int> h_rtag(m_pdata->getRTags(), access_location::host, access_mode::read);

    ArrayHandle<Scalar4> h_force(m_force,access_location::host, access_mode::overwrite);
    ArrayHandle<Scalar> h_virial(m_virial,access_location::host, access_mode::overwrite);
    unsigned int virial_pitch = m_virial.getPitch();

    // there are enough other checks on the input data: but it doesn't hurt to be safe
    assert(h_force.data);
    assert(h_virial.data);
    assert(h_pos.data);
    assert(h_rtag.data);
    
    // Zero data for force calculation.
    memset((void*)h_force.data,0,sizeof(Scalar4)*m_force.getNumElements());
    memset((void*)h_virial.data,0,sizeof(Scalar)*m_virial.getNumElements());
    
    // get a local copy of the simulation box too
    const BoxDim& box = m_pdata->getBox();

    // for each of the angles
    const unsigned int size = (unsigned int)m_angle_data->getNumAngles();
    for (unsigned int i = 0; i < size; i++)
        {
        // lookup the tag of each of the particles participating in the angle
        const Angle& angle = m_angle_data->getAngle(i);
        assert(angle.a < m_pdata->getN());
        assert(angle.b < m_pdata->getN());
        assert(angle.c < m_pdata->getN());
        
        // transform a, b, and c into indicies into the particle data arrays
        // MEM TRANSFER: 6 ints
        unsigned int idx_a = h_rtag.data[angle.a];
        unsigned int idx_b = h_rtag.data[angle.b];
        unsigned int idx_c = h_rtag.data[angle.c];
        assert(idx_a < m_pdata->getN());
        assert(idx_b < m_pdata->getN());
        assert(idx_c < m_pdata->getN());
        
        // calculate d\vec{r}
        Scalar3 dab;
        dab.x = h_pos.data[idx_a].x - h_pos.data[idx_b].x;
        dab.y = h_pos.data[idx_a].y - h_pos.data[idx_b].y;
        dab.z = h_pos.data[idx_a].z - h_pos.data[idx_b].z;
        
        Scalar3 dcb;
        dcb.x = h_pos.data[idx_c].x - h_pos.data[idx_b].x;
        dcb.y = h_pos.data[idx_c].y - h_pos.data[idx_b].y;
        dcb.z = h_pos.data[idx_c].z - h_pos.data[idx_b].z;
        
        Scalar3 dac;
        dac.x = h_pos.data[idx_a].x - h_pos.data[idx_c].x; // used for the 1-3 JL interaction
        dac.y = h_pos.data[idx_a].y - h_pos.data[idx_c].y;
        dac.z = h_pos.data[idx_a].z - h_pos.data[idx_c].z;
        
        // apply minimum image conventions to all 3 vectors
        dab = box.minImage(dab);
        dcb = box.minImage(dcb);
        dac = box.minImage(dac);

        // on paper, the formula turns out to be: F = K*\vec{r} * (r_0/r - 1)
        // FLOPS: 14 / MEM TRANSFER: 2 Scalars
        
        
        // FLOPS: 42 / MEM TRANSFER: 6 Scalars
        Scalar rsqab = dab.x*dab.x+dab.y*dab.y+dab.z*dab.z;
        Scalar rab = sqrt(rsqab);
        Scalar rsqcb = dcb.x*dcb.x+dcb.y*dcb.y+dcb.z*dcb.z;
        Scalar rcb = sqrt(rsqcb);
        
        Scalar c_abbc = dab.x*dcb.x+dab.y*dcb.y+dab.z*dcb.z;
        c_abbc /= rab*rcb;
        
        if (c_abbc > 1.0) c_abbc = 1.0;
        if (c_abbc < -1.0) c_abbc = -1.0;
        
        Scalar s_abbc = sqrt(1.0 - c_abbc*c_abbc);
        if (s_abbc < SMALL) s_abbc = SMALL;
        s_abbc = 1.0/s_abbc;
        
        // actually calculate the force
        Scalar dth = acos(c_abbc) - m_t_0[angle.type];
        Scalar tk = m_K[angle.type]*dth;
        
        Scalar a = -1.0 * tk * s_abbc;
        Scalar a11 = a*c_abbc/rsqab;
        Scalar a12 = -a / (rab*rcb);
        Scalar a22 = a*c_abbc / rsqcb;
        
        Scalar fab[3], fcb[3];
        
        fab[0] = a11*dab.x + a12*dcb.x;
        fab[1] = a11*dab.y + a12*dcb.y;
        fab[2] = a11*dab.z + a12*dcb.z;
        
        fcb[0] = a22*dcb.x + a12*dab.x;
        fcb[1] = a22*dcb.y + a12*dab.y;
        fcb[2] = a22*dcb.z + a12*dab.z;
        
        // compute 1/3 of the energy, 1/3 for each atom in the angle
        Scalar angle_eng = (tk*dth)*Scalar(1.0/6.0);
        
        // compute 1/3 of the virial, 1/3 for each atom in the angle
        // upper triangular version of virial tensor
        Scalar angle_virial[6];
        angle_virial[0] = Scalar(1./3.) * ( dab.x*fab[0] + dcb.x*fcb[0] );
        angle_virial[1] = Scalar(1./3.) * ( dab.y*fab[0] + dcb.y*fcb[0] );
        angle_virial[2] = Scalar(1./3.) * ( dab.z*fab[0] + dcb.z*fcb[0] );
        angle_virial[3] = Scalar(1./3.) * ( dab.y*fab[1] + dcb.y*fcb[1] );
        angle_virial[4] = Scalar(1./3.) * ( dab.z*fab[1] + dcb.z*fcb[1] );
        angle_virial[5] = Scalar(1./3.) * ( dab.z*fab[2] + dcb.z*fcb[2] );
        
        // Now, apply the force to each individual atom a,b,c, and accumlate the energy/virial
        h_force.data[idx_a].x += fab[0];
        h_force.data[idx_a].y += fab[1];
        h_force.data[idx_a].z += fab[2];
        h_force.data[idx_a].w += angle_eng;
        for (int j = 0; j < 6; j++)
            h_virial.data[j*virial_pitch+idx_a]  += angle_virial[j];
        
        h_force.data[idx_b].x -= fab[0] + fcb[0];
        h_force.data[idx_b].y -= fab[1] + fcb[1];
        h_force.data[idx_b].z -= fab[2] + fcb[2];
        h_force.data[idx_b].w += angle_eng;
        for (int j = 0; j < 6; j++)
            h_virial.data[j*virial_pitch+idx_b]  += angle_virial[j];
        
        h_force.data[idx_c].x += fcb[0];
        h_force.data[idx_c].y += fcb[1];
        h_force.data[idx_c].z += fcb[2];
        h_force.data[idx_c].w += angle_eng;
        for (int j = 0; j < 6; j++)
            h_virial.data[j*virial_pitch+idx_c]  += angle_virial[j];
        }
        
    if (m_prof) m_prof->pop();
    }

void export_HarmonicAngleForceCompute()
    {
    class_<HarmonicAngleForceCompute, boost::shared_ptr<HarmonicAngleForceCompute>, bases<ForceCompute>, boost::noncopyable >
    ("HarmonicAngleForceCompute", init< boost::shared_ptr<SystemDefinition> >())
    .def("setParams", &HarmonicAngleForceCompute::setParams)
    ;
    }

#ifdef WIN32
#pragma warning( pop )
#endif

