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

// $Id$
// $URL$
// Maintainer: joaander

#ifdef WIN32
#pragma warning( push )
#pragma warning( disable : 4103 4244 4267)
#endif

#include "SystemDefinition.h"

#include <boost/python.hpp>
using namespace boost::python;

/*! \post All shared pointers contained in SystemDefinition are NULL
*/
SystemDefinition::SystemDefinition()
    {
    }

/*! \param N Number of particles to allocate
    \param box Initial box particles are in
    \param n_types Number of particle types to set
    \param n_bond_types Number of bond types to create
    \param n_angle_types Number of angle types to create
    \param n_dihedral_types Number of diehdral types to create
    \param n_improper_types Number of improper types to create
    \param exec_conf The ExecutionConfiguration HOOMD is to be run on

    Creating SystemDefinition with this constructor results in
     - ParticleData constructed with the arguments \a N, \a box, \a n_types, and \a exec_conf.
     - BondData constructed with the arguments \a n_bond_types
     - All other data structures are default constructed.
*/
SystemDefinition::SystemDefinition(unsigned int N,
                                   const BoxDim &box,
                                   unsigned int n_types,
                                   unsigned int n_bond_types,
                                   unsigned int n_angle_types,
                                   unsigned int n_dihedral_types,
                                   unsigned int n_improper_types,
                                   const ExecutionConfiguration& exec_conf)
    {
    m_n_dimensions = 3;
    m_particle_data = boost::shared_ptr<ParticleData>(new ParticleData(N, box, n_types, exec_conf));
    m_bond_data = boost::shared_ptr<BondData>(new BondData(m_particle_data, n_bond_types));
    m_wall_data = boost::shared_ptr<WallData>(new WallData());
    
    m_angle_data = boost::shared_ptr<AngleData>(new AngleData(m_particle_data, n_angle_types));
    m_dihedral_data = boost::shared_ptr<DihedralData>(new DihedralData(m_particle_data, n_dihedral_types));
    m_improper_data = boost::shared_ptr<DihedralData>(new DihedralData(m_particle_data, n_improper_types));
    m_integrator_data = boost::shared_ptr<IntegratorData>(new IntegratorData());
    }

/*! Calls the initializer's members to determine the number of particles, box size and then
    uses it to fill out the position and velocity data.
    \param init Initializer to use
    \param exec_conf Execution configuration to run on

    \b TEMPORARY!!!!! Initializers are planned to be rewritten
*/
SystemDefinition::SystemDefinition(const ParticleDataInitializer& init, const ExecutionConfiguration& exec_conf)
    {
    m_n_dimensions = init.getNumDimensions();
    
    m_particle_data = boost::shared_ptr<ParticleData>(new ParticleData(init, exec_conf));
    
    m_bond_data = boost::shared_ptr<BondData>(new BondData(m_particle_data, init.getNumBondTypes()));
    init.initBondData(m_bond_data);
    
    m_wall_data = boost::shared_ptr<WallData>(new WallData());
    init.initWallData(m_wall_data);
    
    m_angle_data = boost::shared_ptr<AngleData>(new AngleData(m_particle_data, init.getNumAngleTypes()));
    init.initAngleData(m_angle_data);
    
    m_dihedral_data = boost::shared_ptr<DihedralData>(new DihedralData(m_particle_data, init.getNumDihedralTypes()));
    init.initDihedralData(m_dihedral_data);
    
    m_improper_data = boost::shared_ptr<DihedralData>(new DihedralData(m_particle_data, init.getNumImproperTypes()));
    init.initImproperData(m_improper_data);

    m_integrator_data = boost::shared_ptr<IntegratorData>(new IntegratorData());
    init.initIntegratorData(m_integrator_data);
    }

/*! Sets the dimensionality of the system.  When quantities involving the dof of 
    the system are computed, such as T, P, etc., the dimensionality is needed.
    Therefore, the dimensionality must be set before any temperature/pressure 
    computes, thermostats/barostats, etc. are added to the system.
    \param n_dimensions Number of dimensions
*/
void SystemDefinition::setNDimensions(unsigned int n_dimensions)
    {
    if (!(n_dimensions == 2 || n_dimensions == 3))
        {
        cerr << endl << "***Error! hoomd supports only 2D or 3D simulations" << endl << endl;
        throw runtime_error("Error setting dimensions");        
        }
    m_n_dimensions = n_dimensions;
    }


void export_SystemDefinition()
    {
    class_<SystemDefinition, boost::shared_ptr<SystemDefinition> >("SystemDefinition", init<>())
    .def(init<unsigned int, const BoxDim&, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, const ExecutionConfiguration&>())
    .def(init<unsigned int, const BoxDim&, unsigned int>())
    .def(init<const ParticleDataInitializer&, const ExecutionConfiguration&>())
    .def("setNDimensions", &SystemDefinition::setNDimensions)
    .def("getNDimensions", &SystemDefinition::getNDimensions)
    .def("getParticleData", &SystemDefinition::getParticleData)
    .def("getBondData", &SystemDefinition::getBondData)
    .def("getAngleData", &SystemDefinition::getAngleData)
    .def("getDihedralData", &SystemDefinition::getDihedralData)
    .def("getImproperData", &SystemDefinition::getImproperData)
    .def("getWallData", &SystemDefinition::getWallData)
    .def("getIntegratorData", &SystemDefinition::getIntegratorData)
    ;
    }

#ifdef WIN32
#pragma warning( pop )
#endif

