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

/*! \file SystemDefinition.cc
    \brief Defines SystemDefinition
*/

#ifdef WIN32
#pragma warning( push )
#pragma warning( disable : 4103 4244 4267)
#endif

#include "SystemDefinition.h"

#include "SnapshotSystemData.h"

#ifdef ENABLE_MPI
#include "Communicator.h"
#endif

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
     - ParticleData constructed with the arguments \a N, \a box, \a n_types, and \a exec_conf->
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
                                   boost::shared_ptr<ExecutionConfiguration> exec_conf)
    {
    m_n_dimensions = 3;
    m_particle_data = boost::shared_ptr<ParticleData>(new ParticleData(N, box, n_types, exec_conf));
    m_bond_data = boost::shared_ptr<BondData>(new BondData(m_particle_data, n_bond_types));
    m_wall_data = boost::shared_ptr<WallData>(new WallData());
    
    m_rigid_data = boost::shared_ptr<RigidData>(new RigidData(m_particle_data));
    m_rigid_data->initializeData();

    m_angle_data = boost::shared_ptr<AngleData>(new AngleData(m_particle_data, n_angle_types));
    m_dihedral_data = boost::shared_ptr<DihedralData>(new DihedralData(m_particle_data, n_dihedral_types));
    m_improper_data = boost::shared_ptr<DihedralData>(new DihedralData(m_particle_data, n_improper_types));
    m_integrator_data = boost::shared_ptr<IntegratorData>(new IntegratorData());
    }

/*! Evaluates the snapshot and initializes the respective *Data classes using
   its contents (box dimensions and sub-snapshots) 
    \param snapshot Snapshot to use
    \param exec_conf Execution configuration to run on
    \param decomposition (optional) The domain decomposition layout
*/
SystemDefinition::SystemDefinition(boost::shared_ptr<const SnapshotSystemData> snapshot,
                                   boost::shared_ptr<ExecutionConfiguration> exec_conf
#ifdef ENABLE_MPI
                                   , boost::shared_ptr<DomainDecomposition> decomposition
#endif
                                   )
    {
    m_n_dimensions = snapshot->dimensions;

    m_particle_data = boost::shared_ptr<ParticleData>(new ParticleData(snapshot->particle_data,
                 snapshot->global_box,
                 exec_conf
#ifdef ENABLE_MPI
                 , decomposition
#endif
                 ));
 
    m_bond_data = boost::shared_ptr<BondData>(new BondData(m_particle_data, snapshot->bond_data));
   
    m_wall_data = boost::shared_ptr<WallData>(new WallData(snapshot->wall_data));
    
    m_rigid_data = boost::shared_ptr<RigidData>(new RigidData(m_particle_data));
    
    // the follwing calls may cause confusion, but their tasks are completely different
    // This makes sure that the rigid bodies are initialized correctly based on the particle data (body flags)
    // It computes relevant static data, i.e. body mass, body size, inertia of momentia, particle pos, and particle indices.
    m_rigid_data->initializeData();
    
    // If the initializer is from a binary file, then this reads in the body COM, velocities, angular momenta and body images; 
    // otherwise, nothing is done here.
    if (snapshot->rigid_data.size) m_rigid_data->initializeFromSnapshot(snapshot->rigid_data);
        
    m_angle_data = boost::shared_ptr<AngleData>(new AngleData(m_particle_data, snapshot->angle_data));
    
    m_dihedral_data = boost::shared_ptr<DihedralData>(new DihedralData(m_particle_data, snapshot->dihedral_data));
    
    m_improper_data = boost::shared_ptr<DihedralData>(new DihedralData(m_particle_data, snapshot->improper_data));

    m_integrator_data = boost::shared_ptr<IntegratorData>(new IntegratorData(snapshot->integrator_data));
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
        m_particle_data->getExecConf()->msg->error() << "hoomd supports only 2D or 3D simulations" << endl;
        throw runtime_error("Error setting dimensions");
        }
    m_n_dimensions = n_dimensions;
    }

/*! \param particles True if particle data should be saved
 *  \param bonds True if bond data should be saved
 *  \param angles True if angle data should be saved
 *  \param dihedrals True if dihedral data should be saved
 *  \param impropers True if improper data should be saved
 *  \param rigid True if rigid data should be saved
 *  \param wall True if wall data should be saved
 *  \param integrator True if integrator data should be saved
 */
boost::shared_ptr<SnapshotSystemData> SystemDefinition::takeSnapshot(bool particles,
                                                   bool bonds,
                                                   bool angles,
                                                   bool dihedrals,
                                                   bool impropers,
                                                   bool rigid,
                                                   bool walls,
                                                   bool integrators)
    {
    boost::shared_ptr<SnapshotSystemData> snap(new SnapshotSystemData);

    // always save dimensions and global box
    snap->dimensions = m_n_dimensions;
    snap->global_box = m_particle_data->getGlobalBox();

    if (particles)
        {
        m_particle_data->takeSnapshot(snap->particle_data);
        snap->has_particle_data = true;
        }
    else
        snap->has_particle_data = false;

    if (bonds)
        {
        m_bond_data->takeSnapshot(snap->bond_data);
        snap->has_bond_data = true;
        }
    else
        snap->has_bond_data = false;

    if (angles)
        {
        m_angle_data->takeSnapshot(snap->angle_data);
        snap->has_angle_data = true;
        }
    else
        snap->has_angle_data = false;

    if (dihedrals)
        {
        m_dihedral_data->takeSnapshot(snap->dihedral_data);
        snap->has_dihedral_data = true;
        }
    else
        snap->has_dihedral_data = false;

    if (impropers)
        {
        m_improper_data->takeSnapshot(snap->improper_data);
        snap->has_improper_data = true;
        }
    else
        snap->has_improper_data = false;

    if (rigid)
        {
        m_rigid_data->takeSnapshot(snap->rigid_data);
        snap->has_rigid_data = true;
        }
    else
        snap->has_rigid_data = false;

    if (walls)
        {
        for (unsigned int i = 0; i < m_wall_data->getNumWalls(); ++i)
            snap->wall_data.push_back(m_wall_data->getWall(i));
        snap->has_wall_data = true;
        }
    else
        snap->has_wall_data = false;

    if (integrators)
        {
        for (unsigned int i = 0; i < m_integrator_data->getNumIntegrators(); ++i)
            snap->integrator_data.push_back(m_integrator_data->getIntegratorVariables(i));
        }
    else
        snap->has_integrator_data = false;

    return snap;
    }

//! Re-initialize the system from a snapshot
void SystemDefinition::initializeFromSnapshot(boost::shared_ptr<SnapshotSystemData> snapshot)
    {
    m_n_dimensions = snapshot->dimensions;

    boost::shared_ptr<const ExecutionConfiguration> exec_conf = m_particle_data->getExecConf();

    if (snapshot->has_particle_data)
        {
        m_particle_data->setGlobalBox(snapshot->global_box);
    
        m_particle_data->initializeFromSnapshot(snapshot->particle_data);
        }
   
    if (snapshot->has_bond_data)
        m_bond_data->initializeFromSnapshot(snapshot->bond_data);

    if (snapshot->has_angle_data)
        m_angle_data->initializeFromSnapshot(snapshot->angle_data);

    if (snapshot->has_dihedral_data)
        m_dihedral_data->initializeFromSnapshot(snapshot->dihedral_data);

    if (snapshot->has_improper_data)
        m_improper_data->initializeFromSnapshot(snapshot->improper_data);

    if (snapshot->has_rigid_data)
        m_rigid_data->initializeFromSnapshot(snapshot->rigid_data);

    if (snapshot->has_wall_data)
        {
        m_wall_data->removeAllWalls();
        for (unsigned int i = 0; i < snapshot->wall_data.size(); ++i)
            m_wall_data->addWall(snapshot->wall_data[i]);
        }

    // it is an error to load variables for more integrators than are
    // currently registered
    if (snapshot->has_integrator_data)
        {
        unsigned int n_integrators = m_integrator_data->getNumIntegrators();
        if (n_integrators != snapshot->integrator_data.size())
            {
            exec_conf->msg->error() << "init.restart_from_snapshot: Snapshot contains data for "
                                    << snapshot->integrator_data.size() << " integrators," << std::endl
                                    << "but " << n_integrators << " are currently registered."
                                    << std::endl << std::endl;
            throw std::runtime_error("Error initializing from snapshot");
            }

        for (unsigned int i = 0; i < n_integrators; ++i)
            m_integrator_data->setIntegratorVariables(i, snapshot->integrator_data[i]);
        }
    }

void export_SystemDefinition()
    {
    class_<SystemDefinition, boost::shared_ptr<SystemDefinition> >("SystemDefinition", init<>())
    .def(init<unsigned int, const BoxDim&, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, boost::shared_ptr<ExecutionConfiguration> >())
    #ifdef ENABLE_MPI
    .def(init<boost::shared_ptr<const SnapshotSystemData>, boost::shared_ptr<ExecutionConfiguration>, boost::shared_ptr<DomainDecomposition> >())
    #else
    .def(init<boost::shared_ptr<const SnapshotSystemData>, boost::shared_ptr<ExecutionConfiguration> >())
    #endif
    .def("setNDimensions", &SystemDefinition::setNDimensions)
    .def("getNDimensions", &SystemDefinition::getNDimensions)
    .def("getParticleData", &SystemDefinition::getParticleData)
    .def("getBondData", &SystemDefinition::getBondData)
    .def("getAngleData", &SystemDefinition::getAngleData)
    .def("getDihedralData", &SystemDefinition::getDihedralData)
    .def("getImproperData", &SystemDefinition::getImproperData)
    .def("getWallData", &SystemDefinition::getWallData)
    .def("getIntegratorData", &SystemDefinition::getIntegratorData)
    .def("getRigidData", &SystemDefinition::getRigidData)
    .def("getPDataRefs", &SystemDefinition::getPDataRefs)
    .def("takeSnapshot", &SystemDefinition::takeSnapshot)
    .def("initializeFromSnapshot", &SystemDefinition::initializeFromSnapshot)
    ;
    }

#ifdef WIN32
#pragma warning( pop )
#endif

