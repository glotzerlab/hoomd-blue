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

/*! \file SystemDefinition.h
    \brief Defines the SystemDefinition class
 */

#ifdef NVCC
#error This header cannot be compiled by nvcc
#endif

#include "ParticleData.h"
#include "BondData.h"
#include "WallData.h"
#include "RigidData.h"
#include "AngleData.h"
#include "DihedralData.h"
#include "IntegratorData.h"

#include <boost/shared_ptr.hpp>

#ifndef __SYSTEM_DEFINITION_H__
#define __SYSTEM_DEFINITION_H__

#ifdef ENABLE_MPI
//! Forward declaration of Communicator
class Communicator;
#endif

//! Forward declaration of SnapshotSytemData
class SnapshotSystemData;

//! Container class for all data needed to define the MD system
/*! SystemDefinition is a big bucket where all of the data defining the MD system goes.
    Everything is stored as a shared pointer for quick and easy access from within C++
    and python without worrying about data management.

    <b>Background and intended usage</b>

    The most fundamental data structure stored in SystemDefinition is the ParticleData.
    It stores essential data on a per particle basis (position, velocity, type, mass, etc...)
    as well as defining the number of particles in the system and the simulation box. Many other
    data structures in SystemDefinition also refer to particles and store other data related to
    them (i.e. BondData lists bonds between particles). These will need access to information such
    as the number of particles in the system or potentially some of the per-particle data stored
    in ParticleData. To facilitate this, ParticleData will always be initialized \b fist and its
    shared pointer can then be passed to any future data structure in SystemDefinition that needs
    such a reference.

    More generally, any data structure class in SystemDefinition can potentially reference any other,
    simply by giving the shared pointer to the referenced class to the constructor of the one that
    needs to refer to it. Note that using this setup, there can be no circular references. This is a
    \b good \b thing ^TM, as it promotes good separation and isolation of the various classes responsibilities.

    In rare circumstances, a references back really is required (i.e. notification of referring classes when
    ParticleData resorts particles). Any event based notifications of such should be managed with boost::signals.
    Any ongoing references where two data structure classes are so interwoven that they must constantly refer to
    each other should be avoided (consider merging them into one class).

    <b>Initializing</b>

    A default constructed SystemDefinition is full of NULL shared pointers. Such is intended to be assigned to
    by one created by a SystemInitializer.

    Several other default constructors are provided, mainly to provide backward compatibility to unit tests that
    relied on the simple initialization constructors provided by ParticleData.

    \ingroup data_structs
*/
class SystemDefinition
    {
    public:
        //! Constructs a NULL SystemDefinition
        SystemDefinition();
        //! Conctructs a SystemDefinition with a simply initialized ParticleData
        SystemDefinition(unsigned int N,
                         const BoxDim &box,
                         unsigned int n_types=1,
                         unsigned int n_bond_types=0,
                         unsigned int n_angle_types=0,
                         unsigned int n_dihedral_types=0,
                         unsigned int n_improper_types=0,
                         boost::shared_ptr<ExecutionConfiguration> exec_conf=boost::shared_ptr<ExecutionConfiguration>(new ExecutionConfiguration()));
                         
        //! Construct from a snapshot
        SystemDefinition(boost::shared_ptr<const SnapshotSystemData> snapshot,
                         boost::shared_ptr<ExecutionConfiguration> exec_conf=boost::shared_ptr<ExecutionConfiguration>(new ExecutionConfiguration())
#ifdef ENABLE_MPI
                        , boost::shared_ptr<DomainDecomposition> decomposition=boost::shared_ptr<DomainDecomposition>()
#endif
                        );
                        
        //! Set the dimensionality of the system
        void setNDimensions(unsigned int);
        
        //! Get the dimensionality of the system
        unsigned int getNDimensions() const
            {
            return m_n_dimensions;
            }
        //! Get the particle data
        boost::shared_ptr<ParticleData> getParticleData() const
            {
            return m_particle_data;
            }
        //! Get the bond data
        boost::shared_ptr<BondData> getBondData() const
            {
            return m_bond_data;
            }
        //! Get the wall data
        boost::shared_ptr<WallData> getWallData() const
            {
            return m_wall_data;
            }
        //! Get the rigid body data
        boost::shared_ptr<RigidData> getRigidData() const
            {
            return m_rigid_data;
            }
        //! Access the angle data defined for the simulation
        boost::shared_ptr<AngleData> getAngleData()
            {
            return m_angle_data;
            }
        //! Access the dihedral data defined for the simulation
        boost::shared_ptr<DihedralData> getDihedralData()
            {
            return m_dihedral_data;
            }
        //! Access the improper data defined for the simulation
        boost::shared_ptr<DihedralData> getImproperData()
            {
            return m_improper_data;
            }

        //! Returns the integrator variables (if applicable)
        boost::shared_ptr<IntegratorData> getIntegratorData() 
            {
            return m_integrator_data;
            }

        //! Helper for python memory managment in init.reset
        long getPDataRefs()
            {
            return m_particle_data.use_count();
            }
           
        //! Return a snapshot of the current system data
        boost::shared_ptr<SnapshotSystemData> takeSnapshot(bool particles,
                                                           bool bonds,
                                                           bool angles,
                                                           bool dihedrals,
                                                           bool impropers,
                                                           bool rigid,
                                                           bool walls,
                                                           bool integrators);

        //! Re-initialize the system from a snapshot
        void initializeFromSnapshot(boost::shared_ptr<SnapshotSystemData> snapshot);

    private:
        unsigned int m_n_dimensions;                        //!< Dimensionality of the system
        boost::shared_ptr<ParticleData> m_particle_data;    //!< Particle data for the system
        boost::shared_ptr<BondData> m_bond_data;            //!< Bond data for the system
        boost::shared_ptr<WallData> m_wall_data;            //!< Wall data for the system
        boost::shared_ptr<RigidData> m_rigid_data;          //!< Rigid bodies data for the system
        boost::shared_ptr<AngleData> m_angle_data;          //!< Angle data for the system
        boost::shared_ptr<DihedralData> m_dihedral_data;    //!< Dihedral data for the system
        boost::shared_ptr<DihedralData> m_improper_data;    //!< Improper data for the system
        boost::shared_ptr<IntegratorData> m_integrator_data;    //!< Integrator data for the system
    };

//! Exports SystemDefinition to python
void export_SystemDefinition();

#endif

