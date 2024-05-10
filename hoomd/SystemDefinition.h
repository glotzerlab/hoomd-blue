// Copyright (c) 2009-2024 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

/*! \file SystemDefinition.h
    \brief Defines the SystemDefinition class
 */

#ifdef __HIPCC__
#error This header cannot be compiled by nvcc
#endif

#include "BondedGroupData.h"
#include "ParticleData.h"
#ifdef BUILD_MPCD
#include "hoomd/mpcd/ParticleData.h"
#endif

#include <memory>
#include <pybind11/pybind11.h>

#ifndef __SYSTEM_DEFINITION_H__
#define __SYSTEM_DEFINITION_H__

namespace hoomd
    {
#ifdef ENABLE_MPI
//! Forward declaration of Communicator
class Communicator;
#endif

//! Forward declaration of SnapshotSystemData
template<class Real> struct SnapshotSystemData;

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

    More generally, any data structure class in SystemDefinition can potentially reference any
    other, simply by giving the shared pointer to the referenced class to the constructor of the one
   that needs to refer to it. Note that using this setup, there can be no circular references. This
   is a \b good \b thing ^TM, as it promotes good separation and isolation of the various classes
   responsibilities.

    In rare circumstances, a references back really is required (i.e. notification of referring
   classes when ParticleData resorts particles). Any event based notifications of such should be
   managed with Nano::Signal. Any ongoing references where two data structure classes are so
   interwoven that they must constantly refer to each other should be avoided (consider merging them
   into one class).

    <b>Initializing</b>

    A default constructed SystemDefinition is full of NULL shared pointers. Such is intended to be
   assigned to by one created by a SystemInitializer.

    Several other default constructors are provided, mainly to provide backward compatibility to
   unit tests that relied on the simple initialization constructors provided by ParticleData.

    \ingroup data_structs
*/
class PYBIND11_EXPORT SystemDefinition
    {
    public:
    //! Constructs a NULL SystemDefinition
    SystemDefinition();
    //! Constructs a SystemDefinition with a simply initialized ParticleData
    SystemDefinition(unsigned int N,
                     const std::shared_ptr<BoxDim> box,
                     unsigned int n_types = 1,
                     unsigned int n_bond_types = 0,
                     unsigned int n_angle_types = 0,
                     unsigned int n_dihedral_types = 0,
                     unsigned int n_improper_types = 0,
                     std::shared_ptr<ExecutionConfiguration> exec_conf
                     = std::shared_ptr<ExecutionConfiguration>(new ExecutionConfiguration()),
                     std::shared_ptr<DomainDecomposition> decomposition
                     = std::shared_ptr<DomainDecomposition>());

    // Mostly exists as test pass a plain box rather than a std::shared_ptr.
    //! Constructs a SystemDefinition with a simply initialized ParticleData
    SystemDefinition(unsigned int N,
                     const BoxDim& box,
                     unsigned int n_types = 1,
                     unsigned int n_bond_types = 0,
                     unsigned int n_angle_types = 0,
                     unsigned int n_dihedral_types = 0,
                     unsigned int n_improper_types = 0,
                     std::shared_ptr<ExecutionConfiguration> exec_conf
                     = std::shared_ptr<ExecutionConfiguration>(new ExecutionConfiguration()),
                     std::shared_ptr<DomainDecomposition> decomposition
                     = std::shared_ptr<DomainDecomposition>());

    //! Construct from a snapshot
    template<class Real>
    SystemDefinition(std::shared_ptr<SnapshotSystemData<Real>> snapshot,
                     std::shared_ptr<ExecutionConfiguration> exec_conf
                     = std::shared_ptr<ExecutionConfiguration>(new ExecutionConfiguration()),
                     std::shared_ptr<DomainDecomposition> decomposition
                     = std::shared_ptr<DomainDecomposition>());

    //! Set the dimensionality of the system
    void setNDimensions(unsigned int);

    //! Get the dimensionality of the system
    unsigned int getNDimensions() const
        {
        return m_n_dimensions;
        }

    /// Check if the system is decomposed across MPI ranks
    bool isDomainDecomposed()
        {
#ifdef ENABLE_MPI
        return bool(this->m_particle_data->getDomainDecomposition());
#else
        return false;
#endif
        }

    /// Set the random numbers seed
    void setSeed(uint16_t seed)
        {
        m_seed = seed;

#ifdef ENABLE_MPI
        // In case of MPI run, every rank should be initialized with the same seed.
        // Broadcast the seed of rank 0 to all ranks to correct cases where the user provides
        // different seeds

        if (isDomainDecomposed())
            bcast(m_seed, 0, this->m_particle_data->getExecConf()->getMPICommunicator());
#endif
        }

#ifdef ENABLE_MPI
    void setCommunicator(std::shared_ptr<Communicator> communicator)
        {
        // Communicator holds a shared pointer to the SystemDefinition, so hold a weak pointer
        // to break the circular reference.
        m_communicator = communicator;
        }

    std::weak_ptr<Communicator> getCommunicator()
        {
        return m_communicator;
        }
#endif

    /// Get the random number seed
    uint16_t getSeed() const
        {
        return m_seed;
        }

    //! Get the particle data
    std::shared_ptr<ParticleData> getParticleData() const
        {
        return m_particle_data;
        }
    //! Get the bond data
    std::shared_ptr<BondData> getBondData() const
        {
        return m_bond_data;
        }
    //! Access the angle data defined for the simulation
    std::shared_ptr<AngleData> getAngleData()
        {
        return m_angle_data;
        }
    //! Access the dihedral data defined for the simulation
    std::shared_ptr<DihedralData> getDihedralData()
        {
        return m_dihedral_data;
        }
    //! Access the improper data defined for the simulation
    std::shared_ptr<ImproperData> getImproperData()
        {
        return m_improper_data;
        }

    //! Access the constraint data defined for the simulation
    std::shared_ptr<ConstraintData> getConstraintData()
        {
        return m_constraint_data;
        }

    //! Get the pair data
    std::shared_ptr<PairData> getPairData() const
        {
        return m_pair_data;
        }

#ifdef BUILD_MPCD
    //! Get the MPCD particle data
    std::shared_ptr<mpcd::ParticleData> getMPCDParticleData() const
        {
        return m_mpcd_data;
        }

    //! Set the MPCD particle data
    void setMPCDParticleData(std::shared_ptr<mpcd::ParticleData> mpcd_data)
        {
        m_mpcd_data = mpcd_data;
        }
#endif

    //! Return a snapshot of the current system data
    template<class Real> std::shared_ptr<SnapshotSystemData<Real>> takeSnapshot();

    //! Re-initialize the system from a snapshot
    template<class Real>
    void initializeFromSnapshot(std::shared_ptr<SnapshotSystemData<Real>> snapshot);

    private:
    unsigned int m_n_dimensions;                       //!< Dimensionality of the system
    uint16_t m_seed = 0;                               //!< Random number seed
    std::shared_ptr<ParticleData> m_particle_data;     //!< Particle data for the system
    std::shared_ptr<BondData> m_bond_data;             //!< Bond data for the system
    std::shared_ptr<AngleData> m_angle_data;           //!< Angle data for the system
    std::shared_ptr<DihedralData> m_dihedral_data;     //!< Dihedral data for the system
    std::shared_ptr<ImproperData> m_improper_data;     //!< Improper data for the system
    std::shared_ptr<ConstraintData> m_constraint_data; //!< Improper data for the system
    std::shared_ptr<PairData> m_pair_data;             //!< Special pairs data for the system
#ifdef BUILD_MPCD
    std::shared_ptr<mpcd::ParticleData> m_mpcd_data; //!< MPCD particle data
#endif

#ifdef ENABLE_MPI
    /// The system communicator
    std::weak_ptr<Communicator> m_communicator;
#endif
    };

namespace detail
    {
//! Exports SystemDefinition to python
void export_SystemDefinition(pybind11::module& m);

    } // end namespace detail

    } // end namespace hoomd

#endif
