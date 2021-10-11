// Copyright (c) 2009-2021 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.

// Maintainer: joaander

/*! \file SystemDefinition.h
    \brief Defines the SystemDefinition class
 */

#ifdef __HIPCC__
#error This header cannot be compiled by nvcc
#endif

#include "BondedGroupData.h"
#include "MeshGroupData.h"
#include "IntegratorData.h"
#include "ParticleData.h"

#include <memory>
#include <pybind11/pybind11.h>

#ifndef __SYSTEM_DEFINITION_H__
#define __SYSTEM_DEFINITION_H__

#ifdef ENABLE_MPI
//! Forward declaration of Communicator
class Communicator;
#endif

//! Container class for all data needed to define the MD system
/*! MeshDefinition is a big bucket where all of the data defining the MD system goes.
    Everything is stored as a shared pointer for quick and easy access from within C++
    and python without worrying about data management.

    <b>Background and intended usage</b>

    The data structure stored in MeshDefinition is all the MeshData.These will need access to
    information such as the number of particles in the system or potentially some of the
    per-particle data. To facilitate this,

    More generally, any data structure class in MeshDefinition can potentially reference any
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

    A default constructed MeshDefinition is full of NULL shared pointers. Such is intended to be
   assigned to by one created by a SystemInitializer.

    Several other default constructors are provided, mainly to provide backward compatibility to
   unit tests that relied on the simple initialization constructors provided by ParticleData.

    \ingroup data_structs
*/
class PYBIND11_EXPORT MeshDefinition
    {
    public:
    //! Constructs a NULL MeshDefinition
    MeshDefinition();
    //! Constructs a MeshDefinition with a simply initialized ParticleData
    MeshDefinition(std::shared_ptr<ParticleData> pdata,
                     unsigned int n_triangle_types);

    //! Construct from a snapshot
    MeshDefinition( std::shared_ptr<ParticleData> pdata,
		    TriangleData::Snapshot snapshot);

    std::shared_ptr<TriangleData> getTriangleData();

    void setTriangleData();

    //! Access the mesh triangle data defined for the simulation
    std::shared_ptr<MeshTriangleData> getMeshTriangleData()
        {
        return m_meshtriangle_data;
        }
    //! Access the mesh bond data defined for the simulation
    std::shared_ptr<MeshBondData> getMeshBondData()
        {
        return m_meshbond_data;
        }

    //! Return a snapshot of the current system data
     TriangleData::Snapshot takeSnapshot();

    //! Re-initialize the system from a snapshot
    void initializeFromSnapshot(TriangleData::Snapshot snapshot);

    private:
    std::shared_ptr<TriangleData> m_triangle_data;           //!< Triangle data for the system
    std::shared_ptr<MeshBondData> m_meshbond_data;     //!< Bond data for the mesh
    std::shared_ptr<MeshTriangleData> m_meshtriangle_data; //!< Triangle data for the mesh
    Scalar m_mesh_energy;
    Scalar m_mesh_energy_old;
    bool m_triangle_change;
    bool m_mesh_change;
    };

//! Exports MeshDefinition to python
void export_MeshDefinition(pybind11::module& m);

#endif
