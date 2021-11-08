/*! \file MeshDefinition.h
    \brief Defines the MeshDefinition class
 */

#ifdef __HIPCC__
#error This header cannot be compiled by nvcc
#endif

#include "BondedGroupData.h"
#include "IntegratorData.h"
#include "MeshGroupData.h"
#include "SystemDefinition.h"

#include <memory>
#include <pybind11/pybind11.h>

#ifndef __MESH_DEFINITION_H__
#define __MESH_DEFINITION_H__

namespace hoomd
    {
#ifdef ENABLE_MPI
//! Forward declaration of Communicator
class Communicator;
#endif

//! Container class for all data needed to define a mesh system
/*! MeshDefinition is a big bucket where all of the data for a mesh structure goes.
    Everything is stored as a shared pointer for quick and easy access from within C++
    and python without worrying about data management.

    <b>Background and intended usage</b>

    The data structure stored in MeshDefinition is all the bonds and triangles within a mesh.
    These will need access to information such as the number of particles in the system or
    potentially some of the per-particle data to allow for dynamic bonding and meshes.

    More generally, any data structure class in MeshDefinition can potentially reference any
   other, simply by giving the shared pointer to the referenced class to the constructor of the one
   that needs to refer to it. Note that using this setup, there can be no circular references. This
   is a \b good \b thing ^TM, as it promotes good separation and isolation of the various classes
   responsibilities.

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
    MeshDefinition(std::shared_ptr<SystemDefinition> sysdef);

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

    BondData::Snapshot getBondData();

    void updateTriangleData();

    void updateMeshData();

    TriangleData::Snapshot triangle_data;             //!< The triangle data accessible in python
    Scalar m_mesh_energy;         //!< storing energy for dynamic bonding later

    private:
    std::shared_ptr<SystemDefinition>
        m_sysdef; //!< System definition later needed for dynamic bonding
    std::shared_ptr<MeshBondData> m_meshbond_data;         //!< Bond data for the mesh
    std::shared_ptr<MeshTriangleData> m_meshtriangle_data; //!< Triangle data for the mesh
    Scalar m_mesh_energy_old;    //!< storing old energy for dynamic bonding later
    bool m_data_changed;         //!< check if dynamic bonding has changed the mesh data
    };

namespace detail
    {
//! Exports MeshDefinition to python
void export_MeshDefinition(pybind11::module& m);

    } // end namespace detail

    } // end namespace hoomd

#endif
