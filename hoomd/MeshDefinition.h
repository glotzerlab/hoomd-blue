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

//! Mesh class that contains all infrastructure necessary to combine a set of particles into a mesh
//! triangulation
/*! MeshDefinition is a container class to define a mesh tringulation comprised of the
    particles within the simulation system. The vertices of the mesh are given by the
    particle positions. The information which vertex tuples are directly bonded (i.e. share an
    edge) and which vertex triplets consitute triangles is stored as two shared pointer for
    quick and easy access from within C++ and python without worrying about data management.
    The pointers also encode simplex labels such that the we can detect easily which bonds make
    up a triangle and which triangles share a common edge bond.


    <b>Background and intended usage</b>

    The data structure stored in MeshDefinition is all the edge bonds and triangles within
    a particle mesh. It is used to combine a set of particles into a bonded mesh. In doing so
    we can asign different potentials (as bond or surface potentials) to the mesh data structure.
    The class has to access system information such as the particle positions to define the mesh
    vertices.

    As any data structure class in MeshDefinition can potentially reference any
   other, other classes can simply use the mesh data by giving the shared pointer to the referenced
   class to the constructor of the onem that needs to refer to it. Note that using this setup, there
   can be no circular references. This is a \b good \b thing ^TM, as it promotes good separation
   and isolation of the various classes responsibilities.

    <b>Initializing</b>

    A default constructed MeshDefinition is full of NULL shared pointers. Afterwards the user has to
    specify the triangulation information via a Triangle snapshot (see MeshGroupData.h). The
    corresponding edge information is calculated automatically. MeshDefinition allows for only one
    mesh type to prevent ambiguity how to set triangle and edge bond types.


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

    pybind11::list getTypes() const
        {
        return m_meshtriangle_data->getTypesPy();
        }

    unsigned int getSize() const
        {
        return m_meshtriangle_data->getN();
        }

    BondData::Snapshot getBondData();

    Scalar getEnergy()
        {
        return m_mesh_energy;
        }

    void updateTriangleData();

    void updateMeshData();

    void MeshDataChange();

    TriangleData::Snapshot triangle_data; //!< The triangle data accessible in python

    private:
    std::shared_ptr<SystemDefinition>
        m_sysdef; //!< System definition later needed for dynamic bonding
    std::shared_ptr<MeshBondData> m_meshbond_data;         //!< Bond data for the mesh
    std::shared_ptr<MeshTriangleData> m_meshtriangle_data; //!< Triangle data for the mesh
    GlobalVector<Scalar3> m_triangle_normals;              //! normal vectors of the triangles
    Scalar m_mesh_energy; //!< storing energy for dynamic bonding later
    bool m_data_changed;  //!< check if dynamic bonding has changed the mesh data
    };

namespace detail
    {
//! Exports MeshDefinition to python
void export_MeshDefinition(pybind11::module& m);

    } // end namespace detail

    } // end namespace hoomd

#endif