// Copyright (c) 2009-2024 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

/*! \file MeshDefinition.h
    \brief Defines the MeshDefinition class
 */

#ifdef __HIPCC__
#error This header cannot be compiled by nvcc
#endif

#include "BondedGroupData.h"
#include "MeshGroupData.h"
#include "SystemDefinition.h"

#include <memory>
#include <pybind11/pybind11.h>

#ifndef __MESH_DEFINITION_H__
#define __MESH_DEFINITION_H__

namespace hoomd
    {
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
   class to the constructor of the one that needs to refer to it. Note that using this setup, there
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
    MeshDefinition(std::shared_ptr<SystemDefinition> sysdef, unsigned int n_types);

    //! Access the mesh triangle data defined for the simulation
    std::shared_ptr<TriangleData> getMeshTriangleData()
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

    unsigned int getSize()
        {
        TriangleData::Snapshot triangles = getTriangleData();
        return triangles.getSize();
        }

    void setTypes(pybind11::list types);

    BondData::Snapshot getBondData();

    TriangleData::Snapshot getTriangleData();

    pybind11::object getTriangulationData();

    void setTriangulationData(pybind11::dict triangulation);

    private:
    std::shared_ptr<SystemDefinition>
        m_sysdef; //!< System definition later needed for dynamic bonding
    std::shared_ptr<MeshBondData> m_meshbond_data;     //!< Bond data for the mesh
    std::shared_ptr<TriangleData> m_meshtriangle_data; //!< Triangle data for the mesh
    };

namespace detail
    {
//! Exports MeshDefinition to python
void export_MeshDefinition(pybind11::module& m);

    } // end namespace detail

    } // end namespace hoomd

#endif
