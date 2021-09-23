// Copyright (c) 2009-2021 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.

// Maintainer: jglaser

/*! \file BondedGroupData.h
    \brief Declares BondedGroupData
 */

#ifdef __HIPCC__
#error This header cannot be compiled by nvcc
#endif

#ifndef __MESH_GROUP_DATA_H__
#define __MESH_GROUP_DATA_H__

#include "BondedGroupData.h"
#include "ExecutionConfiguration.h"
#include "GPUVector.h"
#include "HOOMDMPI.h"
#include "HOOMDMath.h"
#include "Index1D.h"
#include "ParticleData.h"
#include "Profiler.h"

#ifdef ENABLE_HIP
#include "BondedGroupData.cuh"
#include "CachedAllocator.h"
#endif

#include <hoomd/extern/nano-signal-slot/nano_signal_slot.hpp>
#include <memory>
#include <type_traits>
#ifndef __HIPCC__
#include <pybind11/pybind11.h>
#endif

#include <map>
#include <set>
#include <sstream>
#include <stack>
#include <string>
#include <vector>

/*! BondedGroupData is a generic storage class for small particle groups of fixed
 *  size N=2,3,4..., such as bonds, angles or dihedrals, which form part of a molecule.
 *
 *  \tpp group_size Size of groups
 *  \tpp name Name of element, i.e. bond, angle, dihedral, ..
 */
template<unsigned int group_size, typename Group, const char* name, bool bond>
class MeshGroupData: public BondedGroupData<group_size,Group,name,true>
    {
    public:
    //! Group size
    //
    //! Group data element type
    typedef union group_storage<group_size> members_t;

    //! Constructor for empty BondedGroupData
    MeshGroupData(std::shared_ptr<ParticleData> pdata, unsigned int n_group_types);

    //! Constructor to initialize from a snapshot
    MeshGroupData(std::shared_ptr<ParticleData> pdata, const TriangleData::Snapshot& snapshot);

    virtual ~MeshGroupData();

    //! Initialize from a snapshot
    //using BondedGroupData<group_size,Group,name,true>::initializeFromSnapshot;
    void initializeFromSnapshot(const TriangleData::Snapshot& snapshot);

    //! Take a snapshot
    //using BondedGroupData<group_size,Group,name,true>::takeSnapshot;
    virtual std::map<unsigned int, unsigned int> takeSnapshot(TriangleData::Snapshot& snapshot) const;

    /*
     * add/remove groups globally
     */

    //! Add a single bonded group on all processors
    /*! \param g Definition of group to add
     */
    unsigned int addBondedGroup(Group g);

    protected:

    };

//! Exports BondData to python
template<class T, class Group>
void export_MeshGroupData(pybind11::module& m,
                            std::string name,
                            std::string snapshot_name,
                            bool export_struct = true);

/*!
 * Typedefs for template instantiations
 */

/*
 * MeshBondData
 */
extern char name_meshbond_data[];

// Definition of a bond
struct MeshBond
    {
    typedef group_storage<4> members_t;

    //! Constructor
    /*! \param type Type of bond
     * \param _a First bond member
     * \param _b Second bond member
     * \param _ta First triangle
     * \param _tb Second triangle
     */
    MeshBond(unsigned int _type, unsigned int _a, unsigned int _b, int _ta, int _tb ) : type(_type), a(_a), b(_b), ta(_ta), tb(_tb) { }

    //! Constructor that takes a members_t (used internally by MeshBondData)
    /*! \param type
     *  \param members group members
     */
    MeshBond(typeval_t _typeval, members_t _members)
        : type(_typeval.type), a(_members.tag[0]), b(_members.tag[1]), ta(_members.tag[2]), tb(_members.tag[3])
        {
        }

    //! This helper function needs to be provided for the templated MeshBondData to work correctly
    members_t get_members() const
        {
        members_t m;
        m.tag[0] = a;
        m.tag[1] = b;
        m.tag[2] = ta;
        m.tag[3] = tb;
        return m;
        }

    //! This helper function needs to be provided for the templated MeshBondData to work correctly
    typeval_t get_typeval() const
        {
        typeval_t t;
        t.type = type;
        return t;
        }

    //! This helper function needs to be provided for the templated MeshBondData to work correctly
    static void export_to_python(pybind11::module& m)
        {
        pybind11::class_<MeshBond>(m, "MeshBond")
            .def(pybind11::init<unsigned int, unsigned int, unsigned int, int, int>())
            .def_readonly("type", &MeshBond::type)
            .def_readonly("a", &MeshBond::a)
            .def_readonly("b", &MeshBond::b)
            .def_readonly("ta", &MeshBond::ta)
            .def_readonly("tb", &MeshBond::tb);
        }

    unsigned int type; //!< Group type
    unsigned int a;    //!< First bond member
    unsigned int b;    //!< Second bond member
    unsigned int ta;    //!< First triangle
    unsigned int tb;    //!< Second triangle
    };

//! Definition of MeshBondData
typedef MeshGroupData<4, MeshBond, name_meshbond_data, true> MeshBondData;


/*
 * MeshTriangleData
 */
extern char name_meshtriangle_data[];

// Definition of an dihedral
struct MeshTriangle
    {
    typedef group_storage<6> members_t;

    //! Constructor
    /*! \param type Type of triangle
     * \param _a First dihedral member
     * \param _b Second dihedral member
     */
    MeshTriangle(unsigned int _type, unsigned int _a, unsigned int _b, unsigned int _c, int _ea, int _eb , int _ec)
        : type(_type), a(_a), b(_b), c(_c), ea(_ea), eb(_eb), ec(_ec)
        {
        }

    //! Constructor that takes a members_t (used internally by MeshTriangleData)
    /*! \param type
     *  \param members group members
     */
    MeshTriangle(typeval_t _typeval, members_t _members)
        : type(_typeval.type), a(_members.tag[0]), b(_members.tag[1]), c(_members.tag[2]),
          ea(_members.tag[3]), eb(_members.tag[4]), ec(_members.tag[5])
        {
        }

    //! This helper function needs to be provided for the templated MeshTriangleData to work correctly
    members_t get_members() const
        {
        members_t m;
        m.tag[0] = a;
        m.tag[1] = b;
        m.tag[2] = c;
        m.tag[3] = ea;
        m.tag[4] = eb;
        m.tag[5] = ec;
        return m;
        }

    //! This helper function needs to be provided for the templated MeshTriangleData to work correctly
    typeval_t get_typeval() const
        {
        typeval_t t;
        t.type = type;
        return t;
        }

    //! This helper function needs to be provided for the templated MeshTriangleData to work correctly
    static void export_to_python(pybind11::module& m)
        {
        pybind11::class_<MeshTriangle>(m, "MeshTriangle")
            .def(pybind11::
                     init<unsigned int, unsigned int, unsigned int, unsigned int, int, int , int>())
            .def_readonly("type", &MeshTriangle::type)
            .def_readonly("a", &MeshTriangle::a)
            .def_readonly("b", &MeshTriangle::b)
            .def_readonly("c", &MeshTriangle::c)
            .def_readonly("ea", &MeshTriangle::ea)
            .def_readonly("eb", &MeshTriangle::eb)
            .def_readonly("ec", &MeshTriangle::ec);
        }

    unsigned int type; //!< Group type
    unsigned int a;    //!< First dihedral member
    unsigned int b;    //!< Second dihedral member
    unsigned int c;    //!< Third dihedral member
    unsigned int ea;    //!< First endge
    unsigned int eb;    //!< Second edge
    unsigned int ec;    //!< Third edge
    };

//! Definition of MeshTriangleData
typedef MeshGroupData<6, MeshTriangle, name_meshtriangle_data, false> MeshTriangleData;

#endif
