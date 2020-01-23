// Copyright (c) 2009-2018 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause
// License.

// Maintainer: atravitz

#include <math.h>
#include <memory>
#include "hoomd/Index1D.h"
#include "hoomd/ParticleGroup.h"
#include "hoomd/Updater.h"
#include "hoomd/md/NeighborList.h"

#include <hoomd/extern/pybind/include/pybind11/pybind11.h>

/*! \file PopBD.h
    \brief Declares a class for computing bond breakage/formation
*/

#ifdef NVCC
#error This header cannot be compiled by nvcc
#endif

#ifndef __POPBD_H__
#define __POPBD_H__

//! Creates or breaks bonds with a given probability
/*!
 */
class PYBIND11_EXPORT PopBD : public Updater
{
public:
    //! Constructs the compute
    PopBD(std::shared_ptr<SystemDefinition> sysdef,
          std::shared_ptr<ParticleGroup> group,
          std::shared_ptr<NeighborList> nlist,
          int seed,
          Scalar delta_t,
          int period,
          unsigned int table_width);

    //! Destructor
    virtual ~PopBD();

    virtual void setParams(Scalar r_cut, std::string bond_type, int n_polymer);

    virtual void setTable(const std::vector<Scalar> &XB,
                          const std::vector<Scalar> &M,
                          const std::vector<Scalar> &L,
                          Scalar rmin,
                          Scalar rmax);
    //! Take one timestep forward
    virtual void update(unsigned int timestep);

protected:
    std::shared_ptr<ParticleGroup> m_group; //!< Group of particles to operate on
    std::shared_ptr<NeighborList> m_nlist;  //!< neighborlist
    std::shared_ptr<BondData> m_bond_data;  //!< Bond data to use in computing bonds
    int period;                             //!< period to create/destroy bonds
    int m_type;                             //!< bond type to create and break
    int m_seed;                             //!< seed for random number generator
    Scalar m_r_cut;                         //!< cut off distance for computing bonds
    unsigned int m_table_width;             //!< Width of the tables in memory
    GPUArray<Scalar2> m_tables;             //!< Stored V and F tables
    GPUArray<Scalar4> m_params;             //!< Parameters stored for each table
    Index2D m_table_value;                  //!< Index table helper
    Scalar m_delta_t;                       //!< time step from integrator
    std::vector<int> m_nloops;              //!< structure of size N to store number of loops for each colloid
    // std::map<std::pair<int, int>, int> m_nbonds;
    // std::map<std::pair<int, int>, int> m_delta_nbonds;
    int n_polymer;                          //!< number of polymers per colloid
};

//! Exports the PopBD class to python
void export_PopBD(pybind11::module &m);

#endif
