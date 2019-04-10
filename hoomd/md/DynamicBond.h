// Copyright (c) 2009-2018 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.


// Maintainer: ?

#include "hoomd/ParticleGroup.h"
#include "hoomd/Updater.h"
#include "NeighborList.h"
#include <memory>

#include <hoomd/extern/pybind/include/pybind11/pybind11.h>
/*! \file DynamicBond.h
    \brief Declares a class for computing bond breakage/formation
*/

#ifdef NVCC
#error This header cannot be compiled by nvcc
#endif

//
#ifndef __DYNAMICBOND_H__
#define __DYNAMICBOND_H__

//! Creates or breaks bonds with a given probability
/*!
*/
class PYBIND11_EXPORT DynamicBond : public Updater
    {
    public:
        //! Constructs the compute
        DynamicBond(std::shared_ptr<SystemDefinition> sysdef,
                 // std::shared_ptr<ParticleGroup> group,
                 // std::shared_ptr<NeighborList> nlist,
                 int seed,
                 int period);

        //! Destructor
        virtual ~DynamicBond();

        //! Take one timestep forward
        virtual void update(unsigned int timestep);

        // void set_params(Scalar r_cut, std::string bond_type);
    protected:
        std::shared_ptr<ParticleGroup> m_group;   //!< Group of particles to which the dynamic bonding is applied
        std::shared_ptr<NeighborList> m_nlist;                 //!< neighborlist

        int period;         //!< period to create/destroy bonds
        int seed;              //!< a seed for the random number generator
        // bond_type;             //!< type of bond to be created or destroyed
        // Scalar m_r_cut;            //!< cutoff radius
        // Scalar prob_create;    //!< probability that a bond will be formed
        // Scalar prob_destroy;   //!< probability that a bond will be destroyed
    // private:
    };

//! Exports the DynamicBond class to python
void export_DynamicBond(pybind11::module& m);

#endif
