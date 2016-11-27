// Copyright (c) 2009-2016 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.

#include "hoomd/ForceCompute.h"
#include "hoomd/md/NeighborList.h"

#include <memory>

/*! \file EAMForceCompute.h
    \brief Declares the EAMForceCompute class
*/

#ifdef NVCC
#error This header cannot be compiled by nvcc
#endif

#include <hoomd/extern/pybind/include/pybind11/pybind11.h>

#ifndef __EAMFORCECOMPUTE_H__
#define __EAMFORCECOMPUTE_H__

//! Computes Lennard-Jones forces on each particle
/*! The total pair force is summed for each particle when compute() is called. Forces are only summed between
    neighboring particles with a separation distance less than \c r_cut. A NeighborList must be provided
    to identify these neighbors. Calling compute() in this class will in turn result in a call to the
    NeighborList's compute() to make sure that the neighbor list is up to date.

    Usage: Construct a EAMForceCompute, providing it an already constructed ParticleData and NeighborList.
    Then set parameters for all possible pairs of types by calling setParams.

    Forces can be computed directly by calling compute() and then retrieved with a call to acquire(), but
    a more typical usage will be to add the force compute to NVEUpdater or NVTUpdater.

    \ingroup computes
*/
class EAMForceCompute : public ForceCompute
    {
    public:
        //! Constructs the compute
        EAMForceCompute(std::shared_ptr<SystemDefinition> sysdef,  char *filename, int type_of_file);

        //! Destructor
        virtual ~EAMForceCompute();

        //! Sets the neighbor list to be used for the EAM force
        virtual void set_neighbor_list(std::shared_ptr<NeighborList> nlist);

        //! Get the r cut value read from the EAM potential file
        virtual Scalar get_r_cut();

        //! Returns a list of log quantities this compute calculates
        virtual std::vector< std::string > getProvidedLogQuantities();

        //! Calculates the requested log value and returns it
        virtual Scalar getLogValue(const std::string& quantity, unsigned int timestep);

        //! Shifting modes that can be applied to the energy
        virtual void loadFile(char *filename, int type_of_file);


    protected:
        std::shared_ptr<NeighborList> m_nlist;       //!< The neighborlist to use for the computation
        Scalar m_r_cut;                                //!< Cuttoff radius beyond which the force is set to 0
        unsigned int m_ntypes;                         //!< Store the width and height of lj1 and lj2 here

        Scalar drho;                                   //!< Undocumented parameter
        Scalar dr;                                     //!< Undocumented parameter
        Scalar rdrho;                                  //!< Undocumented parameter
        Scalar rdr;                                    //!< Undocumented parameter
        std::vector<Scalar> mass;                           //!< Undocumented parameter
        std::vector<int> types;                             //!< Undocumented parameter
        std::vector<std::string> names;                          //!< Undocumented parameter
        unsigned int nr;                               //!< Undocumented parameter
        unsigned int nrho;                             //!< Undocumented parameter


        std::vector<Scalar> electronDensity;                //!< array rho(r)
        std::vector<Scalar2> pairPotential;                  //!< array Z(r)
        std::vector<Scalar> embeddingFunction;              //!< array F(rho)

        std::vector<Scalar> derivativeElectronDensity;      //!< array rho'(r)
        std::vector<Scalar> derivativePairPotential;        //!< array Z'(r)
        std::vector<Scalar> derivativeEmbeddingFunction;    //!< array F'(rho)

        //! Actually compute the forces
        virtual void computeForces(unsigned int timestep);

        //! Method to be called when number of types changes
        virtual void slotNumTypesChange()
            {
            m_exec_conf->msg->error() << "Changing the number of types is unsupported for pair.eam" << std::endl;
            throw std::runtime_error("Unsupported feature");
            }
    };

//! Exports the EAMForceCompute class to python
void export_EAMForceCompute(pybind11::module& m);

#endif
