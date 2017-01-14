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

//! Computes EAM forces on each particle
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
class EAMForceCompute : public ForceCompute {
public:
    //! Constructs the compute
    EAMForceCompute(std::shared_ptr<SystemDefinition> sysdef, char *filename, int type_of_file);

    //! Destructor
    virtual ~EAMForceCompute();

    //! Sets the neighbor list to be used for the EAM force
    virtual void set_neighbor_list(std::shared_ptr<NeighborList> nlist);

    //! Get the r cut value read from the EAM potential file
    virtual Scalar get_r_cut();

    //! Returns a list of log quantities this compute calculates
    virtual std::vector<std::string> getProvidedLogQuantities();

    //! Calculates the requested log value and returns it
    virtual Scalar getLogValue(const std::string &quantity, unsigned int timestep);

    //! Load EAM potential file
    virtual void loadFile(char *filename, int type_of_file);

protected:
    std::shared_ptr<NeighborList> m_nlist;       //!< the neighborlist to use for the computation
    Scalar m_r_cut;                              //!< cut-off radius
    unsigned int m_ntypes;                       //!< number of potential element types
    unsigned int nr;                             //!< number of tabulated values of rho(r), r*phi(r)
    unsigned int nrho;                           //!< number of tabulated values of F(rho)
    Scalar dr;                                   //!< interval of r
    Scalar rdr;                                  //!< 1.0 / dr
    Scalar drho;                                 //!< interval of rho
    Scalar rdrho;                                //!< 1.0 / drho
    std::vector<Scalar> mass;                    //!< array mass(type)
    std::vector<int> types;                      //!< array type(id)
    std::vector<int> nproton;                    //!< atomic number
    std::vector<Scalar> lconst;                  //!< lattice constant
    std::vector<std::string> atomcomment;        //!< atom comment
    std::vector<std::string> names;              //!< array names(type)


    std::vector<Scalar> electronDensity;         //!< array rho(r), electron density
    std::vector<Scalar> pairPotential;           //!< array r*phi(r), pairwise energy
    std::vector<Scalar> embeddingFunction;       //!< array F(rho), embedding energy

    std::vector<Scalar> derivativeElectronDensity;    //!< array d(rho(r))/dr
    std::vector<Scalar> derivativePairPotential;      //!< array d(r*phi(r))/dr
    std::vector<Scalar> derivativeEmbeddingFunction;  //!< array d(F(rho))/drho

    // TODO: interpolation
    /* begin */
    std::vector<Scalar> newelectronDensity;         //!< array rho(r), electron density
    std::vector<Scalar> newpairPotential;           //!< array r*phi(r), pairwise energy
    std::vector<Scalar> newembeddingFunction;       //!< array F(rho), embedding energy
    std::vector<Scalar> newderivativeElectronDensity;    //!< array d(rho(r))/dr
    std::vector<Scalar> newderivativePairPotential;      //!< array d(r*phi(r))/dr
    std::vector<Scalar> newderivativeEmbeddingFunction;  //!< array d(F(rho))/drho
    //! 3rd order interpolation
    std::vector< std::vector< Scalar > > irho;  // as the same as LAMMPS's rhor_spline
    std::vector< std::vector< Scalar > > irphi; // as the same as LAMMPS's z2r_spline
    std::vector< std::vector< Scalar > > iemb; // as the same as LAMMPS's frho_spline
    virtual void interpolate(int num_all, int num_per, Scalar delta, std::vector< Scalar >* f, std::vector< std::vector< Scalar > >* spline);
    /* end */

    //! Actually compute the forces
    virtual void computeForces(unsigned int timestep);

    //! Method to be called when number of types changes
    virtual void slotNumTypesChange() {
        m_exec_conf->msg->error() << "Changing the number of types is unsupported for pair.eam" << std::endl;
        throw std::runtime_error("Unsupported feature");
    }
};

//! Exports the EAMForceCompute class to python
void export_EAMForceCompute(pybind11::module &m);

#endif
