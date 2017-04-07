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
    EAMForceCompute(std::shared_ptr<SystemDefinition> sysdef, char *filename, int type_of_file, int ifinter,
                    int setnrho, int setnr);

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
    virtual void loadFile(char *filename, int type_of_file, int ifinter, int setnrho, int setnr);

protected:
    std::shared_ptr<NeighborList> m_nlist;       //!< the neighborlist to use for the computation
    Scalar m_r_cut;                              //!< cut-off radius
    unsigned int m_ntypes;                       //!< number of potential element types
    unsigned int rawnrho;                        //!< number of tabulated values of F(rho) in file
    Scalar rawdrho;                              //!< interval of rho in file
    unsigned int rawnr;                          //!< number of tabulated values of rho(r), r*phi(r) in file
    Scalar rawdr;                                //!< interval of r in file
    unsigned int nrho;                           //!< number of tabulated values of interpolated F(rho)
    Scalar drho;                                 //!< interval of rho in interpolated table
    unsigned int nr;                             //!< number of tabulated values of interpolated rho(r), r*phi(r)
    Scalar dr;                                   //!< interval of r in interpolated table
    Scalar rawrdrho;                             //!< 1.0 / rawdrho
    Scalar rawrdr;                               //!< 1.0 / rawdr
    Scalar rdrho;                                //!< 1.0 / drho
    Scalar rdr;                                  //!< 1.0 / dr
    std::vector<Scalar> mass;                    //!< array mass(type)
    std::vector<int> types;                      //!< array type(id)
    std::vector<int> nproton;                    //!< atomic number
    std::vector<Scalar> lconst;                  //!< lattice constant
    std::vector<std::string> atomcomment;        //!< atom comment
    std::vector<std::string> names;              //!< array names(type)

    std::vector<Scalar> rawembeddingFunction;    //!< array F(rho), embedding energy in file
    std::vector<Scalar> rawelectronDensity;      //!< array rho(r), electron density in file
    std::vector<Scalar> rawpairPotential;        //!< array r*phi(r), pairwise energy in file

    // interpolation
    std::vector<Scalar> embeddingFunction;       //!< array F(rho), interpolated embedding energy
    std::vector<Scalar> electronDensity;         //!< array rho(r), interpolated electron density
    std::vector<Scalar> pairPotential;           //!< array r*phi(r), interpolated pairwise energy
    std::vector<Scalar> derivativeEmbeddingFunction;  //!< interpolated array d(F(rho))/drho
    std::vector<Scalar> derivativeElectronDensity;    //!< interpolated array d(rho(r))/dr
    std::vector<Scalar> derivativePairPotential;      //!< interpolated array d(r*phi(r))/dr

    //! 3rd order interpolation parameters
    std::vector<std::vector<Scalar> > iemb;  // param for F
    std::vector<std::vector<Scalar> > irho;  // param for rho
    std::vector<std::vector<Scalar> > irphi; // param for r*phi
    virtual void interpolate(int num_all, int num_per, Scalar delta, std::vector<Scalar> *f,
                             std::vector<std::vector<Scalar> > *spline);

    //! Actually compute the forces
    virtual void computeForces(unsigned int timestep);

    //! Method to be called when number of types changes
    virtual void slotNumTypesChange() {
        m_exec_conf->msg->error() << "Changing the number of types is unsupported for pair.eam" << std::endl;
        throw std::runtime_error("Unsupported feature");
    }

    // 0405 begin
    GPUArray<Scalar> m_F;
    GPUArray<Scalar> m_rho;
    GPUArray<Scalar> m_rphi;

    virtual void inter(int num_all, int num_per, Scalar delta, ArrayHandle<Scalar> *f);
    // 0405 end
};

//! Exports the EAMForceCompute class to python
void export_EAMForceCompute(pybind11::module &m);

#endif
