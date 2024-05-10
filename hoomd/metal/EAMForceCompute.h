// Copyright (c) 2009-2024 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

#include "hoomd/ForceCompute.h"
#include "hoomd/md/NeighborList.h"

#include <memory>

/*! \file EAMForceCompute.h
 \brief Declares the EAMForceCompute class
 */

#ifdef __HIPCC__
#error This header cannot be compiled by nvcc
#endif

#include <pybind11/pybind11.h>

#ifndef __EAMFORCECOMPUTE_H__
#define __EAMFORCECOMPUTE_H__

namespace hoomd
    {
namespace metal
    {
//! Computes the potential and force on each particle based on values given in a EAM potential
/*! \b Overview
 The total potential and force is computed for each particle when compute() is called. Potentials
 and forces are only computed between neighbouring particles with a separation distance less than \c
 r_cut. A NeighborList must be provided to identify these neighbours.

 \b Interpolation
 The cubic interpolation is used. For each data point, including the value of the point, there are 3
 coefficients.

 \b Potential memory layout
 The potential data and the coefficients are stored in six GPUArray<Scalar> arrays: the embedded
 potential function (m_F) and its derivative (m_dF), the electron density function (m_rho) and its
 derivative (m_drho), the pair potential function (m_rphi) and its derivative (m_drphi). The 3
 coefficients for a data point is stored continuously, for example, h_F.data[100].w is the embedded
 potential function's value read from the 100st position of the potential file,
 h_F.data[100].z, h_F.data[100].y, h_F.data[100*].x, are for interpolating embedded function,
 h_dF.data[100].z, h_dF.data[100].y, h_dF.data[100].x, are for interpolating derivative embedded
 function.

 \ingroup computes
 */
class EAMForceCompute : public ForceCompute
    {
    public:
    //! Constructs the compute
    EAMForceCompute(std::shared_ptr<SystemDefinition> sysdef, char* filename, int type_of_file);

    //! Destructor
    virtual ~EAMForceCompute();

    //! Sets the neighbor list to be used for the EAM force
    virtual void set_neighbor_list(std::shared_ptr<md::NeighborList> nlist);

    //! Get the r cut value read from the EAM potential file
    virtual Scalar get_r_cut();

    //! Load EAM potential file
    virtual void loadFile(char* filename, int type_of_file);

    protected:
    std::shared_ptr<md::NeighborList> m_nlist; //!< the neighborlist to use for the computation
    Scalar m_r_cut;                            //!< cut-off radius
    unsigned int m_ntypes;                     //!< number of potential element types
    unsigned int nrho;          //!< number of tabulated values of interpolated F(rho)
    Scalar drho;                //!< interval of rho in interpolated table
    Scalar rdrho;               //!< 1.0 / drho
    unsigned int nr;            //!< number of tabulated values of interpolated rho(r), r*phi(r)
    Scalar dr;                  //!< interval of r in interpolated table
    Scalar rdr;                 //!< 1.0 / dr
    std::vector<double> mass;   //!< array mass(type)
    std::vector<int> types;     //!< array type(id)
    std::vector<int> nproton;   //!< atomic number
    std::vector<double> lconst; //!< lattice constant
    std::vector<std::string> atomcomment; //!< atom comment
    std::vector<std::string> names;       //!< array names(type)

    GPUArray<Scalar4> m_F;     //!< embedded function and its coefficients
    GPUArray<Scalar4> m_rho;   //!< electron density and its coefficients
    GPUArray<Scalar4> m_rphi;  //!< pair wise function and its coefficients
    GPUArray<Scalar4> m_dF;    //!< derivative embedded function and its coefficients
    GPUArray<Scalar4> m_drho;  //!< derivative electron density and its coefficients
    GPUArray<Scalar4> m_drphi; //!< derivative pair wise function and its coefficients
    GPUArray<Scalar> m_dFdP;   //!< derivative F / derivative P

    //! Actually compute the forces
    virtual void computeForces(uint64_t timestep);

    //! cubic interpolation
    virtual void interpolation(int num_all,
                               int num_per,
                               Scalar delta,
                               ArrayHandle<Scalar4>* f,
                               ArrayHandle<Scalar4>* df);
    };

namespace detail
    {
//! Exports the EAMForceCompute class to python
void export_EAMForceCompute(pybind11::module& m);

    } // end namespace detail
    } // end namespace metal
    } // end namespace hoomd

#endif
