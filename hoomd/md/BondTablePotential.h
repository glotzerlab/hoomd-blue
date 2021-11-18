// Copyright (c) 2009-2021 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.

// Maintainer: phillicl

#include "hoomd/ForceCompute.h"
#include "hoomd/GPUArray.h"
#include "hoomd/Index1D.h"

#include <memory>

/*! \file BondTablePotential.h
    \brief Declares the BondTablePotential class
*/

#ifdef __HIPCC__
#error This header cannot be compiled by nvcc
#endif

#include <pybind11/pybind11.h>

#ifndef __BONDTABLEPOTENTIAL_H__
#define __BONDTABLEPOTENTIAL_H__

namespace hoomd
    {
namespace md
    {
//! Computes the potential and force on bonds based on values given in a table
/*! \b Overview
    Bond potentials and forces are evaluated for all bonded particle pairs in the system.
    Both the potentials and forces are provided the tables V(r) and F(r) at discreet \a r values
   between \a rmin and \a rmax. Evaluations are performed by simple linear interpolation, thus why
   F(r) must be explicitly specified to avoid large errors resulting from the numerical derivative.
   Note that F(r) should store - dV/dr.

    \b Table memory layout

    V(r) and F(r) are specified for each bond type.

    Three parameters need to be stored for each bond potential: rmin, rmax, and dr, the minimum r,
   maximum r, and spacing between r values in the table respectively. For simple access on the GPU,
   these will be stored in a float4 where x is rmin, y is rmax, and z is dr.

    V(0) is the value of V at r=rmin. V(i) is the value of V at r=rmin + dr * i where i is chosen
   such that r >= rmin and r <= rmax. V(r) for r < rmin and > rmax is 0. The same goes for F. Thus V
   and F are defined between the region [rmin,rmax], inclusive.

    For ease of storing the data, all tables must be of the same number of points for all bonds.

    \b Interpolation
    Values are interpolated linearly between two points straddling the given r. For a given r, the
   first point needed, i can be calculated via i = floorf((r - rmin) / dr). The fraction between ri
   and ri+1 can be calculated via f = (r - rmin) / dr - float(i). And the linear interpolation can
   then be performed via V(r) ~= Vi + f * (Vi+1 - Vi) \ingroup computes
*/
class PYBIND11_EXPORT BondTablePotential : public ForceCompute
    {
    public:
    //! Constructs the compute
    BondTablePotential(std::shared_ptr<SystemDefinition> sysdef, unsigned int table_width);

    //! Destructor
    virtual ~BondTablePotential();

    //! Set the table for a given type pair
    virtual void setTable(unsigned int type,
                          const std::vector<Scalar>& V,
                          const std::vector<Scalar>& F,
                          Scalar rmin,
                          Scalar rmax);

#ifdef ENABLE_MPI
    //! Get ghost particle fields requested by this pair potential
    /*! \param timestep Current time step
     */
    virtual CommFlags getRequestedCommFlags(uint64_t timestep)
        {
        CommFlags flags = CommFlags(0);
        flags[comm_flag::tag] = 1;
        flags |= ForceCompute::getRequestedCommFlags(timestep);
        return flags;
        }
#endif

    protected:
    std::shared_ptr<BondData> m_bond_data; //!< Bond data to use in computing bonds
    unsigned int m_table_width;            //!< Width of the tables in memory
    GPUArray<Scalar2> m_tables;            //!< Stored V and F tables
    GPUArray<Scalar4> m_params;            //!< Parameters stored for each table
    Index2D m_table_value;                 //!< Index table helper

    //! Actually compute the forces
    virtual void computeForces(uint64_t timestep);
    };

namespace detail
    {
//! Exports the TablePotential class to python
void export_BondTablePotential(pybind11::module& m);

    } // end namespace detail
    } // end namespace md
    } // end namespace hoomd

#endif
