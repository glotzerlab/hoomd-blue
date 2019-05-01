// Copyright (c) 2009-2019 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.


// Maintainer: phillicl

#include "hoomd/ForceCompute.h"
#include "hoomd/BondedGroupData.h"
#include "hoomd/Index1D.h"
#include "hoomd/GPUArray.h"

#include <memory>

/*! \file TableDihedralForceCompute.h
    \brief Declares the TableDihedralForceCompute class
*/

#ifdef NVCC
#error This header cannot be compiled by nvcc
#endif

#include <hoomd/extern/pybind/include/pybind11/pybind11.h>

#ifndef __TABLEDIHEDRALFORCECOMPUTE_H__
#define __TABLEDIHEDRALFORCECOMPUTE_H__

//! Computes the potential and force on dihedrals based on values given in a table
/*! \b Overview
    Bond potentials and forces are evaluated for all dihedraled particle pairs in the system.
    Both the potentials and forces are provided the tables V(r) and F(r) at discreet \a r values between \a rmin and
    \a rmax. Evaluations are performed by simple linear interpolation, thus why F(r) must be explicitly specified to
    avoid large errors resulting from the numerical derivative. Note that F(r) should store - dV/dr.

    \b Table memory layout

    V(r) and F(r) are specified for each dihedral type.

    Three parameters need to be stored for each dihedral potential: rmin, rmax, and dr, the minimum r, maximum r, and spacing
    between r values in the table respectively. For simple access on the GPU, these will be stored in a Scalar4 where
    x is rmin, y is rmax, and z is dr.

    V(0) is the value of V at r=rmin. V(i) is the value of V at r=rmin + dr * i where i is chosen such that r >= rmin
    and r <= rmax. V(r) for r < rmin and > rmax is 0. The same goes for F. Thus V and F are defined between the region
    [rmin,rmax], inclusive.

    For ease of storing the data, all tables must be of the same number of points for all dihedrals.

    \b Interpolation
    Values are interpolated linearly between two points straddling the given r. For a given r, the first point needed, i
    can be calculated via i = floorf((r - rmin) / dr). The fraction between ri and ri+1 can be calculated via
    f = (r - rmin) / dr - Scalar(i). And the linear interpolation can then be performed via V(r) ~= Vi + f * (Vi+1 - Vi)
    \ingroup computes
*/
class PYBIND11_EXPORT TableDihedralForceCompute : public ForceCompute
    {
    public:
        //! Constructs the compute
        TableDihedralForceCompute(std::shared_ptr<SystemDefinition> sysdef,
                       unsigned int table_width,
                       const std::string& log_suffix="");

        //! Destructor
        virtual ~TableDihedralForceCompute();

        //! Set the table for a given type pair
        virtual void setTable(unsigned int type,
                              const std::vector<Scalar> &V,
                              const std::vector<Scalar> &T);

        //! Returns a list of log quantities this compute calculates
        virtual std::vector< std::string > getProvidedLogQuantities();

        //! Calculates the requested log value and returns it
        virtual Scalar getLogValue(const std::string& quantity, unsigned int timestep);

        #ifdef ENABLE_MPI
        //! Get ghost particle fields requested by this pair potential
        /*! \param timestep Current time step
        */
        virtual CommFlags getRequestedCommFlags(unsigned int timestep)
            {
                CommFlags flags = CommFlags(0);
                flags[comm_flag::tag] = 1;
                flags |= ForceCompute::getRequestedCommFlags(timestep);
                return flags;
            }
        #endif

        //! Get table entry
        /*! \param type type index
            \param i index to access
            \returns the table entries at the index

            For unit testing
        */
        Scalar2 getEntry(unsigned int type, unsigned int i)
            {
            ArrayHandle<Scalar2> h_tables(m_tables, access_location::host, access_mode::read);
            return h_tables.data[m_table_value(i, type)];
            }

    protected:
        std::shared_ptr<DihedralData> m_dihedral_data;    //!< Bond data to use in computing dihedrals
        unsigned int m_table_width;                 //!< Width of the tables in memory
        GPUArray<Scalar2> m_tables;                  //!< Stored V and F tables
        Index2D m_table_value;                      //!< Index table helper
        std::string m_log_name;                     //!< Cached log name

        //! Actually compute the forces
        virtual void computeForces(unsigned int timestep);
    };

//! Exports the TablePotential class to python
void export_TableDihedralForceCompute(pybind11::module& m);

#endif
