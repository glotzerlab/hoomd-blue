/*
Highly Optimized Object-oriented Many-particle Dynamics -- Blue Edition
(HOOMD-blue) Open Source Software License Copyright 2008, 2009 Ames Laboratory
Iowa State University and The Regents of the University of Michigan All rights
reserved.

HOOMD-blue may contain modifications ("Contributions") provided, and to which
copyright is held, by various Contributors who have granted The Regents of the
University of Michigan the right to modify and/or distribute such Contributions.

Redistribution and use of HOOMD-blue, in source and binary forms, with or
without modification, are permitted, provided that the following conditions are
met:

* Redistributions of source code must retain the above copyright notice, this
list of conditions, and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright notice, this
list of conditions, and the following disclaimer in the documentation and/or
other materials provided with the distribution.

* Neither the name of the copyright holder nor the names of HOOMD-blue's
contributors may be used to endorse or promote products derived from this
software without specific prior written permission.

Disclaimer

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDER AND CONTRIBUTORS ``AS IS''
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, AND/OR
ANY WARRANTIES THAT THIS SOFTWARE IS FREE OF INFRINGEMENT ARE DISCLAIMED.

IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT,
INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE
OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF
ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

// $Id$
// $URL$
// Maintainer: joaander

#include <boost/shared_ptr.hpp>

#include "ForceCompute.h"
#include "NeighborList.h"
#include "Index1D.h"
#include "GPUArray.h"

/*! \file TablePotential.h
    \brief Declares the TablePotential class
*/

#ifndef __TABLEPOTENTIAL_H__
#define __TABLEPOTENTIAL_H__

//! Computes the potential and force on each particle based on values given in a table
/*! \b Overview
    Pair potentials and forces are evaluated for all particle pairs in the system within the given cutoff distances.
    Both the potentials and forces** are provided the tables V(r) and F(r) at discreet \a r values between \a rmin and
    \a rmax. Evaluations are performed by simple linear interpolation, thus why F(r) must be explicitly specified to
    avoid large errors resulting from the numerical derivative. Note that F(r) should store - dV/dr.

    \b Table memory layout

    V(r) and F(r) are specified for each unique particle type pair. They will be indexed so that values of increasing r
    go along the rows in memory for cache efficiency when reading values. The row index to put the potential at can be
    determined using an Index2DUpperTriangular (typei, typej), as it will uniquely index each unique pair.

    To improve cache coherency even further, values for V and F will be interleaved like so: V1 F1 V2 F2 V3 F3 ... To
    accomplish this, tables are stored with a value type of float2, elem.x will be V and elem.y will be F. Since Fn,
    Vn+1 and Fn+1 are read right after Vn, these are likely to be cache hits. Furthermore, on the GPU a single float2
    texture read can be used to access Vn and Fn.

    Three parameters need to be stored for each potential: rmin, rmax, and dr, the minimum r, maximum r, and spacing
    between r values in the table respectively. For simple access on the GPU, these will be stored in a float4 where
    x is rmin, y is rmax, and z is dr. They are indexed with the same Index2DUpperTriangular as the tables themselves.

    V(0) is the value of V at r=rmin. V(i) is the value of V at r=rmin + dr * i where i is chosen such that r >= rmin
    and r <= rmax. V(r) for r < rmin and > rmax is 0. The same goes for F. Thus V and F are defined between the region
    [rmin,rmax], inclusive.

    For ease of storing the data, all tables must be of the same number of points for all type pairs.

    \b Interpolation
    Values are interpolated linearly between two points straddling the given r. For a given r, the first point needed, i
    can be calculated via i = floorf((r - rmin) / dr). The fraction between ri and ri+1 can be calculated via
    f = (r - rmin) / dr - float(i). And the linear interpolation can then be performed via V(r) ~= Vi + f * (Vi+1 - Vi)
    \ingroup computes
*/
class TablePotential : public ForceCompute
    {
    public:
        //! Constructs the compute
        TablePotential(boost::shared_ptr<SystemDefinition> sysdef,
                       boost::shared_ptr<NeighborList> nlist,
                       unsigned int table_width);
                       
        //! Destructor
        virtual ~TablePotential() { }
        
        //! Set the table for a given type pair
        virtual void setTable(unsigned int typ1,
                              unsigned int typ2,
                              const std::vector<Scalar> &V,
                              const std::vector<Scalar> &F,
                              Scalar rmin,
                              Scalar rmax);
                              
        //! Returns a list of log quantities this compute calculates
        virtual std::vector< std::string > getProvidedLogQuantities();
        
        //! Calculates the requested log value and returns it
        virtual Scalar getLogValue(const std::string& quantity, unsigned int timestep);
        
    protected:
        boost::shared_ptr<NeighborList> m_nlist;    //!< The neighborlist to use for the computation
        unsigned int m_table_width;                 //!< Width of the tables in memory
        unsigned int m_ntypes;                      //!< Store the number of particle types
        GPUArray<Scalar2> m_tables;                 //!< Stored V and F tables
        GPUArray<Scalar4> m_params;                 //!< Parameters stored for each table
        
        //! Actually compute the forces
        virtual void computeForces(unsigned int timestep);
    };

//! Exports the TablePotential class to python
void export_TablePotential();

#endif

