/*
Highly Optimized Object-oriented Many-particle Dynamics -- Blue Edition
(HOOMD-blue) Open Source Software License Copyright 2008-2011 Ames Laboratory
Iowa State University and The Regents of the University of Michigan All rights
reserved.

HOOMD-blue may contain modifications ("Contributions") provided, and to which
copyright is held, by various Contributors who have granted The Regents of the
University of Michigan the right to modify and/or distribute such Contributions.

You may redistribute, use, and create derivate works of HOOMD-blue, in source
and binary forms, provided you abide by the following conditions:

* Redistributions of source code must retain the above copyright notice, this
list of conditions, and the following disclaimer both in the code and
prominently in any materials provided with the distribution.

* Redistributions in binary form must reproduce the above copyright notice, this
list of conditions, and the following disclaimer in the documentation and/or
other materials provided with the distribution.

* All publications and presentations based on HOOMD-blue, including any reports
or published results obtained, in whole or in part, with HOOMD-blue, will
acknowledge its use according to the terms posted at the time of submission on:
http://codeblue.umich.edu/hoomd-blue/citations.html

* Any electronic documents citing HOOMD-Blue will link to the HOOMD-Blue website:
http://codeblue.umich.edu/hoomd-blue/

* Apart from the above required attributions, neither the name of the copyright
holder nor the names of HOOMD-blue's contributors may be used to endorse or
promote products derived from this software without specific prior written
permission.

Disclaimer

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDER AND CONTRIBUTORS ``AS IS'' AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, AND/OR ANY
WARRANTIES THAT THIS SOFTWARE IS FREE OF INFRINGEMENT ARE DISCLAIMED.

IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT,
INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE
OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF
ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

// Maintainer: phillicl

#include <boost/shared_ptr.hpp>

#include "ForceCompute.h"
#include "BondData.h"
#include "GPUArray.h"

/*! \file BondTablePotential.h
    \brief Declares the BondTablePotential class
*/

#ifdef NVCC
#error This header cannot be compiled by nvcc
#endif

#ifndef __BONDTABLEPOTENTIAL_H__
#define __BONDTABLEPOTENTIAL_H__

//! Computes the potential and force on bonds based on values given in a table
/*! \b Overview
    Bond potentials and forces are evaluated for all bonded particle pairs in the system.
    Both the potentials and forces are provided the tables V(r) and F(r) at discreet \a r values between \a rmin and
    \a rmax. Evaluations are performed by simple linear interpolation, thus why F(r) must be explicitly specified to
    avoid large errors resulting from the numerical derivative. Note that F(r) should store - dV/dr.

    \b Table memory layout

    V(r) and F(r) are specified for each bond type.

    Three parameters need to be stored for each bond potential: rmin, rmax, and dr, the minimum r, maximum r, and spacing
    between r values in the table respectively. For simple access on the GPU, these will be stored in a float4 where
    x is rmin, y is rmax, and z is dr.

    V(0) is the value of V at r=rmin. V(i) is the value of V at r=rmin + dr * i where i is chosen such that r >= rmin
    and r <= rmax. V(r) for r < rmin and > rmax is 0. The same goes for F. Thus V and F are defined between the region
    [rmin,rmax], inclusive.

    For ease of storing the data, all tables must be of the same number of points for all bonds.

    \b Interpolation
    Values are interpolated linearly between two points straddling the given r. For a given r, the first point needed, i
    can be calculated via i = floorf((r - rmin) / dr). The fraction between ri and ri+1 can be calculated via
    f = (r - rmin) / dr - float(i). And the linear interpolation can then be performed via V(r) ~= Vi + f * (Vi+1 - Vi)
    \ingroup computes
*/
class BondTablePotential : public ForceCompute
    {
    public:
        //! Constructs the compute
        BondTablePotential(boost::shared_ptr<SystemDefinition> sysdef,
                       unsigned int table_width,
                       const std::string& log_suffix="");

        //! Destructor
        virtual ~BondTablePotential();

        //! Set the table for a given type pair
        virtual void setTable(unsigned int type,
                              const std::vector<float> &V,
                              const std::vector<float> &F,
                              Scalar rmin,
                              Scalar rmax);

        //! Returns a list of log quantities this compute calculates
        virtual std::vector< std::string > getProvidedLogQuantities();

        //! Calculates the requested log value and returns it
        virtual Scalar getLogValue(const std::string& quantity, unsigned int timestep);

    protected:
        boost::shared_ptr<BondData> m_bond_data;    //!< Bond data to use in computing bonds
        unsigned int m_table_width;                 //!< Width of the tables in memory
        GPUArray<float2> m_tables;                  //!< Stored V and F tables
        GPUArray<Scalar4> m_params;                 //!< Parameters stored for each table
        std::string m_log_name;                     //!< Cached log name

        //! Actually compute the forces
        virtual void computeForces(unsigned int timestep);
    };

//! Exports the TablePotential class to python
void export_BondTablePotential();

#endif

