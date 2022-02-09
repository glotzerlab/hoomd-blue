// Copyright (c) 2009-2017 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.


// Maintainer: akohlmey

#include "hoomd/ForceCompute.h"
#include "hoomd/md/NeighborList.h"

#include <memory>

/*! \file PCNDForceCompute.h
    \brief Declares the PCNDForceCompute class
*/

#ifdef __HIPCC__
#error This header cannot be compiled by nvcc
#endif

#include <pybind11/pybind11.h>

#ifndef __PCNDFORCECOMPUTE_H__
#define __PCNDFORCECOMPUTE_H__

namespace hoomd
    {
namespace md
    {
//! Computes PCND forces on each particle
/*! The total pair force is summed for each particle when compute() is called. Forces are only summed between
    neighboring particles with a separation distance less than \c r_cut. A NeighborList must be provided
    to identify these neighbors. Calling compute() in this class will in turn result in a call to the
    NeighborList's compute() to make sure that the neighbor list is up to date.

    Usage: Construct a PCNDForceCompute, providing it an already constructed ParticleData and NeighborList.
    Then set parameters for all possible pairs of types by calling setParams.

    Forces can be computed directly by calling compute() and then retrieved with a call to acquire(), but
    a more typical usage will be to add the force compute to NVEUpdater or NVTUpdater.

    \ingroup computes
*/
class PYBIND11_EXPORT PCNDForceCompute : public ForceCompute
    {
    public:
        //! Constructs the compute
        PCNDForceCompute(std::shared_ptr<SystemDefinition> sysdef,
                         std::shared_ptr<ParticleGroup> group,  
			 std::shared_ptr<NeighborList> nlist,
                         Scalar r_cut);

        //! Destructor
        virtual ~PCNDForceCompute();

        //! Set the parameters for a single type pair
        virtual void setParams(unsigned int typ1, unsigned int typ2, Scalar lj12, Scalar lj9, Scalar lj6, Scalar lj4);

        //! Returns a list of log quantities this compute calculates
        virtual std::vector< std::string > getProvidedLogQuantities();

        //! Calculates the requested log value and returns it
        virtual Scalar getLogValue(const std::string& quantity, uint64_t timestep);

    protected:
        std::shared_ptr<NeighborList> m_nlist;    //!< The neighborlist to use for the computation
	std::shared_ptr<ParticleGroup> m_group;
	Scalar m_r_cut;         //!< Cutoff radius beyond which the force is set to 0
        unsigned int m_ntypes;  //!< Rank of the lj12, lj9, lj6, and lj4 parameter matrices.

        // This is a low level force summing class, it ONLY sums forces, and doesn't do high
        // level concepts like mixing. That is for the caller to handle. So, I only store
        // lj12, lj9, lj6, and lj4 here
        Scalar * __restrict__ m_lj12;   //!< Parameter for computing forces (m_ntypes by m_ntypes array)
        Scalar * __restrict__ m_lj9;    //!< Parameter for computing forces (m_ntypes by m_ntypes array)
        Scalar * __restrict__ m_lj6;    //!< Parameter for computing forces (m_ntypes by m_ntypes array)
        Scalar * __restrict__ m_lj4;    //!< Parameter for computing forces (m_ntypes by m_ntypes array)

        //! Actually compute the forces
        virtual void computeForces(uint64_t timestep);
    };

namespace detail
    {
//! Exports the PCNDForceCompute class to python
void export_PCNDForceCompute(pybind11::module& m);

    } // end namespace detail
    } // end namespace md
    } // end namespace hoomd

#endif
