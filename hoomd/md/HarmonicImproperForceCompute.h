// Copyright (c) 2009-2019 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.


// Maintainer: dnlebard

#include "hoomd/ForceCompute.h"
#include "hoomd/BondedGroupData.h"

#include <memory>

#include <vector>

/*! \file HarmonicImproperForceCompute.h
    \brief Declares a class for computing harmonic impropers
*/

#ifdef NVCC
#error This header cannot be compiled by nvcc
#endif

#include <hoomd/extern/pybind/include/pybind11/pybind11.h>

#ifndef __HARMONICIMPROPERFORCECOMPUTE_H__
#define __HARMONICIMPROPERFORCECOMPUTE_H__

//! Computes harmonic improper forces on each particle
/*! Harmonic improper forces are computed on every particle in the simulation.

    The impropers which forces are computed on are accessed from ParticleData::getImproperData
    \ingroup computes
*/
class PYBIND11_EXPORT HarmonicImproperForceCompute : public ForceCompute
    {
    public:
        //! Constructs the compute
        HarmonicImproperForceCompute(std::shared_ptr<SystemDefinition> sysdef);

        //! Destructor
        virtual ~HarmonicImproperForceCompute();

        //! Set the parameters
        virtual void setParams(unsigned int type, Scalar K, Scalar chi);

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

    protected:
        Scalar *m_K;    //!< K parameter for multiple improper tyes
        Scalar *m_chi;  //!< Chi parameter for multiple impropers

        std::shared_ptr<ImproperData> m_improper_data;    //!< Improper data to use in computing impropers

        //! Actually compute the forces
        virtual void computeForces(unsigned int timestep);
    };

//! Exports the ImproperForceCompute class to python
void export_HarmonicImproperForceCompute(pybind11::module& m);

#endif
