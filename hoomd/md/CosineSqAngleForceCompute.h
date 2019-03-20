// Copyright (c) 2009-2019 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.



#include "hoomd/ForceCompute.h"
#include "hoomd/BondedGroupData.h"

#include <memory>
#include <vector>

/*! \file CosineSqAngleForceCompute.h
    \brief Declares a class for computing cosine squared angles
*/

#ifdef NVCC
#error This header cannot be compiled by nvcc
#endif

#include <hoomd/extern/pybind/include/pybind11/pybind11.h>

#ifndef __COSINESQANGLEFORCECOMPUTE_H__
#define __COSINESQANGLEFORCECOMPUTE_H__

//! Computes cosine squared angle forces on each particle
/*! Cosine squared angle forces are computed on every particle in the simulation.

    The angles which forces are computed on are accessed from ParticleData::getAngleData
    \ingroup computes
*/
class PYBIND11_EXPORT CosineSqAngleForceCompute : public ForceCompute
    {
    public:
        //! Constructs the compute
        CosineSqAngleForceCompute(std::shared_ptr<SystemDefinition> sysdef);

        //! Destructor
        virtual ~CosineSqAngleForceCompute();

        //! Set the parameters
        virtual void setParams(unsigned int type, Scalar K, Scalar t_0);

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
        Scalar* m_K;    //!< K parameter for multiple angle types
        Scalar* m_t_0;  //!< r_0 parameter for multiple angle types

        std::shared_ptr<AngleData> m_angle_data;  //!< Angle data to use in computing angles

        //! Actually compute the forces
        virtual void computeForces(unsigned int timestep);
    };

//! Exports the AngleForceCompute class to python
void export_CosineSqAngleForceCompute(pybind11::module& m);

#endif
