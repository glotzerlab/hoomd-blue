// Copyright (c) 2009-2019 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.



#include "hoomd/ForceCompute.h"
#include "hoomd/BondedGroupData.h"

#include <memory>
#include <vector>

/*! \file CosineSqAngleForceCompute.h
    \brief Declares a class for computing cosine squared angles
*/

#ifdef __HIPCC__
#error This header cannot be compiled by nvcc
#endif

#include <pybind11/pybind11.h>

#ifndef __COSINESQANGLEFORCECOMPUTE_H__
#define __COSINESQANGLEFORCECOMPUTE_H__

struct cosinesq_params
    {
    Scalar k;
    Scalar t_0;

    #ifndef __HIPCC__
    cosinesq_params(): k(0), t_0(0) {}

    cosinesq_params(pybind11::dict params)
        :k(params["k"].cast<Scalar>()), t_0(params["t0"].cast<Scalar>()){}

    pybind11::dict asDict()
        {
            pybind11::dict v;
            v["k"] = k;
            v["t0"] = t_0;
            return v;
        }
    #endif
    }
    #ifdef SINGLE_PRECISION
    __attribute__((aligned(8)));
    #else
    __attribute__((aligned(16)));
    #endif

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

        virtual void setParamsPython(std::string type, pybind11::dict params);

        /// Get the parameters for a given type
        virtual pybind11::dict getParams(std::string type);

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
