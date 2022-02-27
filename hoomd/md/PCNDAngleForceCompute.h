// Copyright (c) 2009-2017 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.

#include "hoomd/ForceCompute.h"
#include "hoomd/BondedGroupData.h"

#include <memory>

#include <vector>

/*! \file PCNDAngleForceCompute.h
    \brief Declares a class for computing harmonic bonds
*/

#ifdef __HIPCC__
#error This header cannot be compiled by nvcc
#endif

#include <pybind11/pybind11.h>

#ifndef __PCNDANGLEFORCECOMPUTE_H__
#define __PCNDANGLEFORCECOMPUTE_H__

namespace hoomd
    {
namespace md
    {
struct angle_pcnd_params
    {
    Scalar Xi;
    Scalar Tau;

#ifndef __HIPCC__
    angle_pcnd_params() : Xi(0), Tau(0) { }

    angle_pcnd_params(pybind11::dict params)
	    : Xi(params["Xi"].cast<Scalar>()), Tau(params["Tau"].cast<Scalar>())
	    {
            }
    
    pybind11::dict asDict()
            {
            pybind11::dict v;
            v["Xi"] = Xi;
	    v["Tau"] = Tau;
	    return v;
	    }
#endif
    }
#ifdef SINGLE_PRECISION
    __attribute__((aligned(8)));
#else
    __attribute__((aligned(16)));
#endif

//! Computes PCND angle forces on the central particle of each angle in PCND-enabled systems.
/*! PCND angle forces are computed on every particle in the simulation, OTHER THAN POLYMER END BEADS.

    The angles which forces are computed on are accessed from ParticleData::getAngleData
    \ingroup computes
*/
class PYBIND11_EXPORT PCNDAngleForceCompute : public ForceCompute
    {
    public:
    //! Constructs the compute
    PCNDAngleForceCompute(std::shared_ptr<SystemDefinition> sysdef);

    //! Destructor
    virtual ~PCNDAngleForceCompute();

    //! Set the parameters
    virtual void setParams(unsigned int type, Scalar Xi, Scalar Tau);

    virtual void setParamsPython(std::string type, pybind11::dict params);

    /// Get the parameters for a type
    pybind11::dict getParams(std::string type);

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
    Scalar* m_Xi;    //!< Xi parameter for multiple angle tyes
    Scalar* m_Tau;  //!< Tau parameter for multiple angle types
    
    std::shared_ptr<AngleData> m_pcnd_angle_data; //!< Angle data to use in computing angles

    //! Actually compute the forces
    virtual void computeForces(uint64_t timestep, uint64_t PCNDtimestep);
    };

namespace detail
    {
//! Exports the PCNDAngleForceCompute class to python
void export_PCNDAngleForceCompute(pybind11::module& m);

    } // end namespace detail
    } // end namespace md
    } // end namespace hoomd

#endif
