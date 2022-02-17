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
    unsigned int PCND_type;
    uint16_t particle_sum;
    Scalar particle_index;

#ifndef __HIPCC__
    angle_pcnd_params() : Xi(0), Tau(0), PCND_type(0), particle_sum(0), particle_index(0) { }

    angle_pcnd_params(pybind11::dict params)
	    : Xi(params["Xi"].cast<Scalar>()), Tau(params["Tau"].cast<Scalar>()),
	      PCND_type(params["PCND_type"].cast<unsigned int>()), particle_sum(params["particle_sum"].cast<uint16_t>()),
	      particle_index(params["particle_index"].cast<Scalar>())
	    {
            }
    
    pybind11::dict asDict()
            {
            pybind11::dict v;
            v["Xi"] = Xi;
	    v["Tau"] = Tau;
	    v["PCND_type"] = PCND_type;
	    v["particle_sum"] = particle_sum;
            v["particle_index"] = particle_index;
	    return v;
	    }
#endif
    }
#ifdef SINGLE_PRECISION
    __attribute__((aligned(8)));
#else
    __attribute__((aligned(16)));
#endif

//! Computes harmonic angle forces for PCND-enabled systems.
/*! Harmonic angle forces are computed on every particle in the simulation.

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
    virtual void setParams(unsigned int type, Scalar Xi, Scalar Tau, unsigned int PCND_type, uint16_t particle_sum, Scalar particle_index);

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
    
    //! Returns a list of log quantities this compute calculates
    virtual std::vector< std::string > getProvidedLogQuantities();

    //! Calculates the requested log value and returns it
    virtual Scalar getLogValue(const std::string& quantity, uint64_t timestep);

    protected:
    Scalar* m_Xi;    //!< K parameter for multiple angle tyes
    Scalar* m_Tau;  //!< t_0 parameter for multiple angle types

    // THESE ARE NEW FOR GC ANGLES
    unsigned int* m_PCND_type; //!< coarse grain angle type index (0-3)
    uint16_t* m_particle_sum;  //!< epsilon parameter for 1-3 repulsion of multiple angle tyes
    Scalar* m_particle_index;//!< sigma parameter for 1-3 repulsion of multiple angle types
    Scalar* m_rcut;//!< cutoff parameter for 1-3 repulsion of multiple angle types

    Scalar prefact[4]; //!< prefact precomputed prefactors for CG-CMM angles
    Scalar cgPow1[4];  //!< list of 1st powers for CG-CMM angles
    Scalar cgPow2[4];  //!< list of 2nd powers for CG-CMM angles

    std::shared_ptr<AngleData> m_pcnd_angle_data; //!< Angle data to use in computing angles

    //! Actually compute the forces
    virtual void computeForces(uint64_t timestep);
    };

namespace detail
    {
//! Exports the PCNDAngleForceCompute class to python
void export_PCNDAngleForceCompute(pybind11::module& m);

    } // end namespace detail
    } // end namespace md
    } // end namespace hoomd

#endif
