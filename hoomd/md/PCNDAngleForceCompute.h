// Copyright (c) 2009-2017 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.


// Maintainer: dnlebard

#include "hoomd/ForceCompute.h"
#include "hoomd/BondedGroupData.h"

#include <memory>

#include <vector>

/*! \file HarmonicAngleForceCompute.h
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
//! Computes harmonic angle forces for PCND coarse grain systems.
/*! Harmonic angle forces are computed on every particle in the simulation.

    The angles which forces are computed on are accessed from ParticleData::getAngleData
    \ingroup computes
*/
class PYBIND11_EXPORT PCNDAngleForceCompute : public ForceCompute
    {
    public:
        //! Constructs the compute
        PCNDAngleForceCompute(std::shared_ptr<SystemDefinition> pdata);

        //! Destructor
        ~PCNDAngleForceCompute();

        //! Set the parameters
        virtual void setParams(unsigned int type, Scalar K, Scalar t_0, unsigned int cg_type, uint16_t eps, Scalar sigma);

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
        Scalar *m_K;    //!< K parameter for multiple angle tyes
        Scalar *m_t_0;  //!< t_0 parameter for multiple angle types

        // THESE ARE NEW FOR PCND ANGLES
        Scalar *m_eps;  //!< epsilon parameter for 1-3 repulsion of multiple angle tyes
        Scalar *m_sigma;//!< sigma parameter for 1-3 repulsion of multiple angle types
        Scalar *m_rcut;//!< cutoff parameter for 1-3 repulsion of multiple angle types
        unsigned int *m_cg_type; //!< coarse grain angle type index (0-3)

        Scalar prefact[4]; //!< prefact precomputed prefactors for PCND angles
        Scalar cgPow1[4];  //!< list of 1st powers for PCND angles
        Scalar cgPow2[4];  //!< list of 2nd powers for PCND angles

        std::shared_ptr<AngleData> m_PCNDAngle_data; //!< Angle data to use in computing angles

        //! Actually compute the forces
        virtual void computeForces(unsigned int timestep);
    };

namespace detail
    {
//! Exports the BondForceCompute class to python
void export_PCNDAngleForceCompute(pybind11::module& m);

    } // end namespace detail
    } // end namespace md
    } // end namespace hoomd

#endif
