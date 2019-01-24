// Copyright (c) 2009-2019 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.


// Maintainer: ksil

#include "hoomd/ForceCompute.h"
#include "hoomd/BondedGroupData.h"

#include <memory>
#include <vector>

/*! \file OPLSDihedralForceCompute.h
    \brief Declares a class for computing OPLS dihedrals
*/

#ifdef NVCC
#error This header cannot be compiled by nvcc
#endif

#include <hoomd/extern/pybind/include/pybind11/pybind11.h>

#ifndef __OPLSDIHEDRALFORCECOMPUTE_H__
#define __OPLSDIHEDRALFORCECOMPUTE_H__

//! Computes OPLS dihedral forces on each particle
/*! OPLS dihedral forces are computed on every particle in the simulation.

    The dihedrals which forces are computed on are accessed from ParticleData::getDihedralData
    \ingroup computes
*/
class PYBIND11_EXPORT OPLSDihedralForceCompute : public ForceCompute
    {
    public:
        //! Constructs the compute
        OPLSDihedralForceCompute(std::shared_ptr<SystemDefinition> sysdef);

        //! Destructor
        virtual ~OPLSDihedralForceCompute();

        //! Set the parameters
        virtual void setParams(unsigned int type, Scalar k1, Scalar k2, Scalar k3, Scalar k4);

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
        GPUArray<Scalar4> m_params;

        //!< Dihedral data to use in computing dihedrals
        std::shared_ptr<DihedralData> m_dihedral_data;

        //! Actually compute the forces
        virtual void computeForces(unsigned int timestep);
    };

//! Exports the DihedralForceCompute class to python
void export_OPLSDihedralForceCompute(pybind11::module& m);

#endif
