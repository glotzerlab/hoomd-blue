// Copyright (c) 2009-2019 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.


// Maintainer: dnlebard

#include "hoomd/ForceCompute.h"
#include "hoomd/BondedGroupData.h"

#include <memory>

#include <vector>

/*! \file HarmonicDihedralForceCompute.h
    \brief Declares a class for computing harmonic dihedrals
*/

#ifdef NVCC
#error This header cannot be compiled by nvcc
#endif

#include <hoomd/extern/pybind/include/pybind11/pybind11.h>

#ifndef __HARMONICDIHEDRALFORCECOMPUTE_H__
#define __HARMONICDIHEDRALFORCECOMPUTE_H__

//! Computes harmonic dihedral forces on each particle
/*! Harmonic dihedral forces are computed on every particle in the simulation.

    The dihedrals which forces are computed on are accessed from ParticleData::getDihedralData
    \ingroup computes
*/
class PYBIND11_EXPORT HarmonicDihedralForceCompute : public ForceCompute
    {
    public:
        //! Constructs the compute
        HarmonicDihedralForceCompute(std::shared_ptr<SystemDefinition> sysdef);

        //! Destructor
        virtual ~HarmonicDihedralForceCompute();

        //! Set the parameters
        virtual void setParams(unsigned int type, Scalar K, int sign, unsigned int multiplicity, Scalar phi_0);

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
        Scalar *m_K;     //!< K parameter for multiple dihedral tyes
        Scalar *m_sign;  //!< sign parameter for multiple dihedral types
        Scalar *m_multi; //!< multiplicity parameter for multiple dihedral types
        Scalar *m_phi_0; //!< phi_0 parameter for multiple dihedral types

        std::shared_ptr<DihedralData> m_dihedral_data;    //!< Dihedral data to use in computing dihedrals

        //! Actually compute the forces
        virtual void computeForces(unsigned int timestep);
    };

//! Exports the DihedralForceCompute class to python
void export_HarmonicDihedralForceCompute(pybind11::module& m);

#endif
