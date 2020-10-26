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

#ifdef __HIPCC__
#error This header cannot be compiled by nvcc
#endif

#include <pybind11/pybind11.h>

#ifndef __HARMONICDIHEDRALFORCECOMPUTE_H__
#define __HARMONICDIHEDRALFORCECOMPUTE_H__

struct dihedral_harmonic_params
    {
    Scalar k;
    Scalar d;
    Scalar n;
    Scalar phi_0;

    #ifndef __HIPCC__
    dihedral_harmonic_params(): k(0.), d(0.), n(0.), phi_0(0.){}

    dihedral_harmonic_params(pybind11::dict v)
        : k(v["k"].cast<Scalar>()), d(v["d"].cast<Scalar>()),
          n(v["n"].cast<Scalar>()), phi_0(v["phi0"].cast<Scalar>()){}

    pybind11::dict asDict()
        {
        pybind11::dict v;
        v["k"] = k;
        v["d"] = d;
        v["n"] = n;
        v["phi0"] = phi_0;
        return v;
        }
    #endif
    }
    __attribute__((aligned(32)));

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

        virtual void setParamsPython(std::string type, pybind11::dict params);

        /// Get the parameters for a particular type
        pybind11::dict getParams(std::string type);

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
