// Copyright (c) 2009-2024 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

#include "hoomd/BondedGroupData.h"
#include "hoomd/ForceCompute.h"

#include <memory>
#include <vector>

/*! \file OPLSDihedralForceCompute.h
    \brief Declares a class for computing OPLS dihedrals
*/

#ifdef __HIPCC__
#error This header cannot be compiled by nvcc
#endif

#include <pybind11/pybind11.h>

#ifndef __OPLSDIHEDRALFORCECOMPUTE_H__
#define __OPLSDIHEDRALFORCECOMPUTE_H__

namespace hoomd
    {
namespace md
    {
struct dihedral_opls_params
    {
    Scalar k1;
    Scalar k2;
    Scalar k3;
    Scalar k4;

#ifndef __HIPCC__
    dihedral_opls_params() : k1(0.), k2(0.), k3(0.), k4(0.) { }

    dihedral_opls_params(pybind11::dict v)
        : k1(v["k1"].cast<Scalar>()), k2(v["k2"].cast<Scalar>()), k3(v["k3"].cast<Scalar>()),
          k4(v["k4"].cast<Scalar>())
        {
        }

    pybind11::dict asDict()
        {
        pybind11::dict v;
        v["k1"] = k1;
        v["k2"] = k2;
        v["k3"] = k3;
        v["k4"] = k4;
        return v;
        }
#endif
    } __attribute__((aligned(32)));
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

    virtual void setParamsPython(std::string type, pybind11::dict params);

    /// Get the parameters for a specified type
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
    GPUArray<Scalar4> m_params;

    //!< Dihedral data to use in computing dihedrals
    std::shared_ptr<DihedralData> m_dihedral_data;

    //! Actually compute the forces
    virtual void computeForces(uint64_t timestep);
    };

    } // end namespace md
    } // end namespace hoomd

#endif
