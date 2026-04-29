// Copyright (c) 2009-2025 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

#include "MeshForceCompute.h"

#include <memory>

/*! \file LineTensionForceCompute.h
    \brief Declares a class for computing line tension forces
*/

#ifdef __HIPCC__
#error This header cannot be compiled by nvcc
#endif

#include <pybind11/pybind11.h>

#ifndef __LINETENSIONFORCECOMPUTE_H__
#define __LINETENSIONFORCECOMPUTE_H__

namespace hoomd
{
namespace md
{

//! Parameters for line tension potential
struct line_tension_param_t
{
    Scalar l;
    unsigned int type_i;
    unsigned int type_j;

    line_tension_param_t()
        : l(0), type_i(0), type_j(0) {}

    line_tension_param_t(pybind11::dict params)
    {
        l = pybind11::cast<Scalar>(params["l"]);

        // Important to populate via TypeParameter system to resolve to ints
    }

    pybind11::dict asDict() const
    {
        pybind11::dict d;
        d["l"] = l;
        d["types"] = pybind11::make_tuple(type_i, type_j);
        return d;
    }
};

//! Computes line tension forces on mesh interfaces
class PYBIND11_EXPORT LineTensionForceCompute
    : public MeshForceCompute
{
public:
    //! Constructor
    LineTensionForceCompute(
        std::shared_ptr<SystemDefinition> sysdef,
        std::shared_ptr<MeshDefinition> meshdef);

    //! Destructor
    virtual ~LineTensionForceCompute();

    //! Set parameters
    virtual void setParams(
        unsigned int type,
        const line_tension_param_t& params);

    //! Set parameters from Python
    virtual void setParamsPython(
        std::string type,
        pybind11::dict params);

    //! Get parameters
    pybind11::dict getParams(std::string type);

#ifdef ENABLE_MPI
    CommFlags getRequestedCommFlags(
        uint64_t timestep) override
    {
        CommFlags flags = CommFlags(0);
        flags[comm_flag::tag] = 1;
        flags |=
            ForceCompute::getRequestedCommFlags(
                timestep);
        return flags;
    }
#endif

protected:
    GPUArray<line_tension_param_t> m_params;

    //! Compute forces
    void computeForces(
        uint64_t timestep) override;
};

namespace detail
{
void export_LineTensionForceCompute(
    pybind11::module& m);
}

} // namespace md
} // namespace hoomd

#endif
