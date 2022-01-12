// Copyright (c) 2009-2022 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

#include "hoomd/ForceCompute.h"

#include <map>
#include <memory>

/*! \file CustomForceCompute.h
    \brief Declares the backend for computing custom forces in python classes
*/

#ifdef __HIPCC__
#error This header cannot be compiled by nvcc
#endif

#include <pybind11/pybind11.h>

#ifndef __CUSTOMFORCECOMPUTE_H__
#define __CUSTOMFORCECOMPUTE_H__

namespace hoomd
    {
namespace md
    {
//! Adds a custom force
/*! \ingroup computes
 */
class PYBIND11_EXPORT CustomForceCompute : public ForceCompute
    {
    public:
    //! Constructs the compute
    CustomForceCompute(std::shared_ptr<hoomd::SystemDefinition> sysdef,
                       pybind11::object py_setForces);

    //! Destructor
    ~CustomForceCompute();

    bool isAnisotropic()
        {
        return m_aniso;
        }

    void setAnisotropic(bool aniso)
        {
        m_aniso = aniso;
        }

    protected:
    //! Actually compute the forces
    virtual void computeForces(uint64_t timestep);

    private:
    //! A python callback when the force is updated
    pybind11::object m_setForces;

    //! flag for anisotropic python custom forces
    bool m_aniso = false;
    };

namespace detail
    {
//! Exports the CustomForceComputeClass to python
void export_CustomForceCompute(pybind11::module& m);

    } // end namespace detail
    } // end namespace md
    } // end namespace hoomd

#endif
