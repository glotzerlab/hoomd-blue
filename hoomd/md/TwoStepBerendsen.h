// Copyright (c) 2009-2022 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

#include "ComputeThermo.h"
#include "IntegrationMethodTwoStep.h"
#include "TwoStepNVTBase.h"
#include "hoomd/Variant.h"

// inclusion guard
#ifndef __BERENDSEN_H__
#define __BERENDSEN_H__

/*! \file TwoStepBerendsen.h
    \brief Declaration of Berendsen thermostat
*/

#ifdef __HIPCC__
#error This header cannot be compiled by nvcc
#endif

#include <pybind11/pybind11.h>

namespace hoomd
    {
namespace md
    {
/*! Implements the Berendsen thermostat \cite Berendsen1984
 */
class PYBIND11_EXPORT TwoStepBerendsen : public virtual TwoStepNVTBase
    {
    public:
    //! Constructor
    TwoStepBerendsen(std::shared_ptr<SystemDefinition> sysdef,
                     std::shared_ptr<ParticleGroup> group,
                     std::shared_ptr<ComputeThermo> thermo,
                     Scalar tau,
                     std::shared_ptr<Variant> T);
    virtual ~TwoStepBerendsen();

    //! Update the tau value
    //! \param tau New time constant to set
    virtual void setTau(Scalar tau)
        {
        m_tau = tau;
        }

    //! Get the tau value
    virtual Scalar getTau()
        {
        return m_tau;
        }

    virtual std::array<Scalar, 2> NVT_rescale_factor_one(uint64_t timestep);


    protected:
    Scalar m_tau;                                  //!< time constant for Berendsen thermostat
    };

    } // end namespace md
    } // end namespace hoomd

#endif // _BERENDSEN_H_
