// Copyright (c) 2009-2021 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.

// Maintainer: joaander

#include "ComputeThermo.h"
#include "IntegrationMethodTwoStep.h"
#include "hoomd/Variant.h"

#ifndef __TWO_STEP_NVT_MTK_H__
#define __TWO_STEP_NVT_MTK_H__

/*! \file TwoStepNVTMTK.h
    \brief Declares the TwoStepNVTMTK class
*/

#ifdef __HIPCC__
#error This header cannot be compiled by nvcc
#endif

#include <pybind11/pybind11.h>

namespace hoomd
    {
namespace md
    {
//! Integrates part of the system forward in two steps in the NVT ensemble
/*! Implements Martyna-Tobias-Klein (MTK) NVT integration through the IntegrationMethodTwoStep
   interface

    Integrator variables mapping:
     - [0] -> xi
     - [1] -> eta

    The instantaneous temperature of the system is computed with the provided ComputeThermo. Correct
   dynamics require that the thermo computes the temperature of the assigned group and with D*N-D
   degrees of freedom. TwoStepNVTMTK does not check for these conditions.

    For the update equations of motion, see Refs. \cite{Martyna1994,Martyna1996}

    \ingroup updaters
*/
class PYBIND11_EXPORT TwoStepNVTMTK : public IntegrationMethodTwoStep
    {
    public:
    //! Constructs the integration method and associates it with the system
    TwoStepNVTMTK(std::shared_ptr<SystemDefinition> sysdef,
                  std::shared_ptr<ParticleGroup> group,
                  std::shared_ptr<ComputeThermo> thermo,
                  Scalar tau,
                  std::shared_ptr<Variant> T);
    virtual ~TwoStepNVTMTK();

    //! Update the temperature
    /*! \param T New temperature to set
     */
    virtual void setT(std::shared_ptr<Variant> T)
        {
        m_T = T;
        }

    /// Get the current temperature variant
    std::shared_ptr<Variant> getT()
        {
        return m_T;
        }

    //! Update the tau value
    /*! \param tau New time constant to set
     */
    virtual void setTau(Scalar tau)
        {
        m_tau = tau;
        }

    /// get the tau value
    Scalar getTau()
        {
        return m_tau;
        }

    //! Set the value of xi (for unit tests)
    void setXi(Scalar new_xi)
        {
        IntegratorVariables v = getIntegratorVariables();
        Scalar& xi = v.variable[0];
        xi = new_xi;
        setIntegratorVariables(v);
        }

    //! Performs the first step of the integration
    virtual void integrateStepOne(uint64_t timestep);

    //! Performs the second step of the integration
    virtual void integrateStepTwo(uint64_t timestep);

    //! Get needed pdata flags
    /*! in anisotropic mode, we need the rotational kinetic energy
     */
    virtual PDataFlags getRequestedPDataFlags()
        {
        PDataFlags flags;
        if (m_aniso)
            flags[pdata_flag::rotational_kinetic_energy] = 1;
        return flags;
        }

    //! Initialize integrator variables
    virtual void initializeIntegratorVariables()
        {
        IntegratorVariables v = getIntegratorVariables();
        v.type = "nvt_mtk";
        v.variable.clear();
        v.variable.resize(4);
        v.variable[0] = Scalar(0.0);
        v.variable[1] = Scalar(0.0);
        v.variable[2] = Scalar(0.0);
        v.variable[3] = Scalar(0.0);
        setIntegratorVariables(v);
        }

    /// Randomize the thermostat variables
    void thermalizeThermostatDOF(uint64_t timestep);

    /// Get the translational thermostat degrees of freedom
    pybind11::tuple getTranslationalThermostatDOF();

    /// Set the translational thermostat degrees of freedom
    void setTranslationalThermostatDOF(pybind11::tuple v);

    /// Get the rotational thermostat degrees of freedom
    pybind11::tuple getRotationalThermostatDOF();

    /// Set the rotational thermostat degrees of freedom
    void setRotationalThermostatDOF(pybind11::tuple v);

    Scalar getThermostatEnergy(uint64_t timestep);

    protected:
    std::shared_ptr<ComputeThermo> m_thermo; //!< compute for thermodynamic quantities

    Scalar m_tau;                 //!< tau value for Nose-Hoover
    std::shared_ptr<Variant> m_T; //!< Temperature set point

    Scalar m_exp_thermo_fac; //!< Thermostat rescaling factor

    //! advance the thermostat
    /*!\param timestep The time step
     * \param broadcast True if we should broadcast the integrator variables via MPI
     */
    void advanceThermostat(uint64_t timestep, bool broadcast = true);
    };

namespace detail
    {
//! Exports the TwoStepNVTMTK class to python
void export_TwoStepNVTMTK(pybind11::module& m);

    } // end namespace detail
    } // end namespace md
    } // end namespace hoomd

#endif // #ifndef __TWO_STEP_NVT_MTK_H__
