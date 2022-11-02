// Copyright (c) 2009-2022 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

#include "ComputeThermo.h"
#include "IntegrationMethodTwoStep.h"
#include "TwoStepNPTMTTKBase.h"
#include "hoomd/Variant.h"

#ifndef __TWO_STEP_NPT_MTK_H__
#define __TWO_STEP_NPT_MTK_H__

/*! \file TwoStepNPTMTK.h
    \brief Declares the TwoStepNPTMTK class
*/

#ifdef __HIPCC__
#error This header cannot be compiled by nvcc
#endif

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace hoomd
    {
namespace md
    {
//! Integrates part of the system forward in two steps in the NPT ensemble
/*! Implements the Martyna Tobias Klein (MTK) equations for rigorous integration in the NPT
   ensemble. The update equations are derived from a strictly measure-preserving and time-reversal
   symmetric integration scheme, closely following the one proposed by Tuckerman et al.

    Supports anisotropic (orthorhombic or tetragonal) integration modes, by implementing a special
    version of the the fully flexible cell update equations proposed in Yu et al.

    Triclinic integration for an upper triangular cell parameter matrix is supported with
    fully time-reversible and measure-preserving update equations (Glaser et al. 2013 to be
   published)

    \cite Martyna1994
    \cite Tuckerman2006
    \cite Yu2010
    \ingroup updaters
*/
class PYBIND11_EXPORT TwoStepNPTMTK : public virtual TwoStepNPTMTTKBase
    {
    public:
    /*! Flags to indicate which degrees of freedom of the simulation box should be put under
        barostat control
     */


    //! Constructs the integration method and associates it with the system
    TwoStepNPTMTK(std::shared_ptr<SystemDefinition> sysdef,
                  std::shared_ptr<ParticleGroup> group,
                  std::shared_ptr<ComputeThermo> thermo_half_step,
                  std::shared_ptr<ComputeThermo> thermo_full_step,
                  Scalar tau,
                  Scalar tauS,
                  std::shared_ptr<Variant> T,
                  const std::vector<std::shared_ptr<Variant>>& S,
                  const std::string& couple,
                  const std::vector<bool>& flags,
                  const bool nph = false);

    virtual ~TwoStepNPTMTK();

    //! Update the tau value
    /*! \param tau New time constant to set
     */
    void setTau(Scalar tau)
        {
        m_tau = tau;
        }

    /// get the tau value
    Scalar getTau()
        {
        return m_tau;
        }

    //! Set an optional damping factor for the box degrees of freedom
    void setGamma(Scalar gamma)
        {
        m_gamma = gamma;
        }

    // Get gamma
    Scalar getGamma()
        {
        return m_gamma;
        }


    /// Randomize the thermostat and barostat variables
    void thermalizeThermostatAndBarostatDOF(uint64_t timestep);

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
    /// Thermostat variables
    struct Thermostat
        {
        Scalar xi = 0;
        Scalar eta = 0;
        Scalar xi_rot = 0;
        Scalar eta_rot = 0;
        };

    std::array<Scalar, 2> NPT_thermo_rescale_factor_one(uint64_t timestep) override;
    std::array<Scalar, 2> NPT_thermo_rescale_factor_two(uint64_t timestep) override;

    //Scalar m_ndof;          //!< Number of degrees of freedom from ComputeThermo
    Scalar m_tau;                 //!< tau value for Nose-Hoover
    Scalar m_gamma; //!< Optional damping factor for box degrees of freedom
    Thermostat m_thermostat; //!< thermostat degrees of freedom


    //! Helper function to advance the barostat parameters
    virtual void advanceBarostat(uint64_t timestep);

    virtual void advanceThermostat(uint64_t timestep);

   // void updatePropagator() override;
    };

    } // end namespace md
    } // end namespace hoomd

#endif // #ifndef __TWO_STEP_NPT_MTK_H__
