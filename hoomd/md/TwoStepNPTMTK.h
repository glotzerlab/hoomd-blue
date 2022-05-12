// Copyright (c) 2009-2022 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

#include "ComputeThermo.h"
#include "IntegrationMethodTwoStep.h"
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
class PYBIND11_EXPORT TwoStepNPTMTK : public IntegrationMethodTwoStep
    {
    public:
    //! Specify possible couplings between the diagonal elements of the pressure tensor
    enum couplingMode
        {
        couple_none = 0,
        couple_xy,
        couple_xz,
        couple_yz,
        couple_xyz
        };

    /*! Flags to indicate which degrees of freedom of the simulation box should be put under
        barostat control
     */
    enum baroFlags
        {
        baro_x = 1,
        baro_y = 2,
        baro_z = 4,
        baro_xy = 8,
        baro_xz = 16,
        baro_yz = 32
        };

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

    //! Update the temperature
    /*! \param T New temperature to set
     */
    virtual void setT(std::shared_ptr<Variant> T)
        {
        m_T = T;
        }

    //! Update the stress components
    /*! \param S list of stress components: [xx, yy, zz, yz, xz, xy]
     */
    virtual void setS(const std::vector<std::shared_ptr<Variant>>& S)
        {
        m_S = S;
        }

    //! Update the tau value
    /*! \param tau New time constant to set
     */
    virtual void setTau(Scalar tau)
        {
        m_tau = tau;
        }

    //! Update the nuP value
    /*! \param tauS New pressure constant to set
     */
    virtual void setTauS(Scalar tauS)
        {
        m_tauS = tauS;
        }

    //! Set the scale all particles option
    /*! \param rescale_all If true, rescale all particles
     */
    void setRescaleAll(bool rescale_all)
        {
        m_rescale_all = rescale_all;
        }

    //! Set an optional damping factor for the box degrees of freedom
    void setGamma(Scalar gamma)
        {
        m_gamma = gamma;
        }
    //! declaration for setting the parameter couple
    void setCouple(const std::string& value);

    //! declaration for setting the parameter flags
    void setFlags(const std::vector<bool>& value);

    //! Get temperature
    virtual std::shared_ptr<Variant> getT()
        {
        return m_T;
        }

    // Get stress
    std::vector<std::shared_ptr<Variant>> getS()
        {
        return m_S;
        }

    // Get tau
    virtual Scalar getTau()
        {
        return m_tau;
        }

    // Get tauS
    virtual Scalar getTauS()
        {
        return m_tauS;
        }

    // Get rescale_all
    bool getRescaleAll()
        {
        return m_rescale_all;
        }

    // Get gamma
    Scalar getGamma()
        {
        return m_gamma;
        }

    // declaration get function of couple
    std::string getCouple();

    // declaration get function of flags
    std::vector<bool> getFlags();

    //! Performs the first step of the integration
    virtual void integrateStepOne(uint64_t timestep);

    //! Performs the second step of the integration
    virtual void integrateStepTwo(uint64_t timestep);

    //! Get needed pdata flags
    /*! TwoStepNPTMTK needs the pressure, so the pressure_tensor flag is set
     */
    virtual PDataFlags getRequestedPDataFlags()
        {
        PDataFlags flags;
        flags[pdata_flag::pressure_tensor] = 1;
        if (m_aniso)
            {
            flags[pdata_flag::rotational_kinetic_energy] = 1;
            }
        flags[pdata_flag::external_field_virial] = 1;
        return flags;
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

    /// Get the barostat degrees of freedom
    pybind11::tuple getBarostatDOF();

    /// Set the barostat degrees of freedom
    void setBarostatDOF(pybind11::tuple v);

    Scalar getBarostatEnergy(uint64_t timestep);

    protected:
    /// Thermostat variables
    struct Thermostat
        {
        Scalar xi = 0;
        Scalar eta = 0;
        Scalar xi_rot = 0;
        Scalar eta_rot = 0;
        };

    /// Barostat variables
    struct Barostat
        {
        Scalar nu_xx;
        Scalar nu_xy;
        Scalar nu_xz;
        Scalar nu_yy;
        Scalar nu_yz;
        Scalar nu_zz;
        };

    std::shared_ptr<ComputeThermo>
        m_thermo_half_step; //!< ComputeThermo operating on the integrated group at t+dt/2
    std::shared_ptr<ComputeThermo>
        m_thermo_full_step; //!< ComputeThermo operating on the integrated group at t
    Scalar m_ndof;          //!< Number of degrees of freedom from ComputeThermo

    Scalar m_tau;                 //!< tau value for Nose-Hoover
    Scalar m_tauS;                //!< tauS value for the barostat
    std::shared_ptr<Variant> m_T; //!< Temperature set point
    std::vector<std::shared_ptr<Variant>>
        m_S;    //!< Stress matrix (upper diagonal, components [xx, yy, zz, yz, xz, xy])
    Scalar m_V; //!< Current volume

    couplingMode m_couple;     //!< Coupling of diagonal elements
    unsigned int m_flags;      //!< Coupling flags for barostat
    bool m_nph;                //!< True if integrating without thermostat
    Scalar m_mat_exp_v[6];     //!< Matrix exponential for velocity update (upper triangular)
    Scalar m_mat_exp_r[6];     //!< Matrix exponential for position update (upper triangular)
    Scalar m_mat_exp_r_int[6]; //!< Integrated matrix exp. for velocity update (upper triangular)

    bool m_rescale_all; //!< If true, rescale all particles in the system irrespective of group

    Scalar m_gamma; //!< Optional damping factor for box degrees of freedom

    Thermostat m_thermostat; //!< thermostat degrees of freedom
    Barostat m_barostat;     //!< barostat degrees of freedom

    //! Helper function to advance the barostat parameters
    void advanceBarostat(uint64_t timestep);

    //! advance the thermostat
    /*!\param timestep The time step
     * \param broadcast True if we should broadcast the integrator variables via MPI
     */
    void advanceThermostat(uint64_t timestep);

    //! Helper function to update the propagator elements
    void updatePropagator();

    //! Get the relevant couplings for the active box degrees of freedom
    couplingMode getRelevantCouplings();
    };

    } // end namespace md
    } // end namespace hoomd

#endif // #ifndef __TWO_STEP_NPT_MTK_H__
