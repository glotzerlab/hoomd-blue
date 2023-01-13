// Copyright (c) 2009-2022 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

#ifndef HOOMD_TWOSTEPCONSTANTPRESSURE_H
#define HOOMD_TWOSTEPCONSTANTPRESSURE_H

#include "ComputeThermo.h"
#include "IntegrationMethodTwoStep.h"
// #include "TwoStepConstantVolume.h"

#include "Thermostat.h"
#include "hoomd/Variant.h"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace hoomd::md
    {

class PYBIND11_EXPORT TwoStepConstantPressure : public IntegrationMethodTwoStep
    {
    public:
    TwoStepConstantPressure(std::shared_ptr<SystemDefinition> sysdef,
                            std::shared_ptr<ParticleGroup> group,
                            std::shared_ptr<ComputeThermo> thermo_full_step,
                            Scalar tauS,
                            const std::vector<std::shared_ptr<Variant>>& S,
                            const std::string& couple,
                            const std::vector<bool>& flags,
                            std::shared_ptr<Thermostat> thermostat,
                            Scalar gamma);

    //! Specify possible couplings between the diagonal elements of the pressure tensor
    enum couplingMode
        {
        couple_none = 0,
        couple_xy,
        couple_xz,
        couple_yz,
        couple_xyz
        };
    enum baroFlags
        {
        baro_x = 1,
        baro_y = 2,
        baro_z = 4,
        baro_xy = 8,
        baro_xz = 16,
        baro_yz = 32
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
    // Get tauS
    virtual Scalar getTauS()
        {
        return m_tauS;
        }

    //! Update the stress components
    /*! \param S list of stress components: [xx, yy, zz, yz, xz, xy]
     */
    virtual void setS(const std::vector<std::shared_ptr<Variant>>& S)
        {
        m_S = S;
        }

    void thermalizeBarostatDOF(uint64_t timestep);

    // declaration get function of couple
    std::string getCouple();

    // declaration get function of flags
    std::vector<bool> getFlags();

    // Get rescale_all
    bool getRescaleAll()
        {
        return m_rescale_all;
        }

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

    //! Performs the first step of the integration
    virtual void integrateStepOne(uint64_t timestep);

    //! Performs the second step of the integration
    virtual void integrateStepTwo(uint64_t timestep);

    // Get stress
    std::vector<std::shared_ptr<Variant>> getS()
        {
        return m_S;
        }
    //! Update the nuP value
    /*! \param tauS New pressure constant to set
     */
    void setTauS(Scalar tauS)
        {
        m_tauS = tauS;
        }
    //! declaration for setting the parameter couple
    void setCouple(const std::string& value);

    //! declaration for setting the parameter flags
    void setFlags(const std::vector<bool>& value);

    //! Set the scale all particles option
    /*! \param rescale_all If true, rescale all particles
     */
    void setRescaleAll(bool rescale_all)
        {
        m_rescale_all = rescale_all;
        }

    /// Get the barostat degrees of freedom
    pybind11::tuple getBarostatDOF();

    /// Set the barostat degrees of freedom
    void setBarostatDOF(pybind11::tuple v);

    Scalar getBarostatEnergy(uint64_t timestep);

    void setThermostat(std::shared_ptr<Thermostat> thermostat)
        {
        m_thermostat = thermostat;
        }

    protected:
    Barostat m_barostat {}; //!< barostat degrees of freedom

    std::vector<std::shared_ptr<Variant>>
        m_S;       //!< Stress matrix (upper diagonal, components [xx, yy, zz, yz, xz, xy])
    Scalar m_V;    //!< Current volume
    Scalar m_tauS; //!< tauS value for the barostat
    Scalar m_ndof;
    couplingMode m_couple;     //!< Coupling of diagonal elements
    unsigned int m_flags;      //!< Coupling flags for barostat
    Scalar m_mat_exp_v[6];     //!< Matrix exponential for velocity update (upper triangular)
    Scalar m_mat_exp_r[6];     //!< Matrix exponential for position update (upper triangular)
    Scalar m_mat_exp_r_int[6]; //!< Integrated matrix exp. for velocity update (upper triangular)
    Scalar m_gamma;
    std::shared_ptr<Thermostat> m_thermostat;
    std::shared_ptr<ComputeThermo>
        m_thermo_full_step; //!< ComputeThermo operating on the integrated group at t
    bool m_rescale_all;     //!< If true, rescale all particles in the system irrespective of group

    //! Helper function to update the propagator elements
    void updatePropagator();

    //! Get the relevant couplings for the active box degrees of freedom
    couplingMode getRelevantCouplings();

    //! Helper function to advance the barostat parameters
    virtual void advanceBarostat(uint64_t timestep);
    };

    } // namespace hoomd::md

#endif // HOOMD_TWOSTEPCONSTANTPRESSURE_H
