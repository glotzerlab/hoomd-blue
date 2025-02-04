// Copyright (c) 2009-2024 The Regents of the University of Michigan.
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

/** Perform constant volume simulation.

    Implement the the Velocity-Verlet integration scheme with MTTK+Langevin piston barostat along
    with optional velocity rescaling Thermostat.

    See:

    * G. J. Martyna, D. J. Tobias, M. L. Klein  1994 <https://dx.doi.org/10.1063/1.467468>
    * S. E. Feller, Y. Zhang, R. W. Pastor 1995 <https://doi.org/10.1063/1.470648>
    * M. E. Tuckerman et. al. 2006 <https://dx.doi.org/10.1088/0305-4470/39/19/S18>
    * T. Yu et. al. 2010 <https://dx.doi.org/10.1016/j.chemphys.2010.02.014>

*/
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

    /// Define possible couplings between the diagonal elements of the pressure tensor
    enum couplingMode
        {
        couple_none = 0,
        couple_xy,
        couple_xz,
        couple_yz,
        couple_xyz
        };

    /// Define possible barostat degrees of freedom.
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

    /// Get the barostat time constant.
    virtual Scalar getTauS()
        {
        return m_tauS;
        }

    /** Set the target values of the stress tensor.

        @param S list of stress components: [xx, yy, zz, yz, xz, xy]
    */
    virtual void setS(const std::vector<std::shared_ptr<Variant>>& S)
        {
        m_S = S;
        }

    /// Choose random initial values for the barostat degrees of freedom.
    void thermalizeBarostatDOF(uint64_t timestep);

    /// Get the coupling mode as a string (for use in the Python API).
    std::string getCouple();

    /// Get the barostat degrees of freedom as a bool vector.
    std::vector<bool> getFlags();

    /// Returns true when all particles are scaled, false when only the integration group is.
    bool getRescaleAll()
        {
        return m_rescale_all;
        }

    /// Get needed pdata flags.
    virtual PDataFlags getRequestedPDataFlags()
        {
        // Always compute the pressure tensor and the external field virials. Compute the rotational
        // kinetic energy when integrating anistropic degrees of freedom.

        PDataFlags flags;
        flags[pdata_flag::pressure_tensor] = 1;
        if (m_aniso)
            {
            flags[pdata_flag::rotational_kinetic_energy] = 1;
            }
        flags[pdata_flag::external_field_virial] = 1;
        return flags;
        }

    /** Performs the first half-step of the integration.

        @param timestep Current simulation timestep.

        @post Particle positions are moved forward to timestep+1 and velocities to timestep+1/2 per
        the Velocity-Verlet method. Advance the barostat 1/2 step.
    */
    virtual void integrateStepOne(uint64_t timestep);

    /** Performs the second half-step of the integration.

        @param timestep Current simulation timestep.

        @post Particle velocities are moved forward to timestep+1. Advance the barostat 1/2 step.
    */
    virtual void integrateStepTwo(uint64_t timestep);

    // Get the target stress tensor.
    std::vector<std::shared_ptr<Variant>> getS()
        {
        return m_S;
        }

    /// Set the barostat time constant.
    void setTauS(Scalar tauS)
        {
        m_tauS = tauS;
        }

    /// Set the coupling mode by string name (for use with the Python API).
    void setCouple(const std::string& value);

    /// Set the degrees of freedom flags.
    void setFlags(const std::vector<bool>& value);

    /// Set rescale_all=true to rescale all particles, false to only rescale the integration group.
    void setRescaleAll(bool rescale_all)
        {
        m_rescale_all = rescale_all;
        }

    /// Get the barostat degrees of freedom as a Python tuple.
    pybind11::tuple getBarostatDOF();

    /// Set the barostat degrees of freedom from a Python tuple.
    void setBarostatDOF(pybind11::tuple v);

    /// Get the barostat contribution to the Hamiltonian of the system.
    Scalar getBarostatEnergy(uint64_t timestep);

    /// Set the thermostat to use when rescaling velocities (may be null).
    void setThermostat(std::shared_ptr<Thermostat> thermostat)
        {
        m_thermostat = thermostat;
        }

    protected:
    /// The barostat degrees of freedom.
    Barostat m_barostat {};

    /// Target stress matrix (upper triangular, components [xx, yy, zz, yz, xz, xy])/
    std::vector<std::shared_ptr<Variant>> m_S;

    /// Current box volume.
    Scalar m_V;

    /// Barostat time constant.
    Scalar m_tauS;

    /// Cached number of translational degrees of freedom.
    Scalar m_ndof;

    /// The coupling mode to use.
    couplingMode m_couple;

    /// Bit flag field that sets which box degrees of freedom to integrate.
    unsigned int m_flags;

    /// Matrix exponential for velocity update (upper triangular).
    Scalar m_mat_exp_v[6];

    /// Matrix exponential for position update (upper triangular).
    Scalar m_mat_exp_r[6];

    /// Integrated matrix exp. for velocity update (upper triangular)
    Scalar m_mat_exp_r_int[6];

    /// Damping coefficient in the Langevin piston.
    Scalar m_gamma;

    /// Thermostat to use when rescaling velocities (may be null).
    std::shared_ptr<Thermostat> m_thermostat;

    /// Compute thermodynamic properties of the integrated group.
    std::shared_ptr<ComputeThermo> m_thermo_full_step;

    /// When true, rescale all particles in the system irrespective of group.
    bool m_rescale_all;

    /// Helper function to update the propagator elements.
    void updatePropagator();

    /// Get the relevant couplings for the active box degrees of freedom.
    couplingMode getRelevantCouplings();

    /// Helper function to advance the barostat parameters.
    virtual void advanceBarostat(uint64_t timestep);
    };

    } // namespace hoomd::md

#endif // HOOMD_TWOSTEPCONSTANTPRESSURE_H
