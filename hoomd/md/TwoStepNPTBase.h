//
// Created by girard01 on 10/26/22.
//

#ifndef HOOMD_TWOSTEPNPTBASE_H
#define HOOMD_TWOSTEPNPTBASE_H

#include "ComputeThermo.h"
#include "IntegrationMethodTwoStep.h"
//#include "TwoStepNVTBase.h"

#include "hoomd/Variant.h"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace hoomd::md
    {

class PYBIND11_EXPORT TwoStepNPTBase : public IntegrationMethodTwoStep
    {
    public:

    TwoStepNPTBase(std::shared_ptr<SystemDefinition> sysdef,
                   std::shared_ptr<ParticleGroup> group,
                   std::shared_ptr<ComputeThermo> thermo_half_step,
                   std::shared_ptr<ComputeThermo> thermo_full_step,
                   std::shared_ptr<Variant> T,
                   const std::vector<std::shared_ptr<Variant>>& S,
                   const std::string& couple,
                   const std::vector<bool>& flags,
                   const bool nph = false);

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


    //! Update the stress components
    /*! \param S list of stress components: [xx, yy, zz, yz, xz, xy]
     */
    virtual void setS(const std::vector<std::shared_ptr<Variant>>& S)
        {
        m_S = S;
        }

    virtual std::array<Scalar, 2> NPT_thermo_rescale_factor_one(uint64_t timestep)
        {
        return {Scalar(1.0), Scalar(1.0)};
        }

    virtual std::array<Scalar, 2> NPT_thermo_rescale_factor_two(uint64_t timestep)
        {
        return {Scalar(1.0), Scalar(1.0)};
        }

    std::shared_ptr<Variant> getT()
        {
        return m_T;
        }


    /*! \param T New temperature to set
     */
    virtual void setT(std::shared_ptr<Variant> T)
        {
        m_T = T;
        }

    std::shared_ptr<ComputeThermo>  m_thermo_half_step; //!< ComputeThermo operating on the integrated group at t
    std::shared_ptr<ComputeThermo>  m_thermo_full_step; //!< ComputeThermo operating on the integrated group at t

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

    protected:
    std::vector<std::shared_ptr<Variant>>  m_S;    //!< Stress matrix (upper diagonal, components [xx, yy, zz, yz, xz, xy])
    Scalar m_V; //!< Current volume

    Scalar m_ndof;
    couplingMode m_couple;     //!< Coupling of diagonal elements
    unsigned int m_flags;      //!< Coupling flags for barostat
    bool m_nph;                //!< True if integrating without thermostat
    Scalar m_mat_exp_v[6];     //!< Matrix exponential for velocity update (upper triangular)
    Scalar m_mat_exp_r[6];     //!< Matrix exponential for position update (upper triangular)
    Scalar m_mat_exp_r_int[6]; //!< Integrated matrix exp. for velocity update (upper triangular)

    bool m_rescale_all; //!< If true, rescale all particles in the system irrespective of group

    //! Helper function to update the propagator elements
    virtual void updatePropagator(){};

    //! Get the relevant couplings for the active box degrees of freedom
    couplingMode getRelevantCouplings();

    //! Helper function to advance the barostat parameters
    virtual void advanceBarostat(uint64_t timestep){};

    std::shared_ptr<Variant> m_T;

    /*!\param timestep The time step
     * \param broadcast True if we should broadcast the integrator variables via MPI
     */
    virtual void advanceThermostat(uint64_t timestep){};

    };

    } // namespace hoomd

#endif // HOOMD_TWOSTEPNPTBASE_H
