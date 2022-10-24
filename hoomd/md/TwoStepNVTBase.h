//
// Created by girard01 on 10/21/22.
//

#ifndef HOOMD_TWOSTEPNVTBASE_H
#define HOOMD_TWOSTEPNVTBASE_H

#include "ComputeThermo.h"
#include "IntegrationMethodTwoStep.h"
#include "hoomd/Variant.h"
#include <pybind11/pybind11.h>
namespace hoomd
    {
namespace md
    {

class PYBIND11_EXPORT TwoStepNVTBase : public IntegrationMethodTwoStep
    {
    public:
    TwoStepNVTBase(std::shared_ptr<SystemDefinition> sysdef,
                   std::shared_ptr<ParticleGroup> group,
                   std::shared_ptr<ComputeThermo> thermo,
                   std::shared_ptr<Variant> T) : IntegrationMethodTwoStep(sysdef, group), m_thermo(thermo), m_T(T) {}
    /// Get the current temperature variant
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

    //! Performs the first step of the integration
    virtual void integrateStepOne(uint64_t timestep);
    //! Performs the second step of the integration
    virtual void integrateStepTwo(uint64_t timestep);

    virtual void thermalizeThermostatDOF(uint64_t timestep){};

    virtual std::array<Scalar, 2> NVT_rescale_factor_one(uint64_t timestep)
        {
        return {Scalar(1.0), Scalar(1.0)};
        }

    virtual std::array<Scalar, 2> NVT_rescale_factor_two(uint64_t timestep)
        {
        return {Scalar(1.0), Scalar(1.0)};
        }

    protected:

    std::shared_ptr<ComputeThermo> m_thermo; //!< compute for thermodynamic quantities

    std::shared_ptr<Variant> m_T;

    /*!\param timestep The time step
     * \param broadcast True if we should broadcast the integrator variables via MPI
     */
    virtual void advanceThermostat(uint64_t timestep, bool broadcast = true){};
    };

    } // namespace md
    } // namespace hoomd

#endif // HOOMD_TWOSTEPNVTBASE_H
