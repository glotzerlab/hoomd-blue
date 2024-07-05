// Copyright (c) 2009-2024 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

#include "ComputeThermoHMATypes.h"
#include "hoomd/Compute.h"
#include "hoomd/GPUArray.h"
#include "hoomd/ParticleGroup.h"

#include <limits>
#include <memory>

/*! \file ComputeThermoHMA.h
    \brief Declares a class for computing thermodynamic quantities
*/

#ifdef NVCC
#error This header cannot be compiled by nvcc
#endif

#include <pybind11/pybind11.h>

#ifndef __COMPUTE_THERMO_HMA_H__
#define __COMPUTE_THERMO_HMA_H__

namespace hoomd
    {
namespace md
    {
//! Computes thermodynamic properties of a group of particles
/*! ComputeThermoHMA calculates instantaneous thermodynamic properties and provides them for Python.
    All computed values are stored in a GPUArray so that they can be accessed on the GPU without
   intermediate copies. Use the enum values in thermoHMA_index to index the array and extract the
   properties of interest. Convenience functions are provided for accessing the values on the CPU.

    Computed quantities available in the GPUArray:
     - pressure (valid for the all group)
     - potential energy

    All quantities are made available in Python as properties.

    \ingroup computes
*/
class PYBIND11_EXPORT ComputeThermoHMA : public Compute
    {
    public:
    //! Constructs the compute
    ComputeThermoHMA(std::shared_ptr<SystemDefinition> sysdef,
                     std::shared_ptr<ParticleGroup> group,
                     const double temperature,
                     const double harmonicPressure);

    //! Destructor
    virtual ~ComputeThermoHMA();

    //! Compute the temperature
    virtual void compute(uint64_t timestep);

    //! Returns the potential energy last computed by compute()
    /*! \returns Instantaneous potential energy of the system, or NaN if the energy is not valid
     */
    Scalar getPotentialEnergyHMA()
        {
#ifdef ENABLE_MPI
        if (!m_properties_reduced)
            reduceProperties();
#endif

        // return NaN if the flags are not valid
        ArrayHandle<Scalar> h_properties(m_properties, access_location::host, access_mode::read);
        return h_properties.data[thermoHMA_index::potential_energyHMA];
        }

    //! Returns the pressure last computed by compute()
    /*! \returns Instantaneous pressure of the system
     */
    Scalar getPressureHMA()
        {
        // return NaN if the flags are not valid
        PDataFlags flags = m_pdata->getFlags();
        if (flags[pdata_flag::pressure_tensor])
            {
// return the pressure
#ifdef ENABLE_MPI
            if (!m_properties_reduced)
                reduceProperties();
#endif

            ArrayHandle<Scalar> h_properties(m_properties,
                                             access_location::host,
                                             access_mode::read);
            return h_properties.data[thermoHMA_index::pressureHMA];
            }
        else
            {
            return std::numeric_limits<Scalar>::quiet_NaN();
            }
        }

    //! Get the gpu array of properties
    const GPUArray<Scalar>& getProperties()
        {
#ifdef ENABLE_MPI
        if (!m_properties_reduced)
            reduceProperties();
#endif

        return m_properties;
        }

    /*! Set the temperature
        \param temperature Temperature to set
    */
    void setTemperature(Scalar temperature)
        {
        m_temperature = temperature;
        }

    //! Get the temperature
    Scalar getTemperature()
        {
        return m_temperature;
        }

    /*! Set the harmonic pressure
        \param harmonicPressure Harmonic pressure to set
    */
    void setHarmonicPressure(Scalar harmonicPressure)
        {
        m_harmonicPressure = harmonicPressure;
        }

    //! Get the harmonic pressure
    Scalar getHarmonicPressure()
        {
        return m_harmonicPressure;
        }

    //! Method to be called when particles are added/removed/sorted
    void slotParticleSort();

    protected:
    std::shared_ptr<ParticleGroup> m_group; //!< Group to compute properties for
    GPUArray<Scalar> m_properties;          //!< Stores the computed properties

    //! Does the actual computation
    virtual void computeProperties();

#ifdef ENABLE_MPI
    bool m_properties_reduced; //!< True if properties have been reduced across MPI

    //! Reduce properties over MPI
    virtual void reduceProperties();
#endif

    Scalar m_temperature, m_harmonicPressure;
    GlobalArray<Scalar3> m_lattice_site;
    };

    } // end namespace md
    } // end namespace hoomd

#endif
