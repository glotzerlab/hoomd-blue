// Copyright (c) 2009-2018 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.


// Maintainer: joaander

#include "Compute.h"
#include "GPUArray.h"
#include "ComputeThermoTypes.h"
#include "ParticleGroup.h"

#include <memory>
#include <limits>

/*! \file ComputeThermoHMA.h
    \brief Declares a class for computing thermodynamic quantities
*/

#ifdef NVCC
#error This header cannot be compiled by nvcc
#endif

#include <hoomd/extern/pybind/include/pybind11/pybind11.h>

#ifndef __COMPUTE_THERMO_HMA_H__
#define __COMPUTE_THERMO_HMA_H__

//! Computes thermodynamic properties of a group of particles
/*! ComputeThermoHMA calculates instantaneous thermodynamic properties and provides them for the logger.
    All computed values are stored in a GPUArray so that they can be accessed on the GPU without intermediate copies.
    Use the enum values in thermoHMA_index to index the array and extract the properties of interest. Convenience functions
    are provided for accessing the values on the CPU. Certain properties, loke ndof and num_particles are always known
    and there is no need for them to be accessible via the GPUArray.

    Computed quantities available in the GPUArray:
     - temperature of the group from translational degrees of freedom
     - temperature of the group from rotational degrees of freedom
     - pressure (valid for the all group)
     - translational kinetic energy
     - rotational kinetic energy
     - potential energy

    Values available all the time
     - number of degrees of freedom (ndof)
     - number of particles in the group

    ndof is utilized in calculating the temperature from the kinetic energy. setNDOF() changes it to any value
    the user desires (the default is one!). In standard usage, the python interface queries the number of degrees
    of freedom from the integrators and sets that value for each ComputeThermoHMA so that it is always correct.

    All quantities are made available for the logger. ComputerThermo can be given a suffix which it will append
    to each quantity provided to the logger. Typical usage is to provide _groupname as the suffix so that properties
    of different groups can be logged seperately (e.g. temperature_group1 and temperature_group2).

    \ingroup computes
*/
class PYBIND11_EXPORT ComputeThermoHMA : public Compute
    {
    public:
        //! Constructs the compute
        ComputeThermoHMA(std::shared_ptr<SystemDefinition> sysdef,
                      std::shared_ptr<ParticleGroup> group, double temperature,
                      const std::string& suffix = std::string(""));

        //! Destructor
        virtual ~ComputeThermoHMA();

        //! Compute the temperature
        virtual void compute(unsigned int timestep);

        void setNDOF(unsigned int ndof) {}
        void setRotationalNDOF(unsigned int ndof) {}

        //! Returns the potential energy last computed by compute()
        /*! \returns Instantaneous potential energy of the system, or NaN if the energy is not valid
        */
        Scalar getPotentialEnergyHMA()
            {
            #ifdef ENABLE_MPI
            if (!m_properties_reduced) reduceProperties();
            #endif

            // return NaN if the flags are not valid
            PDataFlags flags = m_pdata->getFlags();
            if (flags[pdata_flag::potential_energy])
                {
                ArrayHandle<Scalar> h_properties(m_properties, access_location::host, access_mode::read);
                return h_properties.data[thermoHMA_index::potential_energyHMA];
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
            if (!m_properties_reduced) reduceProperties();
            #endif

            return m_properties;
            }

        //! Returns a list of log quantities this compute calculates
        virtual std::vector< std::string > getProvidedLogQuantities();

        //! Calculates the requested log value and returns it
        virtual Scalar getLogValue(const std::string& quantity, unsigned int timestep);

        //! Control the enable_logging flag
        /*! Set this flag to false to prevent this compute from providing logged quantities.
            This is useful for internal computes that should not appear in the logs.

            \param enable Flag to set
        */
        void setLoggingEnabled(bool enable)
            {
            m_logging_enabled = enable;
            }

        //! Method to be called when particles are added/removed/sorted
        void slotParticleSort();

    protected:
        std::shared_ptr<ParticleGroup> m_group;     //!< Group to compute properties for
        GPUArray<Scalar> m_properties;  //!< Stores the computed properties
        std::vector<std::string> m_logname_list;  //!< Cache all generated logged quantities names
        bool m_logging_enabled;         //!< Set to false to disable communication with the logger

        //! Does the actual computation
        virtual void computeProperties();

        #ifdef ENABLE_MPI
        bool m_properties_reduced;      //!< True if properties have been reduced across MPI

        //! Reduce properties over MPI
        virtual void reduceProperties();
        #endif

        double T;
        std::vector<Scalar> m_lattice_x;
        std::vector<Scalar> m_lattice_y;
        std::vector<Scalar> m_lattice_z;
    };

//! Exports the ComputeThermoHMA class to python
#ifndef NVCC
void export_ComputeThermoHMA(pybind11::module& m);
#endif

#endif
