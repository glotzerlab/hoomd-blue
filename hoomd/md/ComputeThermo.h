// Copyright (c) 2009-2024 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

#include "ComputeThermoTypes.h"
#include "hoomd/Compute.h"
#include "hoomd/GlobalArray.h"
#include "hoomd/ParticleGroup.h"

#include <limits>
#include <memory>

/*! \file ComputeThermo.h
    \brief Declares a class for computing thermodynamic quantities
*/

#ifdef __HIPCC__
#error This header cannot be compiled by nvcc
#endif

#include <pybind11/pybind11.h>

#ifndef __COMPUTE_THERMO_H__
#define __COMPUTE_THERMO_H__

namespace hoomd
    {
namespace md
    {
//! Computes thermodynamic properties of a group of particles
/*! ComputeThermo calculates instantaneous thermodynamic properties and provides them in Python.
    All computed values are stored in a GlobalArray so that they can be accessed on the GPU without
   intermediate copies. Use the enum values in thermo_index to index the array and extract the
   properties of interest. Convenience functions are provided for accessing the values on the CPU.
   Certain properties, like ndof and num_particles are always known and there is no need for them to
   be accessible via the GlobalArray.

    Computed quantities available in the GlobalArray:
     - temperature of the group from translational degrees of freedom
     - temperature of the group from rotational degrees of freedom
     - pressure (valid for the all group)
     - translational kinetic energy
     - rotational kinetic energy
     - potential energy

    Values available all the time
     - number of degrees of freedom (ndof)
     - number of particles in the group

    ndof is utilized in calculating the temperature from the kinetic energy. setNDOF() changes it to
   any value the user desires (the default is one!). In standard usage, the python interface queries
   the number of degrees of freedom from the integrators and sets that value for each ComputeThermo
   so that it is always correct.

    \ingroup computes
*/
class PYBIND11_EXPORT ComputeThermo : public Compute
    {
    public:
    //! Constructs the compute
    ComputeThermo(std::shared_ptr<SystemDefinition> sysdef, std::shared_ptr<ParticleGroup> group);

    //! Destructor
    virtual ~ComputeThermo();

    //! Compute the temperature
    virtual void compute(uint64_t timestep);

    //! Returns the overall temperature last computed by compute()
    /*! \returns Instantaneous overall temperature of the system
     */
    Scalar getTemperature()
        {
#ifdef ENABLE_MPI
        if (!m_properties_reduced)
            reduceProperties();
#endif
        ArrayHandle<Scalar> h_properties(m_properties, access_location::host, access_mode::read);
        if (m_group->getTranslationalDOF() + m_group->getRotationalDOF() > 0)
            {
            Scalar prefactor
                = Scalar(2.0) / (m_group->getTranslationalDOF() + m_group->getRotationalDOF());
            return prefactor
                   * (h_properties.data[thermo_index::translational_kinetic_energy]
                      + h_properties.data[thermo_index::rotational_kinetic_energy]);
            }
        else
            {
            return 0.0;
            }
        }

    //! Returns the translational temperature last computed by compute()
    /*! \returns Instantaneous translational temperature of the system
     */
    Scalar getTranslationalTemperature()
        {
#ifdef ENABLE_MPI
        if (!m_properties_reduced)
            reduceProperties();
#endif
        ArrayHandle<Scalar> h_properties(m_properties, access_location::host, access_mode::read);
        if (m_group->getTranslationalDOF() > 0)
            {
            return Scalar(2.0) / m_group->getTranslationalDOF()
                   * h_properties.data[thermo_index::translational_kinetic_energy];
            }
        else
            {
            return 0.0;
            }
        }

    //! Returns the rotational temperature last computed by compute()
    /*! \returns Instantaneous rotational temperature of the system
     */
    Scalar getRotationalTemperature()
        {
#ifdef ENABLE_MPI
        if (!m_properties_reduced)
            reduceProperties();
#endif
        // return 0.0 if the flags are not valid or we have no rotational DOF
        if (m_computed_flags[pdata_flag::rotational_kinetic_energy]
            && m_group->getRotationalDOF() > 0)
            {
            ArrayHandle<Scalar> h_properties(m_properties,
                                             access_location::host,
                                             access_mode::read);
            return Scalar(2.0) / m_group->getRotationalDOF()
                   * h_properties.data[thermo_index::rotational_kinetic_energy];
            }
        else
            {
            return 0.0;
            }
        }

    //! Returns the pressure last computed by compute()
    /*! \returns Instantaneous pressure of the system
     */
    Scalar getPressure()
        {
        // return NaN if the flags are not valid
        if (m_computed_flags[pdata_flag::pressure_tensor])
            {
// return the pressure
#ifdef ENABLE_MPI
            if (!m_properties_reduced)
                reduceProperties();
#endif

            ArrayHandle<Scalar> h_properties(m_properties,
                                             access_location::host,
                                             access_mode::read);
            return h_properties.data[thermo_index::pressure];
            }
        else
            {
            return std::numeric_limits<Scalar>::quiet_NaN();
            }
        }

    //! Returns the translational kinetic energy last computed by compute()
    /*! \returns Instantaneous translational kinetic energy of the system
     */
    Scalar getTranslationalKineticEnergy()
        {
#ifdef ENABLE_MPI
        if (!m_properties_reduced)
            reduceProperties();
#endif

        ArrayHandle<Scalar> h_properties(m_properties, access_location::host, access_mode::read);
        return h_properties.data[thermo_index::translational_kinetic_energy];
        }

    //! Returns the rotational kinetic energy last computed by compute()
    /*! \returns Instantaneous rotational kinetic energy of the system
     */
    Scalar getRotationalKineticEnergy()
        {
#ifdef ENABLE_MPI
        if (!m_properties_reduced)
            reduceProperties();
#endif

        // return 0.0 if the flags are not valid
        if (m_computed_flags[pdata_flag::rotational_kinetic_energy])
            {
            ArrayHandle<Scalar> h_properties(m_properties,
                                             access_location::host,
                                             access_mode::read);
            return h_properties.data[thermo_index::rotational_kinetic_energy];
            }
        else
            {
            return 0.0;
            }
        }

    //! Returns the total kinetic energy last computed by compute()
    /*! \returns Instantaneous total kinetic energy of the system
     */
    Scalar getKineticEnergy()
        {
#ifdef ENABLE_MPI
        if (!m_properties_reduced)
            reduceProperties();
#endif

        // return only translational component if the flags are not valid
        ArrayHandle<Scalar> h_properties(m_properties, access_location::host, access_mode::read);
        if (m_computed_flags[pdata_flag::rotational_kinetic_energy])
            {
            return (h_properties.data[thermo_index::translational_kinetic_energy]
                    + h_properties.data[thermo_index::rotational_kinetic_energy]);
            }
        else
            {
            return h_properties.data[thermo_index::translational_kinetic_energy];
            }
        }

    //! Returns the potential energy last computed by compute()
    /*! \returns Instantaneous potential energy of the system, or NaN if the energy is not valid
     */
    Scalar getPotentialEnergy()
        {
#ifdef ENABLE_MPI
        if (!m_properties_reduced)
            reduceProperties();
#endif

        ArrayHandle<Scalar> h_properties(m_properties, access_location::host, access_mode::read);
        return h_properties.data[thermo_index::potential_energy];
        }

    //! Returns the upper triangular virial tensor last computed by compute()
    /*! \returns Instantaneous virial tensor, or virial tensor containing NaN entries if it is
        not available
    */
    PressureTensor getPressureTensor()
        {
        // return tensor of NaN's if flags are not valid
        PressureTensor p;
        if (m_computed_flags[pdata_flag::pressure_tensor])
            {
#ifdef ENABLE_MPI
            if (!m_properties_reduced)
                reduceProperties();
#endif

            ArrayHandle<Scalar> h_properties(m_properties,
                                             access_location::host,
                                             access_mode::read);

            p.xx = h_properties.data[thermo_index::pressure_xx];
            p.xy = h_properties.data[thermo_index::pressure_xy];
            p.xz = h_properties.data[thermo_index::pressure_xz];
            p.yy = h_properties.data[thermo_index::pressure_yy];
            p.yz = h_properties.data[thermo_index::pressure_yz];
            p.zz = h_properties.data[thermo_index::pressure_zz];
            }
        else
            {
            p.xx = std::numeric_limits<Scalar>::quiet_NaN();
            p.xy = std::numeric_limits<Scalar>::quiet_NaN();
            p.xz = std::numeric_limits<Scalar>::quiet_NaN();
            p.yy = std::numeric_limits<Scalar>::quiet_NaN();
            p.yz = std::numeric_limits<Scalar>::quiet_NaN();
            p.zz = std::numeric_limits<Scalar>::quiet_NaN();
            }
        return p;
        }

    //! Returns the pressure tensor as a python list to be used for logging
    /*! \returns the pressure tensor as a python list
     */
    pybind11::list getPressureTensorPython()
        {
        pybind11::list toReturn;
        PressureTensor p = getPressureTensor();
        toReturn.append(p.xx);
        toReturn.append(p.xy);
        toReturn.append(p.xz);
        toReturn.append(p.yy);
        toReturn.append(p.yz);
        toReturn.append(p.zz);
        return toReturn;
        }

    // <--------------- Degree of Freedom Data

    double getNDOF()
        {
        return m_group->getTranslationalDOF() + m_group->getRotationalDOF();
        }

    double getTranslationalDOF()
        {
        return m_group->getTranslationalDOF();
        }

    double getRotationalDOF()
        {
        return m_group->getRotationalDOF();
        }

    unsigned int getNumParticles()
        {
        return m_group->getNumMembersGlobal();
        }

    //! Get the gpu array of properties
    const GlobalArray<Scalar>& getProperties()
        {
#ifdef ENABLE_MPI
        if (!m_properties_reduced)
            reduceProperties();
#endif

        return m_properties;
        }

    /// Get the box volume (or area in 2D)
    const Scalar getVolume()
        {
        bool two_d = m_sysdef->getNDimensions() == 2;
        return m_sysdef->getParticleData()->getGlobalBox().getVolume(two_d);
        }

    protected:
    std::shared_ptr<ParticleGroup> m_group; //!< Group to compute properties for
    GlobalArray<Scalar> m_properties;       //!< Stores the computed properties

    /// Store the particle data flags used during the last computation
    PDataFlags m_computed_flags;

    //! Does the actual computation
    virtual void computeProperties();

#ifdef ENABLE_MPI
    bool m_properties_reduced; //!< True if properties have been reduced across MPI

    //! Reduce properties over MPI
    virtual void reduceProperties();
#endif
    };

    } // end namespace md
    } // end namespace hoomd

#endif
