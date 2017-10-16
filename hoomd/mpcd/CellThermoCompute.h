// Copyright (c) 2009-2019 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.

// Maintainer: mphoward

/*!
 * \file mpcd/CellThermoCompute.h
 * \brief Declaration of mpcd::CellThermoCompute
 */

#ifndef MPCD_CELL_THERMO_COMPUTE_H_
#define MPCD_CELL_THERMO_COMPUTE_H_

#ifdef NVCC
#error This header cannot be compiled by nvcc
#endif

#include "CellThermoTypes.h"
#include "CellList.h"
#include "SystemData.h"
#ifdef ENABLE_MPI
#include "CellCommunicator.h"
#endif // ENABLE_MPI

#include "hoomd/Compute.h"
#include "hoomd/extern/nano-signal-slot/nano_signal_slot.hpp"
#include "hoomd/extern/pybind/include/pybind11/pybind11.h"

namespace mpcd
{
//! Computes the cell (thermodynamic) properties
class PYBIND11_EXPORT CellThermoCompute : public Compute
    {
    public:
        //! Constructor
        CellThermoCompute(std::shared_ptr<mpcd::SystemData> sysdata,
                          const std::string& suffix = std::string(""));

        //! Destructor
        virtual ~CellThermoCompute();

        //! Compute the cell thermodynamic properties
        void compute(unsigned int timestep);

        //! Get the cell indexer for the attached cell list
        const Index3D& getCellIndexer() const
            {
            return m_cl->getCellIndexer();
            }

        //! Get the cell velocities from the last call to compute
        const GPUArray<double4>& getCellVelocities() const
            {
            return m_cell_vel;
            }

        //! Get the cell energies from the last call to compute
        const GPUArray<double3>& getCellEnergies() const
            {
            return m_cell_energy;
            }

        //! Get the net momentum of the particles from the last call to compute
        Scalar3 getNetMomentum()
            {
            if (m_needs_net_reduce) computeNetProperties();

            ArrayHandle<double> h_net_properties(m_net_properties, access_location::host, access_mode::read);
            const Scalar3 net_momentum = make_scalar3(h_net_properties.data[mpcd::detail::thermo_index::momentum_x],
                                                      h_net_properties.data[mpcd::detail::thermo_index::momentum_y],
                                                      h_net_properties.data[mpcd::detail::thermo_index::momentum_z]);
            return net_momentum;
            }

        //! Get the net energy of the particles from the last call to compute
        Scalar getNetEnergy()
            {
            if (!m_flags[mpcd::detail::thermo_options::energy])
                {
                m_exec_conf->msg->error() << "Energy requested from CellThermoCompute, but was not computed." << std::endl;
                throw std::runtime_error("Net cell energy not available");
                }

            if (m_needs_net_reduce) computeNetProperties();

            ArrayHandle<double> h_net_properties(m_net_properties, access_location::host, access_mode::read);
            return h_net_properties.data[mpcd::detail::thermo_index::energy];
            }

        //! Get the average cell temperature from the last call to compute
        Scalar getTemperature()
            {
            if (!m_flags[mpcd::detail::thermo_options::energy])
                {
                m_exec_conf->msg->error() << "Temperature requested from CellThermoCompute, but was not computed." << std::endl;
                throw std::runtime_error("Net cell temperature not available");
                }
            if (m_needs_net_reduce) computeNetProperties();

            ArrayHandle<double> h_net_properties(m_net_properties, access_location::host, access_mode::read);
            return h_net_properties.data[mpcd::detail::thermo_index::temperature];
            }

        //! Set autotuner parameters
        /*!
         * \param enable Enable/disable autotuning
         * \param period period (approximate) in time steps when returning occurs
         */
        virtual void setAutotunerParams(bool enable, unsigned int period)
            {
            #ifdef ENABLE_MPI
            if (m_vel_comm)
                m_vel_comm->setAutotunerParams(enable, period);
            if (m_energy_comm)
                m_energy_comm->setAutotunerParams(enable, period);
            #endif // ENABLE_MPI
            }

        //! Get the signal for requested thermo flags
        /*!
         * \returns A signal that subscribers can attach a callback to in order
         *          to request certain data.
         *
         * For performance reasons, the CellThermoCompute should be able to
         * supply many related cell-level quantities from a single kernel launch.
         * However, sometimes these quantities are not needed, and it is better
         * to skip calculating them. Subscribing classes can optionally request
         * some of these quantities via a callback return mpcd::detail::ThermoFlags
         * with the appropriate bits set.
         */
        Nano::Signal<mpcd::detail::ThermoFlags ()>& getFlagsSignal()
            {
            return m_flag_signal;
            }

        //! Get the signal to attach a callback function
        Nano::Signal<void (unsigned int)>& getCallbackSignal()
            {
            return m_callbacks;
            }

    protected:
        //! Compute the cell properties
        void computeCellProperties(unsigned int timestep);

        #ifdef ENABLE_MPI
        //! Begin the calculation of outer cell properties
        virtual void beginOuterCellProperties();

        //! Finish the calculation of outer cell properties
        virtual void finishOuterCellProperties();
        #endif // ENABLE_MPI

        //! Calculate the inner cell properties
        virtual void calcInnerCellProperties();

        //! Compute the net properties from the cell properties
        virtual void computeNetProperties();

        std::shared_ptr<mpcd::ParticleData> m_mpcd_pdata;       //!< MPCD particle data
        std::shared_ptr<mpcd::CellList> m_cl;                   //!< MPCD cell list
        #ifdef ENABLE_MPI
        bool m_use_mpi;                                         //!< Flag if communication is required
        std::shared_ptr<CellCommunicator> m_vel_comm;           //!< Cell velocity communicator
        std::shared_ptr<CellCommunicator> m_energy_comm;        //!< Cell energy communicator
        #endif // ENABLE_MPI

        bool m_needs_net_reduce;            //!< Flag if a net reduction is necessary
        GPUArray<double> m_net_properties;  //!< Scalar properties of the system

        GPUVector<double4> m_cell_vel;      //!< Average velocity of a cell + cell mass
        GPUVector<double3> m_cell_energy;   //!< Kinetic energy, unscaled temperature, dof in each cell
        unsigned int m_ncells_alloc;        //!< Number of cells allocated for

        Nano::Signal<mpcd::detail::ThermoFlags ()> m_flag_signal; //!< Signal for requested flags
        mpcd::detail::ThermoFlags m_flags;  //!< Requested thermo flags
        //! Updates the requested optional flags
        void updateFlags();

        Nano::Signal<void (unsigned int)> m_callbacks;  //!< Signal for callback functions

    private:
        //! Allocate memory per cell
        void reallocate(unsigned int ncells);

        //! Slot for the number of virtual particles changing
        /*!
         * All thermo properties should be recomputed if the number of virtual particles changes.
         * This is because we cannot distinguish times when the thermo is used for logging vs.
         * in a collision method. The cases where the thermo is first used without virtual particles
         * followed by with virtual particles are small, and so it is easiest to just recompute.
         */
        void slotNumVirtual()
            {
            m_force_compute = true;
            }
    };

namespace detail
{
//! Export the CellThermoCompute class to python
void export_CellThermoCompute(pybind11::module& m);
} // end namespace detail

} // end namespace mpcd
#endif // #define MPCD_CELL_THERMO_COMPUTE_H_
