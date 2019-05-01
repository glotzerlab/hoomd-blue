// Copyright (c) 2009-2019 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.


// Maintainer: askeys

#include "IntegratorTwoStep.h"

#include <memory>

#ifndef __FIRE_ENERGY_MINIMIZER_H__
#define __FIRE_ENERGY_MINIMIZER_H__

/*! \file FIREEnergyMinimizer.h
    \brief Declares the FIRE energy minimizer class
*/

#ifdef NVCC
#error This header cannot be compiled by nvcc
#endif

#include <hoomd/extern/pybind/include/pybind11/pybind11.h>

//! Finds the nearest basin in the potential energy landscape
/*! \b Overview

    \ingroup updaters
*/
class PYBIND11_EXPORT FIREEnergyMinimizer : public IntegratorTwoStep
    {
    public:
        //! Constructs the minimizer and associates it with the system
        FIREEnergyMinimizer(std::shared_ptr<SystemDefinition>,  Scalar);
        virtual ~FIREEnergyMinimizer();

        //! Reset the minimization
        virtual void reset();

        //! Perform one minimization iteration
        virtual void update(unsigned int);

        //! Return whether or not the minimization has converged
        bool hasConverged() const {return m_converged;}

        //! Return the potential energy after the last iteration
        Scalar getEnergy() const
            {
            if (m_was_reset)
                {
                m_exec_conf->msg->warning() << "FIRE has just been initialized. Return energy==0."
                    << std::endl;
                return Scalar(0.0);
                }

            return m_energy_total;
            }

        //! Set the minimum number of steps for which the search direction must be bad before finding a new direction
        /*! \param nmin is the new nmin to set
        */
        void setNmin(unsigned int nmin) {m_nmin = nmin;}

        //! Set the fractional increase in the timestep upon a valid search direction
        void setFinc(Scalar finc);

        //! Set the fractional increase in the timestep upon a valid search direction
        void setFdec(Scalar fdec);

        //! Set the relative strength of the coupling between the "f dot v" vs the "v" term
        void setAlphaStart(Scalar alpha0);

        //! Set the fractional decrease in alpha upon finding a valid search direction
        void setFalpha(Scalar falpha);

        //! Set the stopping criterion based on the total force on all particles in the system
        /*! \param ftol is the new force tolerance to set
        */
        void setFtol(Scalar ftol) {m_ftol = ftol;}

        //! Set the stopping criterion based on the total torque on all particles in the system
        /*! \param wtol is the new torque tolerance to set
        */
        void setWtol(Scalar wtol) {m_wtol = wtol;}

        //! Set the stopping criterion based on the change in energy between successive iterations
        /*! \param etol is the new energy tolerance to set
        */
        void setEtol(Scalar etol) {m_etol = etol;}

        //! Set the a minimum number of steps before the other stopping criteria will be evaluated
        /*! \param steps is the minimum number of steps (attempts) that will be made
        */
        void setMinSteps(unsigned int steps) {m_run_minsteps = steps;}

        //! Get needed pdata flags
        /*! FIREEnergyMinimizer needs the potential energy, so its flag is set
        */
        virtual PDataFlags getRequestedPDataFlags()
            {
            PDataFlags flags = IntegratorTwoStep::getRequestedPDataFlags();
            flags[pdata_flag::potential_energy] = 1;
            return flags;
            }

    protected:
        //! Function to create the underlying integrator
        unsigned int m_nmin;                //!< minimum number of consecutive successful search directions before modifying alpha
        unsigned int m_n_since_negative;    //!< counts the number of consecutive successful search directions
        unsigned int m_n_since_start;       //!< counts the number of consecutive search attempts
        Scalar m_finc;                      //!< fractional increase in timestep upon successful search
        Scalar m_fdec;                      //!< fractional decrease in timestep upon unsuccessful search
        Scalar m_alpha;                     //!< relative coupling strength between alpha
        Scalar m_alpha_start;               //!< starting value of alpha
        Scalar m_falpha;                    //!< fraction to rescale alpha on successful search direction
        Scalar m_ftol;                      //!< stopping tolerance based on total force
        Scalar m_wtol;                      //!< stopping tolerance based on total torque
        Scalar m_etol;                      //!< stopping tolerance based on the chance in energy
        Scalar m_energy_total;              //!< Total energy of all integrator groups
        Scalar m_old_energy;                //!< energy from the previous iteration
        bool m_converged;                   //!< whether the minimization has converged
        Scalar m_deltaT_max;                //!< maximum timesteps after rescaling (set by user)
        Scalar m_deltaT_set;                //!< the initial timestep
        unsigned int m_run_minsteps;        //!< A minimum number of search attempts the search will use
        bool m_was_reset;                   //!< whether or not the minimizer was reset

    private:

    };

//! Exports the FIREEnergyMinimizer class to python
void export_FIREEnergyMinimizer(pybind11::module& m);

#endif // #ifndef __FIRE_ENERGY_MINIMIZER_H__
