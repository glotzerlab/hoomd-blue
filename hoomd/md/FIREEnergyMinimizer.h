// Copyright (c) 2009-2024 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

#include "IntegratorTwoStep.h"

#include <memory>

#ifndef __FIRE_ENERGY_MINIMIZER_H__
#define __FIRE_ENERGY_MINIMIZER_H__

/*! \file FIREEnergyMinimizer.h
    \brief Declares the FIRE energy minimizer class
*/

#ifdef __HIPCC__
#error This header cannot be compiled by nvcc
#endif

#include <pybind11/pybind11.h>

namespace hoomd
    {
namespace md
    {
//! Finds the nearest basin in the potential energy landscape
/*! \b Overview

    \ingroup updaters
*/
class PYBIND11_EXPORT FIREEnergyMinimizer : public IntegratorTwoStep
    {
    public:
    //! Constructs the minimizer and associates it with the system
    FIREEnergyMinimizer(std::shared_ptr<SystemDefinition>, Scalar);
    virtual ~FIREEnergyMinimizer();

    //! Reset the minimization
    virtual void reset();

    //! Perform one minimization iteration
    virtual void update(uint64_t timestep);

    //! Return whether or not the minimization has converged
    bool hasConverged() const
        {
        return m_converged;
        }

    //! Return the potential energy after the last iteration
    Scalar getEnergy() const
        {
        if (m_was_reset)
            {
            m_exec_conf->msg->warning()
                << "FIRE has just been initialized. Return energy==0." << std::endl;
            return Scalar(0.0);
            }

        return m_energy_total;
        }

    //! Set the minimum number of steps for which the search direction must be bad before finding a
    //! new direction
    /*! \param nmin is the new nmin to set
     */
    void setNmin(unsigned int nmin)
        {
        m_nmin = nmin;
        }

    //! Get the minimum number of steps for which the search direction must be
    //! bad before finding a new direction
    unsigned int getNmin()
        {
        return m_nmin;
        }

    //! Set the fractional increase in the timestep upon a valid search direction
    void setFinc(Scalar finc);

    //! get the fractional increase in the timestep upon a valid search direction
    Scalar getFinc()
        {
        return m_finc;
        }

    //! Set the fractional decrease in the timestep upon system energy increasing
    void setFdec(Scalar fdec);

    //! Get the fractional decrease in the timestep upon system energy increasing
    Scalar getFdec()
        {
        return m_fdec;
        }

    //! Set the relative strength of the coupling between the "f dot v" vs the "v" term
    void setAlphaStart(Scalar alpha0);

    //! Get the relative strength of the coupling between the "f dot v" vs the "v" term
    Scalar getAlphaStart()
        {
        return m_alpha_start;
        }

    //! Set the fractional decrease in alpha upon finding a valid search direction
    void setFalpha(Scalar falpha);

    //! Get the fractional decrease in alpha upon finding a valid search direction
    Scalar getFalpha()
        {
        return m_falpha;
        }

    //! Set the stopping criterion based on the total force on all particles in the system
    /*! \param ftol is the new force tolerance to set
     */
    void setFtol(Scalar ftol)
        {
        m_ftol = ftol;
        }

    //! get the stopping criterion based on the total force on all particles in the system
    Scalar getFtol()
        {
        return m_ftol;
        }

    //! Set the stopping criterion based on the total torque on all particles in the system
    /*! \param wtol is the new torque tolerance to set
     */
    void setWtol(Scalar wtol)
        {
        m_wtol = wtol;
        }

    //! Get the stopping criterion based on the total torque on all particles in the system
    Scalar getWtol()
        {
        return m_wtol;
        }

    //! Set the stopping criterion based on the change in energy between successive iterations
    /*! \param etol is the new energy tolerance to set
     */
    void setEtol(Scalar etol)
        {
        m_etol = etol;
        }

    //! Get the stopping criterion based on the change in energy between successive iterations
    Scalar getEtol()
        {
        return m_etol;
        }

    //! Set the a minimum number of steps before the other stopping criteria will be evaluated
    /*! \param steps is the minimum number of steps (attempts) that will be made
     */
    void setMinSteps(unsigned int steps)
        {
        m_run_minsteps = steps;
        }

    //! Get the minimum number of steps before the other stopping criteria will be evaluated
    unsigned int getMinSteps()
        {
        return m_run_minsteps;
        }

    protected:
    //! Function to create the underlying integrator
    unsigned int m_nmin; //!< minimum number of consecutive successful search directions before
                         //!< modifying alpha
    unsigned int
        m_n_since_negative;       //!< counts the number of consecutive successful search directions
    unsigned int m_n_since_start; //!< counts the number of consecutive search attempts
    Scalar m_finc;                //!< fractional increase in timestep upon successful search
    Scalar m_fdec;                //!< fractional decrease in timestep upon unsuccessful search
    Scalar m_alpha;               //!< relative coupling strength between alpha
    Scalar m_alpha_start;         //!< starting value of alpha
    Scalar m_falpha;              //!< fraction to rescale alpha on successful search direction
    Scalar m_ftol;                //!< stopping tolerance based on total force
    Scalar m_wtol;                //!< stopping tolerance based on total torque
    Scalar m_etol;                //!< stopping tolerance based on the chance in energy
    Scalar m_energy_total;        //!< Total energy of all integrator groups
    Scalar m_old_energy;          //!< energy from the previous iteration
    bool m_converged;             //!< whether the minimization has converged
    Scalar m_deltaT_max;          //!< maximum timesteps after rescaling (set by user)
    Scalar m_deltaT_set;          //!< the initial timestep
    unsigned int m_run_minsteps;  //!< A minimum number of search attempts the search will use
    bool m_was_reset;             //!< whether or not the minimizer was reset

    private:
    };

    } // end namespace md
    } // end namespace hoomd

#endif // #ifndef __FIRE_ENERGY_MINIMIZER_H__
