// Copyright (c) 2009-2019 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.


// Maintainer: jglaser

#include "IntegrationMethodTwoStep.h"
#include "hoomd/Variant.h"
#include "hoomd/ComputeThermo.h"

#ifndef __TWO_STEP_NPT_MTK_H__
#define __TWO_STEP_NPT_MTK_H__

/*! \file TwoStepNPTMTK.h
    \brief Declares the TwoStepNPTMTK class
*/

#ifdef NVCC
#error This header cannot be compiled by nvcc
#endif

#include <hoomd/extern/pybind/include/pybind11/pybind11.h>
#include <hoomd/extern/pybind/include/pybind11/stl.h>

//! Integrates part of the system forward in two steps in the NPT ensemble
/*! Implements the Martyna Tobias Klein (MTK) equations for rigorous integration in the NPT ensemble.
    The update equations are derived from a strictly measure-preserving and
    time-reversal symmetric integration scheme, closely following the one proposed by Tuckerman et al.

    Supports anisotropic (orthorhombic or tetragonal) integration modes, by implementing a special
    version of the the fully flexible cell update equations proposed in Yu et al.

    Triclinic integration for an upper triangular cell parameter matrix is supported with
    fully time-reversible and measure-preserving update equations (Glaser et al. 2013 to be published)

    \cite Martyna1994
    \cite Tuckerman2006
    \cite Yu2010
    \ingroup updaters
*/
class PYBIND11_EXPORT TwoStepNPTMTK : public IntegrationMethodTwoStep
    {
    public:
        //! Specify possible couplings between the diagonal elements of the pressure tensor
        enum couplingMode
            {
            couple_none = 0,
            couple_xy,
            couple_xz,
            couple_yz,
            couple_xyz};

        /*! Flags to indicate which degrees of freedom of the simulation box should be put under
            barostat control
         */
        enum baroFlags
            {
            baro_x = 1,
            baro_y = 2,
            baro_z = 4,
            baro_xy = 8,
            baro_xz = 16,
            baro_yz = 32
            };

        //! Constructs the integration method and associates it with the system
        TwoStepNPTMTK(std::shared_ptr<SystemDefinition> sysdef,
                   std::shared_ptr<ParticleGroup> group,
                   std::shared_ptr<ComputeThermo> thermo_group,
                   std::shared_ptr<ComputeThermo> thermo_group_t,
                   Scalar tau,
                   Scalar tauP,
                   std::shared_ptr<Variant> T,
                   pybind11::list S,
                   couplingMode couple,
                   unsigned int flags,
                   const bool nph=false);

        virtual ~TwoStepNPTMTK();

        //! Update the temperature
        /*! \param T New temperature to set
        */
        virtual void setT(std::shared_ptr<Variant> T)
            {
            m_T = T;
            }

    //! Update the stress components
    /*! \param S list of stress components: [xx, yy, zz, yz, xz, xy]
     */
    virtual void setS(pybind11::list S)
            {
            std::vector<std::shared_ptr<Variant> > swapS;
            swapS.resize(0);
            for (int i = 0; i< 6; ++i)
                   {
                swapS.push_back(pybind11::cast<std::shared_ptr<Variant>>(S[i]));
                }
            m_S.swap(swapS);
            }

        //! Update the tau value
        /*! \param tau New time constant to set
        */
        virtual void setTau(Scalar tau)
            {
            m_tau = tau;
            }

        //! Update the nuP value
        /*! \param tauP New pressure constant to set
        */
        virtual void setTauP(Scalar tauP)
            {
            m_tauP = tauP;
            }

        //! Set the scale all particles option
        /*! \param rescale_all If true, rescale all particles
        */
        void setRescaleAll(bool rescale_all)
            {
            m_rescale_all = rescale_all;
            }

        //! Set an optional damping factor for the box degrees of freedom
        void setGamma(Scalar gamma)
            {
            m_gamma = gamma;
            }

        //! Performs the first step of the integration
        virtual void integrateStepOne(unsigned int timestep);

        //! Performs the second step of the integration
        virtual void integrateStepTwo(unsigned int timestep);

        //! Get needed pdata flags
        /*! TwoStepNPTMTK needs the pressure, so the isotropic_virial or pressure_tensor flag is set,
            depending on the integration mode
        */
        virtual PDataFlags getRequestedPDataFlags()
            {
            PDataFlags flags;
            flags[pdata_flag::pressure_tensor] = 1;
            if (m_aniso)
                {
                flags[pdata_flag::rotational_kinetic_energy] = 1;
//                flags[pdata_flag::rotational_virial] = 1;
                }
            flags[pdata_flag::external_field_virial]=1;
            return flags;
            }

        //! Returns a list of log quantities this compute calculates
        virtual std::vector< std::string > getProvidedLogQuantities();

        //! Returns logged values
        virtual Scalar getLogValue(const std::string& quantity, unsigned int timestep, bool &my_quantity_flag);

        //! Initialize integrator variables
        virtual void initializeIntegratorVariables()
            {
            IntegratorVariables v = getIntegratorVariables();
            v.type = "npt_mtk";
            v.variable.clear();
            v.variable.resize(10,Scalar(0.0));
            setIntegratorVariables(v);
            }

        //! Randomize the barostat variables
        virtual void randomizeVelocities(unsigned int timestep);

    protected:
        std::shared_ptr<ComputeThermo> m_thermo_group;   //!< ComputeThermo operating on the integrated group at t+dt/2
        std::shared_ptr<ComputeThermo> m_thermo_group_t; //!< ComputeThermo operating on the integrated group at t
        unsigned int m_ndof;            //!< Number of degrees of freedom from ComputeThermo

        Scalar m_tau;                   //!< tau value for Nose-Hoover
        Scalar m_tauP;                  //!< tauP value for the barostat
        std::shared_ptr<Variant> m_T; //!< Temperature set point
        std::vector<std::shared_ptr<Variant>> m_S;  //!< Stress matrix (upper diagonal, components [xx, yy, zz, yz, xz, xy])
        Scalar m_V;                     //!< Current volume

        couplingMode m_couple;          //!< Coupling of diagonal elements
        unsigned int m_flags;             //!< Coupling flags for barostat
        bool m_nph;                     //!< True if integrating without thermostat
        Scalar m_mat_exp_v[6];          //!< Matrix exponential for velocity update (upper triangular)
        Scalar m_mat_exp_r[6];          //!< Matrix exponential for position update (upper triangular)
        Scalar m_mat_exp_r_int[6];      //!< Integrated matrix exp. for velocity update (upper triangular)

        bool m_rescale_all;             //!< If true, rescale all particles in the system irrespective of group

        Scalar m_gamma;                 //!< Optional damping factor for box degrees of freedom

        std::vector<std::string> m_log_names; //!< Name of the barostat and thermostat quantities that we log

        //! Helper function to advance the barostat parameters
        void advanceBarostat(unsigned int timestep);

        //! advance the thermostat
        /*!\param timestep The time step
         * \param broadcast True if we should broadcast the integrator variables via MPI
         */
        void advanceThermostat(unsigned int timestep);

        //! Helper function to update the propagator elements
        void updatePropagator(Scalar nuxx, Scalar nuxy, Scalar nuxz, Scalar nuyy, Scalar nuyz, Scalar nuzz);

        //! Get the relevant couplings for the active box degrees of freedom
        couplingMode getRelevantCouplings();
        };

//! Exports the TwoStepNPTMTK class to python
void export_TwoStepNPTMTK(pybind11::module& m);

#endif // #ifndef __TWO_STEP_NPT_MTK_H__
