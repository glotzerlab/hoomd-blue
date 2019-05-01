// Copyright (c) 2009-2019 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.

// inclusion guard
#ifndef _UPDATER_HPMC_BOX_MC_
#define _UPDATER_HPMC_BOX_MC_

/*! \file UpdaterBoxMC.h
    \brief Declaration of UpdaterBoxMC
*/

#include <hoomd/Updater.h>
#include <hoomd/Variant.h>
#include "hoomd/RandomNumbers.h"
#include <cmath>

#include "IntegratorHPMC.h"

#ifndef NVCC
#include <hoomd/extern/pybind/include/pybind11/pybind11.h>
#endif

namespace hpmc
{

//! Update box for HPMC simulation in the NPT ensemble, etc.
/*! The pressure parameter is beta*P. For a unitless reduced pressure, the user must adopt and apply the
    convention of their choice externally. E.g. \f$ P^* \equiv \frac{P \sigma^3}{k_B T} \f$ implies a user should pass
    \f$ P^* / \sigma^3 \f$ as the UpdaterBoxMC P parameter.
*/
class UpdaterBoxMC : public Updater
    {
    public:
        //! Constructor
        /*! \param sysdef System definition
            \param mc HPMC integrator object
            \param P Pressure times thermodynamic beta to apply in isobaric ensembles
            \param frequency average number of box updates per particle super-move
            \param seed PRNG seed

            Variant parameters are possible, but changing MC parameters violates detailed balance.
        */
        UpdaterBoxMC(std::shared_ptr<SystemDefinition> sysdef,
                      std::shared_ptr<IntegratorHPMC> mc,
                      std::shared_ptr<Variant> P,
                      const Scalar frequency,
                      const unsigned int seed);

        //! Destructor
        virtual ~UpdaterBoxMC();

        //! Sets parameters for box volume moves
        /*! \param delta maximum size of volume change
            \param weight relative likelihood of volume move
        */
        void volume(const Scalar delta,
                           const float weight)
            {
            m_Volume_delta = delta;
            m_Volume_weight = weight;
            // Calculate aspect ratio
            computeAspectRatios();
            };

        //! Sets parameters for box volume moves
        /*! \param delta_lnV log of maximum relative size of volume change
            \param weight relative likelihood of volume move
        */
        void ln_volume(const Scalar delta_lnV,
                       const float weight)
            {
            m_lnVolume_delta = delta_lnV;
            m_lnVolume_weight = weight;
            // Calculate aspect ratio
            computeAspectRatios();
            };


        //! Sets parameters for box length moves
        /*! \param dLx Extent of length change distribution in first lattice vector for box resize moves
            \param dLy Extent of length change distribution in second lattice vector for box resize moves
            \param dLz Extent of length change distribution in third lattice vector for box resize moves
            \param weight relative likelihood of volume move
        */
        void length(const Scalar dLx,
                           const Scalar dLy,
                           const Scalar dLz,
                           const float weight)
            {
            m_Length_delta[0] = dLx;
            m_Length_delta[1] = dLy;
            m_Length_delta[2] = dLz;
            m_Length_weight = weight;
            };


        //! Sets parameters for box shear moves
        /*! \param dxy Extent of shear parameter distribution for shear moves in x,y plane
            \param dxz Extent of shear parameter distribution for shear moves in x,z plane
            \param dyz Extent of shear parameter distribution for shear moves in y,z plane
            \param reduce Maximum number of lattice vectors of shear to allow before applying lattice reduction.
                Shear of +/- 0.5 cannot be lattice reduced, so set to a value < 0.5 to disable (default 0)
                Note that due to precision errors, lattice reduction may introduce small overlaps which can be resolved,
                but which temporarily break detailed balance.
            \param weight relative likelihood of shear move
        */
        void shear(const Scalar dxy,
                          const Scalar dxz,
                          const Scalar dyz,
                          const Scalar reduce,
                          const float weight)
            {
            m_Shear_delta[0] = dxy;
            m_Shear_delta[1] = dxz;
            m_Shear_delta[2] = dyz;
            m_Shear_reduce = reduce;
            m_Shear_weight = weight;
            };

        //! Sets parameters for box aspect moves
        /*! \param dA maximum relative aspect ratio change.
            \param weight relative likelihood of aspect move.
        */
        void aspect(const Scalar dA,
                           const float weight)
            {
            m_Aspect_delta = dA;
            m_Aspect_weight = weight;
            };

        //! Calculate aspect ratios for use in isotropic volume changes
        void computeAspectRatios()
            {
            // when volume is changed, we want to set Ly = m_rLy * Lx, etc.
            BoxDim curBox = m_pdata->getGlobalBox();
            Scalar Lx = curBox.getLatticeVector(0).x;
            Scalar Ly = curBox.getLatticeVector(1).y;
            Scalar Lz = curBox.getLatticeVector(2).z;
            m_Volume_A1 = Lx / Ly;
            m_Volume_A2 = Lx / Lz;
            }

        //! Get pressure parameter
        /*! \returns pressure variant object
        */
        std::shared_ptr<Variant> getP()
            {
            return m_P;
            }

        //! Set pressure parameter
        void setP(const std::shared_ptr<Variant>& betaP)
            {
            m_P = betaP;
            }

        //! Print statistics about the MC box update steps taken
        void printStats()
            {
            hpmc_boxmc_counters_t counters = getCounters(1);
            m_exec_conf->msg->notice(2) << "-- HPMC box change stats:" << std::endl;

            if (counters.shear_accept_count + counters.shear_reject_count > 0)
                {
                m_exec_conf->msg->notice(2) << "Average shear acceptance: " << counters.getShearAcceptance() << "\n";
                }
            if (counters.volume_accept_count + counters.volume_reject_count > 0)
                {
                m_exec_conf->msg->notice(2) << "Average volume acceptance: " << counters.getVolumeAcceptance() << std::endl;
                }
            if (counters.ln_volume_accept_count + counters.ln_volume_reject_count > 0)
                {
                m_exec_conf->msg->notice(2) << "Average ln_V acceptance: " << counters.getLogVolumeAcceptance() << std::endl;
                }
            if (counters.aspect_accept_count + counters.aspect_reject_count > 0)
                {
                m_exec_conf->msg->notice(2) << "Average aspect acceptance: " << counters.getAspectAcceptance() << std::endl;
                }

            m_exec_conf->msg->notice(2) << "Total box changes:        " << counters.getNMoves() << std::endl;
            }

        //! Get a list of logged quantities
        virtual std::vector< std::string > getProvidedLogQuantities();

        //! Get the value of a logged quantity
        virtual Scalar getLogValue(const std::string& quantity, unsigned int timestep);

        //! Reset statistics counters
        void resetStats()
            {
            m_count_run_start = m_count_total;
            }

        //! Handle MaxParticleNumberChange signal
        /*! Resize the m_pos_backup array
        */
        void slotMaxNChange()
            {
            unsigned int MaxN = m_pdata->getMaxN();
            m_pos_backup.resize(MaxN);
            }

        //! Take one timestep forward
        /*! \param timestep timestep at which update is being evaluated
        */
        virtual void update(unsigned int timestep);

        //! Get the current counter values
        hpmc_boxmc_counters_t getCounters(unsigned int mode=0);

        //! Perform box update in NpT box length distribution
        /*! \param timestep timestep at which update is being evaluated
            \param rng pseudo random number generator instance
        */
        void update_L(unsigned int timestep, hoomd::RandomGenerator& rng);

        //! Perform box update in NpT volume distribution
        /*! \param timestep timestep at which update is being evaluated
            \param rng pseudo random number generator instance
        */
        void update_V(unsigned int timestep, hoomd::RandomGenerator& rng);

        //! Perform box update in NpT ln(V) distribution
        /*! \param timestep timestep at which update is being evaluated
            \param rng pseudo random number generator instance
        */
        void update_lnV(unsigned int timestep, hoomd::RandomGenerator& rng);


        //! Perform box update in NpT shear distribution
        /*! \param timestep timestep at which update is being evaluated
            \param rng pseudo random number generator instance
        */
        void update_shear(unsigned int timestep, hoomd::RandomGenerator& rng);

        //! Perform non-thermodynamic MC move in aspect ratio.
        /*! \param timestep timestep at which update is being evaluated
            \param rng pseudo random number generator instance
        */
        void update_aspect(unsigned int timestep, hoomd::RandomGenerator& rng);

        //! Get volume change parameter
        const Scalar get_volume_delta() const
            {
            return m_Volume_delta;
            }

        //! Get delta_lnV
        const Scalar get_ln_volume_delta() const
            {
            return m_lnVolume_delta;
            }


        //! Get aspect ratio trial parameter
        const Scalar get_aspect_delta() const
            {
            return m_Aspect_delta;
            }

        //! Get box length trial parameters
        pybind11::tuple get_length_delta() const
            {
            return pybind11::make_tuple(m_Length_delta[0], m_Length_delta[1], m_Length_delta[2]);
            }

        //! Get box shear trial parameters
        pybind11::tuple get_shear_delta() const
            {
            return pybind11::make_tuple(m_Shear_delta[0], m_Shear_delta[1], m_Shear_delta[2]);
            }


    private:
        std::shared_ptr<IntegratorHPMC> m_mc;     //!< HPMC integrator object
        std::shared_ptr<Variant> m_P;             //!< Reduced pressure in isobaric ensembles
        Scalar m_frequency;                         //!< Frequency of BoxMC moves versus HPMC integrator moves

        Scalar m_Volume_delta;                      //!< Amount by which to change parameter during box-change
        float m_Volume_weight;                     //!< relative weight of volume moves
        Scalar m_lnVolume_delta;                      //!< Amount by which to change parameter during box-change
        float m_lnVolume_weight;                   //!< relative weight of volume moves
        Scalar m_Volume_A1;                         //!< Ratio of Lx to Ly to use in isotropic volume changes
        Scalar m_Volume_A2;                         //!< Ratio of Lx to Lz to use in isotropic volume changes

        Scalar m_Length_delta[3];                   //!< Max length change in each dimension
        float m_Length_weight;                     //!< relative weight of length change moves

        Scalar m_Shear_delta[3];                    //!< Max tilt factor change in each dimension
        float m_Shear_weight;                      //!< relative weight of shear moves
        Scalar m_Shear_reduce;                      //!< Tolerance for automatic box lattice reduction

        Scalar m_Aspect_delta;                      //!< Maximum relative aspect ratio change in randomly selected dimension
        float m_Aspect_weight;                     //!< relative weight of aspect ratio moves

        GPUArray<Scalar4> m_pos_backup;             //!< hold backup copy of particle positions

        hpmc_boxmc_counters_t m_count_total;          //!< Accept/reject total count
        hpmc_boxmc_counters_t m_count_run_start;      //!< Count saved at run() start
        hpmc_boxmc_counters_t m_count_step_start;     //!< Count saved at the start of the last step

        unsigned int m_seed;                        //!< Seed for pseudo-random number generator

        inline bool is_oversheared();               //!< detect oversheared box
        inline bool remove_overshear();             //!< detect and remove overshear
        inline bool box_resize(Scalar Lx,
                               Scalar Ly,
                               Scalar Lz,
                               Scalar xy,
                               Scalar xz,
                               Scalar yz
                               );
                               //!< perform specified box change, if possible
        inline bool box_resize_trial(Scalar Lx,
                                     Scalar Ly,
                                     Scalar Lz,
                                     Scalar xy,
                                     Scalar xz,
                                     Scalar yz,
                                     unsigned int timestep,
                                     Scalar boltzmann,
                                     hoomd::RandomGenerator& rng
                                     );
                                     //!< attempt specified box change and undo if overlaps generated
        inline bool safe_box(const Scalar newL[3], const unsigned int& Ndim);
                                                    //!< Perform appropriate checks for box validity
    };

//! Export UpdaterBoxMC to Python
void export_UpdaterBoxMC(pybind11::module& m);

} // end namespace hpmc

#endif // _UPDATER_HPMC_BOX_MC_
