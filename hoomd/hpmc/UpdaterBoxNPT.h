// Copyright (c) 2009-2016 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.

// inclusion guard
#ifndef _UPDATER_HPMC_NPT_
#define _UPDATER_HPMC_NPT_

/*! \file UpdaterBoxNPT.h
    \brief Declaration of UpdaterBoxNPT
*/


#include "hoomd/Updater.h"
#include "hoomd/Variant.h"
#include "hoomd/extern/saruprng.h"
#include <cmath>

// Need Moves.h for rand_select
#include "Moves.h"

#include "IntegratorHPMC.h"

namespace hpmc
{

//! Update box for HPMC simulation in the NPT ensemble
/*! The pressure parameter is beta*P. For a unitless reduced pressure, the user must adopt and apply the
    convention of their choice externally. E.g. \f$ P^* \equiv \frac{P \sigma^3}{k_B T} \f$ implies a user should pass
    \f$ P^* / \sigma^3 \f$ as the UpdaterBoxNPT P parameter.
*/
class UpdaterBoxNPT : public Updater
    {
    public:
        //! Constructor
        /*! \param sysdef System definition
            \param mc HPMC integrator object
            \param P Pressure times thermodynamic beta to apply in NPT ensemble
            \param dLx Extent of length change distribution in first lattice vector for box resize moves
            \param dLy Extent of length change distribution in second lattice vector for box resize moves
            \param dLz Extent of length change distribution in third lattice vector for box resize moves
            \param dxy Extent of shear parameter distribution for shear moves in x,y plane
            \param dxz Extent of shear parameter distribution for shear moves in x,z plane
            \param dyz Extent of shear parameter distribution for shear moves in y,z plane
            \param move_ratio Probability of attempting lattice vector length moves
                (1.0 minus probability of attempting shearing moves)
            \param reduce Maximum number of lattice vectors of shear to allow before applying lattice reduction.
                Shear of +/- 0.5 cannot be lattice reduced, so set to a value < 0.5 to disable (default 0)
                Note that due to precision errors, lattice reduction may introduce small overlaps which can be resolved,
                but which temporarily break detailed balance.
            \param isotropic Set to true to link Lx, Ly, and Lz. The Ly and Lz parameters are then ignored
            \param seed PRNG seed

            Variant parameters are possible, but changing MC parameters violates detailed balance.
        */
        UpdaterBoxNPT(std::shared_ptr<SystemDefinition> sysdef,
                      std::shared_ptr<IntegratorHPMC> mc,
                      std::shared_ptr<Variant> P,
                      Scalar dLx,
                      Scalar dLy,
                      Scalar dLz,
                      Scalar dxy,
                      Scalar dxz,
                      Scalar dyz,
                      Scalar move_ratio,
                      Scalar reduce,
                      bool isotropic,
                      const unsigned int seed);

        //! Destructor
        virtual ~UpdaterBoxNPT();

        //! Sets parameters
        /*! \param P Pressure times thermodynamic beta to apply in NPT ensemble
            \param dLx Extent of length change distribution in first lattice vector for box resize moves
            \param dLy Extent of length change distribution in second lattice vector for box resize moves
            \param dLz Extent of length change distribution in third lattice vector for box resize moves
            \param dxy Extent of shear parameter distribution for shear moves in x,y plane
            \param dxz Extent of shear parameter distribution for shear moves in x,z plane
            \param dyz Extent of shear parameter distribution for shear moves in y,z plane
            \param move_ratio Probably of attempting lattice vector length moves
                (1.0 minus probability of attempting shearing moves)
            \param reduce Maximum number of lattice vectors of shear to allow before applying lattice reduction.
                Shear of +/- 0.5 cannot be lattice reduced, so set to a value < 0.5 to disable (default 0)
                Note that due to precision errors, lattice reduction may introduce small overlaps which can be resolved,
                but which temporarily break detailed balance.
            \param isotropic Set to true to link Lx, Ly, and Lz. The Ly and Lz parameters are then ignored

            Variant parameters are possible, but changing MC parameters violates detailed balance.
        */
        void setParams(std::shared_ptr<Variant> P,
                       Scalar dLx,
                       Scalar dLy,
                       Scalar dLz,
                       Scalar dxy,
                       Scalar dxz,
                       Scalar dyz,
                       Scalar move_ratio,
                       Scalar reduce,
                       bool isotropic)
            {
            m_P = P;
            m_dLx = dLx;
            m_dLy = dLy;
            m_dLz = dLz;
            m_dxy = dxy;
            m_dxz = dxz;
            m_dyz = dyz;
            m_move_ratio = move_ratio;
            m_reduce = reduce;

            // Calculate aspect ratio when switching from anisotropic to isotropic box changes
            if (isotropic && !m_isotropic)
                {
                computeAspectRatios();
                }
            // Update derived quantity for isotropic volume changes
            if (isotropic)
                {
                BoxDim curBox = m_pdata->getGlobalBox();
                m_dV = dLx * curBox.getLatticeVector(1).y * curBox.getLatticeVector(2).z;
                }

            m_isotropic = isotropic;
            };

        //! Calculate aspect ratios for use in isotropic volume changes
        void computeAspectRatios()
            {
            // when volume is changed, we want to set Ly = m_rLy * Lx, etc.
            BoxDim curBox = m_pdata->getGlobalBox();
            Scalar Lx = curBox.getLatticeVector(0).x;
            Scalar Ly = curBox.getLatticeVector(1).y;
            Scalar Lz = curBox.getLatticeVector(2).z;
            m_rLy = Ly / Lx;
            m_rLz = Lz / Lx;
            }

        //! Get pressure parameter
        /*! \returns pressure variant object
        */
        std::shared_ptr<Variant> getP()
            {
            return m_P;
            }

        //! Get dLx max trial size parameter
        /*! \returns dLx parameter
        */
        Scalar getdLx()
            {
            return m_dLx;
            }

        //! Get dLy max trial size parameter
        /*! \returns dLy parameter
        */
        Scalar getdLy()
            {
            return m_dLy;
            }

        //! Get dLz max trial size parameter
        /*! \returns dLz parameter
        */
        Scalar getdLz()
            {
            return m_dLz;
            }

        //! Get dxy max trial size parameter
        /*! \returns dxy parameter
        */
        Scalar getdxy()
            {
            return m_dxy;
            }

        //! Get dxz max trial size parameter
        /*! \returns dxz parameter
        */
        Scalar getdxz()
            {
            return m_dxz;
            }

        //! Get dyz max trial size parameter
        /*! \returns dyz parameter
        */
        Scalar getdyz()
            {
            return m_dyz;
            }

        //! Get move_ratio parameter
        /*! \returns move_ratio parameter
        */
        Scalar getMoveRatio()
            {
            return m_move_ratio;
            }

        //! Get lattice reduction threshold
        /*! \returns reduce parameter
        */
        Scalar getReduce()
            {
            return m_reduce;
            }

        //! Get status of isotropic versus anisotropic volume changes
        /*! \returns True if isotropic parameter is set
        */
        bool getIsotropic()
            {
            return m_isotropic;
            }

        //! Print statistics about the NPT box update steps taken
        void printStats()
            {
            hpmc_npt_counters_t counters = getCounters(1);
            m_exec_conf->msg->notice(2) << "-- HPMC NPT box change stats:" << std::endl;

            if (counters.shear_accept_count + counters.shear_reject_count > 0)
                {
                m_exec_conf->msg->notice(2) << "Average shear acceptance: " << counters.getShearAcceptance() << "\n";
                }
            if (counters.volume_accept_count + counters.volume_reject_count > 0)
                {
                m_exec_conf->msg->notice(2) << "Average volume acceptance: " << counters.getVolumeAcceptance() << std::endl;
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
        hpmc_npt_counters_t getCounters(unsigned int mode=0);

    private:
        std::shared_ptr<IntegratorHPMC> m_mc;     //!< HPMC integrator object
        std::shared_ptr<Variant> m_P;             //!< Reduced pressure in NPT ensemble
        Scalar m_dLx;                               //!< Amount by which to change lattice vector length during volume-change
        Scalar m_dLy;                               //!< Amount by which to change lattice vector length during volume-change
        Scalar m_dLz;                               //!< Amount by which to change lattice vector length during volume-change
        Scalar m_dV;                                //!< Amount by which to change volume during volume-change
        Scalar m_dxy;                               //!< Amount by which to adjust box angle parameter during shear
        Scalar m_dxz;                               //!< Amount by which to adjust box angle parameter during shear
        Scalar m_dyz;                               //!< Amount by which to adjust box angle parameter during shear
        Scalar m_move_ratio;                        //!< Ratio of lattice vector length versus shearing move frequency
        Scalar m_reduce;                            //!< Threshold for lattice reduction
        bool m_isotropic;                           //!< If true, dLx and dLy are ignored and volume changes are linked to Lx
        Scalar m_rLy;                               //!< Ratio of Ly to Lx to use in isotropic volume changes
        Scalar m_rLz;                               //!< Ratio of Lz to Lx to use in isotropic volume changes

        GPUArray<Scalar4> m_pos_backup;             //!< hold backup copy of particle positions
        boost::signals2::connection m_maxparticlenumberchange_connection;
                                                    //!< Connection to MaxParticleNumberChange signal

        hpmc_npt_counters_t m_count_total;          //!< Accept/reject total count
        hpmc_npt_counters_t m_count_run_start;      //!< Count saved at run() start
        hpmc_npt_counters_t m_count_step_start;     //!< Count saved at the start of the last step

        unsigned int m_seed;                        //!< Seed for pseudo-random number generator

        inline bool is_oversheared();               //!< detect oversheared box
        inline bool remove_overshear();             //!< detect and remove overshear
        inline bool box_resize(Scalar Lx, Scalar Ly, Scalar Lz, Scalar xy, Scalar xz, Scalar yz);
                                                    //!< perform specified box change, if possible
        inline bool box_resize_trial( Scalar Lx,
                                      Scalar Ly,
                                      Scalar Lz,
                                      Scalar xy,
                                      Scalar xz,
                                      Scalar yz,
                                      unsigned int timestep,
                                      Scalar boltzmann,
                                      Saru& rng
                                      );
                                      //!< attempt specified box change and undo if overlaps generated
    };

//! Export UpdaterBoxNPT to Python
void export_UpdaterBoxNPT();

} // end namespace hpmc

#endif // _UPDATER_HPMC_NPT_
