// Copyright (c) 2009-2024 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

// inclusion guard
#ifndef _UPDATER_HPMC_BOX_MC_
#define _UPDATER_HPMC_BOX_MC_

/*! \file UpdaterBoxMC.h
    \brief Declaration of UpdaterBoxMC
*/

#include "hoomd/RandomNumbers.h"
#include <cmath>
#include <hoomd/Updater.h>
#include <hoomd/Variant.h>
#include <vector>

#include "IntegratorHPMC.h"

#ifndef __HIPCC__
#include <pybind11/pybind11.h>
#endif

namespace hoomd
    {
namespace hpmc
    {
//! Update box for HPMC simulation in the NPT ensemble, etc.
/*! The pressure parameter is beta*P. For a unitless reduced pressure, the user must adopt and apply
   the convention of their choice externally. E.g. \f$ P^* \equiv \frac{P \sigma^3}{k_B T} \f$
   implies a user should pass \f$ P^* / \sigma^3 \f$ as the UpdaterBoxMC P parameter.
*/
class UpdaterBoxMC : public Updater
    {
    public:
    //! Constructor
    /*! \param sysdef System definition
        \param mc HPMC integrator object
        \param P Pressure times thermodynamic beta to apply in isobaric ensembles

        Variant parameters are possible, but changing MC parameters violates detailed balance.
    */
    UpdaterBoxMC(std::shared_ptr<SystemDefinition> sysdef,
                 std::shared_ptr<Trigger> trigger,
                 std::shared_ptr<IntegratorHPMC> mc,
                 std::shared_ptr<Variant> P);

    //! Destructor
    virtual ~UpdaterBoxMC();

    //! Get parameters for box volume moves as a dictionary
    /*! dict keys:
         delta - maximum size of volume change
         weight - relative likelihood of volume move.
    */
    pybind11::dict getVolumeParams()
        {
        pybind11::dict d;
        d["mode"] = m_volume_mode;
        d["weight"] = m_volume_mode == "standard" ? m_volume_weight : m_ln_volume_weight;
        d["delta"] = m_volume_mode == "standard" ? m_volume_delta : m_ln_volume_delta;
        return d;
        }

    //! Set parameters for box volume moves from a dictionary
    /*! dict keys:
         mode - choose between standard and log volume moves
         delta - maximum size of volume change
         weight - relative likelihood of volume move.
    */
    void setVolumeParams(pybind11::dict d)
        {
        m_volume_mode = d["mode"].cast<std::string>();
        if (m_volume_mode == "standard")
            {
            m_volume_weight = d["weight"].cast<Scalar>();
            m_volume_delta = d["delta"].cast<Scalar>();
            m_ln_volume_weight = 0.0;
            m_ln_volume_delta = 0.0;
            }
        else if (m_volume_mode == "ln")
            {
            m_ln_volume_weight = d["weight"].cast<Scalar>();
            m_ln_volume_delta = d["delta"].cast<Scalar>();
            m_volume_weight = 0.0;
            m_volume_delta = 0.0;
            }
        else
            {
            throw std::runtime_error("Unknown mode for volume moves");
            }
        // Calculate aspect ratio
        computeAspectRatios();
        updateChangedWeights();
        }

    //! Gets parameters for box length moves as a dictionary
    /*! dict keys:
         delta - list ([dLx, dLy, dLz]) containing the extent of the length change
                 distribution in the first, second and third lattice vector respectively
         weight - relative likelihood of length moves
    */
    pybind11::dict getLengthParams()
        {
        pybind11::dict d;
        d["weight"] = m_length_weight;
        d["delta"] = pybind11::make_tuple(m_length_delta[0], m_length_delta[1], m_length_delta[2]);
        return d;
        }

    //! Sets parameters for box length moves from a dictionary
    /*! dict keys:
         delta - list ([dLx, dLy, dLz]) containing the extent of the length change
                 distribution in the first, second and third lattice vector respectively
         weight - relative likelihood of length moves
    */
    void setLengthParams(pybind11::dict d)
        {
        m_length_weight = d["weight"].cast<Scalar>();
        pybind11::tuple t = d["delta"];
        m_length_delta[0] = t[0].cast<Scalar>();
        m_length_delta[1] = t[1].cast<Scalar>();
        m_length_delta[2] = t[2].cast<Scalar>();
        updateChangedWeights();
        }

    //! Gets parameters for box shear moves as a dictionary
    /*! dict keys:
         delta - list ([dxy, dxz, dyz]) containing extent of shear parameter
                 distribution for shear moves in xy, xz and yz planes respectively
         weight - relative likelihood of shear move
         reduce - maximum number of lattice vectors of shear to allow before applying lattice
       reduction. Shear of +/- 0.5 cannot be lattice reduced, so set to a value < 0.5 to disable
       (default 0) Note that due to precision errors, lattice reduction may introduce small overlaps
       which can be resolved, but which temporarily break detailed balance.
    */
    pybind11::dict getShearParams()
        {
        pybind11::dict d;
        d["weight"] = m_shear_weight;
        d["reduce"] = m_shear_reduce;
        d["delta"] = pybind11::make_tuple(m_shear_delta[0], m_shear_delta[1], m_shear_delta[2]);
        return d;
        }

    //! Gets parameters for box shear moves as a dictionary
    /*! dict keys:
         delta - list ([dxy, dxz, dyz]) containing extent of shear parameter
                 distribution for shear moves in xy, xz and yz planes respectively
         weight - relative likelihood of shear move
         reduce - maximum number of lattice vectors of shear to allow before applying lattice
       reduction. Shear of +/- 0.5 cannot be lattice reduced, so set to a value < 0.5 to disable
       (default 0) Note that due to precision errors, lattice reduction may introduce small overlaps
       which can be resolved, but which temporarily break detailed balance.
    */
    void setShearParams(pybind11::dict d)
        {
        m_shear_weight = d["weight"].cast<Scalar>();
        pybind11::tuple t = d["delta"];
        m_shear_delta[0] = t[0].cast<Scalar>();
        m_shear_delta[1] = t[1].cast<Scalar>();
        m_shear_delta[2] = t[2].cast<Scalar>();
        m_shear_reduce = d["reduce"].cast<Scalar>();
        updateChangedWeights();
        }

    //! Get parameters for box aspect moves as a dictionary
    /*! dict keys:
         delta - maximum relative aspect ratio change.
         weight - relative likelihood of aspect move.
    */
    pybind11::dict getAspectParams()
        {
        pybind11::dict d;
        d["weight"] = m_aspect_weight;
        d["delta"] = m_aspect_delta;
        return d;
        }

    //! Set parameters for box aspect moves from a dictionary
    /*! dict keys:
         delta - maximum relative aspect ratio change.
         weight - relative likelihood of aspect move.
    */
    void setAspectParams(pybind11::dict d)
        {
        m_aspect_weight = d["weight"].cast<Scalar>();
        m_aspect_delta = d["delta"].cast<Scalar>();
        updateChangedWeights();
        }

    //! Calculate aspect ratios for use in isotropic volume changes
    void computeAspectRatios()
        {
        // when volume is changed, we want to set Ly = m_rLy * Lx, etc.
        BoxDim curBox = m_pdata->getGlobalBox();
        Scalar Lx = curBox.getLatticeVector(0).x;
        Scalar Ly = curBox.getLatticeVector(1).y;
        Scalar Lz = curBox.getLatticeVector(2).z;
        m_volume_A1 = Lx / Ly;
        m_volume_A2 = Lx / Lz;
        }

    //! Get pressure parameter
    /*! \returns pressure variant object
     */
    std::shared_ptr<Variant> getBetaP()
        {
        return m_beta_P;
        }

    //! Set pressure parameter
    void setBetaP(const std::shared_ptr<Variant>& betaP)
        {
        m_beta_P = betaP;
        }

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
    virtual void update(uint64_t timestep);

    //! Get the current counter values
    hpmc_boxmc_counters_t getCounters(unsigned int mode = 0);

    //! Perform box update in NpT box length distribution
    /*! \param timestep timestep at which update is being evaluated
        \param rng pseudo random number generator instance
    */
    void update_L(uint64_t timestep, hoomd::RandomGenerator& rng);

    //! Perform box update in NpT volume distribution
    /*! \param timestep timestep at which update is being evaluated
        \param rng pseudo random number generator instance
    */
    void update_V(uint64_t timestep, hoomd::RandomGenerator& rng);

    //! Perform box update in NpT ln(V) distribution
    /*! \param timestep timestep at which update is being evaluated
        \param rng pseudo random number generator instance
    */
    void update_lnV(uint64_t timestep, hoomd::RandomGenerator& rng);

    //! Perform box update in NpT shear distribution
    /*! \param timestep timestep at which update is being evaluated
        \param rng pseudo random number generator instance
    */
    void update_shear(uint64_t timestep, hoomd::RandomGenerator& rng);

    //! Perform non-thermodynamic MC move in aspect ratio.
    /*! \param timestep timestep at which update is being evaluated
        \param rng pseudo random number generator instance
    */
    void update_aspect(uint64_t timestep, hoomd::RandomGenerator& rng);

    /// Set the RNG instance
    void setInstance(unsigned int instance)
        {
        m_instance = instance;
        }

    /// Get the RNG instance
    unsigned int getInstance()
        {
        return m_instance;
        }

    private:
    std::shared_ptr<IntegratorHPMC> m_mc; //!< HPMC integrator object
    std::shared_ptr<Variant> m_beta_P;    //!< Reduced pressure in isobaric ensembles

    unsigned int m_instance = 0; //!< Unique ID for RNG seeding

    Scalar m_volume_delta;     //!< Amount by which to change volume during box-change
    Scalar m_volume_weight;    //!< relative weight of volume moves
    Scalar m_ln_volume_delta;  //!< Amount by which to log volume parameter during box-change
    Scalar m_ln_volume_weight; //!< relative weight of log volume moves
    std::string m_volume_mode; //!< volume moves mode: standard or logarithmic
    Scalar m_volume_A1;        //!< Ratio of Lx to Ly to use in isotropic volume changes
    Scalar m_volume_A2;        //!< Ratio of Lx to Lz to use in isotropic volume changes

    Scalar m_length_delta[3]; //!< Max length change in each dimension
    Scalar m_length_weight;   //!< relative weight of length change moves

    Scalar m_shear_delta[3]; //!< Max tilt factor change in each dimension
    Scalar m_shear_weight;   //!< relative weight of shear moves
    Scalar m_shear_reduce;   //!< Tolerance for automatic box lattice reduction

    Scalar m_aspect_delta;  //!< Maximum relative aspect ratio change in randomly selected dimension
    Scalar m_aspect_weight; //!< relative weight of aspect ratio moves

    GPUArray<Scalar4> m_pos_backup; //!< hold backup copy of particle positions

    hpmc_boxmc_counters_t m_count_total;      //!< Accept/reject total count
    hpmc_boxmc_counters_t m_count_run_start;  //!< Count saved at run() start
    hpmc_boxmc_counters_t m_count_step_start; //!< Count saved at the start of the last step

    std::vector<Scalar> m_weight_partial_sums; //!< Partial sums of all weights used to select moves

    inline bool is_oversheared();   //!< detect oversheared box
    inline bool remove_overshear(); //!< detect and remove overshear
    inline bool box_resize(Scalar Lx, Scalar Ly, Scalar Lz, Scalar xy, Scalar xz, Scalar yz);
    //!< perform specified box change, if possible
    inline bool box_resize_trial(Scalar Lx,
                                 Scalar Ly,
                                 Scalar Lz,
                                 Scalar xy,
                                 Scalar xz,
                                 Scalar yz,
                                 uint64_t timestep,
                                 Scalar boltzmann,
                                 hoomd::RandomGenerator& rng);
    //!< attempt specified box change and undo if overlaps generated
    inline bool safe_box(const Scalar newL[3], const unsigned int& Ndim);
    //!< Perform appropriate checks for box validity

    //! Update the internal vector of partial sums of weights
    void updateChangedWeights();
    };

namespace detail
    {
//! Export UpdaterBoxMC to Python
void export_UpdaterBoxMC(pybind11::module& m);
    } // end namespace detail
    } // end namespace hpmc
    } // end namespace hoomd

#endif // _UPDATER_HPMC_BOX_MC_
