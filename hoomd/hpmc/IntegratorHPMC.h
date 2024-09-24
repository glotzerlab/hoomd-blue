// Copyright (c) 2009-2024 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

// inclusion guard
#ifndef _INTEGRATOR_HPMC_H_
#define _INTEGRATOR_HPMC_H_

/*! \file IntegratorHPMC.h
    \brief Declaration of IntegratorHPMC
*/

#ifdef ENABLE_HIP
#include <hip/hip_runtime.h>
#endif

#include "hoomd/CellList.h"
#include "hoomd/Integrator.h"
#include <hoomd/Variant.h>

#include "ExternalField.h"
#include "ExternalPotential.h"
#include "HPMCCounters.h"
#include "PairPotential.h"

#ifndef __HIPCC__
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#endif

#ifdef ENABLE_HIP
#include "hoomd/Autotuner.h"
#include "hoomd/GPUPartition.cuh"
#endif

namespace hoomd
    {
namespace hpmc
    {
namespace detail
    {
#ifdef ENABLE_HIP
//! Wraps arguments to kernel::narow_phase_patch functions
struct hpmc_patch_args_t
    {
    //! Construct a hpmc_patch_args_t
    hpmc_patch_args_t(const Scalar4* _d_postype,
                      const Scalar4* _d_orientation,
                      const Scalar4* _d_trial_postype,
                      const Scalar4* _d_trial_orientation,
                      const unsigned int* _d_trial_move_type,
                      const Index3D& _ci,
                      const uint3& _cell_dim,
                      const Scalar3& _ghost_width,
                      const unsigned int _N,
                      const uint16_t _seed,
                      const unsigned int _rank,
                      const uint64_t _timestep,
                      const unsigned int _select,
                      const unsigned int _num_types,
                      const BoxDim& _box,
                      const unsigned int* _d_excell_idx,
                      const unsigned int* _d_excell_size,
                      const Index2D& _excli,
                      const Scalar _r_cut_patch,
                      const Scalar* _d_additive_cutoff,
                      const unsigned int* _d_update_order_by_ptl,
                      const unsigned int* _d_reject_in,
                      unsigned int* _d_reject_out,
                      const Scalar* _d_charge,
                      const Scalar* _d_diameter,
                      const unsigned int* _d_reject_out_of_cell,
                      const GPUPartition& _gpu_partition)
        : d_postype(_d_postype), d_orientation(_d_orientation), d_trial_postype(_d_trial_postype),
          d_trial_orientation(_d_trial_orientation), d_trial_move_type(_d_trial_move_type), ci(_ci),
          cell_dim(_cell_dim), ghost_width(_ghost_width), N(_N), seed(_seed), rank(_rank),
          timestep(_timestep), select(_select), num_types(_num_types), box(_box),
          d_excell_idx(_d_excell_idx), d_excell_size(_d_excell_size), excli(_excli),
          r_cut_patch(_r_cut_patch), d_additive_cutoff(_d_additive_cutoff),
          d_update_order_by_ptl(_d_update_order_by_ptl), d_reject_in(_d_reject_in),
          d_reject_out(_d_reject_out), d_charge(_d_charge), d_diameter(_d_diameter),
          d_reject_out_of_cell(_d_reject_out_of_cell), gpu_partition(_gpu_partition)
        {
        }

    const Scalar4* d_postype;              //!< postype array
    const Scalar4* d_orientation;          //!< orientation array
    const Scalar4* d_trial_postype;        //!< New positions (and type) of particles
    const Scalar4* d_trial_orientation;    //!< New orientations of particles
    const unsigned int* d_trial_move_type; //!< 0=no move, 1/2 = translate/rotate
    const Index3D& ci;                     //!< Cell indexer
    const uint3& cell_dim;                 //!< Cell dimensions
    const Scalar3& ghost_width;            //!< Width of the ghost layer
    const unsigned int N;                  //!< Number of particles
    const uint16_t seed;                   //!< RNG seed
    const unsigned int rank;               //!< MPI Rank
    const uint64_t timestep;               //!< Current timestep
    const unsigned int select;
    const unsigned int num_types;              //!< Number of particle types
    const BoxDim box;                          //!< Current simulation box
    const unsigned int* d_excell_idx;          //!< Expanded cell list
    const unsigned int* d_excell_size;         //!< Size of expanded cells
    const Index2D& excli;                      //!< Excell indexer
    const Scalar r_cut_patch;                  //!< Global cutoff radius
    const Scalar* d_additive_cutoff;           //!< Additive contribution to cutoff per type
    const unsigned int* d_update_order_by_ptl; //!< Order of the update sequence
    const unsigned int* d_reject_in;           //!< Previous reject flags
    unsigned int* d_reject_out;                //!< New reject flags
    const Scalar* d_charge;                    //!< Particle charges
    const Scalar* d_diameter;                  //!< Particle diameters
    const unsigned int*
        d_reject_out_of_cell;          //!< Flag if a particle move has been rejected a priori
    const GPUPartition& gpu_partition; //!< split particles among GPUs
    };
#endif

    } // end namespace detail

//! Integrator that implements the HPMC approach
/*! **Overview** <br>
    IntegratorHPMC is an non-templated base class that implements the basic methods that all HPMC
   integrators have. This provides a base interface that any other code can use when given a shared
   pointer to an IntegratorHPMC.

    The move ratio is stored as an unsigned int (0xffff = 100%) to avoid numerical issues when the
   move ratio is exactly at 100%.

    \ingroup hpmc_integrators
*/
class PYBIND11_EXPORT IntegratorHPMC : public Integrator
    {
    public:
    //! Constructor
    IntegratorHPMC(std::shared_ptr<SystemDefinition> sysdef);

    virtual ~IntegratorHPMC();

    //! Take one timestep forward
    virtual void update(uint64_t timestep)
        {
        ArrayHandle<hpmc_counters_t> h_counters(m_count_total,
                                                access_location::host,
                                                access_mode::read);
        m_count_step_start = h_counters.data[0];
        }

    //! Change maximum displacement
    /*! \param typ Name of type to set
     *! \param d new d to set
     */
    inline void setD(std::string name, Scalar d)
        {
        unsigned int id = this->m_pdata->getTypeByName(name);

            {
            ArrayHandle<Scalar> h_d(m_d, access_location::host, access_mode::readwrite);
            h_d.data[id] = d;
            }

        updateCellWidth();
        }

    //! Get maximum displacement (by type name)
    inline Scalar getD(std::string name)
        {
        unsigned int id = this->m_pdata->getTypeByName(name);
        ArrayHandle<Scalar> h_d(m_d, access_location::host, access_mode::read);
        return h_d.data[id];
        }

    //! Get array of translation move sizes
    const GPUArray<Scalar>& getDArray() const
        {
        return m_d;
        }

    //! Get the maximum particle translational move size
    virtual Scalar getMaxTransMoveSize()
        {
        // access the type parameters
        ArrayHandle<Scalar> h_d(m_d, access_location::host, access_mode::read);

        // for each type, create a temporary shape and return the maximum diameter
        Scalar maxD = Scalar(0.0);
        for (unsigned int typ = 0; typ < this->m_pdata->getNTypes(); typ++)
            {
            maxD = std::max(maxD, h_d.data[typ]);
            }

        return maxD;
        }

    //! Get the minimum particle translational move size
    virtual Scalar getMinTransMoveSize()
        {
        // access the type parameters
        ArrayHandle<Scalar> h_d(m_d, access_location::host, access_mode::read);

        // for each type, create a temporary shape and return the maximum diameter
        Scalar minD = h_d.data[0];
        for (unsigned int typ = 1; typ < this->m_pdata->getNTypes(); typ++)
            {
            minD = std::max(minD, h_d.data[typ]);
            }

        return minD;
        }

    //! Change maximum rotation
    /*! \param name Type name to set
     *! \param a new a to set
     */
    inline void setA(std::string name, Scalar a)
        {
        unsigned int id = this->m_pdata->getTypeByName(name);
        ArrayHandle<Scalar> h_a(m_a, access_location::host, access_mode::readwrite);
        h_a.data[id] = a;
        }

    //! Get maximum rotation by name
    inline Scalar getA(std::string name)
        {
        unsigned int id = this->m_pdata->getTypeByName(name);
        ArrayHandle<Scalar> h_a(m_a, access_location::host, access_mode::read);
        return h_a.data[id];
        }

    //! Get array of rotation move sizes
    const GPUArray<Scalar>& getAArray() const
        {
        return m_a;
        }

    //! Change translation move probability.
    /*! \param translation_move_probability new translation_move_probability to set
     */
    void setTranslationMoveProbability(Scalar translation_move_probability)
        {
        m_translation_move_probability = unsigned(translation_move_probability * 65536);
        }

    //! Get translation move probability.
    //! \returns Fraction of moves that are translation moves.
    inline double getTranslationMoveProbability()
        {
        return m_translation_move_probability / 65536.0;
        }

    //! Set nselect
    /*! \param nselect new nselect value to set
     */
    void setNSelect(unsigned int nselect)
        {
        m_nselect = nselect;
        updateCellWidth();
        }

    //! Get nselect
    //! \returns current value of nselect parameter
    inline unsigned int getNSelect()
        {
        return m_nselect;
        }

    //! Set kT variant
    /*! \param kT new k_BT variant to set
     */
    void setkT(std::shared_ptr<Variant> kT)
        {
        m_kT = kT;
        }

    //! Get kT variant
    //! \returns current value of kT parameter
    std::shared_ptr<Variant> getkT()
        {
        return m_kT;
        }

    /** Evaluate the kT variant at the given timestep.

        @param timestep The simulation timestep.
        @returns The value of the kT variant at the given timestep.
    */
    Scalar getTimestepkT(uint64_t timestep)
        {
        return m_kT->operator()(timestep);
        }

    //! Get performance in moves per second
    virtual double getMPS()
        {
        return m_mps;
        }

    //! Reset statistics counters
    virtual void resetStats()
        {
        ArrayHandle<hpmc_counters_t> h_counters(m_count_total,
                                                access_location::host,
                                                access_mode::read);
        m_count_run_start = h_counters.data[0];
        m_clock = ClockSource();
        }

    //! Get the diameter of the largest circumscribing sphere for objects handled by this integrator
    virtual Scalar getMaxCoreDiameter()
        {
        return 1.0;
        }

    //! Count the number of particle overlaps
    /*! \param timestep current step
        \param early_exit exit at first overlap found if true
        \returns number of overlaps if early_exit=false, 1 if early_exit=true
    */
    virtual unsigned int countOverlaps(bool early_exit)
        {
        return 0;
        }

    //! Get the number of degrees of freedom granted to a given group
    /*! \param group Group over which to count degrees of freedom.
        \return a non-zero dummy value to suppress warnings.

        MC does not integrate with the MD computations that use this value.
    */
    virtual Scalar getTranslationalDOF(std::shared_ptr<ParticleGroup> group)
        {
        return 1;
        }

    //! Check the particle data for non-normalized orientations
    virtual bool checkParticleOrientations();

    //! Get the current counter values
    hpmc_counters_t getCounters(unsigned int mode = 0);

    //! Communicate particles
    /*! \param migrate Set to true to both migrate and exchange, set to false to only exchange

        This method exists so that the python API can force MPI communication when needed, e.g.
       before a count_overlaps call to ensure that particle data is up to date.

        The base class does nothing and leaves derived classes to implement.
    */
    virtual void communicate(bool migrate) { }

    //! Set extra ghost width
    /*! \param extra Extra width to add to the ghost layer

        This method is called by AnalyzerSDF when needed to note that an extra padding on the ghost
       layer is needed
    */
    void setExtraGhostWidth(Scalar extra)
        {
        m_extra_ghost_width = extra;
        updateCellWidth();
        }
    //! Method to scale the box
    virtual bool attemptBoxResize(uint64_t timestep, const BoxDim& new_box);

    ExternalField* getExternalField()
        {
        return m_external_base;
        }

    /// Compute the total energy due to potentials in m_external_potentials
    /** Does NOT include external energies in the soon to be removed m_external_base.
     */
    double computeTotalExternalEnergy(bool trial = false)
        {
        double total_energy = 0.0;

        for (const auto& external : m_external_potentials)
            {
            total_energy += external->totalEnergy(trial);
            }

        return total_energy;
        }

    //! Compute the total energy from pair interactions.
    /*! \param timestep the current time step
     * \returns the total patch energy
     */
    virtual double computeTotalPairEnergy(uint64_t timestep)
        {
        // base class method returns 0
        return 0.0;
        }

    //! Prepare for the run
    virtual void prepRun(uint64_t timestep)
        {
        Integrator::prepRun(timestep);
        m_past_first_run = true;
        }

    /// Test if this has pairwise interactions.
    bool hasPairInteractions()
        {
        return m_pair_potentials.size() > 0;
        }

    /// Get pairwise interaction maximum non-additive r_cut.
    LongReal getMaxPairEnergyRCutNonAdditive() const
        {
        LongReal r_cut = 0;
        for (const auto& pair : m_pair_potentials)
            {
            r_cut = std::max(r_cut, pair->getMaxRCutNonAdditive());
            }

        return r_cut;
        }

    /// Get pairwise interaction maximum additive r_cut.
    LongReal getMaxPairInteractionAdditiveRCut(unsigned int type) const
        {
        LongReal r_cut = 0;
        for (const auto& pair : m_pair_potentials)
            {
            r_cut = std::max(r_cut, pair->getRCutAdditive(type));
            }

        return r_cut;
        }

    __attribute__((always_inline)) inline LongReal computeOnePairEnergy(const LongReal r_squared,
                                                                        const vec3<LongReal>& r_ij,
                                                                        unsigned int type_i,
                                                                        const quat<LongReal>& q_i,
                                                                        LongReal d_i,
                                                                        LongReal charge_i,
                                                                        unsigned int type_j,
                                                                        const quat<LongReal>& q_j,
                                                                        LongReal d_j,
                                                                        LongReal charge_j)
        {
        LongReal energy = 0;
        for (const auto& pair : m_pair_potentials)
            {
            if (r_squared < pair->getRCutSquaredTotal(type_i, type_j))
                {
                energy
                    += pair->energy(r_squared, r_ij, type_i, q_i, charge_i, type_j, q_j, charge_j);
                }
            }

        return energy;
        }

    /*** Evaluate the total energy of all external fields interacting with one particle.

        @param type_i Type index of the particle.
        @param r_i Posiion of the particle in the box.
        @param q_i Orientation of the particle.
        @param charge_i Charge of the particle.
        @param trial Set to false when evaluating the energy of a current configuration. Set to
               true when evaluating a trial move.
        @returns Energy of the external interaction (possibly INFINITY).

        Note: Potentials that may return INFINITY should assume valid old configurations and return
        0 when trial is false. This avoids computing INFINITY - INFINITY -> NaN.
    */
    inline LongReal computeOneExternalEnergy(unsigned int type_i,
                                             const vec3<LongReal>& r_i,
                                             const quat<LongReal>& q_i,
                                             LongReal charge_i,
                                             bool trial = true)
        {
        LongReal energy = 0;
        for (const auto& external : m_external_potentials)
            {
            energy += external->particleEnergy(type_i, r_i, q_i, charge_i, trial);
            }

        return energy;
        }

    /// Get the list of pair potentials.
    std::vector<std::shared_ptr<PairPotential>>& getPairPotentials()
        {
        return m_pair_potentials;
        }

    /// Get the list of external potentials.
    std::vector<std::shared_ptr<ExternalPotential>>& getExternalPotentials()
        {
        return m_external_potentials;
        }

    /// Returns an array (indexed by type) of the AABB tree search radius needed.
    const std::vector<LongReal>& getPairEnergySearchRadius()
        {
        const LongReal max_pair_interaction_r_cut = getMaxPairEnergyRCutNonAdditive();
        const unsigned int n_types = m_pdata->getNTypes();
        m_pair_energy_search_radius.resize(n_types);

        for (unsigned int type = 0; type < n_types; type++)
            {
            m_pair_energy_search_radius[type]
                = max_pair_interaction_r_cut
                  + LongReal(0.5) * getMaxPairInteractionAdditiveRCut(type);
            }

        return m_pair_energy_search_radius;
        }

    protected:
    unsigned int m_translation_move_probability; //!< Fraction of moves that are translation moves.
    unsigned int m_nselect;                      //!< Number of particles to select for trial moves

    GPUVector<Scalar> m_d; //!< Maximum move displacement by type
    GPUVector<Scalar> m_a; //!< Maximum angular displacement by type

    GlobalArray<hpmc_counters_t> m_count_total; //!< Accept/reject total count

    Scalar m_nominal_width;     //!< nominal cell width
    Scalar m_extra_ghost_width; //!< extra ghost width to add
    ClockSource m_clock;        //!< Timer for self-benchmarking

    /// Moves-per-second value last recorded
    double m_mps = 0;

    ExternalField* m_external_base; //! This is a cast of the derived class's m_external that can be
                                    //! used in a more general setting.

    bool m_past_first_run; //!< Flag to test if the first run() has started

    std::shared_ptr<Variant> m_kT; //!< kT variant

    /// Pair potential evaluators.
    std::vector<std::shared_ptr<PairPotential>> m_pair_potentials;

    /// External potential evaluators
    std::vector<std::shared_ptr<ExternalPotential>> m_external_potentials;

    /// Cached pair energy search radius.
    std::vector<LongReal> m_pair_energy_search_radius;

    //! Update the nominal width of the cells
    /*! This method is virtual so that derived classes can set appropriate widths
        (for example, some may want max diameter while others may want a buffer distance).
    */
    virtual void updateCellWidth() { }

    //! Return the requested ghost layer width
    virtual Scalar getGhostLayerWidth(unsigned int type)
        {
        return Scalar(0.0);
        }

#ifdef ENABLE_MPI
    //! Return the requested communication flags for ghost particles
    virtual CommFlags getCommFlags(uint64_t timestep)
        {
        return CommFlags(0);
        }

#endif

    private:
    hpmc_counters_t m_count_run_start;  //!< Count saved at run() start
    hpmc_counters_t m_count_step_start; //!< Count saved at the start of the last step
    };

namespace detail
    {
//! Export the IntegratorHPMC class to python
void export_IntegratorHPMC(pybind11::module& m);

    } // end namespace detail

    } // end namespace hpmc

    } // end namespace hoomd
#endif // _INTEGRATOR_HPMC_H_
