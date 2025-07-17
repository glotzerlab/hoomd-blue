// Copyright (c) 2009-2025 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

#ifndef __UPDATER_MUVT_H__
#define __UPDATER_MUVT_H__

#include "hoomd/Updater.h"
#include "hoomd/Variant.h"
#include "hoomd/VectorMath.h"

#include "IntegratorHPMCMono.h"
#include "Moves.h"
#include "hoomd/RandomNumbers.h"

#ifndef __HIPCC__
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#endif

namespace hoomd
    {
namespace hpmc
    {
/*!
 * This class implements an Updater for simulations in the grand-canonical ensemble (mu-V-T).
 *
 * Gibbs ensemble integration between two MPI partitions is also supported.
 */
template<class Shape> class UpdaterMuVT : public Updater
    {
    public:
    //! Constructor
    UpdaterMuVT(std::shared_ptr<SystemDefinition> sysdef,
                std::shared_ptr<Trigger> trigger,
                std::shared_ptr<IntegratorHPMCMono<Shape>> mc,
                unsigned int npartition);
    virtual ~UpdaterMuVT();

    //! The entry method for this updater
    /*! \param timestep Current simulation step
     */
    virtual void update(uint64_t timestep);

    //! Set the fugacity of a particle type
    /*! \param type The type id for which to set the fugacity
     * \param fugacity The value of the fugacity (variant)
     */
    void setFugacity(const std::string& typ, std::shared_ptr<Variant> fugacity)
        {
        unsigned int id = this->m_pdata->getTypeByName(typ);
        m_fugacity[id] = fugacity;
        }

    //! Get the fugacity of a particle type
    /*! \param type The type id for which to get the fugacity
     */
    std::shared_ptr<Variant> getFugacity(const std::string& typ)
        {
        unsigned int id = this->m_pdata->getTypeByName(typ);
        return m_fugacity[id];
        }

    //! Set maximum factor for volume rescaling (Gibbs ensemble only)
    void setMaxVolumeRescale(Scalar fac)
        {
        m_max_vol_rescale = fac;
        }

    //! Get maximum factor for volume rescaling (Gibbs ensemble only)
    Scalar getMaxVolumeRescale()
        {
        return m_max_vol_rescale;
        }

    //! In the Gibbs ensemble, set fraction of moves that are volume moves (remainder are
    //! exchange/transfer moves)
    void setVolumeMoveProbability(Scalar volume_move_probability)
        {
        if (volume_move_probability < Scalar(0.0) || volume_move_probability > Scalar(1.0))
            {
            throw std::runtime_error("Move ratio has to be between 0 and 1.\n");
            }
        m_volume_move_probability = volume_move_probability;
        }

    //! Get the volume move probability
    Scalar getVolumeMoveProbability()
        {
        return m_volume_move_probability;
        }

    //! List of types that are inserted/removed/transferred
    void setTransferTypes(const std::vector<std::string>& transfer_types)
        {
        assert(transfer_types.size() <= m_pdata->getNTypes());
        if (transfer_types.size() == 0)
            {
            throw std::runtime_error("Must transfer at least one type.\n");
            }
        m_transfer_types.clear();
        for (auto t : transfer_types)
            {
            unsigned int id = this->m_pdata->getTypeByName(t);
            m_transfer_types.push_back(id);
            }
        }

    //! Get the list of types transferred
    std::vector<std::string> getTransferTypes()
        {
        std::vector<std::string> transfer_types;
        for (auto id : m_transfer_types)
            {
            transfer_types.push_back(this->m_pdata->getNameByType(id));
            }
        return transfer_types;
        }

    //! Get the number of particles per type
    std::map<std::string, unsigned int> getN()
        {
        std::map<std::string, unsigned int> m;

        for (unsigned int i = 0; i < this->m_pdata->getNTypes(); ++i)
            {
            m[this->m_pdata->getNameByType(i)] = getNumParticlesType(i);
            }
        return m;
        }

    //! Reset statistics counters
    void resetStats()
        {
        m_count_run_start = m_count_total;
        }

    //! Get the current counter values
    hpmc_muvt_counters_t getCounters(unsigned int mode = 0);

    protected:
    std::vector<std::shared_ptr<Variant>> m_fugacity; //!< Reservoir concentration per particle-type
    std::shared_ptr<IntegratorHPMCMono<Shape>>
        m_mc;                  //!< The MC Integrator this Updater is associated with
    unsigned int m_npartition; //!< The number of partitions to use for Gibbs ensemble
    bool m_gibbs;              //!< True if we simulate a Gibbs ensemble
    uint16_t m_move_type_seed; //!< Random number seed to use for move types.

    GPUVector<Scalar4> m_postype_backup; //!< Backup of postype array

    Scalar m_max_vol_rescale;         //!< Maximum volume ratio rescaling factor
    Scalar m_volume_move_probability; //!< Ratio between exchange/transfer and volume moves

    unsigned int m_gibbs_other; //!< The root-rank of the other partition

    hpmc_muvt_counters_t m_count_total;      //!< Accept/reject total count
    hpmc_muvt_counters_t m_count_run_start;  //!< Count saved at run() start
    hpmc_muvt_counters_t m_count_step_start; //!< Count saved at the start of the last step

    std::vector<std::vector<unsigned int>> m_type_map; //!< Local list of particle tags per type
    std::vector<unsigned int>
        m_transfer_types; //!< List of types being insert/removed/transferred between boxes

    GPUVector<Scalar4> m_pos_backup;         //!< Backup of particle positions for volume move
    GPUVector<Scalar4> m_orientation_backup; //!< Backup of particle orientations for volume move
    GPUVector<Scalar> m_charge_backup;       //!< Backup of particle charges for volume move
    GPUVector<Scalar> m_diameter_backup;     //!< Backup of particle diameters for volume move

    unsigned int m_n_trial;

    /*! Check for overlaps of a fictitious particle
     * \param timestep Current time step
     * \param type Type of particle to test
     * \param pos Position of fictitious particle
     * \param orientation Orientation of particle
     * \param delta_u Change in energy from insertion attempt (return value)
     * \returns True if boltzmann weight is non-zero
     */
    virtual bool tryInsertParticle(uint64_t timestep,
                                   unsigned int type,
                                   vec3<Scalar> pos,
                                   quat<Scalar> orientation,
                                   Scalar& delta_u);

    /*! Try removing a particle
        \param timestep Current time step
        \param tag Tag of particle being removed
        \param delta_u Change in energy from removal attempt (return value)
        \returns True if boltzmann weight is non-zero
     */
    virtual bool tryRemoveParticle(uint64_t timestep, unsigned int tag, Scalar& lnboltzmann);

    /*! Rescale box to new dimensions and scale particles
     * \param timestep current timestep
     * \param new_box the old BoxDim
     * \param new_box the new BoxDim
     * \param extra_ndof (return value) extra degrees of freedom added before box resize
     * \param delta_u (return value) change in energy from the box resize
     * \returns true if no overlaps
     */
    virtual bool boxResizeAndScale(uint64_t timestep,
                                   const BoxDim old_box,
                                   const BoxDim new_box,
                                   unsigned int& extra_ndof,
                                   Scalar& delta_u);

    //! Map particles by type
    virtual void mapTypes();

    //! Get the nth particle of a given type
    /*! \param type the requested type of the particle
     *  \param type_offs offset of the particle in the list of particles per type
     */
    virtual unsigned int getNthTypeTag(unsigned int type, unsigned int type_offs);

    //! Get number of particles of a given type
    unsigned int getNumParticlesType(unsigned int type);

    private:
    //! Handle MaxParticleNumberChange signal
    /*! Resize the m_pos_backup array
     */
    void slotMaxNChange()
        {
        unsigned int MaxN = m_pdata->getMaxN();
        m_pos_backup.resize(MaxN);
        }
    };

/*! Constructor
    \param sysdef The system definition
    \param mc The HPMC integrator
    \param seed RNG seed
    \param npartition How many partitions to use in parallel for Gibbs ensemble (n=1 == grand
   canonical)
 */
template<class Shape>
UpdaterMuVT<Shape>::UpdaterMuVT(std::shared_ptr<SystemDefinition> sysdef,
                                std::shared_ptr<Trigger> trigger,
                                std::shared_ptr<IntegratorHPMCMono<Shape>> mc,
                                unsigned int npartition)
    : Updater(sysdef, trigger), m_mc(mc), m_npartition(npartition), m_gibbs(false),
      m_max_vol_rescale(0.1), m_volume_move_probability(0.5), m_gibbs_other(0), m_n_trial(1)
    {
    m_fugacity.resize(m_pdata->getNTypes(), std::shared_ptr<Variant>(new VariantConstant(0.0)));
    m_type_map.resize(m_pdata->getNTypes());

    m_pdata->getParticleSortSignal()
        .template connect<UpdaterMuVT<Shape>, &UpdaterMuVT<Shape>::mapTypes>(this);

    if (npartition > 1)
        {
        m_gibbs = true;
        }

    m_move_type_seed = m_sysdef->getSeed();

#ifdef ENABLE_MPI
    if (m_gibbs)
        {
        if (m_exec_conf->getNPartitions() % npartition)
            {
            throw std::runtime_error("Total number of partitions not a multiple of the number "
                                     "of Gibbs ensemble partitions.");
            }

        GPUVector<Scalar4> postype_backup(m_exec_conf);
        m_postype_backup.swap(postype_backup);

        m_exec_conf->msg->notice(5) << "Constructing UpdaterMuVT: Gibbs ensemble with "
                                    << m_npartition << " partitions" << std::endl;

        // Ensure that the user sets unique seeds on all partitions so that local trial moves
        // are decorrelated.
        unsigned int my_partition = this->m_exec_conf->getPartition();
        unsigned int my_group = this->m_exec_conf->getPartition() / npartition;
        uint16_t my_seed = this->m_sysdef->getSeed();

        for (unsigned int check_partition = 0;
             check_partition < this->m_exec_conf->getNPartitions();
             check_partition++)
            {
            unsigned int check_group = check_partition / npartition;
            uint16_t check_seed = my_seed;
            MPI_Bcast(&check_seed,
                      1,
                      MPI_UINT16_T,
                      check_partition * m_exec_conf->getMPIConfig()->getNRanks(),
                      m_exec_conf->getHOOMDWorldMPICommunicator());

            if (my_group == check_group && my_partition != check_partition && my_seed == check_seed)
                {
                std::ostringstream s;
                s << "Each partition within a group must set a unique seed. ";
                s << "Partition " << check_partition << "'s " << "seed (" << check_seed << ") ";
                s << "matches partition " << my_partition << "'s";
                throw std::runtime_error(s.str());
                }
            }

        // synchronize move types across all ranks within each group
        for (unsigned int group = 0; group < this->m_exec_conf->getNPartitions() / npartition;
             group++)
            {
            uint16_t tmp = m_move_type_seed;
            MPI_Bcast(&tmp,
                      1,
                      MPI_UINT16_T,
                      group * npartition * this->m_exec_conf->getNRanks(),
                      m_exec_conf->getHOOMDWorldMPICommunicator());

            unsigned int my_group = this->m_exec_conf->getPartition() / npartition;
            if (my_group == group)
                {
                m_move_type_seed = tmp;
                }
            }
        }
    else
#endif
        {
        m_exec_conf->msg->notice(5) << "Constructing UpdaterMuVT" << std::endl;
        }

#ifndef ENABLE_MPI
    if (m_gibbs)
        {
        throw std::runtime_error("Gibbs ensemble integration only supported with MPI.");
        }
#endif

    // initialize list of tags per type
    mapTypes();

    // Connect to the MaxParticleNumberChange signal
    m_pdata->getMaxParticleNumberChangeSignal()
        .template connect<UpdaterMuVT<Shape>, &UpdaterMuVT<Shape>::slotMaxNChange>(this);
    }

//! Destructor
template<class Shape> UpdaterMuVT<Shape>::~UpdaterMuVT()
    {
    m_pdata->getParticleSortSignal()
        .template disconnect<UpdaterMuVT<Shape>, &UpdaterMuVT<Shape>::mapTypes>(this);
    m_pdata->getMaxParticleNumberChangeSignal()
        .template disconnect<UpdaterMuVT<Shape>, &UpdaterMuVT<Shape>::slotMaxNChange>(this);
    }

template<class Shape> void UpdaterMuVT<Shape>::mapTypes()
    {
    ArrayHandle<Scalar4> h_postype(m_pdata->getPositions(),
                                   access_location::host,
                                   access_mode::read);
    ArrayHandle<unsigned int> h_tag(m_pdata->getTags(), access_location::host, access_mode::read);

    assert(m_pdata->getNTypes() == m_type_map.size());
    for (unsigned int itype = 0; itype < m_pdata->getNTypes(); ++itype)
        {
        m_type_map[itype].clear();
        }

    unsigned int nptl = m_pdata->getN();
    for (unsigned int idx = 0; idx < nptl; idx++)
        {
        unsigned int typei = __scalar_as_int(h_postype.data[idx].w);
        unsigned int tag = h_tag.data[idx];

        // store tag in per-type list
        assert(m_type_map.size() > typei);
        m_type_map[typei].push_back(tag);
        }
    }

template<class Shape>
unsigned int UpdaterMuVT<Shape>::getNthTypeTag(unsigned int type, unsigned int type_offs)
    {
    unsigned int tag = UINT_MAX;

    assert(m_type_map.size() > type);
#ifdef ENABLE_MPI
    if (m_pdata->getDomainDecomposition())
        {
        // get number of particles of given type
        unsigned int nptl = (unsigned int)(m_type_map[type].size());

        // have to initialize correctly for prefix sum
        unsigned int begin_offs = 0;
        unsigned int end_offs = 0;

        // exclusive scan
        MPI_Exscan(&nptl, &begin_offs, 1, MPI_UNSIGNED, MPI_SUM, m_exec_conf->getMPICommunicator());

        // inclusive scan
        MPI_Scan(&nptl, &end_offs, 1, MPI_UNSIGNED, MPI_SUM, m_exec_conf->getMPICommunicator());

        bool is_local = type_offs >= begin_offs && type_offs < end_offs;

        unsigned int rank = is_local ? m_exec_conf->getRank() : 0;

        MPI_Allreduce(MPI_IN_PLACE,
                      &rank,
                      1,
                      MPI_UNSIGNED,
                      MPI_SUM,
                      m_exec_conf->getMPICommunicator());
        assert(rank == m_exec_conf->getRank() || !is_local);

        // broadcast the chosen particle tag
        if (is_local)
            {
            assert(type_offs - begin_offs < m_type_map[type].size());
            tag = m_type_map[type][type_offs - begin_offs];
            }

        MPI_Bcast(&tag, 1, MPI_UNSIGNED, rank, m_exec_conf->getMPICommunicator());
        }
    else
#endif
        {
        assert(type_offs < m_type_map[type].size());
        tag = m_type_map[type][type_offs];
        }

    assert(tag <= m_pdata->getMaximumTag());
    return tag;
    }

template<class Shape> unsigned int UpdaterMuVT<Shape>::getNumParticlesType(unsigned int type)
    {
    assert(type < m_type_map.size());
    unsigned int nptl_type = (unsigned int)m_type_map[type].size();

#ifdef ENABLE_MPI
    if (m_pdata->getDomainDecomposition())
        {
        MPI_Allreduce(MPI_IN_PLACE,
                      &nptl_type,
                      1,
                      MPI_UNSIGNED,
                      MPI_SUM,
                      m_exec_conf->getMPICommunicator());
        }
#endif
    return nptl_type;
    }

/*! Set new box and scale positions
 */
template<class Shape>
bool UpdaterMuVT<Shape>::boxResizeAndScale(uint64_t timestep,
                                           const BoxDim old_box,
                                           const BoxDim new_box,
                                           unsigned int& extra_ndof,
                                           Scalar& delta_u)
    {
    delta_u = Scalar(0.0);

    unsigned int N_old = m_pdata->getN();

    // energy of old configuration
    delta_u += m_mc->computeTotalPairEnergy(timestep);

        {
        // Get particle positions
        ArrayHandle<Scalar4> h_pos(m_pdata->getPositions(),
                                   access_location::host,
                                   access_mode::readwrite);

        // move the particles to be inside the new box
        for (unsigned int i = 0; i < N_old; i++)
            {
            Scalar3 old_pos = make_scalar3(h_pos.data[i].x, h_pos.data[i].y, h_pos.data[i].z);

            // obtain scaled coordinates in the old global box
            Scalar3 f = old_box.makeFraction(old_pos);

            // scale particles
            Scalar3 scaled_pos = new_box.makeCoordinates(f);
            h_pos.data[i].x = scaled_pos.x;
            h_pos.data[i].y = scaled_pos.y;
            h_pos.data[i].z = scaled_pos.z;
            }
        } // end lexical scope

    m_pdata->setGlobalBox(new_box);

    // we have changed particle neighbors, communicate those changes
    m_mc->communicate(false);

    // check for overlaps
    bool overlap = m_mc->countOverlaps(true);

    if (!overlap)
        {
        // energy of new configuration
        delta_u -= m_mc->computeTotalPairEnergy(timestep);
        }

    return !overlap;
    }

template<class Shape> void UpdaterMuVT<Shape>::update(uint64_t timestep)
    {
    Updater::update(timestep);
    m_count_step_start = m_count_total;
    unsigned int ndim = this->m_sysdef->getNDimensions();

    const Scalar kT = (*m_mc->getKT())(timestep);

    m_exec_conf->msg->notice(10) << "UpdaterMuVT update: " << timestep << std::endl;

    // initialize random number generator
    unsigned int group = (m_exec_conf->getPartition() / m_npartition);
    unsigned int partition = (m_exec_conf->getPartition() % m_npartition);

    // Make a RNG that is seeded the same across the whole group
    hoomd::RandomGenerator rng_group(
        hoomd::Seed(hoomd::RNGIdentifier::UpdaterMuVTGroup, timestep, m_move_type_seed),
        hoomd::Counter(group));

    bool active = true;
    unsigned int mod = 0;

    bool volume_move = false;

    bool is_root = (m_exec_conf->getRank() == 0);

#ifdef ENABLE_MPI
    unsigned int src = 0;
    unsigned int dest = 1;

    // the other MPI partition
    if (m_gibbs)
        {
        // choose a random pair of communicating boxes
        src = hoomd::UniformIntDistribution(m_npartition - 1)(rng_group);
        dest = hoomd::UniformIntDistribution(m_npartition - 2)(rng_group);
        if (src <= dest)
            dest++;

        if (partition == src)
            {
            m_gibbs_other = (dest + group * m_npartition) * m_exec_conf->getNRanks();
            mod = 0;
            }
        if (partition == dest)
            {
            m_gibbs_other = (src + group * m_npartition) * m_exec_conf->getNRanks();
            mod = 1;
            }
        if (partition != src && partition != dest)
            {
            active = false;
            }

        // order the expanded ensembles
        Scalar r = hoomd::detail::generate_canonical<Scalar>(rng_group);
        volume_move = r < m_volume_move_probability;

        if (active && m_exec_conf->getRank() == 0)
            {
            unsigned int other_timestep = 0;
            // make sure random seeds are equal
            if (mod == 0)
                {
                MPI_Status stat;
                MPI_Recv(&other_timestep,
                         1,
                         MPI_UNSIGNED,
                         m_gibbs_other,
                         0,
                         m_exec_conf->getHOOMDWorldMPICommunicator(),
                         &stat);
                MPI_Send(&timestep,
                         1,
                         MPI_UNSIGNED,
                         m_gibbs_other,
                         0,
                         m_exec_conf->getHOOMDWorldMPICommunicator());
                }
            else
                {
                MPI_Status stat;
                MPI_Send(&timestep,
                         1,
                         MPI_UNSIGNED,
                         m_gibbs_other,
                         0,
                         m_exec_conf->getHOOMDWorldMPICommunicator());
                MPI_Recv(&other_timestep,
                         1,
                         MPI_UNSIGNED,
                         m_gibbs_other,
                         0,
                         m_exec_conf->getHOOMDWorldMPICommunicator(),
                         &stat);
                }

            if (other_timestep != timestep)
                {
                m_exec_conf->msg->error()
                    << "UpdaterMuVT: Boxes are at different time steps " << timestep
                    << " != " << other_timestep << ". Aborting." << std::endl;
                throw std::runtime_error("Error in update.muvt.");
                }
            }
        }
#endif

    if (active && !volume_move)
        {
#ifdef ENABLE_MPI
        if (m_gibbs)
            {
            m_exec_conf->msg->notice(10)
                << "UpdaterMuVT: Gibbs ensemble transfer " << src << "->" << dest << " " << timestep
                << " (Gibbs ensemble partition " << m_exec_conf->getPartition() % m_npartition
                << ")" << std::endl;
            }
#endif

        // whether we insert or remove a particle
        bool insert = m_gibbs ? mod : hoomd::UniformIntDistribution(1)(rng_group);

        // Use a partition specific RNG stream on each partition in Gibbs ensembles.
        hoomd::RandomGenerator rng_insert_remove(
            hoomd::Seed(hoomd::RNGIdentifier::UpdaterMuVTInsertRemove,
                        timestep,
                        this->m_sysdef->getSeed()),
            hoomd::Counter(group, partition));

        if (insert)
            {
            // Try inserting a particle
            unsigned int type = 0;
            std::string type_name;
            Scalar lnboltzmann(0.0);

            unsigned int nptl_type = 0;

            Scalar V = m_pdata->getGlobalBox().getVolume();

            assert(m_transfer_types.size() > 0);

            if (!m_gibbs)
                {
                // choose a random particle type out of those being inserted or removed
                type = m_transfer_types[hoomd::UniformIntDistribution(
                    (unsigned int)(m_transfer_types.size() - 1))(rng_insert_remove)];
                }
            else
                {
                if (is_root)
                    {
#ifdef ENABLE_MPI
                    MPI_Status stat;

                    // receive type of particle
                    unsigned int n;
                    MPI_Recv(&n,
                             1,
                             MPI_UNSIGNED,
                             m_gibbs_other,
                             0,
                             m_exec_conf->getHOOMDWorldMPICommunicator(),
                             &stat);
                    std::vector<char> s(n);
                    MPI_Recv(s.data(),
                             n,
                             MPI_CHAR,
                             m_gibbs_other,
                             0,
                             m_exec_conf->getHOOMDWorldMPICommunicator(),
                             &stat);
                    type_name = std::string(s.data());

                    // resolve type name
                    type = m_pdata->getTypeByName(type_name);
#endif
                    }

#ifdef ENABLE_MPI
                if (m_sysdef->isDomainDecomposed())
                    {
                    bcast(type, 0, m_exec_conf->getMPICommunicator());
                    }
#endif
                }

            // number of particles of that type
            nptl_type = getNumParticlesType(type);

                {
                const std::vector<typename Shape::param_type,
                                  hoomd::detail::managed_allocator<typename Shape::param_type>>&
                    params
                    = m_mc->getParams();
                const typename Shape::param_type& param = params[type];

                // Propose a random position uniformly in the box
                Scalar3 f;
                f.x = hoomd::detail::generate_canonical<Scalar>(rng_insert_remove);
                f.y = hoomd::detail::generate_canonical<Scalar>(rng_insert_remove);
                if (m_sysdef->getNDimensions() == 2)
                    {
                    f.z = Scalar(0.5);
                    }
                else
                    {
                    f.z = hoomd::detail::generate_canonical<Scalar>(rng_insert_remove);
                    }
                vec3<Scalar> pos_test = vec3<Scalar>(m_pdata->getGlobalBox().makeCoordinates(f));

                Shape shape_test(quat<Scalar>(), param);
                if (shape_test.hasOrientation())
                    {
                    // set particle orientation
                    shape_test.orientation = generateRandomOrientation(rng_insert_remove, ndim);
                    }

                if (m_gibbs)
                    {
                    // acceptance probability
                    lnboltzmann = log((Scalar)V / (Scalar)(nptl_type + 1));
                    }
                else
                    {
                    // get fugacity value
                    Scalar fugacity = (*m_fugacity[type])(timestep);

                    // sanity check
                    if (fugacity <= Scalar(0.0))
                        {
                        m_exec_conf->msg->error()
                            << "Fugacity has to be greater than zero." << std::endl;
                        throw std::runtime_error("Error in UpdaterMuVT");
                        }

                    // acceptance probability
                    lnboltzmann = log(fugacity * V / ((Scalar)(nptl_type + 1) * kT));
                    }

                // check if particle can be inserted without overlaps
                Scalar delta_u(0.0);
                unsigned int nonzero
                    = tryInsertParticle(timestep, type, pos_test, shape_test.orientation, delta_u);

                if (nonzero)
                    {
                    lnboltzmann += delta_u / kT;
                    }

#ifdef ENABLE_MPI
                if (m_gibbs && is_root)
                    {
                    // receive Boltzmann factor for removal from other rank
                    MPI_Status stat;
                    Scalar remove_lnb;
                    unsigned int remove_nonzero;
                    MPI_Recv(&remove_lnb,
                             1,
                             MPI_HOOMD_SCALAR,
                             m_gibbs_other,
                             0,
                             m_exec_conf->getHOOMDWorldMPICommunicator(),
                             &stat);
                    MPI_Recv(&remove_nonzero,
                             1,
                             MPI_UNSIGNED,
                             m_gibbs_other,
                             0,
                             m_exec_conf->getHOOMDWorldMPICommunicator(),
                             &stat);

                    // avoid divide/multiply by infinity
                    if (remove_nonzero)
                        {
                        lnboltzmann += remove_lnb;
                        }
                    else
                        {
                        nonzero = 0;
                        }
                    }

                if (m_sysdef->isDomainDecomposed())
                    {
                    bcast(lnboltzmann, 0, m_exec_conf->getMPICommunicator());
                    bcast(nonzero, 0, m_exec_conf->getMPICommunicator());
                    }
#endif

                // apply acceptance criterion
                bool accept = false;
                if (nonzero)
                    {
                    accept = (hoomd::detail::generate_canonical<double>(rng_insert_remove)
                              < exp(lnboltzmann));
                    }

#ifdef ENABLE_MPI
                if (m_gibbs && is_root)
                    {
                    // send result of acceptance test
                    unsigned result = accept;
                    MPI_Send(&result,
                             1,
                             MPI_UNSIGNED,
                             m_gibbs_other,
                             0,
                             m_exec_conf->getHOOMDWorldMPICommunicator());
                    }
#endif

                if (accept)
                    {
                    // insertion was successful

                    // create a new particle with given type
                    unsigned int tag;

                    tag = m_pdata->addParticle(type);

                    // set the position of the particle

                    // setPosition() takes into account the grid shift, so subtract that one
                    Scalar3 p = vec_to_scalar3(pos_test) - m_pdata->getOrigin();
                    int3 tmp = make_int3(0, 0, 0);
                    m_pdata->getGlobalBox().wrap(p, tmp);
                    m_pdata->setPosition(tag, p);
                    if (shape_test.hasOrientation())
                        {
                        m_pdata->setOrientation(tag, quat_to_scalar4(shape_test.orientation));
                        }
                    m_count_total.insert_accept_count++;
                    }
                else
                    {
                    m_count_total.insert_reject_count++;
                    }
                }
            }
        else
            {
            // try removing a particle
            unsigned int tag = UINT_MAX;

            // choose a random particle type out of those being transferred
            assert(m_transfer_types.size() > 0);
            unsigned int type = m_transfer_types[hoomd::UniformIntDistribution(
                (unsigned int)(m_transfer_types.size() - 1))(rng_insert_remove)];

            // choose a random particle of that type
            unsigned int nptl_type = getNumParticlesType(type);

            if (nptl_type)
                {
                // get random tag of given type
                unsigned int type_offset
                    = hoomd::UniformIntDistribution(nptl_type - 1)(rng_insert_remove);
                tag = getNthTypeTag(type, type_offset);
                }

            Scalar V = m_pdata->getGlobalBox().getVolume();
            Scalar lnboltzmann(0.0);

            if (!m_gibbs)
                {
                // get fugacity value
                Scalar fugacity = (*m_fugacity[type])(timestep);

                // sanity check
                if (fugacity <= Scalar(0.0))
                    {
                    m_exec_conf->msg->error()
                        << "Fugacity has to be greater than zero." << std::endl;
                    throw std::runtime_error("Error in UpdaterMuVT");
                    }

                lnboltzmann -= log(fugacity / kT);
                }
            else
                {
                if (is_root)
                    {
#ifdef ENABLE_MPI
                    // determine type name
                    std::string type_name = m_pdata->getNameByType(type);

                    // send particle type to other rank
                    unsigned int n = (unsigned int)(type_name.size() + 1);
                    MPI_Send(&n,
                             1,
                             MPI_UNSIGNED,
                             m_gibbs_other,
                             0,
                             m_exec_conf->getHOOMDWorldMPICommunicator());
                    std::vector<char> s(n);
                    memcpy(s.data(), type_name.c_str(), n);
                    MPI_Send(s.data(),
                             n,
                             MPI_CHAR,
                             m_gibbs_other,
                             0,
                             m_exec_conf->getHOOMDWorldMPICommunicator());
#endif
                    }
                }

            // acceptance probability
            unsigned int nonzero = 1;
            if (nptl_type)
                {
                lnboltzmann += log((Scalar)nptl_type / V);
                }
            else
                {
                nonzero = 0;
                }

            bool accept = true;

            // get weight for removal
            Scalar delta_u(0.0);
            if (tryRemoveParticle(timestep, tag, delta_u))
                {
                lnboltzmann += delta_u / kT;
                }
            else
                {
                nonzero = 0;
                }

            if (m_gibbs)
                {
                if (is_root)
                    {
#ifdef ENABLE_MPI
                    // send result of removal attempt
                    MPI_Send(&lnboltzmann,
                             1,
                             MPI_HOOMD_SCALAR,
                             m_gibbs_other,
                             0,
                             m_exec_conf->getHOOMDWorldMPICommunicator());
                    MPI_Send(&nonzero,
                             1,
                             MPI_UNSIGNED,
                             m_gibbs_other,
                             0,
                             m_exec_conf->getHOOMDWorldMPICommunicator());

                    // wait for result of insertion on other rank
                    unsigned int result;
                    MPI_Status stat;
                    MPI_Recv(&result,
                             1,
                             MPI_UNSIGNED,
                             m_gibbs_other,
                             0,
                             m_exec_conf->getHOOMDWorldMPICommunicator(),
                             &stat);
                    accept = result;
#endif
                    }
                }
            else
                {
                // apply acceptance criterion
                if (nonzero)
                    {
                    accept = (hoomd::detail::generate_canonical<double>(rng_insert_remove)
                              < exp(lnboltzmann));
                    }
                else
                    {
                    accept = false;
                    }
                }

#ifdef ENABLE_MPI
            if (m_gibbs && m_sysdef->isDomainDecomposed())
                {
                bcast(accept, 0, m_exec_conf->getMPICommunicator());
                }
#endif

            if (accept)
                {
                // remove particle
                m_pdata->removeParticle(tag);
                m_count_total.remove_accept_count++;
                }
            else
                {
                m_count_total.remove_reject_count++;
                }
            } // end remove particle
        }
#ifdef ENABLE_MPI
    if (active && volume_move)
        {
        if (m_gibbs)
            {
            m_exec_conf->msg->notice(10)
                << "UpdaterMuVT: Gibbs ensemble volume move " << timestep << std::endl;
            }

        // perform volume move

        Scalar V_other = 0;
        const BoxDim global_box_old = m_pdata->getGlobalBox();
        Scalar V = global_box_old.getVolume();
        unsigned int nglobal = m_pdata->getNGlobal();

        Scalar V_new, V_new_other;
        if (is_root)
            {
            if (mod == 0)
                {
                // send volume to other rank
                MPI_Send(&V,
                         1,
                         MPI_HOOMD_SCALAR,
                         m_gibbs_other,
                         0,
                         m_exec_conf->getHOOMDWorldMPICommunicator());

                MPI_Status stat;

                // receive other box volume
                MPI_Recv(&V_other,
                         1,
                         MPI_HOOMD_SCALAR,
                         m_gibbs_other,
                         0,
                         m_exec_conf->getHOOMDWorldMPICommunicator(),
                         &stat);
                }
            else
                {
                // receive other box volume
                MPI_Status stat;
                MPI_Recv(&V_other,
                         1,
                         MPI_HOOMD_SCALAR,
                         m_gibbs_other,
                         0,
                         m_exec_conf->getHOOMDWorldMPICommunicator(),
                         &stat);

                // send volume to other rank
                MPI_Send(&V,
                         1,
                         MPI_HOOMD_SCALAR,
                         m_gibbs_other,
                         0,
                         m_exec_conf->getHOOMDWorldMPICommunicator());
                }

            if (mod == 0)
                {
                Scalar ln_V_new
                    = log(V / V_other)
                      + (hoomd::detail::generate_canonical<Scalar>(rng_group) - Scalar(0.5))
                            * m_max_vol_rescale;
                V_new = (V + V_other) * exp(ln_V_new) / (Scalar(1.0) + exp(ln_V_new));
                V_new_other
                    = (V + V_other) * (Scalar(1.0) - exp(ln_V_new) / (Scalar(1.0) + exp(ln_V_new)));
                }
            else
                {
                Scalar ln_V_new
                    = log(V_other / V)
                      + (hoomd::detail::generate_canonical<Scalar>(rng_group) - Scalar(0.5))
                            * m_max_vol_rescale;
                V_new
                    = (V + V_other) * (Scalar(1.0) - exp(ln_V_new) / (Scalar(1.0) + exp(ln_V_new)));
                }
            }

        if (m_sysdef->isDomainDecomposed())
            {
            bcast(V_new, 0, m_exec_conf->getMPICommunicator());
            }

        // apply volume rescale to box
        BoxDim global_box_new = m_pdata->getGlobalBox();
        Scalar3 L_old = global_box_new.getL();
        Scalar3 L_new = global_box_new.getL();
        Scalar power(0.0);
        if (m_sysdef->getNDimensions() == 2)
            {
            power = Scalar(1.0 / 2.0);
            }
        else
            {
            power = Scalar(1.0 / 3.0);
            }
        L_new = L_old * pow(V_new / V, power);
        global_box_new.setL(L_new);

        m_postype_backup.resize(m_pdata->getN());

        // Make a backup copy of position data
        unsigned int N_backup = m_pdata->getN();
            {
            ArrayHandle<Scalar4> h_postype(m_pdata->getPositions(),
                                           access_location::host,
                                           access_mode::read);
            ArrayHandle<Scalar4> h_postype_backup(m_postype_backup,
                                                  access_location::host,
                                                  access_mode::readwrite);
            memcpy(h_postype_backup.data, h_postype.data, sizeof(Scalar4) * N_backup);
            }

        //  number of degrees of freedom the old volume (it doesn't change during a volume move)
        unsigned int ndof = nglobal;

        unsigned int extra_ndof = 0;

        // set new box and rescale coordinates
        Scalar delta_u(0.0);
        bool has_overlaps
            = !boxResizeAndScale(timestep, global_box_old, global_box_new, extra_ndof, delta_u);
        ndof += extra_ndof;

        unsigned int other_result;
        Scalar other_delta_u;

        if (is_root)
            {
            if (mod == 0)
                {
                // receive result from other rank
                MPI_Status stat;
                MPI_Recv(&other_result,
                         1,
                         MPI_UNSIGNED,
                         m_gibbs_other,
                         0,
                         m_exec_conf->getHOOMDWorldMPICommunicator(),
                         &stat);
                MPI_Recv(&other_delta_u,
                         1,
                         MPI_HOOMD_SCALAR,
                         m_gibbs_other,
                         1,
                         m_exec_conf->getHOOMDWorldMPICommunicator(),
                         &stat);
                }
            else
                {
                // send result to other rank
                unsigned int result = has_overlaps;
                MPI_Send(&result,
                         1,
                         MPI_UNSIGNED,
                         m_gibbs_other,
                         0,
                         m_exec_conf->getHOOMDWorldMPICommunicator());
                MPI_Send(&delta_u,
                         1,
                         MPI_HOOMD_SCALAR,
                         m_gibbs_other,
                         1,
                         m_exec_conf->getHOOMDWorldMPICommunicator());
                }
            }

        bool accept = true;

        if (is_root)
            {
            if (mod == 0)
                {
                // receive number of particles from other rank
                unsigned int other_ndof;
                MPI_Status stat;
                MPI_Recv(&other_ndof,
                         1,
                         MPI_UNSIGNED,
                         m_gibbs_other,
                         0,
                         m_exec_conf->getHOOMDWorldMPICommunicator(),
                         &stat);

                // apply criterion on rank zero
                Scalar arg = log(V_new / V) * (Scalar)(ndof + 1)
                             + log(V_new_other / V_other) * (Scalar)(other_ndof + 1)
                             + (delta_u + other_delta_u) / kT;

                accept = hoomd::detail::generate_canonical<double>(rng_group) < exp(arg);
                accept &= !(has_overlaps || other_result);

                // communicate if accepted
                unsigned result = accept;
                MPI_Send(&result,
                         1,
                         MPI_UNSIGNED,
                         m_gibbs_other,
                         0,
                         m_exec_conf->getHOOMDWorldMPICommunicator());
                }
            else
                {
                // send number of particles
                MPI_Send(&ndof,
                         1,
                         MPI_UNSIGNED,
                         m_gibbs_other,
                         0,
                         m_exec_conf->getHOOMDWorldMPICommunicator());

                // wait for result of acceptance criterion
                MPI_Status stat;
                unsigned int result;
                MPI_Recv(&result,
                         1,
                         MPI_UNSIGNED,
                         m_gibbs_other,
                         0,
                         m_exec_conf->getHOOMDWorldMPICommunicator(),
                         &stat);
                accept = result;
                }
            }

        if (m_sysdef->isDomainDecomposed())
            {
            bcast(accept, 0, m_exec_conf->getMPICommunicator());
            }

        if (!accept)
            {
                // volume move rejected

                // restore particle positions and orientations
                {
                ArrayHandle<Scalar4> h_postype(m_pdata->getPositions(),
                                               access_location::host,
                                               access_mode::readwrite);
                ArrayHandle<Scalar4> h_postype_backup(m_postype_backup,
                                                      access_location::host,
                                                      access_mode::read);
                unsigned int N = m_pdata->getN();
                if (N != N_backup)
                    {
                    this->m_exec_conf->msg->error()
                        << "update.muvt"
                        << ": Number of particles mismatch when rejecting volume move" << std::endl;
                    throw std::runtime_error("Error resizing box");
                    // note, this error should never appear (because particles are not migrated
                    // after a box resize), but is left here as a sanity check
                    }
                memcpy(h_postype.data, h_postype_backup.data, sizeof(Scalar4) * N);
                }

            m_pdata->setGlobalBox(global_box_old);

            // increment counter
            m_count_total.volume_reject_count++;
            }
        else
            {
            // volume move accepted
            m_count_total.volume_accept_count++;
            }
        } // end volume move
#endif

#ifdef ENABLE_MPI
    if (m_sysdef->isDomainDecomposed())
        {
        // We have inserted or removed particles or changed box volume, so update ghosts
        m_mc->communicate(false);
        }
#endif
    }

template<class Shape>
bool UpdaterMuVT<Shape>::tryRemoveParticle(uint64_t timestep, unsigned int tag, Scalar& delta_u)
    {
    delta_u = Scalar(0.0);

    // guard against trying to modify empty particle data
    bool nonzero = true;
    if (tag == UINT_MAX)
        nonzero = false;

    if (nonzero)
        {
        bool is_local = this->m_pdata->isParticleLocal(tag);

        // do we have to compute a wall contribution?
        unsigned int p = m_exec_conf->getPartition() % m_npartition;
        bool has_field = !m_mc->getExternalPotentials().empty();

        if (has_field && (!m_gibbs || p == 0))
            {
            // getPosition() takes into account grid shift, undo that shift as
            // computeOneExternalEnergy expects unshifted inputs.
            Scalar3 p = m_pdata->getPosition(tag) + m_pdata->getOrigin();
            int3 tmp = make_int3(0, 0, 0);
            m_pdata->getGlobalBox().wrap(p, tmp);
            vec3<Scalar> pos(p);

            unsigned int type = this->m_pdata->getType(tag);
            quat<Scalar> orientation(m_pdata->getOrientation(tag));
            Scalar charge = m_pdata->getCharge(tag);
            if (is_local)
                {
                delta_u += m_mc->computeOneExternalEnergy(timestep,
                                                          tag,
                                                          type,
                                                          pos,
                                                          orientation,
                                                          charge,
                                                          ExternalPotential::Trial::Old);
                }
            }

        // if not, no overlaps generated
        if (m_mc->hasPairInteractions())
            {
            // type
            unsigned int type = this->m_pdata->getType(tag);

            // read in the current position and orientation
            quat<Scalar> orientation(m_pdata->getOrientation(tag));

            // charge and diameter
            Scalar diameter = m_pdata->getDiameter(tag);
            Scalar charge = m_pdata->getCharge(tag);

            // getPosition() takes into account grid shift, correct for that
            Scalar3 p = m_pdata->getPosition(tag) + m_pdata->getOrigin();
            int3 tmp = make_int3(0, 0, 0);
            m_pdata->getGlobalBox().wrap(p, tmp);
            vec3<Scalar> pos(p);

            if (is_local)
                {
                // update the aabb tree
                const hoomd::detail::AABBTree& aabb_tree = m_mc->buildAABBTree();

                // update the image list
                const std::vector<vec3<Scalar>>& image_list = m_mc->updateImageList();

                // check for overlaps
                ArrayHandle<unsigned int> h_tag(m_pdata->getTags(),
                                                access_location::host,
                                                access_mode::read);
                ArrayHandle<Scalar4> h_postype(m_pdata->getPositions(),
                                               access_location::host,
                                               access_mode::read);
                ArrayHandle<Scalar4> h_orientation(m_pdata->getOrientationArray(),
                                                   access_location::host,
                                                   access_mode::read);
                ArrayHandle<Scalar> h_diameter(m_pdata->getDiameters(),
                                               access_location::host,
                                               access_mode::read);
                ArrayHandle<Scalar> h_charge(m_pdata->getCharges(),
                                             access_location::host,
                                             access_mode::read);

                // Check particle against AABB tree for neighbors
                Scalar r_cut_patch
                    = m_mc->getMaxPairEnergyRCutNonAdditive()
                      + LongReal(0.5) * m_mc->getMaxPairInteractionAdditiveRCut(type);

                Scalar R_query = std::max(0.0, r_cut_patch - m_mc->getMinCoreDiameter() / 2.0);
                hoomd::detail::AABB aabb_local
                    = hoomd::detail::AABB(vec3<Scalar>(0, 0, 0), R_query);

                const unsigned int n_images = (unsigned int)image_list.size();
                for (unsigned int cur_image = 0; cur_image < n_images; cur_image++)
                    {
                    vec3<Scalar> pos_image = pos + image_list[cur_image];

                    if (cur_image != 0)
                        {
                        vec3<Scalar> r_ij = pos - pos_image;
                        // self-energy
                        delta_u += m_mc->computeOnePairEnergy(dot(r_ij, r_ij),
                                                              r_ij,
                                                              type,
                                                              orientation,
                                                              diameter,
                                                              charge,
                                                              type,
                                                              orientation,
                                                              diameter,
                                                              charge);
                        }

                    hoomd::detail::AABB aabb = aabb_local;
                    aabb.translate(pos_image);

                    // stackless search
                    for (unsigned int cur_node_idx = 0; cur_node_idx < aabb_tree.getNumNodes();
                         cur_node_idx++)
                        {
                        if (aabb.overlaps(aabb_tree.getNodeAABB(cur_node_idx)))
                            {
                            if (aabb_tree.isNodeLeaf(cur_node_idx))
                                {
                                for (unsigned int cur_p = 0;
                                     cur_p < aabb_tree.getNodeNumParticles(cur_node_idx);
                                     cur_p++)
                                    {
                                    // read in its position and orientation
                                    unsigned int j = aabb_tree.getNodeParticle(cur_node_idx, cur_p);

                                    Scalar4 postype_j = h_postype.data[j];
                                    quat<LongReal> orientation_j(h_orientation.data[j]);

                                    // put particles in coordinate system of particle i
                                    vec3<Scalar> r_ij = vec3<Scalar>(postype_j) - pos_image;

                                    unsigned int typ_j = __scalar_as_int(postype_j.w);

                                    // we computed the self-interaction above
                                    if (h_tag.data[j] == tag)
                                        continue;

                                    delta_u += m_mc->computeOnePairEnergy(dot(r_ij, r_ij),
                                                                          r_ij,
                                                                          type,
                                                                          orientation,
                                                                          diameter,
                                                                          charge,
                                                                          typ_j,
                                                                          orientation_j,
                                                                          h_diameter.data[j],
                                                                          h_charge.data[j]);
                                    }
                                }
                            }
                        else
                            {
                            // skip ahead
                            cur_node_idx += aabb_tree.getNodeSkip(cur_node_idx);
                            }
                        } // end loop over AABB nodes
                    } // end loop over images
                }
            }

#ifdef ENABLE_MPI
        if (m_sysdef->isDomainDecomposed())
            {
            MPI_Allreduce(MPI_IN_PLACE,
                          &delta_u,
                          1,
                          MPI_HOOMD_SCALAR,
                          MPI_SUM,
                          m_exec_conf->getMPICommunicator());
            }
#endif
        }

    return nonzero;
    }

template<class Shape>
bool UpdaterMuVT<Shape>::tryInsertParticle(uint64_t timestep,
                                           unsigned int type,
                                           vec3<Scalar> pos,
                                           quat<Scalar> orientation,
                                           Scalar& delta_u)
    {
    // do we have to compute a wall contribution?
    bool has_field = !m_mc->getExternalPotentials().empty();

    delta_u = Scalar(0.0);

    unsigned int overlap = 0;

    bool is_local = true;
#ifdef ENABLE_MPI
    if (this->m_pdata->getDomainDecomposition())
        {
        const BoxDim global_box = this->m_pdata->getGlobalBox();
        ArrayHandle<unsigned int> h_cart_ranks(
            this->m_pdata->getDomainDecomposition()->getCartRanks(),
            access_location::host,
            access_mode::read);
        is_local = this->m_exec_conf->getRank()
                   == this->m_pdata->getDomainDecomposition()->placeParticle(global_box,
                                                                             vec_to_scalar3(pos),
                                                                             h_cart_ranks.data);
        }
#endif

    unsigned int nptl_local = m_pdata->getN() + m_pdata->getNGhosts();

    if (is_local)
        {
        // get some data structures from the integrator
        auto& image_list = m_mc->updateImageList();
        const unsigned int n_images = (unsigned int)image_list.size();
        auto& params = m_mc->getParams();

        const Index2D& overlap_idx = m_mc->getOverlapIndexer();

        LongReal r_cut_patch(0.0);

        unsigned int p = m_exec_conf->getPartition() % m_npartition;

        if (has_field && (!m_gibbs || p == 0))
            {
            delta_u -= m_mc->computeOneExternalEnergy(timestep,
                                                      0, // particle has no tag yet
                                                      type,
                                                      pos,
                                                      orientation,
                                                      0.0,
                                                      ExternalPotential::Trial::New);
            }

        if (m_mc->hasPairInteractions())
            {
            r_cut_patch = m_mc->getMaxPairEnergyRCutNonAdditive()
                          + LongReal(0.5) * m_mc->getMaxPairInteractionAdditiveRCut(type);
            }

        unsigned int err_count = 0;

            {
            // check for overlaps
            ArrayHandle<Scalar4> h_postype(m_pdata->getPositions(),
                                           access_location::host,
                                           access_mode::read);
            ArrayHandle<Scalar4> h_orientation(m_pdata->getOrientationArray(),
                                               access_location::host,
                                               access_mode::read);
            ArrayHandle<Scalar> h_diameter(m_pdata->getDiameters(),
                                           access_location::host,
                                           access_mode::read);
            ArrayHandle<Scalar> h_charge(m_pdata->getCharges(),
                                         access_location::host,
                                         access_mode::read);

            ArrayHandle<unsigned int> h_overlaps(m_mc->getInteractionMatrix(),
                                                 access_location::host,
                                                 access_mode::read);

            // read in the current position and orientation
            Shape shape(orientation, params[type]);

            for (unsigned int cur_image = 0; cur_image < n_images; cur_image++)
                {
                vec3<Scalar> pos_image = pos + image_list[cur_image];

                if (cur_image != 0)
                    {
                    // check for self-overlap with all images except the original
                    vec3<Scalar> r_ij = pos - pos_image;
                    if (h_overlaps.data[overlap_idx(type, type)]
                        && check_circumsphere_overlap(r_ij, shape, shape)
                        && test_overlap(r_ij, shape, shape, err_count))
                        {
                        overlap = 1;
                        break;
                        }

                    // self-energy
                    delta_u -= m_mc->computeOnePairEnergy(dot(r_ij, r_ij),
                                                          r_ij,
                                                          type,
                                                          orientation,
                                                          1.0, // diameter i
                                                          0.0, // charge i
                                                          type,
                                                          orientation,
                                                          1.0, // diameter i
                                                          0.0  // charge i
                    );
                    }
                }
            }

        // we cannot rely on a valid AABB tree when there are 0 particles
        if (!overlap && nptl_local > 0)
            {
            // Check particle against AABB tree for neighbors
            const hoomd::detail::AABBTree& aabb_tree = m_mc->buildAABBTree();

            ArrayHandle<Scalar4> h_postype(m_pdata->getPositions(),
                                           access_location::host,
                                           access_mode::read);
            ArrayHandle<Scalar4> h_orientation(m_pdata->getOrientationArray(),
                                               access_location::host,
                                               access_mode::read);
            ArrayHandle<Scalar> h_diameter(m_pdata->getDiameters(),
                                           access_location::host,
                                           access_mode::read);
            ArrayHandle<Scalar> h_charge(m_pdata->getCharges(),
                                         access_location::host,
                                         access_mode::read);
            ArrayHandle<unsigned int> h_overlaps(m_mc->getInteractionMatrix(),
                                                 access_location::host,
                                                 access_mode::read);

            Shape shape(orientation, params[type]);
            LongReal R_query = std::max(shape.getCircumsphereDiameter() / LongReal(2.0),
                                        r_cut_patch - m_mc->getMinCoreDiameter() / LongReal(2.0));
            hoomd::detail::AABB aabb_local = hoomd::detail::AABB(vec3<Scalar>(0, 0, 0), R_query);

            for (unsigned int cur_image = 0; cur_image < n_images; cur_image++)
                {
                vec3<Scalar> pos_image = pos + image_list[cur_image];

                hoomd::detail::AABB aabb = aabb_local;
                aabb.translate(pos_image);

                // stackless search
                for (unsigned int cur_node_idx = 0; cur_node_idx < aabb_tree.getNumNodes();
                     cur_node_idx++)
                    {
                    if (aabb.overlaps(aabb_tree.getNodeAABB(cur_node_idx)))
                        {
                        if (aabb_tree.isNodeLeaf(cur_node_idx))
                            {
                            for (unsigned int cur_p = 0;
                                 cur_p < aabb_tree.getNodeNumParticles(cur_node_idx);
                                 cur_p++)
                                {
                                // read in its position and orientation
                                unsigned int j = aabb_tree.getNodeParticle(cur_node_idx, cur_p);

                                Scalar4 postype_j = h_postype.data[j];
                                quat<LongReal> orientation_j(h_orientation.data[j]);

                                // put particles in coordinate system of particle i
                                vec3<Scalar> r_ij = vec3<Scalar>(postype_j) - pos_image;

                                unsigned int typ_j = __scalar_as_int(postype_j.w);
                                Shape shape_j(orientation_j, params[typ_j]);

                                if (h_overlaps.data[overlap_idx(type, typ_j)]
                                    && check_circumsphere_overlap(r_ij, shape, shape_j)
                                    && test_overlap(r_ij, shape, shape_j, err_count))
                                    {
                                    overlap = 1;
                                    break;
                                    }

                                delta_u -= m_mc->computeOnePairEnergy(dot(r_ij, r_ij),
                                                                      r_ij,
                                                                      type,
                                                                      orientation,
                                                                      1.0, // diameter i
                                                                      0.0, // charge i
                                                                      typ_j,
                                                                      orientation_j,
                                                                      h_diameter.data[j],
                                                                      h_charge.data[j]);
                                }
                            }
                        }
                    else
                        {
                        // skip ahead
                        cur_node_idx += aabb_tree.getNodeSkip(cur_node_idx);
                        }

                    if (overlap)
                        {
                        break;
                        }
                    } // end loop over AABB nodes

                if (overlap)
                    {
                    break;
                    }
                } // end loop over images
            } // end if nptl_local > 0
        } // end if local

#ifdef ENABLE_MPI
    if (m_sysdef->isDomainDecomposed())
        {
        MPI_Allreduce(MPI_IN_PLACE,
                      &delta_u,
                      1,
                      MPI_HOOMD_SCALAR,
                      MPI_SUM,
                      m_exec_conf->getMPICommunicator());
        MPI_Allreduce(MPI_IN_PLACE,
                      &overlap,
                      1,
                      MPI_UNSIGNED,
                      MPI_MAX,
                      m_exec_conf->getMPICommunicator());
        }
#endif

    bool nonzero = !overlap;

    return nonzero;
    }

/*! \param mode 0 -> Absolute count, 1 -> relative to the start of the run, 2 -> relative to the
   last executed step \return The current state of the acceptance counters

    UpdaterMuVT maintains a count of the number of accepted and rejected moves since instantiation.
   getCounters() provides the current value. The parameter *mode* controls whether the returned
   counts are absolute, relative to the start of the run, or relative to the start of the last
   executed step.
*/
template<class Shape> hpmc_muvt_counters_t UpdaterMuVT<Shape>::getCounters(unsigned int mode)
    {
    hpmc_muvt_counters_t result;

    if (mode == 0)
        result = m_count_total;
    else if (mode == 1)
        result = m_count_total - m_count_run_start;
    else
        result = m_count_total - m_count_step_start;

    // don't MPI_AllReduce counters because all ranks count the same thing
    return result;
    }

namespace detail
    {
//! Export the UpdaterMuVT class to python
/*! \param name Name of the class in the exported python module
    \tparam Shape An instantiation of UpdaterMuVT<Shape> will be exported
*/
template<class Shape> void export_UpdaterMuVT(pybind11::module& m, const std::string& name)
    {
    pybind11::class_<UpdaterMuVT<Shape>, Updater, std::shared_ptr<UpdaterMuVT<Shape>>>(m,
                                                                                       name.c_str())
        .def(pybind11::init<std::shared_ptr<SystemDefinition>,
                            std::shared_ptr<Trigger>,
                            std::shared_ptr<IntegratorHPMCMono<Shape>>,
                            unsigned int>())
        .def("setFugacity", &UpdaterMuVT<Shape>::setFugacity)
        .def("getFugacity", &UpdaterMuVT<Shape>::getFugacity)
        .def_property("max_volume_rescale",
                      &UpdaterMuVT<Shape>::getMaxVolumeRescale,
                      &UpdaterMuVT<Shape>::setMaxVolumeRescale)
        .def_property("volume_move_probability",
                      &UpdaterMuVT<Shape>::getVolumeMoveProbability,
                      &UpdaterMuVT<Shape>::setVolumeMoveProbability)
        .def_property("transfer_types",
                      &UpdaterMuVT<Shape>::getTransferTypes,
                      &UpdaterMuVT<Shape>::setTransferTypes)
        .def_property_readonly("N", &UpdaterMuVT<Shape>::getN)
        .def("getCounters", &UpdaterMuVT<Shape>::getCounters);
    }

inline void export_hpmc_muvt_counters(pybind11::module& m)
    {
    pybind11::class_<hpmc_muvt_counters_t>(m, "hpmc_muvt_counters_t")
        .def_property_readonly("insert", &hpmc_muvt_counters_t::getInsertCounts)
        .def_property_readonly("remove", &hpmc_muvt_counters_t::getRemoveCounts)
        .def_property_readonly("exchange", &hpmc_muvt_counters_t::getExchangeCounts)
        .def_property_readonly("volume", &hpmc_muvt_counters_t::getVolumeCounts);
    }

    } // end namespace detail
    } // end namespace hpmc
    } // end namespace hoomd

#endif
