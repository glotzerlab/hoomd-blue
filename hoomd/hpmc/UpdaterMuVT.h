// Copyright (c) 2009-2024 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

#ifndef __UPDATER_MUVT_H__
#define __UPDATER_MUVT_H__

#include "hoomd/HOOMDMPI.h"
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

    //! Set ntrial parameter for configurational bias attempts per depletant
    void setNTrial(unsigned int n_trial)
        {
        m_n_trial = n_trial;
        }

    //! Get the number of configurational bias attempts
    unsigned int getNTrial()
        {
        return m_n_trial;
        }

    //! Get the current counter values
    hpmc_muvt_counters_t getCounters(unsigned int mode = 0);

    protected:
    std::vector<std::shared_ptr<Variant>> m_fugacity; //!< Reservoir concentration per particle-type
    std::shared_ptr<IntegratorHPMCMono<Shape>>
        m_mc;                  //!< The MC Integrator this Updater is associated with
    unsigned int m_npartition; //!< The number of partitions to use for Gibbs ensemble
    bool m_gibbs;              //!< True if we simulate a Gibbs ensemble

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
     * \param lnboltzmann Log of Boltzmann weight of insertion attempt (return value)
     * \returns True if boltzmann weight is non-zero
     */
    virtual bool tryInsertParticle(uint64_t timestep,
                                   unsigned int type,
                                   vec3<Scalar> pos,
                                   quat<Scalar> orientation,
                                   Scalar& lnboltzmann);

    /*! Try removing a particle
        \param timestep Current time step
        \param tag Tag of particle being removed
        \param lnboltzmann Log of Boltzmann weight of removal attempt (return value)
        \returns True if boltzmann weight is non-zero
     */
    virtual bool tryRemoveParticle(uint64_t timestep, unsigned int tag, Scalar& lnboltzmann);

    /*! Rescale box to new dimensions and scale particles
     * \param timestep current timestep
     * \param new_box the old BoxDim
     * \param new_box the new BoxDim
     * \param extra_ndof (return value) extra degrees of freedom added before box resize
     * \param lnboltzmann (return value) exponent of Boltzmann factor (-delta_E)
     * \returns true if no overlaps
     */
    virtual bool boxResizeAndScale(uint64_t timestep,
                                   const BoxDim old_box,
                                   const BoxDim new_box,
                                   unsigned int& extra_ndof,
                                   Scalar& lnboltzmann);

    //! Map particles by type
    virtual void mapTypes();

    //! Get the nth particle of a given type
    /*! \param type the requested type of the particle
     *  \param type_offs offset of the particle in the list of particles per type
     */
    virtual unsigned int getNthTypeTag(unsigned int type, unsigned int type_offs);

    //! Get number of particles of a given type
    unsigned int getNumParticlesType(unsigned int type);

    /*
     *! Depletant related methods
     */

    /*! Insert depletants into such that they overlap with a particle of given tag
     * \param timestep time step
     * \param n_insert Number of depletants to insert
     * \param delta Sphere diameter
     * \param tag Tag of the particle depletants must overlap with
     * \param n_trial Number of insertion trials per depletant
     * \param lnboltzmann Log of Boltzmann factor for insertion (return value)
     * \param need_overlap_shape If true, successful insertions need to overlap with shape at old
     * position \param type_d Depletant type \returns True if Boltzmann factor is non-zero
     */
    bool moveDepletantsIntoOldPosition(uint64_t timestep,
                                       unsigned int n_insert,
                                       Scalar delta,
                                       unsigned int tag,
                                       unsigned int n_trial,
                                       Scalar& lnboltzmann,
                                       bool need_overlap_shape,
                                       unsigned int type_d);

    /*! Insert depletants such that they overlap with a fictitious particle at a specified position
     * \param timestep time step
     * \param n_insert Number of depletants to insert
     * \param delta Sphere diameter
     * \param pos Position of inserted particle
     * \param orientation Orientation of inserted particle
     * \param type Type of inserted particle
     * \param n_trial Number of insertion trials per depletant
     * \param lnboltzmann Log of Boltzmann factor for insertion (return value)
     * \param type_d Depletant type
     * \returns True if Boltzmann factor is non-zero
     */
    bool moveDepletantsIntoNewPosition(uint64_t timestep,
                                       unsigned int n_insert,
                                       Scalar delta,
                                       vec3<Scalar> pos,
                                       quat<Scalar> orientation,
                                       unsigned int type,
                                       unsigned int n_trial,
                                       Scalar& lnboltzmann,
                                       unsigned int type_d);

    /*! Count overlapping depletants due to insertion of a fictitious particle
     * \param timestep time step
     * \param n_insert Number of depletants in circumsphere
     * \param delta Sphere diameter
     * \param pos Position of new particle
     * \param orientation Orientation of new particle
     * \param type Type of new particle (ignored, if ignore==True)
     * \param n_free Depletants that were free in old configuration
     * \param type_d Depletant type
     * \returns Number of overlapping depletants
     */
    unsigned int countDepletantOverlapsInNewPosition(uint64_t timestep,
                                                     unsigned int n_insert,
                                                     Scalar delta,
                                                     vec3<Scalar> pos,
                                                     quat<Scalar> orientation,
                                                     unsigned int type,
                                                     unsigned int& n_free,
                                                     unsigned int type_d);

    /*! Count overlapping depletants in a sphere of diameter delta
     * \param timestep time step
     * \param n_insert Number of depletants in circumsphere
     * \param delta Sphere diameter
     * \param pos Center of sphere
     * \param type_d Depletant type
     * \returns Number of overlapping depletants
     */
    unsigned int countDepletantOverlaps(uint64_t timestep,
                                        unsigned int n_insert,
                                        Scalar delta,
                                        vec3<Scalar> pos,
                                        unsigned int type_d);

    //! Get the random number of depletants
    virtual unsigned int
    getNumDepletants(uint64_t timestep, Scalar V, bool local, unsigned int type_d);

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

//! Get a poisson-distributed number of depletants
template<class Shape>
unsigned int
UpdaterMuVT<Shape>::getNumDepletants(uint64_t timestep, Scalar V, bool local, unsigned int type_d)
    {
    // parameter for Poisson distribution
    Scalar lambda = this->m_mc->getDepletantFugacity(type_d) * V;

    unsigned int n = 0;
    if (lambda > Scalar(0.0))
        {
        hoomd::PoissonDistribution<Scalar> poisson(lambda);

        // RNG for poisson distribution
        hoomd::RandomGenerator rng(hoomd::Seed(hoomd::RNGIdentifier::UpdaterMuVTPoisson,
                                               timestep,
                                               this->m_sysdef->getSeed()),
                                   hoomd::Counter(local ? this->m_exec_conf->getRank() : 0,
                                                  this->m_exec_conf->getPartition()));

        n = poisson(rng);
        }
    return n;
    }

/*! Set new box and scale positions
 */
template<class Shape>
bool UpdaterMuVT<Shape>::boxResizeAndScale(uint64_t timestep,
                                           const BoxDim old_box,
                                           const BoxDim new_box,
                                           unsigned int& extra_ndof,
                                           Scalar& lnboltzmann)
    {
    lnboltzmann = Scalar(0.0);
    unsigned int ndim = this->m_sysdef->getNDimensions();

    unsigned int N_old = m_pdata->getN();

    // energy of old configuration
    lnboltzmann += m_mc->computeTotalPairEnergy(timestep);

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
        lnboltzmann -= m_mc->computeTotalPairEnergy(timestep);
        }

    if (!overlap)
        {
        // check depletants

        // update the aabb tree
        const hoomd::detail::AABBTree& aabb_tree = this->m_mc->buildAABBTree();

        // update the image list
        const std::vector<vec3<Scalar>>& image_list = this->m_mc->updateImageList();

        // access particle data and system box
        ArrayHandle<Scalar4> h_postype(this->m_pdata->getPositions(),
                                       access_location::host,
                                       access_mode::read);
        ArrayHandle<Scalar4> h_orientation(this->m_pdata->getOrientationArray(),
                                           access_location::host,
                                           access_mode::read);

        // access parameters
        auto& params = this->m_mc->getParams();
        ArrayHandle<unsigned int> h_overlaps(this->m_mc->getInteractionMatrix(),
                                             access_location::host,
                                             access_mode::read);

        const Index2D& overlap_idx = this->m_mc->getOverlapIndexer();

        bool overlap = false;

        // get old local box
        BoxDim old_local_box = old_box;
#ifdef ENABLE_MPI
        if (this->m_pdata->getDomainDecomposition())
            {
            old_local_box = this->m_pdata->getDomainDecomposition()->calculateLocalBox(old_box);
            }
#endif

        unsigned int overlap_count = 0;

        // loop over depletant types
        for (unsigned int type_d = 0; type_d < this->m_pdata->getNTypes(); ++type_d)
            {
            if (m_mc->getDepletantFugacity(type_d) == 0.0)
                continue;

            if (m_mc->getDepletantFugacity(type_d) < 0.0)
                throw std::runtime_error("Negative fugacties not supported in update.muvt()\n");

            // draw number from Poisson distribution (using old box)
            unsigned int n = getNumDepletants(timestep, old_local_box.getVolume(), true, type_d);

            unsigned int err_count = 0;

            // draw a random vector in the box
            hoomd::RandomGenerator rng(
                hoomd::Seed(hoomd::RNGIdentifier::UpdaterMuVTDepletants4,
                            timestep,
                            this->m_sysdef->getSeed()),
                hoomd::Counter(this->m_exec_conf->getRank(), this->m_exec_conf->getPartition()));

            uint3 dim = make_uint3(1, 1, 1);
            uint3 grid_pos = make_uint3(0, 0, 0);
#ifdef ENABLE_MPI
            if (this->m_pdata->getDomainDecomposition())
                {
                Index3D didx = this->m_pdata->getDomainDecomposition()->getDomainIndexer();
                dim = make_uint3(didx.getW(), didx.getH(), didx.getD());
                grid_pos = this->m_pdata->getDomainDecomposition()->getGridPos();
                }
#endif

            // for every test depletant
            for (unsigned int k = 0; k < n; ++k)
                {
                Scalar xrand = hoomd::detail::generate_canonical<Scalar>(rng);
                Scalar yrand = hoomd::detail::generate_canonical<Scalar>(rng);
                Scalar zrand = hoomd::detail::generate_canonical<Scalar>(rng);

                Scalar3 f_test = make_scalar3(xrand, yrand, zrand);
                f_test = (f_test + make_scalar3(grid_pos.x, grid_pos.y, grid_pos.z))
                         / make_scalar3(dim.x, dim.y, dim.z);
                vec3<Scalar> pos_test = vec3<Scalar>(new_box.makeCoordinates(f_test));

                Shape shape_test(quat<Scalar>(), params[type_d]);
                if (shape_test.hasOrientation())
                    {
                    // if the depletant is anisotropic, generate orientation
                    shape_test.orientation = generateRandomOrientation(rng, ndim);
                    }

                // check against overlap in old box
                overlap = false;
                bool overlap_old = false;
                hoomd::detail::AABB aabb_test_local = shape_test.getAABB(vec3<Scalar>(0, 0, 0));

                // All image boxes (including the primary)
                const unsigned int n_images = (unsigned int)image_list.size();
                for (unsigned int cur_image = 0; cur_image < n_images; cur_image++)
                    {
                    vec3<Scalar> pos_test_image = pos_test + image_list[cur_image];
                    Scalar3 f = new_box.makeFraction(vec_to_scalar3(pos_test_image));
                    vec3<Scalar> pos_test_image_old = vec3<Scalar>(old_box.makeCoordinates(f));

                    // set up AABB in old coordinates
                    hoomd::detail::AABB aabb = aabb_test_local;
                    aabb.translate(pos_test_image_old);

                    // scale AABB to new coordinates (the AABB tree contains new coordinates)
                    vec3<Scalar> lower, upper;
                    lower = aabb.getLower();
                    f = old_box.makeFraction(vec_to_scalar3(lower));
                    lower = vec3<Scalar>(new_box.makeCoordinates(f));
                    upper = aabb.getUpper();
                    f = old_box.makeFraction(vec_to_scalar3(upper));
                    upper = vec3<Scalar>(new_box.makeCoordinates(f));
                    aabb = hoomd::detail::AABB(lower, upper);

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

                                    Scalar4 postype_j;
                                    Scalar4 orientation_j;

                                    // load the old position and orientation of the j particle
                                    postype_j = h_postype.data[j];
                                    orientation_j = h_orientation.data[j];

                                    // compute the particle position scaled in the old box
                                    f = new_box.makeFraction(
                                        make_scalar3(postype_j.x, postype_j.y, postype_j.z));
                                    vec3<Scalar> pos_j_old(old_box.makeCoordinates(f));

                                    // put particles in coordinate system of particle i
                                    vec3<Scalar> r_ij = pos_j_old - pos_test_image_old;

                                    unsigned int typ_j = __scalar_as_int(postype_j.w);
                                    Shape shape_j(quat<Scalar>(orientation_j), params[typ_j]);

                                    if (h_overlaps.data[overlap_idx(typ_j, type_d)]
                                        && check_circumsphere_overlap(r_ij, shape_test, shape_j)
                                        && test_overlap(r_ij, shape_test, shape_j, err_count))
                                        {
                                        overlap = true;

                                        // depletant is ignored for any overlap in the old
                                        // configuration
                                        overlap_old = true;
                                        break;
                                        }
                                    }
                                }
                            }
                        else
                            {
                            // skip ahead
                            cur_node_idx += aabb_tree.getNodeSkip(cur_node_idx);
                            }
                        if (overlap)
                            break;
                        } // end loop over AABB nodes

                    if (overlap)
                        break;

                    } // end loop over images

                if (!overlap)
                    {
                    // depletant in free volume
                    extra_ndof++;

                    // check for overlap in new configuration

                    // new depletant coordinates
                    vec3<Scalar> pos_test(new_box.makeCoordinates(f_test));

                    for (unsigned int cur_image = 0; cur_image < n_images; cur_image++)
                        {
                        vec3<Scalar> pos_test_image = pos_test + image_list[cur_image];
                        hoomd::detail::AABB aabb = aabb_test_local;
                        aabb.translate(pos_test_image);

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
                                        unsigned int j
                                            = aabb_tree.getNodeParticle(cur_node_idx, cur_p);

                                        Scalar4 postype_j;
                                        Scalar4 orientation_j;

                                        // load the new position and orientation of the j particle
                                        postype_j = h_postype.data[j];
                                        orientation_j = h_orientation.data[j];

                                        // put particles in coordinate system of particle i
                                        vec3<Scalar> r_ij
                                            = vec3<Scalar>(postype_j) - pos_test_image;

                                        unsigned int typ_j = __scalar_as_int(postype_j.w);
                                        Shape shape_j(quat<Scalar>(orientation_j), params[typ_j]);

                                        if (h_overlaps.data[overlap_idx(typ_j, type_d)]
                                            && check_circumsphere_overlap(r_ij, shape_test, shape_j)
                                            && test_overlap(r_ij, shape_test, shape_j, err_count))
                                            {
                                            overlap = true;
                                            break;
                                            }
                                        }
                                    }
                                }
                            else
                                {
                                // skip ahead
                                cur_node_idx += aabb_tree.getNodeSkip(cur_node_idx);
                                }

                            } // end loop over AABB nodes

                        if (overlap)
                            break;
                        } // end loop over images
                    } // end overlap check in new configuration

                if (overlap_old)
                    {
                    overlap = false;
                    continue;
                    }

                if (overlap)
                    break;
                } // end loop over test depletants

            overlap_count = overlap;

#ifdef ENABLE_MPI
            if (this->m_sysdef->isDomainDecomposed())
                {
                MPI_Allreduce(MPI_IN_PLACE,
                              &overlap_count,
                              1,
                              MPI_UNSIGNED,
                              MPI_SUM,
                              this->m_exec_conf->getMPICommunicator());
                MPI_Allreduce(MPI_IN_PLACE,
                              &extra_ndof,
                              1,
                              MPI_UNSIGNED,
                              MPI_SUM,
                              this->m_exec_conf->getMPICommunicator());
                }
#endif

            if (overlap_count)
                break;
            } // end loop over depletant types

        overlap = overlap_count;
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

    hoomd::RandomGenerator rng(
        hoomd::Seed(hoomd::RNGIdentifier::UpdaterMuVT, timestep, this->m_sysdef->getSeed()),
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
        unsigned int p = m_exec_conf->getPartition() % m_npartition;

        // choose a random pair of communicating boxes
        src = hoomd::UniformIntDistribution(m_npartition - 1)(rng);
        dest = hoomd::UniformIntDistribution(m_npartition - 2)(rng);
        if (src <= dest)
            dest++;

        if (p == src)
            {
            m_gibbs_other = (dest + group * m_npartition) * m_exec_conf->getNRanks();
            mod = 0;
            }
        if (p == dest)
            {
            m_gibbs_other = (src + group * m_npartition) * m_exec_conf->getNRanks();
            mod = 1;
            }
        if (p != src && p != dest)
            {
            active = false;
            }

        // order the expanded ensembles
        volume_move = hoomd::detail::generate_canonical<Scalar>(rng) < m_volume_move_probability;

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
        bool insert = m_gibbs ? mod : hoomd::UniformIntDistribution(1)(rng);

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
                    (unsigned int)(m_transfer_types.size() - 1))(rng)];
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
                f.x = hoomd::detail::generate_canonical<Scalar>(rng);
                f.y = hoomd::detail::generate_canonical<Scalar>(rng);
                if (m_sysdef->getNDimensions() == 2)
                    {
                    f.z = Scalar(0.5);
                    }
                else
                    {
                    f.z = hoomd::detail::generate_canonical<Scalar>(rng);
                    }
                vec3<Scalar> pos_test = vec3<Scalar>(m_pdata->getGlobalBox().makeCoordinates(f));

                Shape shape_test(quat<Scalar>(), param);
                if (shape_test.hasOrientation())
                    {
                    // set particle orientation
                    shape_test.orientation = generateRandomOrientation(rng, ndim);
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
                Scalar lnb(0.0);
                unsigned int nonzero
                    = tryInsertParticle(timestep, type, pos_test, shape_test.orientation, lnb);

                if (nonzero)
                    {
                    lnboltzmann += lnb / kT;
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
                    accept = (hoomd::detail::generate_canonical<double>(rng) < exp(lnboltzmann));
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

            // in Gibbs ensemble, we should not use correlated random numbers with box 1
            hoomd::RandomGenerator rng_local(hoomd::Seed(hoomd::RNGIdentifier::UpdaterMuVTBox1,
                                                         timestep,
                                                         this->m_sysdef->getSeed()),
                                             hoomd::Counter(group));

            // choose a random particle type out of those being transferred
            assert(m_transfer_types.size() > 0);
            unsigned int type = m_transfer_types[hoomd::UniformIntDistribution(
                (unsigned int)(m_transfer_types.size() - 1))(rng_local)];

            // choose a random particle of that type
            unsigned int nptl_type = getNumParticlesType(type);

            if (nptl_type)
                {
                // get random tag of given type
                unsigned int type_offset = hoomd::UniformIntDistribution(nptl_type - 1)(rng_local);
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
            Scalar lnb(0.0);
            if (tryRemoveParticle(timestep, tag, lnb))
                {
                lnboltzmann += lnb / kT;
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
                    accept
                        = (hoomd::detail::generate_canonical<double>(rng_local) < exp(lnboltzmann));
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
                Scalar ln_V_new = log(V / V_other)
                                  + (hoomd::detail::generate_canonical<Scalar>(rng) - Scalar(0.5))
                                        * m_max_vol_rescale;
                V_new = (V + V_other) * exp(ln_V_new) / (Scalar(1.0) + exp(ln_V_new));
                V_new_other
                    = (V + V_other) * (Scalar(1.0) - exp(ln_V_new) / (Scalar(1.0) + exp(ln_V_new)));
                }
            else
                {
                Scalar ln_V_new = log(V_other / V)
                                  + (hoomd::detail::generate_canonical<Scalar>(rng) - Scalar(0.5))
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
        Scalar lnb(0.0);
        bool has_overlaps
            = !boxResizeAndScale(timestep, global_box_old, global_box_new, extra_ndof, lnb);
        ndof += extra_ndof;

        unsigned int other_result;
        Scalar other_lnb;

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
                MPI_Recv(&other_lnb,
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
                MPI_Send(&lnb,
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
                             + (lnb + other_lnb) / kT;

                accept = hoomd::detail::generate_canonical<double>(rng) < exp(arg);
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
bool UpdaterMuVT<Shape>::tryRemoveParticle(uint64_t timestep, unsigned int tag, Scalar& lnboltzmann)
    {
    lnboltzmann = Scalar(0.0);

    // guard against trying to modify empty particle data
    bool nonzero = true;
    if (tag == UINT_MAX)
        nonzero = false;

    if (nonzero)
        {
        bool is_local = this->m_pdata->isParticleLocal(tag);

        // do we have to compute a wall contribution?
        auto field = m_mc->getExternalField();
        unsigned int p = m_exec_conf->getPartition() % m_npartition;
        bool has_field = field || (!m_mc->getExternalPotentials().empty());

        if (has_field && (!m_gibbs || p == 0))
            {
            // getPosition() takes into account grid shift, undo that shift as
            // computeOneExternalEnergy expects unshifted inputs.
            Scalar3 p = m_pdata->getPosition(tag) + m_pdata->getOrigin();
            int3 tmp = make_int3(0, 0, 0);
            m_pdata->getGlobalBox().wrap(p, tmp);
            vec3<Scalar> pos(p);

            const BoxDim box = this->m_pdata->getGlobalBox();
            unsigned int type = this->m_pdata->getType(tag);
            quat<Scalar> orientation(m_pdata->getOrientation(tag));
            Scalar diameter = m_pdata->getDiameter(tag);
            Scalar charge = m_pdata->getCharge(tag);
            if (is_local)
                {
                if (field)
                    {
                    lnboltzmann += field->energy(box,
                                                 type,
                                                 pos,
                                                 quat<float>(orientation),
                                                 float(diameter), // diameter i
                                                 float(charge)    // charge i
                    );
                    }
                lnboltzmann
                    += m_mc->computeOneExternalEnergy(type, pos, orientation, charge, false);
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
                        lnboltzmann += m_mc->computeOnePairEnergy(dot(r_ij, r_ij),
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

                                    lnboltzmann += m_mc->computeOnePairEnergy(dot(r_ij, r_ij),
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
                          &lnboltzmann,
                          1,
                          MPI_HOOMD_SCALAR,
                          MPI_SUM,
                          m_exec_conf->getMPICommunicator());
            }
#endif
        }

// Depletants
#ifdef ENABLE_MPI
    auto& params = this->m_mc->getParams();
#endif

    for (unsigned int type_d = 0; type_d < this->m_pdata->getNTypes(); ++type_d)
        {
        if (m_mc->getDepletantFugacity(type_d) == 0.0)
            continue;
        if (m_mc->getDepletantFugacity(type_d) < 0.0)
            throw std::runtime_error("Negative fugacties not supported in update.muvt()\n");

#ifdef ENABLE_MPI
        // Depletant and colloid diameter
        quat<Scalar> o;
        Scalar d_dep;
            {
            Shape tmp(o, params[type_d]);
            d_dep = tmp.getCircumsphereDiameter();
            }

        // number of depletants to insert
        unsigned int n_insert = 0;

        // zero overlapping depletants after removal
        unsigned int n_overlap = 0;

        if (this->m_gibbs)
            {
            unsigned int other = this->m_gibbs_other;

            if (this->m_exec_conf->getRank() == 0)
                {
                MPI_Request req[2];
                MPI_Status status[2];
                MPI_Isend(&n_overlap,
                          1,
                          MPI_UNSIGNED,
                          other,
                          0,
                          this->m_exec_conf->getHOOMDWorldMPICommunicator(),
                          &req[0]);
                MPI_Irecv(&n_insert,
                          1,
                          MPI_UNSIGNED,
                          other,
                          0,
                          this->m_exec_conf->getHOOMDWorldMPICommunicator(),
                          &req[1]);
                MPI_Waitall(2, req, status);
                }
            if (this->m_sysdef->isDomainDecomposed())
                {
                bcast(n_insert, 0, this->m_exec_conf->getMPICommunicator());
                }
            }
#endif

        // only if the particle to be removed actually exists
        if (tag != UINT_MAX)
            {
#ifdef ENABLE_MPI
            // old type
            unsigned int type = this->m_pdata->getType(tag);

            // Depletant and colloid diameter
            Scalar d_colloid_old;
                {
                Shape shape_old(o, params[type]);
                d_colloid_old = shape_old.getCircumsphereDiameter();
                }

            if (this->m_gibbs)
                {
                // try inserting depletants in new configuration (where particle is removed)
                Scalar delta = d_dep + d_colloid_old;
                Scalar lnb(0.0);
                if (moveDepletantsIntoOldPosition(timestep,
                                                  n_insert,
                                                  delta,
                                                  tag,
                                                  m_n_trial,
                                                  lnb,
                                                  true,
                                                  type_d))
                    {
                    lnboltzmann += lnb;
                    }
                else
                    {
                    nonzero = false;
                    }
                }
            else
#endif
                {
                // just accept
                }
            } // end if particle exists
        } // end loop over depletants

    return nonzero;
    }

template<class Shape>
bool UpdaterMuVT<Shape>::tryInsertParticle(uint64_t timestep,
                                           unsigned int type,
                                           vec3<Scalar> pos,
                                           quat<Scalar> orientation,
                                           Scalar& lnboltzmann)
    {
    // do we have to compute a wall contribution?
    auto field = m_mc->getExternalField();
    bool has_field = field || (!m_mc->getExternalPotentials().empty());

    lnboltzmann = Scalar(0.0);

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
            lnboltzmann += m_mc->computeOneExternalEnergy(type, pos, orientation, 0.0, true);

            const BoxDim& box = this->m_pdata->getGlobalBox();
            if (field)
                {
                lnboltzmann -= field->energy(box,
                                             type,
                                             pos,
                                             quat<float>(orientation),
                                             1.0, // diameter i
                                             0.0  // charge i
                );
                }

            lnboltzmann += m_mc->computeOneExternalEnergy(type, pos, orientation, 0.0, true);
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
                    lnboltzmann -= m_mc->computeOnePairEnergy(dot(r_ij, r_ij),
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

                                lnboltzmann -= m_mc->computeOnePairEnergy(dot(r_ij, r_ij),
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
                      &lnboltzmann,
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

    quat<Scalar> o;
    auto& params = this->m_mc->getParams();
    Shape shape(o, params[type]);
    Scalar d_colloid = shape.getCircumsphereDiameter();

    // loop over depletant types
    for (unsigned int type_d = 0; type_d < this->m_pdata->getNTypes(); ++type_d)
        {
        if (m_mc->getDepletantFugacity(type_d) == 0.0)
            continue;

        if (m_mc->getDepletantFugacity(type_d) < 0.0)
            throw std::runtime_error("Negative fugacities not supported in update.muvt()\n");

        // Depletant and colloid diameter
        Scalar d_dep;
            {
            Shape tmp(o, params[type_d]);
            d_dep = tmp.getCircumsphereDiameter();
            }

        // test sphere diameter and volume
        Scalar delta = d_dep + d_colloid;
        Scalar V = Scalar(M_PI / 6.0) * delta * delta * delta;

        unsigned int n_overlap = 0;
#ifdef ENABLE_MPI
        // number of depletants to insert
        unsigned int n_insert = 0;

        if (this->m_gibbs)
            {
            // perform cluster move
            if (nonzero)
                {
                // generate random depletant number
                unsigned int n_dep = getNumDepletants(timestep, V, false, type_d);

                unsigned int tmp = 0;

                // count depletants overlapping with new config (but ignore overlap in old one)
                n_overlap = countDepletantOverlapsInNewPosition(timestep,
                                                                n_dep,
                                                                delta,
                                                                pos,
                                                                orientation,
                                                                type,
                                                                tmp,
                                                                type_d);

                Scalar lnb(0.0);

                // try inserting depletants in old configuration (compute configurational bias
                // weight factor)
                if (moveDepletantsIntoNewPosition(timestep,
                                                  n_overlap,
                                                  delta,
                                                  pos,
                                                  orientation,
                                                  type,
                                                  m_n_trial,
                                                  lnb,
                                                  type_d))
                    {
                    lnboltzmann -= lnb;
                    }
                else
                    {
                    nonzero = false;
                    }
                }

            unsigned int other = this->m_gibbs_other;

            if (this->m_exec_conf->getRank() == 0)
                {
                MPI_Request req[2];
                MPI_Status status[2];
                MPI_Isend(&n_overlap,
                          1,
                          MPI_UNSIGNED,
                          other,
                          0,
                          this->m_exec_conf->getHOOMDWorldMPICommunicator(),
                          &req[0]);
                MPI_Irecv(&n_insert,
                          1,
                          MPI_UNSIGNED,
                          other,
                          0,
                          this->m_exec_conf->getHOOMDWorldMPICommunicator(),
                          &req[1]);
                MPI_Waitall(2, req, status);
                }
            if (this->m_sysdef->isDomainDecomposed())
                {
                bcast(n_insert, 0, this->m_exec_conf->getMPICommunicator());
                }

            // if we have to insert depletants in addition, reject
            if (n_insert)
                {
                nonzero = false;
                }
            }
        else
#endif
            {
            if (nonzero)
                {
                // generate random depletant number
                unsigned int n_dep = getNumDepletants(timestep, V, false, type_d);

                // count depletants overlapping with new config (but ignore overlap in old one)
                unsigned int n_free;
                n_overlap = countDepletantOverlapsInNewPosition(timestep,
                                                                n_dep,
                                                                delta,
                                                                pos,
                                                                orientation,
                                                                type,
                                                                n_free,
                                                                type_d);
                nonzero = !n_overlap;
                }
            }
        } // end loop over depletants

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

template<class Shape>
bool UpdaterMuVT<Shape>::moveDepletantsIntoNewPosition(uint64_t timestep,
                                                       unsigned int n_insert,
                                                       Scalar delta,
                                                       vec3<Scalar> pos,
                                                       quat<Scalar> orientation,
                                                       unsigned int type,
                                                       unsigned int n_trial,
                                                       Scalar& lnboltzmann,
                                                       unsigned int type_d)
    {
    lnboltzmann = Scalar(0.0);
    unsigned int zero = 0;

    unsigned int ndim = this->m_sysdef->getNDimensions();

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

    // initialize another rng
    hoomd::RandomGenerator rng(hoomd::Seed(hoomd::RNGIdentifier::UpdaterMuVTDepletants2,
                                           timestep,
                                           this->m_sysdef->getSeed()),
                               hoomd::Counter(this->m_exec_conf->getPartition()));

    // update the aabb tree
    const hoomd::detail::AABBTree& aabb_tree = this->m_mc->buildAABBTree();

    // update the image list
    const std::vector<vec3<Scalar>>& image_list = this->m_mc->updateImageList();

    if (is_local)
        {
        ArrayHandle<Scalar4> h_postype(this->m_pdata->getPositions(),
                                       access_location::host,
                                       access_mode::read);
        ArrayHandle<Scalar4> h_orientation(this->m_pdata->getOrientationArray(),
                                           access_location::host,
                                           access_mode::read);
        ArrayHandle<unsigned int> h_tag(this->m_pdata->getTags(),
                                        access_location::host,
                                        access_mode::read);
        ArrayHandle<unsigned int> h_overlaps(this->m_mc->getInteractionMatrix(),
                                             access_location::host,
                                             access_mode::read);

        auto& params = this->m_mc->getParams();

        const Index2D& overlap_idx = this->m_mc->getOverlapIndexer();

        // for every test depletant
        for (unsigned int k = 0; k < n_insert; ++k)
            {
            // Number of successfully reinsert depletants

            // we start with one because of super-detailed balance (we already inserted one
            // overlapping depletant in the trial move)
            unsigned int n_success = 1;

            // Number of allowed insertion trials (those which overlap with colloid at old position)
            unsigned int n_overlap_shape = 1;

            for (unsigned int itrial = 0; itrial < n_trial; ++itrial)
                {
                // random normalized vector
                vec3<Scalar> n;
                hoomd::SpherePointGenerator<Scalar>()(rng, n);

                // draw random radial coordinate in test sphere
                Scalar r3 = hoomd::detail::generate_canonical<Scalar>(rng);
                Scalar r = Scalar(0.5) * delta * slow::pow(r3, Scalar(1.0 / 3.0));

                // test depletant position
                vec3<Scalar> pos_test = pos + r * n;

                Shape shape_test(quat<Scalar>(), params[type_d]);
                if (shape_test.hasOrientation())
                    {
                    // if the depletant is anisotropic, generate orientation
                    shape_test.orientation = generateRandomOrientation(rng, ndim);
                    }

                // check against overlap with old configuration
                bool overlap_old = false;

                hoomd::detail::AABB aabb_test_local = shape_test.getAABB(vec3<Scalar>(0, 0, 0));

                unsigned int err_count = 0;
                // All image boxes (including the primary)
                const unsigned int n_images = (unsigned int)(image_list.size());
                for (unsigned int cur_image = 0; cur_image < n_images; cur_image++)
                    {
                    vec3<Scalar> pos_test_image = pos_test + image_list[cur_image];
                    hoomd::detail::AABB aabb = aabb_test_local;
                    aabb.translate(pos_test_image);

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

                                    Scalar4 postype_j;
                                    Scalar4 orientation_j;

                                    // load the old position and orientation of the j particle
                                    postype_j = h_postype.data[j];
                                    orientation_j = h_orientation.data[j];

                                    // put particles in coordinate system of particle i
                                    vec3<Scalar> r_ij = vec3<Scalar>(postype_j) - pos_test_image;

                                    unsigned int typ_j = __scalar_as_int(postype_j.w);
                                    Shape shape_j(quat<Scalar>(orientation_j), params[typ_j]);

                                    if (h_overlaps.data[overlap_idx(type_d, typ_j)]
                                        && check_circumsphere_overlap(r_ij, shape_test, shape_j)
                                        && test_overlap(r_ij, shape_test, shape_j, err_count))
                                        {
                                        overlap_old = true;
                                        break;
                                        }
                                    }
                                }
                            }
                        else
                            {
                            // skip ahead
                            cur_node_idx += aabb_tree.getNodeSkip(cur_node_idx);
                            }
                        if (overlap_old)
                            break;
                        } // end loop over AABB nodes
                    if (overlap_old)
                        break;
                    } // end loop over images

                // checking the (0,0,0) image is sufficient
                Shape shape(orientation, params[type]);
                vec3<Scalar> r_ij = pos - pos_test;
                if (h_overlaps.data[overlap_idx(type, type_d)]
                    && check_circumsphere_overlap(r_ij, shape_test, shape)
                    && test_overlap(r_ij, shape_test, shape, err_count))
                    {
                    if (!overlap_old)
                        {
                        // insertion counts if it overlaps with inserted particle at new position,
                        // but not with other particles
                        n_success++;
                        }
                    n_overlap_shape++;
                    }
                } // end loop over insertion attempts

            if (n_success)
                {
                lnboltzmann += log((Scalar)n_success / (Scalar)n_overlap_shape);
                }
            else
                {
                zero = 1;
                }
            } // end loop over test depletants
        } // is_local

#ifdef ENABLE_MPI
    if (this->m_sysdef->isDomainDecomposed())
        {
        MPI_Allreduce(MPI_IN_PLACE,
                      &lnboltzmann,
                      1,
                      MPI_HOOMD_SCALAR,
                      MPI_SUM,
                      this->m_exec_conf->getMPICommunicator());
        MPI_Allreduce(MPI_IN_PLACE,
                      &zero,
                      1,
                      MPI_UNSIGNED,
                      MPI_SUM,
                      this->m_exec_conf->getMPICommunicator());
        }
#endif

    return !zero;
    }

template<class Shape>
bool UpdaterMuVT<Shape>::moveDepletantsIntoOldPosition(uint64_t timestep,
                                                       unsigned int n_insert,
                                                       Scalar delta,
                                                       unsigned int tag,
                                                       unsigned int n_trial,
                                                       Scalar& lnboltzmann,
                                                       bool need_overlap_shape,
                                                       unsigned int type_d)
    {
    lnboltzmann = Scalar(0.0);
    unsigned int ndim = this->m_sysdef->getNDimensions();

    // getPosition() corrects for grid shift, add it back
    Scalar3 p = this->m_pdata->getPosition(tag) + this->m_pdata->getOrigin();
    int3 tmp = make_int3(0, 0, 0);
    this->m_pdata->getGlobalBox().wrap(p, tmp);
    vec3<Scalar> pos(p);

    bool is_local = this->m_pdata->isParticleLocal(tag);

    // initialize another rng
    hoomd::RandomGenerator rng(hoomd::Seed(hoomd::RNGIdentifier::UpdaterMuVTDepletants3,
                                           timestep,
                                           this->m_sysdef->getSeed()),
                               hoomd::Counter(this->m_exec_conf->getPartition()));

    // update the aabb tree
    const hoomd::detail::AABBTree& aabb_tree = this->m_mc->buildAABBTree();

    // update the image list
    const std::vector<vec3<Scalar>>& image_list = this->m_mc->updateImageList();

    unsigned int zero = 0;

    if (is_local)
        {
        ArrayHandle<Scalar4> h_postype(this->m_pdata->getPositions(),
                                       access_location::host,
                                       access_mode::read);
        ArrayHandle<Scalar4> h_orientation(this->m_pdata->getOrientationArray(),
                                           access_location::host,
                                           access_mode::read);
        ArrayHandle<unsigned int> h_tag(this->m_pdata->getTags(),
                                        access_location::host,
                                        access_mode::read);
        ArrayHandle<unsigned int> h_rtag(this->m_pdata->getRTags(),
                                         access_location::host,
                                         access_mode::read);
        ArrayHandle<unsigned int> h_overlaps(this->m_mc->getInteractionMatrix(),
                                             access_location::host,
                                             access_mode::read);

        auto& params = this->m_mc->getParams();

        const Index2D& overlap_idx = this->m_mc->getOverlapIndexer();

        // for every test depletant
        for (unsigned int k = 0; k < n_insert; ++k)
            {
            // Number of successfully reinsert depletants
            unsigned int n_success = 0;

            // Number of allowed insertion trials (those which overlap with colloid at old position)
            unsigned int n_overlap_shape = 0;

            for (unsigned int itrial = 0; itrial < n_trial; ++itrial)
                {
                // draw a random vector in the excluded volume sphere of the particle to be inserted
                vec3<Scalar> n;
                hoomd::SpherePointGenerator<Scalar>()(rng, n);

                // draw random radial coordinate in test sphere
                Scalar r3 = hoomd::detail::generate_canonical<Scalar>(rng);
                Scalar r = Scalar(0.5) * delta * slow::pow(r3, Scalar(1.0 / 3.0));

                // test depletant position
                vec3<Scalar> pos_test = pos + r * n;

                Shape shape_test(quat<Scalar>(), params[type_d]);
                if (shape_test.hasOrientation())
                    {
                    // if the depletant is anisotropic, generate orientation
                    shape_test.orientation = generateRandomOrientation(rng, ndim);
                    }

                bool overlap_old = false;
                bool overlap = false;

                hoomd::detail::AABB aabb_test_local = shape_test.getAABB(vec3<Scalar>(0, 0, 0));

                unsigned int err_count = 0;
                // All image boxes (including the primary)
                const unsigned int n_images = (unsigned int)(image_list.size());
                for (unsigned int cur_image = 0; cur_image < n_images; cur_image++)
                    {
                    vec3<Scalar> pos_test_image = pos_test + image_list[cur_image];
                    hoomd::detail::AABB aabb = aabb_test_local;
                    aabb.translate(pos_test_image);

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

                                    Scalar4 postype_j;
                                    Scalar4 orientation_j;

                                    // load the old position and orientation of the j particle
                                    postype_j = h_postype.data[j];
                                    orientation_j = h_orientation.data[j];

                                    if (h_tag.data[j] == tag)
                                        {
                                        // do not check against old particle configuration
                                        continue;
                                        }

                                    // put particles in coordinate system of particle i
                                    vec3<Scalar> r_ij = vec3<Scalar>(postype_j) - pos_test_image;

                                    unsigned int type = __scalar_as_int(postype_j.w);
                                    Shape shape_j(quat<Scalar>(orientation_j), params[type]);

                                    if (h_overlaps.data[overlap_idx(type_d, type)]
                                        && check_circumsphere_overlap(r_ij, shape_test, shape_j)
                                        && test_overlap(r_ij, shape_test, shape_j, err_count))
                                        {
                                        overlap_old = true;
                                        break;
                                        }
                                    }
                                }
                            }
                        else
                            {
                            // skip ahead
                            cur_node_idx += aabb_tree.getNodeSkip(cur_node_idx);
                            }

                        if (overlap_old)
                            break;
                        } // end loop over AABB nodes
                    if (overlap_old)
                        break;
                    } // end loop over images

                // resolve the updated particle tag
                unsigned int j = h_rtag.data[tag];
                assert(j < this->m_pdata->getN());

                // load the old position and orientation of the updated particle
                Scalar4 postype_j = h_postype.data[j];
                Scalar4 orientation_j = h_orientation.data[j];

                // see if it overlaps with depletant
                // only need to consider (0,0,0) image
                vec3<Scalar> r_ij = vec3<Scalar>(postype_j) - pos_test;

                unsigned int typ_j = __scalar_as_int(postype_j.w);
                Shape shape(quat<Scalar>(orientation_j), params[typ_j]);

                if (h_overlaps.data[overlap_idx(type_d, typ_j)]
                    && check_circumsphere_overlap(r_ij, shape_test, shape)
                    && test_overlap(r_ij, shape_test, shape, err_count))
                    {
                    overlap = true;
                    n_overlap_shape++;
                    }

                if (!overlap_old && (overlap || !need_overlap_shape))
                    {
                    // success if it overlaps with the particle identified by the tag
                    n_success++;
                    }
                } // end loop over insertion attempts

            if (n_success)
                {
                lnboltzmann += log((Scalar)n_success / (Scalar)n_overlap_shape);
                }
            else
                {
                zero = 1;
                }
            } // end loop over test depletants
        } // end is_local

#ifdef ENABLE_MPI
    if (this->m_sysdef->isDomainDecomposed())
        {
        MPI_Allreduce(MPI_IN_PLACE,
                      &lnboltzmann,
                      1,
                      MPI_HOOMD_SCALAR,
                      MPI_SUM,
                      this->m_exec_conf->getMPICommunicator());
        MPI_Allreduce(MPI_IN_PLACE,
                      &zero,
                      1,
                      MPI_UNSIGNED,
                      MPI_MAX,
                      this->m_exec_conf->getMPICommunicator());
        }
#endif

    return !zero;
    }

template<class Shape>
unsigned int UpdaterMuVT<Shape>::countDepletantOverlapsInNewPosition(uint64_t timestep,
                                                                     unsigned int n_insert,
                                                                     Scalar delta,
                                                                     vec3<Scalar> pos,
                                                                     quat<Scalar> orientation,
                                                                     unsigned int type,
                                                                     unsigned int& n_free,
                                                                     unsigned int type_d)
    {
    // number of depletants successfully inserted
    unsigned int n_overlap = 0;
    unsigned int ndim = this->m_sysdef->getNDimensions();

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

    // initialize another rng
    hoomd::RandomGenerator rng(hoomd::Seed(hoomd::RNGIdentifier::UpdaterMuVTDepletants5,
                                           timestep,
                                           this->m_sysdef->getSeed()),
                               hoomd::Counter(this->m_exec_conf->getPartition()));

    // update the aabb tree
    const hoomd::detail::AABBTree& aabb_tree = this->m_mc->buildAABBTree();

    // update the image list
    const std::vector<vec3<Scalar>>& image_list = this->m_mc->updateImageList();

    n_free = 0;

    if (is_local)
        {
        ArrayHandle<Scalar4> h_postype(this->m_pdata->getPositions(),
                                       access_location::host,
                                       access_mode::read);
        ArrayHandle<Scalar4> h_orientation(this->m_pdata->getOrientationArray(),
                                           access_location::host,
                                           access_mode::read);
        ArrayHandle<unsigned int> h_tag(this->m_pdata->getTags(),
                                        access_location::host,
                                        access_mode::read);

        auto& params = this->m_mc->getParams();

        ArrayHandle<unsigned int> h_overlaps(this->m_mc->getInteractionMatrix(),
                                             access_location::host,
                                             access_mode::read);
        const Index2D& overlap_idx = this->m_mc->getOverlapIndexer();

        // for every test depletant
        for (unsigned int k = 0; k < n_insert; ++k)
            {
            // draw a random vector in the excluded volume sphere of the particle to be inserted
            vec3<Scalar> n;
            hoomd::SpherePointGenerator<Scalar>()(rng, n);

            // draw random radial coordinate in test sphere
            Scalar r3 = hoomd::detail::generate_canonical<Scalar>(rng);
            Scalar r = Scalar(0.5) * delta * slow::pow(r3, Scalar(1.0 / 3.0));

            // test depletant position
            vec3<Scalar> pos_test = pos + r * n;

            Shape shape_test(quat<Scalar>(), params[type_d]);
            if (shape_test.hasOrientation())
                {
                // if the depletant is anisotropic, generate orientation
                shape_test.orientation = generateRandomOrientation(rng, ndim);
                }

            // check against overlap with old configuration
            bool overlap_old = false;

            hoomd::detail::AABB aabb_test_local = shape_test.getAABB(vec3<Scalar>(0, 0, 0));

            unsigned int err_count = 0;
            // All image boxes (including the primary)
            const unsigned int n_images = (unsigned int)image_list.size();
            for (unsigned int cur_image = 0; cur_image < n_images; cur_image++)
                {
                vec3<Scalar> pos_test_image = pos_test + image_list[cur_image];
                hoomd::detail::AABB aabb = aabb_test_local;
                aabb.translate(pos_test_image);

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

                                Scalar4 postype_j;
                                Scalar4 orientation_j;

                                // load the old position and orientation of the j particle
                                postype_j = h_postype.data[j];
                                orientation_j = h_orientation.data[j];

                                // put particles in coordinate system of particle i
                                vec3<Scalar> r_ij = vec3<Scalar>(postype_j) - pos_test_image;

                                unsigned int typ_j = __scalar_as_int(postype_j.w);
                                Shape shape_j(quat<Scalar>(orientation_j), params[typ_j]);

                                if (h_overlaps.data[overlap_idx(type_d, typ_j)]
                                    && check_circumsphere_overlap(r_ij, shape_test, shape_j)
                                    && test_overlap(r_ij, shape_test, shape_j, err_count))
                                    {
                                    overlap_old = true;
                                    break;
                                    }
                                }
                            }
                        }
                    else
                        {
                        // skip ahead
                        cur_node_idx += aabb_tree.getNodeSkip(cur_node_idx);
                        }
                    if (overlap_old)
                        break;
                    } // end loop over AABB nodes
                if (overlap_old)
                    break;
                } // end loop over images

            if (!overlap_old)
                {
                n_free++;
                // see if it overlaps with inserted particle
                for (unsigned int cur_image = 0; cur_image < n_images; cur_image++)
                    {
                    Shape shape(orientation, params[type]);

                    vec3<Scalar> pos_test_image = pos_test + image_list[cur_image];
                    vec3<Scalar> r_ij = pos - pos_test_image;
                    if (h_overlaps.data[overlap_idx(type_d, type)]
                        && check_circumsphere_overlap(r_ij, shape_test, shape)
                        && test_overlap(r_ij, shape_test, shape, err_count))
                        {
                        n_overlap++;
                        }
                    }
                }

            } // end loop over test depletants
        } // is_local

#ifdef ENABLE_MPI
    if (this->m_sysdef->isDomainDecomposed())
        {
        MPI_Allreduce(MPI_IN_PLACE,
                      &n_overlap,
                      1,
                      MPI_UNSIGNED,
                      MPI_SUM,
                      this->m_exec_conf->getMPICommunicator());
        MPI_Allreduce(MPI_IN_PLACE,
                      &n_free,
                      1,
                      MPI_UNSIGNED,
                      MPI_SUM,
                      this->m_exec_conf->getMPICommunicator());
        }
#endif

    return n_overlap;
    }

template<class Shape>
unsigned int UpdaterMuVT<Shape>::countDepletantOverlaps(uint64_t timestep,
                                                        unsigned int n_insert,
                                                        Scalar delta,
                                                        vec3<Scalar> pos,
                                                        unsigned int type_d)
    {
    // number of depletants successfully inserted
    unsigned int n_overlap = 0;
    unsigned int ndim = this->m_sysdef->getNDimensions();

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

    // initialize another rng
    hoomd::RandomGenerator rng(hoomd::Seed(hoomd::RNGIdentifier::UpdaterMuVTDepletants6,
                                           timestep,
                                           this->m_sysdef->getSeed()),
                               hoomd::Counter(this->m_exec_conf->getPartition()));

    // update the aabb tree
    const hoomd::detail::AABBTree& aabb_tree = this->m_mc->buildAABBTree();

    // update the image list
    const std::vector<vec3<Scalar>>& image_list = this->m_mc->updateImageList();

    if (is_local)
        {
        ArrayHandle<Scalar4> h_postype(this->m_pdata->getPositions(),
                                       access_location::host,
                                       access_mode::read);
        ArrayHandle<Scalar4> h_orientation(this->m_pdata->getOrientationArray(),
                                           access_location::host,
                                           access_mode::read);
        ArrayHandle<unsigned int> h_tag(this->m_pdata->getTags(),
                                        access_location::host,
                                        access_mode::read);

        auto& params = this->m_mc->getParams();

        ArrayHandle<unsigned int> h_overlaps(this->m_mc->getInteractionMatrix(),
                                             access_location::host,
                                             access_mode::read);
        const Index2D& overlap_idx = this->m_mc->getOverlapIndexer();

        // for every test depletant
        for (unsigned int k = 0; k < n_insert; ++k)
            {
            // draw a random vector in the excluded volume sphere of the particle to be inserted
            vec3<Scalar> n;
            hoomd::SpherePointGenerator<Scalar>()(rng, n);

            // draw random radial coordinate in test sphere
            Scalar r3 = hoomd::detail::generate_canonical<Scalar>(rng);
            Scalar r = Scalar(0.5) * delta * slow::pow(r3, Scalar(1.0 / 3.0));

            // test depletant position
            vec3<Scalar> pos_test = pos + r * n;

            Shape shape_test(quat<Scalar>(), params[type_d]);
            if (shape_test.hasOrientation())
                {
                // if the depletant is anisotropic, generate orientation
                shape_test.orientation = generateRandomOrientation(rng, ndim);
                }

            // check against overlap with present configuration
            bool overlap = false;

            hoomd::detail::AABB aabb_test_local = shape_test.getAABB(vec3<Scalar>(0, 0, 0));

            unsigned int err_count = 0;
            // All image boxes (including the primary)
            const unsigned int n_images = image_list.size();
            for (unsigned int cur_image = 0; cur_image < n_images; cur_image++)
                {
                vec3<Scalar> pos_test_image = pos_test + image_list[cur_image];
                hoomd::detail::AABB aabb = aabb_test_local;
                aabb.translate(pos_test_image);

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

                                Scalar4 postype_j;
                                Scalar4 orientation_j;

                                // load the old position and orientation of the j particle
                                postype_j = h_postype.data[j];
                                orientation_j = h_orientation.data[j];

                                // put particles in coordinate system of particle i
                                vec3<Scalar> r_ij = vec3<Scalar>(postype_j) - pos_test_image;

                                unsigned int typ_j = __scalar_as_int(postype_j.w);
                                Shape shape_j(quat<Scalar>(orientation_j), params[typ_j]);

                                if (h_overlaps.data[overlap_idx(typ_j, type_d)]
                                    && check_circumsphere_overlap(r_ij, shape_test, shape_j)
                                    && test_overlap(r_ij, shape_test, shape_j, err_count))
                                    {
                                    overlap = true;
                                    break;
                                    }
                                }
                            }
                        }
                    else
                        {
                        // skip ahead
                        cur_node_idx += aabb_tree.getNodeSkip(cur_node_idx);
                        }
                    if (overlap)
                        break;
                    } // end loop over AABB nodes
                if (overlap)
                    break;
                } // end loop over images

            if (overlap)
                {
                n_overlap++;
                }
            } // end loop over test depletants
        } // is_local

#ifdef ENABLE_MPI
    if (this->m_sysdef->isDomainDecomposed())
        {
        MPI_Allreduce(MPI_IN_PLACE,
                      &n_overlap,
                      1,
                      MPI_UNSIGNED,
                      MPI_SUM,
                      this->m_exec_conf->getMPICommunicator());
        }
#endif

    return n_overlap;
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
        .def_property("ntrial", &UpdaterMuVT<Shape>::getNTrial, &UpdaterMuVT<Shape>::setNTrial)
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
