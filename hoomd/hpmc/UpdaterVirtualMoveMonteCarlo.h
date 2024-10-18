// Copyright (c) 2009-2024 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

#ifndef _UPDATER_HPMC_VIRTUAL_CLUSTER_MOVES_
#define _UPDATER_HPMC_VIRTUAL_CLUSTER_MOVES_

#include "hoomd/Updater.h"
#include "hoomd/RandomNumbers.h"
#include "hoomd/RNGIdentifiers.h"
#include "hoomd/Variant.h"

#include <algorithm>
#include <set>

#include "Moves.h"
#include "HPMCCounters.h"
#include "IntegratorHPMCMono.h"
#include <pybind11/pybind11.h>


namespace hoomd {

namespace hpmc
{

struct ClusterData
    {
    void clear()
        {
        m_linkers.clear();
        m_linkers_added.clear();
        m_linker_rotation_centers.clear();
        }

    void update_cluster(unsigned int idx, vec3<Scalar> rotation_center)
        {
        if (m_linkers_added.find(idx) == m_linkers_added.end())
            {
            m_linkers_added.insert(idx);
            m_linkers.push_back(idx);
            m_linker_rotation_centers.push_back(rotation_center);
            }
        }

    size_t current_cluster_size()
        {
        return m_linkers_added.size();
        }

    std::vector<unsigned int> m_linkers;
    std::set<unsigned int> m_linkers_added;
    std::vector<vec3<Scalar>> m_linker_rotation_centers;

    };

struct PotentialLinkData
    {

    void clear()
        {
        m_potential_intracluster_link_idxs.clear();
        m_potential_intracluster_link_beta_deltaU_fwds.clear();
        }

    void update(unsigned int i, unsigned int j, LongReal beta_delta_u_ij_fwd)
        {
        m_potential_intracluster_link_idxs.push_back(std::make_pair(i, j));
        m_potential_intracluster_link_beta_deltaU_fwds.push_back(beta_delta_u_ij_fwd);
        }

    std::vector<std::pair<unsigned int, unsigned int>> m_potential_intracluster_link_idxs;
    std::vector<LongReal> m_potential_intracluster_link_beta_deltaU_fwds;
    };


/*! Virtual move Monte Carlo Algorithm.

    See Whitelam and Geissler 2007.
*/

template< class Shape >
class UpdaterVMMC : public Updater
    {
    public:
        //! Constructor
        /*! \param sysdef System definition
            \param trigger When to run updater
            \param mc HPMC integrator
        */
        UpdaterVMMC(std::shared_ptr<SystemDefinition> sysdef, std::shared_ptr<Trigger> trigger, std::shared_ptr<IntegratorHPMCMono<Shape> > mc, std::shared_ptr<Variant> beta_ficticious);

        //! Destructor
        virtual ~UpdaterVMMC();

        //! Run the updater
        /*! \param timestep timestep at which update is being evaluated
        */
        virtual void update(uint64_t timestep);

        /// Set the number of moves per particle per call to update()
        void setAttemptsPerParticle(unsigned int attempts_per_particle)
            {
            m_attempts_per_particle = attempts_per_particle;
            }

        unsigned int getAttemptsPerParticle()
            {
            return m_attempts_per_particle;
            }

        void setBetaFicticious(std::shared_ptr<Variant> beta_ficticious)
            {
            m_beta_ficticious = beta_ficticious;
            }

        std::shared_ptr<Variant> getBetaFicticious()
            {
            return m_beta_ficticious;
            }

        void setTranslationMoveProbability(LongReal p)
            {
            m_translation_move_probability = p;
            }

        LongReal getTranslationMoveProbability()
            {
            return m_translation_move_probability;
            }

        void setMaximumTrialRotation(LongReal a)
            {
            m_maximum_rotate_size = a;
            }

        LongReal getMaximumTrialRotation()
            {
            return m_maximum_rotate_size;
            }

        void setMaximumTrialTranslation(LongReal a)
            {
            m_maximum_translate_size = a;
            }

        LongReal getMaximumTrialTranslation()
            {
            return m_maximum_translate_size;
            }

        void setMaximumTrialCenterOfRotationShift(LongReal d)
            {
            m_maximum_trial_center_of_rotation_shift = d;
            }

        LongReal getMaximumTrialCenterOfRotationShift()
            {
            return m_maximum_trial_center_of_rotation_shift;
            }

        unsigned int getMaximumAllowedClusterSize()
            {
            return m_maximum_allowed_cluster_size;
            }

        void setMaximumAllowedClusterSize(unsigned int s)
            {
            m_maximum_allowed_cluster_size = s;
            }

        LongReal getClusterSizeDistributionPrefactor()
            {
            return m_cluster_size_distribution_prefactor;
            }

        void setClusterSizeDistributionPrefactor(LongReal f)
            {
            m_cluster_size_distribution_prefactor = f;
            }

        std::string getClusterSizeLimitMode()
            {
            return m_cluster_size_limit_mode;
            }

        void setClusterSizeLimitMode(std::string s)
            {
            m_cluster_size_limit_mode = s;
            }

        /// Reset statistics counters
        virtual void resetStats()
            {
            m_count_run_start = m_count_total;
            }

        bool getAlwaysRebuildTree()
            {
            return m_always_rebuild_tree;
            }

        void setAlwaysRebuildTree()
            {
            m_always_rebuild_tree = true;
            }

        uint16_t getInstance()
            {
            return m_instance;
            }

        void setInstance(uint16_t instance)
            {
            m_instance = instance;
            }

        /*! \param mode 0 -> Absolute count, 1 -> relative to the start of the run, 2 -> relative to the last executed step
            \return The current state of the acceptance counters
        */
        hpmc_virtual_moves_counters_t getCounters(unsigned int mode)
            {
            hpmc_virtual_moves_counters_t result;
            if (mode == 0)
                result = m_count_total;
            else if (mode == 1)
                result = m_count_total - m_count_run_start;
            else
                result = m_count_total - m_count_step_start;
            return result;
            }

    protected:
        std::shared_ptr< IntegratorHPMCMono<Shape> > m_mc; //!< HPMC integrator
        unsigned int m_attempts_per_particle;  //!< Number of attempted moves per particle each time update() is called
        std::shared_ptr<Variant> m_beta_ficticious;  //!< Ficticious inverse temperature for creating links between particles
        LongReal m_translation_move_probability;
        unsigned int m_maximum_allowed_cluster_size;
        LongReal m_cluster_size_distribution_prefactor;
        std::string m_cluster_size_limit_mode;
        bool m_always_rebuild_tree;

        detail::UpdateOrder m_update_order;

        GlobalVector<Scalar4> m_postype_backup;        //!< Old local positions
        GlobalVector<Scalar4> m_orientation_backup;    //!< Old local orientations
        GlobalVector<int3> m_image_backup;             //!< Old local images
        LongReal m_maximum_translate_size;
        Scalar m_maximum_rotate_size;
        LongReal m_maximum_trial_center_of_rotation_shift;
        uint16_t  m_instance;

        // containers used during creation and execution of cluster moves
        std::vector<vec3<Scalar>> m_positions_after_move;
        ClusterData m_cluster_data;
        PotentialLinkData m_potential_link_data;

        hpmc_virtual_moves_counters_t m_count_total;
        hpmc_virtual_moves_counters_t m_count_run_start;
        hpmc_virtual_moves_counters_t m_count_step_start;
    };

template< class Shape >
UpdaterVMMC<Shape>::UpdaterVMMC(
        std::shared_ptr<SystemDefinition> sysdef,
        std::shared_ptr<Trigger> trigger,
        std::shared_ptr<IntegratorHPMCMono<Shape> > mc,
        std::shared_ptr<Variant> beta_ficticious
        ) : Updater(sysdef, trigger), m_mc(mc), m_beta_ficticious(beta_ficticious), m_update_order(m_pdata->getN())
    {
    m_exec_conf->msg->notice(5) << "Constructing UpdaterVMMC" << std::endl;

    // initialize stats
    resetStats();

    // initialize memory
    GlobalVector<Scalar4>(1,this->m_exec_conf).swap(m_postype_backup);
    TAG_ALLOCATION(m_postype_backup);
    GlobalVector<Scalar4>(1,this->m_exec_conf).swap(m_orientation_backup);
    TAG_ALLOCATION(m_orientation_backup);
    GlobalVector<int3>(1,this->m_exec_conf).swap(m_image_backup);
    TAG_ALLOCATION(m_image_backup);
    }

template< class Shape >
UpdaterVMMC<Shape>::~UpdaterVMMC()
    {
    m_exec_conf->msg->notice(5) << "Destroying UpdaterVMMC" << std::endl;
    }


/*! Perform a cluster move
    \param timestep Current time step of the simulation

    The steps of the algorithm are:
    1. Pick a random seed particle i and a random trial displacement/rotation
    2. Loop over all of i's neighbors (j):
        2a. Add j to the cluster with probability p_ij, which needs u_ij in the current state, and u_ij after only i has been moved
        2b. If j added to cluster, loop over all of its neighbors and add to cluster with probability p_jk
        2c. Continue recursively looping over neighbors until no candidate cluster members are left
    3. Do the move
    4. Do the reverse move virtually on particle i, and repeat the cluster-making process
        - This step gives you all of the information you need for evaluating the acceptance criterion
    5. Accept the total move according to acceptance criterion.

    So in a coarser-grained level of detail, it's:
    1. Calculate stuff based on current configuration and trial moves to find cluster
    2. Move cluster
    3. Calculate stuff based on config with cluster moved
    4. Accept or reject cluster move based on stuff calculated in steps 1 and 3
*/
template< class Shape >
void UpdaterVMMC<Shape>::update(uint64_t timestep)
    {
    m_exec_conf->msg->notice(4) << "VMMC update() timestep " << timestep << std::endl;
    Updater::update(timestep);
    #ifdef ENABLE_MPI
    if (this->m_pdata->getDomainDecomposition())
        throw std::runtime_error("UpdaterVMMC does not work with spatial domain decomposition.");
    #endif

    // if no particles, exit early
    if (! m_pdata->getN()) return;

    m_count_step_start = m_count_total;

    // get stuff needed for the calculation
    const LongReal current_beta_ficticious = m_beta_ficticious->operator()(timestep);
    const LongReal min_core_radius = m_mc->getMinCoreDiameter() * LongReal(0.5);
    const auto& pair_energy_search_radius = m_mc->getPairEnergySearchRadius();
    ArrayHandle<Scalar4> h_postype(m_pdata->getPositions(), access_location::host, access_mode::readwrite);
    ArrayHandle<Scalar4> h_orientation(m_pdata->getOrientationArray(), access_location::host, access_mode::readwrite);
    ArrayHandle<Scalar> h_charge(m_pdata->getCharges(), access_location::host, access_mode::readwrite);
    ArrayHandle<Scalar> h_diameter(m_pdata->getDiameters(), access_location::host, access_mode::readwrite);
    ArrayHandle<int3> h_image(m_pdata->getImages(), access_location::host, access_mode::readwrite);
    const BoxDim box = m_pdata->getBox();
    unsigned int ndim = this->m_sysdef->getNDimensions();
    uint16_t user_seed = m_sysdef->getSeed();
    const auto &mc_params = m_mc->getParams();
    unsigned int maximum_allowed_cluster_size = m_maximum_allowed_cluster_size == 0 ? m_pdata->getN() : m_maximum_allowed_cluster_size;
    ArrayHandle<Scalar4> pos_last_tree_build(m_pdata->getPositions(), access_location::host, access_mode::readwrite);

    #ifdef ENABLE_MPI
    // compute the width of the active region
    Scalar3 npd = box.getNearestPlaneDistance();
    Scalar3 ghost_fraction = m_mc->getNominalWidth() / npd;
    #endif
    Scalar3 npd_global = m_pdata->getGlobalBox().getNearestPlaneDistance();
    Scalar min_npd = detail::min(npd_global.x, npd_global.y);
    if (this->m_sysdef->getNDimensions() == 3)
        {
        min_npd = detail::min(min_npd, npd_global.z);
        }
    /* LongReal min_displacement_rebuild_tree = min_npd; */

    // Shuffle the order of particles for this step
    m_update_order.resize(m_pdata->getN());
    m_update_order.shuffle(timestep, m_sysdef->getSeed(), m_exec_conf->getRank());

    // get the AABB Tree
    const hoomd::detail::AABBTree& mc_aabb_tree = m_mc->buildAABBTree();
    // update the image list
    const std::vector<vec3<Scalar>>& image_list = m_mc->updateImageList();

    // access interaction matrix
    ArrayHandle<unsigned int> h_overlaps(m_mc->getInteractionMatrix(), access_location::host, access_mode::read);

    // counters
    hpmc_counters_t counters;

    m_positions_after_move.resize(m_pdata->getN());

    for (unsigned int i_trial = 0; i_trial < m_attempts_per_particle; i_trial++)
        {
        for (unsigned int current_seed = 0; current_seed < m_pdata->getN(); current_seed++)
            {
            m_mc->buildAABBTree();
            m_cluster_data.clear();
            m_potential_link_data.clear();
            LongReal p_link_probability_ratio = 1.0;  // 3rd product in eqn 13 from the paper
            LongReal p_failed_link_probability_ratio = 1.0;  // 1st product in eqn 13 from the paper
            LongReal beta_deltaU = 0.0;  // change in energy before and after the cluster move
            bool skip_to_next_seed = false;

            // seed particle that starts the virtual move
            unsigned int seed_idx = m_update_order[current_seed];

            // generate the virtual move
            hoomd::RandomGenerator rng_i(hoomd::Seed(hoomd::RNGIdentifier::UpdaterVMMC, timestep, user_seed),
                                         hoomd::Counter(seed_idx, (m_exec_conf->getRank() << 16) + m_instance, i_trial));
            LongReal move_type_select = hoomd::UniformDistribution<LongReal>()(rng_i);
            bool move_type_translate = move_type_select < m_translation_move_probability;
            vec3<Scalar> virtual_translate_move(0, 0, 0);
            quat<Scalar> virtual_rotate_move(1.0, vec3<Scalar>(0, 0, 0));
            if (move_type_translate)
                {
                move_translate(virtual_translate_move, rng_i, m_maximum_translate_size, ndim);
                }
            else
                {
                move_translate(virtual_translate_move, rng_i, m_maximum_trial_center_of_rotation_shift, ndim);
                if (ndim == 2)
                    {
                    move_rotate<2>(virtual_rotate_move, rng_i, m_maximum_rotate_size);
                    }
                else
                    {
                    move_rotate<3>(virtual_rotate_move, rng_i, m_maximum_rotate_size);
                    }
                }

            // add linker and current image rotation center to their sets
            vec3<Scalar> pos_seed = vec3<Scalar>(h_postype.data[seed_idx]);
            m_cluster_data.update_cluster(seed_idx, virtual_translate_move + pos_seed);

            m_exec_conf->msg->notice(6) << " VMMC virtual translation = ("
                << virtual_translate_move.x << ", "
                << virtual_translate_move.y << ", "
                << virtual_translate_move.z << ")" <<std::endl;

            // loop over neighbors of cluster members to find new cluster members
            m_exec_conf->msg->notice(5) << "VMMC seed tag: " << seed_idx << std::endl;
            /* int3 _images = make_int3(0, 0, 0); */
            while (m_cluster_data.m_linkers.size() > 0)
                {
                if (skip_to_next_seed)
                    {
                    break;
                    }
                unsigned int i_linker = m_cluster_data.m_linkers[0];
                m_cluster_data.m_linkers.erase(m_cluster_data.m_linkers.begin());
                vec3<Scalar> center_of_rotation = m_cluster_data.m_linker_rotation_centers[0];
                m_cluster_data.m_linker_rotation_centers.erase(m_cluster_data.m_linker_rotation_centers.begin());
                m_exec_conf->msg->notice(5) << "VMMC linker tag: " << i_linker << std::endl;

                // get position and orientation of linker
                Scalar4 postype_linker = h_postype.data[i_linker];
                vec3<Scalar> pos_linker = vec3<Scalar>(postype_linker);
                vec3<Scalar> pos_linker_after_move_primary_image;
                vec3<Scalar> pos_linker_after_reverse_move_primary_image;
                if (move_type_translate)
                    {
                    pos_linker_after_move_primary_image = pos_linker + virtual_translate_move;
                    pos_linker_after_reverse_move_primary_image = pos_linker - virtual_translate_move;
                    /* box.wrap(pos_linker_after_move_primary_image, _images); */
                    /* box.wrap(pos_linker_after_reverse_move_primary_image, _images); */
                    }
                else
                    {
                    pos_linker_after_move_primary_image = rotate(virtual_rotate_move, pos_linker - center_of_rotation) + center_of_rotation;
                    m_exec_conf->msg->notice(5) << "center of rotation for particle " << i_linker << " = " << center_of_rotation.x << " " << center_of_rotation.y << " " << center_of_rotation.z << std::endl;
                    pos_linker_after_reverse_move_primary_image = rotate(conj(virtual_rotate_move), pos_linker - center_of_rotation) + center_of_rotation;
                    /* box.wrap(pos_linker_after_move_primary_image, _images); */
                    /* box.wrap(pos_linker_after_reverse_move_primary_image, _images); */
                    }
                m_positions_after_move[i_linker] = pos_linker_after_move_primary_image;

                int type_linker = __scalar_as_int(postype_linker.w);
                Shape shape_linker(quat<Scalar>(h_orientation.data[i_linker]), mc_params[type_linker]);
                Shape shape_linker_after_move(
                    virtual_rotate_move * quat<Scalar>(h_orientation.data[i_linker]), mc_params[type_linker]);
                Shape shape_linker_after_reverse_move(
                    conj(virtual_rotate_move) * quat<Scalar>(h_orientation.data[i_linker]), mc_params[type_linker]);


                // linker must be an active particle before and after the move
                #ifdef ENABLE_MPI
                if (m_sysdef->isDomainDecomposed())
                    {
                    if (
                        !isActive(make_scalar3(pos_linker.x, pos_linker.y, pos_linker.z), box, ghost_fraction)
                        || (!isActive(
                            make_scalar3(
                                pos_linker_after_move_primary_image.x,
                                pos_linker_after_move_primary_image.y,
                                pos_linker_after_move_primary_image.z),
                            box,
                            ghost_fraction))

                       )
                        {
                        m_count_total.reject_count_inactive_seed++;
                        if (move_type_translate)
                            {
                            m_count_total.translate_reject_count++;
                            }
                        else
                            {
                            m_count_total.rotate_reject_count++;
                            }
                        skip_to_next_seed = true;
                        break;
                        }
                    }
                #endif

                // search for all particles that might touch this one
                LongReal R_query = m_mc->getShapeCircumsphereRadius()[type_linker];

                if (m_mc->hasPairInteractions())
                    {
                    // Extend the search to include the pair interaction r_cut
                    // subtract minimum AABB extent from search radius
                    R_query = std::max(R_query, pair_energy_search_radius[type_linker] - min_core_radius);
                    }
                hoomd::detail::AABB aabb_linker_local_no_moves = hoomd::detail::AABB(vec3<Scalar>(0, 0, 0), R_query);
                hoomd::detail::AABB aabb_linker_local_forward = hoomd::detail::AABB(pos_linker_after_move_primary_image - pos_linker, R_query);
                hoomd::detail::AABB aabb_linker_local_reverse = hoomd::detail::AABB(pos_linker_after_reverse_move_primary_image - pos_linker, R_query);
                hoomd::detail::AABB _aabb = hoomd::detail::merge(aabb_linker_local_no_moves, aabb_linker_local_forward);
                hoomd::detail::AABB aabb_linker_local = hoomd::detail::merge(_aabb, aabb_linker_local_reverse);

                // loop over all images
                const unsigned int n_images = (unsigned int)image_list.size();
                for (unsigned int current_image = 0; current_image < n_images; current_image++)
                    {
                    if (skip_to_next_seed)
                        {
                        break;
                        }
                    // create an AABB centered on the linker particle in the current box image
                    hoomd::detail::AABB aabb = aabb_linker_local;
                    vec3<Scalar> pos_linker_image = pos_linker + image_list[current_image];
                    vec3<Scalar> center_of_rotation_image = center_of_rotation + image_list[current_image];
                    aabb.translate(pos_linker_image);

                    // stackless search
                    for (unsigned int current_node_index = 0; current_node_index < mc_aabb_tree.getNumNodes(); current_node_index++)
                        {
                        if (skip_to_next_seed)
                            {
                            break;
                            }
                        if (aabb.overlaps(mc_aabb_tree.getNodeAABB(current_node_index)))
                            {
                            if (mc_aabb_tree.isNodeLeaf(current_node_index))
                                {
                                for (unsigned int current_linkee = 0; current_linkee < mc_aabb_tree.getNodeNumParticles(current_node_index); current_linkee++)
                                    {
                                    if (skip_to_next_seed)
                                        {
                                        break;
                                        }
                                    m_exec_conf->msg->notice(6) << "linker_set_size = " << m_cluster_data.m_linkers.size() << std::endl;
                                    unsigned int j_linkee = mc_aabb_tree.getNodeParticle(current_node_index, current_linkee);
                                    if (!(m_cluster_data.m_linkers_added.find(j_linkee) == m_cluster_data.m_linkers_added.end()))
                                        {
                                        // j already in cluster
                                        continue;
                                        }
                                    Scalar4 postype_linkee;
                                    postype_linkee = h_postype.data[j_linkee];
                                    quat<LongReal> orientation_linkee;
                                    orientation_linkee = quat<LongReal>(h_orientation.data[j_linkee]);

                                    unsigned int type_linkee = __scalar_as_int(postype_linkee.w);
                                    Shape shape_linkee(orientation_linkee, mc_params[type_linkee]);
                                    LongReal max_overlap_distance = m_mc->getShapeCircumsphereRadius()[type_linker] + m_mc->getShapeCircumsphereRadius()[type_linkee];

                                    // at this point we test for a link between the linker and linkee
                                    // add j to the cluster with probability p_ij = 1 - exp(E_C - E_I) (hard overlap makes E_I go to infinity and p_ij = 1)
                                    // so we need E_C (current pair energy) and E_I (pair energy after applying virtual move to i)
                                    // we only need E_C if E_I, so we can check for overlaps first

                                    // pair separation in current real (i.e. non-virtual) configuration
                                    vec3<Scalar> r_linker_linkee = vec3<Scalar>(postype_linkee) - pos_linker_image;
                                    LongReal r_squared = dot(r_linker_linkee, r_linker_linkee);

                                    // pair separation after applying virtual move to linker (i)
                                    vec3<Scalar> r_linker_linkee_after_forward_move_of_linker = vec3<Scalar>(postype_linkee) - (pos_linker_after_move_primary_image + image_list[current_image]);
                                    LongReal r_squared_after_forward_move_of_linker = dot(
                                        r_linker_linkee_after_forward_move_of_linker, r_linker_linkee_after_forward_move_of_linker);

                                    // pair separation after applying reverse virtual move to linker (i)
                                    vec3<Scalar> r_linker_linkee_after_reverse_move_of_linker = vec3<Scalar>(postype_linkee) - (pos_linker_after_reverse_move_primary_image + image_list[current_image]);
                                    LongReal r_squared_after_reverse_move_of_linker = dot(
                                        r_linker_linkee_after_reverse_move_of_linker, r_linker_linkee_after_reverse_move_of_linker);

                                    m_exec_conf->msg->notice(5) << "Test for links: pair " << i_linker << " and " << j_linkee << std::endl;
                                    m_exec_conf->msg->notice(5) << "  Checking for overlap after moving linker between pair " << i_linker << " and " << j_linkee << std::endl;
                                    bool overlap_after_forward_move_of_linker =
                                        h_overlaps.data[m_mc->getOverlapIndexer()(type_linker, type_linkee)]
                                        && r_squared_after_forward_move_of_linker < max_overlap_distance * max_overlap_distance
                                        && test_overlap(r_linker_linkee_after_forward_move_of_linker, shape_linker_after_move, shape_linkee, counters.overlap_err_count);
                                    if (overlap_after_forward_move_of_linker)
                                        {
                                        m_exec_conf->msg->notice(5) << "    Overlap found between " << i_linker << " and " << j_linkee << " -> link formed!" << std::endl;

                                        // add linkee to cluster if not in there already and after checking maximum cluster_size
                                        if (
                                                m_cluster_size_limit_mode == "deterministic"
                                                && m_cluster_data.current_cluster_size() == maximum_allowed_cluster_size)
                                            {
                                            // abort the move.
                                            skip_to_next_seed = true;
                                            break;
                                            }
                                        else if (m_cluster_size_limit_mode == "probabilistic"
                                                && hoomd::UniformDistribution<Scalar>()(rng_i) > m_cluster_size_distribution_prefactor / ((LongReal)m_cluster_data.current_cluster_size() + 1))
                                            {
                                            // abort the move.
                                            skip_to_next_seed = true;
                                            break;
                                            }
                                        else
                                            {
                                            m_cluster_data.update_cluster(j_linkee, center_of_rotation_image);
                                            }

                                        // account for probability of forming this link: denominator of p_link_probability_ratio *= 1
                                        p_link_probability_ratio /= 1.0;

                                        // account for probability of forming this link on the reverse move:
                                        // numerator of p_link_probability_ratio p_ij_reverse gets multiplied by p_ij_reverse
                                        // if overlap after reverse move of i, p_ij_reverse = 1, otherwise we look at energies
                                        Scalar p_ij_reverse;
                                        bool overlap_after_reverse_move_of_linker =
                                            h_overlaps.data[m_mc->getOverlapIndexer()(type_linker, type_linkee)]
                                            && r_squared_after_reverse_move_of_linker < max_overlap_distance * max_overlap_distance
                                            && test_overlap(r_linker_linkee_after_reverse_move_of_linker, shape_linker_after_reverse_move, shape_linkee, counters.overlap_err_count);
                                        if (overlap_after_reverse_move_of_linker)
                                            {
                                            p_ij_reverse = 1.0;
                                            m_exec_conf->msg->notice(5) << "  p_ij_reverse = "  << p_ij_reverse << " by overlap on reverse move of linker." << std::endl;
                                            }
                                        else
                                            {
                                            double u_ij =
                                                m_mc->computeOnePairEnergy(
                                                        r_squared,
                                                        r_linker_linkee,
                                                        type_linker,
                                                        shape_linker.orientation,
                                                        h_diameter.data[i_linker],
                                                        h_charge.data[i_linker],
                                                        type_linkee,
                                                        shape_linkee.orientation,
                                                        h_diameter.data[j_linkee],
                                                        h_charge.data[j_linkee]);
                                            double u_ij_after_reverse_move_of_linker =
                                                m_mc->computeOnePairEnergy(r_squared_after_reverse_move_of_linker,
                                                        r_linker_linkee_after_reverse_move_of_linker,
                                                        type_linker,
                                                        shape_linker_after_reverse_move.orientation,
                                                        h_diameter.data[i_linker],
                                                        h_charge.data[i_linker],
                                                        type_linkee,
                                                        shape_linkee.orientation,
                                                        h_diameter.data[j_linkee],
                                                        h_charge.data[j_linkee]);
                                            p_ij_reverse = std::max(0.0, 1 - exp(-current_beta_ficticious*(u_ij_after_reverse_move_of_linker - u_ij)));
                                            m_exec_conf->msg->notice(5) << "  u_ij = "  << u_ij << std::endl;
                                            m_exec_conf->msg->notice(5) << "  u_ij_after_reverse_move = "  << u_ij_after_reverse_move_of_linker << std::endl;
                                            m_exec_conf->msg->notice(5) << "  p_ij_reverse = "  << p_ij_reverse << std::endl;
                                            }
                                        if (p_ij_reverse == 0.0)
                                            {
                                            m_count_total.early_abort_no_reverse_link++;
                                            if (move_type_translate)
                                                {
                                                m_count_total.translate_reject_count++;
                                                }
                                            else
                                                {
                                                m_count_total.rotate_reject_count++;
                                                }
                                            skip_to_next_seed = true;
                                            break;  // break loop over linkees
                                            }
                                        p_link_probability_ratio *= p_ij_reverse;
                                        continue; // to next linkee
                                        } // end if (overlap_after_forward_move_of_linker)
                                    m_exec_conf->msg->notice(5) << "    No overlap after moving linker between " << i_linker << " and " << j_linkee << std::endl;

                                    // if no overlap, we have to calculate u_ij and u_ij_after_move to determine p_ij
                                    double u_ij_after_move =
                                        m_mc->computeOnePairEnergy(r_squared_after_forward_move_of_linker,
                                                r_linker_linkee_after_forward_move_of_linker,
                                                type_linker,
                                                shape_linker_after_move.orientation,
                                                h_diameter.data[i_linker],
                                                h_charge.data[i_linker],
                                                type_linkee,
                                                shape_linkee.orientation,
                                                h_diameter.data[j_linkee],
                                                h_charge.data[j_linkee]);
                                    double u_ij =
                                        m_mc->computeOnePairEnergy(
                                                r_squared,
                                                r_linker_linkee,
                                                type_linker,
                                                shape_linker.orientation,
                                                h_diameter.data[i_linker],
                                                h_charge.data[i_linker],
                                                type_linkee,
                                                shape_linkee.orientation,
                                                h_diameter.data[j_linkee],
                                                h_charge.data[j_linkee]);
                                    if (u_ij_after_move == 0.0 && u_ij == 0.0)
                                        {
                                        // non-interacting -> non-interacting; continue to next linkee
                                        continue;  // to next linkee
                                        }

                                    LongReal p_ij_forward = std::max(0.0, 1.0 - exp(-current_beta_ficticious * (u_ij_after_move - u_ij)));
                                    Scalar r = hoomd::UniformDistribution<Scalar>()(rng_i);
                                    bool link_formed = r < p_ij_forward;
                                    if (link_formed)
                                        {
                                        m_exec_conf->msg->notice(5) << "  p_ij_forward = " << p_ij_forward << " > " << r << " -> link_formed!" << std::endl;
                                        if (
                                                m_cluster_size_limit_mode == "deterministic"
                                                && m_cluster_data.current_cluster_size() == maximum_allowed_cluster_size)
                                            {
                                            // abort the move.
                                            skip_to_next_seed = true;
                                            break;
                                            }
                                        else if (m_cluster_size_limit_mode == "probabilistic"
                                                && hoomd::UniformDistribution<Scalar>()(rng_i) > m_cluster_size_distribution_prefactor / ((LongReal)m_cluster_data.current_cluster_size() + 1))
                                            {
                                            // abort the move.
                                            skip_to_next_seed = true;
                                            break;
                                            }
                                        else
                                            {
                                            m_cluster_data.update_cluster(j_linkee, center_of_rotation_image);
                                            }

                                        // numerator of p_link_probability_ratio gets multiplied by p_ij_reverse
                                        Scalar p_ij_reverse;
                                        bool overlap_after_reverse_move =
                                            h_overlaps.data[m_mc->getOverlapIndexer()(type_linker, type_linkee)]
                                            && r_squared_after_reverse_move_of_linker < max_overlap_distance * max_overlap_distance
                                            && test_overlap(r_linker_linkee_after_reverse_move_of_linker, shape_linker_after_reverse_move, shape_linkee, counters.overlap_err_count);
                                        if (overlap_after_reverse_move)
                                            {
                                            p_ij_reverse = 1.0;
                                            m_exec_conf->msg->notice(5) << "  Overlap found on reverse move -> p_ij_reverse = "  << p_ij_reverse << std::endl;
                                            }
                                        else
                                            {
                                            double u_ij_after_reverse_move =
                                                m_mc->computeOnePairEnergy(r_squared_after_reverse_move_of_linker,
                                                        r_linker_linkee_after_reverse_move_of_linker,
                                                        type_linker,
                                                        shape_linker_after_reverse_move.orientation,
                                                        h_diameter.data[i_linker],
                                                        h_charge.data[i_linker],
                                                        type_linkee,
                                                        shape_linkee.orientation,
                                                        h_diameter.data[j_linkee],
                                                        h_charge.data[j_linkee]);
                                            m_exec_conf->msg->notice(5) << "  u_ij_after_reverse_move = "  << u_ij_after_move << std::endl;
                                            p_ij_reverse = std::max(0.0, 1.0 - exp(-current_beta_ficticious * (u_ij_after_reverse_move - u_ij)));
                                            if (p_ij_reverse == 0.0)
                                                {
                                                m_count_total.early_abort_no_reverse_link++;
                                                m_exec_conf->msg->notice(5) << "  p_ij_reverse = 0; aborting move."  << std::endl;
                                                if (move_type_translate)
                                                    {
                                                    m_count_total.translate_reject_count++;
                                                    }
                                                else
                                                    {
                                                    m_count_total.rotate_reject_count++;
                                                    }
                                                skip_to_next_seed = true;
                                                break;
                                                }
                                            m_exec_conf->msg->notice(5) << "  p_ij_reverse = "  << p_ij_reverse << std::endl;
                                            }
                                        p_link_probability_ratio *= p_ij_reverse / p_ij_forward;
                                        continue;
                                        }
                                    else  // no link formed
                                        {
                                        m_exec_conf->msg->notice(5) << "  p_ij_forward = " << p_ij_forward << "; random number = " << r << " -> no link_formed" << std::endl;
                                        // if j ultimately ends up in the cluster, the dU contribution is 0, but we still have to
                                        // if j ultimately does not end up in the cluster, we do have to account for the energy change,
                                        // so dU_forward += dU
                                        // either way, we still must account
                                        // for the probability of not forming
                                        // that link, so the denominator of
                                        // p_failed_link_probability_ratio gets
                                        // multiplied by 1 - p_ij_forward = 1 -
                                        // (1 - exp(-beta * (u_ij_after_move - u_ij))) = exp(-beta * (u_ij_after_move - u_ij))
                                        // and the numerator of p_failed_link_probability_ratio gets multiplied by 1 - p_ij_reverse
                                        // 1 - p_ij_reverse = 1 - (1 - exp(-beta * (u_ij - u_ij_after_move)))
                                        // = exp(-beta * (u_ij - u_ij_after_move))
                                        // which means no matter what we only need u_ij (already have) and u_ij_after_move (also already have)
                                        // intuitively it makes sense that we don't need u_ij_after_reverse move because the two
                                        // relevant states for a non-linker pair are (x_i, x_j) and (x_i + dx, x_j) (where dx is
                                        // a generic virtual move) so the only relevant energies are u_ij and u_ij_after_move

                                        // account for probabilities of not forming this link
                                        Scalar q_ij_forward = 1.0 - p_ij_forward;
                                        Scalar q_ij_reverse = std::min(1.0, fast::exp(-current_beta_ficticious * (u_ij - u_ij_after_move)));
                                        if (q_ij_reverse == 0.0)
                                            {
                                            m_count_total.early_abort_forced_reverse_link++;
                                            m_exec_conf->msg->notice(5) << "  q_ij_reverse = 0; aborting move." << std::endl;
                                            skip_to_next_seed = true;
                                            if (move_type_translate)
                                                {
                                                m_count_total.translate_reject_count++;
                                                }
                                            else
                                                {
                                                m_count_total.rotate_reject_count++;
                                                }
                                            break;
                                            }
                                        m_exec_conf->msg->notice(5) << "  q_ij_forward = "  << q_ij_forward << std::endl;
                                        m_exec_conf->msg->notice(5) << "  q_ij_reverse = "  << q_ij_reverse << std::endl;
                                        p_failed_link_probability_ratio *= q_ij_reverse / q_ij_forward;

                                        // add to possible contributions to deltaU for the cluster move
                                        m_potential_link_data.update(i_linker, j_linkee, (LongReal)u_ij_after_move - (LongReal)u_ij);
                                        continue;
                                        }
                                    }  // end loop over linkees
                                }  // end if (mc_aabb_tree.isNodeLeaf(current_node_index))
                            }  // end if (aabb.overlaps(mc_aabb_tree.getNodeAABB(current_node_index)))
                        else
                            {
                            // skip ahead
                            current_node_index += mc_aabb_tree.getNodeSkip(current_node_index);
                            }
                        }  // end loop over nodes in AABBTree
                    }  // end loop over images
                }  // end loop over linkers
            if (skip_to_next_seed)
                {
                continue;
                }

            // find which of the potential cluster members ended up in the cluster
            m_exec_conf->msg->notice(5) << "Finding failed intracluster links" << std::endl;
            for (unsigned int i = 0; i < m_potential_link_data.m_potential_intracluster_link_idxs.size(); i++)
                {
                unsigned int linker_i = m_potential_link_data.m_potential_intracluster_link_idxs[i].first;  // recall: linker_i in cluster by construction
                unsigned int linkee_j = m_potential_link_data.m_potential_intracluster_link_idxs[i].second;
                m_exec_conf->msg->notice(10) << "  Checking failed, intracluster links between " << linker_i << " and " << linkee_j << std::endl;
                if (m_cluster_data.m_linkers_added.find(linkee_j) == m_cluster_data.m_linkers_added.end())
                    {
                    // j ultimately did not join the cluster, so there will in general be a contribution to beta_deltaU for this i-j pair
                    // we have already accounted for the probability of not forming the i-j link, so all we do is add to beta_deltaU
                    beta_deltaU += m_potential_link_data.m_potential_intracluster_link_beta_deltaU_fwds[i];
                    }
                else
                    {
                    m_exec_conf->msg->notice(5) << "    particle with tag " << linkee_j << " eventually joined the cluster." << std::endl;
                    // j ultimately joined the cluster so there will be no contribution to deltaU for this i-j pair
                    // we have also already accounted for the probability of not forming that link in the forward and reverse moves
                    // so there is nothing to do here except for explicitly add 0.0 to beta_deltaU for the sake of code readability
                    beta_deltaU += 0.0;
                    }
                }


            // now we can accept or reject the move
            m_count_total.total_num_moves_attempted++;
            m_exec_conf->msg->notice(5) << "  failed_link_probability_ratio = " << p_failed_link_probability_ratio << std::endl;
            m_exec_conf->msg->notice(5) << "  p_link_probability_ratio = " << p_link_probability_ratio << std::endl;
            m_exec_conf->msg->notice(5) << "  beta*dU = " << beta_deltaU << std::endl;
            m_exec_conf->msg->notice(5) << "  exp(-beta*dU) = " << exp(-beta_deltaU) << std::endl;
            LongReal p_acc = std::min(1.0, exp(-beta_deltaU) * p_link_probability_ratio * p_failed_link_probability_ratio);
            Scalar r = hoomd::UniformDistribution<Scalar>()(rng_i);
            bool accept_cluster_move = r <= p_acc;
            m_exec_conf->msg->notice(4) << "  VMMC p_acc: " << p_acc << std::endl;
            size_t cluster_size = m_cluster_data.m_linkers_added.size();
            m_count_total.total_num_particles_in_clusters += (unsigned long long int)cluster_size;
            if(accept_cluster_move)
                {
                m_count_total.total_num_particles_in_accepted_cluster_moves += (unsigned long long int)cluster_size;
                m_exec_conf->msg->notice(4) << "  VMMC move accepted, moving " << cluster_size << " particles as a cluster" << std::endl;
                if (move_type_translate)
                    {
                    m_count_total.translate_accept_count++;
                    }
                else
                    {
                    m_count_total.rotate_accept_count++;
                    }

                // move the particles that are in the cluster
                /* bool rebuild_tree = m_always_rebuild_tree ? true : false; */
                for(unsigned int idx = 0; idx < m_pdata->getN(); idx++)
                    {
                    if ( !(m_cluster_data.m_linkers_added.find(idx) == m_cluster_data.m_linkers_added.end()) )
                        {
                        // particle idx is in the cluster
                        Scalar4 postype_idx = h_postype.data[idx];
                        vec3<Scalar> new_pos = m_positions_after_move[idx];
                        h_postype.data[idx] = make_scalar4(new_pos.x, new_pos.y, new_pos.z, postype_idx.w);
                        box.wrap(h_postype.data[idx], h_image.data[idx]);
                        /* vec3<Scalar> displacement = vec3<Scalar>(h_postype.data[idx]) - vec3<Scalar>(pos_last_tree_build.data[idx]); */
                        /* LongReal displacement_sq = dot(displacement, displacement); */
                        /* if (displacement_sq > min_displacement_rebuild_tree * min_displacement_rebuild_tree) */
                        /*     { */
                        /*     rebuild_tree = true; */
                        /*     } */
                        quat<Scalar> orientation(h_orientation.data[idx]);
                        h_orientation.data[idx] = quat_to_scalar4(virtual_rotate_move * orientation);
                        }
                    }
                m_mc->invalidateAABBTree();
                /* if (rebuild_tree) */
                /*     { */
                /*     m_mc->invalidateAABBTree(); */
                /*     } */
                /* else */
                /*     { */
                /*     for(unsigned int idx = 0; idx < m_pdata->getN(); idx++) */
                /*         { */
                /*         if ( !(m_cluster_data.m_linkers_added.find(idx) == m_cluster_data.m_linkers_added.end()) ) */
                /*             { */
                /*             LongReal R_query = m_mc->getShapeCircumsphereRadius()[h_postype.data[idx].w]; */
                /*             if (m_mc->hasPairInteractions()) */
                /*                 { */
                /*                 // Extend the search to include the pair interaction r_cut */
                /*                 // subtract minimum AABB extent from search radius */
                /*                 R_query = std::max(R_query, pair_energy_search_radius[h_postype.data[idx].w] - min_core_radius); */
                /*                 } */
                /*             hoomd::detail::AABB new_aabb = hoomd::detail::AABB(vec3<LongReal>(h_postype.data[idx]), R_query); */
                /*             mc_aabb_tree.update(idx, new_aabb); */
                /*             } */
                /*         } */
                /*     } */
                }
            else
                {
                m_exec_conf->msg->notice(4) << "VMMC move rejected (p_acc = " << p_acc << " < r = " << r << ")" << std::endl;
                if (move_type_translate)
                    {
                    m_count_total.translate_reject_count++;
                    }
                else
                    {
                    m_count_total.rotate_reject_count++;
                    }
                }
            m_exec_conf->msg->notice(5) << std::endl;
            }  // end loop over seed particles
        }  // end loop over trials
    }  // end update()


namespace detail {

template < class Shape> void export_UpdaterVirtualMoveMonteCarlo(pybind11::module& m, const std::string& name)
    {
    pybind11::class_< UpdaterVMMC<Shape>, Updater, std::shared_ptr< UpdaterVMMC<Shape> > >(m, name.c_str())
        .def(pybind11::init<
                            std::shared_ptr<SystemDefinition>,
                            std::shared_ptr<Trigger>,
                            std::shared_ptr<IntegratorHPMCMono<Shape>>,
                            std::shared_ptr<Variant>
                            >())
        .def("getCounters", &UpdaterVMMC<Shape>::getCounters)
        .def_property("beta_ficticious", &UpdaterVMMC<Shape>::getBetaFicticious, &UpdaterVMMC<Shape>::setBetaFicticious)
        .def_property("translation_move_probability", &UpdaterVMMC<Shape>::getTranslationMoveProbability, &UpdaterVMMC<Shape>::setTranslationMoveProbability)
        .def_property("maximum_trial_rotation", &UpdaterVMMC<Shape>::getMaximumTrialRotation, &UpdaterVMMC<Shape>::setMaximumTrialRotation)
        .def_property("maximum_trial_translation", &UpdaterVMMC<Shape>::getMaximumTrialTranslation, &UpdaterVMMC<Shape>::setMaximumTrialTranslation)
        .def_property("attempts_per_particle", &UpdaterVMMC<Shape>::getAttemptsPerParticle, &UpdaterVMMC<Shape>::setAttemptsPerParticle)
        .def_property("maximum_trial_center_of_rotation_shift", &UpdaterVMMC<Shape>::getMaximumTrialCenterOfRotationShift, &UpdaterVMMC<Shape>::setMaximumTrialCenterOfRotationShift)
        .def_property("maximum_allowed_cluster_size", &UpdaterVMMC<Shape>::getMaximumAllowedClusterSize, &UpdaterVMMC<Shape>::setMaximumAllowedClusterSize)
        .def_property("cluster_size_limit_mode", &UpdaterVMMC<Shape>::getClusterSizeLimitMode, &UpdaterVMMC<Shape>::setClusterSizeLimitMode)
        .def_property("cluster_size_distribution_prefactor", &UpdaterVMMC<Shape>::getClusterSizeDistributionPrefactor, &UpdaterVMMC<Shape>::setClusterSizeDistributionPrefactor)
        .def_property("always_rebuild_tree", &UpdaterVMMC<Shape>::getAlwaysRebuildTree, &UpdaterVMMC<Shape>::setAlwaysRebuildTree)
    ;
    }

inline void export_hpmc_virtual_moves_counters(pybind11::module &m)
    {
pybind11::class_<hpmc_virtual_moves_counters_t>(m, "hpmc_virtual_moves_counters_t")
    .def_property_readonly("translate_counts", &hpmc_virtual_moves_counters_t::getTranslateCounts)
    .def_property_readonly("rotate_counts", &hpmc_virtual_moves_counters_t::getRotateCounts)
    .def_property_readonly("average_cluster_size", &hpmc_virtual_moves_counters_t::getAverageClusterSize)
    .def_property_readonly("total_num_particles_in_clusters", &hpmc_virtual_moves_counters_t::getTotalNumParticlesInClusters)
    .def_property_readonly("total_num_moves_attempted", &hpmc_virtual_moves_counters_t::getTotalNumMovesAttempted)
    .def_property_readonly("early_abort_counts", &hpmc_virtual_moves_counters_t::getEarlyAbortCounts)
    .def_property_readonly("average_cluster_size_accepted", &hpmc_virtual_moves_counters_t::getAverageAcceptedClusterSize)
    ;
    }


} // end namespace detail
} // end namespace hpmc
  //
} // end namespace hoomd

#endif // _UPDATER_HPMC_VIRTUAL_CLUSTER_MOVES_
