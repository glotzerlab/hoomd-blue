// Copyright (c) 2009-2024 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

#ifndef _UPDATER_HPMC_VIRTUAL_CLUSTER_MOVES_
#define _UPDATER_HPMC_VIRTUAL_CLUSTER_MOVES_

#include "hoomd/Updater.h"
#include "hoomd/RandomNumbers.h"
#include "hoomd/RNGIdentifiers.h"

#include <algorithm>
#include <set>

#include "Moves.h"
#include "HPMCCounters.h"
#include "IntegratorHPMCMono.h"
#include <pybind11/pybind11.h>

#include "hoomd/extern/dset.h"


namespace hoomd {

namespace hpmc
{


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
        UpdaterVMMC(std::shared_ptr<SystemDefinition> sysdef, std::shared_ptr<Trigger> trigger, std::shared_ptr<IntegratorHPMCMono<Shape> > mc);

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

        void setBetaFicticious(LongReal beta_ficticious)
            {
            m_beta_ficticious = beta_ficticious;
            }

        LongReal getBetaFicticious()
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

        /// Reset statistics counters
        virtual void resetStats()
            {
            m_count_run_start = m_count_total;
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
        LongReal m_beta_ficticious;  //!< Ficticious inverse temperature for creating links between particles
        LongReal m_translation_move_probability;

        detail::UpdateOrder m_update_order;

        GlobalVector<Scalar4> m_postype_backup;        //!< Old local positions
        GlobalVector<Scalar4> m_orientation_backup;    //!< Old local orientations
        GlobalVector<int3> m_image_backup;             //!< Old local images
        LongReal m_maximum_translate_size;
        Scalar m_maximum_rotate_size;

        // TODO: what all do we need to keep track of with counters?
        // 1. Number of cluster moves attempted and rejected
        // 2. Size of clusters?
        hpmc_virtual_moves_counters_t m_count_total;
        hpmc_virtual_moves_counters_t m_count_run_start;
        hpmc_virtual_moves_counters_t m_count_step_start;
    };

template< class Shape >
UpdaterVMMC<Shape>::UpdaterVMMC(
        std::shared_ptr<SystemDefinition> sysdef,
        std::shared_ptr<Trigger> trigger,
        std::shared_ptr<IntegratorHPMCMono<Shape> > mc
        ) : Updater(sysdef, trigger), m_mc(mc), m_update_order(m_pdata->getN())
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
    const LongReal min_core_radius = m_mc->getMinCoreDiameter() * LongReal(0.5);
    const auto& pair_energy_search_radius = m_mc->getPairEnergySearchRadius();
    ArrayHandle<Scalar4> h_postype(m_pdata->getPositions(), access_location::host, access_mode::readwrite);
    ArrayHandle<Scalar4> h_orientation(m_pdata->getOrientationArray(), access_location::host, access_mode::readwrite);
    ArrayHandle<Scalar> h_charge(m_pdata->getCharges(), access_location::host, access_mode::readwrite);
    ArrayHandle<Scalar> h_diameter(m_pdata->getDiameters(), access_location::host, access_mode::readwrite);
    const BoxDim box = m_pdata->getBox();
    unsigned int ndim = this->m_sysdef->getNDimensions();
    uint16_t seed = m_sysdef->getSeed();
    const auto &mc_params = m_mc->getParams();

    #ifdef ENABLE_MPI
    // compute the width of the active region
    Scalar3 npd = box.getNearestPlaneDistance();
    Scalar3 ghost_fraction = m_mc->getNominalWidth() / npd;
    #endif

    // Shuffle the order of particles for this step
    // TODO: use separate UpdateOrder from m_mc
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


    std::vector<unsigned int> linkers;
    std::set<unsigned int> linkers_added;
    std::vector<vec3<Scalar>> linker_rotation_centers;
    std::vector<vec3<Scalar>> positions_after_move(m_pdata->getN());
    for (unsigned int i_trial = 0; i_trial < m_attempts_per_particle; i_trial++)
        {
        for (unsigned int current_seed = 0; current_seed < m_pdata->getN(); current_seed++)
            {
            m_mc->buildAABBTree();
            linkers_added.clear();
            linkers.clear();
            linker_rotation_centers.clear();
            positions_after_move.clear();
            // set of all "linker" particles that attempt to create links to "linkee" particles
            DisjointSets cluster(m_pdata->getN());  // tags of particles in the cluster
            LongReal p_link_probability_ratio = 1.0;  // 3rd product in eqn 13 from the paper
            LongReal p_failed_link_probability_ratio = 1.0;  // 1st product in eqn 13 from the paper
            LongReal beta_deltaU = 0.0;  // change in energy before and after the cluster move
            std::vector<std::pair<unsigned int, unsigned int>> potential_intracluster_links;
            std::vector<double> u_ij_potential_links;
            std::vector<double> u_ij_forward_potential_links;

            // seed particle that starts the virtual move
            unsigned int seed_idx = m_update_order[current_seed];

            // generate the virtual move
            hoomd::RandomGenerator rng_i(hoomd::Seed(hoomd::RNGIdentifier::UpdaterVMMC, timestep, seed),
                                         hoomd::Counter(seed_idx, m_exec_conf->getRank(), i_trial));
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
                move_translate(virtual_translate_move, rng_i, m_maximum_translate_size, ndim);
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
            linkers.push_back(seed_idx);
            linkers_added.insert(seed_idx);
            linker_rotation_centers.push_back(virtual_translate_move);
            positions_after_move[seed_idx] = virtual_translate_move;


            // loop over neighbors of cluster members to find new cluster members
            bool skip_to_next_seed = false;
            m_exec_conf->msg->notice(5) << "VMMC seed tag: " << seed_idx << std::endl;
            while (linkers.size() > 0)
                {
                /* m_exec_conf->msg->notice(5) << "VMMC linker set size " << linkers.size() << std::endl; */
                unsigned int i_linker = linkers.back();
                linkers.pop_back();
                m_exec_conf->msg->notice(5) << "VMMC linker tag: " << i_linker << std::endl;
                vec3<Scalar> center_of_rotation = linker_rotation_centers.back();
                linker_rotation_centers.pop_back();

                // get position and orientation of linker
                Scalar4 postype_linker = h_postype.data[i_linker];
                vec3<Scalar> pos_linker = vec3<Scalar>(postype_linker);
                vec3<Scalar> pos_linker_after_move_primary_image = rotate(virtual_rotate_move, pos_linker - center_of_rotation) + center_of_rotation;
                vec3<Scalar> pos_linker_after_reverse_move_primary_image = rotate(conj(virtual_rotate_move), pos_linker - center_of_rotation) + center_of_rotation;
                positions_after_move[i_linker] = pos_linker_after_move_primary_image;

                int type_linker = __scalar_as_int(postype_linker.w);
                Shape shape_linker(quat<Scalar>(h_orientation.data[i_linker]), mc_params[type_linker]);

                // linker must be an active particle
                #ifdef ENABLE_MPI
                if (m_sysdef->isDomainDecomposed())
                    {
                    if (!isActive(make_scalar3(postype_linker.x, postype_linker.y, postype_linker.z), box, ghost_fraction))
                        {
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
                // TODO: do we *always* have to expand the box by the forward and reverse moves? could we save some time if we only search the reverse moves when necessary?
                hoomd::detail::AABB aabb_linker_local_no_moves = hoomd::detail::AABB(vec3<Scalar>(0, 0, 0), R_query);
                hoomd::detail::AABB aabb_linker_local_forward = hoomd::detail::AABB(pos_linker_after_move_primary_image - pos_linker, R_query);
                hoomd::detail::AABB aabb_linker_local_reverse = hoomd::detail::AABB(pos_linker_after_reverse_move_primary_image - pos_linker, R_query);
                hoomd::detail::AABB _aabb = hoomd::detail::merge(aabb_linker_local_no_moves, aabb_linker_local_forward);
                hoomd::detail::AABB aabb_linker_local = hoomd::detail::merge(_aabb, aabb_linker_local_reverse);

                // TODO: think about what to do for large rotations that put particles outside of the box

                // loop over all images
                const unsigned int n_images = (unsigned int)image_list.size();
                for (unsigned int current_image = 0; current_image < n_images; current_image++)
                    {
                    // create an AABB centered on the linker particle in the current box image
                    hoomd::detail::AABB aabb = aabb_linker_local;
                    vec3<Scalar> pos_linker_image = pos_linker + image_list[current_image];
                    vec3<Scalar> center_of_rotation_image = center_of_rotation + image_list[current_image];
                    aabb.translate(pos_linker_image);

                    // stackless search
                    for (unsigned int current_node_index = 0; current_node_index < mc_aabb_tree.getNumNodes(); current_node_index++)
                        {
                        if (aabb.overlaps(mc_aabb_tree.getNodeAABB(current_node_index)))
                            {
                            if (mc_aabb_tree.isNodeLeaf(current_node_index))
                                {
                                for (unsigned int current_linkee = 0; current_linkee < mc_aabb_tree.getNodeNumParticles(current_node_index); current_linkee++)
                                    {
                                    m_exec_conf->msg->notice(6) << "linker_set_size = " << linkers.size() << std::endl;
                                    unsigned int j_linkee = mc_aabb_tree.getNodeParticle(current_node_index, current_linkee);
                                    if ( j_linkee == i_linker || cluster.same(i_linker, j_linkee) )
                                        {
                                        // already tested this pair
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

                                    Shape shape_linker_after_move(
                                        quat<Scalar>(h_orientation.data[i_linker]) * virtual_rotate_move, mc_params[type_linker]);
                                    Shape shape_linker_after_reverse_move(quat<Scalar>(
                                        h_orientation.data[i_linker]) * conj(virtual_rotate_move), mc_params[type_linker]);

                                    m_exec_conf->msg->notice(5) << "Test for links: pair " << i_linker << " and " << j_linkee << std::endl;
                                    m_exec_conf->msg->notice(5) << "  Checking for overlap after moving linker between pair " << i_linker << " and " << j_linkee << std::endl;
                                    bool overlap_after_forward_move_of_linker =
                                        h_overlaps.data[m_mc->getOverlapIndexer()(type_linker, type_linkee)]
                                        && r_squared_after_forward_move_of_linker < max_overlap_distance * max_overlap_distance
                                        && test_overlap(r_linker_linkee_after_forward_move_of_linker, shape_linker_after_move, shape_linkee, counters.overlap_err_count);
                                    if (overlap_after_forward_move_of_linker)
                                        {
                                        m_exec_conf->msg->notice(5) << "    Overlap found between " << i_linker << " and " << j_linkee << " -> link formed!" << std::endl;

                                        // add linkee to cluster
                                        if (linkers_added.find(j_linkee) == linkers_added.end())
                                            {
                                            linkers.push_back(j_linkee);
                                            linkers_added.insert(j_linkee);
                                            linker_rotation_centers.push_back(center_of_rotation_image);
                                            cluster.unite(i_linker, j_linkee);
                                            positions_after_move[j_linkee] = virtual_translate_move;
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
                                            p_ij_reverse = std::max(0.0, 1 - exp(-m_beta_ficticious*(u_ij_after_reverse_move_of_linker - u_ij)));
                                            m_exec_conf->msg->notice(5) << "  u_ij = "  << u_ij << std::endl;
                                            m_exec_conf->msg->notice(5) << "  u_ij_after_reverse_move = "  << u_ij_after_reverse_move_of_linker << std::endl;
                                            m_exec_conf->msg->notice(5) << "  p_ij_reverse = "  << p_ij_reverse << std::endl;
                                            }
                                        p_link_probability_ratio *= p_ij_reverse;
                                        continue; // to next linkee
                                        } // end if (overlap_after_forward_move_of_linker)
                                    m_exec_conf->msg->notice(5) << "    No overlap after moving linker between " << i_linker << " and " << j_linkee << std::endl;

                                    // if no overlap, we have to calculate u_ij, u_ij_after_move and u_ij_after_reverse move
                                    // We also need to check for overlaps in the reverse move, before calculating u_ij_after_reverse_move
                                    // Calculate u_ij_after_move first to determine if there is a link
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
                                    m_exec_conf->msg->notice(5) << "  u_ij = "  << u_ij << std::endl;
                                    m_exec_conf->msg->notice(5) << "  u_ij_after_move = "  << u_ij_after_move << std::endl;

                                    LongReal p_ij_forward = std::max(0.0, 1 - exp(-m_beta_ficticious * (u_ij_after_move - u_ij)));
                                    Scalar r = hoomd::UniformDistribution<Scalar>()(rng_i);
                                    bool link_formed = r < p_ij_forward;
                                    if (link_formed)
                                        {
                                        m_exec_conf->msg->notice(5) << "  p_ij_forward = " << p_ij_forward << "; random number = " << r << " -> link_formed!" << std::endl;
                                        }
                                    else
                                        {
                                        m_exec_conf->msg->notice(5) << "  p_ij_forward = " << p_ij_forward << "; random number = " << r << " -> no link_formed" << std::endl;
                                        }

                                    if (link_formed)
                                        {
                                        // do all the stuff we need to do
                                        // add linkee to cluster
                                        if (linkers_added.find(j_linkee) == linkers_added.end())
                                            {
                                            linkers.push_back(j_linkee);
                                            linkers_added.insert(j_linkee);
                                            linker_rotation_centers.push_back(center_of_rotation_image);
                                            cluster.unite(i_linker, j_linkee);
                                            positions_after_move[j_linkee] = virtual_translate_move;
                                            }


                                        // numerator of p_link_probability_ratio gets multiplied by p_ij_reverse
                                        Scalar p_ij_reverse;
                                        bool overlap_after_reverse_move =
                                            h_overlaps.data[m_mc->getOverlapIndexer()(type_linker, type_linkee)]
                                            && r_squared_after_reverse_move_of_linker < max_overlap_distance * max_overlap_distance
                                            && test_overlap(r_linker_linkee_after_reverse_move_of_linker, shape_linker_after_move, shape_linkee, counters.overlap_err_count);
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
                                            p_ij_reverse = std::max(0.0, 1.0 - exp(-m_beta_ficticious * (u_ij_after_reverse_move - u_ij)));
                                            m_exec_conf->msg->notice(5) << "  p_ij_reverse = "  << p_ij_reverse << std::endl;
                                            }
                                        /* p_link_probability_ratio *= p_ij_reverse; */

                                        // denominator of p_link_probability_ratio gets multiplied by p_ij_forward
                                        /* p_link_probability_ratio /= p_ij_forward; */

                                        p_link_probability_ratio *= std::min(1.0, p_ij_reverse / p_ij_forward);
                                        continue;
                                        }
                                    else  // no link formed
                                        {
                                        // if these don't form a link, we still need their pair energies (all 3 mentioned above), but
                                        // we cannot do anything with them yet until we determine whether or not j ends up in the cluster
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
                                        Scalar q_ij_reverse = 1.0 - std::max(0.0, 1 - exp(-m_beta_ficticious * (u_ij - u_ij_after_move)));
                                        m_exec_conf->msg->notice(5) << "  q_ij_forward = "  << q_ij_forward << std::endl;
                                        m_exec_conf->msg->notice(5) << "  q_ij_reverse = "  << q_ij_reverse << std::endl;
                                        /* p_failed_link_probability_ratio *= q_ij_reverse / q_ij_forward; */
                                        p_failed_link_probability_ratio *= std::max(1.0, q_ij_reverse / q_ij_forward);

                                        // add to possible contributions to deltaU for the cluster move
                                        potential_intracluster_links.push_back(std::make_pair(i_linker, j_linkee));
                                        u_ij_potential_links.push_back(u_ij);
                                        u_ij_forward_potential_links.push_back(u_ij_after_move);
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
            for (unsigned int i = 0; i < potential_intracluster_links.size(); i++)
                {
                unsigned int linker_i = potential_intracluster_links[i].first;  // recall: linker_i in cluster by construction
                unsigned int linkee_j = potential_intracluster_links[i].second;
                m_exec_conf->msg->notice(10) << "  Checking failed, intracluster links between " << linker_i << " and " << linkee_j << std::endl;
                if (cluster.same(linker_i, linkee_j))
                    {
                    m_exec_conf->msg->notice(5) << "    particle with tag " << linkee_j << " eventually joined the cluster." << std::endl;
                    // j ultimately joined the cluster so there will be no contribution to deltaU for this i-j pair
                    // we have also already accounted for the probability of not forming that link in the forward and reverse moves
                    // so there is nothing to do here except for explicitly add 0.0 to beta_deltaU for the sake of code readability
                    beta_deltaU += 0.0;
                    }
                else
                    {
                    // j ultimately did not join the cluster, so there will in general be a contribution to beta_deltaU for this i-j pair
                    // we have already accounted for the probability of not forming the i-j link, so all we do is add to beta_deltaU
                    beta_deltaU += u_ij_forward_potential_links[i] - u_ij_potential_links[i];
                    m_exec_conf->msg->notice(5) << "    particle with tag " << linkee_j << " never joined the cluster (failed forming link with " << linker_i << ". beta_deltaU += " << u_ij_forward_potential_links[i] - u_ij_potential_links[i] << std::endl;
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
            size_t cluster_size = linkers_added.size();
            /* for(unsigned int idx = 0; idx < m_pdata->getN(); idx++) */
            /*     { */
            /*     if (cluster.same(seed_idx, idx)) */
            /*         { */
            /*         cluster_size++; */
            /*         } */
            /*     } */
            m_count_total.total_num_particles_in_clusters += (unsigned long long int)cluster_size;
            if(accept_cluster_move)
                {
                m_exec_conf->msg->notice(4) << "  VMMC move accepted, moving " << cluster_size << " particles as a cluster" << std::endl;
                if (move_type_translate)
                    {
                    m_count_total.translate_accept_count++;
                    }
                else
                    {
                    m_count_total.rotate_accept_count++;
                    }
                // apply the cluster move
                for(unsigned int idx = 0; idx < m_pdata->getN(); idx++)
                    {
                    if (cluster.same(seed_idx, idx))
                        {
                        Scalar4 postype_idx = h_postype.data[idx];
                        vec3<Scalar> new_pos = positions_after_move[idx];
                        h_postype.data[idx] = make_scalar4(new_pos.x, new_pos.y, new_pos.z, postype_idx.w);
                        quat<Scalar> orientation(h_orientation.data[idx]);
                        h_orientation.data[idx] = quat_to_scalar4(orientation * virtual_rotate_move);
                        if (idx == 0)
                            {
                            m_exec_conf->msg->notice(6) << "Setting h_orientation[" << idx << "] from: "
                                << orientation.s << " "
                                << orientation.v.x << " "
                                << orientation.v.y << " "
                                << orientation.v.z << " (norm**2 = " << norm2(orientation) << ") to: "
                                << h_orientation.data[idx].w << " "
                                << h_orientation.data[idx].x << " "
                                << h_orientation.data[idx].y << " "
                                << h_orientation.data[idx].z << " (norm**2 = " << norm2(orientation) << "); "
                                << "virtual rotate move =  " << virtual_rotate_move.s << " "
                                << virtual_rotate_move.v.x << " " << virtual_rotate_move.v.y << " " << virtual_rotate_move.v.z << " (norm**2 = "
                << norm2(virtual_rotate_move) << ")" << std::endl;
                            }
                        }
                    }

                // wrap particles back into box
                ArrayHandle<Scalar4> h_postype(m_pdata->getPositions(), access_location::host, access_mode::readwrite);
                ArrayHandle<int3> h_image(m_pdata->getImages(), access_location::host, access_mode::readwrite);
                for (unsigned int i = 0; i < m_pdata->getN(); i++)
                    {
                    box.wrap(h_postype.data[i], h_image.data[i]);
                    }

                m_mc->invalidateAABBTree();
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
                            std::shared_ptr<IntegratorHPMCMono<Shape>>
                            >())
        .def("getCounters", &UpdaterVMMC<Shape>::getCounters)
        .def_property("beta_ficticious", &UpdaterVMMC<Shape>::getBetaFicticious, &UpdaterVMMC<Shape>::setBetaFicticious)
        .def_property("translation_move_probability", &UpdaterVMMC<Shape>::getTranslationMoveProbability, &UpdaterVMMC<Shape>::setTranslationMoveProbability)
        .def_property("maximum_trial_rotation", &UpdaterVMMC<Shape>::getMaximumTrialRotation, &UpdaterVMMC<Shape>::setMaximumTrialRotation)
        .def_property("maximum_trial_translation", &UpdaterVMMC<Shape>::getMaximumTrialTranslation, &UpdaterVMMC<Shape>::setMaximumTrialTranslation)
        .def_property("attempts_per_particle", &UpdaterVMMC<Shape>::getAttemptsPerParticle, &UpdaterVMMC<Shape>::setAttemptsPerParticle)
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
    ;
    }


} // end namespace detail
} // end namespace hpmc
  //
} // end namespace hoomd

#endif // _UPDATER_HPMC_VIRTUAL_CLUSTER_MOVES_
