// Copyright (c) 2009-2024 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

#ifndef _UPDATER_HPMC_VIRTUAL_CLUSTER_MOVES_
#define _UPDATER_HPMC_VIRTUAL_CLUSTER_MOVES_

#include "hoomd/Updater.h"
#include "hoomd/RandomNumbers.h"
#include "hoomd/RNGIdentifiers.h"

#include <set>

#include "Moves.h"
#include "HPMCCounters.h"
#include "IntegratorHPMCMono.h"

#include "dset/dset.h"


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
        void setAttemptsPerParticle(Scalar attempts_per_particle)
            {
            m_attempts_per_particle = attempts_per_particle;
            }

        unsigned int getAttemptsPerParticle()
            {
            return m_attempts_per_particle;
            }

        /// Reset statistics counters
        virtual void resetStats()
            {
            m_count_run_start = m_count_total;
            }

        /*! \param mode 0 -> Absolute count, 1 -> relative to the start of the run, 2 -> relative to the last executed step
            \return The current state of the acceptance counters
        */
        hpmc_clusters_counters_t getCounters(unsigned int mode)
            {
            hpmc_clusters_counters_t result;

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

        hoomd::detail::AABBTree m_aabb_tree_old;              //!< Locality lookup for old configuration

        GlobalVector<Scalar4> m_postype_backup;        //!< Old local positions
        GlobalVector<Scalar4> m_orientation_backup;    //!< Old local orientations
        GlobalVector<int3> m_image_backup;             //!< Old local images

        // TODO: what all do we need to keep track of with counters?
        // 1. Number of cluster moves attempted and rejected
        // 2. Size of clusters?
        hpmc_clusters_counters_t m_count_total;                 //!< Total count since initialization
        hpmc_clusters_counters_t m_count_run_start;             //!< Count saved at run() start
        hpmc_clusters_counters_t m_count_step_start;            //!< Count saved at the start of the last step


        //! Attempt a single cluster move
        virtual void attemptOneClusterMove(uint64_t timestep);

        //! Generate a cluster
        virtual void generateCluster(std::vector<bool> &in_cluster);

        //! Acceptance test for a cluster move
        virtual bool acceptMove();

        //! Restore old configuration
        virtual void restoreState();

        //! Determine connected components of the interaction graph
        virtual void connectedComponents();

    };

template< class Shape >
UpdaterVMMC<Shape>::UpdaterVMMC(
        std::shared_ptr<SystemDefinition> sysdef,
        std::shared_ptr<Trigger> trigger,
        std::shared_ptr<IntegratorHPMCMono<Shape> > mc
        ) : Updater(sysdef, trigger), m_mc(mc), m_attempts_per_particle(1.0)
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
    Updater::update(timestep);
    #ifdef ENABLE_MPI
    if (this->m_pdata->getDomainDecomposition())
        throw std::runtime_error("UpdaterVMMC does not work with spatial domain decomposition.");
    #endif

    // if no particles, exit early
    if (! m_pdata->getNGlobal()) return;

    // get stuff needed for the calculation
    const LongReal min_core_radius = m_mc->getMinCoreDiameter() * LongReal(0.5);
    const auto& pair_energy_search_radius = m_mc->getPairEnergySearchRadius();
    ArrayHandle<Scalar4> h_postype(m_pdata->getPositions(), access_location::host, access_mode::readwrite);
    ArrayHandle<Scalar4> h_orientation(m_pdata->getOrientationArray(), access_location::host, access_mode::readwrite);
    ArrayHandle<Scalar> h_charge(m_pdata->getCharges(), access_location::host, access_mode::readwrite);
    ArrayHandle<Scalar> h_diameter(m_pdata->getDiameters(), access_location::host, access_mode::readwrite);
    ArrayHandle<Scalar> h_d(m_mc->m_d, access_location::host, access_mode::read);
    const BoxDim box = m_pdata->getBox();
    unsigned int ndim = this->m_sysdef->getNDimensions();
    uint16_t seed = m_sysdef->getSeed();

    #ifdef ENABLE_MPI
    // compute the width of the active region
    Scalar3 npd = box.getNearestPlaneDistance();
    Scalar3 ghost_fraction = m_mc->m_nominal_width / npd;
    #endif

    // Shuffle the order of particles for this step
    m_mc->m_update_order.resize(m_pdata->getN());
    m_mc->m_update_order.shuffle(timestep, m_sysdef->getSeed(), m_exec_conf->getRank());

    // update the AABB Tree
    m_mc->buildAABBTree();
    // limit m_d entries so that particles cannot possibly wander more than one box image in one time step
    m_mc->limitMoveDistances();
    // update the image list
    m_mc->updateImageList();

    // access interaction matrix
    ArrayHandle<unsigned int> h_overlaps(m_mc->m_overlaps, access_location::host, access_mode::read);

    // counters
    ArrayHandle<hpmc_counters_t> h_counters(m_mc->m_count_total, access_location::host, access_mode::readwrite);
    hpmc_counters_t& counters = h_counters.data[0];


    for (unsigned int i_trial; i_trial < m_attempts_per_particle; i_trial++)
        {

        for (unsigned int current_seed = 0; current_seed < m_pdata->getN(); current_seed++)
            {
            // set of all "linker" particles that attempt to create links to "linkee" particles
            std::set<unsigned int> linkers;
            std::vector<std::pair<unsigned int, unsigned int>> links;
            std::vector<std::pair<unsigned int, unsigned int>> potential_intracluster_links;
            DisjointSets cluster;  // tags of particles in the cluster
            LongReal p_link_probability_ratio = 1.0;  // 3rd product in eqn 13 from the paper
            LongReal p_failed_link_probability_ratio = 1.0;  // 1st product in eqn 13 from the paper
            LongReal p_n_o = 1.0;  // product of boltzmann factors in new state for particles that go from non-interacting to interacting after the cluster move
            std::vector<double> u_ij_potential_links;
            std::vector<double> u_ij_forward_potential_links;
            std::vector<double> u_ij_reverse_potential_links;


            // seed particle that starts the virtual move
            unsigned int seed_idx = m_mc->m_update_order[current_seed];
            linkers.insert(seed_idx);

            // get position and orientation of seed
            Scalar4 postype_seed = h_postype.data[seed_idx];
            vec3<Scalar> pos_seed = vec3<Scalar>(postype_seed);
            int type_seed = __scalar_as_int(postype_seed.w);

            // generate the virtual move
            // TODO: enable rotations
            hoomd::RandomGenerator rng_i(hoomd::Seed(hoomd::RNGIdentifier::UpdaterVMMC, timestep, seed),
                                         hoomd::Counter(seed_idx, m_exec_conf->getRank(), i_trial));
            vec3<Scalar> virtual_move(0, 0, 0);
            move_translate(virtual_move, rng_i, h_d.data[type_seed], ndim);

            // make sure virtual move doesn't put particle into the inactive region
            #ifdef ENABLE_MPI
            if (m_sysdef->isDomainDecomposed())
                {
                // check if particle has moved into the ghost layer, and skip if it is
                if (!isActive(vec_to_scalar3(pos_seed + virtual_move), box, ghost_fraction))
                    continue;
                }
            #endif

            // while(linkers.size(); take particle i from linkers, loop over neighbors of i
            // loop over neighbors of cluster members to find new cluster members
            while (linkers.size())
                {
                unsigned int i_linker = linkers.extract(linkers.begin()).value();

                // get position and orientation of linker
                Scalar4 postype_linker = h_postype.data[i_linker];
                vec3<Scalar> pos_linker = vec3<Scalar>(postype_linker);
                int type_linker = __scalar_as_int(postype_linker.w);
                Shape shape_linker(quat<LongReal>(h_orientation.data[i_linker]), m_mc->m_params[type_linker]);

                // linker must be an active particle
                #ifdef ENABLE_MPI
                if (m_sysdef->isDomainDecomposed())
                    {
                    if (!isActive(make_scalar3(postype_linker.x, postype_linker.y, postype_linker.z), box, ghost_fraction))
                        continue;
                    }
                #endif

                // search for all particles that might touch this one
                LongReal R_query = m_mc->m_shape_circumsphere_radius[type_linker];

                if (m_mc->hasPairInteractions())
                    {
                    // Extend the search to include the pair interaction r_cut
                    // subtract minimum AABB extent from search radius
                    R_query = std::max(R_query, pair_energy_search_radius[type_linker] - min_core_radius);
                    }
                hoomd::detail::AABB aabb_linker_local = hoomd::detail::AABB(vec3<Scalar>(0, 0, 0), R_query);
                hoomd::detail::AABB aabb_linker_local_forward = hoomd::detail::AABB(virtual_move, R_query);
                hoomd::detail::AABB aabb_linker_local_reverse = hoomd::detail::AABB(-virtual_move, R_query);
                hoomd::detail::AABB _aabb = hoomd::detail::merge(aabb_linker_local, aabb_linker_local_forward);
                hoomd::detail::AABB aabb = hoomd::detail::merge(_aabb, aabb_linker_local_reverse);

                // loop over all images
                const unsigned int n_images = (unsigned int)m_mc->m_image_list.size();
                for (unsigned int current_image = 0; current_image < n_images; current_image++)
                    {
                    vec3<Scalar> pos_linker_image = pos_linker + m_mc->m_image_list[current_image];
                    aabb.translate(pos_linker_image);

                    // stackless search
                    for (unsigned int current_node_index = 0; current_node_index < m_mc->m_aabb_tree.getNumNodes(); current_node_index++)
                        {
                        if (aabb.overlaps(m_mc->m_aabb_tree.getNodeAABB(current_node_index)))
                            {
                            if (m_mc->m_aabb_tree.isNodeLeaf(current_node_index))
                                {
                                for (unsigned int current_linkee = 0; current_linkee < m_mc->m_aabb_tree.getNodeNumParticles(current_node_index); current_linkee++)
                                    {
                                    unsigned int j_linkee = m_mc->m_aabb_tree.getNodeParticle(current_node_index, current_linkee);
                                    Scalar4 postype_linkee;
                                    quat<LongReal> orientation_linkee;
                                    if ( j_linkee != i_linker )
                                        {
                                        postype_linkee = h_postype.data[j_linkee];
                                        orientation_linkee = quat<LongReal>(h_orientation.data[j_linkee]);
                                        }
                                    else
                                        {
                                        // skip since particle i is already in the cluster
                                        continue;
                                        }

                                    unsigned int type_linkee = __scalar_as_int(postype_linkee.w);
                                    Shape shape_linkee(orientation_linkee, m_mc->params[type_linkee]);
                                    LongReal max_overlap_distance = m_mc->m_shape_circumsphere_radius[type_linker] + m_mc->m_shape_circumsphere_radius[type_linkee];

                                    // at this point we can test for a link between the linker and linkee
                                    // add j to the cluster with probability p_ij = 1 - exp(E_C - E_I) (hard overlap: p_ij = 1)
                                    // so we need E_C (current pair energy) and E_I (pair energy after applying virtual move to i)
                                    // we only need E_C if E_I, so we can check for overlaps first

                                    // pair separation in current real configuration
                                    vec3<Scalar> r_linker_linkee = vec3<Scalar>(postype_linkee) - pos_linker_image;
                                    LongReal r_squared = dot(r_linker_linkee, r_linker_linkee);

                                    // pair separation after applying virtual move to linker (i)
                                    vec3<Scalar> r_linker_linkee_after_move = vec3<Scalar>(postype_linkee) - pos_linker_image + virtual_move;
                                    LongReal r_squared_after_move = dot(r_linker_linkee_after_move, r_linker_linkee_after_move);

                                    // pair separation after applying reverse virtual move to linker (i)
                                    vec3<Scalar> r_linker_linkee_after_reverse_move = vec3<Scalar>(postype_linkee) - pos_linker_image - virtual_move;
                                    LongReal r_squared_after_reverse_move = dot(r_linker_linkee_after_reverse_move, r_linker_linkee_after_reverse_move);

                                    bool overlap_after_move = 
                                        h_overlaps.data[m_mc->m_overlap_idx(type_linker, type_linkee)]
                                        && r_squared_after_move < max_overlap_distance * max_overlap_distance
                                        && test_overlap(r_linker_linkee_after_move, shape_linker, shape_linkee, counters.overlap_err_count);
                                    if (overlap_after_move)
                                        {
                                        // do everything needed for when there is a link, which is:
                                        //     1) numerator of p_link_probability_ratio remains unchanged (actually multiplied by 1.0)
                                        //     2) add j to the list of potential linkers
                                        //     3) calculate u_ij after applying the reverse move to particle i (TODO: must we do that here?) and calculate p_ij_reverse
                                        linkers.insert(j_linkee);
                                        cluster.unite(i_linker, j_linkee);
                                        links.push_back(std::make_pair(i_linker, j_linkee));
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
                                        // TODO: both forward and reverse moves can induce overlaps for nonconvex particles. handle that case.
                                        double u_ij_after_reverse_move =
                                            m_mc->computeOnePairEnergy(r_squared_after_reverse_move,
                                                    r_linker_linkee_after_reverse_move,
                                                    type_linker,
                                                    shape_linker.orientation,
                                                    h_diameter.data[i_linker],
                                                    h_charge.data[i_linker],
                                                    type_linkee,
                                                    shape_linkee.orientation,
                                                    h_diameter.data[j_linkee],
                                                    h_charge.data[j_linkee]);
                                        Scalar p_ij_reverse = 1 - exp(u_ij - u_ij_after_reverse_move);
                                        p_link_probability_ratio *= p_ij_reverse;
                                        continue;
                                        }

                                    // if no overlap, we have to calculate u_ij, u_ij_after_move and u_ij_after_reverse move
                                    // We also need to check for overlaps in the reverse move, before calculating u_ij_after_reverse_move 
                                    // Calculate u_ij_after_move first to determine if there is a link
                                    double u_ij_after_move =
                                        m_mc->computeOnePairEnergy(r_squared_after_move,
                                                r_linker_linkee_after_move,
                                                type_linker,
                                                shape_linker.orientation,
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
                                    LongReal p_ij_forward = 1 - exp(u_ij - u_ij_after_move);
                                    Scalar r = hoomd::UniformDistribution<Scalar>()(rng_i);
                                    bool link_formed = r <= p_ij_forward;
                                    if (link_formed)
                                        {
                                        // do all the stuff we need to do
                                        cluster.unite(i_linker, j_linkee);
                                        linkers.insert(j_linkee);
                                        links.push_back(std::make_pair(i_linker, j_linkee));
                                        p_link_probability_ratio /= p_ij_forward;
                                        Scalar p_ij_reverse;
                                        bool overlap_after_reverse_move = 
                                            h_overlaps.data[m_mc->m_overlap_idx(type_linker, type_linkee)]
                                            && r_squared_after_reverse_move < max_overlap_distance * max_overlap_distance
                                            && test_overlap(r_linker_linkee_after_reverse_move, shape_linker, shape_linkee, counters.overlap_err_count);
                                        if (overlap_after_reverse_move)
                                            {
                                            p_ij_reverse = 1.0;
                                            }
                                        else
                                            {
                                            double u_ij_after_reverse_move =
                                                m_mc->computeOnePairEnergy(r_squared_after_reverse_move,
                                                        r_linker_linkee_after_reverse_move,
                                                        type_linker,
                                                        shape_linker.orientation,
                                                        h_diameter.data[i_linker],
                                                        h_charge.data[i_linker],
                                                        type_linkee,
                                                        shape_linkee.orientation,
                                                        h_diameter.data[j_linkee],
                                                        h_charge.data[j_linkee]);
                                            p_ij_reverse = 1.0 - exp(u_ij - u_ij_after_reverse_move);
                                            }
                                        p_link_probability_ratio *= p_ij_reverse;
                                        }
                                    else
                                        {
                                        // if these don't form a link, we still need their pair energies (all 3 mentioned above), but 
                                        // we cannot do anything with them yet until we determine whether or not j ends up in the cluster
                                        // if it does; then we have to account for the probabilities of not finding links
                                        // if j doesn't end up in the cluster then there's nothing left to do
                                        double u_ij_after_reverse_move =
                                            m_mc->computeOnePairEnergy(r_squared_after_reverse_move,
                                                    r_linker_linkee_after_reverse_move,
                                                    type_linker,
                                                    shape_linker.orientation,
                                                    h_diameter.data[i_linker],
                                                    h_charge.data[i_linker],
                                                    type_linkee,
                                                    shape_linkee.orientation,
                                                    h_diameter.data[j_linkee],
                                                    h_charge.data[j_linkee]);
                                        potential_intracluster_links.push_back(std::make_pair(i_linker, j_linkee));
                                        u_ij_potential_links.push_back(u_ij);
                                        u_ij_forward_potential_links.push_back(u_ij_after_move);
                                        u_ij_reverse_potential_links.push_back(u_ij_after_reverse_move);
                                        }
                                    }  // end loop over linkees
                                }  // end if (m_mc->m_aabb_tree.isNodeLeaf(current_node_index))
                            }  // end if (aabb.overlaps(m_mc->m_aabb_tree.getNodeAABB(current_node_index)))
                        else
                            {
                            // skip ahead
                            current_node_index += m_mc->m_aabb_tree.getNodeSkip(current_node_index);
                            }
                        }  // end loop over nodes in AABBTree
                    }  // end loop over images
                }  // end loop over linkers

            // find which of the potential cluster members ended up in the cluster
            for (unsigned int i; i < potential_intracluster_links.size(); i++)
                {
                unsigned int linker_i = potential_intracluster_links[i].first;
                unsigned int linker_j = potential_intracluster_links[i].second;
                if (cluster.same(linker_i, linker_j))
                    {
                    // failed intracluster link
                    Scalar q_ij_forward = exp(u_ij_potential_links[i] - u_ij_forward_potential_links[i]);
                    Scalar q_ij_reverse = exp(u_ij_potential_links[i] - u_ij_reverse_potential_links[i]);
                    p_failed_link_probability_ratio *= q_ij_reverse / q_ij_forward;
                    }
                }

            // now we can accept or reject the move
            LongReal p_acc = p_n_o * p_failed_link_probability_ratio * p_link_probability_ratio;
            Scalar r = hoomd::UniformDistribution<Scalar>()(rng_i);
            bool accept_cluster_move = r <= p_acc;
            if(accept_cluster_move)
                {
                // apply the cluster moves to all of the particles
                for(unsigned int idx; idx < m_pdata->getN(); idx++)
                    {
                    if (cluster.same(seed_idx, idx))
                        {
                        Scalar4 postype_idx = h_postype.data[idx];
                        vec3<Scalar> new_pos = vec3<Scalar>(postype_idx) + virtual_move;
                        h_postype.data[idx] = make_scalar4(new_pos.x, new_pos.y, new_pos.z, postype_idx.w);
                        }
                    }
                }


            }  // end loop over seed particles
        }  // end loop over trials
    }  // end update()


namespace detail {

template < class Shape> void export_UpdaterVirtualMoveMonteCarlo(pybind11::module& m, const std::string& name)
    {
    pybind11::class_< UpdaterVMMC<Shape>, Updater, std::shared_ptr< UpdaterVMMC<Shape> > >(m, name.c_str())
        .def(
            pybind11::init< std::shared_ptr<SystemDefinition>,
            std::shared_ptr<Trigger>, std::shared_ptr<
            IntegratorHPMCMono<Shape> > >()) .def("getCounters",
            &UpdaterVMMC<Shape>::getCounters
        )
        .def_property("attempts_per_particle", &UpdaterVMMC<Shape>::getAttemptsPerParticle, &UpdaterVMMC<Shape>::setAttemptsPerParticle)
    ;
    }

inline void export_hpmc_clusters_counters(pybind11::module &m)
    {
    pybind11::class_< hpmc_clusters_counters_t >(m, "hpmc_clusters_counters_t")
        .def_property_readonly("average_cluster_size", &hpmc_clusters_counters_t::getAverageClusterSize)
        ;
    }

} // end namespace detail
} // end namespace hpmc
  //
} // end namespace hoomd

#endif // _UPDATER_HPMC_VIRTUAL_CLUSTER_MOVES_
