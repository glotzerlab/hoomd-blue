// inclusion guard
#ifndef _UPDATER_HPMC_CLUSTERS_
#define _UPDATER_HPMC_CLUSTERS_

/*! \file UpdaterBoxClusters.h
    \brief Declaration of UpdaterBoxClusters
*/

#include "hoomd/HOOMDMPI.h"
#include "hoomd/Updater.h"
#include "hoomd/RandomNumbers.h"
#include "hoomd/RNGIdentifiers.h"

#include <set>
#include <list>

#include "Moves.h"
#include "HPMCCounters.h"
#include "IntegratorHPMCMono.h"

#ifdef ENABLE_TBB_TASK
#include <tbb/parallel_for.h>
#include <tbb/concurrent_vector.h>
#include <tbb/concurrent_unordered_set.h>
#include <tbb/concurrent_unordered_map.h>
#include <atomic>
#endif

namespace hpmc
{

namespace detail
{

#ifdef ENABLE_TBB_TASK
//! Wrapper around std::atomic_flag to allow use in a std::vector
class my_atomic_flag
    {
    public:
        //! Default constructor
        my_atomic_flag()
            {
            f.clear();
            }

        //! Copy constructor
        /*! \note this constructor doesn't really copy the value of its argument,
            it just reset's the flag to zero
         */
        my_atomic_flag(const my_atomic_flag& other)
            {
            f.clear();
            }

        //! Assignment operator (non-atomic)
        /*! \note this assignment operator doesn't really copy the value of its argument,
            it just reset's the flag to zero
         */
        my_atomic_flag& operator =( const my_atomic_flag& other)
            {
            f.clear();
            return *this;
            }

        //! Sets flag and returns old value
        bool test_and_set()
            {
            return f.test_and_set();
            }

    private:
        std::atomic_flag f;
    };
#endif

// Graph class represents a undirected graph
// using adjacency list representation
class Graph
    {
    public:
        Graph() {}      //!< Default constructor

        inline Graph(unsigned int V);   // Constructor

        inline void resize(unsigned int V);

        inline void addEdge(unsigned int v, unsigned int w);

        #ifdef ENABLE_TBB_TASK
        inline void connectedComponents(std::vector<tbb::concurrent_vector<unsigned int> >& cc);
        #else
        inline void connectedComponents(std::vector<std::vector<unsigned int> >& cc);
        #endif

    private:
        #ifndef ENABLE_TBB_TASK
        std::multimap<unsigned int,unsigned int> adj;
        #else
        tbb::concurrent_unordered_multimap<unsigned int, unsigned int> adj;
        #endif

        #ifndef ENABLE_TBB_TASK
        std::vector<unsigned int> visited;
        #else
        std::vector<my_atomic_flag> visited;
        #endif

        #ifndef ENABLE_TBB_TASK
        // A function used by DFS
        inline void DFSUtil(unsigned int v, std::vector<unsigned int>& visited, std::vector<unsigned int>& cur_cc);
        #endif

        #ifdef ENABLE_TBB_TASK
        class DFSTask : public tbb::task
            {
            public:
                DFSTask(unsigned int _root, std::vector<my_atomic_flag>& _visited,
                    tbb::concurrent_vector<unsigned int>& _cc,
                    const tbb::concurrent_unordered_multimap<unsigned int, unsigned int>& _adj)
                    : root(_root), visited(_visited), cc(_cc), adj(_adj)
                    { }

                tbb::task* execute()
                    {
                    cc.push_back(root);

                    unsigned int count = 0;
                    tbb::task_list list;

                    auto begin = adj.equal_range(root).first;
                    auto end = adj.equal_range(root).second;

                    for (auto it = begin; it != end; ++it)
                        {
                        unsigned int neighbor = it->second;
                        if (!visited[neighbor].test_and_set())
                            {
                            if (count++ == 0)
                                {
                                root = neighbor; // for task recycling
                                continue;
                                }
                            list.push_back(*new(allocate_child()) DFSTask(neighbor, visited, cc, adj));
                            }
                        }

                    if (count)
                        {
                        set_ref_count(count);
                        spawn(list);

                        recycle_as_safe_continuation();
                        }

                    return NULL;
                    }

            private:
                unsigned int root;
                std::vector<my_atomic_flag> & visited;
                tbb::concurrent_vector<unsigned int>& cc;
                const tbb::concurrent_unordered_multimap<unsigned int, unsigned int>& adj;
            };
        #endif // ENABLE_TBB_TASK

    };

// Gather connected components in an undirected graph
#ifdef ENABLE_TBB_TASK
void Graph::connectedComponents(std::vector<tbb::concurrent_vector<unsigned int> >& cc)
#else
void Graph::connectedComponents(std::vector<std::vector<unsigned int> >& cc)
#endif
    {
    #ifdef ENABLE_TBB_TASK
    for (unsigned int v = 0; v < visited.size(); ++v)
        {
        if (! visited[v].test_and_set())
            {
            tbb::concurrent_vector<unsigned int> component;
            DFSTask& a = *new(tbb::task::allocate_root()) DFSTask(v, visited, component, adj);
            tbb::task::spawn_root_and_wait(a);
            cc.push_back(component);
            }
        }
    #else
    std::fill(visited.begin(), visited.end(), 0);

    // Depth first search
    for (unsigned int v=0; v<visited.size(); v++)
        {
        if (visited[v] == false)
            {
            std::vector<unsigned int> cur_cc;
            DFSUtil(v, visited, cur_cc);
            cc.push_back(cur_cc);
            }
        }
    #endif
    }

#ifndef ENABLE_TBB_TASK
void Graph::DFSUtil(unsigned int v, std::vector<unsigned int>& visited, std::vector<unsigned int>& cur_cc)
    {
    visited[v] = 1;
    cur_cc.push_back(v);

    // Recur for all the vertices
    // adjacent to this vertex
    auto begin = adj.equal_range(v).first;
    auto end = adj.equal_range(v).second;
    for(auto i = begin; i != end; ++i)
        {
        if(!visited[i->second])
            DFSUtil(i->second, visited, cur_cc);
        }
    }
#endif
Graph::Graph(unsigned int V)
    {
    #ifndef ENABLE_TBB_TASK
    visited.resize(V, 0);
    #else
    visited.resize(V);
    #endif
    }

void Graph::resize(unsigned int V)
    {
    #ifndef ENABLE_TBB_TASK
    visited.resize(V, 0);
    #else
    visited.clear();
    visited.resize(V);
    #endif

    adj.clear();
    }

// method to add an undirected edge
void Graph::addEdge(unsigned int v, unsigned int w)
    {
    adj.insert(std::make_pair(v,w));
    adj.insert(std::make_pair(w,v));
    }
} // end namespace detail

/*! A generic cluster move for attractive interactions.

    The cluster move set employed consists of pivot (point mirroring) and
    line reflection (pi rotation) moves. The algorithm therefore implements
    a simplified version of the Geometric Cluster algorithm, cf. Liu and Luijten
    PRL 2004 and Sinkovits, Barr and Luijten JCP 2012.

    In order to support anisotropic particles, we have to reject moves that
    cross the PBC, as described in Sinkovits et al.
*/

template< class Shape >
class UpdaterClusters : public Updater
    {
    public:
        //! Constructor
        /*! \param sysdef System definition
            \param mc HPMC integrator
            \param seed PRNG seed
        */
        UpdaterClusters(std::shared_ptr<SystemDefinition> sysdef,
                        std::shared_ptr<IntegratorHPMCMono<Shape> > mc,
                        unsigned int seed);

        //! Destructor
        virtual ~UpdaterClusters();

        //! Get the value of a logged quantity
        virtual Scalar getLogValue(const std::string& quantity, unsigned int timestep)
            {
            hpmc_clusters_counters_t counters = getCounters(2);

            if (quantity == "hpmc_clusters_moves")
                {
                hpmc_clusters_counters_t counters_total = getCounters(0);
                return double(counters_total.getNParticlesMoved()) / double(m_pdata->getNGlobal());
                }
            else if (quantity == "hpmc_clusters_pivot_acceptance")
                {
                return counters.getPivotAcceptance();
                }
            else if (quantity == "hpmc_clusters_reflection_acceptance")
                {
                return counters.getReflectionAcceptance();
                }
            else if (quantity == "hpmc_clusters_swap_acceptance")
                {
                return counters.getSwapAcceptance();
                }
            else if (quantity == "hpmc_clusters_avg_size")
                {
                return counters.getAverageClusterSize();
                }
            return Scalar(0.0);
            }

        /*
            \returns a list of provided quantities
        */
        std::vector< std::string > getProvidedLogQuantities()
            {
            // start with the integrator provided quantities
            std::vector< std::string > result;
            // then add ours
            result.push_back("hpmc_clusters_moves");
            result.push_back("hpmc_clusters_pivot_acceptance");
            result.push_back("hpmc_clusters_reflection_acceptance");
            result.push_back("hpmc_clusters_swap_acceptance");
            result.push_back("hpmc_clusters_avg_size");
            return result;
            }

        //! Take one timestep forward
        /*! \param timestep timestep at which update is being evaluated
        */
        virtual void update(unsigned int timestep);

        //! Set the move ratio
        void setMoveRatio(Scalar move_ratio)
            {
            m_move_ratio = move_ratio;
            }

        //! Set the swap to geometric move ratio
        void setSwapMoveRatio(Scalar move_ratio)
            {
            m_swap_move_ratio = move_ratio;
            }

        //! Set the cluster flip probability
        void setFlipProbability(Scalar flip_probability)
            {
            m_flip_probability = flip_probability;
            }


        //! Set an AB type pair to be used with type swap moves
        /*! \param type_A first type
            \param type_B second type
            \param delta_mu difference in chemical potentials
         */
        void setSwapTypePair(unsigned int type_A, unsigned int type_B)
            {
            m_ab_types.resize(2);
            m_ab_types[0] = type_A;
            m_ab_types[1] = type_B;
            }

        //! Set the difference in chemical potential mu_B - mu_A
        void setDeltaMu(Scalar delta_mu)
            {
            m_delta_mu = delta_mu;
            }

        //! Reset statistics counters
        virtual void resetStats()
            {
            m_count_run_start = m_count_total;
            }

        //! Print statistics about the cluster move updates
        /* We only print the statistics about accepted and rejected moves.
         */
        void printStats()
            {
            hpmc_clusters_counters_t counters = getCounters(1);
            m_exec_conf->msg->notice(2) << "-- HPMC cluster move stats:" << std::endl;
            if (counters.pivot_accept_count + counters.pivot_reject_count != 0)
                {
                m_exec_conf->msg->notice(2) << "Average pivot acceptance:      " << counters.getPivotAcceptance() << std::endl;
                }
            if (counters.reflection_accept_count + counters.reflection_reject_count != 0)
                {
                m_exec_conf->msg->notice(2) << "Average reflection acceptance: " << counters.getReflectionAcceptance() << std::endl;
                }
            if (counters.swap_accept_count + counters.swap_reject_count != 0)
                {
                m_exec_conf->msg->notice(2) << "Average swap acceptance:       " << counters.getSwapAcceptance() << std::endl;
                }
            m_exec_conf->msg->notice(2) <<     "Total particles in clusters:   " << counters.getNParticlesInClusters() << std::endl;
            m_exec_conf->msg->notice(2) <<     "Total particles moved:         " << counters.getNParticlesMoved() << std::endl;
            m_exec_conf->msg->notice(2) <<     "Average cluster size:          " << counters.getAverageClusterSize() << std::endl;
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

                #ifdef ENABLE_MPI
                if (m_pdata->getDomainDecomposition())
                    {
                    bcast(result.pivot_accept_count,0,m_exec_conf->getMPICommunicator());
                    bcast(result.reflection_accept_count,0,m_exec_conf->getMPICommunicator());
                    bcast(result.swap_accept_count,0,m_exec_conf->getMPICommunicator());
                    bcast(result.pivot_reject_count,0,m_exec_conf->getMPICommunicator());
                    bcast(result.reflection_reject_count,0,m_exec_conf->getMPICommunicator());
                    bcast(result.swap_reject_count,0,m_exec_conf->getMPICommunicator());
                    bcast(result.n_clusters,0,m_exec_conf->getMPICommunicator());
                    bcast(result.n_particles_in_clusters,0,m_exec_conf->getMPICommunicator());
                    }
                #endif

                return result;
                }


    protected:
        std::shared_ptr< IntegratorHPMCMono<Shape> > m_mc; //!< HPMC integrator
        unsigned int m_seed;                        //!< RNG seed
        Scalar m_move_ratio;                        //!< Pivot/Reflection move ratio
        Scalar m_swap_move_ratio;                   //!< Type swap / geometric move ratio
        Scalar m_flip_probability;                  //!< Cluster flip probability

        #ifdef ENABLE_TBB_TASK
        std::vector<tbb::concurrent_vector<unsigned int> > m_clusters; //!< Cluster components
        #else
        std::vector<std::vector<unsigned int> > m_clusters; //!< Cluster components
        #endif

        detail::Graph m_G; //!< The graph

        unsigned int m_n_particles_old;                //!< Number of local particles in the old configuration
        detail::AABBTree m_aabb_tree_old;              //!< Locality lookup for old configuration
        std::vector<Scalar4> m_postype_backup;         //!< Old local positions
        std::vector<Scalar4> m_orientation_backup;     //!< Old local orientations
        std::vector<Scalar> m_diameter_backup;         //!< Old local diameters
        std::vector<Scalar> m_charge_backup;           //!< Old local charges
        std::vector<int3> m_image_backup;              //!< Old local images

        std::vector<unsigned int> m_tag_backup;             //!< Old local tags

        #ifndef ENABLE_TBB_TASK
        std::vector<std::pair<unsigned int, unsigned int> > m_overlap;   //!< A local vector of particle pairs due to overlap
        std::vector<std::pair<unsigned int, unsigned int> > m_interact_old_old;  //!< Pairs interacting old-old
        std::vector<std::pair<unsigned int, unsigned int> > m_interact_new_old;  //!< Pairs interacting new-old

        std::set<std::pair<unsigned int, unsigned int> > m_interact_new_new;  //!< Pairs interacting new-old
        std::set<unsigned int> m_local_reject;                   //!< Set of particles whose clusters moves are rejected

        std::map<std::pair<unsigned int, unsigned int>,float > m_energy_old_old;    //!< Energy of interaction old-old
        std::map<std::pair<unsigned int, unsigned int>,float > m_energy_new_old;    //!< Energy of interaction old-old
        std::set<unsigned int> m_ptl_reject;              //!< List of ptls that are not transformed
        #else
        tbb::concurrent_vector<std::pair<unsigned int, unsigned int> > m_overlap;
        tbb::concurrent_vector<std::pair<unsigned int, unsigned int> > m_interact_old_old;
        tbb::concurrent_vector<std::pair<unsigned int, unsigned int> > m_interact_new_old;

        tbb::concurrent_unordered_set<std::pair<unsigned int, unsigned int> > m_interact_new_new;
        tbb::concurrent_unordered_set<unsigned int> m_local_reject;

        tbb::concurrent_unordered_map<std::pair<unsigned int, unsigned int>,float > m_energy_old_old;
        tbb::concurrent_unordered_map<std::pair<unsigned int, unsigned int>,float > m_energy_new_old;
        tbb::concurrent_unordered_set<unsigned int> m_ptl_reject;              //!< List of ptls that are not transformed
        #endif

        #ifdef ENABLE_TBB_TASK
        tbb::concurrent_vector<vec3<Scalar> > m_random_position;
        tbb::concurrent_vector<quat<Scalar> > m_random_orientation;
        #else
        std::vector<Scalar3> m_random_position;      //!< Internal list of coordinates for Gibbs sampler
        std::vector<quat<Scalar> > m_random_orientation; //!< Internal list of orientations
        #endif

        std::vector<unsigned int> m_ab_types;          //!< Two types used for swap move
        Scalar m_delta_mu;                             //!< Difference in chemical potential

        hpmc_clusters_counters_t m_count_total;                 //!< Total count since initialization
        hpmc_clusters_counters_t m_count_run_start;             //!< Count saved at run() start
        hpmc_clusters_counters_t m_count_step_start;            //!< Count saved at the start of the last step

        //! Find interactions between particles due to overlap and depletion interaction
        /*! \param timestep Current time step
            \param pivot The current pivot point
            \param q The current line reflection axis
            \param line True if this is a line reflection
            \param map Map to lookup new tag from old tag
        */
        virtual void findInteractions(unsigned int timestep, vec3<Scalar> pivot, quat<Scalar> q, bool swap,
            bool line, const std::map<unsigned int, unsigned int>& map);

        //! Helper function to get interaction range
        virtual Scalar getNominalWidth()
            {
            Scalar nominal_width = m_mc->getMaxCoreDiameter();
            auto patch = m_mc->getPatchInteraction();
            if (patch)
                {
                Scalar max_extent = 0.0;
                for (unsigned int typ = 0; typ < this->m_pdata->getNTypes(); typ++)
                    {
                    max_extent = std::max(max_extent, patch->getAdditiveCutoff(typ));
                    }

                nominal_width = std::max(nominal_width, max_extent+patch->getRCut());
                }
            return nominal_width;
            }
    };

template< class Shape >
UpdaterClusters<Shape>::UpdaterClusters(std::shared_ptr<SystemDefinition> sysdef,
                                 std::shared_ptr<IntegratorHPMCMono<Shape> > mc,
                                 unsigned int seed)
        : Updater(sysdef), m_mc(mc), m_seed(seed), m_move_ratio(0.5), m_swap_move_ratio(0.5),
            m_flip_probability(0.5), m_n_particles_old(0), m_delta_mu(0.0)
    {
    m_exec_conf->msg->notice(5) << "Constructing UpdaterClusters" << std::endl;

    // initialize logger and stats
    resetStats();
    }

template< class Shape >
UpdaterClusters<Shape>::~UpdaterClusters()
    {
    m_exec_conf->msg->notice(5) << "Destroying UpdaterClusters" << std::endl;
    }

template< class Shape >
void UpdaterClusters<Shape>::findInteractions(unsigned int timestep, vec3<Scalar> pivot, quat<Scalar> q, bool swap,
    bool line, const std::map<unsigned int, unsigned int>& map)
    {
    if (m_prof) m_prof->push(m_exec_conf,"Interactions");

    // access parameters
    auto& params = m_mc->getParams();

    // update the image list
    auto image_list = m_mc->updateImageList();

    // minimum AABB extent
    Scalar min_core_diameter = m_mc->getMinCoreDiameter();

    Index2D overlap_idx = m_mc->getOverlapIndexer();
    ArrayHandle<unsigned int> h_overlaps(m_mc->getInteractionMatrix(), access_location::host, access_mode::read);

    // clear the local bond and rejection lists
    m_overlap.clear();
    m_interact_old_old.clear();
    m_interact_new_old.clear();
    m_interact_new_new.clear();
    m_local_reject.clear();

    auto patch = m_mc->getPatchInteraction();

    Scalar r_cut_patch(0.0);
    if (patch)
        {
        m_energy_old_old.clear();
        m_energy_new_old.clear();
        r_cut_patch = patch->getRCut();
        }

    // cluster according to overlap of excluded volume shells
    // loop over local particles
    unsigned int nptl = m_pdata->getN();

    // locality data in new configuration
    const detail::AABBTree& aabb_tree = m_mc->buildAABBTree();

    // access particle data
    ArrayHandle<Scalar4> h_postype(m_pdata->getPositions(), access_location::host, access_mode::read);
    ArrayHandle<Scalar4> h_orientation(m_pdata->getOrientationArray(), access_location::host, access_mode::read);
    ArrayHandle<Scalar> h_diameter(m_pdata->getDiameters(), access_location::host, access_mode::read);
    ArrayHandle<Scalar> h_charge(m_pdata->getCharges(), access_location::host, access_mode::read);
    ArrayHandle<unsigned int> h_tag(m_pdata->getTags(), access_location::host, access_mode::read);
    ArrayHandle<int3> h_image(m_pdata->getImages(), access_location::host, access_mode::read);

    if (patch)
        {
        // test old configuration against itself
        #ifdef ENABLE_TBB_TASK
        tbb::parallel_for((unsigned int)0,m_n_particles_old, [&](unsigned int i)
        #else
        for (unsigned int i = 0; i < m_n_particles_old; ++i)
        #endif
            {
            unsigned int typ_i = __scalar_as_int(m_postype_backup[i].w);

            vec3<Scalar> pos_i(m_postype_backup[i]);
            quat<Scalar> orientation_i(m_orientation_backup[i]);

            Scalar d_i(m_diameter_backup[i]);
            Scalar charge_i(m_charge_backup[i]);

            // subtract minimum AABB extent from search radius
            Scalar extent_i = 0.5*patch->getAdditiveCutoff(typ_i);
            OverlapReal R_query = std::max(0.0,r_cut_patch+extent_i-min_core_diameter/(OverlapReal)2.0);
            detail::AABB aabb_local = detail::AABB(vec3<Scalar>(0,0,0),R_query);

            const unsigned int n_images = image_list.size();

            for (unsigned int cur_image = 0; cur_image < n_images; cur_image++)
                {
                vec3<Scalar> pos_i_image = pos_i + image_list[cur_image];

                detail::AABB aabb_i_image = aabb_local;
                aabb_i_image.translate(pos_i_image);

                // stackless search
                for (unsigned int cur_node_idx = 0; cur_node_idx < m_aabb_tree_old.getNumNodes(); cur_node_idx++)
                    {
                    if (detail::overlap(m_aabb_tree_old.getNodeAABB(cur_node_idx), aabb_i_image))
                        {
                        if (m_aabb_tree_old.isNodeLeaf(cur_node_idx))
                            {
                            for (unsigned int cur_p = 0; cur_p < m_aabb_tree_old.getNodeNumParticles(cur_node_idx); cur_p++)
                                {
                                // read in its position and orientation
                                unsigned int j = m_aabb_tree_old.getNodeParticle(cur_node_idx, cur_p);

                                if (m_tag_backup[i] == m_tag_backup[j] && cur_image == 0) continue;

                                // load the position and orientation of the j particle
                                vec3<Scalar> pos_j = vec3<Scalar>(m_postype_backup[j]);
                                unsigned int typ_j = __scalar_as_int(m_postype_backup[j].w);

                                // put particles in coordinate system of particle i
                                vec3<Scalar> r_ij = pos_j - pos_i_image;
                                Scalar rsq_ij = dot(r_ij, r_ij);

                                Scalar rcut_ij = r_cut_patch + extent_i + 0.5*patch->getAdditiveCutoff(typ_j);

                                if (rsq_ij <= rcut_ij*rcut_ij)
                                    {
                                    // the particle pair
                                    unsigned int new_tag_i;
                                        {
                                        auto it = map.find(m_tag_backup[i]);
                                        assert(it != map.end());
                                        new_tag_i = it->second;
                                        }

                                    unsigned int new_tag_j;
                                        {
                                        auto it = map.find(m_tag_backup[j]);
                                        assert(it!=map.end());
                                        new_tag_j = it->second;
                                        }
                                    auto p = std::make_pair(new_tag_i,new_tag_j);

                                    // if particle interacts in different image already, add to that energy
                                    float U = 0.0;
                                        {
                                        auto it_energy = m_energy_old_old.find(p);
                                        if (it_energy != m_energy_old_old.end())
                                            U = it_energy->second;
                                        }

                                    U += patch->energy(r_ij, typ_i,
                                                        quat<float>(orientation_i),
                                                        d_i,
                                                        charge_i,
                                                        typ_j,
                                                        quat<float>(m_orientation_backup[j]),
                                                        m_diameter_backup[j],
                                                        m_charge_backup[j]);

                                    // update map
                                    m_energy_old_old[p] = U;

                                    int3 delta_img = m_image_backup[i] - m_image_backup[j];
                                    bool interacts_via_pbc = delta_img.x || delta_img.y || delta_img.z;
                                    interacts_via_pbc |= cur_image != 0;

                                    if (line && !swap && interacts_via_pbc)
                                        {
                                        // if interaction across PBC, reject cluster move
                                        m_local_reject.insert(new_tag_i);
                                        m_local_reject.insert(new_tag_j);
                                        }
                                    } // end if overlap

                                } // end loop over AABB tree leaf
                            } // end is leaf
                        } // end if overlap
                    else
                        {
                        // skip ahead
                        cur_node_idx += m_aabb_tree_old.getNodeSkip(cur_node_idx);
                        }

                    } // end loop over nodes

                } // end loop over images

            } // end loop over old configuration
        #ifdef ENABLE_TBB_TASK
            );
        #endif
        }

    // loop over new configuration
    #ifdef ENABLE_TBB_TASK
    tbb::parallel_for((unsigned int)0,nptl, [&](unsigned int i)
    #else
    for (unsigned int i = 0; i < nptl; ++i)
    #endif
        {
        unsigned int typ_i = __scalar_as_int(h_postype.data[i].w);

        vec3<Scalar> pos_i_new(h_postype.data[i]);
        quat<Scalar> orientation_i_new(h_orientation.data[i]);

        Shape shape_i(orientation_i_new, params[typ_i]);
        Scalar r_excl_i = shape_i.getCircumsphereDiameter()/Scalar(2.0);

        // check for overlap at mirrored position, with other particles in old configuration
        detail::AABB aabb_i_local = shape_i.getAABB(vec3<Scalar>(0,0,0));

        // All image boxes (including the primary)
        const unsigned int n_images = image_list.size();

        // check against old
        for (unsigned int cur_image = 0; cur_image < n_images; cur_image++)
            {
            vec3<Scalar> pos_i_image = pos_i_new + image_list[cur_image];

            detail::AABB aabb_i_image = aabb_i_local;
            aabb_i_image.translate(pos_i_image);

            // stackless search
            for (unsigned int cur_node_idx = 0; cur_node_idx < m_aabb_tree_old.getNumNodes(); cur_node_idx++)
                {
                if (detail::overlap(m_aabb_tree_old.getNodeAABB(cur_node_idx), aabb_i_image))
                    {
                    if (m_aabb_tree_old.isNodeLeaf(cur_node_idx))
                        {
                        for (unsigned int cur_p = 0; cur_p < m_aabb_tree_old.getNodeNumParticles(cur_node_idx); cur_p++)
                            {
                            // read in its position and orientation
                            unsigned int j = m_aabb_tree_old.getNodeParticle(cur_node_idx, cur_p);

                            unsigned int new_tag_j;
                                {
                                auto it = map.find(m_tag_backup[j]);
                                assert(it != map.end());
                                new_tag_j = it->second;
                                }

                            if (h_tag.data[i] == new_tag_j && cur_image == 0) continue;

                            // load the position and orientation of the j particle
                            vec3<Scalar> pos_j = vec3<Scalar>(m_postype_backup[j]);
                            unsigned int typ_j = __scalar_as_int(m_postype_backup[j].w);
                            Shape shape_j(quat<Scalar>(m_orientation_backup[j]), params[typ_j]);

                            // put particles in coordinate system of particle i
                            vec3<Scalar> r_ij = pos_j - pos_i_image;

                            // check for circumsphere overlap
                            Scalar r_excl_j = shape_j.getCircumsphereDiameter()/Scalar(2.0);
                            Scalar RaRb = r_excl_i + r_excl_j;
                            Scalar rsq_ij = dot(r_ij, r_ij);

                            unsigned int err = 0;
                            if (rsq_ij <= RaRb*RaRb)
                                {
                                if (h_overlaps.data[overlap_idx(typ_i,typ_j)]
                                    && test_overlap(r_ij, shape_i, shape_j, err))
                                    {

                                    int3 delta_img = h_image.data[i] - m_image_backup[j];
                                    bool interacts_via_pbc = delta_img.x || delta_img.y || delta_img.z;
                                    interacts_via_pbc |= cur_image != 0;

                                    bool reject = (line &&!swap) && interacts_via_pbc;

                                    if (swap && ((typ_i != m_ab_types[0] && typ_i != m_ab_types[1])
                                        || (typ_j != m_ab_types[0] && typ_j != m_ab_types[1])))
                                        reject = true;

                                    // add connection
                                    m_overlap.push_back(std::make_pair(h_tag.data[i],new_tag_j));

                                    if (reject)
                                        {
                                        // if interaction across PBC, reject cluster move
                                        m_local_reject.insert(h_tag.data[i]);
                                        m_local_reject.insert(new_tag_j);
                                        }
                                    } // end if overlap
                                }

                            } // end loop over AABB tree leaf
                        } // end is leaf
                    } // end if overlap
                else
                    {
                    // skip ahead
                    cur_node_idx += m_aabb_tree_old.getNodeSkip(cur_node_idx);
                    }

                } // end loop over nodes
            } // end loop over images

        if (patch)
            {
            // subtract minimum AABB extent from search radius
            Scalar extent_i = 0.5*patch->getAdditiveCutoff(typ_i);
            OverlapReal R_query = std::max(0.0,r_cut_patch+extent_i-min_core_diameter/(OverlapReal)2.0);
            detail::AABB aabb_local = detail::AABB(vec3<Scalar>(0,0,0),R_query);

            // compute V(r'-r)
            for (unsigned int cur_image = 0; cur_image < n_images; cur_image++)
                {
                vec3<Scalar> pos_i_image = pos_i_new + image_list[cur_image];

                detail::AABB aabb_i_image = aabb_local;
                aabb_i_image.translate(pos_i_image);

                // stackless search
                for (unsigned int cur_node_idx = 0; cur_node_idx < m_aabb_tree_old.getNumNodes(); cur_node_idx++)
                    {
                    if (detail::overlap(m_aabb_tree_old.getNodeAABB(cur_node_idx), aabb_i_image))
                        {
                        if (m_aabb_tree_old.isNodeLeaf(cur_node_idx))
                            {
                            for (unsigned int cur_p = 0; cur_p < m_aabb_tree_old.getNodeNumParticles(cur_node_idx); cur_p++)
                                {
                                // read in its position and orientation
                                unsigned int j = m_aabb_tree_old.getNodeParticle(cur_node_idx, cur_p);

                                unsigned int new_tag_j;
                                    {
                                    auto it = map.find(m_tag_backup[j]);
                                    assert(it != map.end());
                                    new_tag_j = it->second;
                                    }

                                if (h_tag.data[i] == new_tag_j && cur_image == 0) continue;

                                vec3<Scalar> pos_j(m_postype_backup[j]);
                                unsigned int typ_j = __scalar_as_int(m_postype_backup[j].w);

                                // put particles in coordinate system of particle i
                                vec3<Scalar> r_ij = pos_j - pos_i_image;

                                // check for excluded volume sphere overlap
                                Scalar rsq_ij = dot(r_ij, r_ij);

                                Scalar rcut_ij = r_cut_patch + extent_i + 0.5*patch->getAdditiveCutoff(typ_j);

                                if (rsq_ij <= rcut_ij*rcut_ij)
                                    {
                                    auto p = std::make_pair(h_tag.data[i], new_tag_j);

                                    // if particle interacts in different image already, add to that energy
                                    float U = 0.0;
                                        {
                                        auto it_energy = m_energy_new_old.find(p);
                                        if (it_energy != m_energy_new_old.end())
                                            U = it_energy->second;
                                        }

                                    U += patch->energy(r_ij, typ_i,
                                                            quat<float>(shape_i.orientation),
                                                            h_diameter.data[i],
                                                            h_charge.data[i],
                                                            typ_j,
                                                            quat<float>(m_orientation_backup[j]),
                                                            m_diameter_backup[j],
                                                            m_charge_backup[j]);

                                    // update map
                                    m_energy_new_old[p] = U;

                                    int3 delta_img = h_image.data[i] - m_image_backup[j];
                                    bool interacts_via_pbc = delta_img.x || delta_img.y || delta_img.z;
                                    interacts_via_pbc |= cur_image != 0;

                                    if (line && !swap && interacts_via_pbc)
                                        {
                                        // if interaction across PBC, reject cluster move
                                        m_local_reject.insert(h_tag.data[i]);
                                        m_local_reject.insert(new_tag_j);
                                        }
                                    }
                                } // end loop over AABB tree leaf
                            } // end is leaf
                        } // end if overlap
                    else
                        {
                        // skip ahead
                        cur_node_idx += m_aabb_tree_old.getNodeSkip(cur_node_idx);
                        }

                    } // end loop over nodes

                } // end loop over images
            } // end if patch
        } // end loop over local particles
    #ifdef ENABLE_TBB_TASK
        );
    #endif

    if (line && !swap)
        {
        // check if particles are interacting in the new configuration
        #ifdef ENABLE_TBB_TASK
        tbb::parallel_for((unsigned int)0,nptl, [&](unsigned int i)
        #else
        for (unsigned int i = 0; i < nptl; ++i)
        #endif
            {
            unsigned int typ_i = __scalar_as_int(h_postype.data[i].w);

            vec3<Scalar> pos_i_new(h_postype.data[i]);
            quat<Scalar> orientation_i_new(h_orientation.data[i]);

            Shape shape_i(orientation_i_new, params[typ_i]);
            Scalar r_excl_i = shape_i.getCircumsphereDiameter()/Scalar(2.0);

            // subtract minimum AABB extent from search radius
            Scalar extent_i = 0.0;
            if (patch)
                extent_i = 0.5*patch->getAdditiveCutoff(typ_i);

            OverlapReal R_query = std::max(r_excl_i,r_cut_patch+extent_i-min_core_diameter/(OverlapReal)2.0);
            detail::AABB aabb_i = detail::AABB(pos_i_new,R_query);

            // All image boxes (including the primary)
            const unsigned int n_images = image_list.size();

            // check against new AABB tree
            for (unsigned int cur_image = 0; cur_image < n_images; cur_image++)
                {
                vec3<Scalar> pos_i_image = pos_i_new + image_list[cur_image];

                detail::AABB aabb_i_image = aabb_i;
                aabb_i_image.translate(image_list[cur_image]);

                // stackless search
                for (unsigned int cur_node_idx = 0; cur_node_idx < aabb_tree.getNumNodes(); cur_node_idx++)
                    {
                    if (detail::overlap(aabb_tree.getNodeAABB(cur_node_idx), aabb_i_image))
                        {
                        if (aabb_tree.isNodeLeaf(cur_node_idx))
                            {
                            for (unsigned int cur_p = 0; cur_p < aabb_tree.getNodeNumParticles(cur_node_idx); cur_p++)
                                {
                                // read in its position and orientation
                                unsigned int j = aabb_tree.getNodeParticle(cur_node_idx, cur_p);

                                // no trivial bonds
                                if (h_tag.data[i] == h_tag.data[j]) continue;

                                // load the position and orientation of the j particle
                                vec3<Scalar> pos_j = vec3<Scalar>(h_postype.data[j]);
                                unsigned int typ_j = __scalar_as_int(h_postype.data[j].w);
                                Shape shape_j(quat<Scalar>(h_orientation.data[j]), params[typ_j]);

                                // put particles in coordinate system of particle i
                                vec3<Scalar> r_ij = pos_j - pos_i_image;

                                // check for circumsphere overlap
                                Scalar r_excl_j = shape_j.getCircumsphereDiameter()/Scalar(2.0);
                                Scalar RaRb = r_excl_i + r_excl_j;
                                Scalar rsq_ij = dot(r_ij, r_ij);

                                Scalar rcut_ij = 0.0;
                                if (patch)
                                    rcut_ij = r_cut_patch + extent_i + 0.5*patch->getAdditiveCutoff(typ_j);

                                bool interact_patch = patch && rsq_ij <= rcut_ij*rcut_ij;

                                unsigned int err = 0;

                                if (interact_patch || (rsq_ij <= RaRb*RaRb && h_overlaps.data[overlap_idx(typ_i,typ_j)]
                                        && test_overlap(r_ij, shape_i, shape_j, err)))
                                    {
                                    int3 delta_img = h_image.data[i] - h_image.data[j];
                                    bool interacts_via_pbc = delta_img.x || delta_img.y || delta_img.z;
                                    interacts_via_pbc |= cur_image != 0;

                                    if (interacts_via_pbc)
                                        {
                                        // add to reject list
                                        m_local_reject.insert(h_tag.data[i]);
                                        m_local_reject.insert(h_tag.data[j]);

                                        m_interact_new_new.insert(std::make_pair(h_tag.data[i],h_tag.data[j]));
                                        }
                                    } // end if overlap

                                } // end loop over AABB tree leaf
                            } // end is leaf
                        } // end if overlap
                    else
                        {
                        // skip ahead
                        cur_node_idx += aabb_tree.getNodeSkip(cur_node_idx);
                        }

                    } // end loop over nodes
                } // end loop over images
            } // end loop over local particles
        #ifdef ENABLE_TBB_TASK
            );
        #endif
        }

    if (m_prof) m_prof->pop(m_exec_conf);
    }

/*! Perform a cluster move
    \param timestep Current time step of the simulation
*/
template< class Shape >
void UpdaterClusters<Shape>::update(unsigned int timestep)
    {
    m_exec_conf->msg->notice(10) << timestep << " UpdaterClusters" << std::endl;

    m_count_step_start = m_count_total;

    // if no particles, exit early
    if (! m_pdata->getNGlobal()) return;

    if (m_prof) m_prof->push(m_exec_conf,"HPMC Clusters");

    // save a copy of the old configuration
    m_n_particles_old = m_pdata->getN();

    unsigned int nptl = m_pdata->getN()+m_pdata->getNGhosts();
    m_postype_backup.resize(nptl);
    m_orientation_backup.resize(nptl);
    m_diameter_backup.resize(nptl);
    m_charge_backup.resize(nptl);
    m_tag_backup.resize(nptl);
    m_image_backup.resize(nptl);

        {
        ArrayHandle<Scalar4> h_postype(m_pdata->getPositions(), access_location::host, access_mode::read);
        ArrayHandle<Scalar4> h_orientation(m_pdata->getOrientationArray(), access_location::host, access_mode::read);
        ArrayHandle<Scalar> h_diameter(m_pdata->getDiameters(), access_location::host, access_mode::read);
        ArrayHandle<Scalar> h_charge(m_pdata->getCharges(), access_location::host, access_mode::read);
        ArrayHandle<unsigned int> h_tag(m_pdata->getTags(), access_location::host, access_mode::read);
        ArrayHandle<int3> h_image(m_pdata->getImages(), access_location::host, access_mode::read);

        for (unsigned int i = 0; i < nptl; ++i)
            {
            m_postype_backup[i] = h_postype.data[i];
            m_orientation_backup[i] = h_orientation.data[i];
            m_diameter_backup[i] = h_diameter.data[i];
            m_charge_backup[i] = h_charge.data[i];
            m_tag_backup[i] = h_tag.data[i];
            // reset image
            m_image_backup[i] = make_int3(0,0,0);
            }
        }

    if (m_prof) m_prof->push(m_exec_conf,"Transform");

    // generate the move, select a pivot
    hoomd::RandomGenerator rng(hoomd::RNGIdentifier::UpdaterClusters, timestep, this->m_seed);
    BoxDim box = m_pdata->getGlobalBox();
    vec3<Scalar> pivot(0,0,0);

    bool swap = m_ab_types.size() && (hoomd::detail::generate_canonical<Scalar>(rng) <= m_swap_move_ratio);

    if (swap)
        {
        auto params = m_mc->getParams();
        Shape shape_A(quat<Scalar>(), params[m_ab_types[0]]);
        Shape shape_B(quat<Scalar>(), params[m_ab_types[1]]);

        // swap only works with equal sized spheres
        if (shape_A.getCircumsphereDiameter() != shape_B.getCircumsphereDiameter()
            || shape_A.hasOrientation() || shape_B.hasOrientation())
            swap = false;
        }

    // is this a line reflection?
    bool line = !swap && (m_mc->hasOrientation() || (hoomd::detail::generate_canonical<Scalar>(rng) > m_move_ratio));

    quat<Scalar> q;

    if (line)
        {
        // random normalized vector
        vec3<Scalar> n;

        if (m_sysdef->getNDimensions() == 3)
            {
            Scalar theta = hoomd::UniformDistribution<Scalar>(Scalar(0.0),Scalar(2.0*M_PI))(rng);
            Scalar z = hoomd::UniformDistribution<Scalar>(Scalar(-1.0),Scalar(1.0))(rng);
            n = vec3<Scalar>(fast::sqrt(Scalar(1.0)-z*z)*fast::cos(theta),fast::sqrt(Scalar(1.0)-z*z)*fast::sin(theta),z);
            }
        else
            {
            // reflection around z axis
            n = vec3<Scalar>(0,0,1);
            }

        // line reflection
        q = quat<Scalar>(0,n);
        }
    else
        {
        Scalar3 f;
        f.x = hoomd::detail::generate_canonical<Scalar>(rng);
        f.y = hoomd::detail::generate_canonical<Scalar>(rng);
        if (m_sysdef->getNDimensions() == 3)
            {
            f.z = hoomd::detail::generate_canonical<Scalar>(rng);
            }
        else
            {
            f.z = 0.5;
            }

        pivot = vec3<Scalar>(box.makeCoordinates(f));
        if (m_sysdef->getNDimensions() == 2)
            {
            // force z component to be zero
            pivot.z = 0.0;
            }
        }

    SnapshotParticleData<Scalar> snap(m_pdata->getNGlobal());

    // obtain particle data from all ranks

    // save origin information
    Scalar3 origin = m_pdata->getOrigin();
    int3 origin_image = m_pdata->getOriginImage();

    // take a snapshot, and save tag->snap idx mapping

    // reset origin, so that snapshot positions match AABB tree positions
    m_pdata->resetOrigin();
    auto map = m_pdata->takeSnapshot(snap);

    #ifdef ENABLE_MPI
    if (m_comm)
        {
        // we need the particle tag -> snapshot idx map on all ranks
        bcast(map, 0, m_exec_conf->getMPICommunicator());
        }
    #endif


    // transform all particles on rank zero
    bool master = !m_exec_conf->getRank();

    // compute the width of the active region
    Scalar nominal_width = this->getNominalWidth();
    Scalar3 npd = box.getNearestPlaneDistance();
    Scalar3 range = nominal_width / npd;

    if (m_sysdef->getNDimensions() == 2)
        {
        // no interaction along z
        range.z = 0;
        }

    // reset list of rejected particles
    m_ptl_reject.clear();

    // keep a backup copy
    SnapshotParticleData<Scalar> snap_old = snap;

    if (master)
        {
        // access parameters
        auto& params = m_mc->getParams();

        // create a copy of the box without periodic boundaries
        BoxDim global_box_nonperiodic = box;
        global_box_nonperiodic.setPeriodic(m_pdata->getBox().getPeriodic());

        for (unsigned int i = 0; i < snap.size; ++i)
            {
            // reset image
            snap.image[i] = make_int3(0,0,0);

            if (swap)
                {
                // swap move
                if (snap.type[i] == m_ab_types[0])
                    snap.type[i] = m_ab_types[1];
                else if (snap.type[i] == m_ab_types[1])
                    snap.type[i] = m_ab_types[0];
                }
            else
                {
                // if the particle falls outside the active volume of global_box_nonperiodic, reject
                if (!isActive(vec_to_scalar3(snap.pos[i]), global_box_nonperiodic, range))
                    {
                    m_ptl_reject.insert(i);
                    }

                if (!line)
                    {
                    // point reflection
                    snap.pos[i] = pivot-(snap.pos[i]-pivot);
                    }
                else
                    {
                    // line reflection
                    snap.pos[i] = lineReflection(snap.pos[i], pivot, q);
                    Shape shape_i(snap.orientation[i], params[snap.type[i]]);
                    if (shape_i.hasOrientation())
                        snap.orientation[i] = q*snap.orientation[i];
                    }
                // reject if outside active volume of box at new position
                if (!isActive(vec_to_scalar3(snap.pos[i]), global_box_nonperiodic, range))
                    {
                    m_ptl_reject.insert(i);
                    }

                // wrap particle back into box
                snap.image[i] = box.getImage(snap.pos[i]);
                snap.pos[i] = box.shift(snap.pos[i],-snap.image[i]);
                }
            }
        }

    if (m_prof) m_prof->pop(m_exec_conf);

    // store old locality data
    m_aabb_tree_old = m_mc->buildAABBTree();

    // reload particle data
    // now all tags will be consecutive
    m_pdata->initializeFromSnapshot(snap);

    if (m_prof) m_prof->pop(m_exec_conf);

    // update ghosts & signal that AABB tree is invalid
    m_mc->communicate(true);

    if (m_prof) m_prof->push(m_exec_conf,"HPMC Clusters");

    // determine which particles interact
    findInteractions(timestep, pivot, q, swap, line, map);

    if (m_prof) m_prof->push(m_exec_conf,"Move");

    // collect interactions on rank 0
    #ifndef ENABLE_TBB_TASK
    std::vector< std::vector<std::pair<unsigned int, unsigned int> > > all_overlap;
    std::vector< std::vector<std::pair<unsigned int, unsigned int> > > all_interact_old_old;
    std::vector< std::vector<std::pair<unsigned int, unsigned int> > > all_interact_new_old;
    std::vector< std::set<std::pair<unsigned int, unsigned int> > > all_interact_new_new;
    std::vector< std::set<unsigned int> > all_local_reject;

    std::vector< std::map<std::pair<unsigned int, unsigned int>, float> > all_energy_old_old;
    std::vector< std::map<std::pair<unsigned int, unsigned int>, float> > all_energy_new_old;
    #else
    std::vector< tbb::concurrent_vector<std::pair<unsigned int, unsigned int> > > all_overlap;
    std::vector< tbb::concurrent_vector<std::pair<unsigned int, unsigned int> > > all_interact_old_old;
    std::vector< tbb::concurrent_vector<std::pair<unsigned int, unsigned int> > > all_interact_new_old;
    std::vector< tbb::concurrent_unordered_set<std::pair<unsigned int, unsigned int> > > all_interact_new_new;
    std::vector< tbb::concurrent_unordered_set<unsigned int> > all_local_reject;

    std::vector< tbb::concurrent_unordered_map<std::pair<unsigned int, unsigned int>, float> > all_energy_old_old;
    std::vector< tbb::concurrent_unordered_map<std::pair<unsigned int, unsigned int>, float> > all_energy_new_old;
    #endif

    #ifdef ENABLE_MPI
    if (m_comm)
        {
        // combine lists from different ranks
        gather_v(m_overlap, all_overlap, 0, m_exec_conf->getMPICommunicator());
        gather_v(m_interact_old_old, all_interact_old_old, 0, m_exec_conf->getMPICommunicator());
        gather_v(m_interact_new_old, all_interact_new_old, 0, m_exec_conf->getMPICommunicator());
        gather_v(m_interact_new_new, all_interact_new_new, 0, m_exec_conf->getMPICommunicator());
        gather_v(m_local_reject, all_local_reject, 0, m_exec_conf->getMPICommunicator());
        }
    #endif

    if (m_mc->getPatchInteraction())
        {
        // collect energies on rank 0
        #ifdef ENABLE_MPI
        if (m_comm)
            {
            gather_v(m_energy_old_old, all_energy_old_old, 0, m_exec_conf->getMPICommunicator());
            gather_v(m_energy_new_old, all_energy_new_old, 0, m_exec_conf->getMPICommunicator());
            }
        #endif
        }

    if (this->m_prof)
        this->m_prof->push("fill");

    if (master)
        {
        // fill in the cluster bonds, using bond formation probability defined in Liu and Luijten

        if (m_prof)
            m_prof->push("realloc");

        // resize the number of graph nodes in place
        m_G.resize(snap.size);

        if (m_prof)
            m_prof->pop();


        #ifdef ENABLE_MPI
        if (m_comm)
            {
            // complete the list of rejected particles
            for (auto it_i = all_local_reject.begin(); it_i != all_local_reject.end(); ++it_i)
                {
                for (auto it_j = it_i->begin(); it_j != it_i->end(); ++it_j)
                    {
                    m_ptl_reject.insert(*it_j);
                    }
                }
            }
        #endif

        if (line && !swap)
            {
            if (m_prof)
                m_prof->push("new new");

            #ifdef ENABLE_MPI
            if (m_comm)
                {
                for (auto it_i = all_interact_new_new.begin(); it_i != all_interact_new_new.end(); ++it_i)
                    {
                    for (auto it_j = it_i->begin(); it_j != it_i->end(); ++it_j)
                        {
                        unsigned int i = it_j->first;
                        unsigned int j = it_j->second;

                        m_G.addEdge(i, j);
                        }
                    }
                }
            else
            #endif
                {
                #ifdef ENABLE_TBB_TASK
                tbb::parallel_for(m_interact_new_new.range(), [&] (decltype(m_interact_new_new.range()) r)
                #else
                auto &r = m_interact_new_new;
                #endif
                    {
                    for (auto it = r.begin(); it != r.end(); ++it)
                        {
                        unsigned int i = it->first;
                        unsigned int j = it->second;

                        m_G.addEdge(i,j);
                        }
                    }
                #ifdef ENABLE_TBB_TASK
                    );
                #endif
                }

            if (m_prof)
                m_prof->pop();
            }

        #ifdef ENABLE_MPI
        if (m_comm)
            {
            for (auto it_i = all_interact_new_old.begin(); it_i != all_interact_new_old.end(); ++it_i)
                {
                for (auto it_j = it_i->begin(); it_j != it_i->end(); ++it_j)
                    {
                    unsigned int i = it_j->first;
                    unsigned int j = it_j->second;

                    m_G.addEdge(i, j);
                    }
                }
            }
        else
        #endif
            {
            #ifdef ENABLE_TBB_TASK
            tbb::parallel_for(m_interact_new_old.range(), [&] (decltype(m_interact_new_old.range()) r)
            #else
            auto &r = m_interact_new_old;
            #endif
                {
                for (auto it = r.begin(); it != r.end(); ++it)
                    {
                    unsigned int i = it->first;
                    unsigned int j = it->second;

                    m_G.addEdge(i,j);
                    }
                }
            #ifdef ENABLE_TBB_TASK
                );
            #endif
            }


        if (m_prof)
            m_prof->push("overlap");

        #ifdef ENABLE_MPI
        if (m_comm)
            {
            for (auto it_i = all_overlap.begin(); it_i != all_overlap.end(); ++it_i)
                {
                for (auto it_j = it_i->begin(); it_j != it_i->end(); ++it_j)
                    {
                    unsigned int i = it_j->first;
                    unsigned int j = it_j->second;

                    m_G.addEdge(i,j);
                    }
                }
            }
        else
        #endif
            {
            #ifdef ENABLE_TBB_TASK
            tbb::parallel_for(m_overlap.range(), [&] (decltype(m_overlap.range()) r)
            #else
            auto &r = m_overlap;
            #endif
                {
                for (auto it = r.begin(); it != r.end(); ++it)
                    {
                    unsigned int i = it->first;
                    unsigned int j = it->second;

                    m_G.addEdge(i,j);
                    }
                }
            #ifdef ENABLE_TBB_TASK
                );
            #endif
            }

        if (m_prof)
            m_prof->pop();


        // interactions due to hard depletant-excluded volume overlaps (not used in base class)
        #ifdef ENABLE_MPI
        if (m_comm)
            {
            for (auto it_i = all_interact_old_old.begin(); it_i != all_interact_old_old.end(); ++it_i)
                {
                for (auto it_j = it_i->begin(); it_j != it_i->end(); ++it_j)
                    {
                    unsigned int i = it_j->first;
                    unsigned int j = it_j->second;

                    m_G.addEdge(i, j);
                    }
                }
            }
        else
        #endif
            {
            #ifdef ENABLE_TBB_TASK
            tbb::parallel_for(m_interact_old_old.range(), [&] (decltype(m_interact_old_old.range()) r)
            #else
            auto &r = m_interact_old_old;
            #endif
                {
                for (auto it = r.begin(); it != r.end(); ++it)
                    {
                    unsigned int i = it->first;
                    unsigned int j = it->second;

                    m_G.addEdge(i,j);
                    }
                }
            #ifdef ENABLE_TBB_TASK
                );
            #endif
            }

        #ifdef ENABLE_MPI
        if (m_comm)
            {
            for (auto it_i = all_interact_new_old.begin(); it_i != all_interact_new_old.end(); ++it_i)
                {
                for (auto it_j = it_i->begin(); it_j != it_i->end(); ++it_j)
                    {
                    unsigned int i = it_j->first;
                    unsigned int j = it_j->second;

                    m_G.addEdge(i, j);
                    }
                }
            }
        else
        #endif
            {
            #ifdef ENABLE_TBB_TASK
            tbb::parallel_for(m_interact_new_old.range(), [&] (decltype(m_interact_new_old.range()) r)
            #else
            auto &r = m_interact_new_old;
            #endif
                {
                for (auto it = r.begin(); it != r.end(); ++it)
                    {
                    unsigned int i = it->first;
                    unsigned int j = it->second;

                    m_G.addEdge(i,j);
                    }
                }
            #ifdef ENABLE_TBB_TASK
                );
            #endif
            }

        if (m_mc->getPatchInteraction())
            {
            // sum up interaction energies
            #ifdef ENABLE_TBB_TASK
            tbb::concurrent_unordered_map< std::pair<unsigned int, unsigned int>, float> delta_U;
            #else
            std::map< std::pair<unsigned int, unsigned int>, float> delta_U;
            #endif

            #ifdef ENABLE_MPI
            if (m_comm)
                {
                for (auto it_i = all_energy_old_old.begin(); it_i != all_energy_old_old.end(); ++it_i)
                    {
                    for (auto it_j = it_i->begin(); it_j != it_i->end(); ++it_j)
                        {
                        float delU = -it_j->second;
                        unsigned int i = it_j->first.first;
                        unsigned int j = it_j->first.second;

                        auto p = std::make_pair(i,j);

                        // add to energy
                        auto it = delta_U.find(p);
                        if (it != delta_U.end())
                            delU += it->second;

                        // update map with new interaction energy
                        delta_U[p] = delU;
                        }
                    }
                }
            else
            #endif
                {
                for (auto it = m_energy_old_old.begin(); it != m_energy_old_old.end(); ++it)
                    {
                    float delU = -it->second;
                    unsigned int i = it->first.first;
                    unsigned int j = it->first.second;

                    auto p = std::make_pair(i,j);

                    // add to energy
                    auto itj = delta_U.find(p);
                    if (itj != delta_U.end())
                        delU += itj->second;

                    // update map with new interaction energy
                    delta_U[p] = delU;
                    }
                }

            #ifdef ENABLE_MPI
            if (m_comm)
                {
                for (auto it_i = all_energy_new_old.begin(); it_i != all_energy_new_old.end(); ++it_i)
                    {
                    for (auto it_j = it_i->begin(); it_j != it_i->end(); ++it_j)
                        {
                        float delU = it_j->second;
                        unsigned int i = it_j->first.first;
                        unsigned int j = it_j->first.second;

                        // consider each pair uniquely
                        auto p = std::make_pair(i,j);

                        // add to energy
                        auto it = delta_U.find(p);
                        if (it != delta_U.end())
                            delU += it->second;

                        // update map with new interaction energy
                        delta_U[p] = delU;
                        }
                    }
                }
            else
            #endif
                {
                for (auto it = m_energy_new_old.begin(); it != m_energy_new_old.end(); ++it)
                    {
                    float delU = it->second;
                    unsigned int i = it->first.first;
                    unsigned int j = it->first.second;

                    auto p = std::make_pair(i,j);

                    // add to energy
                    auto itj = delta_U.find(p);
                    if (itj != delta_U.end())
                        delU += itj->second;

                    // update map with new interaction energy
                    delta_U[p] = delU;
                    }
                }

            #ifdef ENABLE_TBB_TASK
            tbb::parallel_for(delta_U.range(), [&] (decltype(delta_U.range()) r)
            #else
            auto &r = delta_U;
            #endif
                {
                for (auto it = r.begin(); it != r.end(); ++it)
                    {
                    float delU = it->second;
                    unsigned int i = it->first.first;
                    unsigned int j = it->first.second;

                    // create a RNG specific to this particle pair
                    hoomd::RandomGenerator rng_ij(hoomd::RNGIdentifier::UpdaterClustersPairwise, this->m_seed, timestep, std::min(i,j), std::max(i,j));

                    float pij = 1.0f-exp(-delU);
                    if (hoomd::detail::generate_canonical<float>(rng_ij) <= pij) // GCA
                        {
                        // add bond
                        m_G.addEdge(i,j);
                        }
                    }
                }
            #ifdef ENABLE_TBB_TASK
                );
            #endif
            } // end if (patch)

        if (this->m_prof) this->m_prof->push("connected components");
        // compute connected components
        m_clusters.clear();
        m_G.connectedComponents(m_clusters);
        if (this->m_prof) this->m_prof->pop();

        if (this->m_prof) this->m_prof->push("reject");

        // move every cluster independently
        m_count_total.n_clusters += m_clusters.size();

        for (unsigned int icluster = 0; icluster < m_clusters.size(); icluster++)
            {
            m_count_total.n_particles_in_clusters += m_clusters[icluster].size();

            // if any particle in the cluster is rejected, the cluster is not transformed
            bool reject = false;
            for (auto it = m_clusters[icluster].begin(); it != m_clusters[icluster].end(); ++it)
                {
                bool mpi = false;
                #ifdef ENABLE_MPI
                mpi = (bool)m_comm;
                if (mpi && m_ptl_reject.find(*it) != m_ptl_reject.end())
                    reject = true;
                #endif
                if (!mpi && m_local_reject.find(*it) != m_local_reject.end())
                    reject = true;
                }

            bool flip = hoomd::detail::generate_canonical<float>(rng) <= m_flip_probability;

            // count number of A and B particles in old and new config
            if (swap && m_ab_types.size())
                {
                int n_A_old = 0, n_A_new = 0;
                int n_B_old = 0, n_B_new = 0;

                for (auto it = m_clusters[icluster].begin(); it != m_clusters[icluster].end(); ++it)
                    {
                    unsigned int i = *it;
                    if (snap.type[i] == m_ab_types[0])
                        n_A_new++;
                    if (snap_old.type[i] == m_ab_types[0])
                        n_A_old++;
                    if (snap.type[i] == m_ab_types[1])
                        n_B_new++;
                    if (snap_old.type[i] == m_ab_types[1])
                        n_B_old++;
                    }

                Scalar NdelMu = 0.5*(Scalar)(n_B_new-n_A_new-n_B_old+n_A_old)*m_delta_mu;

                if (hoomd::detail::generate_canonical<float>(rng) > exp(NdelMu))
                    reject = true;
                }

            if (reject || !flip)
                {
                // revert cluster
                for (auto it = m_clusters[icluster].begin(); it != m_clusters[icluster].end(); ++it)
                    {
                    // particle index
                    unsigned int i = *it;

                    snap.pos[i] = snap_old.pos[i];
                    snap.orientation[i] = snap_old.orientation[i];
                    snap.type[i] = snap_old.type[i];
                    snap.image[i] = snap_old.image[i];

                    if (flip)
                        {
                        if (swap)
                            {
                            if (snap.type[i] == m_ab_types[0] || snap.type[i] == m_ab_types[1])
                                m_count_total.swap_reject_count++;
                            }
                        else
                            {
                            if (line)
                                m_count_total.reflection_reject_count++;
                            else
                                m_count_total.pivot_reject_count++;
                            }
                        }
                    }
                }
            else if (flip)
                {
                for (auto it = m_clusters[icluster].begin(); it != m_clusters[icluster].end(); ++it)
                    {
                    // particle index
                    unsigned int i = *it;

                    if (swap)
                        {
                        if (snap.type[i] == m_ab_types[0] || snap.type[i] == m_ab_types[1])
                            m_count_total.swap_accept_count++;
                        }
                    else
                        {
                        if (line)
                            m_count_total.reflection_accept_count++;
                        else
                            m_count_total.pivot_accept_count++;
                        }
                    }
                }
            } // end loop over clusters

        if (this->m_prof) this->m_prof->pop();
        } // if master

    if (this->m_prof) this->m_prof->pop();

    if (this->m_prof) this->m_prof->push("init");
    // finally re-initialize particle data
    m_pdata->initializeFromSnapshot(snap);

    // restore origin, after initializing from translated positions
    m_pdata->setOrigin(origin,origin_image);

    if (this->m_prof) this->m_prof->pop();
    if (m_prof) m_prof->pop(m_exec_conf);

    // in MPI and GPU simulations the integrator takes care of the grid shift
    bool grid_shift = true;
    #ifdef ENABLE_CUDA
    if (m_exec_conf->isCUDAEnabled())
        grid_shift = false;
    #endif

    #ifdef ENABLE_MPI
    if (m_comm)
        grid_shift = false;
    #endif

    if (grid_shift)
        {
        if (m_prof) m_prof->push(m_exec_conf,"Grid shift");

        // nominal width may be larger than nearest plane distance, correct
        Scalar max_shift = std::min(npd.x, std::min(npd.y,npd.z));
        max_shift = std::min(max_shift, nominal_width);

        // perform the grid shift to compensate for the uncrossable boundaries
        ArrayHandle<Scalar4> h_postype(m_pdata->getPositions(), access_location::host, access_mode::readwrite);
        ArrayHandle<int3> h_image(m_pdata->getImages(), access_location::host, access_mode::readwrite);

        Scalar3 shift = make_scalar3(0,0,0);

        hoomd::UniformDistribution<Scalar> shift_gen(-max_shift/Scalar(2.0),max_shift/Scalar(2.0));
        shift.x = shift_gen(rng);
        shift.y = shift_gen(rng);
        if (this->m_sysdef->getNDimensions() == 3)
            {
            shift.z = shift_gen(rng);
            }

        for (unsigned int i = 0; i < m_pdata->getN(); i++)
            {
            // read in the current position and orientation
            Scalar4 postype_i = h_postype.data[i];
            vec3<Scalar> r_i = vec3<Scalar>(postype_i);
            r_i += vec3<Scalar>(shift);
            h_postype.data[i] = vec_to_scalar4(r_i, postype_i.w);
            box.wrap(h_postype.data[i], h_image.data[i]);
            }
        this->m_pdata->translateOrigin(shift);

        m_mc->invalidateAABBTree();

        if (m_prof) m_prof->pop(m_exec_conf);
        }

    if (m_prof) m_prof->pop(m_exec_conf);

    m_mc->communicate(true);
    }


template < class Shape> void export_UpdaterClusters(pybind11::module& m, const std::string& name)
    {
    pybind11::class_< UpdaterClusters<Shape>, std::shared_ptr< UpdaterClusters<Shape> > >(m, name.c_str(), pybind11::base<Updater>())
          .def( pybind11::init< std::shared_ptr<SystemDefinition>,
                         std::shared_ptr< IntegratorHPMCMono<Shape> >,
                         unsigned int >())
        .def("getCounters", &UpdaterClusters<Shape>::getCounters)
        .def("setMoveRatio", &UpdaterClusters<Shape>::setMoveRatio)
        .def("setFlipProbability", &UpdaterClusters<Shape>::setFlipProbability)
        .def("setSwapMoveRatio", &UpdaterClusters<Shape>::setSwapMoveRatio)
        .def("setSwapTypePair", &UpdaterClusters<Shape>::setSwapTypePair)
        .def("setDeltaMu", &UpdaterClusters<Shape>::setDeltaMu)
    ;
    }

inline void export_hpmc_clusters_counters(pybind11::module &m)
    {
    pybind11::class_< hpmc_clusters_counters_t >(m, "hpmc_clusters_counters_t")
        .def("getPivotAcceptance", &hpmc_clusters_counters_t::getPivotAcceptance)
        .def("getReflectionAcceptance", &hpmc_clusters_counters_t::getReflectionAcceptance)
        .def("getSwapAcceptance", &hpmc_clusters_counters_t::getSwapAcceptance)
        .def("getNParticlesMoved", &hpmc_clusters_counters_t::getNParticlesMoved)
        .def("getNParticlesInClusters", &hpmc_clusters_counters_t::getNParticlesInClusters);
    }

} // end namespace hpmc

#endif // _UPDATER_HPMC_CLUSTERS_
