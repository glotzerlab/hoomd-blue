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

#ifdef ENABLE_TBB
#include <tbb/concurrent_unordered_map.h>
#include <tbb/concurrent_unordered_set.h>
#include <tbb/concurrent_vector.h>
#include <tbb/parallel_for.h>
#include <tbb/task.h>
#include <atomic>
#endif

namespace hpmc
{

namespace detail
{

#ifdef ENABLE_TBB
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

        #ifdef ENABLE_TBB
        inline void connectedComponents(std::vector<tbb::concurrent_vector<unsigned int> >& cc);
        #else
        inline void connectedComponents(std::vector<std::vector<unsigned int> >& cc);
        #endif

    private:
        #ifndef ENABLE_TBB
        std::multimap<unsigned int,unsigned int> adj;
        #else
        tbb::concurrent_unordered_multimap<unsigned int, unsigned int> adj;
        #endif

        #ifndef ENABLE_TBB
        std::vector<unsigned int> visited;
        #else
        std::vector<my_atomic_flag> visited;
        #endif

        #ifndef ENABLE_TBB
        // A function used by DFS
        inline void DFSUtil(unsigned int v, std::vector<unsigned int>& visited, std::vector<unsigned int>& cur_cc);
        #endif

        #ifdef ENABLE_TBB
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
        #endif // ENABLE_TBB

    };

// Gather connected components in an undirected graph
#ifdef ENABLE_TBB
void Graph::connectedComponents(std::vector<tbb::concurrent_vector<unsigned int> >& cc)
#else
void Graph::connectedComponents(std::vector<std::vector<unsigned int> >& cc)
#endif
    {
    #ifdef ENABLE_TBB
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

#ifndef ENABLE_TBB
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
    #ifndef ENABLE_TBB
    visited.resize(V, 0);
    #else
    visited.resize(V);
    #endif
    }

void Graph::resize(unsigned int V)
    {
    #ifndef ENABLE_TBB
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

    In order to support anisotropic particles, we allow line reflections orthogonal
    to the PBC axes only, as described in Sinkovits et al.
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

        //! Get the seed
        unsigned int getSeed()
            {
            return m_seed;
            }

        //! Set the move ratio
        void setMoveRatio(Scalar move_ratio)
            {
            m_move_ratio = move_ratio;
            }

        //! Get the move ratio
        Scalar getMoveRatio()
            {
            return m_move_ratio;
            }

        //! Set the swap to geometric move ratio
        void setSwapMoveRatio(Scalar move_ratio)
            {
            m_swap_move_ratio = move_ratio;
            }

        //! Get the swap move ratio
        Scalar getSwapMoveRatio()
            {
            return m_swap_move_ratio;
            }

        //! Set the cluster flip probability
        void setFlipProbability(Scalar flip_probability)
            {
            m_flip_probability = flip_probability;
            }

        //! Get the flip probability
        Scalar getFlipProbability()
            {
            return m_flip_probability;
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

        //! Set the pair type to be used with type swap moves (by name)
        void setSwapTypePairStr(pybind11::list l)
            {
            size_t l_size = pybind11::len(l);
            if (l_size == 0)
                {
                m_ab_types.clear();
                }
            else if (l_size == 2)
                {
                std::string type_A = l[0].cast<std::string>();
                std::string type_B = l[1].cast<std::string>();

                unsigned int id_A = m_pdata->getTypeByName(type_A);
                unsigned int id_B = m_pdata->getTypeByName(type_B);
                setSwapTypePair(id_A, id_B);
                }
            else
                {
                throw std::runtime_error("swap_types must be a list of length 0 or 2");
                }
            }

        //! Get the swap pair types as a python list
        pybind11::list getSwapTypePairStr()
            {
            pybind11::list result;
            if (m_ab_types.size() == 0)
                {
                return result;
                }
            else if (m_ab_types.size() == 2)
                {
                result.append(m_pdata->getNameByType(m_ab_types[0]));
                result.append(m_pdata->getNameByType(m_ab_types[1]));
                return result;
                }
            else
                {
                throw std::runtime_error("invalid m_ab_types");
                }
            }

        //! Set the difference in chemical potential mu_B - mu_A
        void setDeltaMu(Scalar delta_mu)
            {
            m_delta_mu = delta_mu;
            }

        //! Get the the difference in chemical potential mu_B - mu_A
        Scalar getDeltaMu()
            {
            return m_delta_mu;
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

        #ifdef ENABLE_TBB
        std::vector<tbb::concurrent_vector<unsigned int> > m_clusters; //!< Cluster components
        #else
        std::vector<std::vector<unsigned int> > m_clusters; //!< Cluster components
        #endif

        detail::Graph m_G; //!< The graph

        unsigned int m_n_particles_old;                //!< Number of local particles in the old configuration
        detail::AABBTree m_aabb_tree_old;              //!< Locality lookup for old configuration

        GlobalVector<Scalar4> m_postype_backup;         //!< Old local positions
        GlobalVector<Scalar4> m_orientation_backup;     //!< Old local orientations
        GlobalVector<Scalar> m_diameter_backup;         //!< Old local diameters
        GlobalVector<Scalar> m_charge_backup;           //!< Old local charges
        GlobalVector<unsigned int> m_tag_backup;        //!< Old local tags

        #ifndef ENABLE_TBB
        std::set<std::pair<unsigned int, unsigned int> > m_overlap;   //!< A local vector of particle pairs due to overlap
        std::map<std::pair<unsigned int, unsigned int>,float > m_energy_old_old;    //!< Energy of interaction old-old
        std::map<std::pair<unsigned int, unsigned int>,float > m_energy_new_old;    //!< Energy of interaction old-old
        #else
        tbb::concurrent_unordered_set<std::pair<unsigned int, unsigned int> > m_overlap;
        tbb::concurrent_unordered_map<std::pair<unsigned int, unsigned int>,float > m_energy_old_old;
        tbb::concurrent_unordered_map<std::pair<unsigned int, unsigned int>,float > m_energy_new_old;
        #endif

        std::vector<unsigned int> m_ab_types;          //!< Two types used for swap move
        Scalar m_delta_mu;                             //!< Difference in chemical potential

        hpmc_clusters_counters_t m_count_total;                 //!< Total count since initialization
        hpmc_clusters_counters_t m_count_run_start;             //!< Count saved at run() start
        hpmc_clusters_counters_t m_count_step_start;            //!< Count saved at the start of the last step

        //! Check overlaps of a particle with depletants
        inline void checkDepletantOverlap(unsigned int i, vec3<Scalar> pos_i,
            Shape shape_i, unsigned int typ_i, unsigned int tag_i,
            const Scalar4 *h_postype_backup, const Scalar4 *h_orientation_backup, const unsigned int *h_tag_backup,
            unsigned int *h_overlaps, unsigned int timestep, const quat<Scalar> q, const vec3<Scalar> pivot, bool line);

        //! Save current state of particle data
        virtual void backupState(const std::map<unsigned int, unsigned int>& map);

        //! Find interactions between particles due to overlap and depletion interaction
        /*! \param timestep Current time step
            \param map Map to lookup new tag from old tag
        */
        virtual void findInteractions(unsigned int timestep, const quat<Scalar> q, const vec3<Scalar> pivot, bool line);

        //! Determine connected components of the interaction graph
        #ifdef ENABLE_TBB
        virtual void connectedComponents(unsigned int N, std::vector<tbb::concurrent_vector<unsigned int> >& clusters);
        #else
        virtual void connectedComponents(unsigned int N, std::vector<std::vector<unsigned int> >& clusters);
        #endif

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

    // initialize memory
    GlobalVector<Scalar4>(1,this->m_exec_conf).swap(m_postype_backup);
    TAG_ALLOCATION(m_postype_backup);
    GlobalVector<Scalar4>(1,this->m_exec_conf).swap(m_orientation_backup);
    TAG_ALLOCATION(m_orientation_backup);
    GlobalVector<Scalar>(1,this->m_exec_conf).swap(m_charge_backup);
    TAG_ALLOCATION(m_charge_backup);
    GlobalVector<Scalar>(1,this->m_exec_conf).swap(m_diameter_backup);
    TAG_ALLOCATION(m_diameter_backup);
    GlobalVector<unsigned int>(1,this->m_exec_conf).swap(m_tag_backup);
    TAG_ALLOCATION(m_tag_backup);
    }

template< class Shape >
UpdaterClusters<Shape>::~UpdaterClusters()
    {
    m_exec_conf->msg->notice(5) << "Destroying UpdaterClusters" << std::endl;
    }

/*! \param i The particle id in the list
    \param pos_i Particle position being tested
    \param shape_i Particle shape (including orientation) being tested
    \param typ_i Type of the particle being tested
    \param tag_i Tag of particle i
    \param h_postype Pointer to GPUArray containing particle positions
    \param h_orientation Pointer to GPUArray containing particle orientations
    \param h_overlaps Pointer to GPUArray containing interaction matrix
    \param map map from index (old config) to tag
    \param timestep the timestep

    Particle positions are checked against a randomly generated set of depletant positions.

    NOTE: To avoid numerous acquires and releases of GPUArrays, data pointers are passed directly into this const function.
    */
template<class Shape>
inline void UpdaterClusters<Shape>::checkDepletantOverlap(unsigned int i, vec3<Scalar> pos_i,
    Shape shape_i, unsigned int typ_i, unsigned int tag_i,
    const Scalar4 *h_postype_backup, const Scalar4 *h_orientation_backup, const unsigned int *h_tag_backup,
    unsigned int *h_overlaps, unsigned int timestep, const quat<Scalar> q, const vec3<Scalar> pivot, bool line)
    {
    unsigned int ndim = this->m_sysdef->getNDimensions();

    // access parameters
    auto& params = m_mc->getParams();
    auto& image_list = m_mc->updateImageList();
    const unsigned int n_images = image_list.size();

    Index2D overlap_idx = m_mc->getOverlapIndexer();

    detail::AABB aabb_i_local = shape_i.getAABB(vec3<Scalar>(0,0,0));

    // get image of particle i after transformation
    const BoxDim& box = m_pdata->getGlobalBox();
    int3 img_i;
    vec3<Scalar> pos_i_transf = pos_i;
    if (line)
        {
        pos_i_transf = lineReflection(pos_i, pivot, q);
        }
    else
        {
        pos_i_transf = pivot-(pos_i_transf-pivot);
        }
    img_i = box.getImage(pos_i_transf);

    #ifdef ENABLE_TBB
    tbb::parallel_for(tbb::blocked_range<unsigned int>(0, this->m_pdata->getNTypes()),
        [=, &shape_i](const tbb::blocked_range<unsigned int>& x) {
    for (unsigned int type_a = x.begin(); type_a != x.end(); ++type_a)
    #else
    for (unsigned int type_a = 0; type_a < this->m_pdata->getNTypes(); ++type_a)
    #endif
        {
        #ifdef ENABLE_TBB
        tbb::parallel_for(tbb::blocked_range<unsigned int>(type_a, this->m_pdata->getNTypes()),
            [=, &shape_i](const tbb::blocked_range<unsigned int>& w) {
        for (unsigned int type_b = w.begin(); type_b != w.end(); ++type_b)
        #else
        for (unsigned int type_b = type_a; type_b < this->m_pdata->getNTypes(); ++type_b)
        #endif
            {
            if (this->m_mc->getDepletantFugacity(type_a,type_b) == 0.0
                || !h_overlaps[overlap_idx(type_a, typ_i)]
                || !h_overlaps[overlap_idx(type_b, typ_i)])
                continue;

            std::vector<vec3<Scalar> > pos_j;
            std::vector<quat<Scalar> > orientation_j;
            std::vector<unsigned int> type_j;
            std::vector<unsigned int> tag_j;

            bool repulsive = this->m_mc->getDepletantFugacity(type_a,type_b) < 0.0;

            if (repulsive)
                throw std::runtime_error("Negative fugacities not supported in UpdaterClusters.\n");

            // find neighbors whose circumspheres overlap particle i's circumsphere in the old configuration
            // Here, circumsphere refers to the sphere around the depletant-excluded volume

            Shape tmp_a(quat<Scalar>(), params[type_a]);
            Shape tmp_b(quat<Scalar>(), params[type_b]);
            Scalar d_dep_a = tmp_a.getCircumsphereDiameter();
            Scalar d_dep_b = tmp_b.getCircumsphereDiameter();

            // the relevant search radius is the one for the larger depletant
            Scalar d_dep_search = std::max(d_dep_a, d_dep_b);

            // we're sampling in the larger volume, so that it strictly covers the insertion volume of
            // the smaller depletant
            Scalar r_dep_sample = 0.5*d_dep_search;

            // get AABB and extend
            vec3<Scalar> lower = aabb_i_local.getLower();
            vec3<Scalar> upper = aabb_i_local.getUpper();
            lower.x -= d_dep_search; lower.y -= d_dep_search; lower.z -= d_dep_search;
            upper.x += d_dep_search; upper.y += d_dep_search; upper.z += d_dep_search;
            detail::AABB aabb_local = detail::AABB(lower,upper);

            // All image boxes (including the primary)
            for (unsigned int cur_image = 0; cur_image < n_images; cur_image++)
                {
                vec3<Scalar> pos_i_image = pos_i + image_list[cur_image];
                detail::AABB aabb = aabb_local;
                aabb.translate(pos_i_image);

                // stackless search
                for (unsigned int cur_node_idx = 0; cur_node_idx < this->m_aabb_tree_old.getNumNodes(); cur_node_idx++)
                    {
                    if (detail::overlap(this->m_aabb_tree_old.getNodeAABB(cur_node_idx), aabb))
                        {
                        if (this->m_aabb_tree_old.isNodeLeaf(cur_node_idx))
                            {
                            for (unsigned int cur_p = 0; cur_p < this->m_aabb_tree_old.getNodeNumParticles(cur_node_idx); cur_p++)
                                {
                                // read in its position and orientation
                                unsigned int j = this->m_aabb_tree_old.getNodeParticle(cur_node_idx, cur_p);

                                if (tag_i == h_tag_backup[j] && cur_image == 0) continue;

                                // load the position and orientation of the j particle
                                vec3<Scalar> pj = vec3<Scalar>(h_postype_backup[j]);
                                unsigned int typ_j = __scalar_as_int(h_postype_backup[j].w);
                                Shape shape_j(quat<Scalar>(h_orientation_backup[j]), params[typ_j]);

                                // check excluded volume overlap
                                bool overlap_excluded = h_overlaps[overlap_idx(type_a,typ_j)] &&
                                    h_overlaps[overlap_idx(type_b,typ_j)] &&
                                    excludedVolumeOverlap(shape_i, shape_j, pj-pos_i_image, r_dep_sample,
                                    ndim, detail::SamplingMethod::accurate);

                                if (overlap_excluded)
                                    {
                                    // cache the translated position of particle j. If i's image is cur_image, then j's
                                    // image is the negative of that (and we use i's untranslated position below)
                                    pos_j.push_back(pj-image_list[cur_image]);
                                    orientation_j.push_back(shape_j.orientation);
                                    type_j.push_back(typ_j);
                                    tag_j.push_back(h_tag_backup[j]);
                                    }
                                }
                            }
                        }
                    else
                        {
                        // skip ahead
                        cur_node_idx += this->m_aabb_tree_old.getNodeSkip(cur_node_idx);
                        }
                    }  // end loop over AABB nodes
                } // end loop over images

            // now, we have a list of intersecting spheres, sample in the union of intersection volumes
            // we sample from their union by checking if any generated position falls in the intersection
            // of their AABBs, only accepting it if it was generated from neighbor j_min

            // compute insertion volumes
            std::vector<Scalar> V_all;
            Scalar V_tot(0.0);

            for (unsigned int k = 0; k < pos_j.size(); ++k)
                {
                Shape shape_j(orientation_j[k], params[type_j[k]]);
                Scalar V = getSamplingVolumeIntersection(shape_i, shape_j, pos_j[k] - pos_i, r_dep_sample,
                    ndim, detail::SamplingMethod::accurate);
                V_all.push_back(V);
                V_tot += V;
                }

            // chooose the number of depletants in the intersection volume
            hoomd::PoissonDistribution<Scalar> poisson(
                std::abs(this->m_mc->getDepletantFugacity(type_a,type_b))*V_tot);
            unsigned int ntypes = this->m_pdata->getNTypes();
            hoomd::RandomGenerator rng_num(hoomd::RNGIdentifier::HPMCDepletantNumClusters,
                this->m_seed, type_a*ntypes+type_b, i, timestep);

            unsigned int n = poisson(rng_num);

            // try inserting in the overlap volume
            unsigned int n_intersect = pos_j.size();

            std::vector<unsigned int> storage_sz;
            std::vector< std::vector<typename Shape::depletion_storage_type> > temp_storage;

            // temporary storage for depletant insertions
            for (unsigned int k = 0; k < n_intersect; ++k)
                {
                Shape shape_j(orientation_j[k], params[type_j[k]]);

                // allocate
                unsigned int nelem = allocateDepletionTemporaryStorage(shape_i,
                    shape_j, pos_j[k] - pos_i, r_dep_sample,
                    ndim, detail::SamplingMethod::accurate);
                std::vector<typename Shape::depletion_storage_type> storage(nelem);

                // initialize
                unsigned int sz = initializeDepletionTemporaryStorage(shape_i,
                    shape_j, pos_j[k] - pos_i, r_dep_sample,
                    ndim, &storage.front(), V_all[k], detail::SamplingMethod::accurate);

                temp_storage.push_back(storage);
                storage_sz.push_back(sz);
                }

            // for every depletant
            #ifdef ENABLE_TBB
            tbb::parallel_for(tbb::blocked_range<unsigned int>(0, (unsigned int)n),
                [=, &shape_i,
                    &pos_j, &orientation_j, &type_j, &V_all,
                    &temp_storage, &storage_sz](const tbb::blocked_range<unsigned int>& t) {
            for (unsigned int l = t.begin(); l != t.end(); ++l)
            #else
            for (unsigned int l = 0; l < n; ++l)
            #endif
                {
                if (!n_intersect)
                    {
                    // no insertion volume
                    continue;
                    }
                hoomd::RandomGenerator my_rng(
                    (hoomd::RNGIdentifier::HPMCDepletantsClusters)^(this->m_seed), i,
                        type_a+type_b*ntypes, timestep, l);

                Scalar V_rand = hoomd::UniformDistribution<Scalar>(0.0, V_tot)(my_rng);

                Scalar V_sum(0.0);
                unsigned int k;
                for (k = 0; k < n_intersect; ++k)
                    {
                    Scalar V = V_all[k];
                    V_sum += V;
                    if (V_rand < V_sum)
                        break;
                    }

                // rejection sampling
                Shape shape_j(orientation_j[k], params[type_j[k]]);

                vec3<OverlapReal> dr_test;
                if (!sampleInExcludedVolumeIntersection(my_rng, shape_i, shape_j,
                    pos_j[k] - pos_i, r_dep_sample,
                    dr_test, ndim, storage_sz[k], &temp_storage[k].front(),
                    detail::SamplingMethod::accurate))
                    {
                    continue;
                    }

                Shape shape_test_a(quat<Scalar>(), params[type_a]);
                Shape shape_test_b(quat<Scalar>(), params[type_b]);
                quat<Scalar> o;
                if (shape_test_a.hasOrientation() || shape_test_b.hasOrientation())
                    {
                    o = generateRandomOrientation(my_rng, ndim);
                    }
                if (shape_test_a.hasOrientation())
                    shape_test_a.orientation = o;
                if (shape_test_b.hasOrientation())
                    shape_test_b.orientation = o;

                // check if depletant falls in other intersection volumes
                bool active = true;

                for (unsigned int m = 0; m < k; ++m)
                    {
                    Shape shape_m(orientation_j[m], params[type_j[m]]);

                    if (isPointInExcludedVolumeIntersection(shape_i, shape_m, pos_j[m] - pos_i,
                        r_dep_sample, dr_test, ndim, detail::SamplingMethod::accurate))
                        {
                        active = false;
                        break;
                        }
                    }

                if (!active)
                    {
                    // if we cannot insert, ignore
                    continue;
                    }

                // Check if the configuration of particle i generates an overlap
                bool overlap_i_a = false;
                bool overlap_i_b = false;

                    {
                    OverlapReal rsq = dot(dr_test,dr_test);
                    OverlapReal DaDb = shape_test_a.getCircumsphereDiameter() + shape_i.getCircumsphereDiameter();
                    bool circumsphere_overlap = (rsq*OverlapReal(4.0) <= DaDb * DaDb);

                    if (h_overlaps[overlap_idx(type_a, typ_i)])
                        {
                        unsigned int err = 0;
                        if (circumsphere_overlap &&
                            test_overlap(dr_test, shape_i, shape_test_a, err))
                            {
                            overlap_i_a = true;
                            }
                        }
                    }

                if (type_b == type_a)
                    {
                    overlap_i_b = overlap_i_a;
                    }
                else
                    {
                    OverlapReal rsq = dot(dr_test,dr_test);
                    OverlapReal DaDb = shape_test_b.getCircumsphereDiameter() + shape_i.getCircumsphereDiameter();
                    bool circumsphere_overlap = (rsq*OverlapReal(4.0) <= DaDb * DaDb);

                    if (h_overlaps[overlap_idx(type_b, typ_i)])
                        {
                        unsigned int err = 0;
                        if (circumsphere_overlap &&
                            test_overlap(dr_test, shape_i, shape_test_b, err))
                            {
                            overlap_i_b = true;
                            }
                        }
                    }
                if (!overlap_i_a && !overlap_i_b)
                    {
                    // ignore
                    continue;
                    }

                // Check if the old configuration of particle i generates an overlap with the transformed depletant
                bool overlap_transf_a = false;
                bool overlap_transf_b = false;

                // transform the test particle
                vec3<Scalar> pos_test_transf = pos_i + vec3<Scalar>(dr_test);
                if (line)
                    {
                    pos_test_transf = lineReflection(pos_test_transf, pivot, q);
                    }
                else
                    {
                    pos_test_transf = pivot - (pos_test_transf - pivot);
                    }

                // wrap back into into i's image (after transformation)
                pos_test_transf = box.shift(pos_test_transf,-img_i);
                vec3<OverlapReal> dr_test_transf = pos_test_transf - pos_i;

                Shape shape_test_transf_a(shape_test_a.orientation, params[type_a]);
                if (shape_test_a.hasOrientation())
                    shape_test_transf_a.orientation = q*shape_test_transf_a.orientation;
                Shape shape_test_transf_b(shape_test_b.orientation, params[type_b]);
                if (shape_test_b.hasOrientation())
                    shape_test_transf_b.orientation = q*shape_test_transf_b.orientation;

                OverlapReal rsq = dot(dr_test_transf,dr_test_transf);
                OverlapReal DaDb = shape_test_a.getCircumsphereDiameter() + shape_i.getCircumsphereDiameter();
                bool circumsphere_overlap = (rsq*OverlapReal(4.0) <= DaDb * DaDb);

                if (h_overlaps[overlap_idx(type_a, typ_i)])
                    {
                    unsigned int err = 0;
                    if (circumsphere_overlap &&
                        test_overlap(dr_test_transf, shape_i, shape_test_transf_a, err))
                        {
                        overlap_transf_a = true;
                        }
                    }

                if (type_a == type_b)
                    {
                    overlap_transf_b = overlap_transf_a;
                    }
                else
                    {
                    OverlapReal rsq = dot(dr_test_transf,dr_test_transf);
                    OverlapReal DaDb = shape_test_b.getCircumsphereDiameter() + shape_i.getCircumsphereDiameter();
                    bool circumsphere_overlap = (rsq*OverlapReal(4.0) <= DaDb * DaDb);

                    if (h_overlaps[overlap_idx(type_b, typ_i)])
                        {
                        unsigned int err = 0;
                        if (circumsphere_overlap &&
                            test_overlap(dr_test_transf, shape_i, shape_test_transf_b, err))
                            {
                            overlap_transf_b = true;
                            }
                        }
                    }

                if ((overlap_transf_a || !overlap_i_a) && (overlap_transf_b || !overlap_i_b))
                    {
                    // this depletant did never exist
                    continue;
                    }

                unsigned int min_tag = tag_i;

                for (unsigned int m = k; m < n_intersect; ++m)
                    {
                    unsigned int type_m = type_j[m];
                    Shape shape_m(orientation_j[m], params[type_m]);
                    vec3<Scalar> r_mk = pos_j[m] - pos_i - vec3<Scalar>(dr_test);

                    unsigned int err = 0;

                    // check circumsphere overlap
                    OverlapReal rsq = dot(r_mk,r_mk);
                    OverlapReal DaDb = shape_test_a.getCircumsphereDiameter() + shape_m.getCircumsphereDiameter();
                    bool circumsphere_overlap = (rsq*OverlapReal(4.0) <= DaDb * DaDb);

                    bool overlap_j_a = h_overlaps[overlap_idx(type_a,type_m)]
                        && circumsphere_overlap
                        && test_overlap(r_mk, shape_test_a, shape_m, err);

                    bool overlap_j_b;
                    if (type_a == type_b)
                        {
                        overlap_j_b = overlap_j_a;
                        }
                    else
                        {
                        DaDb = shape_test_b.getCircumsphereDiameter() + shape_m.getCircumsphereDiameter();
                        circumsphere_overlap = (rsq*OverlapReal(4.0) <= DaDb * DaDb);

                        overlap_j_b = h_overlaps[overlap_idx(type_b,type_m)]
                            && circumsphere_overlap
                            && test_overlap(r_mk, shape_test_b, shape_m, err);
                        }

                    // additive depletants
                    if ((overlap_i_a && overlap_j_b) || (overlap_i_b && overlap_j_a))
                        {
                        if (tag_j[m] < min_tag)
                            {
                            min_tag = tag_j[m];
                            break;
                            }
                        }
                    }

                if (min_tag == tag_i)
                    {
                    // does the transformed depletant overlap with any other particles?
                    bool overlap_old = false;

                    // get AABB
                    detail::AABB aabb_a_local = shape_test_transf_a.getAABB(vec3<Scalar>(0,0,0));

                    // All image boxes (including the primary)
                    for (unsigned int cur_image = 0; cur_image < n_images; cur_image++)
                        {
                        vec3<Scalar> pos_test_image = pos_test_transf + image_list[cur_image];
                        detail::AABB aabb = aabb_a_local;
                        aabb.translate(pos_test_image);

                        // stackless search
                        for (unsigned int cur_node_idx = 0; cur_node_idx < this->m_aabb_tree_old.getNumNodes(); cur_node_idx++)
                            {
                            if (detail::overlap(this->m_aabb_tree_old.getNodeAABB(cur_node_idx), aabb))
                                {
                                if (this->m_aabb_tree_old.isNodeLeaf(cur_node_idx))
                                    {
                                    for (unsigned int cur_p = 0; cur_p < this->m_aabb_tree_old.getNodeNumParticles(cur_node_idx); cur_p++)
                                        {
                                        // read in its position and orientation
                                        unsigned int j = this->m_aabb_tree_old.getNodeParticle(cur_node_idx, cur_p);

                                        // load the position and orientation of the j particle
                                        vec3<Scalar> pj = vec3<Scalar>(h_postype_backup[j]);
                                        unsigned int typ_j = __scalar_as_int(h_postype_backup[j].w);
                                        Shape shape_j(quat<Scalar>(h_orientation_backup[j]), params[typ_j]);

                                        // check excluded volume overlap
                                        vec3<Scalar> r_jk(pos_test_image - pj);
                                        OverlapReal rsq = dot(r_jk,r_jk);
                                        OverlapReal DaDb = shape_test_transf_a.getCircumsphereDiameter() + shape_j.getCircumsphereDiameter();
                                        unsigned int err = 0;
                                        if (h_overlaps[overlap_idx(type_a, typ_j)] &&
                                            (rsq*OverlapReal(4.0) <= DaDb * DaDb) &&
                                            test_overlap(r_jk, shape_j, shape_test_transf_a, err) &&
                                            ((type_a == type_b) || (h_overlaps[overlap_idx(type_b, typ_j)] &&
                                            test_overlap(r_jk, shape_j, shape_test_transf_b, err))))
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
                                cur_node_idx += this->m_aabb_tree_old.getNodeSkip(cur_node_idx);
                                }
                            }  // end loop over AABB nodes
                        if (overlap_old)
                            break;
                        } // end loop over images

                    if (type_a != type_b)
                        {
                        detail::AABB aabb_b_local = shape_test_transf_b.getAABB(vec3<Scalar>(0,0,0));
                        for (unsigned int cur_image = 0; cur_image < n_images; cur_image++)
                            {
                            vec3<Scalar> pos_test_image = pos_test_transf + image_list[cur_image];
                            detail::AABB aabb = aabb_b_local;
                            aabb.translate(pos_test_image);

                            // stackless search
                            for (unsigned int cur_node_idx = 0; cur_node_idx < this->m_aabb_tree_old.getNumNodes(); cur_node_idx++)
                                {
                                if (detail::overlap(this->m_aabb_tree_old.getNodeAABB(cur_node_idx), aabb))
                                    {
                                    if (this->m_aabb_tree_old.isNodeLeaf(cur_node_idx))
                                        {
                                        for (unsigned int cur_p = 0; cur_p < this->m_aabb_tree_old.getNodeNumParticles(cur_node_idx); cur_p++)
                                            {
                                            // read in its position and orientation
                                            unsigned int j = this->m_aabb_tree_old.getNodeParticle(cur_node_idx, cur_p);

                                            // load the position and orientation of the j particle
                                            vec3<Scalar> pj = vec3<Scalar>(h_postype_backup[j]);
                                            unsigned int typ_j = __scalar_as_int(h_postype_backup[j].w);
                                            Shape shape_j(quat<Scalar>(h_orientation_backup[j]), params[typ_j]);

                                            // check excluded volume overlap
                                            vec3<Scalar> r_jk(pos_test_image - pj);
                                            OverlapReal rsq = dot(r_jk,r_jk);
                                            OverlapReal DaDb = shape_test_transf_b.getCircumsphereDiameter() + shape_j.getCircumsphereDiameter();
                                            unsigned int err = 0;
                                            if (h_overlaps[overlap_idx(type_b, typ_j)] &&
                                                (rsq*OverlapReal(4.0) <= DaDb * DaDb) &&
                                                test_overlap(r_jk, shape_j, shape_test_transf_b, err) &&
                                                h_overlaps[overlap_idx(type_b, typ_j)] &&
                                                test_overlap(r_jk, shape_j, shape_test_transf_b, err))
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
                                    cur_node_idx += this->m_aabb_tree_old.getNodeSkip(cur_node_idx);
                                    }
                                }  // end loop over AABB nodes
                            if (overlap_old)
                                break;
                            } // end loop over images
                        }

                    if (overlap_old)
                        continue;

                    for (unsigned int m = k; m < n_intersect; ++m)
                        {
                        unsigned int type_m = type_j[m];
                        Shape shape_m(orientation_j[m], params[type_m]);
                        vec3<Scalar> r_mk = pos_j[m] - pos_i - vec3<Scalar>(dr_test);

                        unsigned int err = 0;

                        // check circumsphere overlap
                        OverlapReal rsq = dot(r_mk,r_mk);
                        OverlapReal DaDb = shape_test_a.getCircumsphereDiameter() + shape_m.getCircumsphereDiameter();
                        bool circumsphere_overlap = (rsq*OverlapReal(4.0) <= DaDb * DaDb);

                        bool overlap_j_a = h_overlaps[overlap_idx(type_a,type_m)]
                            && circumsphere_overlap
                            && test_overlap(r_mk, shape_test_a, shape_m, err);

                        bool overlap_j_b;
                        if (type_a == type_b)
                            {
                            overlap_j_b = overlap_j_a;
                            }
                        else
                            {
                            DaDb = shape_test_b.getCircumsphereDiameter() + shape_m.getCircumsphereDiameter();
                            circumsphere_overlap = (rsq*OverlapReal(4.0) <= DaDb * DaDb);

                            overlap_j_b = h_overlaps[overlap_idx(type_b,type_m)]
                                && circumsphere_overlap
                                && test_overlap(r_mk, shape_test_b, shape_m, err);
                            }

                        // additive depletants
                        if ((overlap_i_a && !overlap_transf_a && overlap_j_b) || (overlap_i_b && !overlap_transf_b & overlap_j_a))
                            {
                            // add bond
                            this->m_overlap.insert(std::make_pair(tag_i,tag_j[m]));
                            }
                        }
                    } // end loop over intersections
                } // end loop over depletants
            #ifdef ENABLE_TBB
                });
            #endif
            } // end loop over type_b
        #ifdef ENABLE_TBB
            });
        #endif
        } // end loop over type_a
    #ifdef ENABLE_TBB
        });
    #endif
    }

template< class Shape >
void UpdaterClusters<Shape>::findInteractions(unsigned int timestep, const quat<Scalar> q, const vec3<Scalar> pivot, bool line)
    {
    if (m_prof)
        m_prof->push(m_exec_conf,"Interactions");

    // access parameters
    auto& params = m_mc->getParams();

    // update the image list
    auto& image_list = m_mc->updateImageList();

    // minimum AABB extent
    Scalar min_core_diameter = m_mc->getMinCoreDiameter();

    Index2D overlap_idx = m_mc->getOverlapIndexer();
    ArrayHandle<unsigned int> h_overlaps(m_mc->getInteractionMatrix(), access_location::host, access_mode::read);

    // clear the local bond and rejection lists
    m_overlap.clear();

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

    // access particle data
    ArrayHandle<Scalar4> h_postype(m_pdata->getPositions(), access_location::host, access_mode::read);
    ArrayHandle<Scalar4> h_orientation(m_pdata->getOrientationArray(), access_location::host, access_mode::read);
    ArrayHandle<Scalar> h_diameter(m_pdata->getDiameters(), access_location::host, access_mode::read);
    ArrayHandle<Scalar> h_charge(m_pdata->getCharges(), access_location::host, access_mode::read);
    ArrayHandle<unsigned int> h_tag(m_pdata->getTags(), access_location::host, access_mode::read);

    ArrayHandle<Scalar4> h_postype_backup(m_postype_backup, access_location::host, access_mode::read);
    ArrayHandle<Scalar4> h_orientation_backup(m_orientation_backup, access_location::host, access_mode::read);
    ArrayHandle<unsigned int> h_tag_backup(m_tag_backup, access_location::host, access_mode::read);
    ArrayHandle<Scalar> h_diameter_backup(m_diameter_backup, access_location::host, access_mode::read);
    ArrayHandle<Scalar> h_charge_backup(m_charge_backup, access_location::host, access_mode::read);

    if (patch)
        {
        // test old configuration against itself
        #ifdef ENABLE_TBB
        tbb::parallel_for((unsigned int)0,m_n_particles_old, [&](unsigned int i)
        #else
        for (unsigned int i = 0; i < m_n_particles_old; ++i)
        #endif
            {
            unsigned int typ_i = __scalar_as_int(h_postype_backup.data[i].w);

            vec3<Scalar> pos_i(h_postype_backup.data[i]);
            quat<Scalar> orientation_i(h_orientation_backup.data[i]);

            Scalar d_i(h_diameter_backup.data[i]);
            Scalar charge_i(h_charge_backup.data[i]);

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

                                if (h_tag_backup.data[i] == h_tag_backup.data[j] && cur_image == 0) continue;

                                // load the position and orientation of the j particle
                                vec3<Scalar> pos_j = vec3<Scalar>(h_postype_backup.data[j]);
                                unsigned int typ_j = __scalar_as_int(h_postype_backup.data[j].w);

                                // put particles in coordinate system of particle i
                                vec3<Scalar> r_ij = pos_j - pos_i_image;
                                Scalar rsq_ij = dot(r_ij, r_ij);

                                Scalar rcut_ij = r_cut_patch + extent_i + 0.5*patch->getAdditiveCutoff(typ_j);

                                if (rsq_ij <= rcut_ij*rcut_ij)
                                    {
                                    // the particle pair
                                    auto p = std::make_pair(h_tag_backup.data[i],h_tag_backup.data[j]);

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
                                                        quat<float>(h_orientation_backup.data[j]),
                                                        h_diameter_backup.data[j],
                                                        h_charge_backup.data[j]);

                                    // update map
                                    m_energy_old_old[p] = U;
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
        #ifdef ENABLE_TBB
            );
        #endif
        }

    // loop over new configuration
    #ifdef ENABLE_TBB
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

                            if (h_tag.data[i] == h_tag_backup.data[j] && cur_image == 0) continue;

                            // load the position and orientation of the j particle
                            vec3<Scalar> pos_j = vec3<Scalar>(h_postype_backup.data[j]);
                            unsigned int typ_j = __scalar_as_int(h_postype_backup.data[j].w);
                            Shape shape_j(quat<Scalar>(h_orientation_backup.data[j]), params[typ_j]);

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
                                    // add connection
                                    m_overlap.insert(std::make_pair(h_tag.data[i],h_tag_backup.data[j]));
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

                                if (h_tag.data[i] == h_tag_backup.data[j] && cur_image == 0) continue;

                                vec3<Scalar> pos_j(h_postype_backup.data[j]);
                                unsigned int typ_j = __scalar_as_int(h_postype_backup.data[j].w);

                                // put particles in coordinate system of particle i
                                vec3<Scalar> r_ij = pos_j - pos_i_image;

                                // check for excluded volume sphere overlap
                                Scalar rsq_ij = dot(r_ij, r_ij);

                                Scalar rcut_ij = r_cut_patch + extent_i + 0.5*patch->getAdditiveCutoff(typ_j);

                                if (rsq_ij <= rcut_ij*rcut_ij)
                                    {
                                    auto p = std::make_pair(h_tag.data[i], h_tag_backup.data[j]);

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
                                                            quat<float>(h_orientation_backup.data[j]),
                                                            h_diameter_backup.data[j],
                                                            h_charge_backup.data[j]);

                                    // update map
                                    m_energy_new_old[p] = U;
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
    #ifdef ENABLE_TBB
        );
    #endif

    /*
     * Depletants
     */

    bool has_depletants = false;
    for (unsigned int i = 0; i < this->m_pdata->getNTypes(); ++i)
        {
        for (unsigned int j = 0; j < this->m_pdata->getNTypes(); ++j)
            {
            if (this->m_mc->getDepletantFugacity(i,j) != 0.0)
                {
                has_depletants = true;
                break;
                }
            }
        }

    if (!has_depletants)
        return;

    // test old configuration against itself
    #ifdef ENABLE_TBB
    tbb::parallel_for((unsigned int)0,this->m_n_particles_old, [&](unsigned int i) {
    #else
    for (unsigned int i = 0; i < this->m_n_particles_old; ++i)
    #endif
        {
        unsigned int typ_i = __scalar_as_int(h_postype_backup.data[i].w);

        vec3<Scalar> pos_i(h_postype_backup.data[i]);
        quat<Scalar> orientation_i(h_orientation_backup.data[i]);
        Shape shape_i(orientation_i, params[typ_i]);
        unsigned int tag_i = h_tag_backup.data[i];

        checkDepletantOverlap(i, pos_i, shape_i, typ_i, tag_i,
            h_postype_backup.data, h_orientation_backup.data, h_tag_backup.data,
            h_overlaps.data, timestep, q, pivot, line);
        }
    #ifdef ENABLE_TBB
        });
    #endif

    if (this->m_prof)
        this->m_prof->pop(this->m_exec_conf);
    }

template<class Shape>
void UpdaterClusters<Shape>::backupState(const std::map<unsigned int, unsigned int>& map)

    {
    unsigned int nptl = m_pdata->getN()+m_pdata->getNGhosts();

    // resize as necessary
    m_postype_backup.resize(nptl);
    m_orientation_backup.resize(nptl);
    m_diameter_backup.resize(nptl);
    m_charge_backup.resize(nptl);
    m_tag_backup.resize(nptl);

        {
        ArrayHandle<Scalar4> h_postype(m_pdata->getPositions(), access_location::host, access_mode::read);
        ArrayHandle<Scalar4> h_orientation(m_pdata->getOrientationArray(), access_location::host, access_mode::read);
        ArrayHandle<Scalar> h_diameter(m_pdata->getDiameters(), access_location::host, access_mode::read);
        ArrayHandle<Scalar> h_charge(m_pdata->getCharges(), access_location::host, access_mode::read);
        ArrayHandle<unsigned int> h_tag(m_pdata->getTags(), access_location::host, access_mode::read);

        ArrayHandle<Scalar4> h_postype_backup(m_postype_backup, access_location::host, access_mode::overwrite);
        ArrayHandle<Scalar4> h_orientation_backup(m_orientation_backup, access_location::host, access_mode::overwrite);
        ArrayHandle<unsigned int> h_tag_backup(m_tag_backup, access_location::host, access_mode::overwrite);
        ArrayHandle<Scalar> h_diameter_backup(m_diameter_backup, access_location::host, access_mode::overwrite);
        ArrayHandle<Scalar> h_charge_backup(m_charge_backup, access_location::host, access_mode::overwrite);

        // copy over data
        for (unsigned int i = 0; i < nptl; ++i)
            {
            h_postype_backup.data[i] = h_postype.data[i];
            h_orientation_backup.data[i] = h_orientation.data[i];
            h_diameter_backup.data[i] = h_diameter.data[i];
            h_charge_backup.data[i] = h_charge.data[i];
            auto it = map.find(h_tag.data[i]);
            assert(it != map.end());
            h_tag_backup.data[i] = it->second;
            }
        }
    }

template<class Shape>
#ifdef ENABLE_TBB
void UpdaterClusters<Shape>::connectedComponents(unsigned int N, std::vector<tbb::concurrent_vector<unsigned int> >& clusters)
#else
void UpdaterClusters<Shape>::connectedComponents(unsigned int N, std::vector<std::vector<unsigned int> >& clusters)
#endif
    {
    if (this->m_prof) this->m_prof->push("connected components");

    // compute connected components
    clusters.clear();
    m_G.connectedComponents(clusters);
    if (this->m_prof) this->m_prof->pop();
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

    if (m_prof) m_prof->push(m_exec_conf,"Transform");

    // generate the move, select a pivot
    hoomd::RandomGenerator rng(hoomd::RNGIdentifier::UpdaterClusters, timestep, this->m_seed);
    BoxDim box = m_pdata->getGlobalBox();
    vec3<Scalar> pivot(0,0,0);

    bool swap = false && m_ab_types.size() && (hoomd::detail::generate_canonical<Scalar>(rng) <= m_swap_move_ratio);

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
            unsigned int i = hoomd::UniformIntDistribution(2)(rng);
            if (i == 0)
                n = vec3<Scalar>(1,0,0);
            else if (i == 1)
                n = vec3<Scalar>(0,1,0);
            else if (i == 2)
                n = vec3<Scalar>(0,0,1);
            }
        else
            {
            // reflection around z axis
            n = vec3<Scalar>(0,0,1);
            }

        // line reflection
        q = quat<Scalar>(0,n);
        }

    // choose a pivot point
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

    SnapshotParticleData<Scalar> snap(m_pdata->getNGlobal());

    // obtain particle data from all ranks

    // save origin information
    Scalar3 origin = m_pdata->getOrigin();
    int3 origin_image = m_pdata->getOriginImage();

    // take a snapshot, and save tag->snap idx mapping

    // reset origin, so that snapshot positions match AABB tree positions
    m_pdata->resetOrigin();
    auto map = m_pdata->takeSnapshot(snap);

    // store backup of particle data
    backupState(map);

    #ifdef ENABLE_MPI
    if (m_comm)
        {
        // we need the particle tag -> snapshot idx map on all ranks
        bcast(map, 0, m_exec_conf->getMPICommunicator());
        }
    #endif

    // transform all particles on rank zero
    bool master = !m_exec_conf->getRank();

    // keep a backup copy
    SnapshotParticleData<Scalar> snap_old = snap;

    if (master)
        {
        // access parameters
        auto& params = m_mc->getParams();

        for (unsigned int i = 0; i < snap.size; ++i)
            {
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
                // wrap particle back into box, incrementing image flags
                int3 img = box.getImage(snap.pos[i]);
                snap.pos[i] = box.shift(snap.pos[i],-img);
                snap.image[i] += img;
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
    findInteractions(timestep, q, pivot, line);

    if (m_prof) m_prof->push(m_exec_conf,"Move");

    // collect interactions on rank 0
    #ifndef ENABLE_TBB
    std::vector< std::set<std::pair<unsigned int, unsigned int> > > all_overlap;

    std::vector< std::map<std::pair<unsigned int, unsigned int>, float> > all_energy_old_old;
    std::vector< std::map<std::pair<unsigned int, unsigned int>, float> > all_energy_new_old;
    #else
    std::vector< tbb::concurrent_unordered_set<std::pair<unsigned int, unsigned int> > > all_overlap;

    std::vector< tbb::concurrent_unordered_map<std::pair<unsigned int, unsigned int>, float> > all_energy_old_old;
    std::vector< tbb::concurrent_unordered_map<std::pair<unsigned int, unsigned int>, float> > all_energy_new_old;
    #endif

    #ifdef ENABLE_MPI
    if (m_comm)
        {
        // combine lists from different ranks
        gather_v(m_overlap, all_overlap, 0, m_exec_conf->getMPICommunicator());
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
            #ifdef ENABLE_TBB
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
            #ifdef ENABLE_TBB
                );
            #endif
            }

        if (m_prof)
            m_prof->pop();


        if (m_mc->getPatchInteraction())
            {
            // sum up interaction energies
            #ifdef ENABLE_TBB
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

            #ifdef ENABLE_TBB
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
            #ifdef ENABLE_TBB
                );
            #endif
            } // end if (patch)

        // compute connected components
        connectedComponents(snap.size, m_clusters);

        if (this->m_prof) this->m_prof->push("flip");

        // move every cluster independently
        m_count_total.n_clusters += m_clusters.size();

        for (unsigned int icluster = 0; icluster < m_clusters.size(); icluster++)
            {
            m_count_total.n_particles_in_clusters += m_clusters[icluster].size();

            // seed by id of first particle in cluster to make independent of cluster labeling
            hoomd::RandomGenerator rng_i(hoomd::RNGIdentifier::UpdaterClusters+1, this->m_seed, timestep,
                m_clusters[icluster][0]);

            bool flip = hoomd::detail::generate_canonical<float>(rng_i) <= m_flip_probability;

            // count number of A and B particles in old and new config
            bool reject = false;
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

                if (hoomd::detail::generate_canonical<float>(rng_i) > exp(NdelMu))
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
            else
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
    if (m_prof) m_prof->pop(m_exec_conf);

    m_mc->communicate(true);
    }


template < class Shape> void export_UpdaterClusters(pybind11::module& m, const std::string& name)
    {
    pybind11::class_< UpdaterClusters<Shape>, Updater, std::shared_ptr< UpdaterClusters<Shape> > >(m, name.c_str())
          .def( pybind11::init< std::shared_ptr<SystemDefinition>,
                         std::shared_ptr< IntegratorHPMCMono<Shape> >,
                         unsigned int >())
        .def("getCounters", &UpdaterClusters<Shape>::getCounters)
        .def_property("move_ratio", &UpdaterClusters<Shape>::getMoveRatio, &UpdaterClusters<Shape>::setMoveRatio)
        .def_property("flip_probability", &UpdaterClusters<Shape>::getFlipProbability, &UpdaterClusters<Shape>::setFlipProbability)
        .def_property("swap_move_ratio", &UpdaterClusters<Shape>::getSwapMoveRatio, &UpdaterClusters<Shape>::setSwapMoveRatio)
        .def_property("swap_type_pair", &UpdaterClusters<Shape>::getSwapTypePairStr, &UpdaterClusters<Shape>::setSwapTypePairStr)
        .def_property("delta_mu", &UpdaterClusters<Shape>::getDeltaMu, &UpdaterClusters<Shape>::setDeltaMu)
        .def_property_readonly("seed", &UpdaterClusters<Shape>::getSeed)
        ;
    }

inline void export_hpmc_clusters_counters(pybind11::module &m)
    {
    pybind11::class_< hpmc_clusters_counters_t >(m, "hpmc_clusters_counters_t")
        .def_property_readonly("pivot", [](const hpmc_clusters_counters_t &a)
                                              {
                                              pybind11::list result;
                                              result.append(a.pivot_accept_count);
                                              result.append(a.pivot_reject_count);
                                              return result;
                                              }
                              )
        .def_property_readonly("reflection", [](const hpmc_clusters_counters_t &a)
                                                   {
                                                   pybind11::list result;
                                                   result.append(a.reflection_accept_count);
                                                   result.append(a.reflection_reject_count);
                                                   return result;
                                                   }
                              )
        .def_property_readonly("swap", [](const hpmc_clusters_counters_t &a)
                                            {
                                            pybind11::list result;
                                            result.append(a.swap_accept_count);
                                            result.append(a.swap_reject_count);
                                            return result;
                                            }
                              )
        .def_readonly("clusters", &hpmc_clusters_counters_t::n_clusters)
        .def_readonly("particles", &hpmc_clusters_counters_t::n_particles_in_clusters)
        ;
    }

} // end namespace hpmc

#endif // _UPDATER_HPMC_CLUSTERS_
