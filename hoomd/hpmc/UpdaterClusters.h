// Copyright (c) 2009-2024 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

#ifndef _UPDATER_HPMC_CLUSTERS_
#define _UPDATER_HPMC_CLUSTERS_

/*! \file UpdaterBoxClusters.h
    \brief Declaration of UpdaterBoxClusters
*/

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

#if TBB_VERSION_MAJOR < 2021
#define ENABLE_TBB_TASK
#endif

#endif

namespace hoomd {

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
        Graph()
            {
            }

        inline Graph(unsigned int V);   // Constructor

        inline void resize(unsigned int V);

        inline void addEdge(unsigned int v, unsigned int w);

        #ifdef ENABLE_TBB_TASK
        inline void connectedComponents(std::vector<tbb::concurrent_vector<unsigned int> >& cc);
        #else
        inline void connectedComponents(std::vector<std::vector<unsigned int> >& cc);
        #endif

        #ifdef ENABLE_TBB_TASK
        void setTaskArena(std::shared_ptr<tbb::task_arena> task_arena)
            {
            m_task_arena = task_arena;
            }
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

        #ifdef ENABLE_TBB_TASK
        /// The TBB task arena
        std::shared_ptr<tbb::task_arena> m_task_arena;
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
    this->m_task_arena->execute([&]{
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
    }); // end task arena execute()
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

// method to add a directed edge
void Graph::addEdge(unsigned int v, unsigned int w)
    {
    adj.insert(std::make_pair(v,w));
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
        */
        UpdaterClusters(std::shared_ptr<SystemDefinition> sysdef,
                                     std::shared_ptr<Trigger> trigger,
                        std::shared_ptr<IntegratorHPMCMono<Shape> > mc);

        //! Destructor
        virtual ~UpdaterClusters();

        //! Take one timestep forward
        /*! \param timestep timestep at which update is being evaluated
        */
        virtual void update(uint64_t timestep);

        /// Set the move ratio
        void setMoveRatio(Scalar move_ratio)
            {
            m_move_ratio = move_ratio;
            }

        Scalar getMoveRatio()
            {
            return m_move_ratio;
            }

        /// Set the cluster flip probability
        void setFlipProbability(Scalar flip_probability)
            {
            m_flip_probability = flip_probability;
            }

        /// Get the cluster flip probability
        Scalar getFlipProbability()
            {
            return m_flip_probability;
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

    protected:
        std::shared_ptr< IntegratorHPMCMono<Shape> > m_mc; //!< HPMC integrator
        Scalar m_move_ratio;                        //!< Pivot/Reflection move ratio
        Scalar m_flip_probability;                  //!< Cluster flip probability

        unsigned int m_instance=0;                  //!< Unique ID for RNG seeding

        #ifdef ENABLE_TBB_TASK
        std::vector<tbb::concurrent_vector<unsigned int> > m_clusters; //!< Cluster components
        #else
        std::vector<std::vector<unsigned int> > m_clusters; //!< Cluster components
        #endif

        detail::Graph m_G; //!< The graph

        hoomd::detail::AABBTree m_aabb_tree_old;              //!< Locality lookup for old configuration

        GlobalVector<Scalar4> m_postype_backup;        //!< Old local positions
        GlobalVector<Scalar4> m_orientation_backup;    //!< Old local orientations
        GlobalVector<int3> m_image_backup;             //!< Old local images

        #ifndef ENABLE_TBB_TASK
        std::set<std::pair<unsigned int, unsigned int> > m_overlap;   //!< A local vector of particle pairs due to overlap
        std::map<std::pair<unsigned int, unsigned int>,LongReal > m_energy_old_old;    //!< Energy of interaction old-old
        std::map<std::pair<unsigned int, unsigned int>,LongReal > m_energy_new_old;    //!< Energy of interaction old-old
        #else
        tbb::concurrent_unordered_set<std::pair<unsigned int, unsigned int> > m_overlap;
        tbb::concurrent_unordered_map<std::pair<unsigned int, unsigned int>,LongReal > m_energy_old_old;
        tbb::concurrent_unordered_map<std::pair<unsigned int, unsigned int>,LongReal > m_energy_new_old;
        #endif

        hpmc_clusters_counters_t m_count_total;                 //!< Total count since initialization
        hpmc_clusters_counters_t m_count_run_start;             //!< Count saved at run() start
        hpmc_clusters_counters_t m_count_step_start;            //!< Count saved at the start of the last step

        //! Check overlaps of a particle with depletants
        inline void checkDepletantOverlap(unsigned int i, vec3<Scalar> pos_i, Shape shape_i, unsigned int typ_i,
            const Scalar4 *h_postype_backup, const Scalar4 *h_orientation_backup,
            unsigned int *h_overlaps, const Scalar *h_fugacity,
            uint64_t timestep, const quat<Scalar> q, const vec3<Scalar> pivot, bool line);

        //! Save current state of particle data
        virtual void backupState();

        //! Find interactions between particles due to overlap and depletion interaction
        /*! \param timestep Current time step
        */
        virtual void findInteractions(uint64_t timestep, const quat<Scalar> q, const vec3<Scalar> pivot, bool line);

        //! Determine connected components of the interaction graph
        virtual void connectedComponents();

        // Transform particles using an self-inverse, isometric operation
        virtual void transform(const quat<Scalar>& q, const vec3<Scalar>& pivot, bool line);

        //! Flip clusters randomly
        virtual void flip(uint64_t timestep);
    };

template< class Shape >
UpdaterClusters<Shape>::UpdaterClusters(std::shared_ptr<SystemDefinition> sysdef,
                                     std::shared_ptr<Trigger> trigger,
                                 std::shared_ptr<IntegratorHPMCMono<Shape> > mc)
        : Updater(sysdef, trigger), m_mc(mc), m_move_ratio(0.5),
            m_flip_probability(0.5)
    {
    m_exec_conf->msg->notice(5) << "Constructing UpdaterClusters" << std::endl;

    #ifdef ENABLE_TBB_TASK
    m_G.setTaskArena(sysdef->getParticleData()->getExecConf()->getTaskArena());
    #endif

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
UpdaterClusters<Shape>::~UpdaterClusters()
    {
    m_exec_conf->msg->notice(5) << "Destroying UpdaterClusters" << std::endl;
    }

/*! \param i The particle id in the list
    \param pos_i Particle position being tested
    \param shape_i Particle shape (including orientation) being tested
    \param typ_i Type of the particle being tested
    \param h_postype Pointer to GPUArray containing particle positions
    \param h_orientation Pointer to GPUArray containing particle orientations
    \param h_overlaps Pointer to GPUArray containing interaction matrix
    \param timestep the timestep

    Particle positions are checked against a randomly generated set of depletant positions.

    NOTE: To avoid numerous acquires and releases of GPUArrays, data pointers are passed directly into this const function.
    */
template<class Shape>
inline void UpdaterClusters<Shape>::checkDepletantOverlap(unsigned int i, vec3<Scalar> pos_i, Shape shape_i, unsigned int typ_i,
    const Scalar4 *h_postype_backup, const Scalar4 *h_orientation_backup,
    unsigned int *h_overlaps, const Scalar *h_fugacity,
    uint64_t timestep, const quat<Scalar> q, const vec3<Scalar> pivot, bool line)
    {
    unsigned int ndim = this->m_sysdef->getNDimensions();

    // access parameters
    auto& params = m_mc->getParams();
    auto& image_list = m_mc->updateImageList();
    const unsigned int n_images = (unsigned int) image_list.size();

    Index2D overlap_idx = m_mc->getOverlapIndexer();

    hoomd::detail::AABB aabb_i_local = shape_i.getAABB(vec3<Scalar>(0,0,0));

    const uint16_t seed = this->m_sysdef->getSeed();

    // get image of particle i after transformation
    const BoxDim box = m_pdata->getGlobalBox();
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

    #ifdef ENABLE_TBB_TASK
    this->m_exec_conf->getTaskArena()->execute([&]{
    tbb::parallel_for(tbb::blocked_range<unsigned int>(0, this->m_pdata->getNTypes()),
        [=, &shape_i](const tbb::blocked_range<unsigned int>& x) {
    for (unsigned int type_a = x.begin(); type_a != x.end(); ++type_a)
    #else
    for (unsigned int type_a = 0; type_a < this->m_pdata->getNTypes(); ++type_a)
    #endif
        {
        if (h_fugacity[type_a] == 0.0)
            {
            continue;
            }
        #ifdef ENABLE_TBB_TASK
        tbb::parallel_for(tbb::blocked_range<unsigned int>(type_a, this->m_pdata->getNTypes()),
            [=, &shape_i](const tbb::blocked_range<unsigned int>& w) {
        for (unsigned int type_b = w.begin(); type_b != w.end(); ++type_b)
        #else
        for (unsigned int type_b = type_a; type_b < this->m_pdata->getNTypes(); ++type_b)
        #endif
            {
            if (!h_overlaps[overlap_idx(type_a, typ_i)] || !h_overlaps[overlap_idx(type_b, typ_i)])
                continue;

            std::vector<vec3<Scalar> > pos_j;
            std::vector<quat<Scalar> > orientation_j;
            std::vector<unsigned int> type_j;
            std::vector<unsigned int> idx_j;

            bool repulsive = h_fugacity[type_a] < 0.0;

            if (repulsive)
                throw std::runtime_error("Negative fugacities not supported in UpdaterClusters.\n");

            // find neighbors whose circumspheres overlap particle i's circumsphere in the old configuration
            // Here, circumsphere refers to the sphere around the depletant-excluded volume

            Shape tmp_a(quat<Scalar>(), params[type_a]);
            Shape tmp_b(quat<Scalar>(), params[type_b]);
            ShortReal d_dep_a = tmp_a.getCircumsphereDiameter();
            ShortReal d_dep_b = tmp_b.getCircumsphereDiameter();

            // the relevant search radius is the one for the larger depletant
            ShortReal d_dep_search = std::max(d_dep_a, d_dep_b);

            // we're sampling in the larger volume, so that it strictly covers the insertion volume of
            // the smaller depletant
            ShortReal r_dep_sample = 0.5f*d_dep_search;

            // get AABB and extend
            vec3<Scalar> lower = aabb_i_local.getLower();
            vec3<Scalar> upper = aabb_i_local.getUpper();
            lower.x -= d_dep_search; lower.y -= d_dep_search; lower.z -= d_dep_search;
            upper.x += d_dep_search; upper.y += d_dep_search; upper.z += d_dep_search;
            hoomd::detail::AABB aabb_local = hoomd::detail::AABB(lower,upper);

            // All image boxes (including the primary)
            for (unsigned int cur_image = 0; cur_image < n_images; cur_image++)
                {
                vec3<Scalar> pos_i_image = pos_i + image_list[cur_image];
                hoomd::detail::AABB aabb = aabb_local;
                aabb.translate(pos_i_image);

                // stackless search
                for (unsigned int cur_node_idx = 0; cur_node_idx < this->m_aabb_tree_old.getNumNodes(); cur_node_idx++)
                    {
                    if (aabb.overlaps(this->m_aabb_tree_old.getNodeAABB(cur_node_idx)))
                        {
                        if (this->m_aabb_tree_old.isNodeLeaf(cur_node_idx))
                            {
                            for (unsigned int cur_p = 0; cur_p < this->m_aabb_tree_old.getNodeNumParticles(cur_node_idx); cur_p++)
                                {
                                // read in its position and orientation
                                unsigned int j = this->m_aabb_tree_old.getNodeParticle(cur_node_idx, cur_p);

                                if (i == j && cur_image == 0) continue;

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
                                    idx_j.push_back(j);
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
                std::abs(h_fugacity[type_a]*V_tot));
            unsigned int ntypes = this->m_pdata->getNTypes();
            hoomd::RandomGenerator rng_num(hoomd::Seed(hoomd::RNGIdentifier::HPMCDepletantNumClusters, timestep, seed),
                                           hoomd::Counter(type_a, type_b, i));

            unsigned int n = poisson(rng_num);

            // try inserting in the overlap volume
            size_t n_intersect = pos_j.size();

            std::vector<unsigned int> storage_sz;
            std::vector< std::vector<typename Shape::depletion_storage_type> > temp_storage;

            // temporary storage for depletant insertions
            for (size_t k = 0; k < n_intersect; ++k)
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
                    ndim, &storage.front(), (ShortReal) V_all[k], detail::SamplingMethod::accurate);

                temp_storage.push_back(storage);
                storage_sz.push_back(sz);
                }

            // for every depletant
            #ifdef ENABLE_TBB_TASK
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
                hoomd::RandomGenerator my_rng(hoomd::Seed(hoomd::RNGIdentifier::HPMCDepletantsClusters, timestep, seed),
                                              hoomd::Counter(i, type_a+type_b*ntypes, l));

                Scalar V_rand = hoomd::UniformDistribution<Scalar>(0.0, V_tot)(my_rng);

                Scalar V_sum(0.0);
                size_t k;
                for (k = 0; k < n_intersect; ++k)
                    {
                    Scalar V = V_all[k];
                    V_sum += V;
                    if (V_rand < V_sum)
                        break;
                    }

                // rejection sampling
                Shape shape_j(orientation_j[k], params[type_j[k]]);

                vec3<ShortReal> dr_test;
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
                    ShortReal rsq = dot(dr_test,dr_test);
                    ShortReal DaDb = shape_test_a.getCircumsphereDiameter() + shape_i.getCircumsphereDiameter();
                    bool circumsphere_overlap = (rsq*ShortReal(4.0) <= DaDb * DaDb);

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
                    ShortReal rsq = dot(dr_test,dr_test);
                    ShortReal DaDb = shape_test_b.getCircumsphereDiameter() + shape_i.getCircumsphereDiameter();
                    bool circumsphere_overlap = (rsq*ShortReal(4.0) <= DaDb * DaDb);

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
                vec3<ShortReal> dr_test_transf = pos_test_transf - pos_i;

                Shape shape_test_transf_a(shape_test_a.orientation, params[type_a]);
                if (shape_test_a.hasOrientation())
                    shape_test_transf_a.orientation = q*shape_test_transf_a.orientation;
                Shape shape_test_transf_b(shape_test_b.orientation, params[type_b]);
                if (shape_test_b.hasOrientation())
                    shape_test_transf_b.orientation = q*shape_test_transf_b.orientation;

                ShortReal rsq = dot(dr_test_transf,dr_test_transf);
                ShortReal DaDb = shape_test_a.getCircumsphereDiameter() + shape_i.getCircumsphereDiameter();
                bool circumsphere_overlap = (rsq*ShortReal(4.0) <= DaDb * DaDb);

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
                    ShortReal rsq = dot(dr_test_transf,dr_test_transf);
                    ShortReal DaDb = shape_test_b.getCircumsphereDiameter() + shape_i.getCircumsphereDiameter();
                    bool circumsphere_overlap = (rsq*ShortReal(4.0) <= DaDb * DaDb);

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

                unsigned int min_idx = i;

                for (size_t m = k; m < n_intersect; ++m)
                    {
                    unsigned int type_m = type_j[m];
                    Shape shape_m(orientation_j[m], params[type_m]);
                    vec3<Scalar> r_mk = pos_j[m] - pos_i - vec3<Scalar>(dr_test);

                    unsigned int err = 0;

                    // check circumsphere overlap
                    ShortReal rsq = (ShortReal) dot(r_mk,r_mk);
                    ShortReal DaDb = shape_test_a.getCircumsphereDiameter() + shape_m.getCircumsphereDiameter();
                    bool circumsphere_overlap = (rsq*ShortReal(4.0) <= DaDb * DaDb);

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
                        circumsphere_overlap = (rsq*ShortReal(4.0) <= DaDb * DaDb);

                        overlap_j_b = h_overlaps[overlap_idx(type_b,type_m)]
                            && circumsphere_overlap
                            && test_overlap(r_mk, shape_test_b, shape_m, err);
                        }

                    // non-additive depletants
                    if ((overlap_i_a && overlap_j_b) || (overlap_i_b && overlap_j_a))
                        {
                        if (idx_j[m] < min_idx)
                            {
                            min_idx = idx_j[m];
                            break;
                            }
                        }
                    }

                if (min_idx == i)
                    {
                    // does the transformed depletant overlap with any other particles?
                    bool overlap_old = false;

                    // get AABB
                    hoomd::detail::AABB aabb_a_local = shape_test_transf_a.getAABB(vec3<Scalar>(0,0,0));

                    // All image boxes (including the primary)
                    for (unsigned int cur_image = 0; cur_image < n_images; cur_image++)
                        {
                        vec3<Scalar> pos_test_image = pos_test_transf + image_list[cur_image];
                        hoomd::detail::AABB aabb = aabb_a_local;
                        aabb.translate(pos_test_image);

                        // stackless search
                        for (unsigned int cur_node_idx = 0; cur_node_idx < this->m_aabb_tree_old.getNumNodes(); cur_node_idx++)
                            {
                            if (aabb.overlaps(this->m_aabb_tree_old.getNodeAABB(cur_node_idx)))
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
                                        ShortReal rsq = (ShortReal) dot(r_jk,r_jk);
                                        ShortReal DaDb = shape_test_transf_a.getCircumsphereDiameter() + shape_j.getCircumsphereDiameter();
                                        unsigned int err = 0;
                                        if (h_overlaps[overlap_idx(type_a, typ_j)] &&
                                            (rsq*ShortReal(4.0) <= DaDb * DaDb) &&
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
                        hoomd::detail::AABB aabb_b_local = shape_test_transf_b.getAABB(vec3<Scalar>(0,0,0));
                        for (unsigned int cur_image = 0; cur_image < n_images; cur_image++)
                            {
                            vec3<Scalar> pos_test_image = pos_test_transf + image_list[cur_image];
                            hoomd::detail::AABB aabb = aabb_b_local;
                            aabb.translate(pos_test_image);

                            // stackless search
                            for (unsigned int cur_node_idx = 0; cur_node_idx < this->m_aabb_tree_old.getNumNodes(); cur_node_idx++)
                                {
                                if (aabb.overlaps(this->m_aabb_tree_old.getNodeAABB(cur_node_idx)))
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
                                            ShortReal rsq = (ShortReal) dot(r_jk,r_jk);
                                            ShortReal DaDb = shape_test_transf_b.getCircumsphereDiameter() + shape_j.getCircumsphereDiameter();
                                            unsigned int err = 0;
                                            if (h_overlaps[overlap_idx(type_b, typ_j)] &&
                                                (rsq*ShortReal(4.0) <= DaDb * DaDb) &&
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
                        {
                        // this depletant was not in the free volume
                        continue;
                        }

                    for (size_t m = k; m < n_intersect; ++m)
                        {
                        unsigned int type_m = type_j[m];
                        Shape shape_m(orientation_j[m], params[type_m]);
                        vec3<Scalar> r_mk = pos_j[m] - pos_i - vec3<Scalar>(dr_test);

                        unsigned int err = 0;

                        // check circumsphere overlap
                        ShortReal rsq = (ShortReal) dot(r_mk,r_mk);
                        ShortReal DaDb = shape_test_a.getCircumsphereDiameter() + shape_m.getCircumsphereDiameter();
                        bool circumsphere_overlap = (rsq*ShortReal(4.0) <= DaDb * DaDb);

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
                            circumsphere_overlap = (rsq*ShortReal(4.0) <= DaDb * DaDb);

                            overlap_j_b = h_overlaps[overlap_idx(type_b,type_m)]
                                && circumsphere_overlap
                                && test_overlap(r_mk, shape_test_b, shape_m, err);
                            }

                        // additive depletants
                        if ((overlap_i_a && !overlap_transf_a && overlap_j_b) || (overlap_i_b && !overlap_transf_b & overlap_j_a))
                            {
                            // add bond
                            this->m_overlap.insert(std::make_pair(i,idx_j[m]));

                            // since i<=j by construction, make matrix symmetric explicitly
                            this->m_overlap.insert(std::make_pair(idx_j[m],i));
                            }
                        }
                    } // end loop over intersections
                } // end loop over depletants
            #ifdef ENABLE_TBB_TASK
                });
            #endif
            } // end loop over type_b
        #ifdef ENABLE_TBB_TASK
            });
        #endif
        } // end loop over type_a
    #ifdef ENABLE_TBB_TASK
        });
    }); // end task arena execute()
    #endif
    }

template< class Shape >
void UpdaterClusters<Shape>::transform(const quat<Scalar>& q, const vec3<Scalar>& pivot, bool line)
    {
    // store old locality data
    m_aabb_tree_old = m_mc->buildAABBTree();
    const BoxDim box = m_pdata->getGlobalBox();

        {
        ArrayHandle<Scalar4> h_pos(this->m_pdata->getPositions(), access_location::host, access_mode::readwrite);
        ArrayHandle<Scalar4> h_orientation(this->m_pdata->getOrientationArray(), access_location::host, access_mode::readwrite);
        ArrayHandle<int3> h_image(this->m_pdata->getImages(), access_location::host, access_mode::readwrite);

        // access parameters
        auto params = m_mc->getParams();
        unsigned int nptl = this->m_pdata->getN();

        for (unsigned int i = 0; i < nptl; ++i)
            {
            vec3<Scalar> new_pos(h_pos.data[i]);
            if (!line)
                {
                // point reflection
                new_pos = pivot-(new_pos-pivot);
                }
            else
                {
                // line reflection
                new_pos = lineReflection(new_pos, pivot, q);
                Shape shape_i(quat<Scalar>(h_orientation.data[i]), params[__scalar_as_int(h_pos.data[i].w)]);
                if (shape_i.hasOrientation())
                    h_orientation.data[i] = quat_to_scalar4(q*quat<Scalar>(h_orientation.data[i]));
                }
            // wrap particle back into box, incrementing image flags
            int3 img = box.getImage(new_pos);
            new_pos = box.shift(new_pos,-img);
            h_pos.data[i] = make_scalar4(new_pos.x, new_pos.y, new_pos.z, h_pos.data[i].w);
            h_image.data[i] = h_image.data[i] + img;
            }
        }
    }

template< class Shape >
void UpdaterClusters<Shape>::flip(uint64_t timestep)
    {
    // move every cluster independently
    m_count_total.n_clusters += m_clusters.size();

        {
        ArrayHandle<Scalar4> h_pos(this->m_pdata->getPositions(), access_location::host, access_mode::readwrite);
        ArrayHandle<Scalar4> h_orientation(this->m_pdata->getOrientationArray(), access_location::host, access_mode::readwrite);
        ArrayHandle<int3> h_image(this->m_pdata->getImages(), access_location::host, access_mode::readwrite);

        ArrayHandle<Scalar4> h_pos_backup(m_postype_backup, access_location::host, access_mode::read);
        ArrayHandle<Scalar4> h_orientation_backup(m_orientation_backup, access_location::host, access_mode::read);
        ArrayHandle<int3> h_image_backup(m_image_backup, access_location::host, access_mode::read);

        uint16_t seed = this->m_sysdef->getSeed();

        for (unsigned int icluster = 0; icluster < m_clusters.size(); icluster++)
            {
            m_count_total.n_particles_in_clusters += m_clusters[icluster].size();

            // seed by id of first particle in cluster to make independent of cluster labeling
            hoomd::RandomGenerator rng_i(hoomd::Seed(hoomd::RNGIdentifier::UpdaterClusters2, timestep, seed),
                                         hoomd::Counter(m_clusters[icluster][0]));

            bool flip = hoomd::detail::generate_canonical<LongReal>(rng_i) <= m_flip_probability;

            if (!flip)
                {
                // revert cluster
                for (auto it = m_clusters[icluster].begin(); it != m_clusters[icluster].end(); ++it)
                    {
                    // particle index
                    unsigned int i = *it;

                    h_pos.data[i] = h_pos_backup.data[i];
                    h_orientation.data[i] = h_orientation_backup.data[i];
                    h_image.data[i] = h_image_backup.data[i];
                    }
                }
            } // end loop over clusters
        }
    }

template< class Shape >
void UpdaterClusters<Shape>::findInteractions(uint64_t timestep, const quat<Scalar> q, const vec3<Scalar> pivot, bool line)
    {
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

    Scalar r_cut_patch(0.0);
    if (m_mc->hasPairInteractions())
        {
        m_energy_old_old.clear();
        m_energy_new_old.clear();
        r_cut_patch = m_mc->getMaxPairEnergyRCutNonAdditive();
        }

    // cluster according to overlap of excluded volume shells
    // loop over local particles
    unsigned int nptl = m_pdata->getN();

    // access particle data
    ArrayHandle<Scalar4> h_postype(m_pdata->getPositions(), access_location::host, access_mode::read);
    ArrayHandle<Scalar4> h_orientation(m_pdata->getOrientationArray(), access_location::host, access_mode::read);
    ArrayHandle<Scalar> h_diameter(m_pdata->getDiameters(), access_location::host, access_mode::read);
    ArrayHandle<Scalar> h_charge(m_pdata->getCharges(), access_location::host, access_mode::read);

    ArrayHandle<Scalar4> h_postype_backup(m_postype_backup, access_location::host, access_mode::read);
    ArrayHandle<Scalar4> h_orientation_backup(m_orientation_backup, access_location::host, access_mode::read);

    if (m_mc->hasPairInteractions())
        {
        // test old configuration against itself
        #ifdef ENABLE_TBB_TASK
        this->m_exec_conf->getTaskArena()->execute([&]{
        tbb::parallel_for((unsigned int)0,this->m_pdata->getN(), [&](unsigned int i)
        #else
        for (unsigned int i = 0; i < this->m_pdata->getN(); ++i)
        #endif
            {
            unsigned int typ_i = __scalar_as_int(h_postype_backup.data[i].w);

            vec3<Scalar> pos_i(h_postype_backup.data[i]);
            quat<Scalar> orientation_i(h_orientation_backup.data[i]);

            Scalar d_i(h_diameter.data[i]);
            Scalar charge_i(h_charge.data[i]);

            // subtract minimum AABB extent from search radius
            Scalar extent_i = 0.5*m_mc->getMaxPairInteractionAdditiveRCut(typ_i);
            Scalar R_query = std::max(0.0,r_cut_patch+extent_i-min_core_diameter/(ShortReal)2.0);
            hoomd::detail::AABB aabb_local = hoomd::detail::AABB(vec3<Scalar>(0,0,0), R_query);

            const unsigned int n_images = (unsigned int) image_list.size();

            for (unsigned int cur_image = 0; cur_image < n_images; cur_image++)
                {
                vec3<Scalar> pos_i_image = pos_i + image_list[cur_image];

                hoomd::detail::AABB aabb_i_image = aabb_local;
                aabb_i_image.translate(pos_i_image);

                // stackless search
                for (unsigned int cur_node_idx = 0; cur_node_idx < m_aabb_tree_old.getNumNodes(); cur_node_idx++)
                    {
                    if (aabb_i_image.overlaps(m_aabb_tree_old.getNodeAABB(cur_node_idx)))
                        {
                        if (m_aabb_tree_old.isNodeLeaf(cur_node_idx))
                            {
                            for (unsigned int cur_p = 0; cur_p < m_aabb_tree_old.getNodeNumParticles(cur_node_idx); cur_p++)
                                {
                                // read in its position and orientation
                                unsigned int j = m_aabb_tree_old.getNodeParticle(cur_node_idx, cur_p);

                                if (i == j && cur_image == 0) continue;

                                // load the position and orientation of the j particle
                                vec3<Scalar> pos_j = vec3<Scalar>(h_postype_backup.data[j]);
                                unsigned int typ_j = __scalar_as_int(h_postype_backup.data[j].w);

                                // put particles in coordinate system of particle i
                                vec3<Scalar> r_ij = pos_j - pos_i_image;
                                Scalar rsq_ij = dot(r_ij, r_ij);

                                Scalar rcut_ij = r_cut_patch + extent_i + 0.5*m_mc->getMaxPairInteractionAdditiveRCut(typ_j);

                                if (rsq_ij <= rcut_ij*rcut_ij)
                                    {
                                    // the particle pair
                                    auto p = std::make_pair(i,j);

                                    // if particle interacts in different image already, add to that energy
                                    LongReal U = 0.0;
                                        {
                                        auto it_energy = m_energy_old_old.find(p);
                                        if (it_energy != m_energy_old_old.end())
                                            U = it_energy->second;
                                        }

                                    U += m_mc->computeOnePairEnergy(rsq_ij,
                                                        r_ij, typ_i,
                                                        orientation_i,
                                                        d_i,
                                                        charge_i,
                                                        typ_j,
                                                        quat<LongReal>(h_orientation_backup.data[j]),
                                                        h_diameter.data[j],
                                                        h_charge.data[j]);

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
        #ifdef ENABLE_TBB_TASK
            );
        }); // end task arena execute()
        #endif
        }

    // loop over new configuration
    #ifdef ENABLE_TBB_TASK
    this->m_exec_conf->getTaskArena()->execute([&]{
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
        hoomd::detail::AABB aabb_i_local = shape_i.getAABB(vec3<Scalar>(0,0,0));

        // All image boxes (including the primary)
        const unsigned int n_images = (unsigned int) image_list.size();

        // check against old
        for (unsigned int cur_image = 0; cur_image < n_images; cur_image++)
            {
            vec3<Scalar> pos_i_image = pos_i_new + image_list[cur_image];

            hoomd::detail::AABB aabb_i_image = aabb_i_local;
            aabb_i_image.translate(pos_i_image);

            // stackless search
            for (unsigned int cur_node_idx = 0; cur_node_idx < m_aabb_tree_old.getNumNodes(); cur_node_idx++)
                {
                if (aabb_i_image.overlaps(m_aabb_tree_old.getNodeAABB(cur_node_idx)))
                    {
                    if (m_aabb_tree_old.isNodeLeaf(cur_node_idx))
                        {
                        for (unsigned int cur_p = 0; cur_p < m_aabb_tree_old.getNodeNumParticles(cur_node_idx); cur_p++)
                            {
                            // read in its position and orientation
                            unsigned int j = m_aabb_tree_old.getNodeParticle(cur_node_idx, cur_p);

                            if (i == j && cur_image == 0) continue;

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
                                    m_overlap.insert(std::make_pair(i,j));
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

        if (m_mc->hasPairInteractions())
            {
            // subtract minimum AABB extent from search radius
            Scalar extent_i = 0.5*m_mc->getMaxPairInteractionAdditiveRCut(typ_i);
            Scalar R_query = std::max(0.0,r_cut_patch+extent_i-min_core_diameter/(LongReal)2.0);
            hoomd::detail::AABB aabb_local = hoomd::detail::AABB(vec3<Scalar>(0,0,0), R_query);

            // compute V(r'-r)
            for (unsigned int cur_image = 0; cur_image < n_images; cur_image++)
                {
                vec3<Scalar> pos_i_image = pos_i_new + image_list[cur_image];

                hoomd::detail::AABB aabb_i_image = aabb_local;
                aabb_i_image.translate(pos_i_image);

                // stackless search
                for (unsigned int cur_node_idx = 0; cur_node_idx < m_aabb_tree_old.getNumNodes(); cur_node_idx++)
                    {
                    if (aabb_i_image.overlaps(m_aabb_tree_old.getNodeAABB(cur_node_idx)))
                        {
                        if (m_aabb_tree_old.isNodeLeaf(cur_node_idx))
                            {
                            for (unsigned int cur_p = 0; cur_p < m_aabb_tree_old.getNodeNumParticles(cur_node_idx); cur_p++)
                                {
                                // read in its position and orientation
                                unsigned int j = m_aabb_tree_old.getNodeParticle(cur_node_idx, cur_p);

                                if (i == j && cur_image == 0) continue;

                                vec3<Scalar> pos_j(h_postype_backup.data[j]);
                                unsigned int typ_j = __scalar_as_int(h_postype_backup.data[j].w);

                                // put particles in coordinate system of particle i
                                vec3<Scalar> r_ij = pos_j - pos_i_image;

                                // check for excluded volume sphere overlap
                                Scalar rsq_ij = dot(r_ij, r_ij);

                                Scalar rcut_ij = r_cut_patch + extent_i + 0.5*m_mc->getMaxPairInteractionAdditiveRCut(typ_j);

                                if (rsq_ij <= rcut_ij*rcut_ij)
                                    {
                                    auto p = std::make_pair(i, j);

                                    // if particle interacts in different image already, add to that energy
                                    LongReal U = 0.0;
                                        {
                                        auto it_energy = m_energy_new_old.find(p);
                                        if (it_energy != m_energy_new_old.end())
                                            U = it_energy->second;
                                        }

                                    U += m_mc->computeOnePairEnergy(rsq_ij,
                                                        r_ij,
                                                        typ_i,
                                                        shape_i.orientation,
                                                        h_diameter.data[i],
                                                        h_charge.data[i],
                                                        typ_j,
                                                        quat<LongReal>(h_orientation_backup.data[j]),
                                                        h_diameter.data[j],
                                                        h_charge.data[j]);

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
    #ifdef ENABLE_TBB_TASK
        );
    }); // end task arena execute()
    #endif

    /*
     * Depletants
     */

    ArrayHandle<Scalar> h_fugacity(this->m_mc->getFugacityArray(), access_location::host, access_mode::read);
    bool has_depletants = false;
    for (unsigned int i = 0; i < this->m_pdata->getNTypes(); ++i)
        {
        if (h_fugacity.data[i] != 0.0)
            {
            has_depletants = true;
            break;
            }
        }

    if (!has_depletants)
        return;

    // test old configuration against itself
    #ifdef ENABLE_TBB_TASK
    this->m_exec_conf->getTaskArena()->execute([&]{
    tbb::parallel_for((unsigned int)0,this->m_pdata->getN(), [&](unsigned int i) {
    #else
    for (unsigned int i = 0; i < this->m_pdata->getN(); ++i)
    #endif
        {
        unsigned int typ_i = __scalar_as_int(h_postype_backup.data[i].w);

        vec3<Scalar> pos_i(h_postype_backup.data[i]);
        quat<Scalar> orientation_i(h_orientation_backup.data[i]);
        Shape shape_i(orientation_i, params[typ_i]);

        checkDepletantOverlap(i, pos_i, shape_i, typ_i,
            h_postype_backup.data, h_orientation_backup.data,
            h_overlaps.data, h_fugacity.data,
            timestep, q, pivot, line);
        }
    #ifdef ENABLE_TBB_TASK
        });
    }); // end task arena execute()
    #endif
    }

template<class Shape>
void UpdaterClusters<Shape>::backupState()
    {
    unsigned int nptl = m_pdata->getN();

    // resize as necessary
    m_postype_backup.resize(nptl);
    m_orientation_backup.resize(nptl);
    m_image_backup.resize(nptl);

        {
        ArrayHandle<Scalar4> h_postype(m_pdata->getPositions(), access_location::host, access_mode::read);
        ArrayHandle<Scalar4> h_orientation(m_pdata->getOrientationArray(), access_location::host, access_mode::read);
        ArrayHandle<int3> h_image(m_pdata->getImages(), access_location::host, access_mode::read);

        ArrayHandle<Scalar4> h_postype_backup(m_postype_backup, access_location::host, access_mode::overwrite);
        ArrayHandle<Scalar4> h_orientation_backup(m_orientation_backup, access_location::host, access_mode::overwrite);
        ArrayHandle<int3> h_image_backup(m_image_backup, access_location::host, access_mode::overwrite);

        // copy over data
        for (unsigned int i = 0; i < nptl; ++i)
            {
            h_postype_backup.data[i] = h_postype.data[i];
            h_orientation_backup.data[i] = h_orientation.data[i];
            h_image_backup.data[i] = h_image.data[i];
            }
        }
    }

template<class Shape>
void UpdaterClusters<Shape>::connectedComponents()
    {
    // compute connected components
    m_clusters.clear();
    m_G.connectedComponents(m_clusters);
    }

/*! Perform a cluster move
    \param timestep Current time step of the simulation
*/
template< class Shape >
void UpdaterClusters<Shape>::update(uint64_t timestep)
    {
    Updater::update(timestep);
    #ifdef ENABLE_MPI
    if (this->m_pdata->getDomainDecomposition())
        throw std::runtime_error("UpdaterClusters does not work with spatial domain decomposition.");
    #endif

    m_exec_conf->msg->notice(10) << timestep << " UpdaterClusters" << std::endl;

    m_count_step_start = m_count_total;

    // if no particles, exit early
    if (! m_pdata->getNGlobal()) return;

    const uint16_t seed = m_sysdef->getSeed();

    // generate the move, select a pivot
    hoomd::RandomGenerator rng(hoomd::Seed(hoomd::RNGIdentifier::UpdaterClusters, timestep, seed),
                               hoomd::Counter(m_instance));
    vec3<Scalar> pivot(0,0,0);

    // is this a line reflection?
    bool line = (m_mc->hasOrientation() || (hoomd::detail::generate_canonical<Scalar>(rng) > m_move_ratio));

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

    const BoxDim box = m_pdata->getGlobalBox();
    pivot = vec3<Scalar>(box.makeCoordinates(f));
    if (m_sysdef->getNDimensions() == 2)
        {
        // force z component to be zero
        pivot.z = 0.0;
        }

    // store backup of particle data
    backupState();

    // transform particle data
    this->transform(q, pivot, line);

    // signal that AABB tree is invalid
    m_mc->invalidateAABBTree();

    // determine which particles interact
    findInteractions(timestep, q, pivot, line);

    // resize the number of graph nodes in place
    m_G.resize(this->m_pdata->getN());

    #ifdef ENABLE_TBB_TASK
    this->m_exec_conf->getTaskArena()->execute([&]{
    tbb::parallel_for(m_overlap.range(), [&] (decltype(m_overlap.range()) r)
    #else
    auto &r = m_overlap;
    #endif
        {
        for (auto it = r.begin(); it != r.end(); ++it)
            {
            unsigned int i = it->first;
            unsigned int j = it->second;

            // we're adding a directed edge, but the actual graph will be undirected
            // if the symmetry operation is self-inverse
            m_G.addEdge(i,j);
            }
        }
    #ifdef ENABLE_TBB_TASK
        );
    }); // end task arena execute()
    #endif

    if (m_mc->hasPairInteractions())
        {
        // sum up interaction energies
        #ifdef ENABLE_TBB_TASK
        tbb::concurrent_unordered_map< std::pair<unsigned int, unsigned int>, LongReal> delta_U;
        #else
        std::map< std::pair<unsigned int, unsigned int>, LongReal> delta_U;
        #endif

        for (auto it = m_energy_old_old.begin(); it != m_energy_old_old.end(); ++it)
            {
            LongReal delU = -it->second;
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

        for (auto it = m_energy_new_old.begin(); it != m_energy_new_old.end(); ++it)
            {
            LongReal delU = it->second;
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

        const Scalar kT = (*m_mc->getKT())(timestep);

        #ifdef ENABLE_TBB_TASK
        this->m_exec_conf->getTaskArena()->execute([&]{
        tbb::parallel_for(delta_U.range(), [&] (decltype(delta_U.range()) r)
        #else
        auto &r = delta_U;
        #endif
            {
            for (auto it = r.begin(); it != r.end(); ++it)
                {
                LongReal delU = it->second;
                unsigned int i = it->first.first;
                unsigned int j = it->first.second;

                // create a RNG specific to this particle pair
                hoomd::RandomGenerator rng_ij(hoomd::Seed(hoomd::RNGIdentifier::UpdaterClustersPairwise, timestep, seed),
                                              hoomd::Counter(std::min(i,j), std::max(i,j)));

                LongReal pij = 1.0f-exp(-delU / kT);
                if (hoomd::detail::generate_canonical<LongReal>(rng_ij) <= pij) // GCA
                    {
                    // add bond
                    m_G.addEdge(i,j);
                    }
                }
            }
        #ifdef ENABLE_TBB_TASK
            );
        }); // end task arena execute()
        #endif
        } // end if (m_mc->hasPairInteractions)

    // compute connected components
    connectedComponents();

    // flip clusters randomly
    flip(timestep);

    m_mc->invalidateAABBTree();
    }

namespace detail {

template < class Shape> void export_UpdaterClusters(pybind11::module& m, const std::string& name)
    {
    pybind11::class_< UpdaterClusters<Shape>, Updater, std::shared_ptr< UpdaterClusters<Shape> > >(m, name.c_str())
          .def( pybind11::init< std::shared_ptr<SystemDefinition>, std::shared_ptr<Trigger>,
                         std::shared_ptr< IntegratorHPMCMono<Shape> > >())
        .def("getCounters", &UpdaterClusters<Shape>::getCounters)
        .def_property("pivot_move_probability", &UpdaterClusters<Shape>::getMoveRatio, &UpdaterClusters<Shape>::setMoveRatio)
        .def_property("flip_probability", &UpdaterClusters<Shape>::getFlipProbability, &UpdaterClusters<Shape>::setFlipProbability)
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
} // end namespace hoomd

#endif // _UPDATER_HPMC_CLUSTERS_
