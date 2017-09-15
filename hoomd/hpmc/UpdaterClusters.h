// inclusion guard
#ifndef _UPDATER_HPMC_CLUSTERS_
#define _UPDATER_HPMC_CLUSTERS_

/*! \file UpdaterBoxClusters.h
    \brief Declaration of UpdaterBoxClusters
*/

#include "hoomd/Updater.h"
#include "hoomd/Saru.h"

#include <list>

#include "Moves.h"
#include "HPMCCounters.h"

namespace hpmc
{

namespace detail
{

// Graph class represents a undirected graph
// using adjacency list representation
class Graph
    {
    int V;    // No. of vertices

    // Pointer to an array containing adjacency lists
    std::vector<std::list<int> > adj;

    // A function used by DFS
    inline void DFSUtil(int v, std::vector<bool>& visited, std::vector<unsigned int>& cur_cc);

public:
    Graph()         //!< Default constructor
        : V(0) {}

    inline Graph(int V);   // Constructor
    inline void addEdge(int v, int w);
    inline void connectedComponents(std::vector<std::vector<unsigned int> >& cc);
    };

// Gather connected components in an undirected graph
void Graph::connectedComponents(std::vector<std::vector<unsigned int> >& cc)
    {
    // Mark all the vertices as not visited
    std::vector<bool> visited(V,false);
    for (int v=0; v<V; v++)
        {
        if (visited[v] == false)
            {
            std::vector<unsigned int> cur_cc;
            DFSUtil(v, visited, cur_cc);
            cc.push_back(cur_cc);
            }
        }
    }

void Graph::DFSUtil(int v, std::vector<bool>& visited, std::vector<unsigned int>& cur_cc)
    {
    visited[v] = true;
    cur_cc.push_back(v);

    // Recur for all the vertices
    // adjacent to this vertex
    for(auto i = adj[v].begin(); i != adj[v].end(); ++i)
        {
        if(!visited[*i])
            DFSUtil(*i, visited, cur_cc);
        }
    }

Graph::Graph(int V)
    {
    this->V = V;
    adj.resize(V);
    }

// method to add an undirected edge
void Graph::addEdge(int v, int w)
    {
    adj[v].push_back(w);
    adj[w].push_back(v);
    }
} // end namespace detail

/*! A generic cluster move for implicit depletant integrators.

    The cluster move set employed consists of pivot (point mirroring) and
    line reflection (pi rotation) moves. The algorithm therefore implements
    a simplified version of the Geometric Cluster algorithm, cf. Liu and Luijten
    PRL 2004 and Sinkovits, Barr and Luijten JCP 2012.

    The algorithm has been simplified to not perform any detailed overlap
    checks, only circumsphere overlap checks. This choice does not affect
    correctness, it only affects ergodicity. Therefore the cluster move should
    be combined with a local move, that is, IntegratorHPMCMonoImplicit(New).

    It should be straight forward to generalize the updater to general
    enthalpic potentials.

    In order to support anisotropic particles, we have to reject moves that
    cross the PBC, as described in Sinkovits et al. Furthermore, the class
    doesn't support any parallelization - it will work in MPI, but only in serial.
    The reason is that GCA-like algorithm's are not easily parallelizable
    because of the non-local nature of the move set.
*/

template< class Shape, class Integrator >
class UpdaterClusters : public Updater
    {
    public:
        //! Constructor
        /*! \param sysdef System definition
            \param mc_implicit Implicit depletants integrator
            \param seed PRNG seed
        */
        UpdaterClusters(std::shared_ptr<SystemDefinition> sysdef,
                        std::shared_ptr<Integrator> mc_implicit,
                        unsigned int seed);

        //! Destructor
        virtual ~UpdaterClusters();

        //! Get the value of a logged quantity
        virtual Scalar getLogValue(const std::string& quantity, unsigned int timestep)
            {
            hpmc_counters_t counters = getCounters(2);

            if (quantity == "hpmc_cluster_moves")
                {
                hpmc_counters_t counters_total = getCounters(0);
                return double(counters_total.getNMoves()) / double(m_pdata->getNGlobal());
                }
            else if (quantity == "hpmc_cluster_pivot_acceptance")
                {
                return counters.getTranslateAcceptance();
                }
            else if (quantity == "hpmc_cluster_reflection_acceptance")
                {
                return counters.getRotateAcceptance();
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
            result.push_back("hpmc_cluster_sweep");
            result.push_back("hpmc_cluster_translate_acceptance");
            result.push_back("hpmc_cluster_rotate_acceptance");
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
            hpmc_counters_t counters = getCounters(1);
            m_exec_conf->msg->notice(2) << "-- HPMC cluster move stats:" << std::endl;
            if (counters.translate_accept_count + counters.translate_reject_count != 0)
                {
                m_exec_conf->msg->notice(2) << "Average pivot acceptance: " << counters.getTranslateAcceptance() << std::endl;
                }
            if (counters.rotate_accept_count + counters.rotate_reject_count != 0)
                {
                m_exec_conf->msg->notice(2) << "Average reflection acceptance:    " << counters.getRotateAcceptance() << std::endl;
                }

            m_exec_conf->msg->notice(2) << "Total cluster moves:          " << counters.getNMoves() << std::endl;
            }

            /*! \param mode 0 -> Absolute count, 1 -> relative to the start of the run, 2 -> relative to the last executed step
                \return The current state of the acceptance counters
            */
            hpmc_counters_t getCounters(unsigned int mode)
                {
                hpmc_counters_t result;

                if (mode == 0)
                    result = m_count_total;
                else if (mode == 1)
                    result = m_count_total - m_count_run_start;
                else
                    result = m_count_total - m_count_step_start;

                #ifdef ENABLE_MPI
                if (m_pdata->getDomainDecomposition())
                    {
                    bcast(result.translate_accept_count,0,m_exec_conf->getMPICommunicator());
                    bcast(result.rotate_accept_count,0,m_exec_conf->getMPICommunicator());
                    bcast(result.translate_reject_count,0,m_exec_conf->getMPICommunicator());
                    bcast(result.rotate_reject_count,0,m_exec_conf->getMPICommunicator());
                    }
                #endif

                return result;
                }


    protected:
        std::shared_ptr< Integrator> m_mc_implicit; //!< Implicit depletants integrator object
        unsigned int m_seed;                        //!< RNG seed
        Scalar m_move_ratio;                        //!< Pivot/Reflection move ratio

        std::vector<std::vector<unsigned int> > m_clusters; //!< Cluster components

        detail::AABBTree m_aabb_tree;               //!< Bounding volume hierarchy for overlap checks
        detail::AABBTree m_aabb_tree_new;           //!< Bounding volume hierarchy for overlap checks
        detail::AABB* m_aabbs;                      //!< list of AABBs, one per particle
        detail::AABB* m_aabbs_new;                  //!< list of AABBs, one per particle
        unsigned int m_aabbs_capacity;              //!< Capacity of m_aabbs list


        std::vector<int3> m_image_hkl;                       //!< List of potentially interacting simulation box images (image indices)
        std::vector<vec3<Scalar> > m_image_list;             //!< List of potentially interacting simulation box images
        bool m_image_list_warning_issued;                    //!< True if the image list warning has been issued
        bool m_hkl_max_warning_issued;                       //!< True if the image list size warning has been issued
        Scalar m_extra_range;                                //!< Extra image list padding length

        detail::Graph m_G; //!< The graph

        std::vector<unsigned int> m_ptl_reject;        //!< List of flags if ptl belongs to a cluster that is not transformed
        hpmc_counters_t m_count_total;                 //!< Total count since initialization
        hpmc_counters_t m_count_run_start;             //!< Count saved at run() start
        hpmc_counters_t m_count_step_start;            //!< Count saved at the start of the last step

        //! Build the AABB trees in both the old and the new configuration
        void buildAABBTrees(const SnapshotParticleData<Scalar> &snap, vec3<Scalar> pivot, quat<Scalar> q, bool line);

        //! Update the global image list
        void updateImageList();

        //! Grow the m_aabbs list
        virtual void growAABBList(unsigned int N);

        //! Generate a list of clusters
        void generateClusters(unsigned int timestep, const SnapshotParticleData<Scalar>& snap, vec3<Scalar> pivot, quat<Scalar> q, bool line);
    };

template< class Shape, class Integrator >
UpdaterClusters<Shape,Integrator>::UpdaterClusters(std::shared_ptr<SystemDefinition> sysdef,
                                 std::shared_ptr<Integrator> mc_implicit,
                                 unsigned int seed)
        : Updater(sysdef), m_mc_implicit(mc_implicit), m_seed(seed), m_move_ratio(0.5), m_extra_range(0.0)
    {
    m_exec_conf->msg->notice(5) << "Constructing UpdaterClusters" << std::endl;

    m_aabbs = NULL;
    m_aabbs_new = NULL;
    m_aabbs_capacity = 0;

    m_image_list_warning_issued = false;
    m_hkl_max_warning_issued = false;

    // initialize logger and stats
    resetStats();
    }

template< class Shape, class Integrator >
UpdaterClusters<Shape,Integrator>::~UpdaterClusters()
    {
    m_exec_conf->msg->notice(5) << "Destroying UpdaterClusters" << std::endl;

    if (m_aabbs != NULL)
        {
        free(m_aabbs);
        free(m_aabbs_new);
        }
    }

//! Calculate a list of global box images within interaction range of the simulation box, innermost first
template <class Shape, class Integrator>
void UpdaterClusters<Shape,Integrator>::updateImageList()
    {
    if (m_prof) m_prof->push(m_exec_conf, "HPMC clusters image list");

    unsigned int ndim = m_sysdef->getNDimensions();

    m_image_list.clear();
    m_image_hkl.clear();

    // Get box vectors
    const BoxDim& box = m_pdata->getGlobalBox();
    vec3<Scalar> e1 = vec3<Scalar>(box.getLatticeVector(0));
    vec3<Scalar> e2 = vec3<Scalar>(box.getLatticeVector(1));
    // 2D simulations don't necessarily have a zero-size z-dimension, but it is convenient for us if we assume one.
    vec3<Scalar> e3(0,0,0);
    if (ndim == 3)
        e3 = vec3<Scalar>(box.getLatticeVector(2));

    // Maximum interaction range is the sum of system box circumsphere diameter and the max particle circumsphere diameter
    Scalar range = 0.0f;
    // Try four linearly independent body diagonals and find the longest
    vec3<Scalar> body_diagonal;
    body_diagonal = e1 - e2 - e3;
    range = std::max(range, dot(body_diagonal, body_diagonal));
    body_diagonal = e1 - e2 + e3;
    range = std::max(range, dot(body_diagonal, body_diagonal));
    body_diagonal = e1 + e2 - e3;
    range = std::max(range, dot(body_diagonal, body_diagonal));
    body_diagonal = e1 + e2 + e3;
    range = std::max(range, dot(body_diagonal, body_diagonal));
    range = fast::sqrt(range);

    // add extra range
    range += m_extra_range;

    Scalar max_range(0.0);

        {
        // access the type parameters
        const std::vector<typename Shape::param_type, managed_allocator<typename Shape::param_type> > & params = m_mc_implicit->getParams();

        // for each type, create a temporary shape and compute the maximum diameter
        for (unsigned int typ = 0; typ < this->m_pdata->getNTypes(); typ++)
            {
            Shape temp(quat<Scalar>(), params[typ]);
            max_range = std::max(max_range, (Scalar)temp.getCircumsphereDiameter());
            }
        }

    range += max_range;

    Scalar range_sq = range*range;

    // initialize loop
    int3 hkl;
    bool added_images = true;
    int hkl_max = 0;
    const int crazybig = 30;
    while (added_images == true)
        {
        added_images = false;

        int x_max = hkl_max;
        int y_max = hkl_max;
        int z_max = 0;
        if (ndim == 3)
            z_max = hkl_max;

        // for h in -hkl_max..hkl_max
        //  for k in -hkl_max..hkl_max
        //   for l in -hkl_max..hkl_max
        //    check if exterior to box of images: if abs(h) == hkl_max || abs(k) == hkl_max || abs(l) == hkl_max
        //     if abs(h*e1 + k*e2 + l*e3) <= range; then image_list.push_back(hkl) && added_cells = true;
        for (hkl.x = -x_max; hkl.x <= x_max; hkl.x++)
            {
            for (hkl.y = -y_max; hkl.y <= y_max; hkl.y++)
                {
                for (hkl.z = -z_max; hkl.z <= z_max; hkl.z++)
                    {
                    // Note that the logic of the following line needs to work in 2 and 3 dimensions
                    if (abs(hkl.x) == hkl_max || abs(hkl.y) == hkl_max || abs(hkl.z) == hkl_max)
                        {
                        vec3<Scalar> r = Scalar(hkl.x) * e1 + Scalar(hkl.y) * e2 + Scalar(hkl.z) * e3;
                        // include primary image so we can do checks in in one loop
                        if (dot(r,r) <= range_sq)
                            {
                            vec3<Scalar> img = (Scalar)hkl.x*e1+(Scalar)hkl.y*e2+(Scalar)hkl.z*e3;
                            m_image_list.push_back(img);
                            m_image_hkl.push_back(make_int3(hkl.x, hkl.y, hkl.z));
                            added_images = true;
                            }
                        }
                    }
                }
            }
        if (!m_hkl_max_warning_issued && hkl_max > crazybig)
            {
            m_hkl_max_warning_issued = true;
            m_exec_conf->msg->warning() << "Exceeded sanity limit for image list, generated out to " << hkl_max
                                     << " lattice vectors. Logic error?" << std::endl
                                     << "This message will not be repeated." << std::endl;

            break;
            }

        hkl_max++;
        }

    // cout << "built image list" << endl;
    // for (unsigned int i = 0; i < m_image_list.size(); i++)
    //     cout << m_image_list[i].x << " " << m_image_list[i].y << " " << m_image_list[i].z << endl;
    // cout << endl;

    // warn the user if more than one image in each direction is activated
    unsigned int img_warning = 9;
    if (ndim == 3)
        {
        img_warning = 27;
        }
    if (!m_image_list_warning_issued && m_image_list.size() > img_warning)
        {
        m_image_list_warning_issued = true;
        m_exec_conf->msg->warning() << "Box size is too small or move size is too large for the minimum image convention." << std::endl
                                    << "Testing " << m_image_list.size() << " images per trial move, performance may slow." << std::endl
                                    << "This message will not be repeated." << std::endl;
        }

    m_exec_conf->msg->notice(4) << "Updated image list: " << m_image_list.size() << " images" << std::endl;
    if (m_prof) m_prof->pop();
    }


/*! UpdaterClusters uses its own AABB tree since it operates on the global configuration
*/
template <class Shape, class Integrator>
void UpdaterClusters<Shape,Integrator>::buildAABBTrees(const SnapshotParticleData<Scalar>& snap, vec3<Scalar> pivot, quat<Scalar> q, bool line)
    {
    m_exec_conf->msg->notice(8) << "UpdaterClusters building AABB tree: " << m_pdata->getNGlobal() << " ptls " << std::endl;
    if (this->m_prof) this->m_prof->push(this->m_exec_conf, "AABB tree build");

    const BoxDim& global_box = m_pdata->getGlobalBox();

    // build the AABB trees
        {
        const std::vector<typename Shape::param_type, managed_allocator<typename Shape::param_type> > & params = m_mc_implicit->getParams();

        // grow the AABB list to the needed size
        unsigned int n_aabb = snap.size;
        if (n_aabb > 0)
            {
            growAABBList(n_aabb);
            for (unsigned int cur_particle = 0; cur_particle < n_aabb; cur_particle++)
                {
                Shape shape(quat<Scalar>(snap.orientation[cur_particle]), params[snap.type[cur_particle]]);
                m_aabbs[cur_particle] = shape.getAABB(snap.pos[cur_particle]);

                // transform position and orientation
                vec3<Scalar> pos_new;
                Shape shape_new(quat<Scalar>(snap.orientation[cur_particle]), params[snap.type[cur_particle]]);

                if (!line)
                    {
                    // point reflection
                    pos_new = (pivot-(snap.pos[cur_particle]-pivot));
                    }
                else
                    {
                    // line reflection
                    pos_new = lineReflection(snap.pos[cur_particle], pivot, q);

                    if (shape_new.hasOrientation())
                        {
                        shape_new.orientation = q*shape_new.orientation;
                        }
                    }

                // wrap particle back into box
                int3 img = make_int3(0,0,0);
                global_box.wrap(pos_new, img);

                m_aabbs_new[cur_particle] = shape_new.getAABB(pos_new);
                }
            m_aabb_tree.buildTree(m_aabbs, n_aabb);
            m_aabb_tree_new.buildTree(m_aabbs_new, n_aabb);
            }
        }

    if (this->m_prof) this->m_prof->pop(this->m_exec_conf);
    }

template <class Shape, class Integrator>
void UpdaterClusters<Shape,Integrator>::growAABBList(unsigned int N)
    {
    if (N > m_aabbs_capacity)
        {
        m_aabbs_capacity = N;
        if (m_aabbs != NULL)
            {
            free(m_aabbs);
            free(m_aabbs_new);
            }

        int retval = posix_memalign((void**)&m_aabbs, 32, N*sizeof(detail::AABB));
        if (retval != 0)
            {
            m_exec_conf->msg->error() << "Error allocating aligned memory" << std::endl;
            throw std::runtime_error("Error allocating AABB memory");
            }
        retval = posix_memalign((void**)&m_aabbs_new, 32, N*sizeof(detail::AABB));
        if (retval != 0)
            {
            m_exec_conf->msg->error() << "Error allocating aligned memory" << std::endl;
            throw std::runtime_error("Error allocating AABB memory");
            }
        }
    }


template< class Shape, class Integrator >
void UpdaterClusters<Shape,Integrator>::generateClusters(unsigned int timestep, const SnapshotParticleData<Scalar>& snap, vec3<Scalar> pivot,
    quat<Scalar> q, bool line)
    {
    if (m_prof) m_prof->push("Generate clusters");

    // Depletant diameter
    Scalar d_dep;
    unsigned int depletant_type = m_mc_implicit->getDepletantType();
        {
        // add range of depletion interaction
        const std::vector<typename Shape::param_type, managed_allocator<typename Shape::param_type> > & params = m_mc_implicit->getParams();
        quat<Scalar> o;
        Shape tmp(o, params[depletant_type]);
        d_dep = tmp.getCircumsphereDiameter();
        }

    // combine the three seeds
    std::vector<unsigned int> seed_seq(3);
    seed_seq[0] = this->m_seed^0x9172fb3a;
    seed_seq[1] = timestep;
    seed_seq[2] = this->m_exec_conf->getRank();
    std::seed_seq seed(seed_seq.begin(), seed_seq.end());

    // RNG for poisson distribution
    std::mt19937 rng_poisson(seed);

    m_extra_range = d_dep;

    // update the aabb tree
    buildAABBTrees(snap, pivot, q, line);

    // update the image list
    updateImageList();

    // access parameters
    const std::vector<typename Shape::param_type, managed_allocator<typename Shape::param_type> > & params = m_mc_implicit->getParams();

    Index2D overlap_idx = m_mc_implicit->getOverlapIndexer();
    ArrayHandle<unsigned int> h_overlaps(m_mc_implicit->getInteractionMatrix(), access_location::host, access_mode::read);

    // clear the graph
    m_G = detail::Graph(snap.size);

    m_ptl_reject.clear();
    m_ptl_reject.resize(snap.size,false);

    const BoxDim& global_box = m_pdata->getGlobalBox();

    // cluster according to overlap of excluded volume shells
    // loop over local particles
    for (unsigned int i = 0; i < snap.size; ++i)
        {
        vec3<Scalar> pos_i_old(snap.pos[i]);
        unsigned int typ_i = snap.type[i];

        vec3<Scalar> pos_i_new = pos_i_old;

        quat<Scalar> orientation_i_new(snap.orientation[i]);

        if (!line)
            {
            // point reflection
            pos_i_new = (pivot-(pos_i_new-pivot));
            }
        else
            {
            // line reflection
            pos_i_new = lineReflection(pos_i_new, pivot, q);
            orientation_i_new = q*orientation_i_new;
            }

        // wrap particle i back into box
        int3 img_i = make_int3(0,0,0);
        global_box.wrap(pos_i_new, img_i);

        Shape shape_i(orientation_i_new, params[typ_i]);
        Scalar r_excl_i = shape_i.getCircumsphereDiameter()/Scalar(2.0);

        // if this cluster transformation is rejected
        bool reject = false;

        // check for overlap at mirrored position, with other particles in old configuration
        detail::AABB aabb_i = shape_i.getAABB(pos_i_new);

        // All image boxes (including the primary)
        const unsigned int n_images = m_image_list.size();

        for (unsigned int cur_image = 0; cur_image < n_images; cur_image++)
            {
            vec3<Scalar> pos_i_image = pos_i_new + m_image_list[cur_image];

            detail::AABB aabb_i_image = aabb_i;
            aabb_i_image.translate(m_image_list[cur_image]);

            // stackless search
            for (unsigned int cur_node_idx = 0; cur_node_idx < m_aabb_tree.getNumNodes(); cur_node_idx++)
                {
                if (detail::overlap(m_aabb_tree.getNodeAABB(cur_node_idx), aabb_i_image))
                    {
                    if (m_aabb_tree.isNodeLeaf(cur_node_idx))
                        {
                        for (unsigned int cur_p = 0; cur_p < m_aabb_tree.getNodeNumParticles(cur_node_idx); cur_p++)
                            {
                            // read in its position and orientation
                            unsigned int j = m_aabb_tree.getNodeParticle(cur_node_idx, cur_p);

                            if (i == j) continue;

                            // load the position and orientation of the j particle
                            vec3<Scalar> pos_j = vec3<Scalar>(snap.pos[j]);

                            Shape shape_j(snap.orientation[j], params[snap.type[j]]);

                            // put particles in coordinate system of particle i
                            vec3<Scalar> r_ij = pos_j - pos_i_image;

                            // check for circumsphere overlap
                            Scalar r_excl_j = shape_j.getCircumsphereDiameter()/Scalar(2.0);
                            Scalar RaRb = r_excl_i + r_excl_j;
                            Scalar rsq_ij = dot(r_ij, r_ij);

                            unsigned int err = 0;
                            if (rsq_ij <= RaRb*RaRb)
                                {
                                if (h_overlaps.data[overlap_idx(typ_i,snap.type[j])])
//                                    && test_overlap(r_ij, shape_i, shape_j, err))
                                    {
                                    // add connection
                                    m_G.addEdge(i,j);

                                    // no need to check PBC for isometries
                                    if (! line) continue;

                                    int3 delta_img = m_image_hkl[cur_image]-img_i;
                                    if (delta_img.x != 0 || delta_img.y != 0 || delta_img.z != 0)
                                        {
                                        // ptls interact via PBC, do not transform the cluster
                                        reject = true;
                                        }
                                    } // end if overlap
                                }

                            } // end loop over AABB tree leaf
                        } // end is leaf
                    } // end if overlap
                else
                    {
                    // skip ahead
                    cur_node_idx += m_aabb_tree.getNodeSkip(cur_node_idx);
                    }

                } // end loop over nodes
            } // end loop over images

        if (line)
            {
            // line transformations are not isometries, we need to handle the case where both particles
            // only overlap in the new configuration
            for (unsigned int cur_image = 0; cur_image < n_images; cur_image++)
                {
                vec3<Scalar> pos_i_image = pos_i_new + m_image_list[cur_image];

                detail::AABB aabb_i_image = aabb_i;
                aabb_i_image.translate(m_image_list[cur_image]);

                // stackless search in the transformed particle tree
                for (unsigned int cur_node_idx = 0; cur_node_idx < m_aabb_tree_new.getNumNodes(); cur_node_idx++)
                    {
                    if (detail::overlap(m_aabb_tree_new.getNodeAABB(cur_node_idx), aabb_i_image))
                        {
                        if (m_aabb_tree_new.isNodeLeaf(cur_node_idx))
                            {
                            for (unsigned int cur_p = 0; cur_p < m_aabb_tree_new.getNodeNumParticles(cur_node_idx); cur_p++)
                                {
                                // read in its position and orientation
                                unsigned int j = m_aabb_tree_new.getNodeParticle(cur_node_idx, cur_p);

                                if (i == j) continue;

                                // load the position and orientation of the j particle

                                // transform coordinates of ptl j for a line reflection
                                vec3<Scalar> pos_j_new = lineReflection(snap.pos[j], pivot, q);
                                Shape shape_j_new(snap.orientation[j], params[snap.type[j]]);
                                if (shape_j_new.hasOrientation())
                                    {
                                    shape_j_new.orientation = q*shape_j_new.orientation;
                                    }

                                // wrap particle j back into box
                                int3 img_j = make_int3(0,0,0);
                                global_box.wrap(pos_j_new, img_j);

                                // put particles in coordinate system of particle i
                                vec3<Scalar> r_ij = pos_j_new - pos_i_image;

                                // check for circumsphere overlap
                                Scalar r_excl_j = shape_j_new.getCircumsphereDiameter()/Scalar(2.0);
                                Scalar RaRb = r_excl_i + r_excl_j;
                                Scalar rsq_ij = dot(r_ij, r_ij);

                                unsigned int err = 0;
                                if (rsq_ij <= RaRb*RaRb)
                                    {
                                    if (h_overlaps.data[overlap_idx(typ_i,snap.type[j])])
//                                        && test_overlap(r_ij, shape_i, shape_j_new, err))
                                        {
                                        int3 delta_img = m_image_hkl[cur_image]+img_j-img_i;
                                        if (delta_img.x != 0 || delta_img.y != 0 || delta_img.z != 0)
                                            {
                                            // ptl interacts via PBC, do not transform its cluster
                                            m_G.addEdge(i,j);
                                            reject = true;
                                            }
                                        } // end if overlap
                                    } // end if circumsphere overlap
                                } // end loop over AABB tree leaf
                            } // end is leaf
                        } // end if overlap
                    else
                        {
                        // skip ahead
                        cur_node_idx += m_aabb_tree_new.getNodeSkip(cur_node_idx);
                        }

                    } // end loop over nodes
                } // end loop over images
            } // end if line transformation

        // check overlap of depletant-excluded volumes

        // find neighbors whose circumspheres overlap particle i's circumsphere in the old configuration
        // Here, circumsphere refers to the sphere around the depletant-excluded volume
        detail::AABB aabb_local(vec3<Scalar>(0,0,0), Scalar(0.5)*shape_i.getCircumsphereDiameter()+d_dep);

        for (unsigned int cur_image = 0; cur_image < n_images; cur_image++)
            {
            vec3<Scalar> pos_i_image = pos_i_old + m_image_list[cur_image];

            detail::AABB aabb_i_image = aabb_local;
            aabb_i_image.translate(pos_i_image);

            // stackless search
            for (unsigned int cur_node_idx = 0; cur_node_idx < m_aabb_tree.getNumNodes(); cur_node_idx++)
                {
                if (detail::overlap(m_aabb_tree.getNodeAABB(cur_node_idx), aabb_i_image))
                    {
                    if (m_aabb_tree.isNodeLeaf(cur_node_idx))
                        {
                        for (unsigned int cur_p = 0; cur_p < m_aabb_tree.getNodeNumParticles(cur_node_idx); cur_p++)
                            {
                            // read in its position and orientation
                            unsigned int j = m_aabb_tree.getNodeParticle(cur_node_idx, cur_p);

                            if (i == j) continue;

                            // load the position and orientation of the j particle
                            vec3<Scalar> pos_j = vec3<Scalar>(snap.pos[j]);

                            Shape shape_j(snap.orientation[j], params[snap.type[j]]);

                            // put particles in coordinate system of particle i
                            vec3<Scalar> r_ij = pos_j - pos_i_image;

                            // check for excluded volume sphere overlap
                            Scalar r_excl_j = shape_j.getCircumsphereDiameter()/Scalar(2.0);
                            Scalar RaRb = r_excl_i + r_excl_j + d_dep;
                            Scalar rsq_ij = dot(r_ij, r_ij);

                            if (rsq_ij <= RaRb*RaRb)
                                {
                                // are they interacting via PBC, when both i and j are in the old conf?
                                if (line && cur_image != 0)
                                    {
                                    // ptl interacts via PBC, do no transform its cluster
                                    m_G.addEdge(i,j);
                                    reject = true;
                                    }

                                // does i in the new conf overlap with j in the old conf in the same image?
                                vec3<Scalar> r_ij_new = pos_j - pos_i_new;

                                // shift i back into original image
                                global_box.shift(r_ij_new,-img_i);

                                if (! (dot(r_ij_new,r_ij_new) <= RaRb*RaRb))
                                    {
                                    // no, the particles are transformed as part of the same cluster
                                    m_G.addEdge(i,j);
                                    }

                                // no need to check PBC for pivot (isometry) transformations
                                if (!line) continue;

                                // transform coordinates of ptl j
                                vec3<Scalar> pos_j_new;
                                Shape shape_j_new(snap.orientation[j], params[snap.type[j]]);
                                // line reflection
                                pos_j_new = lineReflection(pos_j, pivot, q);

                                if (shape_j_new.hasOrientation())
                                    {
                                    shape_j_new.orientation = q*shape_j_new.orientation;
                                    }

                                // wrap particle j back into box
                                int3 img_j = make_int3(0,0,0);
                                global_box.wrap(pos_j_new, img_j);

                                // does i interact with j in the new configuration in any other image?
                                for (unsigned int cur_image_j = 0; cur_image_j < n_images; cur_image_j++)
                                    {
                                    vec3<Scalar> pos_j_image = pos_j_new + m_image_list[cur_image_j];
                                    vec3<Scalar> r_ij_new = pos_j_image - pos_i_new;
                                    Scalar rsq_ij_new = dot(r_ij_new,r_ij_new);

                                    int3 delta_img = m_image_hkl[cur_image_j]-img_j+img_i;
                                    if (rsq_ij_new <= RaRb*RaRb && (delta_img.x != 0 || delta_img.y != 0 || delta_img.z != 0))
                                        {
                                        m_G.addEdge(i,j);

                                        // do not transform
                                        reject = true;
                                        break;
                                        }
                                    }
                                } // end if overlap

                            } // end loop over AABB tree leaf
                        } // end is leaf
                    } // end if overlap
                else
                    {
                    // skip ahead
                    cur_node_idx += m_aabb_tree.getNodeSkip(cur_node_idx);
                    }

                } // end loop over nodes

            } // end loop over images

        if (line)
            {
            // for non-isometries, we need to check if the particles interact with one or both of them being in the new configuration
            for (unsigned int cur_image = 0; cur_image < n_images; cur_image++)
                {
                vec3<Scalar> pos_i_image = pos_i_new + m_image_list[cur_image];

                detail::AABB aabb_i_image = aabb_local;
                aabb_i_image.translate(pos_i_image);

                // stackless search
                for (unsigned int cur_node_idx = 0; cur_node_idx < m_aabb_tree_new.getNumNodes(); cur_node_idx++)
                    {
                    if (detail::overlap(m_aabb_tree_new.getNodeAABB(cur_node_idx), aabb_i_image))
                        {
                        if (m_aabb_tree_new.isNodeLeaf(cur_node_idx))
                            {
                            for (unsigned int cur_p = 0; cur_p < m_aabb_tree_new.getNodeNumParticles(cur_node_idx); cur_p++)
                                {
                                // read in its position and orientation
                                unsigned int j = m_aabb_tree_new.getNodeParticle(cur_node_idx, cur_p);

                                if (i == j) continue;

                                // transform coordinates of ptl j for a line reflection
                                vec3<Scalar> pos_j_new = lineReflection(snap.pos[j], pivot, q);
                                Shape shape_j_new(snap.orientation[j], params[snap.type[j]]);

                                // wrap particle j back into box
                                int3 img_j = make_int3(0,0,0);
                                global_box.wrap(pos_j_new, img_j);

                                // put particles in coordinate system of particle i
                                vec3<Scalar> r_ij = pos_j_new - pos_i_image;

                                // check for excluded volume sphere overlap
                                Scalar r_excl_j = shape_j_new.getCircumsphereDiameter()/Scalar(2.0);
                                Scalar RaRb = r_excl_i + r_excl_j + d_dep;
                                Scalar rsq_ij = dot(r_ij, r_ij);

                                // are they interacting via PBC, when i and j are both in the new conf?
                                int3 delta_img = m_image_hkl[cur_image]+img_j-img_i;
                                if (rsq_ij <= RaRb*RaRb && (delta_img.x != 0 || delta_img.y != 0 || delta_img.z != 0))
                                    {
                                    // ptl interacts via PBC, do no transform its cluster
                                    m_G.addEdge(i,j);
                                    reject = true;
                                    }
                                } // end loop over AABB tree leaf
                            } // end is leaf
                        } // end if overlap
                    else
                        {
                        // skip ahead
                        cur_node_idx += m_aabb_tree_new.getNodeSkip(cur_node_idx);
                        }

                    } // end loop over nodes

                } // end loop over images

            // query against old positions
            for (unsigned int cur_image = 0; cur_image < n_images; cur_image++)
                {
                vec3<Scalar> pos_i_image = pos_i_new + m_image_list[cur_image];

                detail::AABB aabb_i_image = aabb_local;
                aabb_i_image.translate(pos_i_image);

                // stackless search
                for (unsigned int cur_node_idx = 0; cur_node_idx < m_aabb_tree.getNumNodes(); cur_node_idx++)
                    {
                    if (detail::overlap(m_aabb_tree.getNodeAABB(cur_node_idx), aabb_i_image))
                        {
                        if (m_aabb_tree.isNodeLeaf(cur_node_idx))
                            {
                            for (unsigned int cur_p = 0; cur_p < m_aabb_tree.getNodeNumParticles(cur_node_idx); cur_p++)
                                {
                                // read in its position and orientation
                                unsigned int j = m_aabb_tree.getNodeParticle(cur_node_idx, cur_p);

                                if (i == j) continue;

                                vec3<Scalar> pos_j = snap.pos[j];
                                Shape shape_j(snap.orientation[j], params[snap.type[j]]);

                                // put particles in coordinate system of particle i
                                vec3<Scalar> r_ij = pos_j - pos_i_image;

                                // check for excluded volume sphere overlap
                                Scalar r_excl_j = shape_j.getCircumsphereDiameter()/Scalar(2.0);
                                Scalar RaRb = r_excl_i + r_excl_j + d_dep;
                                Scalar rsq_ij = dot(r_ij, r_ij);

                                // are they interacting via PBC?
                                int3 delta_img = m_image_hkl[cur_image]-img_i;
                                if (rsq_ij <= RaRb*RaRb && (delta_img.x != 0 || delta_img.y != 0 || delta_img.z != 0))
                                    {
                                    // ptl interacts via PBC, do no transform its cluster
                                    m_G.addEdge(i,j);
                                    reject = true;
                                    }

                                } // end loop over AABB tree leaf
                            } // end is leaf
                        } // end if overlap
                    else
                        {
                        // skip ahead
                        cur_node_idx += m_aabb_tree.getNodeSkip(cur_node_idx);
                        }

                    } // end loop over nodes

                } // end loop over images

            }

        // store vertex color
        m_ptl_reject[i] = reject;

        } // end loop over local particles

    // compute connected components
    m_clusters.clear();

    m_G.connectedComponents(m_clusters);

    #if 0
    // do not perform trivial transformations
    if (m_clusters.size()==1 && m_clusters[0].size() == snap.size && m_clusters[0].size())
        {
        m_ptl_reject[m_clusters[0][0]] = true;
        }
    #endif

    if (m_prof) m_prof->pop();
    }

/*! Perform a cluster move
    \param timestep Current time step of the simulation
*/
template< class Shape, class Integrator >
void UpdaterClusters<Shape,Integrator>::update(unsigned int timestep)
    {
    m_count_step_start = m_count_total;
    //hpmc_counters_t& counters = m_count_total;

    if (m_prof) m_prof->push("HPMC Clusters");
    if (m_prof) m_prof->push("Move");

    SnapshotParticleData<Scalar> snap(m_pdata->getNGlobal());

    // obtain particle data from all ranks
    m_pdata->takeSnapshot(snap);

    // perform moves on rank zero
    bool master = !m_exec_conf->getRank();

    if (master)
        {
        // initialize RNG
        hoomd::detail::Saru rng(timestep, this->m_seed, 0x12f40ab9);

        // select a pivot
        Scalar3 f;
        f.x = rng.template s<Scalar>();
        f.y = rng.template s<Scalar>();
        f.z = rng.template s<Scalar>();

        const BoxDim& box = this->m_pdata->getGlobalBox();

        vec3<Scalar> pivot(box.makeCoordinates(f));

        // is this a line reflection?
        bool line = m_mc_implicit->hasOrientation() || (rng.template s<Scalar>() > m_move_ratio);

        quat<Scalar> q;

        if (line)
            {
            // random normalized vector
            Scalar theta = rng.template s<Scalar>(Scalar(0.0),Scalar(2.0*M_PI));
            Scalar z = rng.template s<Scalar>(Scalar(-1.0),Scalar(1.0));
            vec3<Scalar> n(fast::sqrt(Scalar(1.0)-z*z)*fast::cos(theta),fast::sqrt(Scalar(1.0)-z*z)*fast::sin(theta),z);

            // line reflection
            q = quat<Scalar>(0,n);
            }

        if (m_prof) m_prof->pop();

        // generate the cluster definitions
        generateClusters(timestep, snap, pivot, q, line);

        if (m_prof) m_prof->push("Move");

            {
            // access parameters
            const std::vector<typename Shape::param_type, managed_allocator<typename Shape::param_type> > & params = m_mc_implicit->getParams();

            // clusters may be moved independently
            for (unsigned int icluster = 0; icluster < m_clusters.size(); icluster++)
                {
                // loop over cluster members
                bool reject = false;
                for (auto it = m_clusters[icluster].begin(); it != m_clusters[icluster].end(); ++it)
                    {
                    // if any of the ptls in this cluster connects across PBC, reject move
                    if (m_ptl_reject[*it])
                        reject = true;
                    }

                if (!reject)
                    {
                    for (auto it = m_clusters[icluster].begin(); it != m_clusters[icluster].end(); ++it)
                        {
                        // particle index
                        unsigned int i = *it;

                        // read in the current position and cluster id
                        quat<Scalar> orientation_i = snap.orientation[i];
                        vec3<Scalar> pos_i(snap.pos[i]);

                        // make a trial move for i
                        int typ_i = snap.type[i];

                        if (!line)
                            {
                            // point reflection
                            pos_i = pivot-(pos_i-pivot);
                            }
                        else
                            {
                            // line reflection
                            pos_i = lineReflection(pos_i, pivot, q);
                            orientation_i = q*orientation_i;
                            }

                        // update position of particle
                        snap.pos[i] = pos_i;
                        Shape shape_i(orientation_i, params[typ_i]);
                        if (shape_i.hasOrientation())
                            {
                            snap.orientation[i] = orientation_i;
                            }
                        }

                    // use translate for pivot moves, rotate for line reflections
                    if (line)
                        m_count_total.rotate_accept_count++;
                    else
                        m_count_total.translate_accept_count++;
                    }
                else
                    {
                    if (line)
                        m_count_total.rotate_reject_count++;
                    else
                        m_count_total.translate_reject_count++;
                    }
                }
            } // end loop over clusters

            {
            // wrap particles back into box
            for (unsigned int i = 0; i < snap.size; i++)
                {
                box.wrap(snap.pos[i], snap.image[i]);
                }
            }
        }
    // reload particle data
    m_pdata->initializeFromSnapshot(snap);

    if (m_prof) m_prof->pop();
    if (m_prof) m_prof->pop();

    m_mc_implicit->communicate(true);
    }


template < class Shape, class Integrator > void export_UpdaterClusters(pybind11::module& m, const std::string& name)
    {
    pybind11::class_< UpdaterClusters<Shape,Integrator>, std::shared_ptr< UpdaterClusters<Shape,Integrator> > >(m, name.c_str(), pybind11::base<Updater>())
          .def( pybind11::init< std::shared_ptr<SystemDefinition>,
                         std::shared_ptr< Integrator >,
                         unsigned int >())
        .def("getCounters", &UpdaterClusters<Shape,Integrator>::getCounters)
        .def("setMoveRatio", &UpdaterClusters<Shape,Integrator>::setMoveRatio)
    ;
    }

} // end namespace hpmc

#endif // _UPDATER_HPMC_CLUSTERS_
