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
            m_exec_conf->msg->notice(2) << "Average pivot acceptance: " << counters.getTranslateAcceptance() << std::endl;
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
        detail::AABB* m_aabbs;                      //!< list of AABBs, one per particle
        detail::AABBTree m_aabb_tree_new;           //!< Bounding volume hierarchy for overlap checks in the new configuration
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
        virtual void growAABBLists(unsigned int N);

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
        free(m_aabbs);
    if (m_aabbs_new != NULL)
        free(m_aabbs_new);
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

    // Maximum interaction range is the sum of *twice* system box circumsphere diameter and the max particle circumsphere diameter
    // twice, because of reflection and pivot
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
    range = 2*fast::sqrt(range);

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

    // build the AABB tree
        {
        const std::vector<typename Shape::param_type, managed_allocator<typename Shape::param_type> > & params = m_mc_implicit->getParams();

        // grow the AABB list to the needed size
        unsigned int n_aabb = snap.size;
        if (n_aabb > 0)
            {
            growAABBLists(n_aabb);
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
                    pos_new = rotate(q, snap.pos[cur_particle]);

                    if (shape_new.hasOrientation())
                        {
                        shape_new.orientation = q*shape_new.orientation;
                        }
                    }
                m_aabbs_new[cur_particle] = shape_new.getAABB(pos_new);
                }
            m_aabb_tree.buildTree(m_aabbs, n_aabb);
            m_aabb_tree_new.buildTree(m_aabbs_new, n_aabb);
            }
        }

    if (this->m_prof) this->m_prof->pop(this->m_exec_conf);
    }

template <class Shape, class Integrator>
void UpdaterClusters<Shape,Integrator>::growAABBLists(unsigned int N)
    {
    if (N > m_aabbs_capacity)
        {
        m_aabbs_capacity = N;
        if (m_aabbs != NULL)
            free(m_aabbs);

        if (m_aabbs_new != NULL)
            free(m_aabbs_new);

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

    Scalar fugacity = m_mc_implicit->getDepletantDensity();

    // clear the graph
    m_G = detail::Graph(snap.size);

    m_ptl_reject.clear();
    m_ptl_reject.resize(snap.size,false);

    // List of particles whose circumspheres intersect particle i's excluded-volume circumsphere
    std::vector<unsigned int> intersect_i;

    // List of particle images that intersect
    std::vector<unsigned int> image_i;

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
            pos_i_new = rotate(q, pos_i_new);
            orientation_i_new = q*orientation_i_new;
            }

        Shape shape_i(orientation_i_new, params[typ_i]);
        Scalar r_excl_i = shape_i.getCircumsphereDiameter()/Scalar(2.0);

        // if this cluster transformation is rejected
        bool reject = false;

        // check for overlap at mirrored position, with other particles in old configuration
        detail::AABB aabb_i = shape_i.getAABB(pos_i_new);

        // All image boxes (including the primary)
        const unsigned int n_images = m_image_list.size();

        // get new image of particle i
        int3 image_i_new = snap.image[i];
        Scalar3 tmp = vec_to_scalar3(pos_i_new);
        global_box.wrap(tmp,image_i_new);

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
                                if (h_overlaps.data[overlap_idx(typ_i,snap.type[j])]
                                    && test_overlap(r_ij, shape_i, shape_j, err))
                                    {
                                    // add connection
                                    m_G.addEdge(i,j);

                                    int3 delta_img = snap.image[j]-image_i_new-m_image_hkl[cur_image];
                                    if (line && (delta_img.x != 0 || delta_img.y != 0 || delta_img.z != 0))
                                        {
                                        // ptl interacts via PBC, do no transform its cluster
                                        reject = true;
                                        }
                                    }
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
            // line transformations are not isometric, we have to check with both particles in new configuration as well

            // All image boxes (including the primary)
            const unsigned int n_images = m_image_list.size();

            for (unsigned int cur_image = 0; cur_image < n_images; cur_image++)
                {
                vec3<Scalar> pos_i_image = pos_i_new + m_image_list[cur_image];

                detail::AABB aabb_i_image = aabb_i;
                aabb_i_image.translate(m_image_list[cur_image]);

                // stackless search in new configuration
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

                                // transform coordinates
                                vec3<Scalar> pos_j;
                                Shape shape_j(snap.orientation[j], params[snap.type[j]]);
                                if (!line)
                                    {
                                    // point reflection
                                    pos_j = (pivot-(snap.pos[j]-pivot));
                                    }
                                else
                                    {
                                    // line reflection
                                    pos_j = rotate(q, snap.pos[j]);

                                    if (shape_j.hasOrientation())
                                        {
                                        shape_j.orientation = q*shape_j.orientation;
                                        }
                                    }

                                // new image of particle j
                                tmp = vec_to_scalar3(pos_j);
                                int3 image_j_new = snap.image[j];
                                global_box.wrap(tmp,image_j_new);

                                // put particles in coordinate system of particle i
                                vec3<Scalar> r_ij = pos_j - pos_i_image;

                                // check for circumsphere overlap
                                Scalar r_excl_j = shape_j.getCircumsphereDiameter()/Scalar(2.0);
                                Scalar RaRb = r_excl_i + r_excl_j;
                                Scalar rsq_ij = dot(r_ij, r_ij);

                                unsigned int err = 0;
                                if (rsq_ij <= RaRb*RaRb)
                                    {
                                    if (h_overlaps.data[overlap_idx(typ_i,snap.type[j])]
                                        && test_overlap(r_ij, shape_i, shape_j, err))
                                        {
                                        // add connection
                                        m_G.addEdge(i,j);

                                        int3 delta_img = image_j_new-image_i_new-m_image_hkl[cur_image];
                                        if (line && (delta_img.x != 0 || delta_img.y != 0 || delta_img.z != 0))
                                            {
                                            // ptl interacts via PBC, do no transform its cluster
                                            reject = true;
                                            }
                                        }
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
            } // end if line transformation

        /***
         depletant overlap checks in old configuration
         ***/

        // reset collision lists
        intersect_i.clear();
        image_i.clear();

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
                                // check each overlapping pair only once
                                if (i < j && h_overlaps.data[overlap_idx(depletant_type,snap.type[j])])
                                    {
                                    intersect_i.push_back(j);
                                    image_i.push_back(cur_image);
                                    }
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


        // check depletant overlaps
        // for every unique pairwise intersection
        Scalar V(0.0);

        hoomd::detail::Saru rng_i(i, this->m_seed^0xf8bb99a1, timestep);

        for (unsigned int k = 0; k < intersect_i.size(); ++k)
            {
            unsigned int j = intersect_i[k];
            unsigned int image_j = image_i[k];
            vec3<Scalar> ri = pos_i_old;
            vec3<Scalar> rj = snap.pos[j];
            Scalar Ri = Scalar(0.5)*(shape_i.getCircumsphereDiameter()+d_dep);
            Shape shape_j(quat<Scalar>(), params[snap.type[j]]);
            Scalar Rj = Scalar(0.5)*(shape_j.getCircumsphereDiameter()+d_dep);

            vec3<Scalar> rij(rj-ri - m_image_list[image_j]);
            Scalar d = sqrt(dot(rij,rij));

            // heights spherical caps that constitute the intersection volume
            Scalar hi = (Rj*Rj - (d-Ri)*(d-Ri))/(2*d);
            Scalar hj = (Ri*Ri - (d-Rj)*(d-Rj))/(2*d);

            // volumes of spherical caps
            Scalar Vcap_i = Scalar(M_PI/3.0)*hi*hi*(3*Ri-hi);
            Scalar Vcap_j = Scalar(M_PI/3.0)*hj*hj*(3*Rj-hj);

            // volume of intersection
            Scalar V_lens = Vcap_i + Vcap_j;
            V+=V_lens;

            // chooose the number of depletants in the intersection volume
            std::poisson_distribution<unsigned int> poisson(fugacity*V_lens);
            unsigned int n = poisson(rng_poisson);

            // for every depletant
            for (unsigned int l = 0; l < n; ++l)
                {
                // chose one of the two caps randomly, with a weight proportional to their volume
                Scalar s = rng_i.template s<Scalar>();
                bool cap_i = s < Vcap_i/V_lens;

                // generate a depletant position in the spherical cap
                vec3<Scalar> pos_test = cap_i ? generatePositionInSphericalCap(rng_i, ri, Ri, hi, rij)
                    : generatePositionInSphericalCap(rng_i, rj, Rj, hj, -rij)-m_image_list[image_j];

                Shape shape_test(quat<Scalar>(), params[depletant_type]);
                if (shape_test.hasOrientation())
                    {
                    shape_test.orientation = generateRandomOrientation(rng_i);
                    }

                // check if depletant falls in other intersection volumes
                bool active = true;
                for (unsigned int m = 0; m < k; ++m)
                    {
                    unsigned int p = intersect_i[m];
                    vec3<Scalar> rp = vec3<Scalar>(snap.pos[p]);
                    Shape shape_p(quat<Scalar>(), params[snap.type[p]]);

                    vec3<Scalar> delta_r(pos_test + m_image_list[image_i[m]] - rp);
                    OverlapReal rsq = dot(delta_r,delta_r);
                    OverlapReal DaDb = shape_test.getCircumsphereDiameter() + shape_p.getCircumsphereDiameter();
                    bool circumsphere_overlap = (rsq*OverlapReal(4.0) <= DaDb * DaDb);

                    if (circumsphere_overlap)
                        {
                        active = false;
                        break;
                        }
                    }

                if (!active) continue;

                // depletant falls in intersection volume between circumspheres

                // Check if the old configuration of particle i generates an overlap
                bool overlap_old = false;
                    {
                    vec3<Scalar> r_ij = pos_i_old - pos_test;
                    Shape shape_i_old(quat<Scalar>(snap.orientation[i]), params[typ_i]);

                    OverlapReal rsq = dot(r_ij,r_ij);
                    OverlapReal DaDb = shape_test.getCircumsphereDiameter() + shape_i_old.getCircumsphereDiameter();
                    bool circumsphere_overlap = (rsq*OverlapReal(4.0) <= DaDb * DaDb);

                    if (h_overlaps.data[overlap_idx(depletant_type, typ_i)])
                        {
                        unsigned int err = 0;
                        if (circumsphere_overlap && test_overlap(r_ij, shape_test, shape_i_old, err))
                            {
                            overlap_old = true;
                            }
                        }
                    }

                // if not intersecting ptl i in old config, ignore
                if (!overlap_old) continue;

                // is the *mirrored* depletant overlapping with any colloid in the old configuration?
                vec3<Scalar> pos_test_new;
                Shape shape_test_new(quat<Scalar>(), params[depletant_type]);
                if (!line)
                    {
                    // point reflection
                    pos_test_new = (pivot-(pos_test-pivot));
                    }
                else
                    {
                    // line reflection
                    pos_test_new = rotate(q, pos_test);

                    if (shape_test.hasOrientation())
                        {
                        shape_test_new.orientation = q*shape_test.orientation;
                        }
                    }

                // new image of depletant
                tmp = vec_to_scalar3(pos_test_new);
                int3 image_test_new = make_int3(0,0,0);
                global_box.wrap(tmp,image_test_new);

                // if it mirrors to a different image, ignore
                if (image_test_new.x != 0 || image_test_new.y != 0 || image_test_new.z != 0)
                    continue;

                detail::AABB aabb_test_local = shape_test.getAABB(vec3<Scalar>(0,0,0));
                bool overlap_mirrored = false;

                // All image boxes (including the primary)
                const unsigned int n_images = m_image_list.size();
                for (unsigned int cur_image = 0; cur_image < n_images; cur_image++)
                    {
                    vec3<Scalar> pos_test_image = pos_test_new + m_image_list[cur_image];
                    detail::AABB aabb = aabb_test_local;
                    aabb.translate(pos_test_image);

                    // stackless search
                    for (unsigned int cur_node_idx = 0; cur_node_idx < m_aabb_tree.getNumNodes(); cur_node_idx++)
                        {
                        if (detail::overlap(m_aabb_tree.getNodeAABB(cur_node_idx), aabb))
                            {
                            if (m_aabb_tree.isNodeLeaf(cur_node_idx))
                                {
                                for (unsigned int cur_p = 0; cur_p < m_aabb_tree.getNodeNumParticles(cur_node_idx); cur_p++)
                                    {
                                    // read in its position and orientation
                                    unsigned int j = m_aabb_tree.getNodeParticle(cur_node_idx, cur_p);

                                    // put particles in coordinate system of particle i
                                    vec3<Scalar> r_ij = vec3<Scalar>(snap.pos[j]) - pos_test_image;

                                    Shape shape_j(snap.orientation[j], params[snap.type[j]]);

                                    // check circumsphere overlap
                                    OverlapReal rsq = dot(r_ij,r_ij);
                                    OverlapReal DaDb = shape_test_new.getCircumsphereDiameter() + shape_j.getCircumsphereDiameter();
                                    bool circumsphere_overlap = (rsq*OverlapReal(4.0) <= DaDb * DaDb);

                                    unsigned int err = 0;
                                    if (h_overlaps.data[overlap_idx(depletant_type,snap.type[j])]
                                        && circumsphere_overlap
                                        && test_overlap(r_ij, shape_test_new, shape_j, err))
                                        {
                                        // depletant is ignored for any overlap in the old configuration
                                        overlap_mirrored = true;
                                        break;
                                        }
                                    }
                                }
                            }
                        else
                            {
                            // skip ahead
                            cur_node_idx += this->m_aabb_tree.getNodeSkip(cur_node_idx);
                            }

                        if (overlap_mirrored)
                            break;
                        }  // end loop over AABB nodes

                    if (overlap_mirrored)
                        break;
                    } // end loop over images

                if (overlap_mirrored) continue;

                // does the depletant at the original position fall into the overlap volume with other particles?
                for (unsigned int m = 0; m < intersect_i.size(); ++m)
                    {
                    unsigned int l = intersect_i[m];

                    vec3<Scalar> r_il = snap.pos[l] - pos_test - m_image_list[image_i[m]];

                    Shape shape_l(quat<Scalar>(snap.orientation[l]), params[snap.type[l]]);

                    // check circumsphere overlap
                    OverlapReal rsq = dot(r_il,r_il);
                    OverlapReal DaDb = shape_test.getCircumsphereDiameter() + shape_l.getCircumsphereDiameter();
                    bool circumsphere_overlap = (rsq*OverlapReal(4.0) <= DaDb * DaDb);

                    unsigned int err = 0;
                    if (h_overlaps.data[overlap_idx(depletant_type,snap.type[l])]
                        && circumsphere_overlap
                        && test_overlap(r_il, shape_test, shape_l, err))
                        {
                        // depletant induces a bond between i and l
                        m_G.addEdge(i,l);

                        int3 delta_img = snap.image[j]-snap.image[i]-m_image_hkl[image_i[m]];
                        if (line && (delta_img.x != 0 || delta_img.y != 0 || delta_img.z != 0))
                            {
                            // ptl j interacts via PBC, do no transform its cluster
                            reject = true;
                            }
                        }
                    } // end loop over intersecting particles

                } // end loop over depletants

            } // end loop over overlapping spheres

        if (line)
            {
            // again, we can't assume isometry
            // check if depletants overlap between transformed particles

            // reset collision lists
            intersect_i.clear();
            image_i.clear();

            // find neighbors whose circumspheres overlap particle i's circumsphere in the old configuration
            // Here, circumsphere refers to the sphere around the depletant-excluded volume
            detail::AABB aabb_local(vec3<Scalar>(0,0,0), Scalar(0.5)*shape_i.getCircumsphereDiameter()+d_dep);

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

                                // transform coordinates
                                vec3<Scalar> pos_j;
                                Shape shape_j(snap.orientation[j], params[snap.type[j]]);
                                if (!line)
                                    {
                                    // point reflection
                                    pos_j = (pivot-(snap.pos[j]-pivot));
                                    }
                                else
                                    {
                                    // line reflection
                                    pos_j = rotate(q, snap.pos[j]);

                                    if (shape_j.hasOrientation())
                                        {
                                        shape_j.orientation = q*shape_j.orientation;
                                        }
                                    }

                                // put particles in coordinate system of particle i
                                vec3<Scalar> r_ij = pos_j - pos_i_image;

                                // check for circumsphere overlap
                                Scalar r_excl_j = shape_j.getCircumsphereDiameter()/Scalar(2.0);
                                Scalar RaRb = r_excl_i + r_excl_j + d_dep;
                                Scalar rsq_ij = dot(r_ij, r_ij);

                                if (rsq_ij <= RaRb*RaRb)
                                    {
                                    // check each overlapping pair only once
                                    if (i < j && h_overlaps.data[overlap_idx(depletant_type,snap.type[j])])
                                        {
                                        intersect_i.push_back(j);
                                        image_i.push_back(cur_image);
                                        }
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


            // check depletant overlaps
            // for every unique pairwise intersection
            Scalar V(0.0);

            hoomd::detail::Saru rng_i(i, this->m_seed^0xbbfe5622, timestep);

            for (unsigned int k = 0; k < intersect_i.size(); ++k)
                {
                unsigned int j = intersect_i[k];
                unsigned int image_j = image_i[k];
                vec3<Scalar> ri = pos_i_new;

                // transform coordinates
                vec3<Scalar> rj = snap.pos[j];
                Shape shape_j(quat<Scalar>(), params[snap.type[j]]);
                if (!line)
                    {
                    // point reflection
                    rj = (pivot-(rj-pivot));
                    }
                else
                    {
                    // line reflection
                    rj = rotate(q, rj);
                    }

                Scalar Ri = Scalar(0.5)*(shape_i.getCircumsphereDiameter()+d_dep);
                Scalar Rj = Scalar(0.5)*(shape_j.getCircumsphereDiameter()+d_dep);

                vec3<Scalar> rij(rj-ri - m_image_list[image_j]);
                Scalar d = sqrt(dot(rij,rij));

                // heights spherical caps that constitute the intersection volume
                Scalar hi = (Rj*Rj - (d-Ri)*(d-Ri))/(2*d);
                Scalar hj = (Ri*Ri - (d-Rj)*(d-Rj))/(2*d);

                // volumes of spherical caps
                Scalar Vcap_i = Scalar(M_PI/3.0)*hi*hi*(3*Ri-hi);
                Scalar Vcap_j = Scalar(M_PI/3.0)*hj*hj*(3*Rj-hj);

                // volume of intersection
                Scalar V_lens = Vcap_i + Vcap_j;
                V+=V_lens;

                // chooose the number of depletants in the intersection volume
                std::poisson_distribution<unsigned int> poisson(fugacity*V_lens);
                unsigned int n = poisson(rng_poisson);

                // for every depletant
                for (unsigned int l = 0; l < n; ++l)
                    {
                    // chose one of the two caps randomly, with a weight proportional to their volume
                    Scalar s = rng_i.template s<Scalar>();
                    bool cap_i = s < Vcap_i/V_lens;

                    // generate a depletant position in the spherical cap
                    vec3<Scalar> pos_test = cap_i ? generatePositionInSphericalCap(rng_i, ri, Ri, hi, rij)
                        : generatePositionInSphericalCap(rng_i, rj, Rj, hj, -rij)-m_image_list[image_j];

                    Shape shape_test(quat<Scalar>(), params[depletant_type]);
                    if (shape_test.hasOrientation())
                        {
                        shape_test.orientation = generateRandomOrientation(rng_i);
                        }

                    // check if depletant falls in other intersection volumes
                    bool active = true;
                    for (unsigned int m = 0; m < k; ++m)
                        {
                        unsigned int p = intersect_i[m];
                        vec3<Scalar> rp = vec3<Scalar>(snap.pos[p]);

                        // transform coordinates
                        if (!line)
                            {
                            // point reflection
                            rp = (pivot-(rp-pivot));
                            }
                        else
                            {
                            // line reflection
                            rp = rotate(q, rp);
                            }

                        Shape shape_p(quat<Scalar>(), params[snap.type[p]]);

                        vec3<Scalar> delta_r(pos_test + m_image_list[image_i[m]] - rp);
                        OverlapReal rsq = dot(delta_r,delta_r);
                        OverlapReal DaDb = shape_test.getCircumsphereDiameter() + shape_p.getCircumsphereDiameter();
                        bool circumsphere_overlap = (rsq*OverlapReal(4.0) <= DaDb * DaDb);

                        if (circumsphere_overlap)
                            {
                            active = false;
                            break;
                            }
                        }

                    if (!active) continue;

                    // depletant falls in intersection volume between circumspheres

                    // Check if the new configuration of particle i generates an overlap
                    bool overlap_new = false;
                        {
                        vec3<Scalar> r_ij = pos_i_new - pos_test;

                        OverlapReal rsq = dot(r_ij,r_ij);
                        OverlapReal DaDb = shape_test.getCircumsphereDiameter() + shape_i.getCircumsphereDiameter();
                        bool circumsphere_overlap = (rsq*OverlapReal(4.0) <= DaDb * DaDb);

                        if (h_overlaps.data[overlap_idx(depletant_type, typ_i)])
                            {
                            unsigned int err = 0;
                            if (circumsphere_overlap && test_overlap(r_ij, shape_test, shape_i, err))
                                {
                                overlap_new = true;
                                }
                            }
                        }

                    // if not intersecting ptl i in new config, ignore
                    if (!overlap_new) continue;

                    // does the depletant at the original position fall into the overlap volume with other particles?
                    for (unsigned int m = 0; m < intersect_i.size(); ++m)
                        {
                        unsigned int l = intersect_i[m];

                        // transform coordinates
                        vec3<Scalar> pos_l = snap.pos[l];
                        Shape shape_l(snap.orientation[l], params[snap.type[l]]);
                        if (!line)
                            {
                            // point reflection
                            pos_l = (pivot-(pos_l-pivot));
                            }
                        else
                            {
                            // line reflection
                            pos_l = rotate(q, pos_l);

                            if (shape_l.hasOrientation())
                                {
                                shape_l.orientation = q*shape_l.orientation;
                                }
                            }

                        vec3<Scalar> r_il = pos_l - pos_test - m_image_list[image_i[m]];

                        // check circumsphere overlap
                        OverlapReal rsq = dot(r_il,r_il);
                        OverlapReal DaDb = shape_test.getCircumsphereDiameter() + shape_l.getCircumsphereDiameter();
                        bool circumsphere_overlap = (rsq*OverlapReal(4.0) <= DaDb * DaDb);

                        unsigned int err = 0;
                        if (h_overlaps.data[overlap_idx(depletant_type,snap.type[l])]
                            && circumsphere_overlap
                            && test_overlap(r_il, shape_test, shape_l, err))
                            {
                            unsigned int image_l = image_i[m];

                            // we only need to consider particles interacting via the periodic boundaries,
                            // the others we have already accounted for in the old configuration (due to isometry)
                            int3 delta_img = snap.image[l]-snap.image[l]-m_image_hkl[image_l];

                            if ((delta_img.x != 0 || delta_img.y != 0 || delta_img.z != 0))
                                {
                                // ptl l interacts via PBC, do no transform its cluster
                                m_G.addEdge(i,l);
                                reject = true;
                                }
                            }
                        } // end loop over intersecting particles

                    } // end loop over depletants
                } // end loop over overlap volumes
            } // end if (line)

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

    if (m_prof) m_prof->push("Move Clusters");

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

        // generate the cluster definitions
        generateClusters(timestep, snap, pivot, q, line);

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
                            pos_i = rotate(q, pos_i);
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
