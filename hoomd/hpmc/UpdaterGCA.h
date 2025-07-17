// Copyright (c) 2009-2025 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

#pragma once

#include "hoomd/RNGIdentifiers.h"
#include "hoomd/RandomNumbers.h"
#include "hoomd/Updater.h"

#include <list>
#include <set>

#include "HPMCCounters.h"
#include "IntegratorHPMCMono.h"
#include "Moves.h"

namespace hoomd
    {

namespace hpmc
    {

namespace detail
    {
// Graph class represents a undirected graph
// using adjacency list representation
class Graph
    {
    public:
    Graph() { }

    inline Graph(unsigned int V); // Constructor

    inline void resize(unsigned int V);

    inline void addEdge(unsigned int v, unsigned int w);

    inline void connectedComponents(std::vector<std::vector<unsigned int>>& cc);

    private:
    std::multimap<unsigned int, unsigned int> adj;

    std::vector<unsigned int> visited;

    inline void
    DFSUtil(unsigned int v, std::vector<unsigned int>& visited, std::vector<unsigned int>& cur_cc);
    };

// Gather connected components in an undirected graph
void Graph::connectedComponents(std::vector<std::vector<unsigned int>>& cc)
    {
    std::fill(visited.begin(), visited.end(), 0);

    // Depth first search
    for (unsigned int v = 0; v < visited.size(); v++)
        {
        if (visited[v] == false)
            {
            std::vector<unsigned int> cur_cc;
            DFSUtil(v, visited, cur_cc);
            cc.push_back(cur_cc);
            }
        }
    }

void Graph::DFSUtil(unsigned int v,
                    std::vector<unsigned int>& visited,
                    std::vector<unsigned int>& cur_cc)
    {
    visited[v] = 1;
    cur_cc.push_back(v);

    // Recur for all the vertices
    // adjacent to this vertex
    auto begin = adj.equal_range(v).first;
    auto end = adj.equal_range(v).second;
    for (auto i = begin; i != end; ++i)
        {
        if (!visited[i->second])
            DFSUtil(i->second, visited, cur_cc);
        }
    }

Graph::Graph(unsigned int V)
    {
    visited.resize(V, 0);
    }

void Graph::resize(unsigned int V)
    {
    visited.resize(V, 0);

    adj.clear();
    }

// method to add a directed edge
void Graph::addEdge(unsigned int v, unsigned int w)
    {
    adj.insert(std::make_pair(v, w));
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

template<class Shape> class UpdaterGCA : public Updater
    {
    public:
    //! Constructor
    /*! \param sysdef System definition
        \param mc HPMC integrator
    */
    UpdaterGCA(std::shared_ptr<SystemDefinition> sysdef,
               std::shared_ptr<Trigger> trigger,
               std::shared_ptr<IntegratorHPMCMono<Shape>> mc);

    //! Destructor
    virtual ~UpdaterGCA();

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

    /*! \param mode 0 -> Absolute count, 1 -> relative to the start of the run, 2 -> relative to the
       last executed step
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
    std::shared_ptr<IntegratorHPMCMono<Shape>> m_mc; //!< HPMC integrator
    Scalar m_move_ratio;                             //!< Pivot/Reflection move ratio
    Scalar m_flip_probability;                       //!< Cluster flip probability

    unsigned int m_instance = 0; //!< Unique ID for RNG seeding

    std::vector<std::vector<unsigned int>> m_clusters; //!< Cluster components

    detail::Graph m_G; //!< The graph

    hoomd::detail::AABBTree m_aabb_tree_old; //!< Locality lookup for old configuration

    GPUVector<Scalar4> m_postype_backup;     //!< Old local positions
    GPUVector<Scalar4> m_orientation_backup; //!< Old local orientations
    GPUVector<int3> m_image_backup;          //!< Old local images

    std::set<std::pair<unsigned int, unsigned int>>
        m_overlap; //!< A local vector of particle pairs due to overlap
    std::map<std::pair<unsigned int, unsigned int>, LongReal>
        m_energy_old_old; //!< Energy of interaction old-old
    std::map<std::pair<unsigned int, unsigned int>, LongReal>
        m_energy_new_old; //!< Energy of interaction old-old

    hpmc_clusters_counters_t m_count_total;      //!< Total count since initialization
    hpmc_clusters_counters_t m_count_run_start;  //!< Count saved at run() start
    hpmc_clusters_counters_t m_count_step_start; //!< Count saved at the start of the last step

    //! Save current state of particle data
    virtual void backupState();

    //! Find interactions between particles due to overlaps
    /*! \param timestep Current time step
     */
    virtual void
    findInteractions(uint64_t timestep, const quat<Scalar> q, const vec3<Scalar> pivot, bool line);

    //! Determine connected components of the interaction graph
    virtual void connectedComponents();

    // Transform particles using an self-inverse, isometric operation
    virtual void transform(const quat<Scalar>& q, const vec3<Scalar>& pivot, bool line);

    //! Flip clusters randomly
    virtual void flip(uint64_t timestep);
    };

template<class Shape>
UpdaterGCA<Shape>::UpdaterGCA(std::shared_ptr<SystemDefinition> sysdef,
                              std::shared_ptr<Trigger> trigger,
                              std::shared_ptr<IntegratorHPMCMono<Shape>> mc)
    : Updater(sysdef, trigger), m_mc(mc), m_move_ratio(0.5), m_flip_probability(0.5)
    {
    m_exec_conf->msg->notice(5) << "Constructing UpdaterGCA" << std::endl;

    // initialize stats
    resetStats();

    // initialize memory
    GPUVector<Scalar4>(1, this->m_exec_conf).swap(m_postype_backup);
    GPUVector<Scalar4>(1, this->m_exec_conf).swap(m_orientation_backup);
    GPUVector<int3>(1, this->m_exec_conf).swap(m_image_backup);
    }

template<class Shape> UpdaterGCA<Shape>::~UpdaterGCA()
    {
    m_exec_conf->msg->notice(5) << "Destroying UpdaterGCA" << std::endl;
    }

template<class Shape>
void UpdaterGCA<Shape>::transform(const quat<Scalar>& q, const vec3<Scalar>& pivot, bool line)
    {
    // store old locality data
    m_aabb_tree_old = m_mc->buildAABBTree();
    const BoxDim box = m_pdata->getGlobalBox();

        {
        ArrayHandle<Scalar4> h_pos(this->m_pdata->getPositions(),
                                   access_location::host,
                                   access_mode::readwrite);
        ArrayHandle<Scalar4> h_orientation(this->m_pdata->getOrientationArray(),
                                           access_location::host,
                                           access_mode::readwrite);
        ArrayHandle<int3> h_image(this->m_pdata->getImages(),
                                  access_location::host,
                                  access_mode::readwrite);

        // access parameters
        auto params = m_mc->getParams();
        unsigned int nptl = this->m_pdata->getN();

        for (unsigned int i = 0; i < nptl; ++i)
            {
            vec3<Scalar> new_pos(h_pos.data[i]);
            if (!line)
                {
                // point reflection
                new_pos = pivot - (new_pos - pivot);
                }
            else
                {
                // line reflection
                new_pos = lineReflection(new_pos, pivot, q);
                Shape shape_i(quat<Scalar>(h_orientation.data[i]),
                              params[__scalar_as_int(h_pos.data[i].w)]);
                if (shape_i.hasOrientation())
                    h_orientation.data[i]
                        = quat_to_scalar4(q * quat<Scalar>(h_orientation.data[i]));
                }
            // wrap particle back into box, incrementing image flags
            int3 img = box.getImage(new_pos);
            new_pos = box.shift(new_pos, -img);
            h_pos.data[i] = make_scalar4(new_pos.x, new_pos.y, new_pos.z, h_pos.data[i].w);
            h_image.data[i] = h_image.data[i] + img;
            }
        }
    }

template<class Shape> void UpdaterGCA<Shape>::flip(uint64_t timestep)
    {
    // move every cluster independently
    m_count_total.n_clusters += m_clusters.size();

        {
        ArrayHandle<Scalar4> h_pos(this->m_pdata->getPositions(),
                                   access_location::host,
                                   access_mode::readwrite);
        ArrayHandle<Scalar4> h_orientation(this->m_pdata->getOrientationArray(),
                                           access_location::host,
                                           access_mode::readwrite);
        ArrayHandle<int3> h_image(this->m_pdata->getImages(),
                                  access_location::host,
                                  access_mode::readwrite);

        ArrayHandle<Scalar4> h_pos_backup(m_postype_backup,
                                          access_location::host,
                                          access_mode::read);
        ArrayHandle<Scalar4> h_orientation_backup(m_orientation_backup,
                                                  access_location::host,
                                                  access_mode::read);
        ArrayHandle<int3> h_image_backup(m_image_backup, access_location::host, access_mode::read);

        uint16_t seed = this->m_sysdef->getSeed();

        for (unsigned int icluster = 0; icluster < m_clusters.size(); icluster++)
            {
            m_count_total.n_particles_in_clusters += m_clusters[icluster].size();

            // seed by id of first particle in cluster to make independent of cluster labeling
            hoomd::RandomGenerator rng_i(
                hoomd::Seed(hoomd::RNGIdentifier::UpdaterGCA2, timestep, seed),
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

template<class Shape>
void UpdaterGCA<Shape>::findInteractions(uint64_t timestep,
                                         const quat<Scalar> q,
                                         const vec3<Scalar> pivot,
                                         bool line)
    {
    // access parameters
    auto& params = m_mc->getParams();

    // update the image list
    auto& image_list = m_mc->updateImageList();

    // minimum AABB extent
    Scalar min_core_diameter = m_mc->getMinCoreDiameter();

    Index2D overlap_idx = m_mc->getOverlapIndexer();
    ArrayHandle<unsigned int> h_overlaps(m_mc->getInteractionMatrix(),
                                         access_location::host,
                                         access_mode::read);

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
    ArrayHandle<Scalar4> h_postype(m_pdata->getPositions(),
                                   access_location::host,
                                   access_mode::read);
    ArrayHandle<Scalar4> h_orientation(m_pdata->getOrientationArray(),
                                       access_location::host,
                                       access_mode::read);
    ArrayHandle<Scalar> h_diameter(m_pdata->getDiameters(),
                                   access_location::host,
                                   access_mode::read);
    ArrayHandle<Scalar> h_charge(m_pdata->getCharges(), access_location::host, access_mode::read);

    ArrayHandle<Scalar4> h_postype_backup(m_postype_backup,
                                          access_location::host,
                                          access_mode::read);
    ArrayHandle<Scalar4> h_orientation_backup(m_orientation_backup,
                                              access_location::host,
                                              access_mode::read);

    if (m_mc->hasPairInteractions())
        {
        // test old configuration against itself
        for (unsigned int i = 0; i < this->m_pdata->getN(); ++i)
            {
            unsigned int typ_i = __scalar_as_int(h_postype_backup.data[i].w);

            vec3<Scalar> pos_i(h_postype_backup.data[i]);
            quat<Scalar> orientation_i(h_orientation_backup.data[i]);

            Scalar d_i(h_diameter.data[i]);
            Scalar charge_i(h_charge.data[i]);

            // subtract minimum AABB extent from search radius
            Scalar extent_i = 0.5 * m_mc->getMaxPairInteractionAdditiveRCut(typ_i);
            Scalar R_query
                = std::max(0.0, r_cut_patch + extent_i - min_core_diameter / (ShortReal)2.0);
            hoomd::detail::AABB aabb_local = hoomd::detail::AABB(vec3<Scalar>(0, 0, 0), R_query);

            const unsigned int n_images = (unsigned int)image_list.size();

            for (unsigned int cur_image = 0; cur_image < n_images; cur_image++)
                {
                vec3<Scalar> pos_i_image = pos_i + image_list[cur_image];

                hoomd::detail::AABB aabb_i_image = aabb_local;
                aabb_i_image.translate(pos_i_image);

                // stackless search
                for (unsigned int cur_node_idx = 0; cur_node_idx < m_aabb_tree_old.getNumNodes();
                     cur_node_idx++)
                    {
                    if (aabb_i_image.overlaps(m_aabb_tree_old.getNodeAABB(cur_node_idx)))
                        {
                        if (m_aabb_tree_old.isNodeLeaf(cur_node_idx))
                            {
                            for (unsigned int cur_p = 0;
                                 cur_p < m_aabb_tree_old.getNodeNumParticles(cur_node_idx);
                                 cur_p++)
                                {
                                // read in its position and orientation
                                unsigned int j
                                    = m_aabb_tree_old.getNodeParticle(cur_node_idx, cur_p);

                                if (i == j && cur_image == 0)
                                    continue;

                                // load the position and orientation of the j particle
                                vec3<Scalar> pos_j = vec3<Scalar>(h_postype_backup.data[j]);
                                unsigned int typ_j = __scalar_as_int(h_postype_backup.data[j].w);

                                // put particles in coordinate system of particle i
                                vec3<Scalar> r_ij = pos_j - pos_i_image;
                                Scalar rsq_ij = dot(r_ij, r_ij);

                                Scalar rcut_ij
                                    = r_cut_patch + extent_i
                                      + 0.5 * m_mc->getMaxPairInteractionAdditiveRCut(typ_j);

                                if (rsq_ij <= rcut_ij * rcut_ij)
                                    {
                                    // the particle pair
                                    auto p = std::make_pair(i, j);

                                    // if particle interacts in different image already, add to that
                                    // energy
                                    LongReal U = 0.0;
                                        {
                                        auto it_energy = m_energy_old_old.find(p);
                                        if (it_energy != m_energy_old_old.end())
                                            U = it_energy->second;
                                        }

                                    U += m_mc->computeOnePairEnergy(
                                        rsq_ij,
                                        r_ij,
                                        typ_i,
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
        }

    // loop over new configuration
    for (unsigned int i = 0; i < nptl; ++i)
        {
        unsigned int typ_i = __scalar_as_int(h_postype.data[i].w);

        vec3<Scalar> pos_i_new(h_postype.data[i]);
        quat<Scalar> orientation_i_new(h_orientation.data[i]);

        Shape shape_i(orientation_i_new, params[typ_i]);
        Scalar r_excl_i = shape_i.getCircumsphereDiameter() / Scalar(2.0);

        // check for overlap at mirrored position, with other particles in old configuration
        hoomd::detail::AABB aabb_i_local = shape_i.getAABB(vec3<Scalar>(0, 0, 0));

        // All image boxes (including the primary)
        const unsigned int n_images = (unsigned int)image_list.size();

        // check against old
        for (unsigned int cur_image = 0; cur_image < n_images; cur_image++)
            {
            vec3<Scalar> pos_i_image = pos_i_new + image_list[cur_image];

            hoomd::detail::AABB aabb_i_image = aabb_i_local;
            aabb_i_image.translate(pos_i_image);

            // stackless search
            for (unsigned int cur_node_idx = 0; cur_node_idx < m_aabb_tree_old.getNumNodes();
                 cur_node_idx++)
                {
                if (aabb_i_image.overlaps(m_aabb_tree_old.getNodeAABB(cur_node_idx)))
                    {
                    if (m_aabb_tree_old.isNodeLeaf(cur_node_idx))
                        {
                        for (unsigned int cur_p = 0;
                             cur_p < m_aabb_tree_old.getNodeNumParticles(cur_node_idx);
                             cur_p++)
                            {
                            // read in its position and orientation
                            unsigned int j = m_aabb_tree_old.getNodeParticle(cur_node_idx, cur_p);

                            if (i == j && cur_image == 0)
                                continue;

                            // load the position and orientation of the j particle
                            vec3<Scalar> pos_j = vec3<Scalar>(h_postype_backup.data[j]);
                            unsigned int typ_j = __scalar_as_int(h_postype_backup.data[j].w);
                            Shape shape_j(quat<Scalar>(h_orientation_backup.data[j]),
                                          params[typ_j]);

                            // put particles in coordinate system of particle i
                            vec3<Scalar> r_ij = pos_j - pos_i_image;

                            // check for circumsphere overlap
                            Scalar r_excl_j = shape_j.getCircumsphereDiameter() / Scalar(2.0);
                            Scalar RaRb = r_excl_i + r_excl_j;
                            Scalar rsq_ij = dot(r_ij, r_ij);

                            unsigned int err = 0;
                            if (rsq_ij <= RaRb * RaRb)
                                {
                                if (h_overlaps.data[overlap_idx(typ_i, typ_j)]
                                    && test_overlap(r_ij, shape_i, shape_j, err))
                                    {
                                    // add connection
                                    m_overlap.insert(std::make_pair(i, j));
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
            Scalar extent_i = 0.5 * m_mc->getMaxPairInteractionAdditiveRCut(typ_i);
            Scalar R_query
                = std::max(0.0, r_cut_patch + extent_i - min_core_diameter / (LongReal)2.0);
            hoomd::detail::AABB aabb_local = hoomd::detail::AABB(vec3<Scalar>(0, 0, 0), R_query);

            // compute V(r'-r)
            for (unsigned int cur_image = 0; cur_image < n_images; cur_image++)
                {
                vec3<Scalar> pos_i_image = pos_i_new + image_list[cur_image];

                hoomd::detail::AABB aabb_i_image = aabb_local;
                aabb_i_image.translate(pos_i_image);

                // stackless search
                for (unsigned int cur_node_idx = 0; cur_node_idx < m_aabb_tree_old.getNumNodes();
                     cur_node_idx++)
                    {
                    if (aabb_i_image.overlaps(m_aabb_tree_old.getNodeAABB(cur_node_idx)))
                        {
                        if (m_aabb_tree_old.isNodeLeaf(cur_node_idx))
                            {
                            for (unsigned int cur_p = 0;
                                 cur_p < m_aabb_tree_old.getNodeNumParticles(cur_node_idx);
                                 cur_p++)
                                {
                                // read in its position and orientation
                                unsigned int j
                                    = m_aabb_tree_old.getNodeParticle(cur_node_idx, cur_p);

                                if (i == j && cur_image == 0)
                                    continue;

                                vec3<Scalar> pos_j(h_postype_backup.data[j]);
                                unsigned int typ_j = __scalar_as_int(h_postype_backup.data[j].w);

                                // put particles in coordinate system of particle i
                                vec3<Scalar> r_ij = pos_j - pos_i_image;

                                // check for excluded volume sphere overlap
                                Scalar rsq_ij = dot(r_ij, r_ij);

                                Scalar rcut_ij
                                    = r_cut_patch + extent_i
                                      + 0.5 * m_mc->getMaxPairInteractionAdditiveRCut(typ_j);

                                if (rsq_ij <= rcut_ij * rcut_ij)
                                    {
                                    auto p = std::make_pair(i, j);

                                    // if particle interacts in different image already, add to that
                                    // energy
                                    LongReal U = 0.0;
                                        {
                                        auto it_energy = m_energy_new_old.find(p);
                                        if (it_energy != m_energy_new_old.end())
                                            U = it_energy->second;
                                        }

                                    U += m_mc->computeOnePairEnergy(
                                        rsq_ij,
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
    }

template<class Shape> void UpdaterGCA<Shape>::backupState()
    {
    unsigned int nptl = m_pdata->getN();

    // resize as necessary
    m_postype_backup.resize(nptl);
    m_orientation_backup.resize(nptl);
    m_image_backup.resize(nptl);

        {
        ArrayHandle<Scalar4> h_postype(m_pdata->getPositions(),
                                       access_location::host,
                                       access_mode::read);
        ArrayHandle<Scalar4> h_orientation(m_pdata->getOrientationArray(),
                                           access_location::host,
                                           access_mode::read);
        ArrayHandle<int3> h_image(m_pdata->getImages(), access_location::host, access_mode::read);

        ArrayHandle<Scalar4> h_postype_backup(m_postype_backup,
                                              access_location::host,
                                              access_mode::overwrite);
        ArrayHandle<Scalar4> h_orientation_backup(m_orientation_backup,
                                                  access_location::host,
                                                  access_mode::overwrite);
        ArrayHandle<int3> h_image_backup(m_image_backup,
                                         access_location::host,
                                         access_mode::overwrite);

        // copy over data
        for (unsigned int i = 0; i < nptl; ++i)
            {
            h_postype_backup.data[i] = h_postype.data[i];
            h_orientation_backup.data[i] = h_orientation.data[i];
            h_image_backup.data[i] = h_image.data[i];
            }
        }
    }

template<class Shape> void UpdaterGCA<Shape>::connectedComponents()
    {
    // compute connected components
    m_clusters.clear();
    m_G.connectedComponents(m_clusters);
    }

/*! Perform a cluster move
    \param timestep Current time step of the simulation
*/
template<class Shape> void UpdaterGCA<Shape>::update(uint64_t timestep)
    {
    Updater::update(timestep);
#ifdef ENABLE_MPI
    if (this->m_pdata->getDomainDecomposition())
        throw std::runtime_error("UpdaterGCA does not work with spatial domain decomposition.");
#endif

    m_exec_conf->msg->notice(10) << timestep << " UpdaterGCA" << std::endl;

    m_count_step_start = m_count_total;

    // if no particles, exit early
    if (!m_pdata->getNGlobal())
        return;

    const uint16_t seed = m_sysdef->getSeed();

    // generate the move, select a pivot
    hoomd::RandomGenerator rng(hoomd::Seed(hoomd::RNGIdentifier::UpdaterGCA, timestep, seed),
                               hoomd::Counter(m_instance));
    vec3<Scalar> pivot(0, 0, 0);

    // is this a line reflection?
    bool line = (m_mc->hasOrientation()
                 || (hoomd::detail::generate_canonical<Scalar>(rng) > m_move_ratio));

    quat<Scalar> q;

    if (line)
        {
        // random normalized vector
        vec3<Scalar> n;

        if (m_sysdef->getNDimensions() == 3)
            {
            unsigned int i = hoomd::UniformIntDistribution(2)(rng);
            if (i == 0)
                n = vec3<Scalar>(1, 0, 0);
            else if (i == 1)
                n = vec3<Scalar>(0, 1, 0);
            else if (i == 2)
                n = vec3<Scalar>(0, 0, 1);
            }
        else
            {
            // reflection around z axis
            n = vec3<Scalar>(0, 0, 1);
            }

        // line reflection
        q = quat<Scalar>(0, n);
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

    auto& r = m_overlap;
        {
        for (auto it = r.begin(); it != r.end(); ++it)
            {
            unsigned int i = it->first;
            unsigned int j = it->second;

            // we're adding a directed edge, but the actual graph will be undirected
            // if the symmetry operation is self-inverse
            m_G.addEdge(i, j);
            }
        }

    if (m_mc->hasPairInteractions())
        {
        // sum up interaction energies
        std::map<std::pair<unsigned int, unsigned int>, LongReal> delta_U;

        for (auto it = m_energy_old_old.begin(); it != m_energy_old_old.end(); ++it)
            {
            LongReal delU = -it->second;
            unsigned int i = it->first.first;
            unsigned int j = it->first.second;

            auto p = std::make_pair(i, j);

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

            auto p = std::make_pair(i, j);

            // add to energy
            auto itj = delta_U.find(p);
            if (itj != delta_U.end())
                delU += itj->second;

            // update map with new interaction energy
            delta_U[p] = delU;
            }

        const Scalar kT = (*m_mc->getKT())(timestep);

        auto& r = delta_U;
            {
            for (auto it = r.begin(); it != r.end(); ++it)
                {
                LongReal delU = it->second;
                unsigned int i = it->first.first;
                unsigned int j = it->first.second;

                // create a RNG specific to this particle pair
                hoomd::RandomGenerator rng_ij(
                    hoomd::Seed(hoomd::RNGIdentifier::UpdaterGCAPairwise, timestep, seed),
                    hoomd::Counter(std::min(i, j), std::max(i, j)));

                LongReal pij = 1.0f - exp(-delU / kT);
                if (hoomd::detail::generate_canonical<LongReal>(rng_ij) <= pij) // GCA
                    {
                    // add bond
                    m_G.addEdge(i, j);
                    }
                }
            }
        } // end if (m_mc->hasPairInteractions)

    // compute connected components
    connectedComponents();

    // flip clusters randomly
    flip(timestep);

    m_mc->invalidateAABBTree();
    }

namespace detail
    {

template<class Shape> void export_UpdaterGCA(pybind11::module& m, const std::string& name)
    {
    pybind11::class_<UpdaterGCA<Shape>, Updater, std::shared_ptr<UpdaterGCA<Shape>>>(m,
                                                                                     name.c_str())
        .def(pybind11::init<std::shared_ptr<SystemDefinition>,
                            std::shared_ptr<Trigger>,
                            std::shared_ptr<IntegratorHPMCMono<Shape>>>())
        .def("getCounters", &UpdaterGCA<Shape>::getCounters)
        .def_property("pivot_move_probability",
                      &UpdaterGCA<Shape>::getMoveRatio,
                      &UpdaterGCA<Shape>::setMoveRatio)
        .def_property("flip_probability",
                      &UpdaterGCA<Shape>::getFlipProbability,
                      &UpdaterGCA<Shape>::setFlipProbability);
    }

inline void export_hpmc_clusters_counters(pybind11::module& m)
    {
    pybind11::class_<hpmc_clusters_counters_t>(m, "hpmc_clusters_counters_t")
        .def_property_readonly("average_cluster_size",
                               &hpmc_clusters_counters_t::getAverageClusterSize);
    }

    } // end namespace detail
    } // end namespace hpmc
    } // end namespace hoomd
