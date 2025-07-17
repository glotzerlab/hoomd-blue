// Copyright (c) 2009-2025 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

#pragma once

#include <iostream>

#include "GSDHPMCSchema.h"
#include "IntegratorHPMC.h"
#include "Moves.h"
#include "ShapeSpheropolyhedron.h"
#include "hoomd/AABBTree.h"
#include "hoomd/Index1D.h"
#include "hoomd/Integrator.h"
#include "hoomd/RNGIdentifiers.h"
#include "hoomd/RandomNumbers.h"
#include "hoomd/managed_allocator.h"

#ifdef ENABLE_MPI
#include "hoomd/Communicator.h"
#include "hoomd/HOOMDMPI.h"
#endif

#ifndef __HIPCC__
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#endif

namespace hoomd
    {

namespace hpmc
    {

namespace detail
    {

//! Helper class to manage shuffled update orders
/*! Stores an update order from 0 to N-1, inclusive, and can be resized. shuffle() shuffles the
   order of elements to a new random permutation. operator [i] gets the index of the item at order i
   in the current shuffled sequence.

    \ingroup hpmc_data_structs
*/
class UpdateOrder
    {
    public:
    //! Constructor
    /*! \param seed Random number seed
        \param N number of integers to shuffle
    */
    UpdateOrder(unsigned int N = 0)
        {
        resize(N);
        }

    //! Resize the order
    /*! \param N new size
        \post The order is 0, 1, 2, ... N-1
    */
    void resize(unsigned int N)
        {
        // initialize the update order
        m_update_order.resize(N);
        for (unsigned int i = 0; i < N; i++)
            m_update_order[i] = i;
        }

    //! Shuffle the order
    /*! \param timestep Current timestep of the simulation
        \note \a timestep is used to seed the RNG, thus assuming that the order is shuffled only
       once per timestep.
    */
    void shuffle(uint64_t timestep, uint16_t seed, unsigned int rank = 0, unsigned int select = 0)
        {
        hoomd::RandomGenerator rng(
            hoomd::Seed(hoomd::RNGIdentifier::HPMCMonoShuffle, timestep, seed),
            hoomd::Counter(rank, select));

        // reverse the order with 1/2 probability
        if (hoomd::UniformIntDistribution(1)(rng))
            {
            unsigned int N = (unsigned int)m_update_order.size();
            for (unsigned int i = 0; i < N; i++)
                m_update_order[i] = N - i - 1;
            }
        else
            {
            unsigned int N = (unsigned int)m_update_order.size();
            for (unsigned int i = 0; i < N; i++)
                m_update_order[i] = i;
            }
        }

    //! Access element of the shuffled order
    unsigned int operator[](unsigned int i)
        {
        return m_update_order[i];
        }

    private:
    std::vector<unsigned int> m_update_order; //!< Update order
    };

    }; // end namespace detail

//! HPMC on systems of mono-disperse shapes
/*! Implement hard particle monte carlo for a single type of shape on the CPU.
 */
template<class Shape> class IntegratorHPMCMono : public IntegratorHPMC
    {
    public:
    //! Param type from the shape
    //! Each shape has a param_type member that contain
    //! shape-specific descriptors(radius, vertices, etc)
    typedef typename Shape::param_type param_type;

    //! Constructor
    IntegratorHPMCMono(std::shared_ptr<SystemDefinition> sysdef);

    virtual ~IntegratorHPMCMono()
        {
        if (m_aabbs != NULL)
            free(m_aabbs);
        m_pdata->getBoxChangeSignal()
            .template disconnect<IntegratorHPMCMono<Shape>,
                                 &IntegratorHPMCMono<Shape>::slotBoxChanged>(this);
        m_pdata->getParticleSortSignal()
            .template disconnect<IntegratorHPMCMono<Shape>, &IntegratorHPMCMono<Shape>::slotSorted>(
                this);
        }

    //! Take one timestep forward
    virtual void update(uint64_t timestep);

    /*
     * Common HPMC API
     */

    //! Get the maximum particle diameter
    virtual Scalar getMaxCoreDiameter();

    //! Get the minimum particle diameter
    virtual LongReal getMinCoreDiameter();

    //! Set the pair parameters for a single type
    virtual void setParam(unsigned int typ, const param_type& param);

    //! Set shape parameters from python
    void setShape(std::string typ, pybind11::dict v);

    //! Get shape parameter from python
    pybind11::dict getShape(std::string typ);

    //! Set elements of the interaction matrix
    virtual void setInteractionMatrix(std::pair<std::string, std::string> types,
                                      bool check_overlaps);

    //! Get elements of the interaction matrix
    virtual bool getInteractionMatrixPy(std::pair<std::string, std::string> types);

    //! Get the particle parameters
    virtual std::vector<param_type, hoomd::detail::managed_allocator<param_type>>& getParams()
        {
        return m_params;
        }

    //! Get the interaction matrix
    virtual const GPUArray<unsigned int>& getInteractionMatrix()
        {
        return m_overlaps;
        }

    //! Get the indexer for the interaction matrix
    virtual const Index2D& getOverlapIndexer()
        {
        return m_overlap_idx;
        }

    //! Count overlaps with the option to exit early at the first detected overlap
    virtual unsigned int countOverlaps(bool early_exit);

    //! Return a vector that is an unwrapped overlap map
    virtual std::vector<std::pair<unsigned int, unsigned int>> mapOverlaps();

    //! Return the requested ghost layer width
    virtual Scalar getGhostLayerWidth(unsigned int type)
        {
        Scalar ghost_width = m_nominal_width + m_extra_ghost_width;
        m_exec_conf->msg->notice(9)
            << "IntegratorHPMCMono: ghost layer width of " << ghost_width << std::endl;
        return ghost_width;
        }

#ifdef ENABLE_MPI
    //! Return the requested communication flags for ghost particles
    virtual CommFlags getCommFlags(uint64_t timestep)
        {
        CommFlags flags(0);
        flags[comm_flag::position] = 1;
        flags[comm_flag::tag] = 1;

        std::ostringstream o;
        o << "IntegratorHPMCMono: Requesting communication flags for pos tag orientation";

        // many things depend internally on the orientation field (for ghosts) being initialized,
        // therefore always request it
        flags[comm_flag::orientation] = 1;

        if (hasPairInteractions())
            {
            flags[comm_flag::diameter] = 1;
            flags[comm_flag::charge] = 1;
            o << " diameter charge";
            }

        m_exec_conf->msg->notice(9) << o.str() << std::endl;
        return flags;
        }
#endif

    //! Prepare for the run
    virtual void prepRun(uint64_t timestep)
        {
        // base class method
        IntegratorHPMC::prepRun(timestep);

        m_hasOrientation = false;
        quat<Scalar> q(make_scalar4(1, 0, 0, 0));
        for (unsigned int i = 0; i < m_pdata->getNTypes(); i++)
            {
            Shape dummy(q, m_params[i]);
            if (dummy.hasOrientation())
                {
                m_hasOrientation = true;
                break;
                }
            }

        updateCellWidth(); // make sure the cell width is up-to-date and forces a rebuild of the
                           // AABB tree and image list

        communicate(true);
        }

    //! Communicate particles
    virtual void communicate(bool migrate)
        {
// migrate and exchange particles
#ifdef ENABLE_MPI
        if (m_sysdef->isDomainDecomposed())
            {
            if (migrate)
                {
                m_comm->forceMigrate();
                }

            // The timestep flag is not used in HPMC simulations, pass 0.
            m_comm->communicate(0);

            m_aabb_tree_invalid = true;
            }
#endif
        }

    //! Return true if anisotropic particles are present
    virtual bool hasOrientation()
        {
        return m_hasOrientation;
        }

    //! Compute the total energy from pair interactions.
    /*! \param timestep the current time step
     * \returns the total patch energy
     */
    virtual double computeTotalPairEnergy(uint64_t timestep);

    /*** Compute the pair energy from one pair potential or all pair interactions

    @param selected_pair Pair potential to use when computing the energy. When null, compute
    the total energy of all pair interactions.

    Note: selected_pair *must* be a member of pair_potentials to ensure that the AABB tree
    is constructed properly.
    **/
    double computePairEnergy(uint64_t timestep,
                             std::shared_ptr<PairPotential> selected_pair = nullptr);

    //! Build the AABB tree (if needed)
    const hoomd::detail::AABBTree& buildAABBTree();

    //! Make list of image indices for boxes to check in small-box mode
    const std::vector<vec3<Scalar>>& updateImageList();

    //! Return list of integer shift vectors for periodic images
    const std::vector<int3>& getImageHKL()
        {
        updateImageList();
        return m_image_hkl;
        }

    void invalidateAABBTree()
        {
        m_aabb_tree_invalid = true;
        }

    std::vector<std::string> getTypeShapeMapping(
        const std::vector<param_type, hoomd::detail::managed_allocator<param_type>>& params) const
        {
        quat<Scalar> q(make_scalar4(1, 0, 0, 0));
        std::vector<std::string> type_shape_mapping(params.size());
        for (unsigned int i = 0; i < type_shape_mapping.size(); i++)
            {
            Shape shape(q, params[i]);
            type_shape_mapping[i] = getShapeSpec(shape);
            }
        return type_shape_mapping;
        }

    pybind11::list getTypeShapesPy()
        {
        std::vector<std::string> type_shape_mapping = this->getTypeShapeMapping(this->m_params);
        pybind11::list type_shapes;
        for (unsigned int i = 0; i < type_shape_mapping.size(); i++)
            type_shapes.append(type_shape_mapping[i]);
        return type_shapes;
        }

    protected:
    std::vector<param_type, hoomd::detail::managed_allocator<param_type>>
        m_params;                       //!< Parameters for each particle type on GPU
    GPUArray<unsigned int> m_overlaps;  //!< Interaction matrix (0/1) for overlap checks
    detail::UpdateOrder m_update_order; //!< Update order
    bool m_image_list_is_initialized;   //!< true if image list has been used
    bool m_image_list_valid; //!< image list is invalid if the box dimensions or particle parameters
                             //!< have changed.
    std::vector<vec3<Scalar>>
        m_image_list; //!< List of potentially interacting simulation box images
    std::vector<int3>
        m_image_hkl; //!< List of potentially interacting simulation box images (integer shifts)
    unsigned int m_image_list_rebuilds; //!< Number of times the image list has been rebuilt
    bool m_image_list_warning_issued;   //!< True if the image list warning has been issued
    bool m_hkl_max_warning_issued;      //!< True if the image list size warning has been issued
    bool m_hasOrientation; //!< true if there are any orientable particles in the system

    hoomd::detail::AABBTree m_aabb_tree; //!< Bounding volume hierarchy for overlap checks
    hoomd::detail::AABB* m_aabbs;        //!< list of AABBs, one per particle
    unsigned int m_aabbs_capacity;       //!< Capacity of m_aabbs list
    bool m_aabb_tree_invalid;            //!< Flag if the aabb tree has been invalidated

    Index2D m_overlap_idx; //!!< Indexer for interaction matrix

    /// Cached maximum pair additive cutoff by type.
    std::vector<LongReal> m_max_pair_additive_cutoff;

    /// Cached shape radius by type.
    std::vector<LongReal> m_shape_circumsphere_radius;

    //! Set the nominal width appropriate for looped moves
    virtual void updateCellWidth();

    //! Grow the m_aabbs list
    virtual void growAABBList(unsigned int N);

    //! Limit the maximum move distances
    virtual void limitMoveDistances();

    //! callback so that the box change signal can invalidate the image list
    virtual void slotBoxChanged()
        {
        m_image_list_valid = false;
        // changing the box does not necessarily invalidate the AABB tree - however, practically
        // anything that changes the box (i.e. NPT, box_resize) is also moving the particles,
        // so use it as a sign to rebuild the AABB tree
        m_aabb_tree_invalid = true;
        }

    //! callback so that the particle sort signal can invalidate the AABB tree
    virtual void slotSorted()
        {
        m_aabb_tree_invalid = true;
        }
    };

template<class Shape>
IntegratorHPMCMono<Shape>::IntegratorHPMCMono(std::shared_ptr<SystemDefinition> sysdef)
    : IntegratorHPMC(sysdef), m_update_order(m_pdata->getN()), m_image_list_is_initialized(false),
      m_image_list_valid(false), m_hasOrientation(true)
    {
    // allocate the parameter storage, setting the managed flag
    m_params = std::vector<param_type, hoomd::detail::managed_allocator<param_type>>(
        m_pdata->getNTypes(),
        param_type(),
        hoomd::detail::managed_allocator<param_type>(m_exec_conf->isCUDAEnabled()));

    m_overlap_idx = Index2D(m_pdata->getNTypes());
    GPUArray<unsigned int> overlaps(m_overlap_idx.getNumElements(), m_exec_conf);
    m_overlaps.swap(overlaps);
    ArrayHandle<unsigned int> h_overlaps(m_overlaps, access_location::host, access_mode::readwrite);
    for (unsigned int i = 0; i < m_overlap_idx.getNumElements(); i++)
        {
        h_overlaps.data[i] = 1; // Assume we want to check overlaps.
        }

    // Connect to the BoxChange signal
    m_pdata->getBoxChangeSignal()
        .template connect<IntegratorHPMCMono<Shape>, &IntegratorHPMCMono<Shape>::slotBoxChanged>(
            this);
    m_pdata->getParticleSortSignal()
        .template connect<IntegratorHPMCMono<Shape>, &IntegratorHPMCMono<Shape>::slotSorted>(this);

    m_image_list_rebuilds = 0;
    m_image_list_warning_issued = false;
    m_hkl_max_warning_issued = false;

    m_aabbs = NULL;
    m_aabbs_capacity = 0;
    m_aabb_tree_invalid = true;
    }

template<class Shape> void IntegratorHPMCMono<Shape>::update(uint64_t timestep)
    {
    Integrator::update(timestep);
    m_exec_conf->msg->notice(10) << "HPMCMono update: " << timestep << std::endl;
    IntegratorHPMC::update(timestep);

    // get needed vars
    const BoxDim box = m_pdata->getBox();
    unsigned int ndim = this->m_sysdef->getNDimensions();

#ifdef ENABLE_MPI
    // compute the width of the active region
    Scalar3 npd = box.getNearestPlaneDistance();
    Scalar3 ghost_fraction = m_nominal_width / npd;
#endif

    // Shuffle the order of particles for this step
    m_update_order.resize(m_pdata->getN());
    m_update_order.shuffle(timestep, m_sysdef->getSeed(), m_exec_conf->getRank());

    // update the AABB Tree. Assume that it is invalid as previously called Updaters may have
    // moved particles.
    m_aabb_tree_invalid = true;
    buildAABBTree();
    // limit m_d entries so that particles cannot possibly wander more than one box image in one
    // time step
    limitMoveDistances();
    // update the image list
    updateImageList();

    uint16_t seed = m_sysdef->getSeed();

    // access interaction matrix
    ArrayHandle<unsigned int> h_overlaps(m_overlaps, access_location::host, access_mode::read);

    // precompute constants used many times in the loop
    const LongReal min_core_radius = getMinCoreDiameter() * LongReal(0.5);
    const auto& pair_energy_search_radius = getPairEnergySearchRadius();

    const Scalar kT = this->getKT()->operator()(timestep);

    m_max_pair_additive_cutoff.clear();
    m_shape_circumsphere_radius.clear();
    for (unsigned int type = 0; type < m_pdata->getNTypes(); type++)
        {
        quat<LongReal> q;
        Shape shape(q, m_params[type]);
        m_shape_circumsphere_radius.push_back(LongReal(0.5) * shape.getCircumsphereDiameter());
        m_max_pair_additive_cutoff.push_back(getMaxPairInteractionAdditiveRCut(type));
        }

    // loop over local particles nselect times
    for (unsigned int i_nselect = 0; i_nselect < m_nselect; i_nselect++)
        {
        ArrayHandle<hpmc_counters_t> h_counters(m_count_total,
                                                access_location::host,
                                                access_mode::readwrite);
        hpmc_counters_t& counters = h_counters.data[0];

        // access particle data and system box
        ArrayHandle<Scalar4> h_postype(m_pdata->getPositions(),
                                       access_location::host,
                                       access_mode::readwrite);
        ArrayHandle<Scalar4> h_orientation(m_pdata->getOrientationArray(),
                                           access_location::host,
                                           access_mode::readwrite);
        ArrayHandle<Scalar> h_diameter(m_pdata->getDiameters(),
                                       access_location::host,
                                       access_mode::read);
        ArrayHandle<Scalar> h_charge(m_pdata->getCharges(),
                                     access_location::host,
                                     access_mode::read);
        ArrayHandle<unsigned int> h_tag(m_pdata->getTags(),
                                        access_location::host,
                                        access_mode::read);
        ArrayHandle<Scalar4> h_vel(m_pdata->getVelocities(),
                                   access_location::host,
                                   access_mode::readwrite);

        // access move sizes
        ArrayHandle<Scalar> h_d(m_d, access_location::host, access_mode::read);
        ArrayHandle<Scalar> h_a(m_a, access_location::host, access_mode::read);

        // loop through N particles in a shuffled order
        for (unsigned int cur_particle = 0; cur_particle < m_pdata->getN(); cur_particle++)
            {
            unsigned int i = m_update_order[cur_particle];

            // read in the current position and orientation
            Scalar4 postype_i = h_postype.data[i];
            vec3<Scalar> pos_i = vec3<Scalar>(postype_i);

#ifdef ENABLE_MPI
            if (m_sysdef->isDomainDecomposed())
                {
                // only move particle if active
                if (!isActive(make_scalar3(postype_i.x, postype_i.y, postype_i.z),
                              box,
                              ghost_fraction))
                    continue;
                }
#endif

            // make a trial move for i
            hoomd::RandomGenerator rng_i(
                hoomd::Seed(hoomd::RNGIdentifier::HPMCMonoTrialMove, timestep, seed),
                hoomd::Counter(i, m_exec_conf->getRank(), i_nselect));
            int typ_i = __scalar_as_int(postype_i.w);
            Shape shape_i(quat<LongReal>(h_orientation.data[i]), m_params[typ_i]);
            unsigned int move_type_select = hoomd::UniformIntDistribution(0xffff)(rng_i);
            bool move_type_translate
                = !shape_i.hasOrientation() || (move_type_select < m_translation_move_probability);

            Shape shape_old(shape_i.orientation, m_params[typ_i]);
            vec3<Scalar> pos_old = pos_i;

            if (move_type_translate)
                {
                // skip if no overlap check is required
                if (h_d.data[typ_i] == 0.0)
                    {
                    if (!shape_i.ignoreStatistics())
                        counters.translate_accept_count++;
                    continue;
                    }

                move_translate(pos_i, rng_i, h_d.data[typ_i], ndim);

#ifdef ENABLE_MPI
                if (m_sysdef->isDomainDecomposed())
                    {
                    // check if particle has moved into the ghost layer, and skip if it is
                    if (!isActive(vec_to_scalar3(pos_i), box, ghost_fraction))
                        continue;
                    }
#endif
                }
            else
                {
                if (h_a.data[typ_i] == 0.0)
                    {
                    if (!shape_i.ignoreStatistics())
                        counters.rotate_accept_count++;
                    continue;
                    }

                if (ndim == 2)
                    move_rotate<2>(shape_i.orientation, rng_i, h_a.data[typ_i]);
                else
                    move_rotate<3>(shape_i.orientation, rng_i, h_a.data[typ_i]);
                }

            bool overlap = false;

            // search for all particles that might touch this one
            LongReal R_query = m_shape_circumsphere_radius[typ_i];

            if (hasPairInteractions())
                {
                // Extend the search to include the pair interaction r_cut
                // subtract minimum AABB extent from search radius
                R_query = std::max(R_query, pair_energy_search_radius[typ_i] - min_core_radius);
                }

            hoomd::detail::AABB aabb_i_local = hoomd::detail::AABB(vec3<Scalar>(0, 0, 0), R_query);

            // patch + field interaction deltaU
            double patch_field_energy_diff = 0;

            // check for overlaps with neighboring particle's positions (also calculate the new
            // energy) All image boxes (including the primary)
            const unsigned int n_images = (unsigned int)m_image_list.size();
            for (unsigned int cur_image = 0; cur_image < n_images; cur_image++)
                {
                vec3<Scalar> pos_i_image = pos_i + m_image_list[cur_image];
                hoomd::detail::AABB aabb = aabb_i_local;
                aabb.translate(pos_i_image);

                // stackless search
                for (unsigned int cur_node_idx = 0; cur_node_idx < m_aabb_tree.getNumNodes();
                     cur_node_idx++)
                    {
                    if (aabb.overlaps(m_aabb_tree.getNodeAABB(cur_node_idx)))
                        {
                        if (m_aabb_tree.isNodeLeaf(cur_node_idx))
                            {
                            for (unsigned int cur_p = 0;
                                 cur_p < m_aabb_tree.getNodeNumParticles(cur_node_idx);
                                 cur_p++)
                                {
                                // read in its position and orientation
                                unsigned int j = m_aabb_tree.getNodeParticle(cur_node_idx, cur_p);

                                Scalar4 postype_j;
                                quat<LongReal> orientation_j;

                                // handle j==i situations
                                if (j != i)
                                    {
                                    // load the position and orientation of the j particle
                                    postype_j = h_postype.data[j];
                                    orientation_j = quat<LongReal>(h_orientation.data[j]);
                                    }
                                else
                                    {
                                    if (cur_image == 0)
                                        {
                                        // in the first image, skip i == j
                                        continue;
                                        }
                                    else
                                        {
                                        // If this is particle i and we are in an outside image, use
                                        // the translated position and orientation
                                        postype_j
                                            = make_scalar4(pos_i.x, pos_i.y, pos_i.z, postype_i.w);
                                        orientation_j = shape_i.orientation;
                                        }
                                    }

                                // put particles in coordinate system of particle i
                                vec3<Scalar> r_ij = vec3<Scalar>(postype_j) - pos_i_image;

                                unsigned int typ_j = __scalar_as_int(postype_j.w);
                                Shape shape_j(orientation_j, m_params[typ_j]);

                                LongReal r_squared = dot(r_ij, r_ij);
                                LongReal max_overlap_distance
                                    = m_shape_circumsphere_radius[typ_i]
                                      + m_shape_circumsphere_radius[typ_j];

                                counters.overlap_checks++;
                                if (h_overlaps.data[m_overlap_idx(typ_i, typ_j)]
                                    && r_squared < max_overlap_distance * max_overlap_distance
                                    && test_overlap(r_ij,
                                                    shape_i,
                                                    shape_j,
                                                    counters.overlap_err_count))
                                    {
                                    overlap = true;
                                    break;
                                    }

                                // deltaU = U_old - U_new: subtract energy of new configuration
                                patch_field_energy_diff -= computeOnePairEnergy(r_squared,
                                                                                r_ij,
                                                                                typ_i,
                                                                                shape_i.orientation,
                                                                                h_diameter.data[i],
                                                                                h_charge.data[i],
                                                                                typ_j,
                                                                                shape_j.orientation,
                                                                                h_diameter.data[j],
                                                                                h_charge.data[j]);
                                }
                            }
                        }
                    else
                        {
                        // skip ahead
                        cur_node_idx += m_aabb_tree.getNodeSkip(cur_node_idx);
                        }

                    if (overlap)
                        break;
                    } // end loop over AABB nodes

                if (overlap)
                    break;
                } // end loop over images

            // Calculate old pair energy only when there are pair energies to calculate.
            if (hasPairInteractions() && !overlap)
                {
                for (unsigned int cur_image = 0; cur_image < n_images; cur_image++)
                    {
                    vec3<Scalar> pos_i_image = pos_old + m_image_list[cur_image];
                    hoomd::detail::AABB aabb = aabb_i_local;
                    aabb.translate(pos_i_image);

                    // stackless search
                    for (unsigned int cur_node_idx = 0; cur_node_idx < m_aabb_tree.getNumNodes();
                         cur_node_idx++)
                        {
                        if (aabb.overlaps(m_aabb_tree.getNodeAABB(cur_node_idx)))
                            {
                            if (m_aabb_tree.isNodeLeaf(cur_node_idx))
                                {
                                for (unsigned int cur_p = 0;
                                     cur_p < m_aabb_tree.getNodeNumParticles(cur_node_idx);
                                     cur_p++)
                                    {
                                    // read in its position and orientation
                                    unsigned int j
                                        = m_aabb_tree.getNodeParticle(cur_node_idx, cur_p);

                                    Scalar4 postype_j;
                                    quat<LongReal> orientation_j;

                                    // handle j==i situations
                                    if (j != i)
                                        {
                                        // load the position and orientation of the j particle
                                        postype_j = h_postype.data[j];
                                        orientation_j = quat<LongReal>(h_orientation.data[j]);
                                        }
                                    else
                                        {
                                        if (cur_image == 0)
                                            {
                                            // in the first image, skip i == j
                                            continue;
                                            }
                                        else
                                            {
                                            // If this is particle i and we are in an outside image,
                                            // use the translated position and orientation
                                            postype_j = make_scalar4(pos_old.x,
                                                                     pos_old.y,
                                                                     pos_old.z,
                                                                     postype_i.w);
                                            orientation_j = shape_old.orientation;
                                            }
                                        }

                                    // put particles in coordinate system of particle i
                                    vec3<Scalar> r_ij = vec3<Scalar>(postype_j) - pos_i_image;
                                    unsigned int typ_j = __scalar_as_int(postype_j.w);
                                    Shape shape_j(orientation_j, m_params[typ_j]);

                                    // deltaU = U_old - U_new: add energy of old configuration
                                    patch_field_energy_diff
                                        += computeOnePairEnergy(dot(r_ij, r_ij),
                                                                r_ij,
                                                                typ_i,
                                                                shape_old.orientation,
                                                                h_diameter.data[i],
                                                                h_charge.data[i],
                                                                typ_j,
                                                                shape_j.orientation,
                                                                h_diameter.data[j],
                                                                h_charge.data[j]);
                                    }
                                }
                            }
                        else
                            {
                            // skip ahead
                            cur_node_idx += m_aabb_tree.getNodeSkip(cur_node_idx);
                            }
                        } // end loop over AABB nodes
                    } // end loop over images
                }

            // Add external energetic contribution if there are no overlaps
            if (!overlap)
                {
                // U_old - U_new
                patch_field_energy_diff
                    += this->computeOneExternalEnergy(timestep,
                                                      h_tag.data[i],
                                                      typ_i,
                                                      pos_old,
                                                      shape_old.orientation,
                                                      h_charge.data[i],
                                                      ExternalPotential::Trial::Old)
                       - this->computeOneExternalEnergy(timestep,
                                                        h_tag.data[i],
                                                        typ_i,
                                                        pos_i,
                                                        shape_i.orientation,
                                                        h_charge.data[i],
                                                        ExternalPotential::Trial::New);
                }

            bool accept = !overlap
                          && hoomd::detail::generate_canonical<double>(rng_i)
                                 < slow::exp(patch_field_energy_diff / kT);

            // If no overlaps and Metropolis criterion is met, accept
            // trial move and update positions  and/or orientations.
            if (accept)
                {
                // increment accept counter and assign new position
                if (!shape_i.ignoreStatistics())
                    {
                    if (move_type_translate)
                        counters.translate_accept_count++;
                    else
                        counters.rotate_accept_count++;
                    }

                // update the position of the particle in the tree for future updates
                hoomd::detail::AABB aabb;
                if (!hasPairInteractions())
                    {
                    aabb = shape_i.getAABB(pos_i);
                    }
                else
                    {
                    Scalar radius = std::max(m_shape_circumsphere_radius[typ_i],
                                             LongReal(0.5) * m_max_pair_additive_cutoff[typ_i]);
                    aabb = hoomd::detail::AABB(pos_i, radius);
                    }

                m_aabb_tree.update(i, aabb);

                // update position of particle
                h_postype.data[i] = make_scalar4(pos_i.x, pos_i.y, pos_i.z, postype_i.w);

                if (shape_i.hasOrientation())
                    {
                    h_orientation.data[i] = quat_to_scalar4(shape_i.orientation);
                    }
                }
            else
                {
                if (!shape_i.ignoreStatistics())
                    {
                    // increment reject counter
                    if (move_type_translate)
                        counters.translate_reject_count++;
                    else
                        counters.rotate_reject_count++;
                    }
                }
            } // end loop over all particles
        } // end loop over nselect

        {
        ArrayHandle<Scalar4> h_postype(m_pdata->getPositions(),
                                       access_location::host,
                                       access_mode::readwrite);
        ArrayHandle<int3> h_image(m_pdata->getImages(),
                                  access_location::host,
                                  access_mode::readwrite);
        // wrap particles back into box
        for (unsigned int i = 0; i < m_pdata->getN(); i++)
            {
            box.wrap(h_postype.data[i], h_image.data[i]);
            }
        }

// perform the grid shift
#ifdef ENABLE_MPI
    if (m_sysdef->isDomainDecomposed())
        {
        ArrayHandle<Scalar4> h_postype(m_pdata->getPositions(),
                                       access_location::host,
                                       access_mode::readwrite);
        ArrayHandle<int3> h_image(m_pdata->getImages(),
                                  access_location::host,
                                  access_mode::readwrite);

        // precalculate the grid shift
        hoomd::RandomGenerator rng(
            hoomd::Seed(hoomd::RNGIdentifier::HPMCMonoShift, timestep, this->m_sysdef->getSeed()),
            hoomd::Counter());
        Scalar3 shift = make_scalar3(0, 0, 0);
        hoomd::UniformDistribution<Scalar> uniform(-m_nominal_width / Scalar(2.0),
                                                   m_nominal_width / Scalar(2.0));
        shift.x = uniform(rng);
        shift.y = uniform(rng);
        if (this->m_sysdef->getNDimensions() == 3)
            {
            shift.z = uniform(rng);
            }
        for (unsigned int i = 0; i < m_pdata->getN(); i++)
            {
            // read in the current position and orientation
            Scalar4 postype_i = h_postype.data[i];
            vec3<Scalar> r_i
                = vec3<Scalar>(postype_i); // translation from local to global coordinates
            r_i += vec3<Scalar>(shift);
            h_postype.data[i] = vec_to_scalar4(r_i, postype_i.w);
            box.wrap(h_postype.data[i], h_image.data[i]);
            }
        this->m_pdata->translateOrigin(shift);
        }
#endif

    // migrate and exchange particles
    communicate(true);

    // all particle have been moved, the aabb tree is now invalid
    m_aabb_tree_invalid = true;

    // set current MPS value
    hpmc_counters_t run_counters = getCounters(1);
    double cur_time = double(m_clock.getTime()) / Scalar(1e9);
    m_mps = double(run_counters.getNMoves()) / cur_time;
    }

/*! \param timestep current step
    \param early_exit exit at first overlap found if true
    \returns number of overlaps if early_exit=false, 1 if early_exit=true
*/
template<class Shape> unsigned int IntegratorHPMCMono<Shape>::countOverlaps(bool early_exit)
    {
    unsigned int overlap_count = 0;
    unsigned int err_count = 0;

    // build an up to date AABB tree
    buildAABBTree();
    // update the image list
    updateImageList();

    // access particle data and system box
    ArrayHandle<Scalar4> h_postype(m_pdata->getPositions(),
                                   access_location::host,
                                   access_mode::read);
    ArrayHandle<Scalar4> h_orientation(m_pdata->getOrientationArray(),
                                       access_location::host,
                                       access_mode::read);
    ArrayHandle<unsigned int> h_tag(m_pdata->getTags(), access_location::host, access_mode::read);

    // access parameters and interaction matrix
    ArrayHandle<unsigned int> h_overlaps(m_overlaps, access_location::host, access_mode::read);

    // Loop over all particles
    for (unsigned int i = 0; i < m_pdata->getN(); i++)
        {
        // read in the current position and orientation
        Scalar4 postype_i = h_postype.data[i];
        Scalar4 orientation_i = h_orientation.data[i];
        unsigned int typ_i = __scalar_as_int(postype_i.w);
        Shape shape_i(quat<Scalar>(orientation_i), m_params[typ_i]);
        vec3<Scalar> pos_i = vec3<Scalar>(postype_i);

        // Check particle against AABB tree for neighbors
        hoomd::detail::AABB aabb_i_local = shape_i.getAABB(vec3<Scalar>(0, 0, 0));

        const unsigned int n_images = (unsigned int)m_image_list.size();
        for (unsigned int cur_image = 0; cur_image < n_images; cur_image++)
            {
            vec3<Scalar> pos_i_image = pos_i + m_image_list[cur_image];
            hoomd::detail::AABB aabb = aabb_i_local;
            aabb.translate(pos_i_image);

            // stackless search
            for (unsigned int cur_node_idx = 0; cur_node_idx < m_aabb_tree.getNumNodes();
                 cur_node_idx++)
                {
                if (aabb.overlaps(m_aabb_tree.getNodeAABB(cur_node_idx)))
                    {
                    if (m_aabb_tree.isNodeLeaf(cur_node_idx))
                        {
                        for (unsigned int cur_p = 0;
                             cur_p < m_aabb_tree.getNodeNumParticles(cur_node_idx);
                             cur_p++)
                            {
                            // read in its position and orientation
                            unsigned int j = m_aabb_tree.getNodeParticle(cur_node_idx, cur_p);

                            // skip i==j in the 0 image
                            if (cur_image == 0 && i == j)
                                continue;

                            Scalar4 postype_j = h_postype.data[j];
                            Scalar4 orientation_j = h_orientation.data[j];

                            // put particles in coordinate system of particle i
                            vec3<Scalar> r_ij = vec3<Scalar>(postype_j) - pos_i_image;

                            unsigned int typ_j = __scalar_as_int(postype_j.w);
                            Shape shape_j(quat<Scalar>(orientation_j), m_params[typ_j]);

                            if (h_tag.data[i] <= h_tag.data[j]
                                && h_overlaps.data[m_overlap_idx(typ_i, typ_j)]
                                && check_circumsphere_overlap(r_ij, shape_i, shape_j)
                                && test_overlap(r_ij, shape_i, shape_j, err_count)
                                && test_overlap(-r_ij, shape_j, shape_i, err_count))
                                {
                                overlap_count++;
                                if (early_exit)
                                    {
                                    // exit early from loop over neighbor particles
                                    break;
                                    }
                                }
                            }
                        }
                    }
                else
                    {
                    // skip ahead
                    cur_node_idx += m_aabb_tree.getNodeSkip(cur_node_idx);
                    }

                if (overlap_count && early_exit)
                    {
                    break;
                    }
                } // end loop over AABB nodes

            if (overlap_count && early_exit)
                {
                break;
                }
            } // end loop over images

        if (overlap_count && early_exit)
            {
            break;
            }
        } // end loop over particles

#ifdef ENABLE_MPI
    if (this->m_pdata->getDomainDecomposition())
        {
        MPI_Allreduce(MPI_IN_PLACE,
                      &overlap_count,
                      1,
                      MPI_UNSIGNED,
                      MPI_SUM,
                      m_exec_conf->getMPICommunicator());
        if (early_exit && overlap_count > 1)
            overlap_count = 1;
        }
#endif

    return overlap_count;
    }

template<class Shape> double IntegratorHPMCMono<Shape>::computeTotalPairEnergy(uint64_t timestep)
    {
    return computePairEnergy(timestep);
    }

template<class Shape>
double IntegratorHPMCMono<Shape>::computePairEnergy(uint64_t timestep,
                                                    std::shared_ptr<PairPotential> selected_pair)
    {
    // sum up in double precision
    double energy = 0.0;

    // return if nothing to do
    if (!hasPairInteractions())
        return 0;

    m_exec_conf->msg->notice(10) << "HPMC compute patch energy: " << timestep << std::endl;

    if (!m_past_first_run)
        {
        throw std::runtime_error("Call simulation.run(0) before evaluating HPMC pair energies.");
        }

    // build an up to date AABB tree
    buildAABBTree();
    // update the image list
    updateImageList();

    // access particle data and system box
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
    ArrayHandle<unsigned int> h_tag(m_pdata->getTags(), access_location::host, access_mode::read);

    // access parameters and interaction matrix
    ArrayHandle<unsigned int> h_overlaps(m_overlaps, access_location::host, access_mode::read);

    // precompute constants used many times in the loop
    const LongReal min_core_radius = getMinCoreDiameter() * LongReal(0.5);
    const auto& pair_energy_search_radius = getPairEnergySearchRadius();

    // Loop over all particles
    for (unsigned int i = 0; i < m_pdata->getN(); i++)
        {
        // read in the current position and orientation
        const Scalar4 postype_i = h_postype.data[i];
        const quat<LongReal> orientation_i(h_orientation.data[i]);
        unsigned int typ_i = __scalar_as_int(postype_i.w);
        const vec3<Scalar> pos_i = vec3<Scalar>(postype_i);

        const Scalar d_i = h_diameter.data[i];
        const Scalar charge_i = h_charge.data[i];

        // subtract minimum AABB extent from search radius
        LongReal R_query = pair_energy_search_radius[typ_i] - min_core_radius;
        hoomd::detail::AABB aabb_i_local = hoomd::detail::AABB(vec3<Scalar>(0, 0, 0), R_query);

        const unsigned int n_images = (unsigned int)m_image_list.size();
        for (unsigned int cur_image = 0; cur_image < n_images; cur_image++)
            {
            vec3<Scalar> pos_i_image = pos_i + m_image_list[cur_image];
            hoomd::detail::AABB aabb = aabb_i_local;
            aabb.translate(pos_i_image);

            // stackless search
            for (unsigned int cur_node_idx = 0; cur_node_idx < m_aabb_tree.getNumNodes();
                 cur_node_idx++)
                {
                if (aabb.overlaps(m_aabb_tree.getNodeAABB(cur_node_idx)))
                    {
                    if (m_aabb_tree.isNodeLeaf(cur_node_idx))
                        {
                        for (unsigned int cur_p = 0;
                             cur_p < m_aabb_tree.getNodeNumParticles(cur_node_idx);
                             cur_p++)
                            {
                            // read in its position and orientation
                            unsigned int j = m_aabb_tree.getNodeParticle(cur_node_idx, cur_p);

                            // skip i==j in the 0 image
                            if (cur_image == 0 && i == j)
                                continue;

                            const Scalar4 postype_j = h_postype.data[j];
                            const quat<LongReal> orientation_j(h_orientation.data[j]);
                            unsigned int typ_j = __scalar_as_int(postype_j.w);
                            const Scalar d_j = h_diameter.data[j];
                            const Scalar charge_j = h_charge.data[j];

                            // put particles in coordinate system of particle i
                            const vec3<Scalar> r_ij = vec3<Scalar>(postype_j) - pos_i_image;

                            // count unique pairs within range
                            if (h_tag.data[i] <= h_tag.data[j])
                                {
                                LongReal r_squared = dot(r_ij, r_ij);
                                if (selected_pair)
                                    {
                                    if (r_squared
                                        < selected_pair->getRCutSquaredTotal(typ_i, typ_j))
                                        {
                                        energy += selected_pair->energy(r_squared,
                                                                        r_ij,
                                                                        typ_i,
                                                                        orientation_i,
                                                                        h_charge.data[i],
                                                                        typ_j,
                                                                        orientation_j,
                                                                        h_charge.data[j]);
                                        }
                                    }
                                else
                                    {
                                    energy += computeOnePairEnergy(r_squared,
                                                                   r_ij,
                                                                   typ_i,
                                                                   orientation_i,
                                                                   d_i,
                                                                   charge_i,
                                                                   typ_j,
                                                                   orientation_j,
                                                                   d_j,
                                                                   charge_j);
                                    }
                                }
                            }
                        }
                    }
                else
                    {
                    // skip ahead
                    cur_node_idx += m_aabb_tree.getNodeSkip(cur_node_idx);
                    }

                } // end loop over AABB nodes
            } // end loop over images
        } // end loop over particles

#ifdef ENABLE_MPI
    if (this->m_pdata->getDomainDecomposition())
        {
        MPI_Allreduce(MPI_IN_PLACE,
                      &energy,
                      1,
                      MPI_DOUBLE,
                      MPI_SUM,
                      m_exec_conf->getMPICommunicator());
        }
#endif

    return energy;
    }

template<class Shape> Scalar IntegratorHPMCMono<Shape>::getMaxCoreDiameter()
    {
    Scalar max_d(0.0);

    // access the type parameters
    ArrayHandle<Scalar> h_d(m_d, access_location::host, access_mode::read);

    // access interaction matrix
    ArrayHandle<unsigned int> h_overlaps(this->m_overlaps,
                                         access_location::host,
                                         access_mode::read);

    // for each type, create a temporary shape and return the maximum sum of diameter and move size
    for (unsigned int typ_i = 0; typ_i < this->m_pdata->getNTypes(); typ_i++)
        {
        Shape temp_i(quat<Scalar>(), m_params[typ_i]);

        for (unsigned int typ_j = 0; typ_j < this->m_pdata->getNTypes(); typ_j++)
            {
            Shape temp_j(quat<Scalar>(), m_params[typ_j]);

            // ignore non-interacting shapes
            if (h_overlaps.data[m_overlap_idx(typ_i, typ_j)])
                max_d = std::max(
                    0.5 * (temp_i.getCircumsphereDiameter() + temp_j.getCircumsphereDiameter()),
                    max_d);
            }
        }
    return max_d;
    }

template<class Shape> LongReal IntegratorHPMCMono<Shape>::getMinCoreDiameter()
    {
    // for each type, create a temporary shape and return the minimum diameter
    LongReal minD = 0;
    for (unsigned int typ = 0; typ < this->m_pdata->getNTypes(); typ++)
        {
        Shape temp(quat<Scalar>(), m_params[typ]);
        minD = std::min(minD, LongReal(temp.getCircumsphereDiameter()));
        }

    if (hasPairInteractions())
        {
        LongReal max_extent = 0.0;
        for (unsigned int typ = 0; typ < this->m_pdata->getNTypes(); typ++)
            max_extent = std::max(max_extent, getMaxPairInteractionAdditiveRCut(typ));
        minD = std::max(LongReal(0.0), minD - max_extent);
        }

    return minD;
    }

/*! \param typ type name to set
    \param v python dictionary to convert to shape
*/
template<class Shape>
inline void IntegratorHPMCMono<Shape>::setShape(std::string typ, pybind11::dict v)
    {
    unsigned int id = this->m_pdata->getTypeByName(typ);
    setParam(id, typename Shape::param_type(v, m_exec_conf->isCUDAEnabled()));
    }

/*! \param typ type name to get
 */
template<class Shape> inline pybind11::dict IntegratorHPMCMono<Shape>::getShape(std::string typ)
    {
    unsigned int id = this->m_pdata->getTypeByName(typ);
    return m_params[id].asDict();
    }

template<class Shape>
void IntegratorHPMCMono<Shape>::setParam(unsigned int typ, const param_type& param)
    {
    // validate input
    if (typ >= this->m_pdata->getNTypes())
        {
        throw std::runtime_error("Invalid particle type.");
        }

        // need to scope this because updateCellWidth will access it
        {
        // update the parameter for this type
        m_exec_conf->msg->notice(7) << "setParam : " << typ << std::endl;
        m_params[typ] = param;
        }

    updateCellWidth();
    }

template<class Shape>
void IntegratorHPMCMono<Shape>::setInteractionMatrix(std::pair<std::string, std::string> types,
                                                     bool check_overlaps)
    {
    auto typi = m_pdata->getTypeByName(types.first);
    auto typj = m_pdata->getTypeByName(types.second);

    // update the parameter for this type
    ArrayHandle<unsigned int> h_overlaps(m_overlaps, access_location::host, access_mode::readwrite);
    h_overlaps.data[m_overlap_idx(typi, typj)] = check_overlaps;
    h_overlaps.data[m_overlap_idx(typj, typi)] = check_overlaps;

    m_image_list_valid = false;
    }

template<class Shape>
bool IntegratorHPMCMono<Shape>::getInteractionMatrixPy(std::pair<std::string, std::string> types)
    {
    auto typi = m_pdata->getTypeByName(types.first);
    auto typj = m_pdata->getTypeByName(types.second);

    // update the parameter for this type
    ArrayHandle<unsigned int> h_overlaps(m_overlaps, access_location::host, access_mode::read);
    return h_overlaps.data[m_overlap_idx(typi, typj)];
    }

//! Calculate a list of box images within interaction range of the simulation box, innermost first
template<class Shape>
inline const std::vector<vec3<Scalar>>& IntegratorHPMCMono<Shape>::updateImageList()
    {
    // cancel if the image list is up to date
    if (m_image_list_valid)
        return m_image_list;

    // triclinic boxes have 4 linearly independent body diagonals
    // box_circumsphere = max(body_diagonals)
    // range = getMaxCoreDiameter() + box_circumsphere
    // while still adding images, examine successively larger blocks of images, checking the
    // outermost against range

    unsigned int ndim = m_sysdef->getNDimensions();

    m_image_list_valid = true;
    m_image_list_is_initialized = true;
    m_image_list.clear();
    m_image_hkl.clear();
    m_image_list_rebuilds++;

    // Get box vectors
    const BoxDim box = m_pdata->getGlobalBox();
    vec3<Scalar> e1 = vec3<Scalar>(box.getLatticeVector(0));
    vec3<Scalar> e2 = vec3<Scalar>(box.getLatticeVector(1));
    // 2D simulations don't necessarily have a zero-size z-dimension, but it is convenient for us if
    // we assume one.
    vec3<Scalar> e3(0, 0, 0);
    if (ndim == 3)
        e3 = vec3<Scalar>(box.getLatticeVector(2));

    // The maximum interaction range is the sum of the max particle circumsphere diameter
    // (accounting for non-additive interactions) and the patch interaction.
    Scalar max_trans_d_and_diam(0.0);
        {
        // access the type parameters
        ArrayHandle<Scalar> h_d(m_d, access_location::host, access_mode::read);

        // access interaction matrix
        ArrayHandle<unsigned int> h_overlaps(this->m_overlaps,
                                             access_location::host,
                                             access_mode::read);

        // for each type, create a temporary shape and return the maximum sum of diameter and move
        // size
        for (unsigned int typ_i = 0; typ_i < this->m_pdata->getNTypes(); typ_i++)
            {
            Shape temp_i(quat<Scalar>(), m_params[typ_i]);

            Scalar r_cut_patch_i(0.0);
            if (hasPairInteractions())
                {
                r_cut_patch_i
                    = getMaxPairEnergyRCutNonAdditive()
                      + static_cast<ShortReal>(0.5) * getMaxPairInteractionAdditiveRCut(typ_i);
                }

            Scalar range_i(0.0);
            for (unsigned int typ_j = 0; typ_j < this->m_pdata->getNTypes(); typ_j++)
                {
                Scalar r_cut_patch_ij(0.0);
                if (hasPairInteractions())
                    {
                    r_cut_patch_ij
                        = r_cut_patch_i
                          + static_cast<ShortReal>(0.5) * getMaxPairInteractionAdditiveRCut(typ_j);
                    }

                Shape temp_j(quat<Scalar>(), m_params[typ_j]);
                Scalar r_cut_shape(0.0);
                if (h_overlaps.data[m_overlap_idx(typ_i, typ_j)])
                    r_cut_shape
                        = 0.5
                          * (temp_i.getCircumsphereDiameter() + temp_j.getCircumsphereDiameter());
                Scalar range_ij = detail::max(r_cut_shape, r_cut_patch_ij);
                range_i = detail::max(range_i, range_ij);
                }
            max_trans_d_and_diam
                = detail::max(max_trans_d_and_diam, range_i + Scalar(m_nselect) * h_d.data[typ_i]);
            }
        }

    m_exec_conf->msg->notice(6) << "Image list: max_trans_d_and_diam = " << max_trans_d_and_diam
                                << std::endl;

    Scalar range = max_trans_d_and_diam;

    m_exec_conf->msg->notice(6) << "Image list: range = " << range << std::endl;

    // initialize loop
    // start in the middle and add image boxes going out, one index at a time until no more
    // images are added to the list
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

#ifdef ENABLE_MPI
        if (m_pdata->getDomainDecomposition())
            {
            Index3D di = m_pdata->getDomainDecomposition()->getDomainIndexer();
            if (di.getW() > 1)
                x_max = 0;
            if (di.getH() > 1)
                y_max = 0;
            if (di.getD() > 1)
                z_max = 0;
            }
#endif

        // for every possible image, check to see if the primary image box swept out by the
        // interaction range overlaps with the current image box. If there are any overlaps, there
        // is the possibility of a particle pair in the primary image interacting with a particle in
        // the candidate image - add it to the image list.

        // construct the box shapes
        std::vector<vec3<ShortReal>> box_verts;
        if (ndim == 3)
            {
            box_verts.push_back((-e1 + -e2 + -e3) * 0.5);
            box_verts.push_back((-e1 + e2 + -e3) * 0.5);
            box_verts.push_back((e1 + e2 + -e3) * 0.5);
            box_verts.push_back((e1 + -e2 + -e3) * 0.5);
            box_verts.push_back((-e1 + -e2 + e3) * 0.5);
            box_verts.push_back((-e1 + e2 + e3) * 0.5);
            box_verts.push_back((e1 + e2 + e3) * 0.5);
            box_verts.push_back((e1 + -e2 + e3) * 0.5);
            }
        else
            {
            box_verts.push_back((-e1 + -e2) * 0.5);
            box_verts.push_back((-e1 + e2) * 0.5);
            box_verts.push_back((e1 + e2) * 0.5);
            box_verts.push_back((e1 + -e2) * 0.5);
            }

        detail::PolyhedronVertices central_box_params(box_verts, ShortReal(range), 0);
        ShapeSpheropolyhedron central_box(quat<Scalar>(), central_box_params);
        detail::PolyhedronVertices image_box_params(box_verts, 0, 0);
        ShapeSpheropolyhedron image_box(quat<Scalar>(), image_box_params);

        // for each potential image
        for (hkl.x = -x_max; hkl.x <= x_max; hkl.x++)
            {
            for (hkl.y = -y_max; hkl.y <= y_max; hkl.y++)
                {
                for (hkl.z = -z_max; hkl.z <= z_max; hkl.z++)
                    {
                    // only add images on the outer boundary
                    if (abs(hkl.x) == hkl_max || abs(hkl.y) == hkl_max || abs(hkl.z) == hkl_max)
                        {
                        // find the center of the image
                        vec3<Scalar> r_image
                            = Scalar(hkl.x) * e1 + Scalar(hkl.y) * e2 + Scalar(hkl.z) * e3;

                        // check to see if the image box overlaps with the central box
                        unsigned int err = 0;
                        if (test_overlap(r_image, central_box, image_box, err))
                            {
                            vec3<Scalar> img
                                = (Scalar)hkl.x * e1 + (Scalar)hkl.y * e2 + (Scalar)hkl.z * e3;
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
            m_exec_conf->msg->warning() << "Exceeded sanity limit for image list, generated out to "
                                        << hkl_max << " lattice vectors. Logic error?" << std::endl
                                        << "This message will not be repeated." << std::endl;

            break;
            }

        hkl_max++;
        }

    m_exec_conf->msg->notice(6) << "Image list:" << std::endl;
    for (unsigned int i = 0; i < m_image_list.size(); i++)
        m_exec_conf->msg->notice(6) << m_image_list[i].x << " " << m_image_list[i].y << " "
                                    << m_image_list[i].z << std::endl;

    // warn the user if more than one image in each direction is activated
    unsigned int img_warning = 9;
    if (ndim == 3)
        {
        img_warning = 27;
        }
    if (!m_image_list_warning_issued && m_image_list.size() > img_warning)
        {
        m_image_list_warning_issued = true;
        m_exec_conf->msg->warning()
            << "Box size is too small or move size is too large for the minimum image convention."
            << std::endl
            << "Testing " << m_image_list.size() << " images per trial move, performance may slow."
            << std::endl
            << "This message will not be repeated." << std::endl;
        }

    m_exec_conf->msg->notice(6) << "Updated image list: " << m_image_list.size() << " images"
                                << std::endl;
    return m_image_list;
    }

template<class Shape> void IntegratorHPMCMono<Shape>::updateCellWidth()
    {
    this->m_nominal_width = this->getMaxCoreDiameter();

    // Account for patch width
    if (hasPairInteractions())
        {
        Scalar max_extent = 0.0;
        for (unsigned int typ = 0; typ < this->m_pdata->getNTypes(); typ++)
            {
            max_extent
                = std::max(max_extent, static_cast<Scalar>(getMaxPairInteractionAdditiveRCut(typ)));
            }

        this->m_nominal_width
            = std::max(this->m_nominal_width, getMaxPairEnergyRCutNonAdditive() + max_extent);
        }
    this->m_image_list_valid = false;
    this->m_aabb_tree_invalid = true;

    this->m_exec_conf->msg->notice(5)
        << "IntegratorHPMCMono: updating nominal width to " << this->m_nominal_width << std::endl;
    }

template<class Shape> void IntegratorHPMCMono<Shape>::growAABBList(unsigned int N)
    {
    if (N > m_aabbs_capacity)
        {
        m_aabbs_capacity = N;
        if (m_aabbs != NULL)
            free(m_aabbs);

        int retval = posix_memalign((void**)&m_aabbs, 32, N * sizeof(hoomd::detail::AABB));
        if (retval != 0)
            {
            throw std::runtime_error("Error allocating aligned memory.");
            }
        }
    }

/*! Call any time an up to date AABB tree is needed. IntegratorHPMCMono internally tracks whether
    the tree needs to be rebuilt or if the current tree can be used.

    buildAABBTree() relies on the member variable m_aabb_tree_invalid to work correctly. Any time
   particles are moved (and not updated with m_aabb_tree->update()) or the particle list changes
   order, m_aabb_tree_invalid needs to be set to true. Then buildAABBTree() will know to rebuild the
   tree from scratch on the next call. Typically this is on the next timestep. But in some cases
   (i.e. NPT), the tree may need to be rebuilt several times in a single step because of box volume
   moves.

    Subclasses that override update() or other methods must be user to set m_aabb_tree_invalid
   appropriately, or erroneous simulations will result.

    \returns A reference to the tree.
*/
template<class Shape> const hoomd::detail::AABBTree& IntegratorHPMCMono<Shape>::buildAABBTree()
    {
    if (m_aabb_tree_invalid)
        {
        m_exec_conf->msg->notice(8) << "Building AABB tree: " << m_pdata->getN() << " ptls "
                                    << m_pdata->getNGhosts() << " ghosts" << std::endl;
            // build the AABB tree
            {
            ArrayHandle<Scalar4> h_postype(m_pdata->getPositions(),
                                           access_location::host,
                                           access_mode::read);
            ArrayHandle<Scalar4> h_orientation(m_pdata->getOrientationArray(),
                                               access_location::host,
                                               access_mode::read);

            // precompute constants used many times in the loop
            m_max_pair_additive_cutoff.clear();
            m_shape_circumsphere_radius.clear();
            for (unsigned int type = 0; type < m_pdata->getNTypes(); type++)
                {
                quat<LongReal> q;
                Shape shape(q, m_params[type]);
                m_shape_circumsphere_radius.push_back(LongReal(0.5)
                                                      * shape.getCircumsphereDiameter());
                m_max_pair_additive_cutoff.push_back(getMaxPairInteractionAdditiveRCut(type));
                }

            // grow the AABB list to the needed size
            unsigned int n_aabb = m_pdata->getN() + m_pdata->getNGhosts();
            if (n_aabb > 0)
                {
                growAABBList(n_aabb);
                for (unsigned int cur_particle = 0; cur_particle < n_aabb; cur_particle++)
                    {
                    unsigned int i = cur_particle;
                    unsigned int typ_i = __scalar_as_int(h_postype.data[i].w);
                    Shape shape(quat<Scalar>(h_orientation.data[i]), m_params[typ_i]);

                    if (!hasPairInteractions())
                        m_aabbs[i] = shape.getAABB(vec3<Scalar>(h_postype.data[i]));
                    else
                        {
                        Scalar radius = std::max(m_shape_circumsphere_radius[typ_i],
                                                 LongReal(0.5) * m_max_pair_additive_cutoff[typ_i]);
                        m_aabbs[i] = hoomd::detail::AABB(vec3<Scalar>(h_postype.data[i]), radius);
                        }
                    }
                m_aabb_tree.buildTree(m_aabbs, n_aabb);
                }
            }
        }

    m_aabb_tree_invalid = false;
    return m_aabb_tree;
    }

/*! Call to reduce the m_d values down to safe levels for the bvh tree + small box limitations. That
   code path will not work if particles can wander more than one image in a time step.

    In MPI simulations, they may not move more than half a local box length.
*/
template<class Shape> void IntegratorHPMCMono<Shape>::limitMoveDistances()
    {
    Scalar3 npd_global = m_pdata->getGlobalBox().getNearestPlaneDistance();
    Scalar min_npd = detail::min(npd_global.x, npd_global.y);
    if (this->m_sysdef->getNDimensions() == 3)
        {
        min_npd = detail::min(min_npd, npd_global.z);
        }

    ArrayHandle<Scalar> h_d(m_d, access_location::host, access_mode::readwrite);
    for (unsigned int typ = 0; typ < this->m_pdata->getNTypes(); typ++)
        {
        if (m_nselect * h_d.data[typ] > min_npd)
            {
            h_d.data[typ] = min_npd / Scalar(m_nselect);
            m_exec_conf->msg->warning()
                << "Move distance or nselect too big, reducing move distance to " << h_d.data[typ]
                << " for type " << m_pdata->getNameByType(typ) << std::endl;
            m_image_list_valid = false;
            }
        // Sanity check should be performed in code where parameters can be adjusted.
        if (h_d.data[typ] < Scalar(0.0))
            {
            m_exec_conf->msg->warning()
                << "Move distance has become negative for type " << m_pdata->getNameByType(typ)
                << ". This should never happen. Please file a bug report." << std::endl;
            h_d.data[typ] = Scalar(0.0);
            }
        }
    }

/*! Function for finding all overlaps in a system by particle tag. returns an unraveled form of an
 * NxN matrix with true/false indicating the overlap status of the ith and jth particle
 */
template<class Shape>
std::vector<std::pair<unsigned int, unsigned int>> IntegratorHPMCMono<Shape>::mapOverlaps()
    {
#ifdef ENABLE_MPI
    if (m_pdata->getDomainDecomposition())
        {
        throw std::runtime_error("map_overlaps does not support MPI parallel jobs");
        }
#endif

    unsigned int N = m_pdata->getN();

    std::vector<std::pair<unsigned int, unsigned int>> overlap_vector;

    m_exec_conf->msg->notice(10) << "HPMC overlap mapping" << std::endl;

    unsigned int err_count = 0;

    // build an up to date AABB tree
    buildAABBTree();
    // update the image list
    updateImageList();

    // access particle data and system box
    ArrayHandle<Scalar4> h_postype(m_pdata->getPositions(),
                                   access_location::host,
                                   access_mode::read);
    ArrayHandle<Scalar4> h_orientation(m_pdata->getOrientationArray(),
                                       access_location::host,
                                       access_mode::read);
    ArrayHandle<unsigned int> h_tag(m_pdata->getTags(), access_location::host, access_mode::read);

    // Loop over all particles
    for (unsigned int i = 0; i < N; i++)
        {
        // read in the current position and orientation
        Scalar4 postype_i = h_postype.data[i];
        Scalar4 orientation_i = h_orientation.data[i];
        Shape shape_i(quat<Scalar>(orientation_i), m_params[__scalar_as_int(postype_i.w)]);
        vec3<Scalar> pos_i = vec3<Scalar>(postype_i);

        // Check particle against AABB tree for neighbors
        hoomd::detail::AABB aabb_i_local = shape_i.getAABB(vec3<Scalar>(0, 0, 0));

        const unsigned int n_images = (unsigned int)m_image_list.size();
        for (unsigned int cur_image = 0; cur_image < n_images; cur_image++)
            {
            vec3<Scalar> pos_i_image = pos_i + m_image_list[cur_image];
            hoomd::detail::AABB aabb = aabb_i_local;
            aabb.translate(pos_i_image);

            // stackless search
            for (unsigned int cur_node_idx = 0; cur_node_idx < m_aabb_tree.getNumNodes();
                 cur_node_idx++)
                {
                if (aabb.overlaps(m_aabb_tree.getNodeAABB(cur_node_idx)))
                    {
                    if (m_aabb_tree.isNodeLeaf(cur_node_idx))
                        {
                        for (unsigned int cur_p = 0;
                             cur_p < m_aabb_tree.getNodeNumParticles(cur_node_idx);
                             cur_p++)
                            {
                            // read in its position and orientation
                            unsigned int j = m_aabb_tree.getNodeParticle(cur_node_idx, cur_p);

                            // skip i==j in the 0 image
                            if (cur_image == 0 && i == j)
                                {
                                continue;
                                }

                            Scalar4 postype_j = h_postype.data[j];
                            Scalar4 orientation_j = h_orientation.data[j];

                            // put particles in coordinate system of particle i
                            vec3<Scalar> r_ij = vec3<Scalar>(postype_j) - pos_i_image;

                            Shape shape_j(quat<Scalar>(orientation_j),
                                          m_params[__scalar_as_int(postype_j.w)]);

                            if (h_tag.data[i] <= h_tag.data[j]
                                && check_circumsphere_overlap(r_ij, shape_i, shape_j)
                                && test_overlap(r_ij, shape_i, shape_j, err_count)
                                && test_overlap(-r_ij, shape_j, shape_i, err_count))
                                {
                                overlap_vector.push_back(
                                    std::make_pair(h_tag.data[i], h_tag.data[j]));
                                }
                            }
                        }
                    }
                else
                    {
                    // skip ahead
                    cur_node_idx += m_aabb_tree.getNodeSkip(cur_node_idx);
                    }
                } // end loop over AABB nodes
            } // end loop over images
        } // end loop over particles
    return overlap_vector;
    }

namespace detail
    {

//! Export the IntegratorHPMCMono class to python
/*! \param name Name of the class in the exported python module
    \tparam Shape An instantiation of IntegratorHPMCMono<Shape> will be exported
*/
template<class Shape> void export_IntegratorHPMCMono(pybind11::module& m, const std::string& name)
    {
    pybind11::class_<IntegratorHPMCMono<Shape>,
                     IntegratorHPMC,
                     std::shared_ptr<IntegratorHPMCMono<Shape>>>(m, name.c_str())
        .def(pybind11::init<std::shared_ptr<SystemDefinition>>())
        .def("setParam", &IntegratorHPMCMono<Shape>::setParam)
        .def("setInteractionMatrix", &IntegratorHPMCMono<Shape>::setInteractionMatrix)
        .def("getInteractionMatrix", &IntegratorHPMCMono<Shape>::getInteractionMatrixPy)
        .def("mapOverlaps", &IntegratorHPMCMono<Shape>::mapOverlaps)
        .def("getTypeShapesPy", &IntegratorHPMCMono<Shape>::getTypeShapesPy)
        .def("getShape", &IntegratorHPMCMono<Shape>::getShape)
        .def("setShape", &IntegratorHPMCMono<Shape>::setShape)
        .def("computePairEnergy", &IntegratorHPMCMono<Shape>::computePairEnergy);
    }

    } // end namespace detail
    } // end namespace hpmc
    } // end namespace hoomd
