// Copyright (c) 2009-2024 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

// inclusion guard
#ifndef _INTEGRATOR_HPMC_MONO_H_
#define _INTEGRATOR_HPMC_MONO_H_

/*! \file IntegratorHPMCMono.h
    \brief Declaration of IntegratorHPMC
*/

#include <iostream>
#include <iomanip>
#include <sstream>

#include "hoomd/Integrator.h"
#include "IntegratorHPMC.h"
#include "Moves.h"
#include "hoomd/AABBTree.h"
#include "GSDHPMCSchema.h"
#include "hoomd/Index1D.h"
#include "hoomd/RandomNumbers.h"
#include "hoomd/RNGIdentifiers.h"
#include "hoomd/managed_allocator.h"
#include "ShapeSpheropolyhedron.h"

#ifdef ENABLE_TBB
#include <thread>
#include <tbb/blocked_range.h>
#include <tbb/enumerable_thread_specific.h>
#include <tbb/parallel_for.h>
#include <tbb/parallel_reduce.h>
#endif

#ifdef ENABLE_MPI
#include "hoomd/Communicator.h"
#include "hoomd/HOOMDMPI.h"
#endif

#ifndef __HIPCC__
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#endif

namespace hoomd {

namespace hpmc
{

namespace detail
{

//! Helper class to manage shuffled update orders
/*! Stores an update order from 0 to N-1, inclusive, and can be resized. shuffle() shuffles the order of elements
    to a new random permutation. operator [i] gets the index of the item at order i in the current shuffled sequence.

    \ingroup hpmc_data_structs
*/
class UpdateOrder
    {
    public:
        //! Constructor
        /*! \param seed Random number seed
            \param N number of integers to shuffle
        */
        UpdateOrder(unsigned int N=0)
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
            \note \a timestep is used to seed the RNG, thus assuming that the order is shuffled only once per
            timestep.
        */
        void shuffle(uint64_t timestep, uint16_t seed, unsigned int rank = 0, unsigned int select = 0)
            {
            hoomd::RandomGenerator rng(hoomd::Seed(hoomd::RNGIdentifier::HPMCMonoShuffle, timestep, seed),
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

    TODO: I need better documentation

    \ingroup hpmc_integrators
*/
template < class Shape >
class IntegratorHPMCMono : public IntegratorHPMC
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
            m_pdata->getBoxChangeSignal().template disconnect<IntegratorHPMCMono<Shape>, &IntegratorHPMCMono<Shape>::slotBoxChanged>(this);
            m_pdata->getParticleSortSignal().template disconnect<IntegratorHPMCMono<Shape>, &IntegratorHPMCMono<Shape>::slotSorted>(this);
            }

        virtual void resetStats();

        //! Take one timestep forward
        virtual void update(uint64_t timestep);

        /*
         * Depletant related options
         */

        //! Set number of reinsertion attempts
        void setNtrialPy(std::string type, unsigned int ntrial)
            {
            unsigned int ptype = this->m_pdata->getTypeByName(type);
            m_ntrial[ptype] = ntrial;
            }

        //! Set the depletant density in the free volume
        void setDepletantFugacityPy(std::string type, Scalar fugacity)
            {
            unsigned int ptype = this->m_pdata->getTypeByName(type);
            m_fugacity[ptype] = fugacity;
            }

        //! Returns the depletant fugacity
        Scalar getDepletantFugacityPy(std::string type) const
            {
            unsigned int ptype = this->m_pdata->getTypeByName(type);
            return m_fugacity[ptype];
            }

        unsigned int getNtrialPy(std::string type) const
            {
            unsigned int ptype = this->m_pdata->getTypeByName(type);
            return m_ntrial[ptype];
            }

        unsigned int getNtrial(unsigned int type) const
            {
            return m_ntrial[type];
            }

        //! Set the depletant density in the free volume
        /*! \param type type of depletant
            \param the fugacity
         */
        void setDepletantFugacity(unsigned int type, Scalar fugacity)
            {
            if (type >= this->m_pdata->getNTypes())
                throw std::runtime_error("Unknown type.");
            m_fugacity[type] = fugacity;
            }

        //! Returns the depletant fugacity
        Scalar getDepletantFugacity(unsigned int type)
            {
            return m_fugacity[type];
            }

        //! Returns a GlobalVector of the depletant fugacities
        const GlobalVector<Scalar>& getFugacityArray()
            {
            return m_fugacity;
            }

        //! Returns a GlobalVector of the depletant ntrial values
        const GlobalVector<unsigned int>& getNtrialArray()
            {
            return m_ntrial;
            }

        //! Get the current counter values
        virtual std::vector<hpmc_implicit_counters_t> getImplicitCounters(unsigned int mode=0);

        //! Method to scale the box
        virtual bool attemptBoxResize(uint64_t timestep, const BoxDim& new_box);

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

        //! Set the external field for the integrator
        void setExternalField(std::shared_ptr< ExternalFieldMono<Shape> > external)
            {
            m_external = external;
            this->m_external_base = (ExternalField*)external.get();
            }

        //! Get the particle parameters
        virtual std::vector<param_type, hoomd::detail::managed_allocator<param_type> >& getParams()
            {
            return m_params;
            }

        //! Get the interaction matrix
        virtual const GlobalArray<unsigned int>& getInteractionMatrix()
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
        virtual std::vector<std::pair<unsigned int, unsigned int> > mapOverlaps();

        //! Return the requested ghost layer width
        virtual Scalar getGhostLayerWidth(unsigned int type)
            {
            Scalar ghost_width = m_nominal_width + m_extra_ghost_width;
            m_exec_conf->msg->notice(9) << "IntegratorHPMCMono: ghost layer width of " << ghost_width << std::endl;
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

            // many things depend internally on the orientation field (for ghosts) being initialized, therefore always request it
            flags[comm_flag::orientation] = 1;

            if (hasPairInteractions())
                {
                flags[comm_flag::diameter] = 1;
                flags[comm_flag::charge] = 1;
                o << " diameter charge";
                }

            bool have_auxilliary_variables = false;
            for (unsigned int i = 0; i < this->m_pdata->getNTypes(); ++i)
                {
                if (m_fugacity[i] != 0.0 && m_ntrial[i] > 0)
                    {
                    have_auxilliary_variables = true;
                    break;
                    }
                }

            if (have_auxilliary_variables)
                {
                flags[comm_flag::velocity] = 1;
                o << " velocity";
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
            quat<Scalar> q(make_scalar4(1,0,0,0));
            for (unsigned int i=0; i < m_pdata->getNTypes(); i++)
                {
                Shape dummy(q, m_params[i]);
                if (dummy.hasOrientation())
                    {
                    m_hasOrientation = true;
                    break;
                    }
                }

            updateCellWidth(); // make sure the cell width is up-to-date and forces a rebuild of the AABB tree and image list

            communicate(true);
            }

        //! Communicate particles
        virtual void communicate(bool migrate)
            {
            // migrate and exchange particles
            #ifdef ENABLE_MPI
            if (m_sysdef->isDomainDecomposed())
                {
                // this is kludgy but necessary since we are calling the communications methods directly
                m_comm->setFlags(getCommFlags(0));

                if (migrate)
                    m_comm->migrateParticles();
                else
                    m_pdata->removeAllGhostParticles();

                m_comm->exchangeGhosts();

                m_aabb_tree_invalid = true;
                }
            #endif
            }

        //! Return true if anisotropic particles are present
        virtual bool hasOrientation() { return m_hasOrientation; }

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
        double computePairEnergy(uint64_t timestep, std::shared_ptr<PairPotential> selected_pair = nullptr);

        //! Build the AABB tree (if needed)
        const hoomd::detail::AABBTree& buildAABBTree();

        //! Make list of image indices for boxes to check in small-box mode
        const std::vector<vec3<Scalar> >& updateImageList();

        //! Return list of integer shift vectors for periodic images
        const std::vector<int3>& getImageHKL()
            {
            updateImageList();
            return m_image_hkl;
            }

        void invalidateAABBTree(){ m_aabb_tree_invalid = true; }

        std::vector<std::string> getTypeShapeMapping(const std::vector<param_type, hoomd::detail::managed_allocator<param_type> > &params) const
            {
            quat<Scalar> q(make_scalar4(1,0,0,0));
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
        std::vector<param_type, hoomd::detail::managed_allocator<param_type> > m_params;   //!< Parameters for each particle type on GPU
        GlobalArray<unsigned int> m_overlaps;          //!< Interaction matrix (0/1) for overlap checks
        detail::UpdateOrder m_update_order;         //!< Update order
        bool m_image_list_is_initialized;                    //!< true if image list has been used
        bool m_image_list_valid;                             //!< image list is invalid if the box dimensions or particle parameters have changed.
        std::vector<vec3<Scalar> > m_image_list;             //!< List of potentially interacting simulation box images
        std::vector<int3> m_image_hkl;               //!< List of potentially interacting simulation box images (integer shifts)
        unsigned int m_image_list_rebuilds;                  //!< Number of times the image list has been rebuilt
        bool m_image_list_warning_issued;                    //!< True if the image list warning has been issued
        bool m_hkl_max_warning_issued;                       //!< True if the image list size warning has been issued
        bool m_hasOrientation;                               //!< true if there are any orientable particles in the system

        std::shared_ptr< ExternalFieldMono<Shape> > m_external;//!< External Field
        hoomd::detail::AABBTree m_aabb_tree;               //!< Bounding volume hierarchy for overlap checks
        hoomd::detail::AABB* m_aabbs;                      //!< list of AABBs, one per particle
        unsigned int m_aabbs_capacity;              //!< Capacity of m_aabbs list
        bool m_aabb_tree_invalid;                   //!< Flag if the aabb tree has been invalidated

        Scalar m_extra_image_width;                 //! Extra width to extend the image list

        Index2D m_overlap_idx;                      //!!< Indexer for interaction matrix

        /// Cached maximum pair additive cutoff by type.
        std::vector<LongReal> m_max_pair_additive_cutoff;

        /// Cached shape radius by type.
        std::vector<LongReal> m_shape_circumsphere_radius;

        /* Depletants related data members */

        GlobalVector<Scalar> m_fugacity;            //!< Average depletant number density in free volume, per type
        GlobalVector<unsigned int> m_ntrial;        //!< Number of reinsertion attempts per depletant in overlap volume, per type

        GlobalArray<hpmc_implicit_counters_t> m_implicit_count;               //!< Counter of depletant insertions
        std::vector<hpmc_implicit_counters_t> m_implicit_count_run_start;     //!< Counter of depletant insertions at run start
        std::vector<hpmc_implicit_counters_t> m_implicit_count_step_start;    //!< Counter of depletant insertions at step start

        //! Test whether to reject the current particle move based on depletants
        inline bool checkDepletantOverlap(unsigned int i, vec3<Scalar> pos_i, Shape shape_i, unsigned int typ_i,
            Scalar4 *h_postype, Scalar4 *h_orientation, const unsigned int *h_tag, const Scalar4 *h_vel,
            unsigned int *h_overlaps, hpmc_counters_t& counters, hpmc_implicit_counters_t *implicit_counters,
            uint64_t timestep, hoomd::RandomGenerator& rng_depletants,
            unsigned int seed_i_old, unsigned int seed_i_new);

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

template <class Shape>
IntegratorHPMCMono<Shape>::IntegratorHPMCMono(std::shared_ptr<SystemDefinition> sysdef)
            : IntegratorHPMC(sysdef),
              m_update_order(m_pdata->getN()),
              m_image_list_is_initialized(false),
              m_image_list_valid(false),
              m_hasOrientation(true),
              m_extra_image_width(0.0),
              m_fugacity(m_exec_conf),
              m_ntrial(m_exec_conf)
    {
    // allocate the parameter storage, setting the managed flag
    m_params = std::vector<param_type, hoomd::detail::managed_allocator<param_type> >(m_pdata->getNTypes(),
                                                                       param_type(),
                                                                       hoomd::detail::managed_allocator<param_type>(m_exec_conf->isCUDAEnabled()));

    m_overlap_idx = Index2D(m_pdata->getNTypes());
    GlobalArray<unsigned int> overlaps(m_overlap_idx.getNumElements(), m_exec_conf);
    m_overlaps.swap(overlaps);
    TAG_ALLOCATION(m_overlaps);
    ArrayHandle<unsigned int> h_overlaps(m_overlaps, access_location::host, access_mode::readwrite);
    for(unsigned int i = 0; i < m_overlap_idx.getNumElements(); i++)
        {
        h_overlaps.data[i] = 1; // Assume we want to check overlaps.
        }

    // Connect to the BoxChange signal
    m_pdata->getBoxChangeSignal().template connect<IntegratorHPMCMono<Shape>, &IntegratorHPMCMono<Shape>::slotBoxChanged>(this);
    m_pdata->getParticleSortSignal().template connect<IntegratorHPMCMono<Shape>, &IntegratorHPMCMono<Shape>::slotSorted>(this);

    m_image_list_rebuilds = 0;
    m_image_list_warning_issued = false;
    m_hkl_max_warning_issued = false;

    m_aabbs = NULL;
    m_aabbs_capacity = 0;
    m_aabb_tree_invalid = true;

    m_fugacity.resize(this->m_pdata->getNTypes(), 0.0);
    m_ntrial.resize(m_fugacity.getNumElements(), 1);
    TAG_ALLOCATION(m_fugacity);
    TAG_ALLOCATION(m_ntrial);

    GlobalArray<hpmc_implicit_counters_t> implicit_count(m_fugacity.getNumElements(),this->m_exec_conf);
    m_implicit_count.swap(implicit_count);
    TAG_ALLOCATION(m_implicit_count);

    m_implicit_count_run_start.resize(m_fugacity.getNumElements());
    m_implicit_count_step_start.resize(m_fugacity.getNumElements());
    }

/*! \param mode 0 -> Absolute count, 1 -> relative to the start of the run, 2 -> relative to the last executed step
    \return The current state of the acceptance counters

    IntegratorHPMCMonoImplicit maintains a count of the number of accepted and rejected moves since instantiation. getCounters()
    provides the current value. The parameter *mode* controls whether the returned counts are absolute, relative
    to the start of the run, or relative to the start of the last executed step.
*/
template<class Shape>
std::vector<hpmc_implicit_counters_t> IntegratorHPMCMono<Shape>::getImplicitCounters(unsigned int mode)
    {
    ArrayHandle<hpmc_implicit_counters_t> h_counters(m_implicit_count, access_location::host, access_mode::read);
    std::vector<hpmc_implicit_counters_t> result(m_fugacity.getNumElements());

    std::copy(h_counters.data, h_counters.data + m_fugacity.getNumElements(), result.begin());

    if (mode == 1)
        {
        for (unsigned int i = 0; i < m_fugacity.getNumElements(); ++i)
            result[i] = result[i] - m_implicit_count_run_start[i];
        }
    else if (mode == 2)
        {
        for (unsigned int i = 0; i < m_fugacity.getNumElements(); ++i)
            result[i] = result[i] - m_implicit_count_step_start[i];
        }

    #ifdef ENABLE_MPI
    if (this->m_sysdef->isDomainDecomposed())
        {
        // MPI Reduction to total result values on all ranks
        for (unsigned int i = 0; i < m_fugacity.getNumElements(); ++i)
            {
            MPI_Allreduce(MPI_IN_PLACE, &result[i].insert_count, 1, MPI_LONG_LONG_INT, MPI_SUM, this->m_exec_conf->getMPICommunicator());
            MPI_Allreduce(MPI_IN_PLACE, &result[i].insert_accept_count, 1, MPI_LONG_LONG_INT, MPI_SUM, this->m_exec_conf->getMPICommunicator());
            MPI_Allreduce(MPI_IN_PLACE, &result[i].insert_accept_count_sq, 1, MPI_LONG_LONG_INT, MPI_SUM, this->m_exec_conf->getMPICommunicator());
            }
        }
    #endif

    return result;
    }

template <class Shape>
void IntegratorHPMCMono<Shape>::resetStats()
    {
    IntegratorHPMC::resetStats();

    ArrayHandle<hpmc_implicit_counters_t> h_counters(m_implicit_count, access_location::host, access_mode::read);
    for (unsigned int i = 0; i < m_fugacity.getNumElements(); ++i)
        m_implicit_count_run_start[i] = h_counters.data[i];
    }

template <class Shape>
void IntegratorHPMCMono<Shape>::update(uint64_t timestep)
    {
    Integrator::update(timestep);
    m_exec_conf->msg->notice(10) << "HPMCMono update: " << timestep << std::endl;
    IntegratorHPMC::update(timestep);

    // get needed vars
    ArrayHandle<hpmc_counters_t> h_counters(m_count_total, access_location::host, access_mode::readwrite);
    hpmc_counters_t& counters = h_counters.data[0];

    ArrayHandle<hpmc_implicit_counters_t> h_implicit_counters(m_implicit_count, access_location::host, access_mode::readwrite);
    std::copy(h_implicit_counters.data, h_implicit_counters.data + m_fugacity.getNumElements(), m_implicit_count_step_start.begin());

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

    // update the AABB Tree
    buildAABBTree();
    // limit m_d entries so that particles cannot possibly wander more than one box image in one time step
    limitMoveDistances();
    // update the image list
    updateImageList();

    bool has_depletants = false;
    for (unsigned int i = 0; i < m_fugacity.getNumElements(); ++i)
        {
        if (m_fugacity[i] != 0.0)
            {
            has_depletants = true;
            break;
            }
        }

    // Combine the three seeds to generate RNG for poisson distribution
    hoomd::RandomGenerator rng_depletants(hoomd::Seed(hoomd::RNGIdentifier::HPMCDepletants,
                                                      timestep,
                                                      this->m_sysdef->getSeed()),
                                          hoomd::Counter(this->m_exec_conf->getRank()));

    uint16_t seed = m_sysdef->getSeed();

    // access interaction matrix
    ArrayHandle<unsigned int> h_overlaps(m_overlaps, access_location::host, access_mode::read);

    // precompute constants used many times in the loop
    const LongReal min_core_radius = getMinCoreDiameter() * LongReal(0.5);
    const auto& pair_energy_search_radius = getPairEnergySearchRadius();

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
        // access particle data and system box
        ArrayHandle<Scalar4> h_postype(m_pdata->getPositions(), access_location::host, access_mode::readwrite);
        ArrayHandle<Scalar4> h_orientation(m_pdata->getOrientationArray(), access_location::host, access_mode::readwrite);
        ArrayHandle<Scalar> h_diameter(m_pdata->getDiameters(), access_location::host, access_mode::read);
        ArrayHandle<Scalar> h_charge(m_pdata->getCharges(), access_location::host, access_mode::read);
        ArrayHandle<unsigned int> h_tag(m_pdata->getTags(), access_location::host, access_mode::read);
        ArrayHandle<Scalar4> h_vel(m_pdata->getVelocities(), access_location::host, access_mode::readwrite);

        //access move sizes
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
                if (!isActive(make_scalar3(postype_i.x, postype_i.y, postype_i.z), box, ghost_fraction))
                    continue;
                }
            #endif

            // make a trial move for i
            hoomd::RandomGenerator rng_i(hoomd::Seed(hoomd::RNGIdentifier::HPMCMonoTrialMove, timestep, seed),
                                         hoomd::Counter(i, m_exec_conf->getRank(), i_nselect));
            int typ_i = __scalar_as_int(postype_i.w);
            Shape shape_i(quat<LongReal>(h_orientation.data[i]), m_params[typ_i]);
            unsigned int move_type_select = hoomd::UniformIntDistribution(0xffff)(rng_i);
            bool move_type_translate = !shape_i.hasOrientation() || (move_type_select < m_translation_move_probability);

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


            bool overlap=false;

            // search for all particles that might touch this one
            LongReal R_query = m_shape_circumsphere_radius[typ_i];

            if (hasPairInteractions())
                {
                // Extend the search to include the pair interaction r_cut
                // subtract minimum AABB extent from search radius
                R_query = std::max(R_query, pair_energy_search_radius[typ_i] - min_core_radius);
                }

            hoomd::detail::AABB aabb_i_local = hoomd::detail::AABB(vec3<Scalar>(0,0,0),R_query);

            // patch + field interaction deltaU
            double patch_field_energy_diff = 0;

            // check for overlaps with neighboring particle's positions (also calculate the new energy)
            // All image boxes (including the primary)
            const unsigned int n_images = (unsigned int)m_image_list.size();
            for (unsigned int cur_image = 0; cur_image < n_images; cur_image++)
                {
                vec3<Scalar> pos_i_image = pos_i + m_image_list[cur_image];
                hoomd::detail::AABB aabb = aabb_i_local;
                aabb.translate(pos_i_image);

                // stackless search
                for (unsigned int cur_node_idx = 0; cur_node_idx < m_aabb_tree.getNumNodes(); cur_node_idx++)
                    {
                    if (aabb.overlaps(m_aabb_tree.getNodeAABB(cur_node_idx)))
                        {
                        if (m_aabb_tree.isNodeLeaf(cur_node_idx))
                            {
                            for (unsigned int cur_p = 0; cur_p < m_aabb_tree.getNodeNumParticles(cur_node_idx); cur_p++)
                                {
                                // read in its position and orientation
                                unsigned int j = m_aabb_tree.getNodeParticle(cur_node_idx, cur_p);

                                Scalar4 postype_j;
                                quat<LongReal> orientation_j;

                                // handle j==i situations
                                if ( j != i )
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
                                        // If this is particle i and we are in an outside image, use the translated position and orientation
                                        postype_j = make_scalar4(pos_i.x, pos_i.y, pos_i.z, postype_i.w);
                                        orientation_j = shape_i.orientation;
                                        }
                                    }

                                // put particles in coordinate system of particle i
                                vec3<Scalar> r_ij = vec3<Scalar>(postype_j) - pos_i_image;

                                unsigned int typ_j = __scalar_as_int(postype_j.w);
                                Shape shape_j(orientation_j, m_params[typ_j]);

                                LongReal r_squared = dot(r_ij, r_ij);
                                LongReal max_overlap_distance = m_shape_circumsphere_radius[typ_i] + m_shape_circumsphere_radius[typ_j];

                                counters.overlap_checks++;
                                if (h_overlaps.data[m_overlap_idx(typ_i, typ_j)]
                                    && r_squared < max_overlap_distance * max_overlap_distance
                                    && test_overlap(r_ij, shape_i, shape_j, counters.overlap_err_count))
                                    {
                                    overlap = true;
                                    break;
                                    }

                                // deltaU = U_old - U_new: subtract energy of new configuration
                                patch_field_energy_diff -= computeOnePairEnergy(r_squared, r_ij, typ_i,
                                                        shape_i.orientation,
                                                        h_diameter.data[i],
                                                        h_charge.data[i],
                                                        typ_j,
                                                        shape_j.orientation,
                                                        h_diameter.data[j],
                                                        h_charge.data[j]
                                                        );
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
                    }  // end loop over AABB nodes

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
                    for (unsigned int cur_node_idx = 0; cur_node_idx < m_aabb_tree.getNumNodes(); cur_node_idx++)
                        {
                        if (aabb.overlaps(m_aabb_tree.getNodeAABB(cur_node_idx)))
                            {
                            if (m_aabb_tree.isNodeLeaf(cur_node_idx))
                                {
                                for (unsigned int cur_p = 0; cur_p < m_aabb_tree.getNodeNumParticles(cur_node_idx); cur_p++)
                                    {
                                    // read in its position and orientation
                                    unsigned int j = m_aabb_tree.getNodeParticle(cur_node_idx, cur_p);

                                    Scalar4 postype_j;
                                    quat<LongReal> orientation_j;

                                    // handle j==i situations
                                    if ( j != i )
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
                                            // If this is particle i and we are in an outside image, use the translated position and orientation
                                            postype_j = make_scalar4(pos_old.x, pos_old.y, pos_old.z, postype_i.w);
                                            orientation_j = shape_old.orientation;
                                            }
                                        }

                                    // put particles in coordinate system of particle i
                                    vec3<Scalar> r_ij = vec3<Scalar>(postype_j) - pos_i_image;
                                    unsigned int typ_j = __scalar_as_int(postype_j.w);
                                    Shape shape_j(orientation_j, m_params[typ_j]);

                                    // deltaU = U_old - U_new: add energy of old configuration
                                    patch_field_energy_diff += computeOnePairEnergy(dot(r_ij, r_ij),
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
                        }  // end loop over AABB nodes
                    } // end loop over images
                }

            // Add external energetic contribution if there are no overlaps
            if (!overlap)
                {
                // Legacy external field energy difference
                if (m_external)
                    {
                    patch_field_energy_diff -= m_external->energydiff(timestep, i, pos_old, shape_old, pos_i, shape_i);
                    }

                // U_old - U_new
                patch_field_energy_diff +=
                    this->computeOneExternalEnergy(typ_i, pos_old, shape_old.orientation, h_charge.data[i], false) -
                    this->computeOneExternalEnergy(typ_i, pos_i, shape_i.orientation, h_charge.data[i], true);
                }

            bool accept = !overlap && hoomd::detail::generate_canonical<double>(rng_i) < slow::exp(patch_field_energy_diff);

            // The trial move is valid, so check if it is invalidated by depletants
            unsigned int seed_i_new = hoomd::detail::generate_u32(rng_i);
            unsigned int seed_i_old = __scalar_as_int(h_vel.data[i].x);

            if (has_depletants && accept)
                {
                accept = checkDepletantOverlap(i, pos_i, shape_i, typ_i, h_postype.data,
                    h_orientation.data, h_tag.data, h_vel.data, h_overlaps.data, counters, h_implicit_counters.data,
                    timestep^i_nselect, rng_depletants, seed_i_old, seed_i_new);
                }

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
                h_postype.data[i] = make_scalar4(pos_i.x,pos_i.y,pos_i.z,postype_i.w);

                if (shape_i.hasOrientation())
                    {
                    h_orientation.data[i] = quat_to_scalar4(shape_i.orientation);
                    }

                // store new seed
                if (has_depletants)
                    h_vel.data[i].x = __int_as_scalar(seed_i_new);
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
        ArrayHandle<Scalar4> h_postype(m_pdata->getPositions(), access_location::host, access_mode::readwrite);
        ArrayHandle<int3> h_image(m_pdata->getImages(), access_location::host, access_mode::readwrite);
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
        ArrayHandle<Scalar4> h_postype(m_pdata->getPositions(), access_location::host, access_mode::readwrite);
        ArrayHandle<int3> h_image(m_pdata->getImages(), access_location::host, access_mode::readwrite);

        // precalculate the grid shift
        hoomd::RandomGenerator rng(hoomd::Seed(hoomd::RNGIdentifier::HPMCMonoShift, timestep, this->m_sysdef->getSeed()),
                                   hoomd::Counter());
        Scalar3 shift = make_scalar3(0,0,0);
        hoomd::UniformDistribution<Scalar> uniform(-m_nominal_width/Scalar(2.0),m_nominal_width/Scalar(2.0));
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
            vec3<Scalar> r_i = vec3<Scalar>(postype_i); // translation from local to global coordinates
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
template <class Shape>
unsigned int IntegratorHPMCMono<Shape>::countOverlaps(bool early_exit)
    {
    unsigned int overlap_count = 0;
    unsigned int err_count = 0;

    // build an up to date AABB tree
    buildAABBTree();
    // update the image list
    updateImageList();

    // access particle data and system box
    ArrayHandle<Scalar4> h_postype(m_pdata->getPositions(), access_location::host, access_mode::read);
    ArrayHandle<Scalar4> h_orientation(m_pdata->getOrientationArray(), access_location::host, access_mode::read);
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
        hoomd::detail::AABB aabb_i_local = shape_i.getAABB(vec3<Scalar>(0,0,0));

        const unsigned int n_images = (unsigned int)m_image_list.size();
        for (unsigned int cur_image = 0; cur_image < n_images; cur_image++)
            {
            vec3<Scalar> pos_i_image = pos_i + m_image_list[cur_image];
            hoomd::detail::AABB aabb = aabb_i_local;
            aabb.translate(pos_i_image);

            // stackless search
            for (unsigned int cur_node_idx = 0; cur_node_idx < m_aabb_tree.getNumNodes(); cur_node_idx++)
                {
                if (aabb.overlaps(m_aabb_tree.getNodeAABB(cur_node_idx)))
                    {
                    if (m_aabb_tree.isNodeLeaf(cur_node_idx))
                        {
                        for (unsigned int cur_p = 0; cur_p < m_aabb_tree.getNodeNumParticles(cur_node_idx); cur_p++)
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
                                && h_overlaps.data[m_overlap_idx(typ_i,typ_j)]
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
        MPI_Allreduce(MPI_IN_PLACE, &overlap_count, 1, MPI_UNSIGNED, MPI_SUM, m_exec_conf->getMPICommunicator());
        if (early_exit && overlap_count > 1)
            overlap_count = 1;
        }
    #endif

    return overlap_count;
    }

template<class Shape>
double IntegratorHPMCMono<Shape>::computeTotalPairEnergy(uint64_t timestep)
    {
    return computePairEnergy(timestep);
    }

template <class Shape>
double IntegratorHPMCMono<Shape>::computePairEnergy(uint64_t timestep, std::shared_ptr<PairPotential> selected_pair)
    {
    // sum up in double precision
    double energy = 0.0;

    // return if nothing to do
    if (!hasPairInteractions()) return 0;

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
    ArrayHandle<Scalar4> h_postype(m_pdata->getPositions(), access_location::host, access_mode::read);
    ArrayHandle<Scalar4> h_orientation(m_pdata->getOrientationArray(), access_location::host, access_mode::read);
    ArrayHandle<Scalar> h_diameter(m_pdata->getDiameters(), access_location::host, access_mode::read);
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
        hoomd::detail::AABB aabb_i_local = hoomd::detail::AABB(vec3<Scalar>(0,0,0),R_query);

        const unsigned int n_images = (unsigned int)m_image_list.size();
        for (unsigned int cur_image = 0; cur_image < n_images; cur_image++)
            {
            vec3<Scalar> pos_i_image = pos_i + m_image_list[cur_image];
            hoomd::detail::AABB aabb = aabb_i_local;
            aabb.translate(pos_i_image);

            // stackless search
            for (unsigned int cur_node_idx = 0; cur_node_idx < m_aabb_tree.getNumNodes(); cur_node_idx++)
                {
                if (aabb.overlaps(m_aabb_tree.getNodeAABB(cur_node_idx)))
                    {
                    if (m_aabb_tree.isNodeLeaf(cur_node_idx))
                        {
                        for (unsigned int cur_p = 0; cur_p < m_aabb_tree.getNodeNumParticles(cur_node_idx); cur_p++)
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
                                if (selected_pair && r_squared < selected_pair->getRCutSquaredTotal(typ_i, typ_j))
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
        MPI_Allreduce(MPI_IN_PLACE, &energy, 1, MPI_DOUBLE, MPI_SUM, m_exec_conf->getMPICommunicator());
        }
    #endif

    return energy;
    }


template <class Shape>
Scalar IntegratorHPMCMono<Shape>::getMaxCoreDiameter()
    {
    Scalar max_d(0.0);

    // access the type parameters
    ArrayHandle<Scalar> h_d(m_d, access_location::host, access_mode::read);

    // access interaction matrix
    ArrayHandle<unsigned int> h_overlaps(this->m_overlaps, access_location::host, access_mode::read);

    // for each type, create a temporary shape and return the maximum sum of diameter and move size
    for (unsigned int typ_i = 0; typ_i < this->m_pdata->getNTypes(); typ_i++)
        {
        Shape temp_i(quat<Scalar>(), m_params[typ_i]);

        for (unsigned int typ_j = 0; typ_j < this->m_pdata->getNTypes(); typ_j++)
            {
            Shape temp_j(quat<Scalar>(), m_params[typ_j]);

            // ignore non-interacting shapes
            if (h_overlaps.data[m_overlap_idx(typ_i,typ_j)])
                max_d = std::max(0.5*(temp_i.getCircumsphereDiameter()+temp_j.getCircumsphereDiameter()),max_d);
            }
        }
    return max_d;
    }

template <class Shape>
LongReal IntegratorHPMCMono<Shape>::getMinCoreDiameter()
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
        for (unsigned int typ =0; typ < this->m_pdata->getNTypes(); typ++)
            max_extent = std::max(max_extent, getMaxPairInteractionAdditiveRCut(typ));
        minD = std::max(LongReal(0.0), minD-max_extent);
        }

    return minD;
    }

/*! \param typ type name to set
    \param v python dictionary to convert to shape
*/
template <class Shape> inline
void IntegratorHPMCMono<Shape>::setShape(std::string typ, pybind11::dict v)
    {
    unsigned int id = this->m_pdata->getTypeByName(typ);
    setParam(id, typename Shape::param_type(v, m_exec_conf->isCUDAEnabled()));
    }

/*! \param typ type name to get
*/
template <class Shape> inline
pybind11::dict IntegratorHPMCMono<Shape>::getShape(std::string typ)
    {
    unsigned int id = this->m_pdata->getTypeByName(typ);
    return m_params[id].asDict();
    }

template <class Shape>
void IntegratorHPMCMono<Shape>::setParam(unsigned int typ,  const param_type& param)
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

template <class Shape>
void IntegratorHPMCMono<Shape>::setInteractionMatrix(std::pair<std::string, std::string> types,
                                                     bool check_overlaps)
    {
    auto typi = m_pdata->getTypeByName(types.first);
    auto typj = m_pdata->getTypeByName(types.second);

    // update the parameter for this type
    ArrayHandle<unsigned int> h_overlaps(m_overlaps, access_location::host, access_mode::readwrite);
    h_overlaps.data[m_overlap_idx(typi,typj)] = check_overlaps;
    h_overlaps.data[m_overlap_idx(typj,typi)] = check_overlaps;

    m_image_list_valid = false;
    }

template <class Shape>
bool IntegratorHPMCMono<Shape>::getInteractionMatrixPy(std::pair<std::string, std::string> types)
    {
    auto typi = m_pdata->getTypeByName(types.first);
    auto typj = m_pdata->getTypeByName(types.second);

    // update the parameter for this type
    ArrayHandle<unsigned int> h_overlaps(m_overlaps, access_location::host, access_mode::read);
    return h_overlaps.data[m_overlap_idx(typi,typj)];
    }

//! Calculate a list of box images within interaction range of the simulation box, innermost first
template <class Shape>
inline const std::vector<vec3<Scalar> >& IntegratorHPMCMono<Shape>::updateImageList()
    {
    // cancel if the image list is up to date
    if (m_image_list_valid)
        return m_image_list;

    // triclinic boxes have 4 linearly independent body diagonals
    // box_circumsphere = max(body_diagonals)
    // range = getMaxCoreDiameter() + box_circumsphere
    // while still adding images, examine successively larger blocks of images, checking the outermost against range

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
    // 2D simulations don't necessarily have a zero-size z-dimension, but it is convenient for us if we assume one.
    vec3<Scalar> e3(0,0,0);
    if (ndim == 3)
        e3 = vec3<Scalar>(box.getLatticeVector(2));

    // The maximum interaction range is the sum of the max particle circumsphere diameter
    // (accounting for non-additive interactions), the patch interaction and interaction with
    // any depletants in the system
    Scalar max_trans_d_and_diam(0.0);
        {
        // access the type parameters
        ArrayHandle<Scalar> h_d(m_d, access_location::host, access_mode::read);

        // access interaction matrix
        ArrayHandle<unsigned int> h_overlaps(this->m_overlaps, access_location::host, access_mode::read);

        // for each type, create a temporary shape and return the maximum sum of diameter and move size
        for (unsigned int typ_i = 0; typ_i < this->m_pdata->getNTypes(); typ_i++)
            {
            Shape temp_i(quat<Scalar>(), m_params[typ_i]);

            Scalar r_cut_patch_i(0.0);
            if (hasPairInteractions())
                {
                r_cut_patch_i = getMaxPairEnergyRCutNonAdditive() +
                    static_cast<ShortReal>(0.5) * getMaxPairInteractionAdditiveRCut(typ_i);
                }

            Scalar range_i(0.0);
            for (unsigned int typ_j = 0; typ_j < this->m_pdata->getNTypes(); typ_j++)
                {
                Scalar r_cut_patch_ij(0.0);
                if (hasPairInteractions())
                    {
                    r_cut_patch_ij = r_cut_patch_i + static_cast<ShortReal>(0.5)*getMaxPairInteractionAdditiveRCut(typ_j);
                    }

                Shape temp_j(quat<Scalar>(), m_params[typ_j]);
                Scalar r_cut_shape(0.0);
                if (h_overlaps.data[m_overlap_idx(typ_i,typ_j)])
                    r_cut_shape = 0.5*(temp_i.getCircumsphereDiameter()+temp_j.getCircumsphereDiameter());
                Scalar range_ij = detail::max(r_cut_shape,r_cut_patch_ij);
                range_i = detail::max(range_i,range_ij);
                }
            max_trans_d_and_diam = detail::max(max_trans_d_and_diam, range_i+Scalar(m_nselect)*h_d.data[typ_i]);
            }
        }

    m_exec_conf->msg->notice(6) << "Image list: max_trans_d_and_diam = " << max_trans_d_and_diam << std::endl;

    Scalar range = max_trans_d_and_diam;

    m_exec_conf->msg->notice(6) << "Image list: extra_image_width = " << m_extra_image_width << std::endl;

    // add any extra requested width
    range += m_extra_image_width;

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
            if (di.getW() > 1) x_max = 0;
            if (di.getH() > 1) y_max = 0;
            if (di.getD() > 1) z_max = 0;
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
        ShapeSpheropolyhedron image_box(quat<Scalar>(),  image_box_params);


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
                        vec3<Scalar> r_image = Scalar(hkl.x) * e1 + Scalar(hkl.y) * e2 + Scalar(hkl.z) * e3;

                        // check to see if the image box overlaps with the central box
                        unsigned int err = 0;
                        if (test_overlap(r_image, central_box, image_box, err))
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

    m_exec_conf->msg->notice(6) << "Image list:" << std::endl;
    for (unsigned int i = 0; i < m_image_list.size(); i++)
        m_exec_conf->msg->notice(6) << m_image_list[i].x << " " << m_image_list[i].y << " " << m_image_list[i].z << std::endl;

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

    m_exec_conf->msg->notice(6) << "Updated image list: " << m_image_list.size() << " images" << std::endl;
    return m_image_list;
    }

template <class Shape>
void IntegratorHPMCMono<Shape>::updateCellWidth()
    {
    this->m_nominal_width = this->getMaxCoreDiameter();

    Scalar max_d(0.0);

    for (unsigned int type_i = 0; type_i < this->m_pdata->getNTypes(); ++type_i)
        {
        if (m_fugacity[type_i] != Scalar(0.0))
            {
            // add range of depletion interaction
            quat<Scalar> o;
            Shape tmp_shape(o, this->m_params[type_i]);
            max_d = std::max(max_d, static_cast<Scalar>(tmp_shape.getCircumsphereDiameter()));
            }
        }

    // extend the image list by the depletant diameter, since we're querying
    // AABBs that are larger than the shape diameters themselves
    this->m_extra_image_width = max_d;

    this->m_nominal_width += this->m_extra_image_width;

    // Account for patch width
    if (hasPairInteractions())
        {
        Scalar max_extent = 0.0;
        for (unsigned int typ = 0; typ < this->m_pdata->getNTypes(); typ++)
            {
            max_extent = std::max(max_extent, static_cast<Scalar>(getMaxPairInteractionAdditiveRCut(typ)));
            }

        this->m_nominal_width = std::max(this->m_nominal_width, getMaxPairEnergyRCutNonAdditive() + max_extent);
        }
    this->m_image_list_valid = false;
    this->m_aabb_tree_invalid = true;

    this->m_exec_conf->msg->notice(5) << "IntegratorHPMCMono: updating nominal width to " << this->m_nominal_width << std::endl;
    }

template <class Shape>
void IntegratorHPMCMono<Shape>::growAABBList(unsigned int N)
    {
    if (N > m_aabbs_capacity)
        {
        m_aabbs_capacity = N;
        if (m_aabbs != NULL)
            free(m_aabbs);

        int retval = posix_memalign((void**)&m_aabbs, 32, N*sizeof(hoomd::detail::AABB));
        if (retval != 0)
            {
            throw std::runtime_error("Error allocating aligned memory.");
            }
        }
    }


/*! Call any time an up to date AABB tree is needed. IntegratorHPMCMono internally tracks whether
    the tree needs to be rebuilt or if the current tree can be used.

    buildAABBTree() relies on the member variable m_aabb_tree_invalid to work correctly. Any time particles
    are moved (and not updated with m_aabb_tree->update()) or the particle list changes order, m_aabb_tree_invalid
    needs to be set to true. Then buildAABBTree() will know to rebuild the tree from scratch on the next call. Typically
    this is on the next timestep. But in some cases (i.e. NPT), the tree may need to be rebuilt several times in a
    single step because of box volume moves.

    Subclasses that override update() or other methods must be user to set m_aabb_tree_invalid appropriately, or
    erroneous simulations will result.

    \returns A reference to the tree.
*/
template <class Shape>
const hoomd::detail::AABBTree& IntegratorHPMCMono<Shape>::buildAABBTree()
    {
    if (m_aabb_tree_invalid)
        {
        m_exec_conf->msg->notice(8) << "Building AABB tree: " << m_pdata->getN() << " ptls " << m_pdata->getNGhosts() << " ghosts" << std::endl;
        // build the AABB tree
            {
            ArrayHandle<Scalar4> h_postype(m_pdata->getPositions(), access_location::host, access_mode::read);
            ArrayHandle<Scalar4> h_orientation(m_pdata->getOrientationArray(), access_location::host, access_mode::read);

            // precompute constants used many times in the loop
            m_max_pair_additive_cutoff.clear();
            m_shape_circumsphere_radius.clear();
            for (unsigned int type = 0; type < m_pdata->getNTypes(); type++)
                {
                quat<LongReal> q;
                Shape shape(q, m_params[type]);
                m_shape_circumsphere_radius.push_back(LongReal(0.5) * shape.getCircumsphereDiameter());
                m_max_pair_additive_cutoff.push_back(getMaxPairInteractionAdditiveRCut(type));
                }

            // grow the AABB list to the needed size
            unsigned int n_aabb = m_pdata->getN()+m_pdata->getNGhosts();
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
                            LongReal(0.5)*m_max_pair_additive_cutoff[typ_i]);
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

/*! Call to reduce the m_d values down to safe levels for the bvh tree + small box limitations. That code path
    will not work if particles can wander more than one image in a time step.

    In MPI simulations, they may not move more than half a local box length.
*/
template <class Shape>
void IntegratorHPMCMono<Shape>::limitMoveDistances()
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
            m_exec_conf->msg->warning() << "Move distance or nselect too big, reducing move distance to "
                                        << h_d.data[typ] << " for type " << m_pdata->getNameByType(typ) << std::endl;
            m_image_list_valid = false;
            }
        // Sanity check should be performed in code where parameters can be adjusted.
        if (h_d.data[typ] < Scalar(0.0))
            {
            m_exec_conf->msg->warning() << "Move distance has become negative for type " << m_pdata->getNameByType(typ)
                                        << ". This should never happen. Please file a bug report." << std::endl;
            h_d.data[typ] = Scalar(0.0);
            }
        }
    }

/*! Function for finding all overlaps in a system by particle tag. returns an unraveled form of an NxN matrix
 * with true/false indicating the overlap status of the ith and jth particle
 */
template <class Shape>
std::vector<std::pair<unsigned int, unsigned int> > IntegratorHPMCMono<Shape>::mapOverlaps()
    {
    #ifdef ENABLE_MPI
    if (m_pdata->getDomainDecomposition())
        {
        throw std::runtime_error("map_overlaps does not support MPI parallel jobs");
        }
    #endif

    unsigned int N = m_pdata->getN();

    std::vector<std::pair<unsigned int, unsigned int> > overlap_vector;

    m_exec_conf->msg->notice(10) << "HPMC overlap mapping" << std::endl;

    unsigned int err_count = 0;

    // build an up to date AABB tree
    buildAABBTree();
    // update the image list
    updateImageList();

    // access particle data and system box
    ArrayHandle<Scalar4> h_postype(m_pdata->getPositions(), access_location::host, access_mode::read);
    ArrayHandle<Scalar4> h_orientation(m_pdata->getOrientationArray(), access_location::host, access_mode::read);
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
        hoomd::detail::AABB aabb_i_local = shape_i.getAABB(vec3<Scalar>(0,0,0));

        const unsigned int n_images = (unsigned int)m_image_list.size();
        for (unsigned int cur_image = 0; cur_image < n_images; cur_image++)
            {
            vec3<Scalar> pos_i_image = pos_i + m_image_list[cur_image];
            hoomd::detail::AABB aabb = aabb_i_local;
            aabb.translate(pos_i_image);

            // stackless search
            for (unsigned int cur_node_idx = 0; cur_node_idx < m_aabb_tree.getNumNodes(); cur_node_idx++)
                {
                if (aabb.overlaps(m_aabb_tree.getNodeAABB(cur_node_idx)))
                    {
                    if (m_aabb_tree.isNodeLeaf(cur_node_idx))
                        {
                        for (unsigned int cur_p = 0; cur_p < m_aabb_tree.getNodeNumParticles(cur_node_idx); cur_p++)
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

                            Shape shape_j(quat<Scalar>(orientation_j), m_params[__scalar_as_int(postype_j.w)]);

                            if (h_tag.data[i] <= h_tag.data[j]
                                && check_circumsphere_overlap(r_ij, shape_i, shape_j)
                                && test_overlap(r_ij, shape_i, shape_j, err_count)
                                && test_overlap(-r_ij, shape_j, shape_i, err_count))
                                {
                                overlap_vector.push_back(std::make_pair(h_tag.data[i],
                                                                        h_tag.data[j]));
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

/*! \param i The particle id in the list
    \param pos_i Particle position being tested
    \param shape_i Particle shape (including orientation) being tested
    \param typ_i Type of the particle being tested
    \param h_postype Pointer to GPUArray containing particle positions
    \param h_orientation Pointer to GPUArray containing particle orientations
    \param h_tag Array of permanent particle tags
    \param h_overlaps Pointer to GPUArray containing interaction matrix
    \param hpmc_counters_t&  Pointer to current counters
    \param hpmc_implicit_counters_t&  Pointer to current implicit counters
    \param rng_depletants The RNG used within this algorithm
    \param rng_i The RNG used for evaluating the Metropolis criterion

    In order to determine whether or not moves are accepted, particle positions are checked against a randomly generated set of depletant positions.

    NOTE: To avoid numerous acquires and releases of GPUArrays, data pointers are passed directly into this const function.
    */
template<class Shape>
inline bool IntegratorHPMCMono<Shape>::checkDepletantOverlap(unsigned int i, vec3<Scalar> pos_i,
    Shape shape_i, unsigned int typ_i, Scalar4 *h_postype, Scalar4 *h_orientation, const unsigned int *h_tag,
    const Scalar4 *h_vel, unsigned int *h_overlaps, hpmc_counters_t& counters, hpmc_implicit_counters_t *implicit_counters,
    uint64_t timestep, hoomd::RandomGenerator& rng_depletants,
    unsigned int seed_i_old, unsigned int seed_i_new)
    {
    const unsigned int n_images = (unsigned int) this->m_image_list.size();
    unsigned int ndim = this->m_sysdef->getNDimensions();

    Shape shape_old(quat<Scalar>(h_orientation[i]), this->m_params[typ_i]);
    hoomd::detail::AABB aabb_i_local = shape_i.getAABB(vec3<Scalar>(0,0,0));
    hoomd::detail::AABB aabb_i_local_old = shape_old.getAABB(vec3<Scalar>(0,0,0));

    #ifdef ENABLE_TBB
    std::vector< tbb::enumerable_thread_specific<hpmc_implicit_counters_t> >
        thread_implicit_counters(this->m_pdata->getNTypes());
    tbb::enumerable_thread_specific<hpmc_counters_t> thread_counters;
    #endif

    Scalar ln_numerator_tot(0.0);
    Scalar ln_denominator_tot(0.0);

    #ifdef ENABLE_TBB
    m_exec_conf->getTaskArena()->execute([&]{
    try {
    #endif

    for (unsigned int type_a = 0; type_a < this->m_pdata->getNTypes(); ++type_a)
        {
        // GlobalVector is not thread-safe, access it outside the parallel loop
        Scalar fugacity = m_fugacity[type_a];
        if (fugacity == 0.0 || !h_overlaps[this->m_overlap_idx(type_a, typ_i)])
            continue;

        unsigned int ntrial = m_ntrial[type_a];

        #ifdef ENABLE_TBB
        // deltaF == free energy difference
        tbb::enumerable_thread_specific<std::vector<Scalar> > thread_ln_numerator(ntrial,Scalar(0.0));
        tbb::enumerable_thread_specific<std::vector<Scalar> > thread_ln_denominator(ntrial,Scalar(0.0));
        #endif

        std::vector<Scalar> ln_numerator(ntrial,0.0);
        std::vector<Scalar> ln_denominator(ntrial,0.0);

        std::vector<vec3<Scalar> > pos_j_old;
        std::vector<quat<Scalar> > orientation_j_old;
        std::vector<unsigned int> type_j_old;
        std::vector<unsigned int> tag_j_old;
        std::vector<unsigned int> seed_j_old;

        std::vector<vec3<Scalar> > pos_j_new;
        std::vector<quat<Scalar> > orientation_j_new;
        std::vector<unsigned int> type_j_new;
        std::vector<unsigned int> tag_j_new;
        std::vector<unsigned int> seed_j_new;

        bool repulsive = fugacity < 0.0;

        // the relevant search radius is the one for the larger depletant
        Shape tmp_a(quat<Scalar>(), this->m_params[type_a]);
        ShortReal d_dep_search = tmp_a.getCircumsphereDiameter();

        // we're sampling in the larger volume, so that it strictly covers the insertion volume of
        // the smaller depletant
        ShortReal r_dep_sample = 0.5f*d_dep_search;

        // get old AABB and extend
        vec3<Scalar> lower = aabb_i_local_old.getLower();
        vec3<Scalar> upper = aabb_i_local_old.getUpper();
        lower.x -= d_dep_search; lower.y -= d_dep_search; lower.z -= d_dep_search;
        upper.x += d_dep_search; upper.y += d_dep_search; upper.z += d_dep_search;
        hoomd::detail::AABB aabb_local = hoomd::detail::AABB(lower,upper);

        vec3<Scalar> pos_i_old(h_postype[i]);

        detail::OBB obb_i_old = shape_old.getOBB(pos_i_old);

        // extend by depletant radius
        obb_i_old.lengths.x += r_dep_sample;
        obb_i_old.lengths.y += r_dep_sample;
        obb_i_old.lengths.z += r_dep_sample;

        // All image boxes (including the primary)
        for (unsigned int cur_image = 0; cur_image < n_images; cur_image++)
            {
            vec3<Scalar> pos_i_old_image = pos_i_old + this->m_image_list[cur_image];
            hoomd::detail::AABB aabb = aabb_local;
            aabb.translate(pos_i_old_image);

            // stackless search
            for (unsigned int cur_node_idx = 0; cur_node_idx < this->m_aabb_tree.getNumNodes(); cur_node_idx++)
                {
                if (aabb.overlaps(this->m_aabb_tree.getNodeAABB(cur_node_idx)))
                    {
                    if (this->m_aabb_tree.isNodeLeaf(cur_node_idx))
                        {
                        for (unsigned int cur_p = 0; cur_p < this->m_aabb_tree.getNodeNumParticles(cur_node_idx); cur_p++)
                            {
                            // read in its position and orientation
                            unsigned int j = this->m_aabb_tree.getNodeParticle(cur_node_idx, cur_p);

                            if (i == j && cur_image == 0) continue;

                            // load the old position and orientation of the j particle
                            Scalar4 postype_j = h_postype[j];
                            vec3<Scalar> pos_j(postype_j);

                            unsigned int typ_j = __scalar_as_int(postype_j.w);
                            Shape shape_j(quat<Scalar>(), this->m_params[typ_j]);
                            if (shape_j.hasOrientation())
                                shape_j.orientation = quat<Scalar>(h_orientation[j]);

                            // get shape OBB
                            detail::OBB obb_j = shape_j.getOBB(pos_j-this->m_image_list[cur_image]);

                            // extend by depletant radius
                            Shape shape_test_a(quat<Scalar>(), m_params[type_a]);

                            obb_j.lengths.x += r_dep_sample;
                            obb_j.lengths.y += r_dep_sample;
                            obb_j.lengths.z += r_dep_sample;

                            // check excluded volume overlap
                            bool overlap_excluded = (h_overlaps[this->m_overlap_idx(type_a,typ_j)]
                                    && detail::overlap(obb_j, obb_i_old));

                            if (overlap_excluded)
                                {
                                // cache the translated position of particle j. If i's image is cur_image, then j's
                                // image is the negative of that (and we use i's untranslated position below)
                                pos_j_old.push_back(pos_j-this->m_image_list[cur_image]);
                                orientation_j_old.push_back(shape_j.orientation);
                                type_j_old.push_back(typ_j);
                                tag_j_old.push_back(h_tag[j]);
                                seed_j_old.push_back(__scalar_as_int(h_vel[j].x));
                                }
                            }
                        }
                    }
                else
                    {
                    // skip ahead
                    cur_node_idx += this->m_aabb_tree.getNodeSkip(cur_node_idx);
                    }
                }  // end loop over AABB nodes
            } // end loop over images

        // get new AABB and extend
        lower = aabb_i_local.getLower();
        upper = aabb_i_local.getUpper();
        lower.x -= d_dep_search; lower.y -= d_dep_search; lower.z -= d_dep_search;
        upper.x += d_dep_search; upper.y += d_dep_search; upper.z += d_dep_search;
        aabb_local = hoomd::detail::AABB(lower,upper);

        detail::OBB obb_i_new = shape_i.getOBB(pos_i);
        obb_i_new.lengths.x += r_dep_sample;
        obb_i_new.lengths.y += r_dep_sample;
        obb_i_new.lengths.z += r_dep_sample;

        // find neighbors at new position
        for (unsigned int cur_image = 0; cur_image < n_images; cur_image++)
            {
            vec3<Scalar> pos_i_image = pos_i + this->m_image_list[cur_image];
            hoomd::detail::AABB aabb = aabb_local;
            aabb.translate(pos_i_image);

            // stackless search
            for (unsigned int cur_node_idx = 0; cur_node_idx < this->m_aabb_tree.getNumNodes(); cur_node_idx++)
                {
                if (aabb.overlaps(this->m_aabb_tree.getNodeAABB(cur_node_idx)))
                    {
                    if (this->m_aabb_tree.isNodeLeaf(cur_node_idx))
                        {
                        for (unsigned int cur_p = 0; cur_p < this->m_aabb_tree.getNodeNumParticles(cur_node_idx); cur_p++)
                            {
                            // read in its position and orientation
                            unsigned int j = this->m_aabb_tree.getNodeParticle(cur_node_idx, cur_p);

                            unsigned int typ_j;
                            vec3<Scalar> pos_j;
                            if (i == j)
                                {
                                if (cur_image == 0)
                                    continue;
                                else
                                    {
                                    pos_j = pos_i;
                                    typ_j = typ_i;
                                    }
                                }
                            else
                                {
                                // load the old position and orientation of the j particle
                                Scalar4 postype_j = h_postype[j];
                                pos_j = vec3<Scalar>(postype_j);
                                typ_j = __scalar_as_int(postype_j.w);
                                }

                            Shape shape_j(quat<Scalar>(), this->m_params[typ_j]);

                            if (shape_j.hasOrientation())
                                {
                                if (i == j)
                                    shape_j.orientation = shape_i.orientation;
                                else
                                    shape_j.orientation = quat<Scalar>(h_orientation[j]);
                                }

                            // get shape OBB
                            detail::OBB obb_j = shape_j.getOBB(pos_j-this->m_image_list[cur_image]);

                            // extend by depletant radius
                            Shape shape_test_a(quat<Scalar>(), m_params[type_a]);

                            obb_j.lengths.x += r_dep_sample;
                            obb_j.lengths.y += r_dep_sample;
                            obb_j.lengths.z += r_dep_sample;

                            // check excluded volume overlap
                            bool overlap_excluded = h_overlaps[this->m_overlap_idx(type_a,typ_j)] &&
                                detail::overlap(obb_j, obb_i_new);

                            if (overlap_excluded)
                                {
                                // cache the translated position of particle j. If i's image is cur_image, then j's
                                // image is the negative of that (and we use i's untranslated position below)
                                pos_j_new.push_back(pos_j-this->m_image_list[cur_image]);
                                orientation_j_new.push_back(shape_j.orientation);
                                type_j_new.push_back(typ_j);
                                tag_j_new.push_back(h_tag[j]);
                                seed_j_new.push_back(__scalar_as_int(h_vel[j].x));
                                }
                            }
                        }
                    }
                else
                    {
                    // skip ahead
                    cur_node_idx += this->m_aabb_tree.getNodeSkip(cur_node_idx);
                    }
                }  // end loop over AABB nodes
            } // end loop over images

        // insert into particle OBBs
        #ifdef ENABLE_TBB
        tbb::parallel_for(tbb::blocked_range<unsigned int>(0, 2),
            [=, &shape_old, &shape_i,
                &pos_j_new, &orientation_j_new, &type_j_new,
                &pos_j_old, &orientation_j_old, &type_j_old,
                &thread_ln_denominator, &thread_ln_numerator,
                &thread_counters, &thread_implicit_counters](const tbb::blocked_range<unsigned int>& v) {
        for (unsigned int new_config = v.begin(); new_config != v.end(); ++new_config)
        #else
        for (unsigned int new_config = 0; new_config < 2; ++new_config)
        #endif
            {
            detail::OBB obb_i = new_config ? obb_i_new : obb_i_old;
            Scalar V_i = obb_i.getVolume(ndim);

            #ifdef ENABLE_TBB
            tbb::parallel_for(tbb::blocked_range<unsigned int>(0, ntrial),
                [=, &shape_old, &shape_i,
                    &pos_j_new, &orientation_j_new, &type_j_new,
                    &pos_j_old, &orientation_j_old, &type_j_old,
                    &thread_ln_denominator, &thread_ln_numerator,
                    &thread_counters, &thread_implicit_counters]
                    (const tbb::blocked_range<unsigned int>& u) {
            for (unsigned int i_trial = u.begin(); i_trial != u.end(); ++i_trial)
            #else
            for (unsigned int i_trial = 0; i_trial < ntrial; ++i_trial)
            #endif
                {
                // chooose the number of depletants in the insertion OBB
                Scalar lambda = std::abs(fugacity)*V_i;
                hoomd::PoissonDistribution<Scalar> poisson(lambda);
                hoomd::RandomGenerator rng_num(hoomd::Seed(hoomd::RNGIdentifier::HPMCDepletantNum,
                                                            0,
                                                            0),
                                                hoomd::Counter(type_a,
                                                                new_config ? seed_i_new : seed_i_old,
                                                                i_trial)
                                                );

                unsigned int n = poisson(rng_num);

                // try inserting in the overlap volume
                size_t n_intersect = new_config ? pos_j_new.size() : pos_j_old.size();

                // for every depletant
                #ifdef ENABLE_TBB
                tbb::parallel_for(tbb::blocked_range<unsigned int>(0, (unsigned int)n),
                    [=, &shape_old, &shape_i,
                        &pos_j_new, &orientation_j_new, &type_j_new,
                        &pos_j_old, &orientation_j_old, &type_j_old,
                        &thread_ln_denominator, &thread_ln_numerator,
                        &thread_counters, &thread_implicit_counters](const tbb::blocked_range<unsigned int>& t) {
                for (unsigned int l = t.begin(); l != t.end(); ++l)
                #else
                for (unsigned int l = 0; l < n; ++l)
                #endif
                    {
                    hoomd::RandomGenerator my_rng(hoomd::Seed(hoomd::RNGIdentifier::HPMCDepletants,
                                                                0,
                                                                0),
                                                    hoomd::Counter(new_config ? seed_i_new : seed_i_old,
                                                                    l,
                                                                    i_trial,
                                                                    static_cast<uint16_t>(type_a))
                                                    );

                    if (! shape_i.ignoreStatistics())
                        {
                        #ifdef ENABLE_TBB
                        thread_implicit_counters[type_a].local().insert_count++;
                        #else
                        implicit_counters[type_a].insert_count++;
                        #endif
                        }

                    // rejection-free sampling
                    vec3<Scalar> pos_test(generatePositionInOBB(my_rng, obb_i, ndim));

                    Shape shape_test_a(quat<Scalar>(), this->m_params[type_a]);
                    quat<Scalar> o;
                    if (shape_test_a.hasOrientation())
                        {
                        o = generateRandomOrientation(my_rng, ndim);
                        }
                    if (shape_test_a.hasOrientation())
                        shape_test_a.orientation = o;

                    // Check if the new (old) configuration of particle i generates an overlap
                    bool overlap_i_a = false;
                    vec3<Scalar> r_i_test = pos_test - (new_config ? pos_i : pos_i_old);
                        {
                        const Shape& shape = !new_config ? shape_old : shape_i;

                        ShortReal rsq = (ShortReal) dot(r_i_test,r_i_test);
                        ShortReal DaDb = shape_test_a.getCircumsphereDiameter() + shape.getCircumsphereDiameter();
                        bool circumsphere_overlap = (rsq*ShortReal(4.0) <= DaDb * DaDb);

                        if (h_overlaps[this->m_overlap_idx(type_a, typ_i)])
                            {
                            #ifdef ENABLE_TBB
                            thread_counters.local().overlap_checks++;
                            #else
                            counters.overlap_checks++;
                            #endif

                            unsigned int err = 0;
                            if (circumsphere_overlap &&
                                test_overlap(r_i_test, shape, shape_test_a, err))
                                {
                                overlap_i_a = true;
                                }
                            if (err)
                            #ifdef ENABLE_TBB
                                thread_counters.local().overlap_err_count++;
                            #else
                                counters.overlap_err_count++;
                            #endif
                            }
                        }

                    if (!overlap_i_a)
                        {
                        // reject because we can't insert in overlap volume
                        continue;
                        }

                    unsigned int n_overlap = 0;
                    unsigned int tag_i = h_tag[i];
                    for (size_t m = 0; m < n_intersect; ++m)
                        {
                        unsigned int type_m = new_config ? type_j_new[m] : type_j_old[m];
                        Shape shape_m(new_config ? orientation_j_new[m] : orientation_j_old[m], this->m_params[type_m]);
                        vec3<Scalar> r_mk = (new_config ? pos_j_new[m] : pos_j_old[m]) - pos_test;

                        #ifdef ENABLE_TBB
                        thread_counters.local().overlap_checks++;
                        #else
                        counters.overlap_checks++;
                        #endif

                        unsigned int err = 0;

                        // check circumsphere overlap
                        ShortReal rsq = (ShortReal) dot(r_mk,r_mk);
                        ShortReal DaDb = shape_test_a.getCircumsphereDiameter() + shape_m.getCircumsphereDiameter();
                        bool circumsphere_overlap = (rsq*ShortReal(4.0) <= DaDb * DaDb);

                        bool overlap_j_a = h_overlaps[this->m_overlap_idx(type_a,type_m)]
                            && circumsphere_overlap
                            && test_overlap(r_mk, shape_test_a, shape_m, err);

                        // non-additive depletants
                        // Tim says: not sure what to do here for getting rid of
                        // non-additive depletants
                        if (overlap_i_a || overlap_j_a)
                            {
                            unsigned int tag_m = new_config ? tag_j_new[m] : tag_j_old[m];
                            if (tag_i < tag_m)
                                {
                                n_overlap++;
                                }
                            }

                        if (err)
                        #ifdef ENABLE_TBB
                            thread_counters.local().overlap_err_count++;
                        #else
                            counters.overlap_err_count++;
                        #endif
                        } // end loop over intersections

                    unsigned int chi = n_overlap > 0;

                    Scalar betaF = log(1.0+(Scalar)chi/(Scalar)ntrial);

                    if ((repulsive && new_config) || (!repulsive && !new_config))
                        {
                        #ifdef ENABLE_TBB
                        thread_ln_denominator.local()[i_trial] += betaF;
                        #else
                        ln_denominator[i_trial] += betaF;
                        #endif
                        }
                    else
                        {
                        #ifdef ENABLE_TBB
                        thread_ln_numerator.local()[i_trial] += betaF;
                        #else
                        ln_numerator[i_trial] += betaF;
                        #endif
                        }
                    } // end loop over depletants
                #ifdef ENABLE_TBB
                    });
                #endif

                // insert into each neighbor volume
                #ifdef ENABLE_TBB
                tbb::parallel_for(tbb::blocked_range<size_t>(0, n_intersect),
                    [=, &shape_old, &shape_i,
                        &pos_j_new, &orientation_j_new, &type_j_new,
                        &pos_j_old, &orientation_j_old, &type_j_old,
                        &thread_ln_denominator, &thread_ln_numerator,
                        &thread_counters, &thread_implicit_counters](const tbb::blocked_range<size_t>& y) {
                for (size_t k = y.begin(); k != y.end(); ++k)
                #else
                for (size_t k = 0; k < n_intersect; ++k)
                #endif
                    {
                    detail::OBB obb_k;
                    Scalar V_k;
                    Shape shape_k(new_config ? orientation_j_new[k] : orientation_j_old[k],
                        this->m_params[new_config ? type_j_new[k] : type_j_old[k]]);

                        {
                        // get shape OBB
                        obb_k = shape_k.getOBB(new_config ? pos_j_new[k] : pos_j_old[k]);

                        // extend by depletant radius
                        Shape shape_test_a(quat<Scalar>(), m_params[type_a]);

                        ShortReal r = 0.5f * shape_test_a.getCircumsphereDiameter();
                        obb_k.lengths.x += r;
                        obb_k.lengths.y += r;
                        obb_k.lengths.z += r;

                        V_k = obb_k.getVolume(ndim);
                        }

                    // random number of depletants uniquely for this volume
                    Scalar lambda = std::abs(fugacity)*V_k;
                    hoomd::PoissonDistribution<Scalar> poisson(lambda);
                    unsigned int seed_j = new_config ? seed_j_new[k] : seed_j_old[k];
                    hoomd::RandomGenerator rng_num(hoomd::Seed(hoomd::RNGIdentifier::HPMCDepletantNum,
                                                                0,
                                                                0),
                                                    hoomd::Counter(type_a,
                                                                    seed_j,
                                                                    i_trial)
                                                    );

                    unsigned int n = poisson(rng_num);

                    // for every depletant
                    #ifdef ENABLE_TBB
                    tbb::parallel_for(tbb::blocked_range<unsigned int>(0, (unsigned int)n),
                        [=, &shape_old, &shape_i,
                            &pos_j_new, &orientation_j_new, &type_j_new,
                            &pos_j_old, &orientation_j_old, &type_j_old,
                            &thread_ln_denominator, &thread_ln_numerator,
                            &thread_counters, &thread_implicit_counters](const tbb::blocked_range<unsigned int>& t) {
                    for (unsigned int l = t.begin(); l != t.end(); ++l)
                    #else
                    for (unsigned int l = 0; l < n; ++l)
                    #endif
                        {
                        hoomd::RandomGenerator my_rng(hoomd::Seed(hoomd::RNGIdentifier::HPMCDepletants,
                                                                    0,
                                                                    0),
                                                        hoomd::Counter(seed_j,
                                                                        l,
                                                                        i_trial,
                                                                        static_cast<uint16_t>(type_a))
                                                    );

                        if (! shape_i.ignoreStatistics())
                            {
                            #ifdef ENABLE_TBB
                            thread_implicit_counters[type_a].local().insert_count++;
                            #else
                            implicit_counters[type_a].insert_count++;
                            #endif
                            }

                        // rejection-free sampling
                        vec3<Scalar> pos_test(generatePositionInOBB(my_rng, obb_k, ndim));

                        Shape shape_test_a(quat<Scalar>(), this->m_params[type_a]);
                        quat<Scalar> o;
                        if (shape_test_a.hasOrientation())
                            {
                            o = generateRandomOrientation(my_rng, ndim);
                            }
                        if (shape_test_a.hasOrientation())
                            shape_test_a.orientation = o;
                        // Check if the particle j overlaps
                        bool overlap_k_a = false;

                        vec3<Scalar> r_k_test = pos_test - (new_config ? pos_j_new[k] : pos_j_old[k]);

                            {
                            ShortReal rsq = (ShortReal) dot(r_k_test,r_k_test);
                            ShortReal DaDb = shape_test_a.getCircumsphereDiameter() + shape_k.getCircumsphereDiameter();
                            bool circumsphere_overlap = (rsq*ShortReal(4.0) <= DaDb * DaDb);

                            if (h_overlaps[this->m_overlap_idx(type_a, new_config ? type_j_new[k] : type_j_old[k])])
                                {
                                #ifdef ENABLE_TBB
                                thread_counters.local().overlap_checks++;
                                #else
                                counters.overlap_checks++;
                                #endif

                                unsigned int err = 0;
                                if (circumsphere_overlap &&
                                    test_overlap(r_k_test, shape_k, shape_test_a, err))
                                    {
                                    overlap_k_a = true;
                                    }
                                if (err)
                                #ifdef ENABLE_TBB
                                    thread_counters.local().overlap_err_count++;
                                #else
                                    counters.overlap_err_count++;
                                #endif
                                }
                            }

                        if (!overlap_k_a)
                            {
                            // not in j's excluded volume
                            continue;
                            }

                        // does particle i overlap in current configuration?
                        bool overlap_i_a = false;
                        vec3<Scalar> r_i_test = pos_test - (new_config ? pos_i : pos_i_old);
                            {
                            const Shape& shape = new_config ? shape_i : shape_old;

                            ShortReal rsq = (ShortReal) dot(r_i_test,r_i_test);
                            ShortReal DaDb = shape_test_a.getCircumsphereDiameter() + shape.getCircumsphereDiameter();
                            bool circumsphere_overlap = (rsq*ShortReal(4.0) <= DaDb * DaDb);

                            if (h_overlaps[this->m_overlap_idx(type_a, typ_i)])
                                {
                                #ifdef ENABLE_TBB
                                thread_counters.local().overlap_checks++;
                                #else
                                counters.overlap_checks++;
                                #endif

                                unsigned int err = 0;
                                if (circumsphere_overlap &&
                                    test_overlap(r_i_test, shape, shape_test_a, err))
                                    {
                                    overlap_i_a = true;
                                    }
                                if (err)
                                #ifdef ENABLE_TBB
                                    thread_counters.local().overlap_err_count++;
                                #else
                                    counters.overlap_err_count++;
                                #endif
                                }
                            }

                        // does particle i overlap in the other configuration?
                        bool overlap_i_other_a = false;

                        vec3<Scalar> r_i_test_other = pos_test - (!new_config ? pos_i : pos_i_old);
                            {
                            const Shape& shape = !new_config ? shape_i : shape_old;

                            ShortReal rsq = (ShortReal) dot(r_i_test_other,r_i_test_other);
                            ShortReal DaDb = shape_test_a.getCircumsphereDiameter() + shape.getCircumsphereDiameter();
                            bool circumsphere_overlap = (rsq*ShortReal(4.0) <= DaDb * DaDb);

                            if (h_overlaps[this->m_overlap_idx(type_a, typ_i)])
                                {
                                #ifdef ENABLE_TBB
                                thread_counters.local().overlap_checks++;
                                #else
                                counters.overlap_checks++;
                                #endif

                                unsigned int err = 0;
                                if (circumsphere_overlap &&
                                    test_overlap(r_i_test_other, shape, shape_test_a, err))
                                    {
                                    overlap_i_other_a = true;
                                    }
                                if (err)
                                #ifdef ENABLE_TBB
                                    thread_counters.local().overlap_err_count++;
                                #else
                                    counters.overlap_err_count++;
                                #endif
                                }
                            }

                        unsigned int tag_i = h_tag[i];
                        unsigned int tag_k = new_config ? tag_j_new[k] : tag_j_old[k];
                        unsigned int n_overlap = 0;

                        for (unsigned int m = 0; m < n_intersect; ++m)
                            {
                            unsigned int type_m = new_config ? type_j_new[m] : type_j_old[m];
                            Shape shape_m(new_config ? orientation_j_new[m] : orientation_j_old[m],
                                this->m_params[type_m]);
                            vec3<Scalar> r_m_test = vec3<Scalar>(pos_test) - (new_config ? pos_j_new[m] : pos_j_old[m]);

                            #ifdef ENABLE_TBB
                            thread_counters.local().overlap_checks++;
                            #else
                            counters.overlap_checks++;
                            #endif

                            unsigned int err = 0;

                            // check circumsphere overlap
                            ShortReal rsq = (ShortReal) dot(r_m_test,r_m_test);
                            ShortReal DaDb = shape_test_a.getCircumsphereDiameter() + shape_m.getCircumsphereDiameter();
                            bool circumsphere_overlap = (rsq*ShortReal(4.0) <= DaDb * DaDb);

                            bool overlap_m_a = h_overlaps[this->m_overlap_idx(type_a,type_m)]
                                && circumsphere_overlap
                                && test_overlap(r_m_test, shape_m, shape_test_a, err);

                            // non-additive depletants
                            // Tim says: not sure what to do here about
                            // removing non-additive depletants
                            if (overlap_m_a || overlap_k_a)
                                {
                                unsigned int tag_m = new_config ? tag_j_new[m] : tag_j_old[m];
                                if (tag_m > tag_k) // also excludes self-overlap, doesn't work in small boxes
                                    {
                                    n_overlap++;
                                    break;
                                    }
                                }

                            if (err)
                            #ifdef ENABLE_TBB
                                thread_counters.local().overlap_err_count++;
                            #else
                                counters.overlap_err_count++;
                            #endif
                            }

                        bool overlap_ik = (overlap_k_a || overlap_i_a) && (tag_i > tag_k);
                        bool overlap_ik_other = (overlap_k_a || overlap_i_other_a) && (tag_i > tag_k);

                        // indicator function for MC integration
                        unsigned int chi = 0;

                        if (!overlap_ik_other && overlap_ik && !n_overlap)
                            chi = 1;

                        Scalar betaF = log(1.0+(Scalar)chi/(Scalar)ntrial);

                        if ((repulsive && new_config) || (!repulsive && !new_config))
                            {
                            #ifdef ENABLE_TBB
                            thread_ln_denominator.local()[i_trial] += betaF;
                            #else
                            ln_denominator[i_trial] += betaF;
                            #endif
                            }
                        else
                            {
                            #ifdef ENABLE_TBB
                            thread_ln_numerator.local()[i_trial] += betaF;
                            #else
                            ln_numerator[i_trial] += betaF;
                            #endif
                            }
                        } // end loop over depletants
                        #ifdef ENABLE_TBB
                        });
                        #endif
                    } // end loop over intersections
                    #ifdef ENABLE_TBB
                    });
                    #endif
                } // end loop over i_trial
                #ifdef ENABLE_TBB
                });
                #endif
            } // end loop over configurations
        #ifdef ENABLE_TBB
            });
        #endif

        #ifdef ENABLE_TBB
        for (auto ln_denom_thread : thread_ln_denominator)
            {
            for (unsigned int itrial = 0; itrial < ntrial; itrial++)
                ln_denominator[itrial] += ln_denom_thread[itrial];
            }
        #endif
        for (auto term_itrial : ln_denominator)
            ln_denominator_tot += term_itrial;

        #ifdef ENABLE_TBB
        for (auto ln_num_thread : thread_ln_numerator)
            {
            for (unsigned int itrial = 0; itrial < ntrial; itrial++)
                ln_numerator[itrial] += ln_num_thread[itrial];
            }
        #endif
        for (auto term_itrial : ln_numerator)
            ln_numerator_tot += term_itrial;
        } // end loop over type_a

    #ifdef ENABLE_TBB
    } catch (bool b) { }
    }); // end task arena execute()
    #endif

    Scalar u = hoomd::UniformDistribution<Scalar>()(rng_depletants);
    bool accept = u <= exp(ln_numerator_tot-ln_denominator_tot);

    #ifdef ENABLE_TBB
    // reduce counters
    for (auto i = thread_counters.begin(); i != thread_counters.end(); ++i)
        {
        counters = counters + *i;
        }

    for (unsigned int i = 0; i < this->m_pdata->getNTypes(); ++i)
        for (auto it = thread_implicit_counters[i].begin(); it != thread_implicit_counters[i].end(); ++it)
            {
            implicit_counters[i] = implicit_counters[i] + *it;
            }
    #endif

    return accept;
    }  // end IntegratorHPMCMono<Shape>::checkDepletantOverlap

template<class Shape>
bool IntegratorHPMCMono<Shape>::attemptBoxResize(uint64_t timestep, const BoxDim& new_box)
    {
    // call parent class method
    bool result = IntegratorHPMC::attemptBoxResize(timestep, new_box);

    if (result)
        {
        for (unsigned int ptype = 0; ptype < this->m_pdata->getNTypes(); ++ptype)
            {
            if (getDepletantFugacity(ptype) != 0.0)
                throw std::runtime_error("Implicit depletants not supported with NPT ensemble\n");
            }
        }

    return result;
    }

namespace detail {

//! Export the IntegratorHPMCMono class to python
/*! \param name Name of the class in the exported python module
    \tparam Shape An instantiation of IntegratorHPMCMono<Shape> will be exported
*/
template < class Shape > void export_IntegratorHPMCMono(pybind11::module& m, const std::string& name)
    {
    pybind11::class_< IntegratorHPMCMono<Shape>, IntegratorHPMC, std::shared_ptr< IntegratorHPMCMono<Shape> > >(m, name.c_str())
          .def(pybind11::init< std::shared_ptr<SystemDefinition> >())
          .def("setParam", &IntegratorHPMCMono<Shape>::setParam)
          .def("setInteractionMatrix", &IntegratorHPMCMono<Shape>::setInteractionMatrix)
          .def("getInteractionMatrix", &IntegratorHPMCMono<Shape>::getInteractionMatrixPy)
          .def("setExternalField", &IntegratorHPMCMono<Shape>::setExternalField)
          .def("setPatchEnergy", &IntegratorHPMCMono<Shape>::setPatchEnergy)
          .def("getPatchEnergy", &IntegratorHPMCMono<Shape>::getPatchEnergy)
          .def("mapOverlaps", &IntegratorHPMCMono<Shape>::mapOverlaps)
          .def("getImplicitCounters", &IntegratorHPMCMono<Shape>::getImplicitCounters)
          .def("getDepletantNtrial", &IntegratorHPMCMono<Shape>::getNtrialPy)
          .def("setDepletantNtrial", &IntegratorHPMCMono<Shape>::setNtrialPy)
          .def("setDepletantFugacity", &IntegratorHPMCMono<Shape>::setDepletantFugacityPy)
          .def("getDepletantFugacity", &IntegratorHPMCMono<Shape>::getDepletantFugacityPy)
          .def("getTypeShapesPy", &IntegratorHPMCMono<Shape>::getTypeShapesPy)
          .def("getShape", &IntegratorHPMCMono<Shape>::getShape)
          .def("setShape", &IntegratorHPMCMono<Shape>::setShape)
          .def("computePairEnergy", &IntegratorHPMCMono<Shape>::computePairEnergy)
          ;
    }

//! Export the counters for depletants
inline void export_hpmc_implicit_counters(pybind11::module& m)
    {
    pybind11::class_< hpmc_implicit_counters_t >(m, "hpmc_implicit_counters_t")
    .def_readwrite("insert_count", &hpmc_implicit_counters_t::insert_count)
    .def_readwrite("insert_accept_count", &hpmc_implicit_counters_t::insert_accept_count)
    .def_readwrite("insert_accept_count_sq", &hpmc_implicit_counters_t::insert_accept_count_sq)
    ;
    }

} // end namespace detail
} // end namespace hpmc
} // end namespace hoomd

#endif // _INTEGRATOR_HPMC_MONO_H_
