// Copyright (c) 2009-2019 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.

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
#include "HPMCPrecisionSetup.h"
#include "IntegratorHPMC.h"
#include "Moves.h"
#include "hoomd/AABBTree.h"
#include "GSDHPMCSchema.h"
#include "hoomd/Index1D.h"
#include "hoomd/RNGIdentifiers.h"
#include "hoomd/managed_allocator.h"
#include "hoomd/GSDShapeSpecWriter.h"

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
#endif


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
        UpdateOrder(unsigned int seed, unsigned int N=0)
            : m_seed(seed)
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
        void shuffle(unsigned int timestep, unsigned int select = 0)
            {
            hoomd::RandomGenerator rng(hoomd::RNGIdentifier::HPMCMonoShuffle, m_seed, timestep, select);

            // reverse the order with 1/2 probability
            if (hoomd::UniformIntDistribution(1)(rng))
                {
                unsigned int N = m_update_order.size();
                for (unsigned int i = 0; i < N; i++)
                    m_update_order[i] = N - i - 1;
                }
            else
                {
                unsigned int N = m_update_order.size();
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
        unsigned int m_seed;                       //!< Random number seed
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
        IntegratorHPMCMono(std::shared_ptr<SystemDefinition> sysdef,
                      unsigned int seed);

        virtual ~IntegratorHPMCMono()
            {
            if (m_aabbs != NULL)
                free(m_aabbs);
            m_pdata->getBoxChangeSignal().template disconnect<IntegratorHPMCMono<Shape>, &IntegratorHPMCMono<Shape>::slotBoxChanged>(this);
            m_pdata->getParticleSortSignal().template disconnect<IntegratorHPMCMono<Shape>, &IntegratorHPMCMono<Shape>::slotSorted>(this);
            }

        virtual void printStats();

        virtual void resetStats();

        //! Take one timestep forward
        virtual void update(unsigned int timestep);

        /*
         * Depletant related options
         */

        //! Set the depletant density in the free volume
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

        //! Set quermass integration mode
        void setQuermassMode(bool enable_quermass)
            {
            if (enable_quermass && !Shape::supportsSweepRadius())
                throw std::runtime_error("Quermass integration not supported for this shape");

            m_quermass = enable_quermass;
            }

        //! Get the quermass integration state
        bool getQuermassMode()
            {
            return m_quermass;
            }

        //! Set up the additional sweep radius around every shape
        void setSweepRadius(Scalar sweep_radius)
            {
            // check if supported
            if (sweep_radius != 0.0 && !Shape::supportsSweepRadius())
                throw std::runtime_error("This shape doesn't support setting a sweep radius to extend the surface out.\n");

            m_sweep_radius = sweep_radius;
            }

        //! Get the sweep radius
        Scalar getSweepRadius()
             {
             return m_sweep_radius;
             }

        //! Get the current counter values
        std::vector<hpmc_implicit_counters_t> getImplicitCounters(unsigned int mode=0);

        //! Method to scale the box
        virtual bool attemptBoxResize(unsigned int timestep, const BoxDim& new_box);

        /*
         * Common HPMC API
         */

        //! Get the maximum particle diameter
        virtual Scalar getMaxCoreDiameter();

        //! Get the minimum particle diameter
        virtual OverlapReal getMinCoreDiameter();

        //! Set the pair parameters for a single type
        virtual void setParam(unsigned int typ, const param_type& param);

        //! Set elements of the interaction matrix
        virtual void setOverlapChecks(unsigned int typi, unsigned int typj, bool check_overlaps);

        //! Set the external field for the integrator
        void setExternalField(std::shared_ptr< ExternalFieldMono<Shape> > external)
            {
            m_external = external;
            this->m_external_base = (ExternalField*)external.get();
            }

        //! Get a list of logged quantities
        virtual std::vector< std::string > getProvidedLogQuantities();

        //! Get the value of a logged quantity
        virtual Scalar getLogValue(const std::string& quantity, unsigned int timestep);

        //! Get the particle parameters
        virtual std::vector<param_type, managed_allocator<param_type> >& getParams()
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
        virtual unsigned int countOverlaps(unsigned int timestep, bool early_exit);

        //! Return a vector that is an unwrapped overlap map
        virtual std::vector<bool> mapOverlaps();

        //! Return a python list that is an unwrapped overlap map
        virtual pybind11::list PyMapOverlaps();

        //! Return a vector that is an unwrapped energy map
        virtual std::vector<float> mapEnergies();

        //! Return a python list that is an unwrapped energy map
        virtual pybind11::list PyMapEnergies();

        //! Test overlap for a given pair of particle coordinates
        /*! \param type_i Type of first particle
            \param type_j Type of second particle
            \param rij Separation vector rj-ri
            \param qi Orientation quaternion of first particle
            \param qj Orientation quaternion of second particle
            \param use_images if true, take into account periodic boundary conditions
            \param exclude_self if true, exclude the self-image

            \returns true if particles overlap
         */
        virtual bool py_test_overlap(unsigned int type_i, unsigned int type_j,
            pybind11::list rij, pybind11::list qi, pybind11::list qj,
            bool use_images, bool exclude_self);

        //! Return the requested ghost layer width
        virtual Scalar getGhostLayerWidth(unsigned int)
            {
            Scalar ghost_width = m_nominal_width + m_extra_ghost_width;
            m_exec_conf->msg->notice(9) << "IntegratorHPMCMono: ghost layer width of " << ghost_width << std::endl;
            return ghost_width;
            }

        #ifdef ENABLE_MPI
        //! Return the requested communication flags for ghost particles
        virtual CommFlags getCommFlags(unsigned int)
            {
            CommFlags flags(0);
            flags[comm_flag::position] = 1;
            flags[comm_flag::tag] = 1;

            std::ostringstream o;
            o << "IntegratorHPMCMono: Requesting communication flags for pos tag orientation";

            // many things depend internally on the orientation field (for ghosts) being initialized, therefore always request it
            flags[comm_flag::orientation] = 1;

            if (m_patch && (!m_patch_log || m_pdata->getFlags()[pdata_flag::potential_energy]))
                {
                flags[comm_flag::diameter] = 1;
                flags[comm_flag::charge] = 1;
                o << "diameter charge";
                }

            m_exec_conf->msg->notice(9) << o.str() << std::endl;
            return flags;
            }
        #endif

        //! Prepare for the run
        virtual void prepRun(unsigned int timestep)
            {
            // base class method
            IntegratorHPMC::prepRun(timestep);

                {
                // for p in params, if Shape dummy(q_dummy, params).hasOrientation() then m_hasOrientation=true
                m_hasOrientation = false;
                quat<Scalar> q(make_scalar4(1,0,0,0));
                for (unsigned int i=0; i < m_pdata->getNTypes(); i++)
                    {
                    Shape dummy(q, m_params[i]);
                    if (dummy.hasOrientation())
                        m_hasOrientation = true;
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
            if (m_comm)
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

        //! Compute the energy due to patch interactions
        /*! \param timestep the current time step
         * \returns the total patch energy
         */
        virtual float computePatchEnergy(unsigned int timestep);

        //! Build the AABB tree (if needed)
        const detail::AABBTree& buildAABBTree();

        //! Make list of image indices for boxes to check in small-box mode
        const std::vector<vec3<Scalar> >& updateImageList();

        //! Return list of integer shift vectors for periodic images
        const std::vector<int3>& getImageHKL()
            {
            updateImageList();
            return m_image_hkl;
            }

        //! Method to be called when number of types changes
        virtual void slotNumTypesChange();

        void invalidateAABBTree(){ m_aabb_tree_invalid = true; }

        //! Method that is called whenever the GSD file is written if connected to a GSD file.
        int slotWriteGSDState(gsd_handle&, std::string name) const;

        //! Method that is called whenever the GSD file is written if connected to a GSD file.
        int slotWriteGSDShapeSpec(gsd_handle&) const;

        //! Method that is called to connect to the gsd write state signal
        void connectGSDStateSignal(std::shared_ptr<GSDDumpWriter> writer, std::string name);

        //! Method that is called to connect to the gsd write shape signal
        void connectGSDShapeSpec(std::shared_ptr<GSDDumpWriter> writer);

        //! Method that is called to connect to the gsd write state signal
        bool restoreStateGSD(std::shared_ptr<GSDReader> reader, std::string name);

        std::vector<std::string> getTypeShapeMapping(const std::vector<param_type, managed_allocator<param_type> > &params) const
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
        std::vector<param_type, managed_allocator<param_type> > m_params;   //!< Parameters for each particle type on GPU
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
        detail::AABBTree m_aabb_tree;               //!< Bounding volume hierarchy for overlap checks
        detail::AABB* m_aabbs;                      //!< list of AABBs, one per particle
        unsigned int m_aabbs_capacity;              //!< Capacity of m_aabbs list
        bool m_aabb_tree_invalid;                   //!< Flag if the aabb tree has been invalidated

        Scalar m_extra_image_width;                 //! Extra width to extend the image list

        Index2D m_overlap_idx;                      //!!< Indexer for interaction matrix

        /* Depletants related data members */

        std::vector<Scalar> m_fugacity;                          //!< Average depletant number density in free volume, per type

        GlobalArray<hpmc_implicit_counters_t> m_implicit_count;               //!< Counter of depletant insertions
        std::vector<hpmc_implicit_counters_t> m_implicit_count_run_start;     //!< Counter of depletant insertions at run start
        std::vector<hpmc_implicit_counters_t> m_implicit_count_step_start;    //!< Counter of depletant insertions at step start

        bool m_quermass;                                         //!< True if quermass integration mode is enabled
        Scalar m_sweep_radius;                                   //!< Radius of sphere to sweep shapes by

        //! Test whether to reject the current particle move based on depletants
        #ifndef ENABLE_TBB
        inline bool checkDepletantOverlap(unsigned int i, vec3<Scalar> pos_i, Shape shape_i, unsigned int typ_i,
            Scalar4 *h_postype, Scalar4 *h_orientation, unsigned int *h_overlaps,
            hpmc_counters_t& counters, hpmc_implicit_counters_t *implicit_counters,
            hoomd::RandomGenerator& rng_depletants);
        #else
        inline bool checkDepletantOverlap(unsigned int i, vec3<Scalar> pos_i, Shape shape_i, unsigned int typ_i,
            Scalar4 *h_postype, Scalar4 *h_orientation, unsigned int *h_overlaps,
            hpmc_counters_t& counters, hpmc_implicit_counters_t *implicit_counters,
            tbb::enumerable_thread_specific< hoomd::RandomGenerator >& rng_depletants_parallel);
        #endif

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
IntegratorHPMCMono<Shape>::IntegratorHPMCMono(std::shared_ptr<SystemDefinition> sysdef,
                                                   unsigned int seed)
            : IntegratorHPMC(sysdef, seed),
              m_update_order(seed+m_exec_conf->getRank(), m_pdata->getN()),
              m_image_list_is_initialized(false),
              m_image_list_valid(false),
              m_hasOrientation(true),
              m_extra_image_width(0.0),
              m_quermass(false),
              m_sweep_radius(0.0)
    {
    // allocate the parameter storage
    m_params = std::vector<param_type, managed_allocator<param_type> >(m_pdata->getNTypes(), param_type(), managed_allocator<param_type>(m_exec_conf->isCUDAEnabled()));

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

    GlobalArray<hpmc_implicit_counters_t> implicit_count(this->m_pdata->getNTypes(),this->m_exec_conf);
    m_implicit_count.swap(implicit_count);
    TAG_ALLOCATION(m_implicit_count);

        {
        ArrayHandle<hpmc_implicit_counters_t> h_implicit_count(m_implicit_count, access_location::host, access_mode::readwrite);
        for (unsigned int i = 0; i < this->m_pdata->getNTypes(); ++i)
            h_implicit_count.data[i] = hpmc_implicit_counters_t();
        }

    m_implicit_count_run_start.resize(this->m_pdata->getNTypes());
    m_implicit_count_step_start.resize(this->m_pdata->getNTypes());

    m_fugacity.resize(this->m_pdata->getNTypes(),0.0);
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
    std::vector<hpmc_implicit_counters_t> result(this->m_pdata->getNTypes());

    std::copy(h_counters.data, h_counters.data + this->m_pdata->getNTypes(), result.begin());

    if (mode == 1)
        {
        for (unsigned int i = 0; i < this->m_pdata->getNTypes(); ++i)
            result[i] = result[i] - m_implicit_count_run_start[i];
        }
    else if (mode == 2)
        {
        for (unsigned int i = 0; i < this->m_pdata->getNTypes(); ++i)
            result[i] = result[i] - m_implicit_count_step_start[i];
        }

    #ifdef ENABLE_MPI
    if (this->m_comm)
        {
        // MPI Reduction to total result values on all ranks
        for (unsigned int i = 0; i < this->m_pdata->getNTypes(); ++i)
            MPI_Allreduce(MPI_IN_PLACE, &result[i].insert_count, 1, MPI_LONG_LONG_INT, MPI_SUM, this->m_exec_conf->getMPICommunicator());
        }
    #endif

    return result;
    }

template<class Shape>
std::vector< std::string > IntegratorHPMCMono<Shape>::getProvidedLogQuantities()
    {
    // start with the integrator provided quantities
    std::vector< std::string > result = IntegratorHPMC::getProvidedLogQuantities();
    // then add ours
    if(m_patch)
        {
        result.push_back("hpmc_patch_energy");
        result.push_back("hpmc_patch_rcut");
        }

    for (unsigned int typ=0; typ<this->m_pdata->getNTypes();typ++)
      {
      std::ostringstream tmp_str0;
      tmp_str0<<"hpmc_fugacity_"<< this->m_pdata->getNameByType(typ);
      result.push_back(tmp_str0.str());
      }

    result.push_back("hpmc_insert_count");

    return result;
    }

template<class Shape>
Scalar IntegratorHPMCMono<Shape>::getLogValue(const std::string& quantity, unsigned int timestep)
    {
    if (quantity == "hpmc_patch_energy")
        {
        if (m_patch)
            {
            return computePatchEnergy(timestep);
            }
        else
            {
            this->m_exec_conf->msg->error() << "No patch enabled:" << quantity << " not registered." << std::endl;
            throw std::runtime_error("Error getting log value");
            }
        }
    else if (quantity == "hpmc_patch_rcut")
        {
        if (m_patch)
            {
            return (Scalar)m_patch->getRCut();
            }
        else
            {
            this->m_exec_conf->msg->error() << "No patch enabled:" << quantity << " not registered." << std::endl;
            throw std::runtime_error("Error getting log value");
            }
        }

    //loop over per particle fugacities
    for (unsigned int typ=0; typ<this->m_pdata->getNTypes();typ++)
        {
        std::ostringstream tmp_str0;
        tmp_str0<<"hpmc_fugacity_"<<this->m_pdata->getNameByType(typ);
        if (quantity==tmp_str0.str())
            return m_fugacity[typ];
        }

    hpmc_counters_t counters = IntegratorHPMC::getCounters(2);
    const std::vector<hpmc_implicit_counters_t>& implicit_counters = getImplicitCounters(2);

    if (quantity == "hpmc_insert_count")
        {
        // reduce over all types
        unsigned long long int total_insert_count = 0;
        for (unsigned int i = 0; i < this->m_pdata->getNTypes(); ++i)
            total_insert_count += implicit_counters[i].insert_count;

        // return number of depletant insertions per colloid
        if (counters.getNMoves() > 0)
            return (Scalar)total_insert_count/(Scalar)counters.getNMoves();
        else
            return Scalar(0.0);
        }

    //nothing found -> pass on to base class
    return IntegratorHPMC::getLogValue(quantity, timestep);
    }

template <class Shape>
void IntegratorHPMCMono<Shape>::printStats()
    {
    IntegratorHPMC::printStats();

    const std::vector<hpmc_implicit_counters_t>& result = getImplicitCounters(1);
    hpmc_counters_t counters = getCounters(1);

    // reduce over all types
    unsigned long long int total_insert_count = 0;
    for (unsigned int i = 0; i < this->m_pdata->getNTypes(); ++i)
        total_insert_count += result[i].insert_count;

    bool has_depletants = false;
    for (unsigned int i = 0; i < this->m_pdata->getNTypes(); ++i)
        {
        if (m_fugacity[i] != 0.0)
            {
            has_depletants = true;
            break;
            }
        }

    if (!has_depletants)
        return;

    this->m_exec_conf->msg->notice(2) << "-- Implicit depletants stats:" << "\n";
    this->m_exec_conf->msg->notice(2) << "Depletant insertions per trial move:      "
        << double(total_insert_count)/double(counters.getNMoves()) << "\n";

    // supply additional statistics
    for (unsigned int i = 0; i < this->m_pdata->getNTypes(); ++i)
        {
        if (m_fugacity[i] != 0.0)
            {
            this->m_exec_conf->msg->notice(3) << "Type '" << this->m_pdata->getNameByType(i) << "': "
                << double(result[i].insert_count)/double(counters.getNMoves()) << std::endl;
            }
        }

    /*unsigned int max_height = 0;
    unsigned int total_height = 0;

    for (unsigned int i = 0; i < m_pdata->getN(); i++)
        {
        unsigned int height = m_aabb_tree.height(i);
        if (height > max_height)
            max_height = height;
        total_height += height;
        }

    m_exec_conf->msg->notice(2) << "Avg AABB tree height: " << total_height / Scalar(m_pdata->getN()) << std::endl;
    m_exec_conf->msg->notice(2) << "Max AABB tree height: " << max_height << std::endl;*/
    }

template <class Shape>
void IntegratorHPMCMono<Shape>::resetStats()
    {
    IntegratorHPMC::resetStats();

    ArrayHandle<hpmc_implicit_counters_t> h_counters(m_implicit_count, access_location::host, access_mode::read);
    for (unsigned int i = 0; i < this->m_pdata->getNTypes(); ++i)
        m_implicit_count_run_start[i] = h_counters.data[i];
    }

template <class Shape>
void IntegratorHPMCMono<Shape>::slotNumTypesChange()
    {
    // re-allocate the parameter storage
    m_params.resize(m_pdata->getNTypes());

    // skip the reallocation if the number of types does not change
    // this keeps old potential coefficients when restoring a snapshot
    // it will result in invalid coefficients if the snapshot has a different type id -> name mapping
    if (m_pdata->getNTypes() == m_overlap_idx.getW())
        return;

    // re-allocate overlap interaction matrix
    Index2D old_overlap_idx = m_overlap_idx;
    m_overlap_idx = Index2D(m_pdata->getNTypes());

    GlobalArray<unsigned int> overlaps(m_overlap_idx.getNumElements(), m_exec_conf);

        {
        ArrayHandle<unsigned int> h_old_overlaps(m_overlaps, access_location::host, access_mode::read);
        ArrayHandle<unsigned int> h_overlaps(overlaps, access_location::host, access_mode::overwrite);

        for(unsigned int i = 0; i < m_overlap_idx.getNumElements(); i++)
            {
            h_overlaps.data[i] = 1; // Assume we want to check overlaps.
            }

        // copy over old overlap check flags (this assumes the number of types is greater or equal to the old number of types)
        for (unsigned int i = 0; i < old_overlap_idx.getW(); ++i)
            {
            for (unsigned int j = 0; j < old_overlap_idx.getH(); ++j)
                {
                h_overlaps.data[m_overlap_idx(i,j)] = h_old_overlaps.data[old_overlap_idx(i,j)];
                }
            }
        }

    m_overlaps.swap(overlaps);

    // depletant related counters
    unsigned int old_ntypes = m_implicit_count.getNumElements();
    m_implicit_count.resize(this->m_pdata->getNTypes());

        {
        ArrayHandle<hpmc_implicit_counters_t> h_implicit_count(m_implicit_count, access_location::host, access_mode::readwrite);
        for (unsigned int i = old_ntypes; i < this->m_pdata->getNTypes(); ++i)
            h_implicit_count.data[i] = hpmc_implicit_counters_t();
        }
    m_implicit_count_run_start.resize(this->m_pdata->getNTypes());
    m_implicit_count_step_start.resize(this->m_pdata->getNTypes());

    // depletant fugacities
    m_fugacity.resize(this->m_pdata->getNTypes(),0.0);

    // call parent class method
    IntegratorHPMC::slotNumTypesChange();
    }

template <class Shape>
void IntegratorHPMCMono<Shape>::update(unsigned int timestep)
    {
    m_exec_conf->msg->notice(10) << "HPMCMono update: " << timestep << std::endl;
    IntegratorHPMC::update(timestep);

    // get needed vars
    ArrayHandle<hpmc_counters_t> h_counters(m_count_total, access_location::host, access_mode::readwrite);
    hpmc_counters_t& counters = h_counters.data[0];

    ArrayHandle<hpmc_implicit_counters_t> h_implicit_counters(m_implicit_count, access_location::host, access_mode::readwrite);
    std::copy(h_implicit_counters.data, h_implicit_counters.data + this->m_pdata->getNTypes(), m_implicit_count_step_start.begin());

    const BoxDim& box = m_pdata->getBox();
    unsigned int ndim = this->m_sysdef->getNDimensions();

    #ifdef ENABLE_MPI
    // compute the width of the active region
    Scalar3 npd = box.getNearestPlaneDistance();
    Scalar3 ghost_fraction = m_nominal_width / npd;
    #endif

    // Shuffle the order of particles for this step
    m_update_order.resize(m_pdata->getN());
    m_update_order.shuffle(timestep);

    // update the AABB Tree
    buildAABBTree();
    // limit m_d entries so that particles cannot possibly wander more than one box image in one time step
    limitMoveDistances();
    // update the image list
    updateImageList();

    bool has_depletants = false;
    for (unsigned int i = 0; i < this->m_pdata->getNTypes(); ++i)
        {
        if (m_fugacity[i] != 0.0)
            {
            has_depletants = true;
            break;
            }
        }

    // Combine the three seeds to generate RNG for poisson distribution
    #ifndef ENABLE_TBB
    hoomd::RandomGenerator rng_depletants(this->m_seed,
        timestep,
        this->m_exec_conf->getRank(),
        hoomd::RNGIdentifier::HPMCDepletants);
    #else
    // create one RNG per thread
    tbb::enumerable_thread_specific< hoomd::RandomGenerator > rng_depletants_parallel([=]
        {
        std::vector<unsigned int> seed_seq(5);
        std::hash<std::thread::id> hash;
        return hoomd::RandomGenerator(this->m_seed,
            timestep,
            this->m_exec_conf->getRank(),
            hash(std::this_thread::get_id()),
            hoomd::RNGIdentifier::HPMCDepletants);
        });
    #endif

    if (this->m_prof) this->m_prof->push(this->m_exec_conf, "HPMC update");

    if( m_external ) // I think we need this here otherwise I don't think it will get called.
        {
        m_external->compute(timestep);
        }

    // access interaction matrix
    ArrayHandle<unsigned int> h_overlaps(m_overlaps, access_location::host, access_mode::read);

    // loop over local particles nselect times
    for (unsigned int i_nselect = 0; i_nselect < m_nselect; i_nselect++)
        {
        // access particle data and system box
        ArrayHandle<Scalar4> h_postype(m_pdata->getPositions(), access_location::host, access_mode::readwrite);
        ArrayHandle<Scalar4> h_orientation(m_pdata->getOrientationArray(), access_location::host, access_mode::readwrite);
        ArrayHandle<Scalar> h_diameter(m_pdata->getDiameters(), access_location::host, access_mode::read);
        ArrayHandle<Scalar> h_charge(m_pdata->getCharges(), access_location::host, access_mode::read);

        //access move sizes
        ArrayHandle<Scalar> h_d(m_d, access_location::host, access_mode::read);
        ArrayHandle<Scalar> h_a(m_a, access_location::host, access_mode::read);

        // loop through N particles in a shuffled order
        for (unsigned int cur_particle = 0; cur_particle < m_pdata->getN(); cur_particle++)
            {
            unsigned int i = m_update_order[cur_particle];

            // read in the current position and orientation
            Scalar4 postype_i = h_postype.data[i];
            Scalar4 orientation_i = h_orientation.data[i];
            vec3<Scalar> pos_i = vec3<Scalar>(postype_i);

            #ifdef ENABLE_MPI
            if (m_comm)
                {
                // only move particle if active
                if (!isActive(make_scalar3(postype_i.x, postype_i.y, postype_i.z), box, ghost_fraction))
                    continue;
                }
            #endif

            // make a trial move for i
            hoomd::RandomGenerator rng_i(hoomd::RNGIdentifier::HPMCMonoTrialMove, m_seed, i, m_exec_conf->getRank()*m_nselect + i_nselect, timestep);
            int typ_i = __scalar_as_int(postype_i.w);
            Shape shape_i(quat<Scalar>(orientation_i), m_params[typ_i]);
            unsigned int move_type_select = hoomd::UniformIntDistribution(0xffff)(rng_i);
            bool move_type_translate = !shape_i.hasOrientation() || (move_type_select < m_move_ratio);

            Shape shape_old(quat<Scalar>(orientation_i), m_params[typ_i]);
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
                if (m_comm)
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
            OverlapReal r_cut_patch = 0;

            if (m_patch && !m_patch_log)
                {
                r_cut_patch = m_patch->getRCut() + 0.5*m_patch->getAdditiveCutoff(typ_i);
                }

            // subtract minimum AABB extent from search radius
            OverlapReal R_query = std::max(shape_i.getCircumsphereDiameter()/OverlapReal(2.0),
                r_cut_patch-getMinCoreDiameter()/(OverlapReal)2.0);
            detail::AABB aabb_i_local = detail::AABB(vec3<Scalar>(0,0,0),R_query);

            // patch + field interaction deltaU
            double patch_field_energy_diff = 0;

            // check for overlaps with neighboring particle's positions (also calculate the new energy)
            // All image boxes (including the primary)
            const unsigned int n_images = m_image_list.size();
            for (unsigned int cur_image = 0; cur_image < n_images; cur_image++)
                {
                vec3<Scalar> pos_i_image = pos_i + m_image_list[cur_image];
                detail::AABB aabb = aabb_i_local;
                aabb.translate(pos_i_image);

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

                                Scalar4 postype_j;
                                Scalar4 orientation_j;

                                // handle j==i situations
                                if ( j != i )
                                    {
                                    // load the position and orientation of the j particle
                                    postype_j = h_postype.data[j];
                                    orientation_j = h_orientation.data[j];
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
                                        orientation_j = quat_to_scalar4(shape_i.orientation);
                                        }
                                    }

                                // put particles in coordinate system of particle i
                                vec3<Scalar> r_ij = vec3<Scalar>(postype_j) - pos_i_image;

                                unsigned int typ_j = __scalar_as_int(postype_j.w);
                                Shape shape_j(quat<Scalar>(orientation_j), m_params[typ_j]);

                                Scalar rcut = 0.0;
                                if (m_patch)
                                    rcut = r_cut_patch + 0.5 * m_patch->getAdditiveCutoff(typ_j);

                                counters.overlap_checks++;
                                if (h_overlaps.data[m_overlap_idx(typ_i, typ_j)]
                                    && check_circumsphere_overlap(r_ij, shape_i, shape_j)
                                    && test_overlap(r_ij, shape_i, shape_j, counters.overlap_err_count))
                                    {
                                    overlap = true;
                                    break;
                                    }
                                else if (m_patch && !m_patch_log && dot(r_ij,r_ij) <= rcut*rcut) // If there is no overlap and m_patch is not NULL, calculate energy
                                    {
                                    // deltaU = U_old - U_new: subtract energy of new configuration
                                    patch_field_energy_diff -= m_patch->energy(r_ij, typ_i,
                                                               quat<float>(shape_i.orientation),
                                                               h_diameter.data[i],
                                                               h_charge.data[i],
                                                               typ_j,
                                                               quat<float>(orientation_j),
                                                               h_diameter.data[j],
                                                               h_charge.data[j]
                                                               );
                                    }
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

            // calculate old patch energy only if m_patch not NULL and no overlaps
            if (m_patch && !m_patch_log && !overlap)
                {
                for (unsigned int cur_image = 0; cur_image < n_images; cur_image++)
                    {
                    vec3<Scalar> pos_i_image = pos_old + m_image_list[cur_image];
                    detail::AABB aabb = aabb_i_local;
                    aabb.translate(pos_i_image);

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

                                    Scalar4 postype_j;
                                    Scalar4 orientation_j;

                                    // handle j==i situations
                                    if ( j != i )
                                        {
                                        // load the position and orientation of the j particle
                                        postype_j = h_postype.data[j];
                                        orientation_j = h_orientation.data[j];
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
                                            orientation_j = quat_to_scalar4(shape_old.orientation);
                                            }
                                        }

                                    // put particles in coordinate system of particle i
                                    vec3<Scalar> r_ij = vec3<Scalar>(postype_j) - pos_i_image;
                                    unsigned int typ_j = __scalar_as_int(postype_j.w);
                                    Shape shape_j(quat<Scalar>(orientation_j), m_params[typ_j]);

                                    Scalar rcut = r_cut_patch + 0.5 * m_patch->getAdditiveCutoff(typ_j);

                                    // deltaU = U_old - U_new: add energy of old configuration
                                    if (dot(r_ij,r_ij) <= rcut*rcut)
                                        patch_field_energy_diff += m_patch->energy(r_ij,
                                                                   typ_i,
                                                                   quat<float>(orientation_i),
                                                                   h_diameter.data[i],
                                                                   h_charge.data[i],
                                                                   typ_j,
                                                                   quat<float>(orientation_j),
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
                } // end if (m_patch)

            // Add external energetic contribution
            if (m_external)
                {
                patch_field_energy_diff -= m_external->energydiff(i, pos_old, shape_old, pos_i, shape_i);
                }

            bool accept = !overlap && hoomd::detail::generate_canonical<double>(rng_i) < slow::exp(patch_field_energy_diff);

            // The trial move is valid, so check if it is invalidated by depletants
            if (has_depletants && accept)
                {
                #ifndef ENABLE_TBB
                accept = checkDepletantOverlap(i, pos_i, shape_i, typ_i, h_postype.data, h_orientation.data, h_overlaps.data, counters, h_implicit_counters.data, rng_depletants);
                #else
                accept = checkDepletantOverlap(i, pos_i, shape_i, typ_i, h_postype.data, h_orientation.data, h_overlaps.data, counters, h_implicit_counters.data, rng_depletants_parallel);
                #endif
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
                detail::AABB aabb = aabb_i_local;
                aabb.translate(pos_i);
                m_aabb_tree.update(i, aabb);

                // update position of particle
                h_postype.data[i] = make_scalar4(pos_i.x,pos_i.y,pos_i.z,postype_i.w);

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
    if (m_comm)
        {
        ArrayHandle<Scalar4> h_postype(m_pdata->getPositions(), access_location::host, access_mode::readwrite);
        ArrayHandle<int3> h_image(m_pdata->getImages(), access_location::host, access_mode::readwrite);

        // precalculate the grid shift
        hoomd::RandomGenerator rng(hoomd::RNGIdentifier::HPMCMonoShift, this->m_seed, timestep);
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

    if (this->m_prof) this->m_prof->pop(this->m_exec_conf);

    // migrate and exchange particles
    communicate(true);

    // all particle have been moved, the aabb tree is now invalid
    m_aabb_tree_invalid = true;
    }

/*! \param timestep current step
    \param early_exit exit at first overlap found if true
    \returns number of overlaps if early_exit=false, 1 if early_exit=true
*/
template <class Shape>
unsigned int IntegratorHPMCMono<Shape>::countOverlaps(unsigned int timestep, bool early_exit)
    {
    unsigned int overlap_count = 0;
    unsigned int err_count = 0;

    m_exec_conf->msg->notice(10) << "HPMCMono count overlaps: " << timestep << std::endl;

    if (!m_past_first_run)
        {
        m_exec_conf->msg->error() << "count_overlaps only works after a run() command" << std::endl;
        throw std::runtime_error("Error communicating in count_overlaps");
        }

    // build an up to date AABB tree
    buildAABBTree();
    // update the image list
    updateImageList();

    if (this->m_prof) this->m_prof->push(this->m_exec_conf, "HPMC count overlaps");

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
        detail::AABB aabb_i_local = shape_i.getAABB(vec3<Scalar>(0,0,0));

        const unsigned int n_images = m_image_list.size();
        for (unsigned int cur_image = 0; cur_image < n_images; cur_image++)
            {
            vec3<Scalar> pos_i_image = pos_i + m_image_list[cur_image];
            detail::AABB aabb = aabb_i_local;
            aabb.translate(pos_i_image);

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

    if (this->m_prof) this->m_prof->pop(this->m_exec_conf);

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
float IntegratorHPMCMono<Shape>::computePatchEnergy(unsigned int timestep)
    {
    // sum up in double precision
    double energy = 0.0;

    // return if nothing to do
    if (!m_patch) return energy;

    m_exec_conf->msg->notice(10) << "HPMC compute patch energy: " << timestep << std::endl;

    if (!m_past_first_run)
        {
        m_exec_conf->msg->error() << "get_patch_energy only works after a run() command" << std::endl;
        throw std::runtime_error("Error communicating in count_overlaps");
        }

    // build an up to date AABB tree
    buildAABBTree();
    // update the image list
    updateImageList();

    if (this->m_prof) this->m_prof->push(this->m_exec_conf, "HPMC compute patch energy");

    // access particle data and system box
    ArrayHandle<Scalar4> h_postype(m_pdata->getPositions(), access_location::host, access_mode::read);
    ArrayHandle<Scalar4> h_orientation(m_pdata->getOrientationArray(), access_location::host, access_mode::read);
    ArrayHandle<Scalar> h_diameter(m_pdata->getDiameters(), access_location::host, access_mode::read);
    ArrayHandle<Scalar> h_charge(m_pdata->getCharges(), access_location::host, access_mode::read);
    ArrayHandle<unsigned int> h_tag(m_pdata->getTags(), access_location::host, access_mode::read);

    // access parameters and interaction matrix
    ArrayHandle<unsigned int> h_overlaps(m_overlaps, access_location::host, access_mode::read);

    // Loop over all particles
    #ifdef ENABLE_TBB
    energy = tbb::parallel_reduce(tbb::blocked_range<unsigned int>(0, m_pdata->getN()),
        0.0f,
        [&](const tbb::blocked_range<unsigned int>& r, float energy)->float {
        for (unsigned int i = r.begin(); i != r.end(); ++i)
    #else
    for (unsigned int i = 0; i < m_pdata->getN(); i++)
    #endif
        {
        // read in the current position and orientation
        Scalar4 postype_i = h_postype.data[i];
        Scalar4 orientation_i = h_orientation.data[i];
        unsigned int typ_i = __scalar_as_int(postype_i.w);
        Shape shape_i(quat<Scalar>(orientation_i), m_params[typ_i]);
        vec3<Scalar> pos_i = vec3<Scalar>(postype_i);

        Scalar d_i = h_diameter.data[i];
        Scalar charge_i = h_charge.data[i];

        // the cut-off
        float r_cut = m_patch->getRCut() + 0.5*m_patch->getAdditiveCutoff(typ_i);

        // subtract minimum AABB extent from search radius
        OverlapReal R_query = std::max(shape_i.getCircumsphereDiameter()/OverlapReal(2.0),
            r_cut-getMinCoreDiameter()/(OverlapReal)2.0);
        detail::AABB aabb_i_local = detail::AABB(vec3<Scalar>(0,0,0),R_query);

        const unsigned int n_images = m_image_list.size();
        for (unsigned int cur_image = 0; cur_image < n_images; cur_image++)
            {
            vec3<Scalar> pos_i_image = pos_i + m_image_list[cur_image];
            detail::AABB aabb = aabb_i_local;
            aabb.translate(pos_i_image);

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

                            // skip i==j in the 0 image
                            if (cur_image == 0 && i == j)
                                continue;

                            Scalar4 postype_j = h_postype.data[j];
                            Scalar4 orientation_j = h_orientation.data[j];
                            Scalar d_j = h_diameter.data[j];
                            Scalar charge_j = h_charge.data[j];

                            // put particles in coordinate system of particle i
                            vec3<Scalar> r_ij = vec3<Scalar>(postype_j) - pos_i_image;

                            unsigned int typ_j = __scalar_as_int(postype_j.w);
                            Shape shape_j(quat<Scalar>(orientation_j), m_params[typ_j]);

                            // count unique pairs within range
                            Scalar rcut_ij = r_cut + 0.5*m_patch->getAdditiveCutoff(typ_j);

                            if (h_tag.data[i] <= h_tag.data[j] && dot(r_ij,r_ij) <= rcut_ij*rcut_ij)
                                {
                                energy += m_patch->energy(r_ij,
                                       typ_i,
                                       quat<float>(orientation_i),
                                       d_i,
                                       charge_i,
                                       typ_j,
                                       quat<float>(orientation_j),
                                       d_j,
                                       charge_j);
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
    #ifdef ENABLE_TBB
    return energy;
    }, [](float x, float y)->float { return x+y; } );
    #endif

    if (this->m_prof) this->m_prof->pop(this->m_exec_conf);

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
OverlapReal IntegratorHPMCMono<Shape>::getMinCoreDiameter()
    {
    // for each type, create a temporary shape and return the minimum diameter
    OverlapReal minD = OverlapReal(0.0);
    for (unsigned int typ = 0; typ < this->m_pdata->getNTypes(); typ++)
        {
        Shape temp(quat<Scalar>(), m_params[typ]);
        minD = std::min(minD, temp.getCircumsphereDiameter());
        }

    if (m_patch)
        {
        OverlapReal max_extent = 0.0;
        for (unsigned int typ =0; typ < this->m_pdata->getNTypes(); typ++)
            max_extent = std::max(max_extent, (OverlapReal) m_patch->getAdditiveCutoff(typ));
        minD = std::max((OverlapReal) 0.0, minD-max_extent);
        }

    return minD;
    }

template <class Shape>
void IntegratorHPMCMono<Shape>::setParam(unsigned int typ,  const param_type& param)
    {
    // validate input
    if (typ >= this->m_pdata->getNTypes())
        {
        this->m_exec_conf->msg->error() << "integrate.mode_hpmc_?." << /*evaluator::getName() <<*/ ": Trying to set pair params for a non existent type! "
                  << typ << std::endl;
        throw std::runtime_error("Error setting parameters in IntegratorHPMCMono");
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
void IntegratorHPMCMono<Shape>::setOverlapChecks(unsigned int typi, unsigned int typj, bool check_overlaps)
    {
    // validate input
    if (typi >= this->m_pdata->getNTypes())
        {
        this->m_exec_conf->msg->error() << "integrate.mode_hpmc_?." << /*evaluator::getName() <<*/ ": Trying to set interaction matrix for a non existent type! "
                  << typi << std::endl;
        throw std::runtime_error("Error setting interaction matrix in IntegratorHPMCMono");
        }

    if (typj >= this->m_pdata->getNTypes())
        {
        this->m_exec_conf->msg->error() << "integrate.mode_hpmc_?." << /*evaluator::getName() <<*/ ": Trying to set interaction matrix for a non existent type! "
                  << typj << std::endl;
        throw std::runtime_error("Error setting interaction matrix in IntegratorHPMCMono");
        }

    // update the parameter for this type
    m_exec_conf->msg->notice(7) << "setOverlapChecks : " << typi << " " << typj << " " << check_overlaps << std::endl;
    ArrayHandle<unsigned int> h_overlaps(m_overlaps, access_location::host, access_mode::readwrite);
    h_overlaps.data[m_overlap_idx(typi,typj)] = check_overlaps;
    h_overlaps.data[m_overlap_idx(typj,typi)] = check_overlaps;

    m_image_list_valid = false;
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

    if (m_prof) m_prof->push(m_exec_conf, "HPMC image list");

    unsigned int ndim = m_sysdef->getNDimensions();

    m_image_list_valid = true;
    m_image_list_is_initialized = true;
    m_image_list.clear();
    m_image_hkl.clear();
    m_image_list_rebuilds++;

    // Get box vectors
    const BoxDim& box = m_pdata->getGlobalBox();
    vec3<Scalar> e1 = vec3<Scalar>(box.getLatticeVector(0));
    vec3<Scalar> e2 = vec3<Scalar>(box.getLatticeVector(1));
    // 2D simulations don't necessarily have a zero-size z-dimension, but it is convenient for us if we assume one.
    vec3<Scalar> e3(0,0,0);
    if (ndim == 3)
        e3 = vec3<Scalar>(box.getLatticeVector(2));

    // Maximum interaction range is the sum of the system box circumsphere diameter and the max particle circumsphere diameter and move distance
    Scalar range = 0.0f;
    // Try four linearly independent body diagonals and find the longest
    vec3<Scalar> body_diagonal;
    body_diagonal = e1 - e2 - e3;
    range = detail::max(range, dot(body_diagonal, body_diagonal));
    body_diagonal = e1 - e2 + e3;
    range = detail::max(range, dot(body_diagonal, body_diagonal));
    body_diagonal = e1 + e2 - e3;
    range = detail::max(range, dot(body_diagonal, body_diagonal));
    body_diagonal = e1 + e2 + e3;
    range = detail::max(range, dot(body_diagonal, body_diagonal));
    range = fast::sqrt(range);

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
            if (m_patch)
                r_cut_patch_i = (Scalar)m_patch->getRCut() + 0.5*m_patch->getAdditiveCutoff(typ_i);

            Scalar range_i(0.0);
            for (unsigned int typ_j = 0; typ_j < this->m_pdata->getNTypes(); typ_j++)
                {
                Scalar r_cut_patch_ij(0.0);
                if (m_patch)
                    r_cut_patch_ij = r_cut_patch_i + 0.5*m_patch->getAdditiveCutoff(typ_j);

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

    range += max_trans_d_and_diam;

    // add any extra requested width
    range += m_extra_image_width;

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

        #ifdef ENABLE_MPI
        if (m_pdata->getDomainDecomposition())
            {
            Index3D di = m_pdata->getDomainDecomposition()->getDomainIndexer();
            if (di.getW() > 1) x_max = 0;
            if (di.getH() > 1) y_max = 0;
            if (di.getD() > 1) z_max = 0;
            }
        #endif

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

    // cout << "built image list" << std::endl;
    // for (unsigned int i = 0; i < m_image_list.size(); i++)
    //     cout << m_image_list[i].x << " " << m_image_list[i].y << " " << m_image_list[i].z << std::endl;
    // cout << std::endl;

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

    m_exec_conf->msg->notice(8) << "Updated image list: " << m_image_list.size() << " images" << std::endl;
    if (m_prof) m_prof->pop();

    return m_image_list;
    }

template <class Shape>
void IntegratorHPMCMono<Shape>::updateCellWidth()
    {
    this->m_nominal_width = this->getMaxCoreDiameter();

    Scalar max_d(0.0);

    for (unsigned int type = 0; type < this->m_pdata->getNTypes(); ++type)
        {
        if (m_fugacity[type] != Scalar(0.0))
            {
            // add range of depletion interaction
            quat<Scalar> o;
            Shape tmp(o, this->m_params[type]);
            max_d = std::max(max_d, (Scalar) tmp.getCircumsphereDiameter());
            }
        }

    // extend the image list by the depletant diameter, since we're querying
    // AABBs that are larger than the shape diameters themselves
    this->m_extra_image_width = m_quermass ? 2.0*m_sweep_radius : max_d;

    this->m_nominal_width += this->m_extra_image_width;

    // Account for patch width
    if (this->m_patch)
        {
        Scalar max_extent = 0.0;
        for (unsigned int typ = 0; typ < this->m_pdata->getNTypes(); typ++)
            {
            max_extent = std::max(max_extent, this->m_patch->getAdditiveCutoff(typ));
            }

        this->m_nominal_width = std::max(this->m_nominal_width, this->m_patch->getRCut() + max_extent);
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

        int retval = posix_memalign((void**)&m_aabbs, 32, N*sizeof(detail::AABB));
        if (retval != 0)
            {
            m_exec_conf->msg->errorAllRanks() << "Error allocating aligned memory" << std::endl;
            throw std::runtime_error("Error allocating AABB memory");
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
const detail::AABBTree& IntegratorHPMCMono<Shape>::buildAABBTree()
    {
    if (m_aabb_tree_invalid)
        {
        m_exec_conf->msg->notice(8) << "Building AABB tree: " << m_pdata->getN() << " ptls " << m_pdata->getNGhosts() << " ghosts" << std::endl;
        if (this->m_prof) this->m_prof->push(this->m_exec_conf, "AABB tree build");
        // build the AABB tree
            {
            ArrayHandle<Scalar4> h_postype(m_pdata->getPositions(), access_location::host, access_mode::read);
            ArrayHandle<Scalar4> h_orientation(m_pdata->getOrientationArray(), access_location::host, access_mode::read);

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

                    if (!this->m_patch)
                        m_aabbs[i] = shape.getAABB(vec3<Scalar>(h_postype.data[i]));
                    else
                        {
                        Scalar radius = std::max(0.5*shape.getCircumsphereDiameter(),
                            0.5*this->m_patch->getAdditiveCutoff(typ_i));
                        m_aabbs[i] = detail::AABB(vec3<Scalar>(h_postype.data[i]), radius);
                        }
                    }
                m_aabb_tree.buildTree(m_aabbs, n_aabb);
                }
            }

        if (this->m_prof) this->m_prof->pop(this->m_exec_conf);
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
    Scalar min_npd = detail::min((Scalar) npd_global.x, (Scalar) npd_global.y);
    if (this->m_sysdef->getNDimensions() == 3)
        {
        min_npd = detail::min(min_npd, (Scalar) npd_global.z);
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
std::vector<bool> IntegratorHPMCMono<Shape>::mapOverlaps()
    {
    #ifdef ENABLE_MPI
    if (m_pdata->getDomainDecomposition())
        {
        m_exec_conf->msg->error() << "map_overlaps does not support MPI parallel jobs" << std::endl;
        throw std::runtime_error("map_overlaps does not support MPI parallel jobs");
        }
    #endif

    unsigned int N = m_pdata->getN();

    std::vector<bool> overlap_map(N*N, false);

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
        detail::AABB aabb_i_local = shape_i.getAABB(vec3<Scalar>(0,0,0));

        const unsigned int n_images = m_image_list.size();
        for (unsigned int cur_image = 0; cur_image < n_images; cur_image++)
            {
            vec3<Scalar> pos_i_image = pos_i + m_image_list[cur_image];
            detail::AABB aabb = aabb_i_local;
            aabb.translate(pos_i_image);

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
                                overlap_map[h_tag.data[j]+N*h_tag.data[i]] = true;
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
    return overlap_map;
    }

/*! Calculate energy between all pairs of particles
 */
template <class Shape>
std::vector<float> IntegratorHPMCMono<Shape>::mapEnergies()
    {
    if (! m_patch)
        {
        throw std::runtime_error("No patch energy defined.\n");
        }

    #ifdef ENABLE_MPI
    if (m_pdata->getDomainDecomposition())
        {
        throw std::runtime_error("map_energies does not support MPI parallel jobs");
        }
    #endif

    unsigned int N = m_pdata->getN();

    std::vector<float> energy_map(N*N, 0.0);

    m_exec_conf->msg->notice(10) << "HPMC energy mapping" << std::endl;

    // build an up to date AABB tree
    buildAABBTree();
    // update the image list
    updateImageList();

    // access particle data and system box
    ArrayHandle<Scalar4> h_postype(m_pdata->getPositions(), access_location::host, access_mode::read);
    ArrayHandle<Scalar4> h_orientation(m_pdata->getOrientationArray(), access_location::host, access_mode::read);
    ArrayHandle<unsigned int> h_tag(m_pdata->getTags(), access_location::host, access_mode::read);
    ArrayHandle<Scalar> h_diameter(m_pdata->getDiameters(), access_location::host, access_mode::read);
    ArrayHandle<Scalar> h_charge(m_pdata->getCharges(), access_location::host, access_mode::read);

    // Loop over all particles
    #ifdef ENABLE_TBB
    tbb::parallel_for(tbb::blocked_range<unsigned int>(0, N),
        [&](const tbb::blocked_range<unsigned int>& r) {
    for (unsigned int i = r.begin(); i != r.end(); ++i)
    #else
    for (unsigned int i = 0; i < N; i++)
    #endif
        {
        // read in the current position and orientation
        Scalar4 postype_i = h_postype.data[i];
        Scalar4 orientation_i = h_orientation.data[i];
        Shape shape_i(quat<Scalar>(orientation_i), m_params[__scalar_as_int(postype_i.w)]);
        vec3<Scalar> pos_i = vec3<Scalar>(postype_i);
        unsigned int typ_i = __scalar_as_int(postype_i.w);

        Scalar diameter_i = h_diameter.data[i];
        Scalar charge_i = h_charge.data[i];

        // Check particle against AABB tree for neighbors
        Scalar r_cut_patch = m_patch->getRCut() + 0.5*m_patch->getAdditiveCutoff(typ_i);
        OverlapReal R_query = r_cut_patch-getMinCoreDiameter()/(OverlapReal)2.0;
        detail::AABB aabb_i_local = detail::AABB(vec3<Scalar>(0,0,0),R_query);

        const unsigned int n_images = m_image_list.size();
        for (unsigned int cur_image = 0; cur_image < n_images; cur_image++)
            {
            vec3<Scalar> pos_i_image = pos_i + m_image_list[cur_image];
            detail::AABB aabb = aabb_i_local;
            aabb.translate(pos_i_image);

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

                            // skip i==j in the 0 image
                            if (cur_image == 0 && i == j)
                                {
                                continue;
                                }

                            Scalar4 postype_j = h_postype.data[j];
                            Scalar4 orientation_j = h_orientation.data[j];

                            // put particles in coordinate system of particle i
                            vec3<Scalar> r_ij = vec3<Scalar>(postype_j) - pos_i_image;
                            unsigned int typ_j = __scalar_as_int(postype_j.w);

                            Scalar rcut = r_cut_patch + 0.5 * m_patch->getAdditiveCutoff(typ_j);

                            if (dot(r_ij,r_ij) <= rcut*rcut)
                                {
                                energy_map[h_tag.data[j]+N*h_tag.data[i]] +=
                                    m_patch->energy(r_ij,
                                       typ_i,
                                       quat<float>(shape_i.orientation),
                                       diameter_i,
                                       charge_i,
                                       typ_j,
                                       quat<float>(orientation_j),
                                       h_diameter.data[j],
                                       h_charge.data[j]
                                       );
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
    #ifdef ENABLE_TBB
        });
    #endif
    return energy_map;
    }

/*! Function for returning a python list of all overlaps in a system by particle
  tag. returns an unraveled form of an NxN matrix with true/false indicating
  the overlap status of the ith and jth particle
 */
template <class Shape>
pybind11::list IntegratorHPMCMono<Shape>::PyMapOverlaps()
    {
    std::vector<bool> v = IntegratorHPMCMono<Shape>::mapOverlaps();
    pybind11::list overlap_map;
    // for( unsigned int i = 0; i < sizeof(v)/sizeof(v[0]); i++ )
    for (auto i: v)
        {
        overlap_map.append(pybind11::cast<bool>(i));
        }
    return overlap_map;
    }

/*! Function for returning a python list of all overlaps in a system by particle
  tag. returns an unraveled form of an NxN matrix with true/false indicating
  the overlap status of the ith and jth particle
 */
template <class Shape>
pybind11::list IntegratorHPMCMono<Shape>::PyMapEnergies()
    {
    std::vector<float> v = IntegratorHPMCMono<Shape>::mapEnergies();
    pybind11::list energy_map;
    // for( unsigned int i = 0; i < sizeof(v)/sizeof(v[0]); i++ )
    for (auto i: v)
        {
        energy_map.append(pybind11::cast<float>(i));
        }
    return energy_map;
    }

template <class Shape>
void IntegratorHPMCMono<Shape>::connectGSDStateSignal(
                                                    std::shared_ptr<GSDDumpWriter> writer,
                                                    std::string name)
    {
    typedef hoomd::detail::SharedSignalSlot<int(gsd_handle&)> SlotType;
    auto func = std::bind(&IntegratorHPMCMono<Shape>::slotWriteGSDState, this, std::placeholders::_1, name);
    std::shared_ptr<hoomd::detail::SignalSlot> pslot( new SlotType(writer->getWriteSignal(), func));
    addSlot(pslot);
    }

template <class Shape>
void IntegratorHPMCMono<Shape>::connectGSDShapeSpec(std::shared_ptr<GSDDumpWriter> writer)
    {
    typedef hoomd::detail::SharedSignalSlot<int(gsd_handle&)> SlotType;
    auto func = std::bind(&IntegratorHPMCMono<Shape>::slotWriteGSDShapeSpec, this, std::placeholders::_1);
    std::shared_ptr<hoomd::detail::SignalSlot> pslot( new SlotType(writer->getWriteSignal(), func));
    addSlot(pslot);
    }

template <class Shape>
int IntegratorHPMCMono<Shape>::slotWriteGSDState( gsd_handle& handle, std::string name ) const
    {
    m_exec_conf->msg->notice(10) << "IntegratorHPMCMono writing to GSD File to name: "<< name << std::endl;
    int retval = 0;
    // create schema helpers
    #ifdef ENABLE_MPI
    bool mpi=(bool)m_pdata->getDomainDecomposition();
    #else
    bool mpi=false;
    #endif
    gsd_schema_hpmc schema(m_exec_conf, mpi);
    gsd_shape_schema<typename Shape::param_type> schema_shape(m_exec_conf, mpi);

    // access parameters
    ArrayHandle<Scalar> h_d(m_d, access_location::host, access_mode::read);
    ArrayHandle<Scalar> h_a(m_a, access_location::host, access_mode::read);
    schema.write(handle, "state/hpmc/integrate/d", m_pdata->getNTypes(), h_d.data, GSD_TYPE_DOUBLE);
    if(m_hasOrientation)
        {
        schema.write(handle, "state/hpmc/integrate/a", m_pdata->getNTypes(), h_a.data, GSD_TYPE_DOUBLE);
        }
    retval |= schema_shape.write(handle, name, m_pdata->getNTypes(), m_params);

    return retval;
    }

template <class Shape>
int IntegratorHPMCMono<Shape>::slotWriteGSDShapeSpec(gsd_handle& handle) const
    {
    GSDShapeSpecWriter shapespec(m_exec_conf);
    m_exec_conf->msg->notice(10) << "IntegratorHPMCMono writing to GSD File to name: " << shapespec.getName() << std::endl;
    int retval = shapespec.write(handle, this->getTypeShapeMapping(m_params));
    return retval;
    }

template <class Shape>
bool IntegratorHPMCMono<Shape>::restoreStateGSD( std::shared_ptr<GSDReader> reader, std::string name)
    {
    bool success = true;
    m_exec_conf->msg->notice(10) << "IntegratorHPMCMono from GSD File to name: "<< name << std::endl;
    uint64_t frame = reader->getFrame();
    // create schemas
    #ifdef ENABLE_MPI
    bool mpi=(bool)m_pdata->getDomainDecomposition();
    #else
    bool mpi=false;
    #endif
    gsd_schema_hpmc schema(m_exec_conf, mpi);
    gsd_shape_schema<typename Shape::param_type> schema_shape(m_exec_conf, mpi);

    ArrayHandle<Scalar> h_d(m_d, access_location::host, access_mode::readwrite);
    ArrayHandle<Scalar> h_a(m_a, access_location::host, access_mode::readwrite);
    schema.read(reader, frame, "state/hpmc/integrate/d", m_pdata->getNTypes(), h_d.data, GSD_TYPE_DOUBLE);
    if(m_hasOrientation)
        {
        schema.read(reader, frame, "state/hpmc/integrate/a", m_pdata->getNTypes(), h_a.data, GSD_TYPE_DOUBLE);
        }
    schema_shape.read(reader, frame, name, m_pdata->getNTypes(), m_params);
    return success;
    }

template<class Shape>
bool IntegratorHPMCMono<Shape>::py_test_overlap(unsigned int type_i, unsigned int type_j,
    pybind11::list rij, pybind11::list qi, pybind11::list qj,
    bool use_images, bool exclude_self)
    {
    if (len(rij) != 3)
        throw std::runtime_error("rij needs to be a 3d vector.\n");
    if (len(qi) != 4 || len(qj) != 4)
        throw std::runtime_error("qi and qj need to be quaternions.\n");

    assert(type_i <= m_pdata->getNTypes());
    assert(type_j <= m_pdata->getNTypes());

    vec3<Scalar> dr(pybind11::cast<Scalar>(rij[0]), pybind11::cast<Scalar>(rij[1]), pybind11::cast<Scalar>(rij[2]));
    quat<Scalar> quat_i(pybind11::cast<Scalar>(qi[0]),
        vec3<Scalar>(pybind11::cast<Scalar>(qi[1]), pybind11::cast<Scalar>(qi[2]), pybind11::cast<Scalar>(qi[3])));
    quat<Scalar> quat_j(pybind11::cast<Scalar>(qj[0]),
        vec3<Scalar>(pybind11::cast<Scalar>(qj[1]), pybind11::cast<Scalar>(qj[2]), pybind11::cast<Scalar>(qj[3])));

    Shape shape_i(quat_i, m_params[type_i]);
    Shape shape_j(quat_j, m_params[type_j]);

    unsigned int err = 0;
    bool overlap = false;
    if (use_images)
        {
        #ifdef ENABLE_MPI
        if (m_pdata->getDomainDecomposition())
            {
            this->m_exec_conf->msg->error() << "test_overlap does not support MPI parallel jobs with use_images=True" << std::endl;
            throw std::runtime_error("test_overlap does not support MPI parallel jobs");
            }
        #endif

        updateImageList();

        const unsigned int n_images = m_image_list.size();
        for (unsigned int cur_image = 0; cur_image < n_images; cur_image++)
            {
            if (exclude_self && cur_image == 0)
                continue;

            if (check_circumsphere_overlap(dr + m_image_list[cur_image], shape_i, shape_j) &&
                test_overlap(dr + m_image_list[cur_image], shape_i, shape_j, err))
                overlap = true;
            }
        }
    else
        {
        overlap = check_circumsphere_overlap(dr, shape_i, shape_j) && test_overlap(dr, shape_i, shape_j, err);
        }

    if (err)
        m_exec_conf->msg->warning() << "test_overlap() reports an error due to finite numerical precision." << std::endl;

    return overlap;
    }

/*! \param i The particle id in the list
    \param pos_i Particle position being tested
    \param shape_i Particle shape (including orientation) being tested
    \param typ_i Type of the particle being tested
    \param h_postype Pointer to GPUArray containing particle positions
    \param h_orientation Pointer to GPUArray containing particle orientations
    \param h_overlaps Pointer to GPUArray containing interaction matrix
    \param hpmc_counters_t&  Pointer to current counters
    \param hpmc_implicit_counters_t&  Pointer to current implicit counters
    \param rng_depletants The RNG used within this algorithm
    \param rng_i The RNG used for evaluating the Metropolis criterion

    In order to determine whether or not moves are accepted, particle positions are checked against a randomly generated set of depletant positions.

    NOTE: To avoid numerous acquires and releases of GPUArrays, data pointers are passed directly into this const function.
    */
#ifndef ENABLE_TBB
template<class Shape>
inline bool IntegratorHPMCMono<Shape>::checkDepletantOverlap(unsigned int i, vec3<Scalar> pos_i,
    Shape shape_i, unsigned int typ_i, Scalar4 *h_postype, Scalar4 *h_orientation, unsigned int *h_overlaps,
    hpmc_counters_t& counters, hpmc_implicit_counters_t *implicit_counters, hoomd::RandomGenerator& rng_depletants)
#else
template<class Shape>
inline bool IntegratorHPMCMono<Shape>::checkDepletantOverlap(unsigned int i, vec3<Scalar> pos_i,
    Shape shape_i, unsigned int typ_i, Scalar4 *h_postype, Scalar4 *h_orientation, unsigned int *h_overlaps,
    hpmc_counters_t& counters, hpmc_implicit_counters_t *implicit_counters,
    tbb::enumerable_thread_specific< hoomd::RandomGenerator >& rng_depletants_parallel)
#endif
    {
    bool accept = true;

    const unsigned int n_images = this->m_image_list.size();
    unsigned int ndim = this->m_sysdef->getNDimensions();

    Shape shape_old(quat<Scalar>(h_orientation[i]), this->m_params[typ_i]);
    detail::AABB aabb_i_local = shape_i.getAABB(vec3<Scalar>(0,0,0));
    detail::AABB aabb_i_local_old = shape_old.getAABB(vec3<Scalar>(0,0,0));

    #ifdef ENABLE_TBB
    std::vector< tbb::enumerable_thread_specific<hpmc_implicit_counters_t> > thread_implicit_counters(this->m_pdata->getNTypes());
    tbb::enumerable_thread_specific<hpmc_counters_t> thread_counters;
    #endif

    #ifdef ENABLE_TBB
    try {
    #endif
    for (unsigned int type = 0; type < this->m_pdata->getNTypes(); ++type)
        {
        if (m_fugacity[type] == 0.0 || (!h_overlaps[this->m_overlap_idx(type, typ_i)] && !m_quermass))
            continue;

        std::vector<unsigned int> intersect_i;
        std::vector<unsigned int> image_i;
        std::vector<detail::AABB> aabbs_i;

        if (accept && m_fugacity[type] > 0.0)
            {
            // find neighbors whose circumspheres overlap particle i's circumsphere in the old configuration
            // Here, circumsphere refers to the sphere around the depletant-excluded volume

            Shape tmp(quat<Scalar>(), this->m_params[type]);
            Scalar d_dep = tmp.getCircumsphereDiameter();

            Scalar range = m_quermass ? Scalar(2.0)*m_sweep_radius : d_dep;

            // get old AABB and extend
            vec3<Scalar> lower = aabb_i_local_old.getLower();
            vec3<Scalar> upper = aabb_i_local_old.getUpper();
            lower.x -= range; lower.y -= range; lower.z -= range;
            upper.x += range; upper.y += range; upper.z += range;
            detail::AABB aabb_local = detail::AABB(lower,upper);

            vec3<Scalar> pos_i_old(h_postype[i]);

            // All image boxes (including the primary)
            for (unsigned int cur_image = 0; cur_image < n_images; cur_image++)
                {
                vec3<Scalar> pos_i_old_image = pos_i_old + this->m_image_list[cur_image];
                detail::AABB aabb = aabb_local;
                aabb.translate(pos_i_old_image);

                // stackless search
                for (unsigned int cur_node_idx = 0; cur_node_idx < this->m_aabb_tree.getNumNodes(); cur_node_idx++)
                    {
                    if (detail::overlap(this->m_aabb_tree.getNodeAABB(cur_node_idx), aabb))
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
                                vec3<Scalar> r_ij = pos_j - pos_i_old_image;

                                unsigned int typ_j = __scalar_as_int(postype_j.w);
                                Shape shape_j(quat<Scalar>(), this->m_params[typ_j]);
                                if (shape_j.hasOrientation())
                                    shape_j.orientation = quat<Scalar>(h_orientation[j]);

                                detail::AABB aabb_j = shape_j.getAABB(pos_j);

                                // check AABB overlap
                                bool overlap_excluded = detail::overlap(aabb, aabb_j);

                                // currently ignore overlap flags with quermass==true
                                overlap_excluded = overlap_excluded && (m_quermass || h_overlaps[this->m_overlap_idx(type,typ_j)]);

                                if (Shape::supportsSweepRadius() && overlap_excluded)
                                    {
                                    // do i and j overlap with their excluded volumes in the old configuration?
                                    #ifdef ENABLE_TBB
                                    thread_counters.local().overlap_checks++;
                                    #else
                                    counters.overlap_checks++;
                                    #endif

                                    unsigned int err = 0;
                                    Scalar sweep_radius = m_quermass ? m_sweep_radius : Scalar(0.5)*tmp.getCircumsphereDiameter();
                                    overlap_excluded = test_overlap(r_ij, shape_old, shape_j, err, sweep_radius, sweep_radius);

                                    if (err)
                                    #ifdef ENABLE_TBB
                                        thread_counters.local().overlap_err_count++;
                                    #else
                                        counters.overlap_err_count++;
                                    #endif
                                    }
                                if (overlap_excluded)
                                    {
                                    intersect_i.push_back(j);
                                    image_i.push_back(cur_image);

                                    // cache the translated AABB of particle j. If i's image is cur_image, then j's
                                    // image is the negative of that (and we use i's untranslated position below)
                                    aabb_j.translate(-this->m_image_list[cur_image]);
                                    aabbs_i.push_back(aabb_j);
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

            // now, we have a list of intersecting spheres, sample in the union of intersection volumes
            // we sample from their union by checking if any generated position falls in the intersection
            // of their AABBs, only accepting it if it was generated from neighbor j_min

            // world AABB of particle i
            detail::AABB aabb_i_old = aabb_i_local_old;
            aabb_i_old.translate(pos_i_old);

            // extend AABB i by sweep radius
            range = Scalar(0.5)*d_dep+m_sweep_radius;
            vec3<Scalar> lower_i = aabb_i_old.getLower();
            vec3<Scalar> upper_i = aabb_i_old.getUpper();
            lower_i.x -= range; lower_i.y -= range; lower_i.z -= range;
            upper_i.x += range; upper_i.y += range; upper_i.z += range;

            // for every pairwise intersection
            #ifdef ENABLE_TBB
            tbb::parallel_for(tbb::blocked_range<unsigned int>(0, (unsigned int)intersect_i.size()),
                [=, &intersect_i, &image_i, &aabbs_i,
                    &accept, &rng_depletants_parallel,
                    &thread_counters, &thread_implicit_counters](const tbb::blocked_range<unsigned int>& s) {
            for (unsigned int k = s.begin(); k != s.end(); ++k)
            #else
            for (unsigned int k = 0; k < intersect_i.size(); ++k)
            #endif
                {
                // world AABB of shape j, in image of i
                detail::AABB aabb_j = aabbs_i[k];

                // extend AABB j by sweep radius
                vec3<Scalar> lower_j = aabb_j.getLower();
                vec3<Scalar> upper_j = aabb_j.getUpper();
                lower_j.x -= range; lower_j.y -= range; lower_j.z -= range;
                upper_j.x += range; upper_j.y += range; upper_j.z += range;

                // we already know the AABBs are overlapping, compute their intersection
                vec3<Scalar> intersect_lower, intersect_upper;
                intersect_lower.x = std::max(lower_i.x, lower_j.x);
                intersect_lower.y = std::max(lower_i.y, lower_j.y);
                intersect_lower.z = std::max(lower_i.z, lower_j.z);
                intersect_upper.x = std::min(upper_i.x, upper_j.x);
                intersect_upper.y = std::min(upper_i.y, upper_j.y);
                intersect_upper.z = std::min(upper_i.z, upper_j.z);

                detail::AABB aabb_intersect(intersect_lower, intersect_upper);

                // intersection AABB volume
                Scalar V =  (intersect_upper.x-intersect_lower.x)*(intersect_upper.y-intersect_lower.y);
                if(ndim == 3)
                    V *= intersect_upper.z-intersect_lower.z;

                // chooose the number of depletants in the intersection volume
                hoomd::PoissonDistribution<Scalar> poisson(m_fugacity[type]*V);
                #ifdef ENABLE_TBB
                hoomd::RandomGenerator& my_rng = rng_depletants_parallel.local();
                #else
                hoomd::RandomGenerator& my_rng = rng_depletants;
                #endif

                unsigned int n = poisson(my_rng);

                // for every depletant
                #ifdef ENABLE_TBB
                tbb::parallel_for(tbb::blocked_range<unsigned int>(0, (unsigned int)n),
                    [=, &intersect_i, &image_i, &aabbs_i,
                        &accept, &rng_depletants_parallel,
                        &thread_counters, &thread_implicit_counters](const tbb::blocked_range<unsigned int>& t) {
                for (unsigned int l = t.begin(); l != t.end(); ++l)
                #else
                for (unsigned int l = 0; l < n; ++l)
                #endif
                    {
                    #ifdef ENABLE_TBB
                    hoomd::RandomGenerator& my_rng = rng_depletants_parallel.local();
                    #else
                    hoomd::RandomGenerator& my_rng = rng_depletants;
                    #endif

                    #ifdef ENABLE_TBB
                    thread_implicit_counters[type].local().insert_count++;
                    #else
                    implicit_counters[type].insert_count++;
                    #endif

                    vec3<Scalar> pos_test = generatePositionInAABB(my_rng, aabb_intersect, ndim);
                    Shape shape_test(quat<Scalar>(), this->m_params[type]);
                    if (shape_test.hasOrientation())
                        {
                        shape_test.orientation = generateRandomOrientation(my_rng, ndim);
                        }

                    // check if depletant falls in other intersection volumes

                    // load the depletant's AABB and extend it
                    detail::AABB aabb_test = shape_test.getAABB(pos_test);
                    vec3<Scalar> lower_test = aabb_test.getLower();
                    vec3<Scalar> upper_test = aabb_test.getUpper();
                    lower_test.x -= m_sweep_radius; lower_test.y -= m_sweep_radius; lower_test.z -= m_sweep_radius;
                    upper_test.x += m_sweep_radius; upper_test.y += m_sweep_radius; upper_test.z += m_sweep_radius;
                    aabb_test = detail::AABB(lower_test, upper_test);

                    bool active = true;
                    for (unsigned int m = 0; m < k; ++m)
                        {
                        if (detail::overlap(aabb_test,aabbs_i[m]))
                            {
                            active = false;
                            break;
                            }
                        }

                    if (!active)
                        continue;

                    // depletant falls in intersection volume between circumspheres

                    if (!m_quermass)
                        {
                        // Check if the old configuration of particle i generates an overlap
                        bool overlap_old = false;
                            {
                            vec3<Scalar> r_ij = pos_i_old - pos_test;
                            OverlapReal rsq = dot(r_ij,r_ij);
                            OverlapReal DaDb = shape_test.getCircumsphereDiameter() + shape_old.getCircumsphereDiameter() + Scalar(2.0)*m_sweep_radius;
                            bool circumsphere_overlap = (rsq*OverlapReal(4.0) <= DaDb * DaDb);

                            if (m_quermass || h_overlaps[this->m_overlap_idx(type, typ_i)])
                                {
                                #ifdef ENABLE_TBB
                                thread_counters.local().overlap_checks++;
                                #else
                                counters.overlap_checks++;
                                #endif

                                unsigned int err = 0;
                                if (circumsphere_overlap && test_overlap(r_ij, shape_test, shape_old, err, 0.0, m_sweep_radius))
                                    {
                                    overlap_old = true;
                                    }
                                if (err)
                                #ifdef ENABLE_TBB
                                    thread_counters.local().overlap_err_count++;
                                #else
                                    counters.overlap_err_count++;
                                #endif
                                }
                            }

                        // if not intersecting ptl i in old config, ignore
                        if (!overlap_old)
                            continue;

                        // Check if the new configuration of particle i generates an overlap
                        bool overlap_new = false;

                            {
                            vec3<Scalar> r_ij = pos_i - pos_test;

                            OverlapReal rsq = dot(r_ij,r_ij);
                            OverlapReal DaDb = shape_test.getCircumsphereDiameter() + shape_i.getCircumsphereDiameter() + Scalar(2.0)*m_sweep_radius;
                            bool circumsphere_overlap = (rsq*OverlapReal(4.0) <= DaDb * DaDb);

                            if (m_quermass || h_overlaps[this->m_overlap_idx(type, typ_i)])
                                {
                                #ifdef ENABLE_TBB
                                thread_counters.local().overlap_checks++;
                                #else
                                counters.overlap_checks++;
                                #endif

                                unsigned int err = 0;
                                if (circumsphere_overlap && test_overlap(r_ij, shape_test, shape_i, err, 0.0, m_sweep_radius))
                                    {
                                    overlap_new = true;
                                    }
                                if (err)
                                #ifdef ENABLE_TBB
                                    thread_counters.local().overlap_err_count++;
                                #else
                                    counters.overlap_err_count++;
                                #endif
                                }
                            }

                        if (overlap_new)
                            continue;
                        }

                    // does the depletant fall into the overlap volume with other particles?
                    bool in_intersection_volume = false;
                    for (unsigned int m = 0; m < intersect_i.size(); ++m)
                        {
                        // read in its position and orientation
                        unsigned int j = intersect_i[m];

                        // load the old position and orientation of the j particle
                        Scalar4 postype_j = h_postype[j];
                        Scalar4 orientation_j = h_orientation[j];

                        vec3<Scalar> r_jk = vec3<Scalar>(postype_j) - pos_test - this->m_image_list[image_i[m]];

                        unsigned int typ_j = __scalar_as_int(postype_j.w);
                        Shape shape_j(quat<Scalar>(orientation_j), this->m_params[typ_j]);

                        #ifdef ENABLE_TBB
                        thread_counters.local().overlap_checks++;
                        #else
                        counters.overlap_checks++;
                        #endif

                        unsigned int err = 0;

                        if (m_quermass)
                            {
                            // check triple overlap of circumspheres

                            // check triple overlap with i at old position
                            vec3<Scalar> r_ij = vec3<Scalar>(postype_j) - pos_i_old - this->m_image_list[image_i[m]];

                            // need to enable later when we have a better way of excluding particles from the image list calculation
                            bool circumsphere_overlap = true || (h_overlaps[this->m_overlap_idx(type,typ_i)] && h_overlaps[this->m_overlap_idx(type,typ_j)]);
                            circumsphere_overlap = circumsphere_overlap && check_circumsphere_overlap_three(shape_old, shape_j, shape_test,
                                    r_ij, -r_jk+r_ij, m_sweep_radius, m_sweep_radius, 0.0);

                            if (circumsphere_overlap
                                && test_overlap_intersection(shape_old, shape_j, shape_test, r_ij, -r_jk+r_ij, err, m_sweep_radius, m_sweep_radius, 0.0))
                                in_intersection_volume = true;

                            if (in_intersection_volume)
                                {
                                // check triple overlap with i at new position
                                r_ij = vec3<Scalar>(postype_j) - pos_i - this->m_image_list[image_i[m]];
                                r_jk = ((i == j) ? pos_i : vec3<Scalar>(h_postype[j])) - pos_test - this->m_image_list[image_i[m]];

                                // need to enable later when we have a better way of excluding particles from the image list calculation
                                bool circumsphere_overlap = true || (h_overlaps[this->m_overlap_idx(type,typ_i)] && h_overlaps[this->m_overlap_idx(type,typ_j)]);
                                circumsphere_overlap = circumsphere_overlap && check_circumsphere_overlap_three(shape_i, shape_j, shape_test, r_ij, -r_jk+r_ij,
                                    m_sweep_radius, m_sweep_radius, 0.0);

                                if (circumsphere_overlap
                                    && test_overlap_intersection(shape_i, (i == j) ? shape_i : shape_j, shape_test, r_ij, -r_jk+r_ij, err,
                                        m_sweep_radius, m_sweep_radius, 0.0))
                                    in_intersection_volume = false;
                                }
                            }
                        else
                            {
                            // check circumsphere overlap
                            OverlapReal rsq = dot(r_jk,r_jk);
                            OverlapReal DaDb = shape_test.getCircumsphereDiameter() + shape_j.getCircumsphereDiameter() + Scalar(2.0)*m_sweep_radius;
                            bool circumsphere_overlap = (rsq*OverlapReal(4.0) <= DaDb * DaDb);

                            if (h_overlaps[this->m_overlap_idx(type,typ_j)]
                                && circumsphere_overlap
                                && test_overlap(r_jk, shape_test, shape_j, err, 0.0, m_sweep_radius))
                                {
                                in_intersection_volume = true;
                                }
                            }

                        if (err)
                        #ifdef ENABLE_TBB
                            thread_counters.local().overlap_err_count++;
                        #else
                            counters.overlap_err_count++;
                        #endif

                        if (in_intersection_volume)
                            break;
                        } // end loop over intersections

                    // if not part of overlap volume in new config, reject
                    if (in_intersection_volume)
                        {
                        accept = false;
                        #ifndef ENABLE_TBB
                        break;
                        #else
                        throw false;
                        #endif
                        }
                    } // end loop over depletants
                #ifdef ENABLE_TBB
                    });
                #endif

                #ifndef ENABLE_TBB
                if (!accept)
                    break;
                #endif
                } // end loop over overlapping spheres
            #ifdef ENABLE_TBB
                });
            #endif
            }
        // Depletant check for negative fugacity
        else if (accept && m_fugacity[type] < 0.0)
            {
            Shape tmp(quat<Scalar>(), this->m_params[type]);
            Scalar d_dep = tmp.getCircumsphereDiameter();

            // find neighbors whose circumspheres overlap particle i's excluded volume circumsphere in the new configuration
            Scalar range = m_quermass ? Scalar(2.0)*m_sweep_radius : d_dep;

            // get new AABB and extend
            vec3<Scalar> lower = aabb_i_local.getLower();
            vec3<Scalar> upper = aabb_i_local.getUpper();
            lower.x -= range; lower.y -= range; lower.z -= range;
            upper.x += range; upper.y += range; upper.z += range;
            detail::AABB aabb_local = detail::AABB(lower,upper);

            // All image boxes (including the primary)
            for (unsigned int cur_image = 0; cur_image < n_images; cur_image++)
                {
                vec3<Scalar> pos_i_image = pos_i + this->m_image_list[cur_image];
                detail::AABB aabb = aabb_local;
                aabb.translate(pos_i_image);

                // stackless search
                for (unsigned int cur_node_idx = 0; cur_node_idx < this->m_aabb_tree.getNumNodes(); cur_node_idx++)
                    {
                    if (detail::overlap(this->m_aabb_tree.getNodeAABB(cur_node_idx), aabb))
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

                                vec3<Scalar> r_ij = pos_j - pos_i_image;
                                Shape shape_j(quat<Scalar>(), this->m_params[typ_j]);

                                if (shape_j.hasOrientation())
                                    {
                                    if (i == j)
                                        shape_j.orientation = shape_i.orientation;
                                    else
                                        shape_j.orientation = quat<Scalar>(h_orientation[j]);
                                    }

                                // check excluded volume overlap
                                detail::AABB aabb_j = shape_j.getAABB(pos_j);
                                bool overlap_excluded = detail::overlap(aabb,aabb_j);

                                // currently ignore overlap flags with quermass==true
                                overlap_excluded = overlap_excluded && (m_quermass || h_overlaps[this->m_overlap_idx(type,typ_j)]);

                                if (Shape::supportsSweepRadius() && overlap_excluded)
                                    {
                                    // do i and j overlap with their excluded volumes in the new configuration?
                                    #ifdef ENABLE_TBB
                                    thread_counters.local().overlap_checks++;
                                    #else
                                    counters.overlap_checks++;
                                    #endif

                                    unsigned int err = 0;
                                    Scalar sweep_radius = m_quermass ? m_sweep_radius : Scalar(0.5)*tmp.getCircumsphereDiameter();
                                    overlap_excluded = test_overlap(r_ij, shape_i, shape_j, err, sweep_radius, sweep_radius);

                                    if (err)
                                    #ifdef ENABLE_TBB
                                        thread_counters.local().overlap_err_count++;
                                    #else
                                        counters.overlap_err_count++;
                                    #endif
                                    }
                                if (overlap_excluded)
                                    {
                                    intersect_i.push_back(j);
                                    image_i.push_back(cur_image);

                                    // cache the translated AABB of particle j
                                    aabb_j.translate(-this->m_image_list[cur_image]);
                                    aabbs_i.push_back(aabb_j);
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


            // now, we have a list of intersecting spheres, sample in the union of intersection volumes
            // we sample from their union by checking if any generated position falls in the intersection
            // of their AABBs and if so, only accepting it if it was generated from neighbor j_min

            // world AABB of particle i
            detail::AABB aabb_i = aabb_i_local;
            aabb_i.translate(pos_i);

            // extend AABB i by sweep radius
            range = Scalar(0.5)*d_dep+m_sweep_radius;
            vec3<Scalar> lower_i = aabb_i.getLower();
            vec3<Scalar> upper_i = aabb_i.getUpper();
            lower_i.x -= range; lower_i.y -= range; lower_i.z -= range;
            upper_i.x += range; upper_i.y += range; upper_i.z += range;

            // for every pairwise intersection
            #ifdef ENABLE_TBB
            tbb::parallel_for(tbb::blocked_range<unsigned int>(0, (unsigned int)intersect_i.size()),
                [=, &intersect_i, &image_i, &aabbs_i,
                    &accept, &rng_depletants_parallel,
                    &thread_counters, &thread_implicit_counters](const tbb::blocked_range<unsigned int>& s) {
            for (unsigned int k = s.begin(); k != s.end(); ++k)
            #else
            for (unsigned int k = 0; k < intersect_i.size(); ++k)
            #endif
                {
                // world AABB of shape j, in image of i
                detail::AABB aabb_j = aabbs_i[k];

                // extend AABB j by sweep radius
                vec3<Scalar> lower_j = aabb_j.getLower();
                vec3<Scalar> upper_j = aabb_j.getUpper();
                lower_j.x -= range; lower_j.y -= range; lower_j.z -= range;
                upper_j.x += range; upper_j.y += range; upper_j.z += range;

                // we already know the AABBs are overlapping, compute their intersection
                vec3<Scalar> intersect_lower, intersect_upper;
                intersect_lower.x = std::max(lower_i.x, lower_j.x);
                intersect_lower.y = std::max(lower_i.y, lower_j.y);
                intersect_lower.z = std::max(lower_i.z, lower_j.z);
                intersect_upper.x = std::min(upper_i.x, upper_j.x);
                intersect_upper.y = std::min(upper_i.y, upper_j.y);
                intersect_upper.z = std::min(upper_i.z, upper_j.z);

                detail::AABB aabb_intersect(intersect_lower, intersect_upper);

                // intersection AABB volume
                Scalar V =  (intersect_upper.x-intersect_lower.x)*(intersect_upper.y-intersect_lower.y);
                if(ndim == 3)
                    V *= intersect_upper.z-intersect_lower.z;

                // chooose the number of depletants in the intersection volume
                hoomd::PoissonDistribution<Scalar> poisson(-m_fugacity[type]*V);
                #ifdef ENABLE_TBB
                hoomd::RandomGenerator& my_rng = rng_depletants_parallel.local();
                #else
                hoomd::RandomGenerator& my_rng = rng_depletants;
                #endif

                unsigned int n = poisson(my_rng);

                // for every depletant
                #ifdef ENABLE_TBB
                tbb::parallel_for(tbb::blocked_range<unsigned int>(0, (unsigned int)n),
                    [=, &intersect_i, &image_i, &aabbs_i,
                        &accept, &rng_depletants_parallel,
                        &thread_counters, &thread_implicit_counters](const tbb::blocked_range<unsigned int>& t) {
                for (unsigned int l = t.begin(); l != t.end(); ++l)
                #else
                for (unsigned int l = 0; l < n; ++l)
                #endif
                    {
                    #ifdef ENABLE_TBB
                    hoomd::RandomGenerator& my_rng = rng_depletants_parallel.local();
                    #else
                    hoomd::RandomGenerator& my_rng = rng_depletants;
                    #endif

                    #ifdef ENABLE_TBB
                    thread_implicit_counters[type].local().insert_count++;
                    #else
                    implicit_counters[type].insert_count++;
                    #endif

                    vec3<Scalar> pos_test = generatePositionInAABB(my_rng, aabb_intersect, ndim);
                    Shape shape_test(quat<Scalar>(), this->m_params[type]);
                    if (shape_test.hasOrientation())
                        {
                        shape_test.orientation = generateRandomOrientation(my_rng, ndim);
                        }

                    // check if depletant falls in other intersection volumes (new)
                    bool active = true;

                    // check against any other AABB preceding this

                    // load the depletant's AABB and extend it
                    detail::AABB aabb_test = shape_test.getAABB(pos_test);
                    vec3<Scalar> lower_test = aabb_test.getLower();
                    vec3<Scalar> upper_test = aabb_test.getUpper();
                    lower_test.x -= m_sweep_radius; lower_test.y -= m_sweep_radius; lower_test.z -= m_sweep_radius;
                    upper_test.x += m_sweep_radius; upper_test.y += m_sweep_radius; upper_test.z += m_sweep_radius;
                    aabb_test = detail::AABB(lower_test, upper_test);

                    for (unsigned int m = 0; m < k; ++m)
                        {
                        if (detail::overlap(aabb_test,aabbs_i[m]))
                            {
                            active = false;
                            break;
                            }
                        }

                    if (!active)
                        continue;

                    // depletant falls in intersection volume between circumspheres

                    if (!m_quermass)
                        {
                        // Check if the new configuration of particle i generates an overlap
                        bool overlap_new = false;

                            {
                            vec3<Scalar> r_ij = pos_i - pos_test;

                            OverlapReal rsq = dot(r_ij,r_ij);
                            OverlapReal DaDb = shape_test.getCircumsphereDiameter() + shape_i.getCircumsphereDiameter() + OverlapReal(2.0)*m_sweep_radius;
                            bool circumsphere_overlap = (rsq*OverlapReal(4.0) <= DaDb * DaDb);

                            if (m_quermass || h_overlaps[this->m_overlap_idx(type, typ_i)])
                                {
                                #ifdef ENABLE_TBB
                                thread_counters.local().overlap_checks++;
                                #else
                                counters.overlap_checks++;
                                #endif

                                unsigned int err = 0;
                                if (circumsphere_overlap && test_overlap(r_ij, shape_test, shape_i, err, 0.0, m_sweep_radius))
                                    {
                                    overlap_new = true;
                                    }
                                if (err)
                                #ifdef ENABLE_TBB
                                    thread_counters.local().overlap_err_count++;
                                #else
                                    counters.overlap_err_count++;
                                #endif
                                }
                            }

                        if (!overlap_new)
                            continue;

                        // Check if the old configuration of particle i generates an overlap
                        bool overlap_old = false;

                        vec3<Scalar> r_ij = vec3<Scalar>(h_postype[i]) - pos_test;

                        OverlapReal rsq = dot(r_ij,r_ij);
                        OverlapReal DaDb = shape_test.getCircumsphereDiameter() + shape_old.getCircumsphereDiameter() + OverlapReal(2.0)*m_sweep_radius;
                        bool circumsphere_overlap = (rsq*OverlapReal(4.0) <= DaDb * DaDb);

                        if (m_quermass || h_overlaps[this->m_overlap_idx(type, typ_i)])
                            {
                            #ifdef ENABLE_TBB
                            thread_counters.local().overlap_checks++;
                            #else
                            counters.overlap_checks++;
                            #endif

                            unsigned int err = 0;
                            if (circumsphere_overlap && test_overlap(r_ij, shape_test, shape_old, err, 0.0, m_sweep_radius))
                                {
                                overlap_old = true;
                                }
                            if (err)
                            #ifdef ENABLE_TBB
                                thread_counters.local().overlap_err_count++;
                            #else
                                counters.overlap_err_count++;
                            #endif
                            }

                        if (overlap_old)
                            {
                            // all is ok
                            continue;
                            }
                        }

                    bool in_new_intersection_volume = false;

                    vec3<Scalar> pos_i_old(h_postype[i]);

                    for (unsigned int m = 0; m < intersect_i.size(); ++m)
                        {
                        // read in its position and orientation
                        unsigned int j = intersect_i[m];

                        vec3<Scalar> r_jk = ((i == j) ? pos_i : vec3<Scalar>(h_postype[j])) - pos_test -
                            this->m_image_list[image_i[m]];

                        unsigned int typ_j = (i == j) ? typ_i : __scalar_as_int(h_postype[j].w);
                        Shape shape_j((i == j) ? shape_i.orientation : quat<Scalar>(h_orientation[j]), this->m_params[typ_j]);

                        #ifdef ENABLE_TBB
                        thread_counters.local().overlap_checks++;
                        #else
                        counters.overlap_checks++;
                        #endif

                        unsigned int err = 0;
                        if (m_quermass)
                            {
                            // check triple overlap of circumspheres
                            vec3<Scalar> r_ij = ( (i==j) ? pos_i : vec3<Scalar>(h_postype[j])) - pos_i -
                                this->m_image_list[image_i[m]];

                            // need to enable later when we have a better way of excluding particles from the image list calculation
                            bool circumsphere_overlap = true || (h_overlaps[this->m_overlap_idx(type,typ_i)] && h_overlaps[this->m_overlap_idx(type,typ_j)]);
                            circumsphere_overlap = circumsphere_overlap && check_circumsphere_overlap_three(shape_i, shape_j, shape_test, r_ij, r_ij - r_jk,
                                m_sweep_radius, m_sweep_radius, 0.0);

                            // check triple overlap with new configuration
                            unsigned int err = 0;

                            if (circumsphere_overlap
                                && test_overlap_intersection(shape_i, shape_j, shape_test, r_ij, r_ij - r_jk, err, m_sweep_radius, m_sweep_radius, 0.0))
                                in_new_intersection_volume = true;

                            if (in_new_intersection_volume)
                                {
                                // check triple overlap with old configuration
                                r_ij = vec3<Scalar>(h_postype[j]) - pos_i_old - this->m_image_list[image_i[m]];
                                r_jk = vec3<Scalar>(h_postype[j]) - pos_test - this->m_image_list[image_i[m]];

                                // need to enable later when we have a better way of excluding particles from the image list calculation
                                bool circumsphere_overlap = true || (h_overlaps[this->m_overlap_idx(type,typ_i)] && h_overlaps[this->m_overlap_idx(type,typ_j)]);
                                circumsphere_overlap = circumsphere_overlap && check_circumsphere_overlap_three(shape_old, shape_j, shape_test, r_ij, r_ij - r_jk,
                                    m_sweep_radius, m_sweep_radius, 0.0);

                                if (circumsphere_overlap
                                    && test_overlap_intersection(shape_old, (i == j) ? shape_old : shape_j, shape_test, r_ij, r_ij - r_jk, err,
                                        m_sweep_radius, m_sweep_radius, 0.0))
                                    in_new_intersection_volume = false;
                                }
                            if (err)
                            #ifdef ENABLE_TBB
                                thread_counters.local().overlap_err_count++;
                            #else
                                counters.overlap_err_count++;
                            #endif
                            }
                        else
                            {
                            // check circumsphere overlap
                            OverlapReal rsq = dot(r_jk,r_jk);
                            OverlapReal DaDb = shape_test.getCircumsphereDiameter() + shape_j.getCircumsphereDiameter() + OverlapReal(2.0)*m_sweep_radius;
                            bool circumsphere_overlap = (rsq*OverlapReal(4.0) <= DaDb * DaDb);

                            if (h_overlaps[this->m_overlap_idx(type,typ_j)]
                                && circumsphere_overlap
                                && test_overlap(r_jk, shape_test, shape_j, err, 0.0, m_sweep_radius))
                                {
                                in_new_intersection_volume = true;
                                }
                            if (err)
                            #ifdef ENABLE_TBB
                                thread_counters.local().overlap_err_count++;
                            #else
                                counters.overlap_err_count++;
                            #endif
                            }

                        if (in_new_intersection_volume)
                            break;
                        }

                    if (in_new_intersection_volume)
                        {
                        accept = false;

                        #ifndef ENABLE_TBB
                        // early exit
                        break;
                        #else
                        throw false;
                        #endif
                        }
                    } // end loop over depletants
                #ifdef ENABLE_TBB
                    });
                #endif

                #ifndef ENABLE_TBB
                if (!accept) break;
                #endif
                } // end loop over overlapping spheres
            #ifdef ENABLE_TBB
                });
            #endif
            } // end depletant placement
        }
    #ifdef ENABLE_TBB
    } catch (bool b) { }
    #endif

    #ifdef ENABLE_TBB
    // reduce counters
    for (auto i = thread_counters.begin(); i != thread_counters.end(); ++i)
        {
        counters = counters + *i;
        }

    for (unsigned int i = 0; i < this->m_pdata->getNTypes(); ++i)
        for (auto it= thread_implicit_counters[i].begin(); it != thread_implicit_counters[i].end(); ++it)
            {
            implicit_counters[i] = implicit_counters[i] + *it;
            }
    #endif

    return accept;
    }

template<class Shape>
bool IntegratorHPMCMono<Shape>::attemptBoxResize(unsigned int timestep, const BoxDim& new_box)
    {
    // call parent class method
    bool result = IntegratorHPMC::attemptBoxResize(timestep, new_box);

    if (result)
        {
        for (unsigned int type_d = 0; type_d < this->m_pdata->getNTypes(); ++type_d)
            {
            if (getDepletantFugacity(type_d) != 0.0)
                throw std::runtime_error("Implicit depletants not supported with NPT ensemble\n");
            }
        }

    return result;
    }

//! Export the IntegratorHPMCMono class to python
/*! \param name Name of the class in the exported python module
    \tparam Shape An instantiation of IntegratorHPMCMono<Shape> will be exported
*/
template < class Shape > void export_IntegratorHPMCMono(pybind11::module& m, const std::string& name)
    {
    pybind11::class_< IntegratorHPMCMono<Shape>, IntegratorHPMC, std::shared_ptr< IntegratorHPMCMono<Shape> > >(m, name.c_str())
          .def(pybind11::init< std::shared_ptr<SystemDefinition>, unsigned int >())
          .def("setParam", &IntegratorHPMCMono<Shape>::setParam)
          .def("setOverlapChecks", &IntegratorHPMCMono<Shape>::setOverlapChecks)
          .def("setExternalField", &IntegratorHPMCMono<Shape>::setExternalField)
          .def("setPatchEnergy", &IntegratorHPMCMono<Shape>::setPatchEnergy)
          .def("mapOverlaps", &IntegratorHPMCMono<Shape>::PyMapOverlaps)
          .def("mapEnergies", &IntegratorHPMCMono<Shape>::PyMapEnergies)
          .def("connectGSDStateSignal", &IntegratorHPMCMono<Shape>::connectGSDStateSignal)
          .def("connectGSDShapeSpec", &IntegratorHPMCMono<Shape>::connectGSDShapeSpec)
          .def("restoreStateGSD", &IntegratorHPMCMono<Shape>::restoreStateGSD)
          .def("py_test_overlap", &IntegratorHPMCMono<Shape>::py_test_overlap)
          .def("setDepletantFugacity", &IntegratorHPMCMono<Shape>::setDepletantFugacity)
          .def("getImplicitCounters", &IntegratorHPMCMono<Shape>::getImplicitCounters)
          .def("getDepletantFugacity", &IntegratorHPMCMono<Shape>::getDepletantFugacity)
          .def("setQuermassMode", &IntegratorHPMCMono<Shape>::setQuermassMode)
          .def("setSweepRadius", &IntegratorHPMCMono<Shape>::setSweepRadius)
          .def("getQuermassMode", &IntegratorHPMCMono<Shape>::getQuermassMode)
          .def("getSweepRadius", &IntegratorHPMCMono<Shape>::getSweepRadius)
          .def("getTypeShapesPy", &IntegratorHPMCMono<Shape>::getTypeShapesPy)
          ;
    }

//! Export the counters for depletants
inline void export_hpmc_implicit_counters(pybind11::module& m)
    {
    pybind11::class_< hpmc_implicit_counters_t >(m, "hpmc_implicit_counters_t")
    .def_readwrite("insert_count", &hpmc_implicit_counters_t::insert_count)
    ;
    }

} // end namespace hpmc

#endif // _INTEGRATOR_HPMC_MONO_H_
