// Copyright (c) 2009-2019 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.

#ifndef __HPMC_MONO_IMPLICIT__H__
#define __HPMC_MONO_IMPLICIT__H__

#include "IntegratorHPMCMono.h"
#include "hoomd/Autotuner.h"
#include "hoomd/Saru.h"

#include <atomic>
#include <random>
#include <cfloat>

/*! \file IntegratorHPMCMonoImplicit.h
    \brief Defines the template class for HPMC with implicit generated depletant solvent
    \note This header cannot be compiled by nvcc
*/

#ifdef NVCC
#error This header cannot be compiled by nvcc
#endif

#include <hoomd/extern/pybind/include/pybind11/pybind11.h>

#ifdef ENABLE_TBB
#include <tbb/parallel_for.h>
#include <tbb/enumerable_thread_specific.h>
#include <thread>
#endif

namespace hpmc
{

//! Template class for HPMC update with implicit depletants
/*!
    Depletants are generated randomly on the fly according to the semi-grand canonical ensemble.

    The penetrable depletants model is simulated.

    \ingroup hpmc_integrators
*/
template< class Shape >
class IntegratorHPMCMonoImplicit : public IntegratorHPMCMono<Shape>
    {
    public:
        //! Construct the integrator
        IntegratorHPMCMonoImplicit(std::shared_ptr<SystemDefinition> sysdef,
                              unsigned int seed,
                              unsigned int method);
        //! Destructor
        virtual ~IntegratorHPMCMonoImplicit();

        //! Set the depletant density in the free volume
        void setDepletantDensity(Scalar n_R)
            {
            m_n_R = n_R;
            m_need_initialize_poisson = true;
            }

        //! Set the type of depletant particle
        void setDepletantType(unsigned int type)
            {
            m_type = type;
            }

        //! Number of depletant-reinsertions
        /*! \param n_trial Depletant reinsertions per overlapping depletant
         */
        void setNTrial(unsigned int n_trial)
            {
            m_n_trial = n_trial;
            }

        //! Return number of depletant re-insertions
        unsigned int getNTrial()
            {
            return m_n_trial;
            }

        //! Returns the depletant density
        Scalar getDepletantDensity()
            {
            return m_n_R;
            }

        //! Return the depletant type
        unsigned int getDepletantType()
            {
            return m_type;
            }

        //! Return the number of re-insertion trials
        unsigned int getNumTrials() const
            {
            return m_n_trial;
            }

        //! Reset statistics counters
        virtual void resetStats()
            {
            IntegratorHPMCMono<Shape>::resetStats();
            ArrayHandle<hpmc_implicit_counters_t> h_counters(m_implicit_count, access_location::host, access_mode::read);
            m_implicit_count_run_start = h_counters.data[0];
            }

        //! Print statistics about the hpmc steps taken
        virtual void printStats()
            {
            IntegratorHPMCMono<Shape>::printStats();

            hpmc_implicit_counters_t result = getImplicitCounters(1);

            double cur_time = double(this->m_clock.getTime()) / Scalar(1e9);

            this->m_exec_conf->msg->notice(2) << "-- Implicit depletants stats:" << "\n";
            this->m_exec_conf->msg->notice(2) << "Depletant insertions per second:          "
                << double(result.insert_count)/cur_time << "\n";
            if (m_method == 0)
                {
                this->m_exec_conf->msg->notice(2) << "Configurational bias attempts per second: "
                    << double(result.reinsert_count)/cur_time << "\n";
                this->m_exec_conf->msg->notice(2) << "Fraction of depletants in free volume:    "
                    << result.getFreeVolumeFraction() << "\n";
                this->m_exec_conf->msg->notice(2) << "Fraction of overlapping depletants:       "
                    << result.getOverlapFraction()<< "\n";
                }
            }

        //! Get the current counter values
        hpmc_implicit_counters_t getImplicitCounters(unsigned int mode=0);

        /* \returns a list of provided quantities
        */
        std::vector< std::string > getProvidedLogQuantities()
            {
            // start with the integrator provided quantities
            std::vector< std::string > result = IntegratorHPMCMono<Shape>::getProvidedLogQuantities();

            // then add ours
            result.push_back("hpmc_fugacity");
            result.push_back("hpmc_insert_count");
            if (m_method == 0)
                {
                result.push_back("hpmc_ntrial");
                result.push_back("hpmc_reinsert_count");
                result.push_back("hpmc_free_volume_fraction");
                result.push_back("hpmc_overlap_fraction");
                result.push_back("hpmc_configurational_bias_ratio");
                }

            return result;
            }

        //! Get the value of a logged quantity
        virtual Scalar getLogValue(const std::string& quantity, unsigned int timestep);

        //! Method to scale the box
        virtual bool attemptBoxResize(unsigned int timestep, const BoxDim& new_box);

        //! Slot to be called when number of types changes
        void slotNumTypesChange();

    protected:
        Scalar m_n_R;                                            //!< Average depletant number density in free volume
        unsigned int m_type;                                     //!< Type of depletant particle to generate

        GPUArray<hpmc_implicit_counters_t> m_implicit_count;     //!< Counter of active cell cluster moves
        hpmc_implicit_counters_t m_implicit_count_run_start;     //!< Counter of active cell cluster moves at run start
        hpmc_implicit_counters_t m_implicit_count_step_start;    //!< Counter of active cell cluster moves at run start

        std::vector<Scalar> m_lambda;                            //!< Poisson distribution parameters per type
        Scalar m_d_dep;                                          //!< Depletant circumsphere diameter
        GPUArray<Scalar> m_d_min;                                //!< Minimum sphere from which test depletant is excluded
        GPUArray<Scalar> m_d_max;                                //!< Maximum sphere for test depletant insertion

        unsigned int m_n_trial;                                 //!< Number of trial re-insertions per depletant

        bool m_need_initialize_poisson;                             //!< Flag to tell if we need to initialize the poisson distribution

        unsigned int m_method;                                   //!< Whether to use the "overlap_regions" or "circumsphere" integrator

        //! Take one timestep forward
        virtual void update(unsigned int timestep);

        //! Test whether to reject the current particle move based on depletants
        #ifndef ENABLE_TBB
        inline bool checkDepletantOverlap(unsigned int i, vec3<Scalar> pos_i, Shape shape_i, unsigned int typ_i, Scalar d_max, Scalar d_min, Scalar4 *h_postype, Scalar4 *h_orientation, unsigned int *h_overlaps, hpmc_counters_t& counters, hpmc_implicit_counters_t& implicit_counters, std::mt19937& rng_poisson, hoomd::detail::Saru& rng_i);
        #else
        inline bool checkDepletantOverlap(unsigned int i, vec3<Scalar> pos_i, Shape shape_i, unsigned int typ_i, Scalar d_max, Scalar d_min, Scalar4 *h_postype, Scalar4 *h_orientation, unsigned int *h_overlaps, hpmc_counters_t& counters, hpmc_implicit_counters_t& implicit_counters, hoomd::detail::Saru& rng_i, tbb::enumerable_thread_specific< hoomd::detail::Saru >& rng_parallel, tbb::enumerable_thread_specific<std::mt19937>& rng_parallel_mt);
        #endif

        //! Test whether to reject the current particle move based on depletants
        #ifndef ENABLE_TBB
        inline bool checkDepletantCircumsphere(unsigned int i, vec3<Scalar> pos_i, Shape shape_i, unsigned int typ_i, Scalar d_max, Scalar d_min, Scalar4 *h_postype, Scalar4 *h_orientation, unsigned int *h_overlaps, hpmc_counters_t& counters, hpmc_implicit_counters_t& implicit_counters, std::mt19937& rng_poisson, hoomd::detail::Saru& rng_i);
        #else
        inline bool checkDepletantCircumsphere(unsigned int i, vec3<Scalar> pos_i, Shape shape_i, unsigned int typ_i, Scalar d_max, Scalar d_min, Scalar4 *h_postype, Scalar4 *h_orientation, unsigned int *h_overlaps, hpmc_counters_t& counters, hpmc_implicit_counters_t& implicit_counters, hoomd::detail::Saru& rng_i, tbb::enumerable_thread_specific< hoomd::detail::Saru >& rng_parallel, tbb::enumerable_thread_specific<std::mt19937>& rng_parallel_mt);
        #endif

        //! Initialize Poisson distribution parameters
        virtual void updatePoissonParameters();

        //! Set the nominal width appropriate for depletion interaction
        virtual void updateCellWidth();

        //! Generate a random depletant position in a sphere around a particle
        template<class RNG>
        inline void generateDepletant(RNG& rng, vec3<Scalar> pos_sphere, Scalar delta, Scalar d_min,
            vec3<Scalar>& pos, quat<Scalar>& orientation, const typename Shape::param_type& params_depletants);

        /*! Generate a random depletant position in a region including the sphere around a particle,
            restricted so that it does not intersect another sphere
         */
        template<class RNG>
        inline void generateDepletantRestricted(RNG& rng, vec3<Scalar> pos_sphere, Scalar delta, Scalar delta_other,
            vec3<Scalar>& pos, quat<Scalar>& orientation, const typename Shape::param_type& params_depletants,
            vec3<Scalar> pos_sphere_other);

        //! Try inserting a depletant in a configuration such that it overlaps with the particle in the old (new) configuration
        inline bool insertDepletant(vec3<Scalar>& pos_depletant, const Shape& shape_depletant, unsigned int idx,
            typename Shape::param_type *params, unsigned int *h_overlaps, unsigned int typ_i, Scalar4 *h_postype, Scalar4 *h_orientation,
            vec3<Scalar>  pos_new, quat<Scalar>& orientation_new, const typename Shape::param_type& params_new,
            unsigned int &overlap_checks, unsigned int &overlap_err_count, bool &overlap_shape, bool new_config);
    };

/*! \param sysdef System definition
    \param cl Cell list
    \param seed Random number generator seed

    NOTE: only 3d supported at this time
    */

template< class Shape >
IntegratorHPMCMonoImplicit< Shape >::IntegratorHPMCMonoImplicit(std::shared_ptr<SystemDefinition> sysdef,
                                                                   unsigned int seed,
                                                                   unsigned int method)
    : IntegratorHPMCMono<Shape>(sysdef, seed), m_n_R(0), m_type(0), m_d_dep(0.0), m_n_trial(0),
      m_need_initialize_poisson(true), m_method(method)
    {
    this->m_exec_conf->msg->notice(5) << "Constructing IntegratorHPMCImplicit" << std::endl;

    GPUArray<hpmc_implicit_counters_t> implicit_count(1,this->m_exec_conf);
    m_implicit_count.swap(implicit_count);

    GPUArray<Scalar> d_min(this->m_pdata->getNTypes(), this->m_exec_conf);
    m_d_min.swap(d_min);

    GPUArray<Scalar> d_max(this->m_pdata->getNTypes(), this->m_exec_conf);
    m_d_max.swap(d_max);

    m_lambda.resize(this->m_pdata->getNTypes(),FLT_MAX);
    }

//! Destructor
template< class Shape >
IntegratorHPMCMonoImplicit< Shape >::~IntegratorHPMCMonoImplicit()
    {
    }

template <class Shape>
void IntegratorHPMCMonoImplicit<Shape>::slotNumTypesChange()
    {
    // call parent class method
    IntegratorHPMCMono<Shape>::slotNumTypesChange();

    m_lambda.resize(this->m_pdata->getNTypes(),FLT_MAX);

    GPUArray<Scalar> d_min(this->m_pdata->getNTypes(), this->m_exec_conf);
    m_d_min.swap(d_min);

    GPUArray<Scalar> d_max(this->m_pdata->getNTypes(), this->m_exec_conf);
    m_d_max.swap(d_max);

    m_need_initialize_poisson = true;
    }

template< class Shape >
void IntegratorHPMCMonoImplicit< Shape >::updatePoissonParameters()
    {
    // Depletant diameter
    quat<Scalar> o;
    Shape shape_depletant(o, this->m_params[this->m_type]);
    m_d_dep = shape_depletant.getCircumsphereDiameter();

    // access GPUArrays
    ArrayHandle<Scalar> h_d_min(m_d_min, access_location::host, access_mode::overwrite);
    ArrayHandle<Scalar> h_d_max(m_d_max, access_location::host, access_mode::overwrite);

    for (unsigned int i_type = 0; i_type < this->m_pdata->getNTypes(); ++i_type)
        {
        // test sphere diameter and volume
        Shape shape_i(quat<Scalar>(), this->m_params[i_type]);
        Scalar delta = shape_i.getCircumsphereDiameter()+m_d_dep;
        h_d_max.data[i_type] = delta;

        // volume of insertion sphere
        Scalar V = Scalar(M_PI/6.0)*delta*delta*delta;

        // Minimum diameter of colloid sphere in which depletant can be inserted without overlapping with other colloids
//        Scalar d = std::max(Scalar(2.0)*shape_i.getInsphereRadius()-m_d_dep,0.0);
        Scalar d = Scalar(0.0);

        h_d_min.data[i_type] = d;

        // subtract inner sphere from sampling volume
        V -= Scalar(M_PI/6.0)*d*d*d;

        // average number of depletants in volume
        m_lambda[i_type] = this->m_n_R*V;
        }
    }

template< class Shape >
void IntegratorHPMCMonoImplicit< Shape >::updateCellWidth()
    {
    this->m_nominal_width = this->getMaxCoreDiameter();

    if (m_n_R > Scalar(0.0))
        {
        // add range of depletion interaction
        quat<Scalar> o;
        Shape tmp(o, this->m_params[m_type]);
        this->m_nominal_width += tmp.getCircumsphereDiameter();

        // extend the image list by the depletant diameter, since we're querying
        // AABBs that are larger than the shape diameters themselves
        this->m_extra_image_width = tmp.getCircumsphereDiameter();
        }
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

    this->m_exec_conf->msg->notice(5) << "IntegratorHPMCMonoImplicit: updating nominal width to " << this->m_nominal_width << std::endl;
    }

template< class Shape >
void IntegratorHPMCMonoImplicit< Shape >::update(unsigned int timestep)
    {
    this->m_exec_conf->msg->notice(10) << "HPMCMonoImplicit update: " << timestep << std::endl;
    IntegratorHPMC::update(timestep);

    // update poisson distributions
    if (m_need_initialize_poisson)
        {
        updatePoissonParameters();
        m_need_initialize_poisson = false;
        }

    // get needed vars
    ArrayHandle<hpmc_counters_t> h_counters(this->m_count_total, access_location::host, access_mode::readwrite);
    hpmc_counters_t& counters = h_counters.data[0];

    ArrayHandle<hpmc_implicit_counters_t> h_implicit_counters(m_implicit_count, access_location::host, access_mode::readwrite);
    hpmc_implicit_counters_t& implicit_counters = h_implicit_counters.data[0];

    m_implicit_count_step_start = implicit_counters;

    const BoxDim& box = this->m_pdata->getBox();
    unsigned int ndim = this->m_sysdef->getNDimensions();

    #ifdef ENABLE_MPI
    // compute the width of the active region
    Scalar3 npd = box.getNearestPlaneDistance();
    Scalar3 ghost_fraction = this->m_nominal_width / npd;
    #endif

    // Shuffle the order of particles for this step
    this->m_update_order.resize(this->m_pdata->getN());
    this->m_update_order.shuffle(timestep);

    // update the AABB Tree
    this->buildAABBTree();
    // limit m_d entries so that particles cannot possibly wander more than one box image in one time step
    this->limitMoveDistances();
    // update the image list
    this->updateImageList();

    // Combine the three seeds to generate RNG for poisson distribution
    #ifndef ENABLE_TBB
    std::vector<unsigned int> seed_seq(4);
    seed_seq[0] = this->m_seed;
    seed_seq[1] = timestep;
    seed_seq[2] = this->m_exec_conf->getRank();
    seed_seq[3] = 0x91baff72;
    std::seed_seq seed(seed_seq.begin(), seed_seq.end());
    std::mt19937 rng_poisson(seed);
    #else
    // create one RNG per thread
    tbb::enumerable_thread_specific< hoomd::detail::Saru > rng_parallel([=]
        {
        std::vector<unsigned int> seed_seq(5);
        seed_seq[0] = this->m_seed;
        seed_seq[1] = timestep;
        seed_seq[2] = this->m_exec_conf->getRank();
        std::hash<std::thread::id> hash;
        seed_seq[3] = hash(std::this_thread::get_id());
        seed_seq[4] = 0x6b71abc8;
        std::seed_seq seed(seed_seq.begin(), seed_seq.end());
        std::vector<unsigned int> s(1);
        seed.generate(s.begin(),s.end());
        return s[0];
        });
    tbb::enumerable_thread_specific<std::mt19937> rng_parallel_mt([=]
        {
        std::vector<unsigned int> seed_seq(5);
        seed_seq[0] = this->m_seed;
        seed_seq[1] = timestep;
        seed_seq[2] = this->m_exec_conf->getRank();
        std::hash<std::thread::id> hash;
        seed_seq[3] = hash(std::this_thread::get_id());
        seed_seq[4] = 0x91baff72;
        std::seed_seq seed(seed_seq.begin(), seed_seq.end());
        std::vector<unsigned int> s(1);
        seed.generate(s.begin(),s.end());
        return s[0]; // use a single seed
        });
    #endif

    if (this->m_prof) this->m_prof->push(this->m_exec_conf, "HPMC implicit");

    // access depletant insertion sphere dimensions
    ArrayHandle<Scalar> h_d_min(m_d_min, access_location::host, access_mode::read);
    ArrayHandle<Scalar> h_d_max(m_d_max, access_location::host, access_mode::read);

    // loop over local particles nselect times
    for (unsigned int i_nselect = 0; i_nselect < this->m_nselect; i_nselect++)
        {
        // access particle data and system box
        ArrayHandle<Scalar4> h_postype(this->m_pdata->getPositions(), access_location::host, access_mode::readwrite);
        ArrayHandle<Scalar4> h_orientation(this->m_pdata->getOrientationArray(), access_location::host, access_mode::readwrite);
        ArrayHandle<Scalar> h_diameter(this->m_pdata->getDiameters(), access_location::host, access_mode::read);
        ArrayHandle<Scalar> h_charge(this->m_pdata->getCharges(), access_location::host, access_mode::read);

        // access interaction matrix
        ArrayHandle<unsigned int> h_overlaps(this->m_overlaps, access_location::host, access_mode::read);

        //access move sizes
        ArrayHandle<Scalar> h_d(this->m_d, access_location::host, access_mode::read);
        ArrayHandle<Scalar> h_a(this->m_a, access_location::host, access_mode::read);

        // loop through N particles in a shuffled order
        for (unsigned int cur_particle = 0; cur_particle < this->m_pdata->getN(); cur_particle++)
            {
            unsigned int i = this->m_update_order[cur_particle];

            // read in the current position and orientation
            Scalar4 postype_i = h_postype.data[i];
            Scalar4 orientation_i = h_orientation.data[i];
            vec3<Scalar> pos_i = vec3<Scalar>(postype_i);

            #ifdef ENABLE_MPI
            if (this->m_comm)
                {
                // only move particle if active
                if (!isActive(make_scalar3(postype_i.x, postype_i.y, postype_i.z), box, ghost_fraction))
                    continue;
                }
            #endif

            // make a trial move for i
            hoomd::detail::Saru rng_i(i, this->m_seed + this->m_exec_conf->getRank()*this->m_nselect + i_nselect, timestep);
            int typ_i = __scalar_as_int(postype_i.w);
            Shape shape_i(quat<Scalar>(orientation_i), this->m_params[typ_i]);
            unsigned int move_type_select = rng_i.u32() & 0xffff;
            bool move_type_translate = !shape_i.hasOrientation() || (move_type_select < this->m_move_ratio);

            Shape shape_old(quat<Scalar>(orientation_i), this->m_params[typ_i]);
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
                if (this->m_comm)
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

                move_rotate(shape_i.orientation, rng_i, h_a.data[typ_i], ndim);
                }

            // check for overlaps with neighboring particle's positions
            bool overlap=false;
            OverlapReal r_cut_patch = 0;

            if (this->m_patch && !this->m_patch_log)
                {
                r_cut_patch = this->m_patch->getRCut() + 0.5*this->m_patch->getAdditiveCutoff(typ_i);
                }
            OverlapReal R_query = std::max(shape_i.getCircumsphereDiameter()/OverlapReal(2.0), r_cut_patch-this->getMinCoreDiameter()/(OverlapReal)2.0);
            detail::AABB aabb_i_local = detail::AABB(vec3<Scalar>(0,0,0),R_query);

            // patch + field interaction deltaU
            double patch_field_energy_diff = 0;

            // All image boxes (including the primary)
            const unsigned int n_images = this->m_image_list.size();
            for (unsigned int cur_image = 0; cur_image < n_images; cur_image++)
                {
                vec3<Scalar> pos_i_image = pos_i + this->m_image_list[cur_image];
                detail::AABB aabb = aabb_i_local;
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
                                Shape shape_j(quat<Scalar>(orientation_j), this->m_params[typ_j]);

                                counters.overlap_checks++;

                                // check circumsphere overlap
                                OverlapReal rsq = dot(r_ij,r_ij);
                                OverlapReal DaDb = shape_i.getCircumsphereDiameter() + shape_j.getCircumsphereDiameter();
                                bool circumsphere_overlap = (rsq*OverlapReal(4.0) <= DaDb * DaDb);

                                Scalar r_cut_ij = 0.0;
                                if (this->m_patch)
                                    r_cut_ij = r_cut_patch + 0.5*this->m_patch->getAdditiveCutoff(typ_j);

                                if (h_overlaps.data[this->m_overlap_idx(typ_i,typ_j)]
                                    && circumsphere_overlap
                                    && test_overlap(r_ij, shape_i, shape_j, counters.overlap_err_count))
                                    {
                                    overlap = true;
                                    break;
                                    }
                                // If there is no overlap and m_patch is not NULL, calculate energy
                                else if (this->m_patch && !this->m_patch_log && rsq <= r_cut_ij*r_cut_ij)
                                    {
                                    patch_field_energy_diff -= this->m_patch->energy(r_ij, typ_i,
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
                        cur_node_idx += this->m_aabb_tree.getNodeSkip(cur_node_idx);
                        }

                    if (overlap)
                        break;
                    }  // end loop over AABB nodes

                if (overlap)
                    break;

                } // end loop over images

            // whether the move is accepted
            bool accept = !overlap;

            // In most cases checking patch energy should be cheaper than computing
            // depletants, so do that first. Calculate old patch energy only if
            // m_patch not NULL and no overlaps. Note that we are computing U_old-U_new
            // and then exponentiating directly (rather than exp(-(U_new-U_old)))
            if (this->m_patch && !this->m_patch_log && accept)
                {
                for (unsigned int cur_image = 0; cur_image < n_images; cur_image++)
                    {
                    vec3<Scalar> pos_i_image = pos_old + this->m_image_list[cur_image];
                    detail::AABB aabb = aabb_i_local;
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
                                    Shape shape_j(quat<Scalar>(orientation_j), this->m_params[typ_j]);
                                    if (dot(r_ij,r_ij) <= r_cut_patch*r_cut_patch)
                                        patch_field_energy_diff += this->m_patch->energy(r_ij,
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
                            cur_node_idx += this->m_aabb_tree.getNodeSkip(cur_node_idx);
                            }
                        }  // end loop over AABB nodes
                    } // end loop over images
                } // end if (m_patch)

            // Add external energetic contribution
            if (this->m_external)
                {
                patch_field_energy_diff -= this->m_external->energydiff(i, pos_old, shape_old, pos_i, shape_i);
                }

            accept = accept && (rng_i.d() < slow::exp(patch_field_energy_diff));

            // If no overlaps and Metropolis criterion is met, check if it is
            // invalidated by depletants.
            if (accept && h_overlaps.data[this->m_overlap_idx(m_type, typ_i)])
                {
                if (m_method == 0)
                    {
                    // check free volume in circumsphere
                    #ifndef ENABLE_TBB
                    accept = checkDepletantCircumsphere(i, pos_i, shape_i, typ_i, h_d_max.data[typ_i], h_d_min.data[typ_i], h_postype.data, h_orientation.data, h_overlaps.data, counters, implicit_counters, rng_poisson, rng_i);
                    #else
                    accept = checkDepletantCircumsphere(i, pos_i, shape_i, typ_i, h_d_max.data[typ_i], h_d_min.data[typ_i], h_postype.data, h_orientation.data, h_overlaps.data, counters, implicit_counters, rng_i, rng_parallel, rng_parallel_mt);
                    #endif
                    }
                else
                    {
                    // check overlap volume only
                    #ifndef ENABLE_TBB
                    accept = checkDepletantOverlap(i, pos_i, shape_i, typ_i, h_d_max.data[typ_i], h_d_min.data[typ_i], h_postype.data, h_orientation.data, h_overlaps.data, counters, implicit_counters, rng_poisson, rng_i);
                    #else
                    accept = checkDepletantOverlap(i, pos_i, shape_i, typ_i, h_d_max.data[typ_i], h_d_min.data[typ_i], h_postype.data, h_orientation.data, h_overlaps.data, counters, implicit_counters, rng_i, rng_parallel, rng_parallel_mt);
                    #endif
                    }
                } // end depletant placement

            // if the move is accepted
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
                this->m_aabb_tree.update(i, aabb);

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
        ArrayHandle<Scalar4> h_postype(this->m_pdata->getPositions(), access_location::host, access_mode::readwrite);
        ArrayHandle<int3> h_image(this->m_pdata->getImages(), access_location::host, access_mode::readwrite);

        // wrap particles back into box
        for (unsigned int i = 0; i < this->m_pdata->getN(); i++)
            {
            box.wrap(h_postype.data[i], h_image.data[i]);
            }
        }

    // perform the grid shift
    #ifdef ENABLE_MPI
    if (this->m_comm)
        {
        ArrayHandle<Scalar4> h_postype(this->m_pdata->getPositions(), access_location::host, access_mode::readwrite);
        ArrayHandle<int3> h_image(this->m_pdata->getImages(), access_location::host, access_mode::readwrite);

        // precalculate the grid shift
        hoomd::detail::Saru rng(timestep, this->m_seed, 0xf4a3210e);
        Scalar3 shift = make_scalar3(0,0,0);
        shift.x = rng.s(-this->m_nominal_width/Scalar(2.0),this->m_nominal_width/Scalar(2.0));
        shift.y = rng.s(-this->m_nominal_width/Scalar(2.0),this->m_nominal_width/Scalar(2.0));
        if (this->m_sysdef->getNDimensions() == 3)
            {
            shift.z = rng.s(-this->m_nominal_width/Scalar(2.0),this->m_nominal_width/Scalar(2.0));
            }
        for (unsigned int i = 0; i < this->m_pdata->getN(); i++)
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
    this->communicate(true);

    // all particle have been moved, the aabb tree is now invalid
    this->m_aabb_tree_invalid = true;
    }


/*! \param i The particle id in the list
    \param pos_i Particle position being tested
    \param shape_i Particle shape (including orientation) being tested
    \param typ_i Type of the particle being tested
    \param d_max The maximum sphere for test depletant exclusion
    \param d_min The minimum sphere in which depletants may be inserted
    \param h_postype Pointer to GPUArray containing particle positions
    \param h_orientation Pointer to GPUArray containing particle orientations
    \param h_overlaps Pointer to GPUArray containing interaction matrix
    \param hpmc_counters_t&  Pointer to current counters
    \param hpmc_implicit_counters_t&  Pointer to current implicit counters
    \param rng_poisson The RNG used within this algorithm
    \param rng_i The RNG used for evaluating the Metropolis criterion

    In order to determine whether or not moves are accepted, particle positions are checked against a randomly generated set of depletant positions.
    In principle this function should enable multiple depletant modes, although at present only one (cirumsphere) has been implemented here.

    NOTE: To avoid numerous acquires and releases of GPUArrays, ArrayHandles are passed directly into this const function.
    */
#ifndef ENABLE_TBB
template<class Shape>
inline bool IntegratorHPMCMonoImplicit<Shape>::checkDepletantCircumsphere(unsigned int i, vec3<Scalar> pos_i, Shape shape_i, unsigned int typ_i, Scalar d_max, Scalar d_min, Scalar4 *h_postype, Scalar4 *h_orientation, unsigned int *h_overlaps, hpmc_counters_t& counters, hpmc_implicit_counters_t& implicit_counters, std::mt19937& rng_poisson, hoomd::detail::Saru& rng_i)
#else
template<class Shape>
inline bool IntegratorHPMCMonoImplicit<Shape>::checkDepletantCircumsphere(unsigned int i, vec3<Scalar> pos_i, Shape shape_i, unsigned int typ_i, Scalar d_max, Scalar d_min, Scalar4 *h_postype, Scalar4 *h_orientation, unsigned int *h_overlaps, hpmc_counters_t& counters, hpmc_implicit_counters_t& implicit_counters, hoomd::detail::Saru& rng_i, tbb::enumerable_thread_specific< hoomd::detail::Saru >& rng_parallel, tbb::enumerable_thread_specific<std::mt19937>& rng_parallel_mt)
#endif
    {
    // log of acceptance probability
    Scalar lnb(0.0);

    // Whether or not any overlaps are detected
    unsigned int zero = 0;

    // The trial move is valid. Now generate random depletant particles in a sphere
    // of radius (d_max+d_depletant+move size)/2.0 around the original particle position

    // draw number from Poisson distribution
    #ifdef ENABLE_TBB
    std::mt19937& rng_poisson = rng_parallel_mt.local();
    #endif

    unsigned int n = 0;
    if (m_lambda[typ_i] > Scalar(0.0))
        {
        std::poisson_distribution<unsigned int> poisson(m_lambda[typ_i]);
        n = poisson(rng_poisson);
        }

    #ifdef ENABLE_TBB
    std::atomic<unsigned int> n_overlap_checks(0);
    std::atomic<unsigned int> overlap_err_count(0);
    std::atomic<unsigned int> insert_count(0);
    std::atomic<unsigned int> reinsert_count(0);
    std::atomic<unsigned int> free_volume_count(0);
    std::atomic<unsigned int> overlap_count(0);
    #else
    unsigned int n_overlap_checks = 0;
    unsigned int overlap_err_count = 0;
    unsigned int insert_count = 0;
    unsigned int reinsert_count = 0;
    unsigned int free_volume_count = 0;
    unsigned int overlap_count = 0;
    #endif

    volatile bool flag=false;
    const unsigned int n_images = this->m_image_list.size();

    #ifdef ENABLE_TBB
    tbb::parallel_for((unsigned int) 0, (unsigned int) n, [&] (unsigned int k)
    #else
    for (unsigned int k = 0; k < n; ++k)
    #endif
        {
        if (flag)
            {
            #ifndef ENABLE_TBB
            break;
            #else
            return;
            #endif
            }
        insert_count++;

        // generate a random depletant coordinate and orientation in the sphere around the new position
        vec3<Scalar> pos_test;
        quat<Scalar> orientation_test;

        #ifdef ENABLE_TBB
        hoomd::detail::Saru& my_rng = rng_parallel.local();
        #else
        hoomd::detail::Saru& my_rng = rng_i;
        #endif

        generateDepletant(my_rng, pos_i, d_max, d_min, pos_test,
            orientation_test, this->m_params[m_type]);
        Shape shape_test(orientation_test, this->m_params[m_type]);

        detail::AABB aabb_test_local = shape_test.getAABB(vec3<Scalar>(0,0,0));

        bool overlap_depletant = false;

        // Check if the new configuration of particle i generates an overlap
        for (unsigned int cur_image = 0; cur_image < n_images; cur_image++)
            {
            vec3<Scalar> pos_test_image = pos_test + this->m_image_list[cur_image];
            detail::AABB aabb = aabb_test_local;
            aabb.translate(pos_test_image);

            vec3<Scalar> r_ij = pos_i - pos_test_image;

            n_overlap_checks++;

            // check circumsphere overlap
            OverlapReal rsq = dot(r_ij,r_ij);
            OverlapReal DaDb = shape_test.getCircumsphereDiameter() + shape_i.getCircumsphereDiameter();
            bool circumsphere_overlap = (rsq*OverlapReal(4.0) <= DaDb * DaDb);

            if (circumsphere_overlap)
                {
                unsigned int err = 0;
                if (test_overlap(r_ij, shape_test, shape_i, err))
                    {
                    overlap_depletant = true;
                    overlap_count++;
                    break;
                    }
                if (err) overlap_err_count += err;
                }
            }

        // If the depletant overlaps the current configuration,
        // we will still accept if the depletant overlaps the old
        // configuration (indicating an invalid depletant placement).
        // We check for this case now.
        if (overlap_depletant)
            {
            // check against overlap with old position
            bool overlap_old = false;

            // Check if the old configuration of particle i generates an overlap
            for (unsigned int cur_image = 0; cur_image < n_images; cur_image++)
                {
                vec3<Scalar> pos_test_image = pos_test + this->m_image_list[cur_image];
                vec3<Scalar> r_ij = vec3<Scalar>(h_postype[i]) - pos_test_image;

                n_overlap_checks++;

                // check circumsphere overlap
                Shape shape_i_old(quat<Scalar>(h_orientation[i]), this->m_params[typ_i]);
                OverlapReal rsq = dot(r_ij,r_ij);
                OverlapReal DaDb = shape_test.getCircumsphereDiameter() + shape_i_old.getCircumsphereDiameter();
                bool circumsphere_overlap = (rsq*OverlapReal(4.0) <= DaDb * DaDb);

                if (h_overlaps[this->m_overlap_idx(m_type, typ_i)])
                    {
                    unsigned int err = 0;
                    if(circumsphere_overlap
                        && test_overlap(r_ij, shape_test, shape_i_old, err))
                        {
                        overlap_old = true;
                        break;
                        }
                    if (err) overlap_err_count += err;
                    }
                }

            // If the old configuration of particle i does not generate an
            // overlap, check overlaps against all neighboring particles in the
            // old configuration.
            if (!overlap_old)
                {
                // All image boxes (including the primary)
                const unsigned int n_images = this->m_image_list.size();
                for (unsigned int cur_image = 0; cur_image < n_images; cur_image++)
                    {
                    vec3<Scalar> pos_test_image = pos_test + this->m_image_list[cur_image];
                    detail::AABB aabb = aabb_test_local;
                    aabb.translate(pos_test_image);

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

                                    // we checked ptl i first
                                    if (i == j) continue;

                                    Scalar4 postype_j;
                                    Scalar4 orientation_j;

                                    // load the old position and orientation of the j particle
                                    postype_j = h_postype[j];
                                    orientation_j = h_orientation[j];

                                    // put particles in coordinate system of particle i
                                    vec3<Scalar> r_ij = vec3<Scalar>(postype_j) - pos_test_image;

                                    unsigned int typ_j = __scalar_as_int(postype_j.w);
                                    Shape shape_j(quat<Scalar>(orientation_j), this->m_params[typ_j]);

                                    n_overlap_checks++;

                                    // check circumsphere overlap
                                    OverlapReal rsq = dot(r_ij,r_ij);
                                    OverlapReal DaDb = shape_test.getCircumsphereDiameter() + shape_j.getCircumsphereDiameter();
                                    bool circumsphere_overlap = (rsq*OverlapReal(4.0) <= DaDb * DaDb);

                                    if (h_overlaps[this->m_overlap_idx(m_type,typ_j)])
                                        {
                                        unsigned int err = 0;
                                        if (circumsphere_overlap
                                            && test_overlap(r_ij, shape_test, shape_j, err))
                                            {
                                            // depletant is ignored for any overlap in the old configuration
                                            overlap_old = true;
                                            break;
                                            }
                                        if (err) overlap_err_count += err;
                                        }
                                    }
                                }
                            }
                        else
                            {
                            // skip ahead
                            cur_node_idx += this->m_aabb_tree.getNodeSkip(cur_node_idx);
                            }

                        if (overlap_old)
                            break;
                        }  // end loop over AABB nodes

                    if (overlap_old)
                        break;
                    } // end loop over images
                }

            if (!overlap_old)
                {
                //TODO: Fix this counter to work anytime it doesn't overlap in the old configuration
                //It should be adding to free volume any time the old config doesn't overlap even if the new one does, but here it's not doing anything any more
                free_volume_count++;
                }
            else
                {
                // the depletant overlap doesn't count since it was already overlapping
                // in the old configuration
                overlap_depletant = false;
                }
            }

        if (overlap_depletant && !m_n_trial)
            {
            zero = 1;
            // break out of loop
            flag = true;
            }
        else if (overlap_depletant && m_n_trial)
            {
            const typename Shape::param_type& params_depletant = this->m_params[m_type];

            // Number of successful depletant insertions in new configuration
            unsigned int n_success_new = 0;

            // Number of allowed insertion trials (those which overlap with colloid at old position)
            unsigned int n_overlap_shape_new = 0;

            // diameter (around origin) in which we are guaruanteed to intersect with the shape
            Scalar delta_insphere = Scalar(2.0)*shape_i.getInsphereRadius();

            // same for old reverse move. Because we have already sampled one successful insertion
            // that overlaps with the colloid at the new position, we increment by one (super-detailed
            // balance)
            unsigned int n_success_old = 1;
            unsigned int n_overlap_shape_old = 1;

            Scalar4& postype_i_old = h_postype[i];
            vec3<Scalar> pos_i_old(postype_i_old);
            quat<Scalar> orientation_i_old(h_orientation[i]);

            for (unsigned int l = 0; l < m_n_trial; ++l)
                {
                // generate a random depletant position and orientation
                // in both the old and the new configuration of the colloid particle
                vec3<Scalar> pos_depletant_old, pos_depletant_new;
                quat<Scalar> orientation_depletant_old, orientation_depletant_new;

                // try moving the overlapping depletant in the excluded volume
                // such that it overlaps with the particle at the old position
                generateDepletantRestricted(my_rng, pos_i_old, d_max, delta_insphere,
                    pos_depletant_new, orientation_depletant_new, params_depletant, pos_i);

                reinsert_count++;

                Shape shape_depletant_new(orientation_depletant_new, params_depletant);
                const typename Shape::param_type& params_i = this->m_params[__scalar_as_int(postype_i_old.w)];

                bool overlap_shape = false;
                    {
                    unsigned int err = 0, checks = 0;
                    if (insertDepletant(pos_depletant_new, shape_depletant_new, i, this->m_params.data(), h_overlaps, typ_i,
                        h_postype, h_orientation, pos_i, shape_i.orientation, params_i,
                        checks, err, overlap_shape, false))
                        {
                        n_success_new++;
                        }
                    if (err) overlap_err_count += err;
                    if (checks) n_overlap_checks += checks;
                    }

                if (overlap_shape)
                    {
                    // depletant overlaps with colloid at old position
                    n_overlap_shape_new++;
                    }

                if (l >= 1)
                    {
                    // as above, in excluded volume sphere at new position
                    generateDepletantRestricted(my_rng, pos_i, d_max, delta_insphere,
                        pos_depletant_old, orientation_depletant_old, params_depletant, pos_i_old);
                    Shape shape_depletant_old(orientation_depletant_old, params_depletant);
                        {
                        unsigned int err = 0, checks = 0;
                        if (insertDepletant(pos_depletant_old, shape_depletant_old, i, this->m_params.data(), h_overlaps, typ_i,
                            h_postype, h_orientation, pos_i, shape_i.orientation, params_i,
                            checks, err, overlap_shape, true))
                            {
                            n_success_old++;
                            }
                        if (err) overlap_err_count += err;
                        if (checks) n_overlap_checks += checks;
                        }

                    if (overlap_shape)
                        {
                        // depletant overlaps with colloid at new position
                        n_overlap_shape_old++;
                        }
                    reinsert_count++;
                    }

                n_overlap_checks += counters.overlap_checks;
                overlap_err_count += counters.overlap_err_count;
                } // end loop over re-insertion attempts

            if (n_success_new != 0)
                {
                lnb += log((Scalar)n_success_new/(Scalar)n_overlap_shape_new);
                lnb -= log((Scalar)n_success_old/(Scalar)n_overlap_shape_old);
                }
            else
                {
                zero = 1;
                // break out of loop
                flag = true;
                }
            } // end if depletant overlap

        } // end loop over depletants
    #ifdef ENABLE_TBB
        );
    #endif

    // increment counters
    counters.overlap_checks += n_overlap_checks;
    counters.overlap_err_count += overlap_err_count;
    implicit_counters.insert_count += insert_count;
    implicit_counters.free_volume_count += free_volume_count;
    implicit_counters.overlap_count += overlap_count;
    implicit_counters.reinsert_count += reinsert_count;

    //// apply acceptance criterium
    //return (!zero) ? (rng_i.f() < exp(lnb)) : false;

    // apply acceptance criterium
    bool accept;
    if (!zero)
        {
        accept = rng_i.f() < exp(lnb);
        }
    else
        {
        accept = false;
        }
    return accept;
    }


/*! \param i The particle id in the list
    \param pos_i Particle position being tested
    \param shape_i Particle shape (including orientation) being tested
    \param typ_i Type of the particle being tested
    \param d_max The maximum sphere for test depletant exclusion
    \param d_min The minimum sphere in which depletants may be inserted
    \param h_postype Pointer to GPUArray containing particle positions
    \param h_orientation Pointer to GPUArray containing particle orientations
    \param h_overlaps Pointer to GPUArray containing interaction matrix
    \param hpmc_counters_t&  Pointer to current counters
    \param hpmc_implicit_counters_t&  Pointer to current implicit counters
    \param rng_poisson The RNG used within this algorithm
    \param rng_i The RNG used for evaluating the Metropolis criterion

    In order to determine whether or not moves are accepted, particle positions are checked against a randomly generated set of depletant positions.
    In principle this function should enable multiple depletant modes, although at present only one (cirumsphere) has been implemented here.

    NOTE: To avoid numerous acquires and releases of GPUArrays, ArrayHandles are passed directly into this const function.
    */
#ifndef ENABLE_TBB
template<class Shape>
inline bool IntegratorHPMCMonoImplicit<Shape>::checkDepletantOverlap(unsigned int i, vec3<Scalar> pos_i, Shape shape_i, unsigned int typ_i, Scalar d_max, Scalar d_min, Scalar4 *h_postype, Scalar4 *h_orientation, unsigned int *h_overlaps, hpmc_counters_t& counters, hpmc_implicit_counters_t& implicit_counters, std::mt19937& rng_poisson, hoomd::detail::Saru& rng_i)
#else
template<class Shape>
inline bool IntegratorHPMCMonoImplicit<Shape>::checkDepletantOverlap(unsigned int i, vec3<Scalar> pos_i, Shape shape_i, unsigned int typ_i, Scalar d_max, Scalar d_min, Scalar4 *h_postype, Scalar4 *h_orientation, unsigned int *h_overlaps, hpmc_counters_t& counters, hpmc_implicit_counters_t& implicit_counters, hoomd::detail::Saru& rng_i, tbb::enumerable_thread_specific< hoomd::detail::Saru >& rng_parallel, tbb::enumerable_thread_specific<std::mt19937>& rng_parallel_mt)
#endif
    {
    // List of particles whose circumspheres intersect particle i's excluded-volume circumsphere
    std::vector<unsigned int> intersect_i;

    // List of particle images that intersect
    std::vector<unsigned int> image_i;

    // find neighbors whose circumspheres overlap particle i's circumsphere in the old configuration
    // Here, circumsphere refers to the sphere around the depletant-excluded volume
    detail::AABB aabb_local(vec3<Scalar>(0,0,0), Scalar(0.5)*shape_i.getCircumsphereDiameter()+m_d_dep);
    vec3<Scalar> pos_i_old(h_postype[i]);

    const unsigned int n_images = this->m_image_list.size();

    bool accept = true;

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
                        vec3<Scalar> r_ij = vec3<Scalar>(postype_j) - pos_i_old_image;

                        unsigned int typ_j = __scalar_as_int(postype_j.w);
                        Shape shape_j(quat<Scalar>(), this->m_params[typ_j]);

                        // check circumsphere overlap
                        OverlapReal rsq = dot(r_ij,r_ij);
                        OverlapReal DaDb = shape_i.getCircumsphereDiameter() + shape_j.getCircumsphereDiameter() + 2*m_d_dep;
                        bool circumsphere_overlap = (rsq*OverlapReal(4.0) <= DaDb * DaDb);

                        if (h_overlaps[this->m_overlap_idx(m_type,typ_j)] && circumsphere_overlap)
                            {
                            intersect_i.push_back(j);
                            image_i.push_back(cur_image);
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
    // between two 'lenses' and if so, only accepting it if it was generated from neighbor j_min

    #ifdef ENABLE_TBB
    std::atomic<unsigned int> n_overlap_checks(0);
    std::atomic<unsigned int> overlap_err_count(0);
    std::atomic<unsigned int> insert_count(0);
    #else
    unsigned int n_overlap_checks = 0;
    unsigned int overlap_err_count = 0;
    unsigned int insert_count = 0;
    #endif

    // for every pairwise intersection
    #ifdef ENABLE_TBB
    tbb::parallel_for((unsigned int)0, (unsigned int)intersect_i.size(), [&](unsigned int k)
    #else
    for (unsigned int k = 0; k < intersect_i.size(); ++k)
    #endif
        {
        unsigned int j = intersect_i[k];
        vec3<Scalar> ri = pos_i_old;
        Scalar4 postype_j = h_postype[j];
        vec3<Scalar> rj = vec3<Scalar>(postype_j);
        Scalar Ri = Scalar(0.5)*(shape_i.getCircumsphereDiameter()+m_d_dep);
        Shape shape_j(quat<Scalar>(), this->m_params[__scalar_as_int(postype_j.w)]);
        Scalar Rj = Scalar(0.5)*(shape_j.getCircumsphereDiameter()+m_d_dep);

        vec3<Scalar> rij(rj-ri - this->m_image_list[image_i[k]]);
        Scalar d = sqrt(dot(rij,rij));

        // whether the intersection is the entire (smaller) sphere
        bool sphere = false;
        Scalar V;
        Scalar Vcap_i(0.0);
        Scalar Vcap_j(0.0);
        Scalar hi(0.0);
        Scalar hj(0.0);

        if (d + Ri - Rj < 0 || d + Rj - Ri < 0)
            {
            sphere = true;
            V = (Ri < Rj) ? Scalar(M_PI*4.0/3.0)*Ri*Ri*Ri : Scalar(M_PI*4.0/3.0)*Rj*Rj*Rj;
            }
        else
            {
            // heights spherical caps that constitute the intersection volume
            hi = (Rj*Rj - (d-Ri)*(d-Ri))/(2*d);
            hj = (Ri*Ri - (d-Rj)*(d-Rj))/(2*d);

            // volumes of spherical caps
            Vcap_i = Scalar(M_PI/3.0)*hi*hi*(3*Ri-hi);
            Vcap_j = Scalar(M_PI/3.0)*hj*hj*(3*Rj-hj);

            // volume of intersection
            V = Vcap_i + Vcap_j;
            }

        // chooose the number of depletants in the intersection volume
        std::poisson_distribution<unsigned int> poisson(m_n_R*V);
        #ifdef ENABLE_TBB
        std::mt19937& rng_poisson = rng_parallel_mt.local();
        hoomd::detail::Saru& my_rng = rng_parallel.local();
        #else
        hoomd::detail::Saru& my_rng = rng_i;
        #endif

        unsigned int n = poisson(rng_poisson);

        // for every depletant
        for (unsigned int l = 0; l < n; ++l)
            {
            insert_count++;

            vec3<Scalar> pos_test;
            if (!sphere)
                {
                // choose one of the two caps randomly, with a weight proportional to their volume
                Scalar s = my_rng.template s<Scalar>();
                bool cap_i = s < Vcap_i/V;

                // generate a depletant position in the spherical cap
                pos_test = cap_i ? generatePositionInSphericalCap(my_rng, ri, Ri, hi, rij)
                    : generatePositionInSphericalCap(my_rng, rj, Rj, hj, -rij)-this->m_image_list[image_i[k]];
                }
            else
                {
                // generate a random position in the smaller sphere
                if (Ri < Rj)
                    pos_test = generatePositionInSphere(my_rng, ri, Ri);
                else
                    pos_test = generatePositionInSphere(my_rng, rj, Rj) - this->m_image_list[image_i[k]];
                }

            Shape shape_test(quat<Scalar>(), this->m_params[m_type]);
            if (shape_test.hasOrientation())
                {
                shape_test.orientation = generateRandomOrientation(my_rng);
                }

            // check if depletant falls in other intersection volumes
            bool active = true;
            for (unsigned int m = 0; m < k; ++m)
                {
                unsigned int p = intersect_i[m];
                Scalar4 postype_p = h_postype[p];
                vec3<Scalar> rp = vec3<Scalar>(postype_p);
                Shape shape_p(quat<Scalar>(), this->m_params[__scalar_as_int(postype_p.w)]);

                vec3<Scalar> delta_r(pos_test + this->m_image_list[image_i[m]] - rp);
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
                Shape shape_i_old(quat<Scalar>(h_orientation[i]), this->m_params[typ_i]);

                OverlapReal rsq = dot(r_ij,r_ij);
                OverlapReal DaDb = shape_test.getCircumsphereDiameter() + shape_i_old.getCircumsphereDiameter();
                bool circumsphere_overlap = (rsq*OverlapReal(4.0) <= DaDb * DaDb);

                if (h_overlaps[this->m_overlap_idx(m_type, typ_i)])
                    {
                    n_overlap_checks++;
                    unsigned int err = 0;
                    if (circumsphere_overlap && test_overlap(r_ij, shape_test, shape_i_old, err))
                        {
                        overlap_old = true;
                        }
                    if (err) overlap_err_count += err;
                    }
                }

            // if not intersecting ptl i in old config, ignore
            if (!overlap_old) continue;

            // Check if the new configuration of particle i generates an overlap
            bool overlap_new = false;

                {
                vec3<Scalar> r_ij = pos_i - pos_test;

                OverlapReal rsq = dot(r_ij,r_ij);
                OverlapReal DaDb = shape_test.getCircumsphereDiameter() + shape_i.getCircumsphereDiameter();
                bool circumsphere_overlap = (rsq*OverlapReal(4.0) <= DaDb * DaDb);

                if (h_overlaps[this->m_overlap_idx(m_type, typ_i)])
                    {
                    n_overlap_checks++;
                    unsigned int err = 0;
                    if (circumsphere_overlap && test_overlap(r_ij, shape_test, shape_i, err))
                        {
                        overlap_new = true;
                        }
                    if (err) overlap_err_count += err;
                    }
                }

            if (overlap_new) continue;

            // does the depletant fall into the overlap volume with other particles?
            bool in_intersection_volume = false;

            for (unsigned int m = 0; m < intersect_i.size(); ++m)
                {
                // read in its position and orientation
                unsigned int j = intersect_i[m];

                // load the old position and orientation of the j particle
                Scalar4 postype_j = h_postype[j];
                Scalar4 orientation_j = h_orientation[j];

                vec3<Scalar> r_ij = vec3<Scalar>(postype_j) - pos_test - this->m_image_list[image_i[m]];

                unsigned int typ_j = __scalar_as_int(postype_j.w);
                Shape shape_j(quat<Scalar>(orientation_j), this->m_params[typ_j]);

                n_overlap_checks++;

                // check circumsphere overlap
                OverlapReal rsq = dot(r_ij,r_ij);
                OverlapReal DaDb = shape_test.getCircumsphereDiameter() + shape_j.getCircumsphereDiameter();
                bool circumsphere_overlap = (rsq*OverlapReal(4.0) <= DaDb * DaDb);

                unsigned int err = 0;
                if (h_overlaps[this->m_overlap_idx(m_type,typ_j)]
                    && circumsphere_overlap
                    && test_overlap(r_ij, shape_test, shape_j, err))
                    {
                    in_intersection_volume = true;
                    }

                if (err) overlap_err_count+=err;
                if (in_intersection_volume)
                    break;
                } // end loop over intersections

            // if not part of overlap volume in new config, reject
            if (in_intersection_volume)
                {
                accept = false;
                #ifndef ENABLE_TBB
                break;
                #endif
                }
            } // end loop over depletants

        #ifndef ENABLE_TBB
        if (!accept) break;
        #endif
        } // end loop over overlapping spheres
    #ifdef ENABLE_TBB
        );
    #endif

    // increment counters
    counters.overlap_checks += n_overlap_checks;
    counters.overlap_err_count += overlap_err_count;
    implicit_counters.insert_count += insert_count;

    return accept;
    }


/* \param rng The random number generator
 * \param pos_sphere Center of sphere
 * \param delta diameter of sphere
 * \param d_min Diameter of smaller sphere excluding depletant
 * \param pos Position of depletant (return value)
 * \param orientation ion of depletant (return value)
 * \param params_depletant Depletant parameters
 */
template<class Shape>
template<class RNG>
inline void IntegratorHPMCMonoImplicit<Shape>::generateDepletant(RNG& rng, vec3<Scalar> pos_sphere, Scalar delta,
    Scalar d_min, vec3<Scalar>& pos, quat<Scalar>& orientation, const typename Shape::param_type& params_depletant)
    {
    // draw a random vector in the excluded volume sphere of the colloid
    Scalar theta = rng.template s<Scalar>(Scalar(0.0),Scalar(2.0*M_PI));
    Scalar z = rng.template s<Scalar>(Scalar(-1.0),Scalar(1.0));

    // random normalized vector
    vec3<Scalar> n(fast::sqrt(Scalar(1.0)-z*z)*fast::cos(theta),fast::sqrt(Scalar(1.0)-z*z)*fast::sin(theta),z);

    // draw random radial coordinate in test sphere
    Scalar r3 = rng.template s<Scalar>(fast::pow(d_min/delta,Scalar(3.0)),Scalar(1.0));
    Scalar r = Scalar(0.5)*delta*fast::pow(r3,Scalar(1.0/3.0));

    // test depletant position
    vec3<Scalar> pos_depletant = pos_sphere+r*n;

    Shape shape_depletant(quat<Scalar>(), params_depletant);
    if (shape_depletant.hasOrientation())
        {
        orientation = generateRandomOrientation(rng);
        }
    pos = pos_depletant;
    }

/* \param rng The random number generator
 * \param pos_sphere Center of sphere
 * \param delta diameter of sphere
 * \param delta_other diameter of other sphere
 * \param pos Position of depletant (return value)
 * \param orientation ion of depletant (return value)
 * \param params_depletant Depletant parameters
 * \params pos_sphere_other Center of other sphere
 */
template<class Shape>
template<class RNG>
inline void IntegratorHPMCMonoImplicit<Shape>::generateDepletantRestricted(RNG& rng, vec3<Scalar> pos_sphere, Scalar delta,
    Scalar delta_other, vec3<Scalar>& pos, quat<Scalar>& orientation, const typename Shape::param_type& params_depletant,
    vec3<Scalar> pos_sphere_other)
    {
    vec3<Scalar> r_ij = pos_sphere - pos_sphere_other;
    Scalar d = fast::sqrt(dot(r_ij,r_ij));

    Scalar rmin(0.0);
    Scalar rmax = Scalar(0.5)*delta;

    Scalar ctheta_min(-1.0);
    bool do_rotate = false;
    if (d > Scalar(0.0) && delta_other > Scalar(0.0))
        {
        // draw a random direction in the bounded spherical shell
        Scalar ctheta = (delta_other*delta_other+Scalar(4.0)*d*d-delta*delta)/(Scalar(4.0)*delta_other*d);
        if (ctheta >= Scalar(-1.0) && ctheta < Scalar(1.0))
            {
            // true intersection, we can restrict angular sampling
            ctheta_min = ctheta;
            }

        // is there an intersection?
        if (Scalar(2.0)*d < delta+delta_other)
            {
            // sample in shell around smaller sphere
            rmin = delta_other/Scalar(2.0);
            rmax = d+delta/Scalar(2.0);
            do_rotate = true;
            }
        }

    // draw random radial coordinate in a spherical shell
    Scalar r3 = rng.template s<Scalar>(fast::pow(rmin/rmax,Scalar(3.0)),Scalar(1.0));
    Scalar r = rmax*fast::pow(r3,Scalar(1.0/3.0));

    // random direction in spherical shell
    Scalar z = rng.s(ctheta_min,Scalar(1.0));
    Scalar phi = Scalar(2.0*M_PI)*rng.template s<Scalar>();
    vec3<Scalar> n;
    if (do_rotate)
        {
        vec3<Scalar> u(r_ij/d);

        // normal vector
        vec3<Scalar> v(cross(u,vec3<Scalar>(0,0,1)));
        if (dot(v,v) < EPSILON)
            {
            v = cross(u,vec3<Scalar>(0,1,0));
            }
        v *= fast::rsqrt(dot(v,v));

        quat<Scalar> q(quat<Scalar>::fromAxisAngle(u,phi));
        n = z*u+(fast::sqrt(Scalar(1.0)-z*z))*rotate(q,v);
        }
    else
        {
        n = vec3<Scalar>(fast::sqrt(Scalar(1.0)-z*z)*fast::cos(phi),fast::sqrt(Scalar(1.0)-z*z)*fast::sin(phi),z);
        }

    // test depletant position
    pos = r*n;

    if (do_rotate)
        {
        // insert such that it potentially intersects the sphere, but not the other one
        pos += pos_sphere_other;
        }
    else
        {
        // insert in sphere
        pos += pos_sphere;
        }

    Shape shape_depletant(quat<Scalar>(), params_depletant);
    if (shape_depletant.hasOrientation())
        {
        orientation = generateRandomOrientation(rng);
        }
    }


/*! \param pos_depletant Depletant position
 * \param shape_depletant Depletant shape
 * \param idx Index of updated particle
 * \param h_overlaps Interaction matrix
 * \param typ_i type of updated particle
 * \param h_orientation ion array
 * \param pos_new New position of updated particle
 * \param orientation_new New orientation of updated particle
 * \param params_new New shape parameters of updated particle
 * \param counters HPMC overlap counters
 */
template<class Shape>
inline bool IntegratorHPMCMonoImplicit<Shape>::insertDepletant(vec3<Scalar>& pos_depletant,
    const Shape& shape_depletant, unsigned int idx, typename Shape::param_type *params, unsigned int *h_overlaps,
    unsigned int typ_i, Scalar4 *h_postype, Scalar4 *h_orientation, vec3<Scalar>  pos_new, quat<Scalar>& orientation_new,
    const typename Shape::param_type& params_new, unsigned int &n_overlap_checks,
    unsigned int &overlap_err_count, bool& overlap_shape, bool new_config)
    {
    overlap_shape=false;

    detail::AABB aabb_depletant_local = shape_depletant.getAABB(vec3<Scalar>(0,0,0));

    // now check if depletant overlaps with moved particle in the old configuration
    Shape shape_i(quat<Scalar>(), params_new);
    if (shape_i.hasOrientation())
        {
        if (! new_config)
            {
            // load old orientation
            Scalar4 orientation_i = h_orientation[idx];
            shape_i.orientation = quat<Scalar>(orientation_i);
            }
        else
            {
            shape_i.orientation = orientation_new;
            }
        }

    vec3<Scalar> pos_i;

    if (!new_config)
        {
        // load old position
        pos_i = vec3<Scalar>(h_postype[idx]);
        }
    else
        {
        pos_i = pos_new;
        }

    // only need to consider the (0,0,0) image
    detail::AABB aabb = aabb_depletant_local;
    aabb.translate(pos_depletant);

    // put particles in coordinate system of depletant
    vec3<Scalar> r_ij = pos_i - pos_depletant;

    n_overlap_checks++;

    // test circumsphere overlap
    OverlapReal rsq = dot(r_ij,r_ij);
    OverlapReal DaDb = shape_depletant.getCircumsphereDiameter() + shape_i.getCircumsphereDiameter();
    bool circumsphere_overlap = (rsq*OverlapReal(4.0) <= DaDb * DaDb);

    if (h_overlaps[this->m_overlap_idx(typ_i, m_type)] && circumsphere_overlap && test_overlap(r_ij, shape_depletant, shape_i, overlap_err_count))
        {
        overlap_shape = true;
        }

    // same, but for reverse move
    if (shape_i.hasOrientation())
        {
        if (new_config)
            {
            // load old orientation
            Scalar4 orientation_i = h_orientation[idx];
            shape_i.orientation = quat<Scalar>(orientation_i);
            }
        else
            {
            shape_i.orientation = orientation_new;
            }
        }

    if (new_config)
        {
        // load old position
        pos_i = vec3<Scalar>(h_postype[idx]);
        }
    else
        {
        pos_i = pos_new;
        }

    // only need to consider the (0,0,0) image
    aabb = aabb_depletant_local;
    aabb.translate(pos_depletant);

    // put particles in coordinate system of depletant
    r_ij = pos_i - pos_depletant;

    n_overlap_checks++;

    // test circumsphere overlap
    rsq = dot(r_ij,r_ij);
    DaDb = shape_depletant.getCircumsphereDiameter() + shape_i.getCircumsphereDiameter();
    circumsphere_overlap = (rsq*OverlapReal(4.0) <= DaDb * DaDb);

    // check for overlaps with neighboring particle's positions
    bool overlap=false;

    if (h_overlaps[this->m_overlap_idx(m_type, typ_i)] && circumsphere_overlap && test_overlap(r_ij, shape_depletant, shape_i, overlap_err_count))
        {
        // if we are already overlapping in the other configuration, this doesn't count as an insertion
        overlap = true;
        }

    if (!overlap && overlap_shape)
        {
        // All image boxes (including the primary)
        const unsigned int n_images = this->m_image_list.size();
        for (unsigned int cur_image = 0; cur_image < n_images; cur_image++)
            {
            vec3<Scalar> pos_depletant_image = pos_depletant + this->m_image_list[cur_image];
            detail::AABB aabb = aabb_depletant_local;
            aabb.translate(pos_depletant_image);

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


                            // load the position and orientation of the j particle
                            Scalar4 postype_j = h_postype[j];
                            vec3<Scalar> pos_j(postype_j);
                            Scalar4 orientation_j = h_orientation[j];
                            unsigned int type = __scalar_as_int(postype_j.w);
                            Shape shape_j(quat<Scalar>(orientation_j), params[type]);

                            if (j == idx)
                                {
                                // we have already excluded overlap with the moved particle above
                                continue;
                                }

                            // put particles in coordinate system of depletant
                            vec3<Scalar> r_ij = pos_j - pos_depletant_image;

                            n_overlap_checks++;

                            // check circumsphere overlap
                            OverlapReal rsq = dot(r_ij,r_ij);
                            OverlapReal DaDb = shape_depletant.getCircumsphereDiameter() + shape_j.getCircumsphereDiameter();
                            bool circumsphere_overlap = (rsq*OverlapReal(4.0) <= DaDb * DaDb);

                            if (h_overlaps[this->m_overlap_idx(type, m_type)] && circumsphere_overlap && test_overlap(r_ij, shape_depletant, shape_j, overlap_err_count))
                                {
                                overlap = true;
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

                if (overlap)
                    break;
                }  // end loop over AABB nodes

            if (overlap)
                break;

            } // end loop over images
        } // end if overlap with shape

    return overlap_shape && !overlap;
    }

/*! \param quantity Name of the log quantity to get
    \param timestep Current time step of the simulation
    \return the requested log quantity.
*/
template<class Shape>
Scalar IntegratorHPMCMonoImplicit<Shape>::getLogValue(const std::string& quantity, unsigned int timestep)
    {
    if (quantity == "hpmc_fugacity")
        {
        return (Scalar) m_n_R;
        }


    // n_trial is only valid when mode=circumsphere
    if (quantity == "hpmc_ntrial" && m_method == 0)
        {
        return (Scalar) m_n_trial;
        }

    hpmc_counters_t counters = IntegratorHPMC::getCounters(2);
    hpmc_implicit_counters_t implicit_counters = getImplicitCounters(2);

    if (quantity == "hpmc_insert_count")
        {
        // return number of depletant insertions per colloid
        if (counters.getNMoves() > 0)
            return (Scalar)implicit_counters.insert_count/(Scalar)counters.getNMoves();
        else
            return Scalar(0.0);
        }
    // These quantities are not logged when mode=overlap_regions
    if (m_method == 0)
        {
        if (quantity == "hpmc_reinsert_count")
            {
            // return number of overlapping depletants reinserted per colloid
            if (counters.getNMoves() > 0)
                return (Scalar)implicit_counters.reinsert_count/(Scalar)counters.getNMoves();
            else
                return Scalar(0.0);
            }
        if (quantity == "hpmc_free_volume_fraction")
            {
            // return fraction of free volume in depletant insertion sphere
            return (Scalar) implicit_counters.getFreeVolumeFraction();
            }
        if (quantity == "hpmc_overlap_fraction")
            {
            // return fraction of overlapping depletants after trial move
            return (Scalar) implicit_counters.getOverlapFraction();
            }
        if (quantity == "hpmc_configurational_bias_ratio")
            {
            // return fraction of overlapping depletants after trial move
            return (Scalar) implicit_counters.getConfigurationalBiasRatio();
            }
        }


    //nothing found -> pass on to base class
    return IntegratorHPMCMono<Shape>::getLogValue(quantity, timestep);
    }

/*! \param mode 0 -> Absolute count, 1 -> relative to the start of the run, 2 -> relative to the last executed step
    \return The current state of the acceptance counters

    IntegratorHPMCMonoImplicit maintains a count of the number of accepted and rejected moves since instantiation. getCounters()
    provides the current value. The parameter *mode* controls whether the returned counts are absolute, relative
    to the start of the run, or relative to the start of the last executed step.
*/
template<class Shape>
hpmc_implicit_counters_t IntegratorHPMCMonoImplicit<Shape>::getImplicitCounters(unsigned int mode)
    {
    ArrayHandle<hpmc_implicit_counters_t> h_counters(m_implicit_count, access_location::host, access_mode::read);
    hpmc_implicit_counters_t result;

    if (mode == 0)
        result = h_counters.data[0];
    else if (mode == 1)
        result = h_counters.data[0] - m_implicit_count_run_start;
    else
        result = h_counters.data[0] - m_implicit_count_step_start;

    #ifdef ENABLE_MPI
    if (this->m_comm)
        {
        // MPI Reduction to total result values on all ranks
        MPI_Allreduce(MPI_IN_PLACE, &result.insert_count, 1, MPI_LONG_LONG_INT, MPI_SUM, this->m_exec_conf->getMPICommunicator());
        if (m_method == 0)
            {
            MPI_Allreduce(MPI_IN_PLACE, &result.free_volume_count, 1, MPI_LONG_LONG_INT, MPI_SUM, this->m_exec_conf->getMPICommunicator());
            MPI_Allreduce(MPI_IN_PLACE, &result.overlap_count, 1, MPI_LONG_LONG_INT, MPI_SUM, this->m_exec_conf->getMPICommunicator());
            MPI_Allreduce(MPI_IN_PLACE, &result.reinsert_count, 1, MPI_LONG_LONG_INT, MPI_SUM, this->m_exec_conf->getMPICommunicator());
            }
        }
    #endif

    return result;
    }

/*! NPT simulations are not supported with implicit depletants

    (The Nmu_ptPT ensemble is instable)

    \returns false if resize results in overlaps
*/
template<class Shape>
bool IntegratorHPMCMonoImplicit<Shape>::attemptBoxResize(unsigned int timestep, const BoxDim& new_box)
    {
    this->m_exec_conf->msg->error() << "Nmu_pPT simulations are unsupported." << std::endl;
    throw std::runtime_error("Error during implicit depletant integration\n");
    }

//! Export this hpmc integrator to python
/*! \param name Name of the class in the exported python module
    \tparam Shape An instantiation of IntegratorHPMCMono<Shape> will be exported
*/
template < class Shape > void export_IntegratorHPMCMonoImplicit(pybind11::module& m, const std::string& name)
    {
    pybind11::class_<IntegratorHPMCMonoImplicit<Shape>, std::shared_ptr< IntegratorHPMCMonoImplicit<Shape> > >(m, name.c_str(),  pybind11::base< IntegratorHPMCMono<Shape> >())
        .def(pybind11::init< std::shared_ptr<SystemDefinition>, unsigned int, unsigned int>())
        .def("setDepletantDensity", &IntegratorHPMCMonoImplicit<Shape>::setDepletantDensity)
        .def("setDepletantType", &IntegratorHPMCMonoImplicit<Shape>::setDepletantType)
        .def("setNTrial", &IntegratorHPMCMonoImplicit<Shape>::setNTrial)
        .def("getNTrial", &IntegratorHPMCMonoImplicit<Shape>::getNTrial)
        .def("getImplicitCounters", &IntegratorHPMCMonoImplicit<Shape>::getImplicitCounters)
        ;

    }

//! Export the counters for depletants
inline void export_hpmc_implicit_counters(pybind11::module& m)
    {
    pybind11::class_< hpmc_implicit_counters_t >(m, "hpmc_implicit_counters_t")
    .def_readwrite("insert_count", &hpmc_implicit_counters_t::insert_count)
    .def_readwrite("reinsert_count", &hpmc_implicit_counters_t::reinsert_count)
    .def_readwrite("free_volume_count", &hpmc_implicit_counters_t::free_volume_count)
    .def_readwrite("overlap_count", &hpmc_implicit_counters_t::overlap_count)
    .def("getFreeVolumeFraction", &hpmc_implicit_counters_t::getFreeVolumeFraction)
    .def("getOverlapFraction", &hpmc_implicit_counters_t::getOverlapFraction)
    .def("getConfigurationalBiasRatio", &hpmc_implicit_counters_t::getConfigurationalBiasRatio)
    ;
    }
} // end namespace hpmc

#endif // __HPMC_MONO_IMPLICIT__H__
