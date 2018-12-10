// Copyright (c) 2009-2018 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.

#ifndef __HPMC_MONO_IMPLICIT__H__
#define __HPMC_MONO_IMPLICIT__H__

#include "IntegratorHPMCMono.h"
#include "hoomd/Autotuner.h"

#include <random>
#include <cfloat>
#include <sstream>

/*! \file IntegratorHPMCMonoImplicit.h
    \brief Defines the template class for HPMC with implicit generated depletant solvent
    \note This header cannot be compiled by nvcc
*/

#ifdef NVCC
#error This header cannot be compiled by nvcc
#endif

#include <hoomd/extern/pybind/include/pybind11/pybind11.h>

#ifdef ENABLE_TBB
#include <tbb/tbb.h>
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
                              unsigned int seed);
        //! Destructor
        virtual ~IntegratorHPMCMonoImplicit();

        //! Set the depletant density in the free volume
        void setDepletantFugacity(unsigned int type, Scalar fugacity)
            {
            if (type >= this->m_pdata->getNTypes())
                throw std::runtime_error("Unknown type.");
            m_fugacity[type] = fugacity;
            }

        //! Returns the depletant density
        Scalar getDepletantFugacity(unsigned int type)
            {
            return m_fugacity[type];
            }

        //! Set quermass integration mode
        void setQuermassMode(bool enable_quermass)
            {
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
                throw std::runtime_error("This shape doesn's support setting a sweep radius to extend the surface out.\n");

            m_sweep_radius = sweep_radius;
            }

        //! Get the sweep radius
        Scalar getSweepRadius()
             {
             return m_sweep_radius;
             }

        //! Reset statistics counters
        virtual void resetStats()
            {
            IntegratorHPMCMono<Shape>::resetStats();
            ArrayHandle<hpmc_implicit_counters_t> h_counters(m_implicit_count, access_location::host, access_mode::read);
            for (unsigned int i = 0; i < this->m_pdata->getNTypes(); ++i)
                m_implicit_count_run_start[i] = h_counters.data[i];
            }

        //! Print statistics about the hpmc steps taken
        virtual void printStats()
            {
            IntegratorHPMCMono<Shape>::printStats();

            const std::vector<hpmc_implicit_counters_t>& result = getImplicitCounters(1);

            // reduce over all types
            unsigned long long int total_insert_count = 0;
            for (unsigned int i = 0; i < this->m_pdata->getNTypes(); ++i)
                total_insert_count += result[i].insert_count;

            double cur_time = double(this->m_clock.getTime()) / Scalar(1e9);

            this->m_exec_conf->msg->notice(2) << "-- Implicit depletants stats:" << "\n";
            this->m_exec_conf->msg->notice(2) << "Depletant insertions per second:          "
                << double(total_insert_count)/cur_time << "\n";

            // supply additional statistics
            for (unsigned int i = 0; i < this->m_pdata->getNTypes(); ++i)
                if (m_fugacity[i] != 0.0)
                    this->m_exec_conf->msg->notice(3) << "Type '" << this->m_pdata->getNameByType(i) << "': "
                        << double(result[i].insert_count)/cur_time << std::endl;
            }

        //! Get the current counter values
        std::vector<hpmc_implicit_counters_t> getImplicitCounters(unsigned int mode=0);

        /* \returns a list of provided quantities
        */
        std::vector< std::string > getProvidedLogQuantities()
            {
            // start with the integrator provided quantities
            std::vector< std::string > result = IntegratorHPMCMono<Shape>::getProvidedLogQuantities();

            // then add ours
            for (unsigned int typ=0; typ<this->m_pdata->getNTypes();typ++)
              {
              std::ostringstream tmp_str0;
              tmp_str0<<"hpmc_fugacity_"<< this->m_pdata->getNameByType(typ);
              result.push_back(tmp_str0.str());
              }

            result.push_back("hpmc_insert_count");

            return result;
            }

        //! Get the value of a logged quantity
        virtual Scalar getLogValue(const std::string& quantity, unsigned int timestep);

        //! Method to scale the box
        virtual bool attemptBoxResize(unsigned int timestep, const BoxDim& new_box);

        //! Slot to be called when number of types changes
        void slotNumTypesChange();

    protected:
        std::vector<Scalar> m_fugacity;                          //!< Average depletant number density in free volume, per type

        GPUArray<hpmc_implicit_counters_t> m_implicit_count;     //!< Counter of active cell cluster moves
        std::vector<hpmc_implicit_counters_t> m_implicit_count_run_start;     //!< Counter of active cell cluster moves at run start
        std::vector<hpmc_implicit_counters_t> m_implicit_count_step_start;    //!< Counter of active cell cluster moves at run start

        bool m_quermass;                                         //!< True if quermass integration mode is enabled
        Scalar m_sweep_radius;                                   //!< Radius of sphere to sweep shapes by

        //! Take one timestep forward
        virtual void update(unsigned int timestep);

        //! Test whether to reject the current particle move based on depletants
        #ifndef ENABLE_TBB
        inline bool checkDepletantOverlap(unsigned int i, vec3<Scalar> pos_i, Shape shape_i, unsigned int typ_i, Scalar4 *h_postype, Scalar4 *h_orientation, unsigned int *h_overlaps, hpmc_counters_t& counters, hpmc_implicit_counters_t *implicit_counters, std::mt19937& rng_poisson, hoomd::detail::Saru& rng_i);
        #else
        inline bool checkDepletantOverlap(unsigned int i, vec3<Scalar> pos_i, Shape shape_i, unsigned int typ_i, Scalar4 *h_postype, Scalar4 *h_orientation, unsigned int *h_overlaps, hpmc_counters_t& counters, hpmc_implicit_counters_t *implicit_counters, hoomd::detail::Saru& rng_i, tbb::enumerable_thread_specific< hoomd::detail::Saru >& rng_parallel, tbb::enumerable_thread_specific<std::mt19937>& rng_parallel_mt);
        #endif

        //! Set the nominal width appropriate for depletion interaction
        virtual void updateCellWidth();

        //! Generate a random depletant position in a sphere around a particle
        template<class RNG>
        inline void generateDepletant(RNG& rng, vec3<Scalar> pos_sphere, Scalar delta, Scalar d_min,
            vec3<Scalar>& pos, quat<Scalar>& orientation, const typename Shape::param_type& params_depletants);
    };

/*! \param sysdef System definition
    \param cl Cell list
    \param seed Random number generator seed
    */

template< class Shape >
IntegratorHPMCMonoImplicit< Shape >::IntegratorHPMCMonoImplicit(std::shared_ptr<SystemDefinition> sysdef,
                                                                   unsigned int seed)
    : IntegratorHPMCMono<Shape>(sysdef, seed), m_quermass(false), m_sweep_radius(0.0)
    {
    this->m_exec_conf->msg->notice(5) << "Constructing IntegratorHPMCImplicit" << std::endl;

    GPUArray<hpmc_implicit_counters_t> implicit_count(this->m_pdata->getNTypes(),this->m_exec_conf);
    m_implicit_count.swap(implicit_count);

        {
        ArrayHandle<hpmc_implicit_counters_t> h_implicit_count(m_implicit_count, access_location::host, access_mode::readwrite);
        for (unsigned int i = 0; i < this->m_pdata->getNTypes(); ++i)
            h_implicit_count.data[i] = hpmc_implicit_counters_t();
        }

    m_implicit_count_run_start.resize(this->m_pdata->getNTypes());
    m_implicit_count_step_start.resize(this->m_pdata->getNTypes());

    m_fugacity.resize(this->m_pdata->getNTypes(),0.0);
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

    unsigned int old_ntypes = m_implicit_count.getNumElements();
    m_implicit_count.resize(this->m_pdata->getNTypes());

        {
        ArrayHandle<hpmc_implicit_counters_t> h_implicit_count(m_implicit_count, access_location::host, access_mode::readwrite);
        for (unsigned int i = old_ntypes; i < this->m_pdata->getNTypes(); ++i)
            h_implicit_count.data[i] = hpmc_implicit_counters_t();
        }
    m_implicit_count_run_start.resize(this->m_pdata->getNTypes());
    m_implicit_count_step_start.resize(this->m_pdata->getNTypes());

    m_fugacity.resize(this->m_pdata->getNTypes(),0.0);
    }

template< class Shape >
void IntegratorHPMCMonoImplicit< Shape >::updateCellWidth()
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

    this->m_nominal_width += m_quermass ? 2.0*m_sweep_radius : max_d;

    // extend the image list by the depletant diameter, since we're querying
    // AABBs that are larger than the shape diameters themselves
    this->m_extra_image_width = m_quermass ? 2.0*m_sweep_radius : max_d;

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

    // get needed vars
    ArrayHandle<hpmc_counters_t> h_counters(this->m_count_total, access_location::host, access_mode::readwrite);
    hpmc_counters_t& counters = h_counters.data[0];

    ArrayHandle<hpmc_implicit_counters_t> h_implicit_counters(m_implicit_count, access_location::host, access_mode::readwrite);
    std::copy(h_implicit_counters.data, h_implicit_counters.data + this->m_pdata->getNTypes(), m_implicit_count_step_start.begin());

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

            // get AABB and extend
            detail::AABB aabb_i_local = shape_i.getAABB(vec3<Scalar>(0,0,0));
            vec3<Scalar> lower = aabb_i_local.getLower();
            vec3<Scalar> upper = aabb_i_local.getUpper();
            vec3<Scalar> center = Scalar(0.5)*(lower+upper);
            vec3<Scalar> lengths = Scalar(0.5)*(upper-lower);
            lengths.x = std::max(lengths.x, (Scalar) r_cut_patch);
            lengths.y = std::max(lengths.y, (Scalar) r_cut_patch);
            lengths.z = std::max(lengths.z, (Scalar) r_cut_patch);
            detail::AABB aabb_i_local_query = detail::AABB(center-lengths,center+lengths);

            // patch + field interaction deltaU
            double patch_field_energy_diff = 0;

            // All image boxes (including the primary)
            const unsigned int n_images = this->m_image_list.size();
            for (unsigned int cur_image = 0; cur_image < n_images; cur_image++)
                {
                vec3<Scalar> pos_i_image = pos_i + this->m_image_list[cur_image];
                detail::AABB aabb = aabb_i_local_query;
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

            // get old AABB and extend
            detail::AABB aabb_i_local_old = shape_old.getAABB(vec3<Scalar>(0,0,0));
            lower = aabb_i_local_old.getLower();
            upper = aabb_i_local_old.getUpper();
            center = Scalar(0.5)*(lower+upper);
            lengths = Scalar(0.5)*(upper-lower);
            lengths.x = std::max(lengths.x, (Scalar) r_cut_patch);
            lengths.y = std::max(lengths.y, (Scalar) r_cut_patch);
            lengths.z = std::max(lengths.z, (Scalar) r_cut_patch);
            aabb_i_local_old = detail::AABB(center-lengths,center+lengths);

            if (this->m_patch && !this->m_patch_log && accept)
                {
                for (unsigned int cur_image = 0; cur_image < n_images; cur_image++)
                    {
                    vec3<Scalar> pos_i_image = pos_old + this->m_image_list[cur_image];
                    detail::AABB aabb = aabb_i_local_old;
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

                // Add external energetic contribution
                if (this->m_external)
                    {
                    patch_field_energy_diff -= this->m_external->energydiff(i, pos_old, shape_old, pos_i, shape_i);
                    }

                // Update acceptance based on patch, will only be reached if overlap check succeeded
                accept = rng_i.d() < slow::exp(patch_field_energy_diff);
                } // end if (m_patch)

            // The trial move is valid, so check if it is invalidated by depletants
            if (accept)
                {
                #ifndef ENABLE_TBB
                accept = checkDepletantOverlap(i, pos_i, shape_i, typ_i, h_postype.data, h_orientation.data, h_overlaps.data, counters, h_implicit_counters.data, rng_poisson, rng_i);
                #else
                accept = checkDepletantOverlap(i, pos_i, shape_i, typ_i, h_postype.data, h_orientation.data, h_overlaps.data, counters, h_implicit_counters.data, rng_i, rng_parallel, rng_parallel_mt);
                #endif
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
inline bool IntegratorHPMCMonoImplicit<Shape>::checkDepletantOverlap(unsigned int i, vec3<Scalar> pos_i, Shape shape_i, unsigned int typ_i, Scalar4 *h_postype, Scalar4 *h_orientation, unsigned int *h_overlaps, hpmc_counters_t& counters, hpmc_implicit_counters_t *implicit_counters, std::mt19937& rng_poisson, hoomd::detail::Saru& rng_i)
#else
template<class Shape>
inline bool IntegratorHPMCMonoImplicit<Shape>::checkDepletantOverlap(unsigned int i, vec3<Scalar> pos_i, Shape shape_i, unsigned int typ_i, Scalar4 *h_postype, Scalar4 *h_orientation, unsigned int *h_overlaps, hpmc_counters_t& counters, hpmc_implicit_counters_t *implicit_counters, hoomd::detail::Saru& rng_i, tbb::enumerable_thread_specific< hoomd::detail::Saru >& rng_parallel, tbb::enumerable_thread_specific<std::mt19937>& rng_parallel_mt)
#endif
    {
    bool accept = true;

    const unsigned int n_images = this->m_image_list.size();

    Shape shape_old(quat<Scalar>(h_orientation[i]), this->m_params[typ_i]);
    detail::AABB aabb_i_local = shape_i.getAABB(vec3<Scalar>(0,0,0));
    detail::AABB aabb_i_local_old = shape_old.getAABB(vec3<Scalar>(0,0,0));

    #ifdef ENABLE_TBB
    std::vector< tbb::enumerable_thread_specific<hpmc_implicit_counters_t> > thread_implicit_counters(this->m_pdata->getNTypes());
    tbb::enumerable_thread_specific<hpmc_counters_t> thread_counters;
    #endif

    #ifdef ENABLE_TBB
    try {
    tbb::parallel_for((unsigned int)0, (unsigned int)this->m_pdata->getNTypes(), [&](unsigned int type)
    #else
    for (unsigned int type = 0; type < this->m_pdata->getNTypes(); ++type)
    #endif
        {
        if (!h_overlaps[this->m_overlap_idx(type, typ_i)] && !m_quermass)
            #ifdef ENABLE_TBB
            return;
            #else
            continue;
            #endif

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
            tbb::parallel_for((unsigned int)0, (unsigned int)intersect_i.size(), [&](unsigned int k)
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

                // intersectionAABB volume
                Scalar V =  (intersect_upper.x-intersect_lower.x)*(intersect_upper.y-intersect_lower.y)*(intersect_upper.z-intersect_lower.z);

                // chooose the number of depletants in the intersection volume
                std::poisson_distribution<unsigned int> poisson(m_fugacity[type]*V);
                #ifdef ENABLE_TBB
                std::mt19937& rng_poisson = rng_parallel_mt.local();
                #endif

                unsigned int n = poisson(rng_poisson);

                // for every depletant
                #ifdef ENABLE_TBB
                tbb::parallel_for((unsigned int)0, (unsigned int)n, [&](unsigned int l)
                #else
                for (unsigned int l = 0; l < n; ++l)
                #endif
                    {
                    #ifdef ENABLE_TBB
                    if (!accept)
                        return;
                    #endif

                    #ifdef ENABLE_TBB
                    hoomd::detail::Saru& my_rng = rng_parallel.local();
                    #else
                    hoomd::detail::Saru& my_rng = rng_i;
                    #endif

                    #ifdef ENABLE_TBB
                    thread_implicit_counters[type].local().insert_count++;
                    #else
                    implicit_counters[type].insert_count++;
                    #endif

                    vec3<Scalar> pos_test = generatePositionInAABB(my_rng, aabb_intersect);
                    Shape shape_test(quat<Scalar>(), this->m_params[type]);
                    if (shape_test.hasOrientation())
                        {
                        shape_test.orientation = generateRandomOrientation(my_rng);
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
                    #ifdef ENABLE_TBB
                        return;
                    #else
                        continue;
                    #endif

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
                        #ifdef ENABLE_TBB
                            return;
                        #else
                            continue;
                        #endif

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
                        #ifdef ENABLE_TBB
                            return;
                        #else
                            continue;
                        #endif
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
                    );
                #endif

                #ifndef ENABLE_TBB
                if (!accept)
                    break;
                #endif
                } // end loop over overlapping spheres
            #ifdef ENABLE_TBB
                );
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
            tbb::parallel_for((unsigned int)0, (unsigned int)intersect_i.size(), [&](unsigned int k)
            #else
            for (unsigned int k = 0; k < intersect_i.size(); ++k)
            #endif
                {
                #ifdef ENABLE_TBB
                if (!accept)
                    return;
                #endif

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
                Scalar V = (intersect_upper.x-intersect_lower.x)*(intersect_upper.y-intersect_lower.y)*(intersect_upper.z-intersect_lower.z);

                // chooose the number of depletants in the intersection volume
                std::poisson_distribution<unsigned int> poisson(-m_fugacity[type]*V);
                #ifdef ENABLE_TBB
                std::mt19937& rng_poisson = rng_parallel_mt.local();
                #endif

                unsigned int n = poisson(rng_poisson);

                // for every depletant
                #ifdef ENABLE_TBB
                tbb::parallel_for((unsigned int)0, (unsigned int)n, [&](unsigned int l)
                #else
                for (unsigned int l = 0; l < n; ++l)
                #endif
                    {
                    #ifdef ENABLE_TBB
                    hoomd::detail::Saru& my_rng = rng_parallel.local();
                    #else
                    hoomd::detail::Saru& my_rng = rng_i;
                    #endif

                    #ifdef ENABLE_TBB
                    thread_implicit_counters[type].local().insert_count++;
                    #else
                    implicit_counters[type].insert_count++;
                    #endif

                    vec3<Scalar> pos_test = generatePositionInAABB(my_rng, aabb_intersect);
                    Shape shape_test(quat<Scalar>(), this->m_params[type]);
                    if (shape_test.hasOrientation())
                        {
                        shape_test.orientation = generateRandomOrientation(my_rng);
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
                    #ifdef ENABLE_TBB
                        return;
                    #else
                        continue;
                    #endif

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
                        #ifdef ENABLE_TBB
                            return;
                        #else
                            continue;
                        #endif

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
                            // all is ok
                        #ifdef ENABLE_TBB
                            return;
                        #else
                            continue;
                        #endif
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
                    );
                #endif

                #ifndef ENABLE_TBB
                if (!accept) break;
                #endif
                } // end loop over overlapping spheres
            #ifdef ENABLE_TBB
                );
            #endif
            } // end depletant placement
        }
    #ifdef ENABLE_TBB
        ); } catch (bool b) { }
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

/*! \param quantity Name of the log quantity to get
    \param timestep Current time step of the simulation
    \return the requested log quantity.
*/
template<class Shape>
Scalar IntegratorHPMCMonoImplicit<Shape>::getLogValue(const std::string& quantity, unsigned int timestep)
    {
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
    return IntegratorHPMCMono<Shape>::getLogValue(quantity, timestep);
    }

/*! \param mode 0 -> Absolute count, 1 -> relative to the start of the run, 2 -> relative to the last executed step
    \return The current state of the acceptance counters

    IntegratorHPMCMonoImplicit maintains a count of the number of accepted and rejected moves since instantiation. getCounters()
    provides the current value. The parameter *mode* controls whether the returned counts are absolute, relative
    to the start of the run, or relative to the start of the last executed step.
*/
template<class Shape>
std::vector<hpmc_implicit_counters_t> IntegratorHPMCMonoImplicit<Shape>::getImplicitCounters(unsigned int mode)
    {
    ArrayHandle<hpmc_implicit_counters_t> h_counters(m_implicit_count, access_location::host, access_mode::read);
    std::vector<hpmc_implicit_counters_t> result(this->m_pdata->getNTypes());

    std::copy(h_counters.data, h_counters.data + this->m_pdata->getNTypes(), result.begin());
    if (mode == 1)
        {
        for (unsigned int i = 0; i < this->m_pdata->getNTypes(); ++i)
            result[i] = result[i] - m_implicit_count_run_start[i];
        }
    else
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
        .def(pybind11::init< std::shared_ptr<SystemDefinition>, unsigned int>())
        .def("setDepletantFugacity", &IntegratorHPMCMonoImplicit<Shape>::setDepletantFugacity)
        .def("getImplicitCounters", &IntegratorHPMCMonoImplicit<Shape>::getImplicitCounters)
        .def("getDepletantFugacity", &IntegratorHPMCMonoImplicit<Shape>::getDepletantFugacity)
        .def("setQuermassMode", &IntegratorHPMCMonoImplicit<Shape>::setQuermassMode)
        .def("setSweepRadius", &IntegratorHPMCMonoImplicit<Shape>::setSweepRadius)
        .def("getQuermassMode", &IntegratorHPMCMonoImplicit<Shape>::getQuermassMode)
        .def("getSweepRadius", &IntegratorHPMCMonoImplicit<Shape>::getSweepRadius)
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

#endif // __HPMC_MONO_IMPLICIT__H__
