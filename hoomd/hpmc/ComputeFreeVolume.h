// Copyright (c) 2009-2024 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

#ifndef __COMPUTE_FREE_VOLUME__H__
#define __COMPUTE_FREE_VOLUME__H__

#include "hoomd/Autotuner.h"
#include "hoomd/CellList.h"
#include "hoomd/Compute.h"

#include "IntegratorHPMCMono.h"
#include "hoomd/RNGIdentifiers.h"

/*! \file ComputeFreeVolume.h
    \brief Defines the template class for an approximate free volume integration
    \note This header cannot be compiled by nvcc
*/

#ifdef __HIPCC__
#error This header cannot be compiled by nvcc
#endif

#include <pybind11/pybind11.h>

namespace hoomd
    {
namespace hpmc
    {
//! Template class for a free volume integration analyzer
/*!
    \ingroup hpmc_integrators
*/
template<class Shape> class ComputeFreeVolume : public Compute
    {
    public:
    //! Construct the integrator
    ComputeFreeVolume(std::shared_ptr<SystemDefinition> sysdef,
                      std::shared_ptr<IntegratorHPMCMono<Shape>> mc,
                      std::shared_ptr<CellList> cl);
    //! Destructor
    virtual ~ComputeFreeVolume() {};

    //! Get the number of MC samples to perform
    unsigned int getNumSamples()
        {
        return m_n_sample;
        }

    //! Set the number of MC samples to perform
    //! \param n_sample number of MC steps to sample
    void setNumSamples(unsigned int n_sample)
        {
        m_n_sample = n_sample;
        }

    //! Get the type of depletant particle
    std::string getTestParticleType()
        {
        std::string type = m_sysdef->getParticleData()->getNameByType(m_type);
        return type;
        }

    //! Set the type of depletant particle
    //! \param type particle type to set test particle to
    void setTestParticleType(std::string type)
        {
        unsigned int type_int = m_sysdef->getParticleData()->getTypeByName(type);
        m_type = type_int;
        }

    //! Analyze the current configuration
    virtual void compute(uint64_t timestep);

    //! Return an estimate of the overlap volume
    virtual Scalar getFreeVolume();

    protected:
    std::shared_ptr<IntegratorHPMCMono<Shape>> m_mc; //!< The parent integrator
    std::shared_ptr<CellList> m_cl;                  //!< The cell list

    unsigned int m_type;     //!< Type of depletant particle to generate
    unsigned int m_n_sample; //!< Number of sampling depletants to generate

    GPUArray<unsigned int> m_n_overlap_all; //!< Number of overlap volume particles in box

    //! Return an estimate of the overlap volume
    virtual void computeFreeVolume(uint64_t timestep);
    };

template<class Shape>
ComputeFreeVolume<Shape>::ComputeFreeVolume(std::shared_ptr<SystemDefinition> sysdef,
                                            std::shared_ptr<IntegratorHPMCMono<Shape>> mc,
                                            std::shared_ptr<CellList> cl)
    : Compute(sysdef), m_mc(mc), m_cl(cl), m_type(0), m_n_sample(0)
    {
    this->m_exec_conf->msg->notice(5) << "Constructing ComputeFreeVolume" << std::endl;

    this->m_cl->setRadius(1);
    this->m_cl->setComputeTypeBody(false);
    this->m_cl->setFlagType();
    this->m_cl->setComputeIdx(true);

    // allocate mem for overlap counts
    GPUArray<unsigned int> n_overlap_all(1, this->m_exec_conf);
    m_n_overlap_all.swap(n_overlap_all);
    }

template<class Shape> void ComputeFreeVolume<Shape>::compute(uint64_t timestep)
    {
    Compute::compute(timestep);
    if (!shouldCompute(timestep))
        return;

    // update ghost layers
    m_mc->communicate(false);

    this->computeFreeVolume(timestep);
    }

/*! \return the current free volume estimate by MC integration
 */
template<class Shape> void ComputeFreeVolume<Shape>::computeFreeVolume(uint64_t timestep)
    {
    unsigned int overlap_count = 0;
    unsigned int err_count = 0;
    unsigned int ndim = this->m_sysdef->getNDimensions();

    this->m_exec_conf->msg->notice(5) << "HPMC computing free volume " << timestep << std::endl;

    // update AABB tree
    const hoomd::detail::AABBTree& aabb_tree = this->m_mc->buildAABBTree();

    // update the image list
    std::vector<vec3<Scalar>> image_list = this->m_mc->updateImageList();

    uint16_t seed = m_sysdef->getSeed();

    // only check if AABB tree is populated
    if (m_pdata->getN() + m_pdata->getNGhosts())
        {
        // access particle data and system box
        ArrayHandle<Scalar4> h_postype(m_pdata->getPositions(),
                                       access_location::host,
                                       access_mode::read);
        ArrayHandle<Scalar4> h_orientation(m_pdata->getOrientationArray(),
                                           access_location::host,
                                           access_mode::read);
        const BoxDim box = m_pdata->getBox();

        // access parameters and interaction matrix
        const std::vector<typename Shape::param_type,
                          hoomd::detail::managed_allocator<typename Shape::param_type>>& params
            = m_mc->getParams();

        ArrayHandle<unsigned int> h_overlaps(m_mc->getInteractionMatrix(),
                                             access_location::host,
                                             access_mode::read);
        const Index2D& overlap_idx = m_mc->getOverlapIndexer();

        // generate n_sample random test depletants in the global box
        unsigned int n_sample = m_n_sample;

#ifdef ENABLE_MPI
        n_sample /= this->m_exec_conf->getNRanks();
#endif

        for (unsigned int i = 0; i < n_sample; i++)
            {
            // select a random particle coordinate in the box
            hoomd::RandomGenerator rng_i(
                hoomd::Seed(hoomd::RNGIdentifier::ComputeFreeVolume, timestep, seed),
                hoomd::Counter(m_exec_conf->getRank(), i));

            Scalar xrand = hoomd::detail::generate_canonical<Scalar>(rng_i);
            Scalar yrand = hoomd::detail::generate_canonical<Scalar>(rng_i);
            Scalar zrand = hoomd::detail::generate_canonical<Scalar>(rng_i);
            if (this->m_sysdef->getNDimensions() == 2)
                {
                zrand = 0;
                }
            Scalar3 f = make_scalar3(xrand, yrand, zrand);
            vec3<Scalar> pos_i = vec3<Scalar>(box.makeCoordinates(f));

            Shape shape_i(quat<Scalar>(), params[m_type]);
            if (shape_i.hasOrientation())
                {
                shape_i.orientation = generateRandomOrientation(rng_i, ndim);
                }

            // check for overlaps with particles in the system state
            bool overlap = false;
            hoomd::detail::AABB aabb_i_local = shape_i.getAABB(vec3<Scalar>(0, 0, 0));

            // All image boxes (including the primary)
            const unsigned int n_images = (unsigned int)image_list.size();
            for (unsigned int cur_image = 0; cur_image < n_images; cur_image++)
                {
                vec3<Scalar> pos_i_image = pos_i + image_list[cur_image];
                hoomd::detail::AABB aabb = aabb_i_local;
                aabb.translate(pos_i_image);

                // stackless search
                for (unsigned int cur_node_idx = 0; cur_node_idx < aabb_tree.getNumNodes();
                     cur_node_idx++)
                    {
                    if (aabb.overlaps(aabb_tree.getNodeAABB(cur_node_idx)))
                        {
                        if (aabb_tree.isNodeLeaf(cur_node_idx))
                            {
                            for (unsigned int cur_p = 0;
                                 cur_p < aabb_tree.getNodeNumParticles(cur_node_idx);
                                 cur_p++)
                                {
                                // read in its position and orientation
                                unsigned int j = aabb_tree.getNodeParticle(cur_node_idx, cur_p);

                                Scalar4 postype_j;
                                Scalar4 orientation_j;

                                // load the position and orientation of the j particle
                                postype_j = h_postype.data[j];
                                orientation_j = h_orientation.data[j];

                                // put particles in coordinate system of particle i
                                vec3<Scalar> r_ij = vec3<Scalar>(postype_j) - pos_i_image;

                                unsigned int typ_j = __scalar_as_int(postype_j.w);
                                Shape shape_j(quat<Scalar>(orientation_j), params[typ_j]);

                                if (h_overlaps.data[overlap_idx(m_type, typ_j)]
                                    && check_circumsphere_overlap(r_ij, shape_i, shape_j)
                                    && test_overlap(r_ij, shape_i, shape_j, err_count))
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
                        cur_node_idx += aabb_tree.getNodeSkip(cur_node_idx);
                        }

                    if (overlap)
                        break;
                    } // end loop over AABB nodes

                if (overlap)
                    break;
                } // end loop over images

            if (overlap)
                {
                overlap_count++;
                }
            } // end loop through all particles

        } // end lexical scope

#ifdef ENABLE_MPI
    if (m_sysdef->isDomainDecomposed())
        {
        MPI_Allreduce(MPI_IN_PLACE,
                      &overlap_count,
                      1,
                      MPI_UNSIGNED,
                      MPI_SUM,
                      m_exec_conf->getMPICommunicator());
        }
#endif

    ArrayHandle<unsigned int> h_n_overlap_all(m_n_overlap_all,
                                              access_location::host,
                                              access_mode::overwrite);
    *h_n_overlap_all.data = overlap_count;
    }

// \return the free volume.
template<class Shape> Scalar ComputeFreeVolume<Shape>::getFreeVolume()
    {
    // access counters
    ArrayHandle<unsigned int> h_n_overlap_all(m_n_overlap_all,
                                              access_location::host,
                                              access_mode::read);

    // generate n_sample random test depletants in the global box
    unsigned int n_sample = m_n_sample;

#ifdef ENABLE_MPI
    // in MPI, for small n_sample we can encounter round-off issues
    unsigned int n_ranks = this->m_exec_conf->getNRanks();
    n_sample = (n_sample / n_ranks) * n_ranks;
#endif

    // total free volume
    const BoxDim global_box = this->m_pdata->getGlobalBox();
    Scalar V_free = (Scalar)(n_sample - *h_n_overlap_all.data) / (Scalar)n_sample
                    * global_box.getVolume(this->m_sysdef->getNDimensions() == 2);

    return V_free;
    }

namespace detail
    {
//! Export this hpmc analyzer to python
/*! \param name Name of the class in the exported python module
    \tparam Shape An instantiation of ComputeFreeVolume<Shape> will be exported
*/
template<class Shape> void export_ComputeFreeVolume(pybind11::module& m, const std::string& name)
    {
    pybind11::class_<ComputeFreeVolume<Shape>, Compute, std::shared_ptr<ComputeFreeVolume<Shape>>>(
        m,
        name.c_str())
        .def(pybind11::init<std::shared_ptr<SystemDefinition>,
                            std::shared_ptr<IntegratorHPMCMono<Shape>>,
                            std::shared_ptr<CellList>>())
        .def_property("num_samples",
                      &ComputeFreeVolume<Shape>::getNumSamples,
                      &ComputeFreeVolume<Shape>::setNumSamples)
        .def_property("test_particle_type",
                      &ComputeFreeVolume<Shape>::getTestParticleType,
                      &ComputeFreeVolume<Shape>::setTestParticleType)
        .def_property_readonly("free_volume", &ComputeFreeVolume<Shape>::getFreeVolume);
    }

    } // end namespace detail
    } // end namespace hpmc
    } // end namespace hoomd

#endif // __COMPUTE_FREE_VOLUME__H__
