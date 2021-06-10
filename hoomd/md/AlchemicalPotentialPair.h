// Copyright (c) 2009-2021 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.

// Maintainer: jproc

#ifndef __ALCHEMICALPOTENTIALPAIR_H__
#define __ALCHEMICALPOTENTIALPAIR_H__

#include <bitset>
#include <cstddef>

#include "hoomd/AlchemyData.h"
#include "hoomd/Index1D.h"
#include "hoomd/extern/nano-signal-slot/nano_observer.hpp"
#include "hoomd/md/PotentialPair.h"

#ifdef ENABLE_HIP
#include <hip/hip_runtime.h>
#endif

#ifdef ENABLE_MPI
#include "hoomd/Communicator.h"
#endif

/*! \file AlchemicalPotentialPair.h
    \brief Defines the template class for alchemical pair potentials
*/

#ifdef __HIPCC__
#error This header cannot be compiled by nvcc
#endif

template<class evaluator> struct AlchemyPackage
    {
    bool calculate_derivatives;
    ArrayHandle<std::shared_ptr<AlchemicalPairParticle>> h_alchemical_particles;
    ArrayHandle<std::bitset<evaluator::num_alchemical_parameters>> h_alchemy_mask;

    std::vector<std::array<Scalar, evaluator::num_alchemical_parameters>> alphas = {};

    AlchemyPackage(bool a,
                   ArrayHandle<std::shared_ptr<AlchemicalPairParticle>> b,
                   ArrayHandle<std::bitset<evaluator::num_alchemical_parameters>> c)
        : calculate_derivatives(a), h_alchemical_particles(b), h_alchemy_mask(c) {};

    AlchemyPackage(std::nullptr_t)
        : h_alchemical_particles(GPUArray<std::shared_ptr<AlchemicalPairParticle>>()),
          h_alchemy_mask(GPUArray<std::bitset<evaluator::num_alchemical_parameters>>()) {};
    };

//! Template class for computing alchemical pair potentials
/*! <b>Overview:</b>

    <b>Implementation details</b>



    \sa export_PotentialPair()
*/
template<class evaluator>
class AlchemicalPotentialPair : public PotentialPair<evaluator, AlchemyPackage<evaluator>>
    {
    public:
    //! Construct the pair potential
    AlchemicalPotentialPair(std::shared_ptr<SystemDefinition> sysdef,
                            std::shared_ptr<NeighborList> nlist);
    //! Destructor
    virtual ~AlchemicalPotentialPair();

    std::shared_ptr<AlchemicalPairParticle> getAlchemicalPairParticle(int i, int j, int k)
        {
        ArrayHandle<std::shared_ptr<AlchemicalPairParticle>> h_alpha_p(m_alchemical_particles,
                                                                       access_location::host,
                                                                       access_mode::readwrite);
        std::shared_ptr<AlchemicalPairParticle>& alpha_p
            = h_alpha_p.data[k * m_alchemy_index.getNumElements() + m_alchemy_index(i, j)];
        if (alpha_p == nullptr)
            {
            alpha_p
                = std::make_shared<AlchemicalPairParticle>(this->m_exec_conf, make_int3(i, j, k));
            m_needs_alch_force_resize = true;
            }
        return alpha_p;
        }

    void enableAlchemicalPairParticle(std::shared_ptr<AlchemicalPairParticle> alpha_p)
        {
        // ArrayHandle<std::shared_ptr<AlchemicalPairParticle>> h_alpha_p(m_alchemical_particles);
        // h_alpha_p.data[]
        ArrayHandle<mask_type> h_mask(m_alchemy_mask);
        mask_type& mask = h_mask.data[m_alchemy_index(alpha_p->m_type_pair_param.x,
                                                      alpha_p->m_type_pair_param.y)];
        // TODO: make sure only adding a particle to an alchemostat can enable it
        assert(mask[alpha_p->m_type_pair_param.z] == false);
        // TODO: is this where the momentum etc should be initilized?
        mask[alpha_p->m_type_pair_param.z] = true;
        m_needs_alch_force_resize = true;
        }

    // TODO: keep as general python object, or change to an updater? problem with updater is lack of
    // input, but could make a normalizer based on it
    void setNormalizer(pybind11::object callback)
        {
        m_normalizer = callback;
        m_normalized = true;
        }

    // TODO: disable alchemical particle by resetting params, mask, should remain shared_ptr?

    protected:
    typedef std::bitset<evaluator::num_alchemical_parameters> mask_type;
    typedef std::array<Scalar, evaluator::num_alchemical_parameters> alpha_array_t;

    Index2DUpperTriangular m_alchemy_index; //!< upper triangular typepair index
    GlobalArray<mask_type> m_alchemy_mask;  //!< Type pair mask for if alchemical forces are used
    GlobalArray<std::shared_ptr<AlchemicalPairParticle>>
        m_alchemical_particles;                 //!< 2D array (alchemy_index,alchemical param)
    std::set<uint64_t> m_alchemical_time_steps; //!< Next alchemical time step
    bool m_needs_alch_force_resize = true;
    bool m_normalized = false;     //!< The potential should be normalizeds
    pybind11::object m_normalizer; //!< python normalization TODO: allow for cpp normalization as
                                   //!< well, probably via pybind11::vectorize

    //! Method to be called when number of types changes
    void slotNumTypesChange() override;
    void slotNumParticlesChange()
        {
        m_needs_alch_force_resize = true;
        };

    // Extra steps to insert
    inline AlchemyPackage<evaluator> pkgInitialize(const uint64_t& timestep) override;
    inline void pkgPerNeighbor(const unsigned int& i,
                               const unsigned int& j,
                               const unsigned int& typei,
                               const unsigned int& typej,
                               const bool in_rcut,
                               evaluator& eval,
                               AlchemyPackage<evaluator>&) override;
    inline void pkgFinalize(AlchemyPackage<evaluator>&) override;
    };

template<class evaluator>
AlchemicalPotentialPair<evaluator>::AlchemicalPotentialPair(
    std::shared_ptr<SystemDefinition> sysdef,
    std::shared_ptr<NeighborList> nlist)
    : PotentialPair<evaluator, AlchemyPackage<evaluator>>(sysdef, nlist)
    {
    m_alchemy_index = Index2DUpperTriangular(this->m_pdata->getNTypes());
    m_alchemical_particles
        = GlobalArray<std::shared_ptr<AlchemicalPairParticle>>(m_alchemy_index.getNumElements(),
                                                               evaluator::num_alchemical_parameters,
                                                               this->m_exec_conf);
    m_alchemy_mask = GlobalArray<mask_type>(this->m_pdata->getNTypes(), this->m_exec_conf);

    // TODO: proper logging variables
    this->m_exec_conf->msg->notice(5)
        << "Constructing AlchemicalPotentialPair<" << evaluator::getName() << ">" << std::endl;

    this->m_pdata->getNumTypesChangeSignal()
        .template connect<AlchemicalPotentialPair<evaluator>,
                          &AlchemicalPotentialPair<evaluator>::slotNumTypesChange>(this);

    this->m_pdata->getGlobalParticleNumberChangeSignal()
        .template connect<AlchemicalPotentialPair<evaluator>,
                          &AlchemicalPotentialPair<evaluator>::slotNumParticlesChange>(this);
    }

// TODO: constructor from base class and similar demote for easy switching

template<class evaluator> AlchemicalPotentialPair<evaluator>::~AlchemicalPotentialPair()
    {
    this->m_exec_conf->msg->notice(5)
        << "Destroying AlchemicalPotentialPair<" << evaluator::getName() << ">" << std::endl;

    this->m_pdata->getNumTypesChangeSignal()
        .template disconnect<AlchemicalPotentialPair<evaluator>,
                             &AlchemicalPotentialPair<evaluator>::slotNumTypesChange>(this);

    this->m_pdata->getGlobalParticleNumberChangeSignal()
        .template disconnect<AlchemicalPotentialPair<evaluator>,
                             &AlchemicalPotentialPair<evaluator>::slotNumParticlesChange>(this);
    }

template<class evaluator> void AlchemicalPotentialPair<evaluator>::slotNumTypesChange()
    {
    Index2DUpperTriangular new_alchemy_index = Index2DUpperTriangular(this->m_pdata->getNTypes());
    GlobalArray<mask_type> new_mask(new_alchemy_index.getNumElements(), this->m_exec_conf);
    GlobalArray<std::shared_ptr<AlchemicalPairParticle>> new_particles_array(
        new_alchemy_index.getNumElements(),
        evaluator::num_alchemical_parameters,
        this->m_exec_conf);

    ArrayHandle<mask_type> h_new_mask(new_mask, access_location::host, access_mode::overwrite);
    ArrayHandle<std::shared_ptr<AlchemicalPairParticle>> h_new_particles(new_particles_array,
                                                                         access_location::host,
                                                                         access_mode::overwrite);
    for (unsigned int i = 0; i < new_alchemy_index.getNumElements(); i++)
        {
        h_new_mask.data[i].reset(); // set bitsets to all false by default, enable manually
        for (unsigned int j = 0; j < evaluator::num_alchemical_parameters; j++)
            {
            // TODO: determine if this does anything, is default shared_ptr guaranteed to be nullptr
            h_new_particles.data[j * new_alchemy_index.getNumElements() + i] = nullptr;
            }
        }

    ArrayHandle<std::shared_ptr<AlchemicalPairParticle>> h_particles(m_alchemical_particles,
                                                                     access_location::host,
                                                                     access_mode::read);
    ArrayHandle<mask_type> h_mask(m_alchemy_mask, access_location::host, access_mode::read);

    // copy over entries that are valid in both the new and old matrices
    unsigned int copy_w = std::min(new_alchemy_index.getW(), m_alchemy_index.getW());
    for (unsigned int i = 0; i < copy_w; i++)
        for (unsigned int j = 0; j < i; j++)
            {
            h_new_mask.data[new_alchemy_index(i, j)] = h_mask.data[m_alchemy_index(i, j)];
            for (unsigned int k = 0; k < evaluator::num_alchemical_parameters; k++)
                {
                h_new_particles
                    .data[k * new_alchemy_index.getNumElements() + new_alchemy_index(i, j)]
                    = h_particles
                          .data[k * m_alchemy_index.getNumElements() + m_alchemy_index(i, j)];
                }
            }
    m_alchemy_index = new_alchemy_index;
    m_alchemical_particles.swap(new_particles_array);
    m_alchemy_mask.swap(new_mask);
    }

template<class evaluator>
inline AlchemyPackage<evaluator>
AlchemicalPotentialPair<evaluator>::pkgInitialize(const uint64_t& timestep)
    {
    // Create pkg for passing additional variables between specialized code
    AlchemyPackage<evaluator> pkg
        = {false,
           ArrayHandle<std::shared_ptr<AlchemicalPairParticle>>(m_alchemical_particles,
                                                                access_location::host,
                                                                access_mode::read),
           ArrayHandle<mask_type>(m_alchemy_mask,
                                  access_location::host,
                                  access_mode::read)}; // zero force

    // Allocate and read alphas into type pair accessible format
    pkg.alphas.assign(m_alchemy_index.getNumElements(), {});
    for (unsigned int i = 0; i < m_alchemy_index.getNumElements(); i++)
        for (unsigned int j = 0; j < evaluator::num_alchemical_parameters; j++)
            if (pkg.h_alchemy_mask.data[i][j])
                {
                unsigned int idx = j * m_alchemy_index.getNumElements() + i;
                pkg.alphas[i][j] = pkg.h_alchemical_particles.data[idx]->value;
                // TODO: if we copy the mask array and modify the copy we can skip some unneeded
                // calculations
                m_alchemical_time_steps.insert(
                    pkg.h_alchemical_particles.data[idx]->m_nextTimestep);
                // m_alchemical_time_steps.insert(
                // pkg.h_alchemical_particles.data[idx]->m_nextTimestep-1);
                }
            else
                {
                pkg.alphas[i][j] = Scalar(1.0);
                }

    // Check if alchemical forces need to be computed
    if (m_alchemical_time_steps.count(timestep))
        {
        pkg.calculate_derivatives = true;
        this->m_exec_conf->msg->notice(10)
            << "AlchemPotentialPair: Calculating alchemical forces" << std::endl;
        // Only resize when preforming a new calculation so previous results remain accessible
        if (m_needs_alch_force_resize)
            {
            unsigned int N = this->m_pdata->getN();
            for (unsigned int i = 0; i < m_alchemical_particles.getNumElements(); i++)
                if (pkg.h_alchemical_particles.data[i] != nullptr)
                    pkg.h_alchemical_particles.data[i]->resizeForces(N);
            m_needs_alch_force_resize = false;
            }
        // zero the forces of all particles and set timing checks
        for (unsigned int i = 0; i < m_alchemy_index.getNumElements(); i++)
            for (unsigned int j = 0; j < evaluator::num_alchemical_parameters; j++)
                if (pkg.h_alchemy_mask.data[i][j])
                    pkg.h_alchemical_particles.data[m_alchemy_index.getNumElements() * j + i]
                        ->setNetForce(timestep);
        }
    return pkg;
    }

template<class evaluator>
inline void AlchemicalPotentialPair<evaluator>::pkgPerNeighbor(const unsigned int& i,
                                                               const unsigned int& j,
                                                               const unsigned int& typei,
                                                               const unsigned int& typej,
                                                               const bool in_rcut,
                                                               evaluator& eval,
                                                               AlchemyPackage<evaluator>& pkg)
    {
    unsigned int alchemy_index = m_alchemy_index(typei, typej);
    mask_type& mask {pkg.h_alchemy_mask.data[alchemy_index]};
    alpha_array_t& alphas {pkg.alphas[alchemy_index]};

    // TODO: make sure that when we disable an alchemical particle, we rewrite it's parameter
    // TODO: add support of aniso

    // update parameter values with current alphas
    eval.alchemParams(alphas.data());
    if (pkg.calculate_derivatives && in_rcut)
        {
        Scalar alchemical_derivatives[evaluator::num_alchemical_parameters] = {0.0};
        eval.evalAlchDerivatives(alchemical_derivatives, alphas.data());
        for (unsigned int k = 0; k < evaluator::num_alchemical_parameters; k++)
            {
            if (mask[k])
                {
                // TODO: this is likely to be inefficient, design better array handle access
                ArrayHandle<Scalar> h_alchemical_forces(
                    pkg.h_alchemical_particles.data[alchemy_index]->m_alchemical_derivatives);
                h_alchemical_forces.data[i] += alchemical_derivatives[k] * Scalar(0.5);
                }
            alchemy_index += m_alchemy_index.getNumElements();
            }
        }
    }

template<class evaluator>
inline void AlchemicalPotentialPair<evaluator>::pkgFinalize(AlchemyPackage<evaluator>& pkg)
    {
    Scalar norm_value(1.0);
    if (m_normalized)
        {
        std::vector<Scalar> current_params;
            {
            ArrayHandle<typename evaluator::param_type> h_params(this->m_params,
                                                                 access_location::host,
                                                                 access_mode::read);

            current_params.reserve(m_alchemy_index.getNumElements()
                                   * evaluator::num_alchemical_parameters);
            for (unsigned int i = 0; i < this->m_pdata->getNTypes(); i++)
                for (unsigned int j = 0; j < i; j++)
                    for (unsigned int k = 0; k < evaluator::num_alchemical_parameters; k++)
                        current_params.push_back(pkg.alphas[m_alchemy_index(i, j)][k]
                                                 * h_params.data[this->m_typpair_idx(i, j)][k]);
            }

        // auto norm_input = pybind11::memoryview::from_buffer(
        //     current_params.data(),
        //     sizeof(Scalar),
        //     NULL,
        //     {evaluator::num_alchemical_parameters, this->m_pdata->getNTypes()},
        //     {sizeof(Scalar) * evaluator::num_alchemical_parameters, sizeof(Scalar)},
        //     true);
        auto norm_input = pybind11::array({evaluator::num_alchemical_parameters, this->m_pdata->getNTypes()},
                        {sizeof(Scalar) * evaluator::num_alchemical_parameters, sizeof(Scalar)},
                        current_params.data());
        Scalar norm_value = m_normalizer(norm_input).template cast<Scalar>();
        // TODO: major changes to support multiple types, needs to modify force with per type pair
        // norm
        // norm_value = m_normalizer(current_params.data()).cast<Scalar>();

        bool compute_virial = this->m_pdata->getFlags()[pdata_flag::pressure_tensor];

        ArrayHandle<Scalar4> h_force(this->m_force, access_location::host, access_mode::readwrite);
        ArrayHandle<Scalar> h_virial(this->m_virial, access_location::host, access_mode::readwrite);

        for (unsigned int i = 0; i < this->m_pdata->getN(); i++)
            {
            h_force.data[i].x *= norm_value;
            h_force.data[i].y *= norm_value;
            h_force.data[i].z *= norm_value;
            h_force.data[i].w *= norm_value;
            if (compute_virial)
                for (unsigned int j = 0; j < 6; j++)
                    h_virial.data[j * this->m_virial_pitch + i] *= norm_value;
            }
        }

    // for all used variables, store the net force
    for (unsigned int i = 0; i < pkg.alphas.size(); i++)
        for (unsigned int j = 0; j < evaluator::num_alchemical_parameters; j++)
            if (pkg.h_alchemy_mask.data[i][j])
                pkg.h_alchemical_particles.data[pkg.alphas.size() * j + i]->setNetForce(norm_value);
    }

//! Export this pair potential to python
/*! \param name Name of the class in the exported python module
    \tparam T Class type to export. \b Must be an instantiated PotentialPair class template.
*/
template<class evaluator>
void export_AlchemicalPotentialPair(pybind11::module& m, const std::string& name)
    {
    typedef PotentialPair<evaluator, AlchemyPackage<evaluator>> base;
    export_PotentialPair<base>(m, name + std::string("Base").c_str());
    typedef AlchemicalPotentialPair<evaluator> T;
    pybind11::class_<T, base, std::shared_ptr<T>> alchemicalpotentialpair(m, name.c_str());
    alchemicalpotentialpair
        .def(pybind11::init<std::shared_ptr<SystemDefinition>, std::shared_ptr<NeighborList>>())
        .def("getAlchemicalPairParticle", &T::getAlchemicalPairParticle)
        .def("enableAlchemicalPairParticle", &T::enableAlchemicalPairParticle)
        .def("setNormalizer", &T::setNormalizer)
        // .def_property_readonly("")
        ;
    }

#endif // __ALCHEMICALPOTENTIALPAIR_H__
