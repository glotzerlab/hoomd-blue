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

    // TODO: is this good practice or should there be a pkg finalize function call
    ~AlchemyPackage<evaluator>()
        {
        // for all used variables, store the net force
        for (unsigned int i = 0; i < alphas.size(); i++)
            for (unsigned int j = 0; j < evaluator::num_alchemical_parameters; j++)
                if (h_alchemy_mask.data[i][j])
                    h_alchemical_particles.data[alphas.size() * j + i]->setNetForce();
        };
    };

//! Template class for computing alchemical pair potentials
/*! <b>Overview:</b>

    <b>Implementation details</b>



    \sa export_PotentialPair()
*/
template<class evaluator, typename extra_pkg = AlchemyPackage<evaluator>>
class AlchemicalPotentialPair : public PotentialPair<evaluator, extra_pkg>
    {
    public:
    //! Construct the pair potential
    AlchemicalPotentialPair(std::shared_ptr<SystemDefinition> sysdef,
                            std::shared_ptr<NeighborList> nlist);
    //! Destructor
    virtual ~AlchemicalPotentialPair();

    std::shared_ptr<AlchemicalPairParticle> getAlchemicalPairParticle(int i, int j, int k)
        {
        ArrayHandle<std::shared_ptr<AlchemicalPairParticle>> h_alpha_p(m_alchemical_particles);
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

    //! Method to be called when number of types changes
    void slotNumTypesChange() override;
    void slotNumParticlesChange()
        {
        m_needs_alch_force_resize = true;
        };

    // Extra steps to insert
    inline extra_pkg pkgInitialze(const uint64_t& timestep) override;
    inline void pkgPerNeighbor(const unsigned int& i,
                               const unsigned int& j,
                               const bool in_rcut,
                               evaluator& eval,
                               extra_pkg&) override;
    };

template<class evaluator, typename extra_pkg>
AlchemicalPotentialPair<evaluator, extra_pkg>::AlchemicalPotentialPair(
    std::shared_ptr<SystemDefinition> sysdef,
    std::shared_ptr<NeighborList> nlist)
    : PotentialPair<evaluator, extra_pkg>(sysdef, nlist)
    {
    // TODO: proper logging variables
    this->m_exec_conf->msg->notice(5)
        << "Constructing AlchemicalPotentialPair<" << evaluator::getName() << ">" << std::endl;

    this->m_pdata->getNumTypesChangeSignal()
        .template connect<AlchemicalPotentialPair<evaluator, extra_pkg>,
                          &AlchemicalPotentialPair<evaluator, extra_pkg>::slotNumTypesChange>(this);

    this->m_pdata->getGlobalParticleNumberChangeSignal()
        .template connect<AlchemicalPotentialPair<evaluator, extra_pkg>,
                          &AlchemicalPotentialPair<evaluator, extra_pkg>::slotNumParticlesChange>(
            this);
    }

// TODO: constructor from base class and similar demote for easy switching

template<class evaluator, typename extra_pkg>
AlchemicalPotentialPair<evaluator, extra_pkg>::~AlchemicalPotentialPair()
    {
    this->m_exec_conf->msg->notice(5)
        << "Destroying AlchemicalPotentialPair<" << evaluator::getName() << ">" << std::endl;

    this->m_pdata->getNumTypesChangeSignal()
        .template disconnect<AlchemicalPotentialPair<evaluator, extra_pkg>,
                             &AlchemicalPotentialPair<evaluator, extra_pkg>::slotNumTypesChange>(
            this);

    this->m_pdata->getGlobalParticleNumberChangeSignal()
        .template disconnect<
            AlchemicalPotentialPair<evaluator, extra_pkg>,
            &AlchemicalPotentialPair<evaluator, extra_pkg>::slotNumParticlesChange>(this);
    }

template<class evaluator, typename extra_pkg>
void AlchemicalPotentialPair<evaluator, extra_pkg>::slotNumTypesChange()
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

template<class evaluator, typename extra_pkg>
inline extra_pkg
AlchemicalPotentialPair<evaluator, extra_pkg>::pkgInitialze(const uint64_t& timestep)
    {
    // Create pkg for passing additional variables between specialized code
    extra_pkg pkg = {false,
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

template<class evaluator, typename extra_pkg>
inline void AlchemicalPotentialPair<evaluator, extra_pkg>::pkgPerNeighbor(const unsigned int& i,
                                                                          const unsigned int& j,
                                                                          const bool in_rcut,
                                                                          evaluator& eval,
                                                                          extra_pkg& pkg)
    {
    unsigned int alchemy_index = m_alchemy_index(i, j);
    mask_type& mask = pkg.h_alchemy_mask.data[alchemy_index];
    alpha_array_t& alphas = pkg.alphas[alchemy_index];

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

//! Export this pair potential to python
/*! \param name Name of the class in the exported python module
    \tparam T Class type to export. \b Must be an instantiated PotentialPair class template.
*/
template<class evaluator>
void export_AlchemicalPotentialPair(pybind11::module& m, const std::string& name)
    {
    typedef PotentialPair<evaluator, AlchemyPackage<evaluator>> base;
    export_PotentialPair<base>(m,name+std::string("Base").c_str());
    typedef AlchemicalPotentialPair<evaluator> T;
    pybind11::class_<T, base, std::shared_ptr<T>>
        alchemicalpotentialpair(m, name.c_str());
    alchemicalpotentialpair
        .def(pybind11::init<std::shared_ptr<SystemDefinition>, std::shared_ptr<NeighborList>>())
        .def("getAlchemicalPairParticle", &T::getAlchemicalPairParticle)
        .def("enableAlchemicalPairParticle", &T::enableAlchemicalPairParticle);
    }

#endif // __ALCHEMICALPOTENTIALPAIR_H__
