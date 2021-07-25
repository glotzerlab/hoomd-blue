// Copyright (c) 2009-2021 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.

// Maintainer: jproc

#ifndef __ALCHEMICALPOTENTIALPAIR_H__
#define __ALCHEMICALPOTENTIALPAIR_H__

#include <bitset>
#include <cstddef>
#include <numeric>

#include "hoomd/AlchemyData.h"
#include "hoomd/GlobalArray.h"
#include "hoomd/Index1D.h"
#include "hoomd/extern/nano-signal-slot/nano_observer.hpp"
#include "hoomd/md/PotentialPair.h"

#include "pybind11/functional.h"

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
    bool calculate_derivatives = false;
    std::vector<std::array<Scalar, evaluator::num_alchemical_parameters>> alphas = {};
    std::vector<ArrayHandle<Scalar>> force_handles = {};
    std::vector<std::bitset<evaluator::num_alchemical_parameters>> compute_mask = {};
    std::vector<Scalar> normalizationValues = {};

    AlchemyPackage(std::nullptr_t) {};
    AlchemyPackage() {};
    };

template<class evaluator> struct Normalized : public evaluator
    {
    using evaluator::evaluator;
    Scalar m_normValue {1.};

    void setNormalizationValue(Scalar value)
        {
        m_normValue = value;
        }

    bool evalForceAndEnergy(Scalar& force_divr, Scalar& pair_eng, bool energy_shift)
        {
        bool evaluated = evaluator::evalForceAndEnergy(force_divr, pair_eng, energy_shift);
        force_divr *= m_normValue;
        pair_eng *= m_normValue;
        return evaluated;
        }
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
        std::shared_ptr<AlchemicalPairParticle>& alpha_p
            = m_alchemical_particles[k * m_alchemy_index.getNumElements() + m_alchemy_index(i, j)];
        if (alpha_p == nullptr)
            {
            alpha_p
                = std::make_shared<AlchemicalPairParticle>(this->m_exec_conf, make_int3(i, j, k));
            }
        return alpha_p;
        }

    void enableAlchemicalPairParticle(std::shared_ptr<AlchemicalPairParticle> alpha_p)
        {
        // TODO: make sure only adding a particle to an alchemostat can enable it
        // TODO: is this where the momentum etc should be initilized?
        m_alchemy_mask[m_alchemy_index(alpha_p->m_type_pair_param.x, alpha_p->m_type_pair_param.y)]
                      [alpha_p->m_type_pair_param.z]
            = true;
        alpha_p->resizeForces(this->m_pdata->getN());
        }

    void disableAlchemicalPairParticle(std::shared_ptr<AlchemicalPairParticle> alpha_p)
        {
        // TODO: handle resetting param value and alpha
        m_alchemy_mask[m_alchemy_index(alpha_p->m_type_pair_param.x, alpha_p->m_type_pair_param.y)]
                      [alpha_p->m_type_pair_param.z]
            = false;
        }

    // TODO: keep as general python object, or change to an updater? problem with updater is lack of
    // input, but could make a normalizer based on it
    void setNormalizer(std::function<Scalar(pybind11::kwargs)>& callback)
        {
        m_normalized = true;
        m_normalizer = callback;
        }

    // TODO: disable alchemical particle by resetting params, mask, should remain shared_ptr?

    protected:
    typedef std::bitset<evaluator::num_alchemical_parameters> mask_type;
    typedef std::array<Scalar, evaluator::num_alchemical_parameters> alpha_array_t;

    Index2DUpperTriangular m_alchemy_index; //!< upper triangular typepair index
    std::vector<mask_type> m_alchemy_mask;  //!< Type pair mask for if alchemical forces are used
    std::vector<std::shared_ptr<AlchemicalPairParticle>>
        m_alchemical_particles; //!< 2D array (alchemy_index,alchemical param)
    // static constexpr bool m_normalized =
    // std::is_member_function_pointer<decltype(evaluator::setNormalizationValue)>::value;    //!<
    // The potential should be normalizeds

    // pybind11::function m_normalizer;
    // template<typename ...Args> std::function<Scalar(Args...)> m_normalizer;
    std::function<Scalar(pybind11::kwargs)> m_normalizer;
    bool m_normalized = false;

    //! Method to be called when number of types changes
    void slotNumTypesChange() override;
    void slotNumParticlesChange()
        {
        unsigned int N = this->m_pdata->getN();
        for (auto& particle : m_alchemical_particles)
            if (particle)
                particle->resizeForces(N);
        };

    // Extra steps to insert
    inline AlchemyPackage<evaluator> pkgInitialize(const uint64_t& timestep) override;
    inline void pkgPerNeighbor(const unsigned int& i,
                               const unsigned int& j,
                               const unsigned int& typei,
                               const unsigned int& typej,
                               const bool& in_rcut,
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
    m_alchemical_particles.resize(m_alchemy_index.getNumElements()
                                  * evaluator::num_alchemical_parameters);
    m_alchemy_mask.resize(this->m_pdata->getNTypes());

    this->m_exec_conf->msg->notice(5)
        << "Constructing AlchemicalPotentialPair<" << evaluator::getName() << ">" << std::endl;

    this->m_pdata->getNumTypesChangeSignal()
        .template connect<AlchemicalPotentialPair<evaluator>,
                          &AlchemicalPotentialPair<evaluator>::slotNumTypesChange>(this);

    this->m_pdata->getGlobalParticleNumberChangeSignal()
        .template connect<AlchemicalPotentialPair<evaluator>,
                          &AlchemicalPotentialPair<evaluator>::slotNumParticlesChange>(this);
    }

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
    std::vector<mask_type> new_mask(new_alchemy_index.getNumElements());
    std::vector<std::shared_ptr<AlchemicalPairParticle>> new_particles(
        new_alchemy_index.getNumElements() * evaluator::num_alchemical_parameters);

    // copy over entries that are valid in both the new and old matrices
    unsigned int copy_w = std::min(new_alchemy_index.getW(), m_alchemy_index.getW());
    for (unsigned int i = 0; i < copy_w; i++)
        for (unsigned int j = 0; j < i; j++)
            {
            new_mask[new_alchemy_index(i, j)] = m_alchemy_mask[m_alchemy_index(i, j)];
            for (unsigned int k = 0; k < evaluator::num_alchemical_parameters; k++)
                {
                new_particles[k * new_alchemy_index.getNumElements() + new_alchemy_index(i, j)]
                    = m_alchemical_particles[k * m_alchemy_index.getNumElements()
                                             + m_alchemy_index(i, j)];
                }
            }
    m_alchemy_index = new_alchemy_index;
    m_alchemical_particles.swap(new_particles);
    m_alchemy_mask.swap(new_mask);

    PotentialPair<evaluator, AlchemyPackage<evaluator>>::slotNumTypesChange();
    }

template<class evaluator>
inline AlchemyPackage<evaluator>
AlchemicalPotentialPair<evaluator>::pkgInitialize(const uint64_t& timestep)
    {
    // Create pkg for passing additional variables between specialized code
    AlchemyPackage<evaluator> pkg;

    // make an updated mask that is accurate for this timestep per alchemical particle
    std::vector<mask_type> compute_mask(m_alchemy_mask);
    // Allocate and read alphas into type pair accessible format
    pkg.alphas.assign(m_alchemy_index.getNumElements(), {});
    for (unsigned int i = 0; i < m_alchemy_index.getNumElements(); i++)
        for (unsigned int j = 0; j < evaluator::num_alchemical_parameters; j++)
            {
            unsigned int idx = j * m_alchemy_index.getNumElements() + i;
            if (m_alchemy_mask[i][j])
                {
                pkg.alphas[i][j] = m_alchemical_particles[idx]->value;
                if (m_alchemical_particles[idx]->m_nextTimestep == timestep)
                    {
                    pkg.calculate_derivatives = true;
                    }
                else
                    {
                    compute_mask[i][j] = false;
                    }
                }
            else
                {
                pkg.alphas[i][j] = Scalar(1.0);
                }
            }
    pkg.compute_mask.swap(compute_mask);

    // Precompute normalization
    if (m_normalized)
        {
        pkg.normalizationValues.assign(m_alchemy_index.getNumElements(), 1.0);
        for (unsigned int i = 0; i < m_alchemy_index.getW(); i++)
            for (unsigned int j = 0; j <= i; j++)
                {
                unsigned int idx = m_alchemy_index(i, j);
                pkg.normalizationValues[idx] = m_normalizer(
                    evaluator::alchemParams(this->m_params[this->m_typpair_idx(i, j)],
                                            pkg.alphas[idx])
                        .asDict());
                ;
                }
        }

    if (pkg.calculate_derivatives)
        {
        this->m_exec_conf->msg->notice(10)
            << "AlchemPotentialPair: Calculating alchemical forces" << std::endl;

        // Setup alchemical force array handlers
        pkg.force_handles.clear();
        for (auto& particle : m_alchemical_particles)
            if (particle
                && pkg.compute_mask[m_alchemy_index(particle->m_type_pair_param.x,
                                                    particle->m_type_pair_param.y)]
                                   [particle->m_type_pair_param.z])
                {
                // zero force array and set current timestep for tracking
                particle->setNetForce(timestep);
                pkg.force_handles.push_back(
                    ArrayHandle<Scalar>(particle->m_alchemical_derivatives));
                }
            else
                {
                pkg.force_handles.push_back(ArrayHandle<Scalar>(GlobalArray<Scalar>()));
                }
        }
    return pkg;
    }

template<class evaluator>
inline void AlchemicalPotentialPair<evaluator>::pkgPerNeighbor(const unsigned int& i,
                                                               const unsigned int& j,
                                                               const unsigned int& typei,
                                                               const unsigned int& typej,
                                                               const bool& in_rcut,
                                                               evaluator& eval,
                                                               AlchemyPackage<evaluator>& pkg)
    {
    unsigned int alchemy_index = m_alchemy_index(typei, typej);
    mask_type& mask {pkg.compute_mask[alchemy_index]};
    alpha_array_t& alphas {pkg.alphas[alchemy_index]};
    if (m_normalized)
        eval.setNormalizationValue(pkg.normalizationValues[alchemy_index]);

    // TODO: make sure that when we disable an alchemical particle, we rewrite its parameter

    // calculate alchemical derivatives if needed
    if (pkg.calculate_derivatives && in_rcut && mask.any())
        {
        std::array<Scalar, evaluator::num_alchemical_parameters> alchemical_derivatives = {};
        eval.evalAlchDerivatives(alchemical_derivatives, alphas);
        for (unsigned int k = 0; k < evaluator::num_alchemical_parameters; k++)
            {
            if (mask[k])
                {
                pkg.force_handles[alchemy_index].data[i]
                    += alchemical_derivatives[k] * Scalar(-0.5);
                pkg.force_handles[alchemy_index].data[j]
                    += alchemical_derivatives[k] * Scalar(-0.5);
                }
            alchemy_index += m_alchemy_index.getNumElements();
            }
        }

    // update parameter values with current alphas (MUST! be performed after dAlpha calculations)
    eval.alchemParams(alphas);
    }

template<class evaluator>
inline void AlchemicalPotentialPair<evaluator>::pkgFinalize(AlchemyPackage<evaluator>& pkg)
    {
    // for all used variables, store the net force
    for (unsigned int i = 0; i < m_alchemy_index.getNumElements(); i++)
        for (unsigned int j = 0; j < evaluator::num_alchemical_parameters; j++)
            if (pkg.compute_mask[i][j])
                {
                if (m_normalized)
                    m_alchemical_particles[j * m_alchemy_index.getNumElements() + i]->setNetForce(
                        pkg.normalizationValues[i]);
                else
                    m_alchemical_particles[j * m_alchemy_index.getNumElements() + i]->setNetForce();
                }
    }

//! Export this pair potential to python
/*! \param name Name of the class in the exported python module
    \tparam T Class type to export. \b Must be an instantiated PotentialPair class template.
*/
template<class evaluator_base>
void export_AlchemicalPotentialPair(pybind11::module& m, const std::string& name)
    {
    typedef Normalized<evaluator_base> evaluator;
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
