// Copyright (c) 2009-2022 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

// Maintainer: jproc

#ifndef POTENTIAL_PAIR_ALCHEMICAL_H
#define POTENTIAL_PAIR_ALCHEMICAL_H

#include <bitset>
#include <cstddef>
#include <memory>
#include <numeric>

#include "AlchemyData.h"
#include "hoomd/GlobalArray.h"
#include "hoomd/Index1D.h"
#include "hoomd/extern/nano-signal-slot/nano_observer.hpp"
#include "hoomd/md/PotentialPair.h"

#ifdef ENABLE_HIP
#include <hip/hip_runtime.h>
#endif

#ifdef ENABLE_MPI
#include "hoomd/Communicator.h"
#endif

/*! \file PotentialPairAlchemical.h
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

    AlchemyPackage(std::nullptr_t) {};
    AlchemyPackage() {};
    };

//! Template class for computing alchemical pair potentials
/*! <b>Overview:</b>

    <b>Implementation details</b>



    \sa export_PotentialPair()
*/
template<class evaluator,
         typename extra_pkg = AlchemyPackage<evaluator>,
         typename alpha_particle_type = AlchemicalPairParticle>
class PotentialPairAlchemical : public PotentialPair<evaluator, extra_pkg>
    {
    public:
    //! Construct the pair potential
    PotentialPairAlchemical(std::shared_ptr<SystemDefinition> sysdef,
                            std::shared_ptr<NeighborList> nlist);
    //! Destructor
    virtual ~PotentialPairAlchemical();

    std::shared_ptr<alpha_particle_type> getAlchemicalPairParticle(int i, int j, int k)
        {
        std::shared_ptr<alpha_particle_type>& alpha_p
            = m_alchemical_particles[k * m_alchemy_index.getNumElements() + m_alchemy_index(i, j)];
        if (alpha_p == nullptr)
            {
            alpha_p = std::make_shared<alpha_particle_type>(m_exec_conf, make_int3(i, j, k));
            }
        return alpha_p;
        }

    void enableAlchemicalPairParticle(std::shared_ptr<alpha_particle_type> alpha_p)
        {
        // TODO: make sure only adding a particle to an alchemostat can enable it
        // TODO: is this where the momentum etc should be initilized?
        m_alchemy_mask[m_alchemy_index(alpha_p->m_type_pair_param.x, alpha_p->m_type_pair_param.y)]
                      [alpha_p->m_type_pair_param.z]
            = true;
        alpha_p->resizeForces(m_pdata->getN());
        }

    void disableAlchemicalPairParticle(std::shared_ptr<alpha_particle_type> alpha_p)
        {
        // TODO: handle resetting param value and alpha
        m_alchemy_mask[m_alchemy_index(alpha_p->m_type_pair_param.x, alpha_p->m_type_pair_param.y)]
                      [alpha_p->m_type_pair_param.z]
            = false;
        }

    // TODO: disable alchemical particle by resetting params, mask, should remain shared_ptr?

    protected:
    typedef std::bitset<evaluator::num_alchemical_parameters> mask_type;
    typedef std::array<Scalar, evaluator::num_alchemical_parameters> alpha_array_t;

    // templated base classes need us to pull members we want into scope or overuse this->
    using PotentialPair<evaluator, extra_pkg>::m_exec_conf;
    using PotentialPair<evaluator, extra_pkg>::m_pdata;

    Index2DUpperTriangular m_alchemy_index; //!< upper triangular typepair index
    std::vector<mask_type> m_alchemy_mask;  //!< Type pair mask for if alchemical forces are used
    std::vector<std::shared_ptr<alpha_particle_type>>
        m_alchemical_particles; //!< 2D array (alchemy_index,alchemical param)

    //! Method to be called when number of particles changes
    void slotNumParticlesChange()
        {
        unsigned int N = m_pdata->getN();
        for (auto& particle : m_alchemical_particles)
            if (particle)
                particle->resizeForces(N);
        };

    // Extra steps to insert
    inline extra_pkg pkgInitialize(const uint64_t& timestep) override;
    inline void pkgPerNeighbor(const unsigned int& i,
                               const unsigned int& j,
                               const unsigned int& typei,
                               const unsigned int& typej,
                               const bool& in_rcut,
                               evaluator& eval,
                               extra_pkg&) override;
    inline void pkgFinalize(extra_pkg&) override;
    };

template<class evaluator, typename extra_pkg, typename alpha_particle_type>
PotentialPairAlchemical<evaluator, extra_pkg, alpha_particle_type>::PotentialPairAlchemical(
    std::shared_ptr<SystemDefinition> sysdef,
    std::shared_ptr<NeighborList> nlist)
    : PotentialPair<evaluator, extra_pkg>(sysdef, nlist)
    {
    m_alchemy_index = Index2DUpperTriangular(m_pdata->getNTypes());
    m_alchemical_particles.resize(m_alchemy_index.getNumElements()
                                  * evaluator::num_alchemical_parameters);
    m_alchemy_mask.resize(m_pdata->getNTypes());

    m_exec_conf->msg->notice(5) << "Constructing PotentialPairAlchemical<" << evaluator::getName()
                                << ">" << std::endl;

    m_pdata->getGlobalParticleNumberChangeSignal()
        .template connect<PotentialPairAlchemical<evaluator, extra_pkg, alpha_particle_type>,
                          &PotentialPairAlchemical<evaluator, extra_pkg, alpha_particle_type>::
                              slotNumParticlesChange>(this);
    }

template<class evaluator, typename extra_pkg, typename alpha_particle_type>
PotentialPairAlchemical<evaluator, extra_pkg, alpha_particle_type>::~PotentialPairAlchemical()
    {
    m_exec_conf->msg->notice(5) << "Destroying PotentialPairAlchemical<" << evaluator::getName()
                                << ">" << std::endl;

    m_pdata->getGlobalParticleNumberChangeSignal()
        .template disconnect<PotentialPairAlchemical<evaluator, extra_pkg, alpha_particle_type>,
                             &PotentialPairAlchemical<evaluator, extra_pkg, alpha_particle_type>::
                                 slotNumParticlesChange>(this);
    }

template<class evaluator, typename extra_pkg, typename alpha_particle_type>
inline extra_pkg PotentialPairAlchemical<evaluator, extra_pkg, alpha_particle_type>::pkgInitialize(
    const uint64_t& timestep)
    {
    // Create pkg for passing additional variables between specialized code
    extra_pkg pkg;

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

    if (pkg.calculate_derivatives)
        {
        m_exec_conf->msg->notice(10)
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

template<class evaluator, typename extra_pkg, typename alpha_particle_type>
inline void PotentialPairAlchemical<evaluator, extra_pkg, alpha_particle_type>::pkgPerNeighbor(
    const unsigned int& i,
    const unsigned int& j,
    const unsigned int& typei,
    const unsigned int& typej,
    const bool& in_rcut,
    evaluator& eval,
    extra_pkg& pkg)
    {
    unsigned int alchemy_index = m_alchemy_index(typei, typej);
    mask_type& mask {pkg.compute_mask[alchemy_index]};
    alpha_array_t& alphas {pkg.alphas[alchemy_index]};

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

template<class evaluator, typename extra_pkg, typename alpha_particle_type>
inline void
PotentialPairAlchemical<evaluator, extra_pkg, alpha_particle_type>::pkgFinalize(extra_pkg& pkg)
    {
    // for all used variables, store the net force
    for (unsigned int i = 0; i < m_alchemy_index.getNumElements(); i++)
        for (unsigned int j = 0; j < evaluator::num_alchemical_parameters; j++)
            if (pkg.compute_mask[i][j])
                {
                m_alchemical_particles[j * m_alchemy_index.getNumElements() + i]->setNetForce();
                }
    }

//! Export this pair potential to python
/*! \param name Name of the class in the exported python module
    \tparam T Class type to export. \b Must be an instantiated PotentialPair class template.
*/
template<class evaluator,
         typename extra_pkg = AlchemyPackage<evaluator>,
         typename alpha_particle_type = AlchemicalPairParticle>
void export_PotentialPairAlchemical(pybind11::module& m, const std::string& name)
    {
    typedef PotentialPair<evaluator, extra_pkg> base;
    export_PotentialPair<base>(m, name + std::string("Base").c_str());
    typedef PotentialPairAlchemical<evaluator, extra_pkg, alpha_particle_type> T;
    pybind11::class_<T, base, std::shared_ptr<T>> PotentialPairAlchemical(m, name.c_str());
    PotentialPairAlchemical
        .def(pybind11::init<std::shared_ptr<SystemDefinition>, std::shared_ptr<NeighborList>>())
        .def("getAlchemicalPairParticle", &T::getAlchemicalPairParticle)
        .def("enableAlchemicalPairParticle", &T::enableAlchemicalPairParticle)
        .def("disableAlchemicalPairParticle", &T::disableAlchemicalPairParticle);
    }

#endif // POTENTIAL_PAIR_ALCHEMICAL_H
