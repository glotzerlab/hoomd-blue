// Copyright (c) 2009-2024 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

#pragma once

#include <bitset>
#include <cstddef>
#include <memory>
#include <numeric>

#include "AlchemyData.h"
#include "PotentialPair.h"

/*! \file PotentialPairAlchemical.h
    \brief Defines the template class for alchemical pair potentials
*/

#ifdef __HIPCC__
#error This header cannot be compiled by nvcc
#endif

namespace hoomd
    {

namespace md
    {

template<class evaluator> struct AlchemyPackage
    {
    bool calculate_derivatives = false;
    std::vector<std::array<Scalar, evaluator::num_alchemical_parameters>> alphas = {};
    std::vector<ArrayHandle<Scalar>> force_handles = {};
    std::vector<std::bitset<evaluator::num_alchemical_parameters>> compute_mask = {};

    AlchemyPackage(std::nullptr_t) { };
    AlchemyPackage() { };
    };

//! Template class for computing alchemical pair potentials
/*! <b>Overview:</b>

    <b>Implementation details</b>



    \sa export_PotentialPair()
*/
template<class evaluator,
         typename extra_pkg = AlchemyPackage<evaluator>,
         typename alpha_particle_type = AlchemicalPairParticle>
class PotentialPairAlchemical : public PotentialPair<evaluator>
    {
    public:
    //! Construct the pair potential
    PotentialPairAlchemical(std::shared_ptr<SystemDefinition> sysdef,
                            std::shared_ptr<NeighborList> nlist);
    //! Destructor
    virtual ~PotentialPairAlchemical();

    std::shared_ptr<alpha_particle_type> getAlchemicalPairParticle(pybind11::tuple types,
                                                                   std::string param_name)
        {
        int type_i = static_cast<int>(m_pdata->getTypeByName(types[0].cast<std::string>()));
        int type_j = static_cast<int>(m_pdata->getTypeByName(types[1].cast<std::string>()));
        int param_index = evaluator::getAlchemicalParameterIndex(param_name);

        std::shared_ptr<alpha_particle_type>& alpha_p
            = m_alchemical_particles[param_index * m_alchemy_index.getNumElements()
                                     + m_alchemy_index(type_i, type_j)];
        if (alpha_p == nullptr)
            {
            alpha_p = std::make_shared<alpha_particle_type>(m_exec_conf,
                                                            make_int3(type_i, type_j, param_index));
            }
        return alpha_p;
        }

    void enableAlchemicalPairParticle(std::shared_ptr<alpha_particle_type> alpha_p)
        {
        m_alchemy_mask[m_alchemy_index(alpha_p->m_type_pair_param.x, alpha_p->m_type_pair_param.y)]
                      [alpha_p->m_type_pair_param.z]
            = true;
        alpha_p->resizeForces(m_pdata->getN());
        }

    void disableAlchemicalPairParticle(std::shared_ptr<alpha_particle_type> alpha_p)
        {
        m_alchemy_mask[m_alchemy_index(alpha_p->m_type_pair_param.x, alpha_p->m_type_pair_param.y)]
                      [alpha_p->m_type_pair_param.z]
            = false;
        }

    protected:
    typedef std::bitset<evaluator::num_alchemical_parameters> mask_type;
    typedef std::array<Scalar, evaluator::num_alchemical_parameters> alpha_array_t;

    // allow copy and paste from PotentialPair without using this-> on every member
    using PotentialPair<evaluator>::m_exec_conf;
    using PotentialPair<evaluator>::m_sysdef;
    using PotentialPair<evaluator>::m_nlist;
    using PotentialPair<evaluator>::m_virial;
    using PotentialPair<evaluator>::m_ronsq;
    using PotentialPair<evaluator>::m_rcutsq;
    using PotentialPair<evaluator>::m_force;
    using PotentialPair<evaluator>::m_typpair_idx;
    using PotentialPair<evaluator>::m_pdata;
    using PotentialPair<evaluator>::m_shift_mode;
    using PotentialPair<evaluator>::xplor;
    using PotentialPair<evaluator>::shift;
    using PotentialPair<evaluator>::m_virial_pitch;
    using PotentialPair<evaluator>::m_params;
    using PotentialPair<evaluator>::computeTailCorrection;
#ifdef ENABLE_MPI
    using PotentialPair<evaluator>::m_comm;
#endif

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
    virtual inline extra_pkg pkgInitialize(const uint64_t& timestep);
    virtual inline void pkgPerNeighbor(const unsigned int& i,
                                       const unsigned int& j,
                                       const unsigned int& typei,
                                       const unsigned int& typej,
                                       const bool& in_rcut,
                                       evaluator& eval,
                                       extra_pkg&);
    virtual inline void pkgFinalize(extra_pkg&);

    virtual void computeForces(uint64_t timestep);
    };

template<class evaluator, typename extra_pkg, typename alpha_particle_type>
PotentialPairAlchemical<evaluator, extra_pkg, alpha_particle_type>::PotentialPairAlchemical(
    std::shared_ptr<SystemDefinition> sysdef,
    std::shared_ptr<NeighborList> nlist)
    : PotentialPair<evaluator>(sysdef, nlist)
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
                particle->zeroNetForce(timestep);
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

    // calculate alchemical derivatives if needed
    if (pkg.calculate_derivatives && in_rcut && mask.any())
        {
        std::array<Scalar, evaluator::num_alchemical_parameters> alchemical_derivatives = {};
        eval.evalAlchemyDerivatives(alchemical_derivatives, alphas);
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
    eval.updateAlchemyParams(alphas);
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

/*! Compute pair forces with extra alchemical derivatives.
 */
template<class evaluator, typename extra_pkg, typename alpha_particle_type>
void PotentialPairAlchemical<evaluator, extra_pkg, alpha_particle_type>::computeForces(
    uint64_t timestep)
    {
    // start by updating the neighborlist
    m_nlist->compute(timestep);

    // depending on the neighborlist settings, we can take advantage of newton's third law
    // to reduce computations at the cost of memory access complexity: set that flag now
    bool third_law = m_nlist->getStorageMode() == NeighborList::half;

    // access the neighbor list, particle data, and system box
    ArrayHandle<unsigned int> h_n_neigh(m_nlist->getNNeighArray(),
                                        access_location::host,
                                        access_mode::read);
    ArrayHandle<unsigned int> h_nlist(m_nlist->getNListArray(),
                                      access_location::host,
                                      access_mode::read);
    //     Index2D nli = m_nlist->getNListIndexer();
    ArrayHandle<size_t> h_head_list(m_nlist->getHeadList(),
                                    access_location::host,
                                    access_mode::read);

    ArrayHandle<Scalar4> h_pos(m_pdata->getPositions(), access_location::host, access_mode::read);
    ArrayHandle<Scalar> h_charge(m_pdata->getCharges(), access_location::host, access_mode::read);

    // force arrays
    ArrayHandle<Scalar4> h_force(m_force, access_location::host, access_mode::overwrite);
    ArrayHandle<Scalar> h_virial(m_virial, access_location::host, access_mode::overwrite);

    const BoxDim box = m_pdata->getGlobalBox();
    ArrayHandle<Scalar> h_ronsq(m_ronsq, access_location::host, access_mode::read);
    ArrayHandle<Scalar> h_rcutsq(m_rcutsq, access_location::host, access_mode::read);

    PDataFlags flags = this->m_pdata->getFlags();
    bool compute_virial = flags[pdata_flag::pressure_tensor];

    // need to start from a zero force, energy and virial
    memset((void*)h_force.data, 0, sizeof(Scalar4) * m_force.getNumElements());
    memset((void*)h_virial.data, 0, sizeof(Scalar) * m_virial.getNumElements());

    extra_pkg pkg = pkgInitialize(timestep);

    // for each particle
    for (int i = 0; i < (int)m_pdata->getN(); i++)
        {
        // access the particle's position and type (MEM TRANSFER: 4 scalars)
        Scalar3 pi = make_scalar3(h_pos.data[i].x, h_pos.data[i].y, h_pos.data[i].z);
        unsigned int typei = __scalar_as_int(h_pos.data[i].w);

        // sanity check
        assert(typei < m_pdata->getNTypes());

        // access charge (if needed)
        Scalar qi = Scalar(0.0);
        if (evaluator::needsCharge())
            qi = h_charge.data[i];

        // initialize current particle force, potential energy, and virial to 0
        Scalar3 fi = make_scalar3(0, 0, 0);
        Scalar pei = 0.0;
        Scalar virialxxi = 0.0;
        Scalar virialxyi = 0.0;
        Scalar virialxzi = 0.0;
        Scalar virialyyi = 0.0;
        Scalar virialyzi = 0.0;
        Scalar virialzzi = 0.0;

        // loop over all of the neighbors of this particle
        const size_t myHead = h_head_list.data[i];
        const unsigned int size = (unsigned int)h_n_neigh.data[i];
        for (unsigned int k = 0; k < size; k++)
            {
            // access the index of this neighbor (MEM TRANSFER: 1 scalar)
            unsigned int j = h_nlist.data[myHead + k];
            assert(j < m_pdata->getN() + m_pdata->getNGhosts());

            // calculate dr_ji (MEM TRANSFER: 3 scalars / FLOPS: 3)
            Scalar3 pj = make_scalar3(h_pos.data[j].x, h_pos.data[j].y, h_pos.data[j].z);
            Scalar3 dx = pi - pj;

            // access the type of the neighbor particle (MEM TRANSFER: 1 scalar)
            unsigned int typej = __scalar_as_int(h_pos.data[j].w);
            assert(typej < m_pdata->getNTypes());

            // access charge (if needed)
            Scalar qj = Scalar(0.0);
            if (evaluator::needsCharge())
                qj = h_charge.data[j];

            // apply periodic boundary conditions
            dx = box.minImage(dx);

            // calculate r_ij squared (FLOPS: 5)
            Scalar rsq = dot(dx, dx);

            // get parameters for this type pair
            unsigned int typpair_idx = m_typpair_idx(typei, typej);
            const auto& param = m_params[typpair_idx];
            Scalar rcutsq = h_rcutsq.data[typpair_idx];
            Scalar ronsq = Scalar(0.0);
            if (m_shift_mode == xplor)
                ronsq = h_ronsq.data[typpair_idx];

            // design specifies that energies are shifted if
            // 1) shift mode is set to shift
            // or 2) shift mode is explor and ron > rcut
            bool energy_shift = false;
            if (m_shift_mode == shift)
                energy_shift = true;
            else if (m_shift_mode == xplor)
                {
                if (ronsq > rcutsq)
                    energy_shift = true;
                }

            // compute the force and potential energy
            Scalar force_divr = Scalar(0.0);
            Scalar pair_eng = Scalar(0.0);
            evaluator eval(rsq, rcutsq, param);
            if (evaluator::needsCharge())
                eval.setCharge(qi, qj);

            pkgPerNeighbor(i, j, typei, typej, (rsq < rcutsq), eval, pkg);

            bool evaluated = eval.evalForceAndEnergy(force_divr, pair_eng, energy_shift);

            if (evaluated)
                {
                // modify the potential for xplor shifting
                if (m_shift_mode == xplor)
                    {
                    if (rsq >= ronsq && rsq < rcutsq)
                        {
                        // Implement XPLOR smoothing (FLOPS: 16)
                        Scalar old_pair_eng = pair_eng;
                        Scalar old_force_divr = force_divr;

                        // calculate 1.0 / (xplor denominator)
                        Scalar xplor_denom_inv
                            = Scalar(1.0)
                              / ((rcutsq - ronsq) * (rcutsq - ronsq) * (rcutsq - ronsq));

                        Scalar rsq_minus_r_cut_sq = rsq - rcutsq;
                        Scalar s = rsq_minus_r_cut_sq * rsq_minus_r_cut_sq
                                   * (rcutsq + Scalar(2.0) * rsq - Scalar(3.0) * ronsq)
                                   * xplor_denom_inv;
                        Scalar ds_dr_divr
                            = Scalar(12.0) * (rsq - ronsq) * rsq_minus_r_cut_sq * xplor_denom_inv;

                        // make modifications to the old pair energy and force
                        pair_eng = old_pair_eng * s;
                        // note: I'm not sure why the minus sign needs to be there: my notes have a
                        // + But this is verified correct via plotting
                        force_divr = s * old_force_divr - ds_dr_divr * old_pair_eng;
                        }
                    }

                Scalar force_div2r = force_divr * Scalar(0.5);
                // add the force, potential energy and virial to the particle i
                // (FLOPS: 8)
                fi += dx * force_divr;
                pei += pair_eng * Scalar(0.5);
                if (compute_virial)
                    {
                    virialxxi += force_div2r * dx.x * dx.x;
                    virialxyi += force_div2r * dx.x * dx.y;
                    virialxzi += force_div2r * dx.x * dx.z;
                    virialyyi += force_div2r * dx.y * dx.y;
                    virialyzi += force_div2r * dx.y * dx.z;
                    virialzzi += force_div2r * dx.z * dx.z;
                    }

                // add the force to particle j if we are using the third law (MEM TRANSFER: 10
                // scalars / FLOPS: 8) only add force to local particles
                if (third_law && j < m_pdata->getN())
                    {
                    unsigned int mem_idx = j;
                    h_force.data[mem_idx].x -= dx.x * force_divr;
                    h_force.data[mem_idx].y -= dx.y * force_divr;
                    h_force.data[mem_idx].z -= dx.z * force_divr;
                    h_force.data[mem_idx].w += pair_eng * Scalar(0.5);
                    if (compute_virial)
                        {
                        h_virial.data[0 * m_virial_pitch + mem_idx] += force_div2r * dx.x * dx.x;
                        h_virial.data[1 * m_virial_pitch + mem_idx] += force_div2r * dx.x * dx.y;
                        h_virial.data[2 * m_virial_pitch + mem_idx] += force_div2r * dx.x * dx.z;
                        h_virial.data[3 * m_virial_pitch + mem_idx] += force_div2r * dx.y * dx.y;
                        h_virial.data[4 * m_virial_pitch + mem_idx] += force_div2r * dx.y * dx.z;
                        h_virial.data[5 * m_virial_pitch + mem_idx] += force_div2r * dx.z * dx.z;
                        }
                    }
                }
            }

        // finally, increment the force, potential energy and virial for particle i
        unsigned int mem_idx = i;
        h_force.data[mem_idx].x += fi.x;
        h_force.data[mem_idx].y += fi.y;
        h_force.data[mem_idx].z += fi.z;
        h_force.data[mem_idx].w += pei;
        if (compute_virial)
            {
            h_virial.data[0 * m_virial_pitch + mem_idx] += virialxxi;
            h_virial.data[1 * m_virial_pitch + mem_idx] += virialxyi;
            h_virial.data[2 * m_virial_pitch + mem_idx] += virialxzi;
            h_virial.data[3 * m_virial_pitch + mem_idx] += virialyyi;
            h_virial.data[4 * m_virial_pitch + mem_idx] += virialyzi;
            h_virial.data[5 * m_virial_pitch + mem_idx] += virialzzi;
            }
        }
    pkgFinalize(pkg);

    computeTailCorrection();
    }

namespace detail
    {

//! Export this pair potential to python
/*! \param name Name of the class in the exported python module
    \tparam T Evaluator type to export.
*/
template<class evaluator,
         typename extra_pkg = AlchemyPackage<evaluator>,
         typename alpha_particle_type = AlchemicalPairParticle>
void export_PotentialPairAlchemical(pybind11::module& m, const std::string& name)
    {
    typedef PotentialPair<evaluator> base;
    typedef PotentialPairAlchemical<evaluator, extra_pkg, alpha_particle_type> T;
    pybind11::class_<T, base, std::shared_ptr<T>> PotentialPairAlchemical(m, name.c_str());
    PotentialPairAlchemical
        .def(pybind11::init<std::shared_ptr<SystemDefinition>, std::shared_ptr<NeighborList>>())
        .def("getAlchemicalPairParticle", &T::getAlchemicalPairParticle)
        .def("enableAlchemicalPairParticle", &T::enableAlchemicalPairParticle)
        .def("disableAlchemicalPairParticle", &T::disableAlchemicalPairParticle);
    }

    } // end namespace detail

    } // end namespace md

    } // end namespace hoomd
