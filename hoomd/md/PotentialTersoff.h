// Copyright (c) 2009-2024 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

#ifndef __POTENTIAL_TERSOFF_H__
#define __POTENTIAL_TERSOFF_H__

#include <fstream>
#include <iostream>
#include <memory>
#include <stdexcept>

#include "NeighborList.h"
#include "hoomd/ForceCompute.h"
#include "hoomd/GPUArray.h"
#include "hoomd/HOOMDMath.h"
#include "hoomd/Index1D.h"

/*! \file PotentialTersoff.h
    \brief Defines the template class for standard three-body potentials
    \details The heart of the code that computes three-body potentials is in this file.
    \note This header cannot be compiled by nvcc
*/

#ifdef __HIPCC__
#error This header cannot be compiled by nvcc
#endif

#include <pybind11/pybind11.h>

namespace hoomd
    {
namespace md
    {
//! Template class for computing three-body potentials
/*! <b>Overview:</b>
    PotentialTersoff computes standard three-body potentials and forces between all particles in the
    simulation.  It employs the use of a neighbor list to limit the number of computations done to
    only those particles within the cutoff radius of each other.  the computation of the actual
    potential is not performed directly by this class, but by an evaluator class (e.g.
    EvaluatorTersoff) which is passed in as a template parameter so the computations are performed
    as efficiently as possible.

    PotentialTersoff handles most of the internal details common to all standard three-body
   potentials.
     - A cutoff radius to be specified per particle type-pair
     - Per type-pair parameters are stored and a set method is provided
     - All the details about looping through the particles, computing dr, computing the virial, etc.
   are handled

    <b>Implementation details</b>

    Unlike the pair potentials, the three-body potentials offer two force directions: ij and ik.
    In addition, some three-body potentials (such as the Tersoff potential) compute unique forces on
    each of the three particles involved.  Three-body evaluators must thus return six force
   magnitudes: two for each particle.  These values are returned in the Scalar3 values \a
   force_divr_ij and \a force_divr_ik.  The x components refer to particle i, y to particle j, and z
   to particle k. If your particular three-body potential does not compute one of these forces, then
   the evaluator can simply return 0 for that force.  In addition, the potential energy is stored in
   the w component of force_divr_ij.

    rcutsq, ronsq, and the params are stored per particle type-pair. It wastes a little bit of
   space, but benchmarks show that storing the symmetric type pairs and indexing with Index2D is
   faster than not storing redundant pairs and indexing with Index2DUpperTriangular. All of these
   values are stored in GPUArray for easy access on the GPU by a derived class. The type of the
   parameters is defined by \a param_type in the potential evaluator class passed in. See the
   appropriate documentation for the evaluator for the definition of each element of the parameters.

    \sa export_PotentialTersoff()
*/
template<class evaluator> class PotentialTersoff : public ForceCompute
    {
    public:
    //! Param type from evaluator
    typedef typename evaluator::param_type param_type;

    //! Construct the potential
    PotentialTersoff(std::shared_ptr<SystemDefinition> sysdef, std::shared_ptr<NeighborList> nlist);
    //! Destructor
    virtual ~PotentialTersoff();

    //! Set the pair parameters for a single type pair
    virtual void setParams(unsigned int typ1, unsigned int typ2, const param_type& param);
    virtual void setParamsPython(pybind11::tuple typ, pybind11::dict params);
    /// Get params for a single type pair using a tuple of strings
    virtual pybind11::dict getParams(pybind11::tuple typ);
    //! Set the rcut for a single type pair
    virtual void setRcut(unsigned int typ1, unsigned int typ2, Scalar rcut);
    //! Get the rcut for a single type pair
    Scalar getRCut(pybind11::tuple types);
    /// Set the rcut for a single type pair using a tuple of strings
    virtual void setRCutPython(pybind11::tuple types, Scalar r_cut);

    /// Validate that types are within Ntypes
    virtual void validateTypes(unsigned int typ1, unsigned int typ2, std::string action);

    virtual void notifyDetach()
        {
        if (m_attached)
            {
            m_nlist->removeRCutMatrix(m_r_cut_nlist);
            }
        m_attached = false;
        }

#ifdef ENABLE_MPI
    //! Get ghost particle fields requested by this pair potential
    virtual CommFlags getRequestedCommFlags(uint64_t timestep);
#endif

    /// Start autotuning kernel launch parameters
    virtual void startAutotuning()
        {
        ForceCompute::startAutotuning();

        // Start autotuning the neighbor list.
        m_nlist->startAutotuning();
        }

    /// Check if autotuning is complete.
    virtual bool isAutotuningComplete()
        {
        bool result = ForceCompute::isAutotuningComplete();
        return result && m_nlist->isAutotuningComplete();
        }

    protected:
    std::shared_ptr<NeighborList> m_nlist; //!< The neighborlist to use for the computation
    Index2D m_typpair_idx;                 //!< Helper class for indexing per type pair arrays
    GPUArray<Scalar> m_rcutsq;             //!< Cutoff radius squared per type pair
    GPUArray<param_type> m_params;         //!< Pair parameters per type pair

    // track whether we are attached to the simulation
    bool m_attached = true;

    // r_cut (not squared) given to the neighborlist
    std::shared_ptr<GlobalArray<Scalar>> m_r_cut_nlist;

    //! Actually compute the forces
    virtual void computeForces(uint64_t timestep);
    };

/*! \param sysdef System to compute forces on
    \param nlist Neighborlist to use for computing the forces
*/
template<class evaluator>
PotentialTersoff<evaluator>::PotentialTersoff(std::shared_ptr<SystemDefinition> sysdef,
                                              std::shared_ptr<NeighborList> nlist)
    : ForceCompute(sysdef), m_nlist(nlist), m_typpair_idx(m_pdata->getNTypes())
    {
    this->m_exec_conf->msg->notice(5) << "Constructing PotentialTersoff" << std::endl;

    assert(m_pdata);
    assert(m_nlist);

    GPUArray<Scalar> rcutsq(m_typpair_idx.getNumElements(), m_exec_conf);
    m_rcutsq.swap(rcutsq);
    GPUArray<param_type> params(m_typpair_idx.getNumElements(), m_exec_conf);
    m_params.swap(params);

    m_r_cut_nlist
        = std::make_shared<GlobalArray<Scalar>>(m_typpair_idx.getNumElements(), m_exec_conf);
    nlist->addRCutMatrix(m_r_cut_nlist);
    }

template<class evaluator> PotentialTersoff<evaluator>::~PotentialTersoff()
    {
    this->m_exec_conf->msg->notice(5) << "Destroying PotentialTersoff" << std::endl;
    if (m_attached)
        {
        m_nlist->removeRCutMatrix(m_r_cut_nlist);
        }
    }

/*! \param typ1 First type index in the pair
    \param typ2 Second type index in the pair
    \param param Parameter to set
    \note When setting the value for (\a typ1, \a typ2), the parameter for (\a typ2, \a typ1) is
   automatically set.
*/
template<class evaluator>
void PotentialTersoff<evaluator>::setParams(unsigned int typ1,
                                            unsigned int typ2,
                                            const param_type& param)
    {
    validateTypes(typ1, typ2, "set params");
    ArrayHandle<param_type> h_params(m_params, access_location::host, access_mode::readwrite);
    h_params.data[m_typpair_idx(typ1, typ2)] = param;
    h_params.data[m_typpair_idx(typ2, typ1)] = param;
    }

template<class evaluator>
void PotentialTersoff<evaluator>::setParamsPython(pybind11::tuple typ, pybind11::dict params)
    {
    auto typ1 = m_pdata->getTypeByName(typ[0].cast<std::string>());
    auto typ2 = m_pdata->getTypeByName(typ[1].cast<std::string>());
    validateTypes(typ1, typ2, "set params");
    ArrayHandle<param_type> h_params(m_params, access_location::host, access_mode::readwrite);
    h_params.data[m_typpair_idx(typ1, typ2)] = param_type(params);
    h_params.data[m_typpair_idx(typ2, typ1)] = param_type(params);
    }

template<class evaluator> pybind11::dict PotentialTersoff<evaluator>::getParams(pybind11::tuple typ)
    {
    auto typ1 = m_pdata->getTypeByName(typ[0].cast<std::string>());
    auto typ2 = m_pdata->getTypeByName(typ[1].cast<std::string>());
    validateTypes(typ1, typ2, "get params");
    ArrayHandle<param_type> h_params(m_params, access_location::host, access_mode::read);
    return h_params.data[m_typpair_idx(typ1, typ2)].asDict();
    }

/*! \param typ1 First type index in the pair
    \param typ2 Second type index in the pair
    \param rcut Cutoff radius to set
    \note When setting the value for (\a typ1, \a typ2), the parameter for (\a typ2, \a typ1) is
   automatically set.
*/
template<class evaluator>
void PotentialTersoff<evaluator>::setRcut(unsigned int typ1, unsigned int typ2, Scalar rcut)
    {
    validateTypes(typ1, typ2, "set r_cut");
    ArrayHandle<Scalar> h_rcutsq(m_rcutsq, access_location::host, access_mode::readwrite);
    h_rcutsq.data[m_typpair_idx(typ1, typ2)] = rcut * rcut;
    h_rcutsq.data[m_typpair_idx(typ2, typ1)] = rcut * rcut;

    ArrayHandle<Scalar> h_r_cut_nlist(*m_r_cut_nlist,
                                      access_location::host,
                                      access_mode::readwrite);
    h_r_cut_nlist.data[m_typpair_idx(typ1, typ2)] = rcut;
    h_r_cut_nlist.data[m_typpair_idx(typ2, typ1)] = rcut;

    m_nlist->notifyRCutMatrixChange();
    }

template<class evaluator>
void PotentialTersoff<evaluator>::setRCutPython(pybind11::tuple types, Scalar r_cut)
    {
    auto typ1 = m_pdata->getTypeByName(types[0].cast<std::string>());
    auto typ2 = m_pdata->getTypeByName(types[1].cast<std::string>());
    validateTypes(typ1, typ2, "set r_cut");
    setRcut(typ1, typ2, r_cut);
    }

template<class evaluator>
void PotentialTersoff<evaluator>::validateTypes(unsigned int typ1,
                                                unsigned int typ2,
                                                std::string action)
    {
    // TODO change logic to just throw an exception
    auto n_types = this->m_pdata->getNTypes();
    if (typ1 >= n_types || typ2 >= n_types)
        {
        this->m_exec_conf->msg->error()
            << "pair." << evaluator::getName() << ": Trying to " << action
            << " for a non existent type! " << typ1 << "," << typ2 << std::endl;
        throw std::runtime_error("Error setting parameters in PotentialTersoff");
        }
    }

template<class evaluator> Scalar PotentialTersoff<evaluator>::getRCut(pybind11::tuple types)
    {
    auto typ1 = m_pdata->getTypeByName(types[0].cast<std::string>());
    auto typ2 = m_pdata->getTypeByName(types[1].cast<std::string>());
    validateTypes(typ1, typ2, "get rcut.");
    ArrayHandle<Scalar> h_rcutsq(m_rcutsq, access_location::host, access_mode::read);
    return sqrt(h_rcutsq.data[m_typpair_idx(typ1, typ2)]);
    }

/*! \post The forces are computed for the given timestep. The neighborlist's compute method is
   called to ensure that it is up to date before proceeding.

    \param timestep specifies the current time step of the simulation
*/
template<class evaluator> void PotentialTersoff<evaluator>::computeForces(uint64_t timestep)
    {
    // *****  check if we need the structure of the Tersoff or the RevCross potential for evaluation
    if (evaluator::flag_for_RevCross)
        {
        // ***** RevCross potential
        // start by updating the neighborlist
        m_nlist->compute(timestep);

        // The three-body potentials can't handle a half neighbor list, so check now.
        bool third_law = m_nlist->getStorageMode() == NeighborList::half;
        if (third_law)
            {
            m_exec_conf->msg->error()
                << std::endl
                << "PotentialRevCross cannot handle a half neighborlist" << std::endl;
            throw std::runtime_error("Error computing forces in PotentialRevCross");
            }

        // access the neighbor list, particle data, and system box
        ArrayHandle<unsigned int> h_n_neigh(m_nlist->getNNeighArray(),
                                            access_location::host,
                                            access_mode::read);
        ArrayHandle<unsigned int> h_nlist(m_nlist->getNListArray(),
                                          access_location::host,
                                          access_mode::read);
        ArrayHandle<size_t> h_head_list(m_nlist->getHeadList(),
                                        access_location::host,
                                        access_mode::read);

        ArrayHandle<Scalar4> h_pos(m_pdata->getPositions(),
                                   access_location::host,
                                   access_mode::read);

        // force and virial arrays
        ArrayHandle<Scalar4> h_force(m_force, access_location::host, access_mode::overwrite);
        ArrayHandle<Scalar> h_virial(m_virial, access_location::host, access_mode::overwrite);

        PDataFlags flags = this->m_pdata->getFlags();
        bool compute_virial = flags[pdata_flag::pressure_tensor];

        const BoxDim box = m_pdata->getBox();
        ArrayHandle<Scalar> h_rcutsq(m_rcutsq, access_location::host, access_mode::read);
        ArrayHandle<param_type> h_params(m_params, access_location::host, access_mode::read);

        // need to start from a zero force, energy
        memset(h_force.data, 0, sizeof(Scalar4) * (m_pdata->getN() + m_pdata->getNGhosts()));
        memset(h_virial.data, 0, sizeof(Scalar) * 6 * m_virial_pitch);

        // for each particle
        for (int i = 0; i < (int)m_pdata->getN(); i++)
            {
            // access the particle's position and type (MEM TRANSFER: 4 scalars)
            Scalar3 posi = make_scalar3(h_pos.data[i].x, h_pos.data[i].y, h_pos.data[i].z);
            unsigned int typei = __scalar_as_int(h_pos.data[i].w);
            const size_t head_i = h_head_list.data[i];
            // sanity check
            assert(typei < m_pdata->getNTypes());

            // initialize current force and potential energy of particle i to 0
            Scalar3 fi = make_scalar3(0.0, 0.0, 0.0);
            Scalar pei = 0.0;

            Scalar virialixx(0.0);
            Scalar virialixy(0.0);
            Scalar virialixz(0.0);
            Scalar virialiyy(0.0);
            Scalar virialiyz(0.0);
            Scalar virializz(0.0);

            // loop over all of the neighbors of this particle
            const unsigned int size = (unsigned int)h_n_neigh.data[i];
            for (unsigned int j = 0; j < size; j++)
                {
                // access the index of neighbor j (MEM TRANSFER: 1 scalar)
                unsigned int jj = h_nlist.data[head_i + j];
                assert(jj < m_pdata->getN() + m_pdata->getNGhosts());

                // access the position and type of particle j
                Scalar3 posj = make_scalar3(h_pos.data[jj].x, h_pos.data[jj].y, h_pos.data[jj].z);
                unsigned int typej = __scalar_as_int(h_pos.data[jj].w);
                assert(typej < m_pdata->getNTypes());

                // initialize the current force and potential energy of particle j to 0
                Scalar3 fj = make_scalar3(0.0, 0.0, 0.0);
                Scalar pej = 0.0;

                // calculate dr_ij (MEM TRANSFER: 3 scalars / FLOPS: 3)
                Scalar3 dxij = posi - posj;

                // apply periodic boundary conditions
                dxij = box.minImage(dxij);

                // compute rij_sq (FLOPS: 5)
                Scalar rij_sq = dot(dxij, dxij);

                // get parameters for this type pair
                unsigned int typpair_idx = m_typpair_idx(typei, typej);
                const param_type& param = h_params.data[typpair_idx];
                Scalar rcutsq = h_rcutsq.data[typpair_idx];

                // evaluate the base repulsive and attractive terms
                Scalar invratio = 0.0;
                Scalar invratio2 = 0.0;
                evaluator eval(rij_sq, rcutsq, param);
                bool evaluated = eval.evalRepulsiveAndAttractive(invratio, invratio2);

                // Even though the i-j interaction is symmetric so in principle I could consider i>j
                // only, I have to loop over both i-j-k and j-i-k because I search only in neighbors
                // of of the first element (since nl are type-wise I can not even merge them because
                // i, j and k could be different types)
                if (evaluated)
                    {
                    // printf("\nEvaluating the pair (i,j)=(%d, %d)  from inside HOOMD CPU",i,jj);
                    // evaluate the force and energy from the ij interaction
                    Scalar force_divr = Scalar(0.0);
                    Scalar potential_eng = Scalar(0.0);
                    Scalar bij = Scalar(0.0); // not used
                    eval.evalForceij(invratio,
                                     invratio2,
                                     Scalar(0.0),
                                     Scalar(0.0),
                                     bij,
                                     force_divr,
                                     potential_eng);

                    // add this force to particle i
                    fi += force_divr * dxij;
                    pei += potential_eng;

                    // add this force to particle j
                    fj += Scalar(-1.0) * force_divr * dxij;
                    pej += potential_eng;

                    // vir contribute for i j direct interaction on particle i and j
                    if (compute_virial)
                        {
                        virialixx += force_divr * dxij.x * dxij.x;
                        virialixy += force_divr * dxij.x * dxij.y;
                        virialixz += force_divr * dxij.x * dxij.z;
                        virialiyy += force_divr * dxij.y * dxij.y;
                        virialiyz += force_divr * dxij.y * dxij.z;
                        virializz += force_divr * dxij.z * dxij.z;
                        }

                    // evaluate the force from the ik interactions
                    for (unsigned int k = j + 1; k < size;
                         k++) // I want to account only a single time for each triplets
                        {
                        // access the index of neighbor k
                        unsigned int kk = h_nlist.data[head_i + k];
                        assert(kk < m_pdata->getN() + m_pdata->getNGhosts());

                        // access the position and type of neighbor k
                        Scalar3 posk
                            = make_scalar3(h_pos.data[kk].x, h_pos.data[kk].y, h_pos.data[kk].z);
                        unsigned int typek = __scalar_as_int(h_pos.data[kk].w);
                        assert(typek < m_pdata->getNTypes());

                        // access the type pair parameters for i and k
                        typpair_idx = m_typpair_idx(typei, typek);
                        param_type temp_param
                            = h_params.data[typpair_idx]; // use this to control the species wich
                                                          // have to interact

                        // compute dr_ik
                        Scalar3 dxik = posi - posk;
                        // apply periodic boundary conditions
                        dxik = box.minImage(dxik);
                        // compute rik_sq
                        Scalar rik_sq = dot(dxik, dxik);

                        // check if k interacts using a temporary evaluator to analyze i-k
                        // parameters
                        evaluator temp_eval(rij_sq, rcutsq, temp_param);
                        temp_eval.setRik(rik_sq);
                        bool temp_evaluated = temp_eval.areInteractive();

                        // 3 Body interaction ******
                        if (temp_evaluated)
                            {
                            eval.setRik(rik_sq);
                            // compute the total force and energy
                            Scalar3 fk = make_scalar3(0.0, 0.0, 0.0);
                            Scalar3 force_divr_ij_vec = make_scalar3(0.0, 0.0, 0.0);
                            Scalar3 force_divr_ik_vec = make_scalar3(0.0, 0.0, 0.0);
                            bool evaluatedk = eval.evalForceik(invratio,
                                                               invratio2,
                                                               Scalar(0.0),
                                                               Scalar(0.0),
                                                               force_divr_ij_vec,
                                                               force_divr_ik_vec);
                            // k interacts with the i-j as an additional third body
                            if (evaluatedk)
                                {
                                // I stored the modulus of the force in the first component
                                Scalar force_divr_ij = force_divr_ij_vec.x;
                                Scalar force_divr_ik = force_divr_ik_vec.x;

                                // add the force to particle i
                                fi += force_divr_ij * dxij + force_divr_ik * dxik;

                                // add the force to particle j (FLOPS: 17)
                                fj += force_divr_ij * dxij * Scalar(-1.0);

                                // add the force to particle k
                                fk += force_divr_ik * dxik * Scalar(-1.0);

                                if (compute_virial)
                                    {
                                    //***look at 3 body pressure notes
                                    // i just need a single term to account for all of the 3 body
                                    // virial that i decide to store in the i particle's data and i
                                    // just defined the diagonal component of pressure tensor, I
                                    // don't know how the off diagonal terms can be included
                                    virialixx += (force_divr_ij * dxij.x * dxij.x
                                                  + force_divr_ik * dxik.x * dxik.x);
                                    virialiyy += (force_divr_ij * dxij.y * dxij.y
                                                  + force_divr_ik * dxik.y * dxik.y);
                                    virializz += (force_divr_ij * dxij.z * dxij.z
                                                  + force_divr_ik * dxik.z * dxik.z);
                                    virialixy += (force_divr_ij * dxij.x * dxij.y
                                                  + force_divr_ik * dxik.x * dxik.y);
                                    virialixz += (force_divr_ij * dxij.x * dxij.z
                                                  + force_divr_ik * dxik.x * dxik.z);
                                    virialiyz += (force_divr_ij * dxij.y * dxij.z
                                                  + force_divr_ik * dxik.y * dxik.z);
                                    }

                                // increment the force for particle k
                                unsigned int mem_idx = kk;
                                h_force.data[mem_idx].x += fk.x;
                                h_force.data[mem_idx].y += fk.y;
                                h_force.data[mem_idx].z += fk.z;
                                }
                            }
                        }
                    }

                // increment the force and potential energy for particle j
                unsigned int mem_idx = jj;
                h_force.data[mem_idx].x += fj.x;
                h_force.data[mem_idx].y += fj.y;
                h_force.data[mem_idx].z += fj.z;
                h_force.data[mem_idx].w += pej;
                }

            // finally, increment the force and potential energy for particle i
            unsigned int mem_idx = i;
            h_force.data[mem_idx].x += fi.x;
            h_force.data[mem_idx].y += fi.y;
            h_force.data[mem_idx].z += fi.z;
            h_force.data[mem_idx].w += pei;

            // imcrement vir for i
            if (compute_virial)
                {
                h_virial.data[0 * m_virial_pitch + mem_idx] += virialixx;
                h_virial.data[1 * m_virial_pitch + mem_idx] += virialixy;
                h_virial.data[2 * m_virial_pitch + mem_idx] += virialixz;
                h_virial.data[3 * m_virial_pitch + mem_idx] += virialiyy;
                h_virial.data[4 * m_virial_pitch + mem_idx] += virialiyz;
                h_virial.data[5 * m_virial_pitch + mem_idx] += virializz;
                }
            }
        }
    else
        {
        // ****** Tersoff or SquareDensity potential
        // start by updating the neighborlist
        m_nlist->compute(timestep);

        // The three-body potentials can't handle a half neighbor list, so check now.
        bool third_law = m_nlist->getStorageMode() == NeighborList::half;
        if (third_law)
            {
            m_exec_conf->msg->error()
                << std::endl
                << "PotentialTersoff cannot handle a half neighborlist" << std::endl;
            throw std::runtime_error("Error computing forces in PotentialTersoff");
            }

        // access the neighbor list, particle data, and system box
        ArrayHandle<unsigned int> h_n_neigh(m_nlist->getNNeighArray(),
                                            access_location::host,
                                            access_mode::read);
        ArrayHandle<unsigned int> h_nlist(m_nlist->getNListArray(),
                                          access_location::host,
                                          access_mode::read);
        ArrayHandle<size_t> h_head_list(m_nlist->getHeadList(),
                                        access_location::host,
                                        access_mode::read);

        ArrayHandle<Scalar4> h_pos(m_pdata->getPositions(),
                                   access_location::host,
                                   access_mode::read);

        // force and virial arrays
        ArrayHandle<Scalar4> h_force(m_force, access_location::host, access_mode::overwrite);
        ArrayHandle<Scalar> h_virial(m_virial, access_location::host, access_mode::overwrite);

        PDataFlags flags = this->m_pdata->getFlags();
        bool compute_virial = flags[pdata_flag::pressure_tensor];

        const BoxDim box = m_pdata->getBox();
        ArrayHandle<Scalar> h_rcutsq(m_rcutsq, access_location::host, access_mode::read);
        ArrayHandle<param_type> h_params(m_params, access_location::host, access_mode::read);

        // need to start from a zero force, energy
        memset(h_force.data, 0, sizeof(Scalar4) * (m_pdata->getN() + m_pdata->getNGhosts()));
        memset(h_virial.data, 0, sizeof(Scalar) * 6 * m_virial_pitch);

        unsigned int ntypes = m_pdata->getNTypes();

        // for each particle
        for (int i = 0; i < (int)m_pdata->getN(); i++)
            {
            // access the particle's position and type (MEM TRANSFER: 4 scalars)
            Scalar3 posi = make_scalar3(h_pos.data[i].x, h_pos.data[i].y, h_pos.data[i].z);
            unsigned int typei = __scalar_as_int(h_pos.data[i].w);
            const size_t head_i = h_head_list.data[i];
            // sanity check
            assert(typei < m_pdata->getNTypes());

            // initialize current force and potential energy of particle i to 0
            Scalar3 fi = make_scalar3(0.0, 0.0, 0.0);
            Scalar pei = 0.0;

            Scalar viriali_xx(0.0);
            Scalar viriali_xy(0.0);
            Scalar viriali_xz(0.0);
            Scalar viriali_yy(0.0);
            Scalar viriali_yz(0.0);
            Scalar viriali_zz(0.0);

            Scalar phi_ab[ntypes];

            // reset phi
            for (unsigned int typ_b = 0; typ_b < ntypes; ++typ_b)
                {
                phi_ab[typ_b] = Scalar(0.0);
                }

            // all neighbors of this particle
            const unsigned int size = (unsigned int)h_n_neigh.data[i];
            if (evaluator::hasPerParticleEnergy())
                {
                for (unsigned int j = 0; j < size; j++)
                    {
                    // access the index of neighbor j (MEM TRANSFER: 1 scalar)
                    unsigned int jj = h_nlist.data[head_i + j];
                    assert(jj < m_pdata->getN() + m_pdata->getNGhosts());

                    // access the position and type of particle j
                    Scalar3 posj
                        = make_scalar3(h_pos.data[jj].x, h_pos.data[jj].y, h_pos.data[jj].z);
                    unsigned int typej = __scalar_as_int(h_pos.data[jj].w);
                    assert(typej < m_pdata->getNTypes());

                    // calculate dr_ij (MEM TRANSFER: 3 scalars / FLOPS: 3)
                    Scalar3 dxij = posi - posj;

                    // apply periodic boundary conditions
                    dxij = box.minImage(dxij);

                    // compute rij_sq (FLOPS: 5)
                    Scalar rij_sq = dot(dxij, dxij);

                    // get parameters for this type pair
                    unsigned int typpair_idx = m_typpair_idx(typei, typej);
                    const param_type& param = h_params.data[typpair_idx];
                    Scalar rcutsq = h_rcutsq.data[typpair_idx];

                    // evaluate the scalar per-neighbor contribution
                    evaluator eval(rij_sq, rcutsq, param);
                    eval.evalPhi(phi_ab[typej]);
                    }

                // self-energy
                for (unsigned int typ_b = 0; typ_b < ntypes; ++typ_b)
                    {
                    unsigned int typpair_idx = m_typpair_idx(typei, typ_b);
                    const param_type& param = h_params.data[typpair_idx];
                    Scalar rcutsq = h_rcutsq.data[typpair_idx];
                    evaluator eval(Scalar(0.0), rcutsq, param);
                    Scalar energy(0.0);
                    eval.evalSelfEnergy(energy, phi_ab[typ_b]);
                    pei += energy;
                    }
                }

            // loop over all of the neighbors of this particle
            for (unsigned int j = 0; j < size; j++)
                {
                // access the index of neighbor j (MEM TRANSFER: 1 scalar)
                unsigned int jj = h_nlist.data[head_i + j];
                assert(jj < m_pdata->getN() + m_pdata->getNGhosts());

                // access the position and type of particle j
                Scalar3 posj = make_scalar3(h_pos.data[jj].x, h_pos.data[jj].y, h_pos.data[jj].z);
                unsigned int typej = __scalar_as_int(h_pos.data[jj].w);
                assert(typej < m_pdata->getNTypes());

                // initialize the current force and potential energy of particle j to 0
                Scalar3 fj = make_scalar3(0.0, 0.0, 0.0);
                Scalar pej = 0.0;

                // calculate dr_ij (MEM TRANSFER: 3 scalars / FLOPS: 3)
                Scalar3 dxij = posi - posj;

                // apply periodic boundary conditions
                dxij = box.minImage(dxij);

                // compute rij_sq (FLOPS: 5)
                Scalar rij_sq = dot(dxij, dxij);

                // get parameters for this type pair
                unsigned int typpair_idx = m_typpair_idx(typei, typej);
                const param_type& param = h_params.data[typpair_idx];
                Scalar rcutsq = h_rcutsq.data[typpair_idx];

                // evaluate the base repulsive and attractive terms
                Scalar fR = 0.0;
                Scalar fA = 0.0;
                evaluator eval(rij_sq, rcutsq, param);
                bool evaluated = eval.evalRepulsiveAndAttractive(fR, fA);

                Scalar virialj_xx(0.0);
                Scalar virialj_xy(0.0);
                Scalar virialj_xz(0.0);
                Scalar virialj_yy(0.0);
                Scalar virialj_yz(0.0);
                Scalar virialj_zz(0.0);

                if (evaluated)
                    {
                    // evaluate chi
                    Scalar chi = 0.0;
                    if (evaluator::needsChi())
                        {
                        for (unsigned int k = 0; k < size; k++)
                            {
                            // access the index of neighbor k
                            unsigned int kk = h_nlist.data[head_i + k];
                            assert(kk < m_pdata->getN() + m_pdata->getNGhosts());

                            // access the position and type of neighbor k
                            Scalar3 posk = make_scalar3(h_pos.data[kk].x,
                                                        h_pos.data[kk].y,
                                                        h_pos.data[kk].z);
                            unsigned int typek = __scalar_as_int(h_pos.data[kk].w);
                            assert(typek < m_pdata->getNTypes());

                            // access the type pair parameters for i and k
                            typpair_idx = m_typpair_idx(typei, typek);
                            const param_type& temp_param = h_params.data[typpair_idx];

                            evaluator temp_eval(rij_sq, rcutsq, temp_param);
                            bool temp_evaluated = temp_eval.areInteractive();

                            if (kk != jj && temp_evaluated)
                                {
                                // compute drik
                                Scalar3 dxik = posi - posk;

                                // apply periodic boundary conditions
                                dxik = box.minImage(dxik);

                                // compute rik_sq
                                Scalar rik_sq = dot(dxik, dxik);

                                // compute the bond angle (if needed)
                                Scalar cos_th = Scalar(0.0);
                                if (evaluator::needsAngle())
                                    cos_th = dot(dxij, dxik) / fast::sqrt(rij_sq * rik_sq);

                                // evaluate the partial chi term
                                eval.setRik(rik_sq);
                                if (evaluator::needsAngle())
                                    eval.setAngle(cos_th);

                                eval.evalChi(chi);
                                }
                            }
                        }

                    // evaluate the force and energy from the ij interaction
                    Scalar force_divr = Scalar(0.0);
                    Scalar potential_eng = Scalar(0.0);
                    Scalar bij = Scalar(0.0);
                    eval.evalForceij(fR, fA, chi, phi_ab[typej], bij, force_divr, potential_eng);

                    // add this force to particle i
                    fi += force_divr * dxij;
                    pei += potential_eng * Scalar(0.5);

                    if (compute_virial)
                        {
                        Scalar force_div2r = Scalar(0.5) * force_divr;

                        viriali_xx += force_div2r * dxij.x * dxij.x;
                        viriali_xy += force_div2r * dxij.x * dxij.y;
                        viriali_xz += force_div2r * dxij.x * dxij.z;
                        viriali_yy += force_div2r * dxij.y * dxij.y;
                        viriali_yz += force_div2r * dxij.y * dxij.z;
                        viriali_zz += force_div2r * dxij.z * dxij.z;
                        }

                    // add this force to particle j
                    fj += Scalar(-1.0) * force_divr * dxij;
                    pej += potential_eng * Scalar(0.5);

                    if (compute_virial)
                        {
                        Scalar force_div2r = Scalar(0.5) * force_divr;

                        virialj_xx += force_div2r * dxij.x * dxij.x;
                        virialj_xy += force_div2r * dxij.x * dxij.y;
                        virialj_xz += force_div2r * dxij.x * dxij.z;
                        virialj_yy += force_div2r * dxij.y * dxij.y;
                        virialj_yz += force_div2r * dxij.y * dxij.z;
                        virialj_zz += force_div2r * dxij.z * dxij.z;
                        }

                    if (evaluator::hasIkForce())
                        {
                        // evaluate the force from the ik interactions
                        for (unsigned int k = 0; k < size; k++)
                            {
                            // access the index of neighbor k
                            unsigned int kk = h_nlist.data[head_i + k];
                            assert(kk < m_pdata->getN() + m_pdata->getNGhosts());

                            // access the position and type of neighbor k
                            Scalar3 posk = make_scalar3(h_pos.data[kk].x,
                                                        h_pos.data[kk].y,
                                                        h_pos.data[kk].z);
                            unsigned int typek = __scalar_as_int(h_pos.data[kk].w);
                            assert(typek < m_pdata->getNTypes());

                            // access the type pair parameters for i and k
                            typpair_idx = m_typpair_idx(typei, typek);
                            const param_type& temp_param = h_params.data[typpair_idx];

                            evaluator temp_eval(rij_sq, rcutsq, temp_param);
                            bool temp_evaluated = temp_eval.areInteractive();

                            if (kk != jj && temp_evaluated)
                                {
                                // create variable for the force on k
                                Scalar3 fk = make_scalar3(0.0, 0.0, 0.0);

                                // compute dr_ik
                                Scalar3 dxik = posi - posk;

                                // apply periodic boundary conditions
                                dxik = box.minImage(dxik);

                                // compute rik_sq
                                Scalar rik_sq = dot(dxik, dxik);

                                // compute the bond angle (if needed)
                                Scalar cos_th = Scalar(0.0);
                                if (evaluator::needsAngle())
                                    cos_th = dot(dxij, dxik) / sqrt(rij_sq * rik_sq);

                                // set up the evaluator
                                eval.setRik(rik_sq);
                                if (evaluator::needsAngle())
                                    eval.setAngle(cos_th);

                                // compute the total force and energy
                                Scalar3 force_divr_ij = make_scalar3(0.0, 0.0, 0.0);
                                Scalar3 force_divr_ik = make_scalar3(0.0, 0.0, 0.0);
                                eval.evalForceik(fR, fA, chi, bij, force_divr_ij, force_divr_ik);

                                // add the force to particle i
                                // (FLOPS: 17)
                                fi.x += force_divr_ij.x * dxij.x + force_divr_ik.x * dxik.x;
                                fi.y += force_divr_ij.x * dxij.y + force_divr_ik.x * dxik.y;
                                fi.z += force_divr_ij.x * dxij.z + force_divr_ik.x * dxik.z;

                                // NOTE: virial for ik forces not tested
                                if (compute_virial)
                                    {
                                    Scalar force_div2r_ij = Scalar(0.5) * force_divr_ij.x;
                                    Scalar force_div2r_ik = Scalar(0.5) * force_divr_ik.x;
                                    viriali_xx += force_div2r_ij * dxij.x * dxij.x
                                                  + force_div2r_ik * dxik.x * dxik.x;
                                    viriali_xy += force_div2r_ij * dxij.x * dxij.y
                                                  + force_div2r_ik * dxik.x * dxik.y;
                                    viriali_xz += force_div2r_ij * dxij.x * dxij.z
                                                  + force_div2r_ik * dxik.x * dxik.z;
                                    viriali_yy += force_div2r_ij * dxij.y * dxij.y
                                                  + force_div2r_ik * dxik.y * dxik.y;
                                    viriali_yz += force_div2r_ij * dxij.y * dxij.z
                                                  + force_div2r_ik * dxik.y * dxik.z;
                                    viriali_zz += force_div2r_ij * dxij.z * dxij.z
                                                  + force_div2r_ik * dxik.z * dxik.z;
                                    }

                                // add the force to particle j (FLOPS: 17)
                                fj.x += force_divr_ij.y * dxij.x + force_divr_ik.y * dxik.x;
                                fj.y += force_divr_ij.y * dxij.y + force_divr_ik.y * dxik.y;
                                fj.z += force_divr_ij.y * dxij.z + force_divr_ik.y * dxik.z;

                                // NOTE: virial for ik forces not tested
                                if (compute_virial)
                                    {
                                    Scalar force_div2r_ij = Scalar(0.5) * force_divr_ij.y;
                                    Scalar force_div2r_ik = Scalar(0.5) * force_divr_ik.y;
                                    virialj_xx += force_div2r_ij * dxij.x * dxij.x
                                                  + force_div2r_ik * dxik.x * dxik.x;
                                    virialj_xy += force_div2r_ij * dxij.x * dxij.y
                                                  + force_div2r_ik * dxik.x * dxik.y;
                                    virialj_xz += force_div2r_ij * dxij.x * dxij.z
                                                  + force_div2r_ik * dxik.x * dxik.z;
                                    virialj_yy += force_div2r_ij * dxij.y * dxij.y
                                                  + force_div2r_ik * dxik.y * dxik.y;
                                    virialj_yz += force_div2r_ij * dxij.y * dxij.z
                                                  + force_div2r_ik * dxik.y * dxik.z;
                                    virialj_zz += force_div2r_ij * dxij.z * dxij.z
                                                  + force_div2r_ik * dxik.z * dxik.z;
                                    }

                                // add the force to particle k
                                fk.x += force_divr_ij.z * dxij.x + force_divr_ik.z * dxik.x;
                                fk.y += force_divr_ij.z * dxij.y + force_divr_ik.z * dxik.y;
                                fk.z += force_divr_ij.z * dxij.z + force_divr_ik.z * dxik.z;

                                // increment the force for particle k
                                unsigned int mem_idx = kk;
                                h_force.data[mem_idx].x += fk.x;
                                h_force.data[mem_idx].y += fk.y;
                                h_force.data[mem_idx].z += fk.z;

                                if (compute_virial)
                                    {
                                    Scalar force_div2r_ij = Scalar(0.5) * force_divr_ij.z;
                                    Scalar force_div2r_ik = Scalar(0.5) * force_divr_ik.z;
                                    h_virial.data[0 * m_virial_pitch + mem_idx]
                                        += force_div2r_ij * dxij.x * dxij.x
                                           + force_div2r_ik * dxik.x * dxik.x;
                                    h_virial.data[1 * m_virial_pitch + mem_idx]
                                        += force_div2r_ij * dxij.x * dxij.y
                                           + force_div2r_ik * dxik.x * dxik.y;
                                    h_virial.data[2 * m_virial_pitch + mem_idx]
                                        += force_div2r_ij * dxij.x * dxij.z
                                           + force_div2r_ik * dxik.x * dxik.z;
                                    h_virial.data[3 * m_virial_pitch + mem_idx]
                                        += force_div2r_ij * dxij.y * dxij.y
                                           + force_div2r_ik * dxik.y * dxik.y;
                                    h_virial.data[4 * m_virial_pitch + mem_idx]
                                        += force_div2r_ij * dxij.y * dxij.z
                                           + force_div2r_ik * dxik.y * dxik.z;
                                    h_virial.data[5 * m_virial_pitch + mem_idx]
                                        += force_div2r_ij * dxij.z * dxij.z
                                           + force_div2r_ik * dxik.z * dxik.z;
                                    }
                                }
                            }
                        }
                    }
                // increment the force and potential energy for particle j
                unsigned int mem_idx = jj;
                h_force.data[mem_idx].x += fj.x;
                h_force.data[mem_idx].y += fj.y;
                h_force.data[mem_idx].z += fj.z;
                h_force.data[mem_idx].w += pej;

                if (compute_virial)
                    {
                    h_virial.data[0 * m_virial_pitch + mem_idx] += virialj_xx;
                    h_virial.data[1 * m_virial_pitch + mem_idx] += virialj_xy;
                    h_virial.data[2 * m_virial_pitch + mem_idx] += virialj_xz;
                    h_virial.data[3 * m_virial_pitch + mem_idx] += virialj_yy;
                    h_virial.data[4 * m_virial_pitch + mem_idx] += virialj_yz;
                    h_virial.data[5 * m_virial_pitch + mem_idx] += virialj_zz;
                    }
                }
            // finally, increment the force and potential energy for particle i
            unsigned int mem_idx = i;
            h_force.data[mem_idx].x += fi.x;
            h_force.data[mem_idx].y += fi.y;
            h_force.data[mem_idx].z += fi.z;
            h_force.data[mem_idx].w += pei;

            if (compute_virial)
                {
                h_virial.data[0 * m_virial_pitch + mem_idx] += viriali_xx;
                h_virial.data[1 * m_virial_pitch + mem_idx] += viriali_xy;
                h_virial.data[2 * m_virial_pitch + mem_idx] += viriali_xz;
                h_virial.data[3 * m_virial_pitch + mem_idx] += viriali_yy;
                h_virial.data[4 * m_virial_pitch + mem_idx] += viriali_yz;
                h_virial.data[5 * m_virial_pitch + mem_idx] += viriali_zz;
                }
            }
        }
    }

#ifdef ENABLE_MPI
/*! \param timestep Current time step
 */
template<class evaluator>
CommFlags PotentialTersoff<evaluator>::getRequestedCommFlags(uint64_t timestep)
    {
    CommFlags flags = CommFlags(0);

    flags |= ForceCompute::getRequestedCommFlags(timestep);

    // enable reverse communication of forces
    flags[comm_flag::reverse_net_force] = 1;

    // reverse net force requires tags
    flags[comm_flag::tag] = 1;

    return flags;
    }
#endif

namespace detail
    {
//! Export this triplet potential to python
/*! \param name Name of the class in the exported python module
    \tparam T Evaluator type to export.
*/
template<class T> void export_PotentialTersoff(pybind11::module& m, const std::string& name)
    {
    pybind11::class_<PotentialTersoff<T>, ForceCompute, std::shared_ptr<PotentialTersoff<T>>>(
        m,
        name.c_str())
        .def(pybind11::init<std::shared_ptr<SystemDefinition>, std::shared_ptr<NeighborList>>())
        .def("setParams", &PotentialTersoff<T>::setParamsPython)
        .def("getParams", &PotentialTersoff<T>::getParams)
        .def("setRCut", &PotentialTersoff<T>::setRCutPython)
        .def("getRCut", &PotentialTersoff<T>::getRCut);
    }

    } // end namespace detail
    } // end namespace md
    } // end namespace hoomd

#endif
