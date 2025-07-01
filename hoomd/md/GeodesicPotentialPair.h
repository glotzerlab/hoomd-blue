// Copyright (c) 2009-2025 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

#ifndef __GEODESIC_POTENTIAL_PAIR_H__
#define __GEODESIC_POTENTIAL_PAIR_H__

#include <iostream>
#include <memory>
#include <sstream>
#include <stdexcept>

#ifdef ENABLE_HIP
#include <hip/hip_runtime.h>
#endif

#include "NeighborList.h"
#include "PotentialPair.h"

#include "hoomd/ManagedArray.h"
#include "hoomd/VectorMath.h"
#include "hoomd/md/EvaluatorPairLJ.h"

/*! \file GeodesicPotentialPair.h
    \brief Defines the template class for geodesic pair potentials
    \details The heart of the code that computes geodesic pair potentials is in this file.
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
//! Template class for computing pair potentials
/*! <b>Overview:</b>
    GeodesicPotentialPair computes standard pair potentials (and forces) between all particle pairs in
   the simulation. It employs the use of a neighbor list to limit the number of computations done to
   only those particles*/   
 with the cutoff radius of each other. The computation of the actual V(r) is
   not performed directly by this class, but by an evaluator class (e.g. EvaluatorPairLJ)
   which is passed in as a template parameter so the computations are performed as efficiently as
   possible.

    GeodesicPotentialPair handles most of the gory internal details common to all standard pair
   potentials.
     - A cutoff radius to be specified per particle type pair
     - The energy can be globally shifted to 0 at the cutoff
     - Per type pair parameters are stored and a set method is provided
     - And all the details about looping through the particles, computing dr, computing the virial,
   etc. are handled

    \note XPLOR switching is not supported

    <b>Implementation details</b>

    rcutsq and the params are stored per particle type pair. It wastes a little bit of space, but
   benchmarks show that storing the symmetric type pairs and indexing with Index2D is faster than
   not storing redundant pairs and indexing with Index2DUpperTriangular. All of these values are
   stored in GlobalArray for easy access on the GPU by a derived class. The type of the parameters
   is defined by \a param_type in the potential evaluator class passed in. See the appropriate
   documentation for the evaluator for the definition of each element of the parameters.
*/

template<class evaluator> class GeodesicPotentialPair : public PotentialPair<evaluator>
    {
    public:
    //! Param type from evaluator
    typedef typename evaluator::param_type param_type;

    //! Construct the pair potential
    GeodesicPotentialPair(std::shared_ptr<SystemDefinition> sysdef,
                       std::shared_ptr<NeighborList> nlist, 
			Scalar R);
    //! Destructor
    virtual ~GeodesicPotentialPair();

    //! Set the rcut for a single type pair
    virtual void setRcut(unsigned int typ1, unsigned int typ2, Scalar rcut);

    /// Get the r_cut for a single type pair
    Scalar getRCut(pybind11::tuple types);

    protected:
    Scalar m_R; 
    //! Actually compute the forces
    virtual void computeForces(uint64_t timestep);
    };

/*! \param sysdef System to compute forces on
    \param nlist Neighborlist to use for computing the forces
*/
template<class evaluator>
GeodesicPotentialPair<evaluator>::GeodesicPotentialPair(std::shared_ptr<SystemDefinition> sysdef,
                                                        std::shared_ptr<NeighborList> nlist,
							Scalar R)
    : PotentialPair(sysdef, nlist), m_R(R))
    {
    m_exec_conf->msg->notice(5) << "Constructing GeodesicPotentialPair<" << evaluator::getName()
                                << ">" << std::endl;
    }

template<class evaluator> GeodesicPotentialPair<evaluator>::~GeodesicPotentialPair()
    {
    m_exec_conf->msg->notice(5) << "Destroying GeodesicPotentialPair<" << evaluator::getName()
                                << ">" << std::endl;

    if (this->m_attached)
        {
        this->m_nlist->removeRCutMatrix(m_r_cut_nlist);
        }
    }

/*! \param typ1 First type index in the pair
    \param typ2 Second type index in the pair
    \param rcut Cutoff radius to set
    \note When setting the value for (\a typ1, \a typ2), the parameter for (\a typ2, \a typ1) is
   automatically set.
*/
template<class evaluator>
void GeodesicPotentialPair<evaluator>::setRcut(unsigned int typ1, unsigned int typ2, Scalar rcut)
    {
    this->validateTypes(typ1, typ2, "setting r_cut");
        {
	Scalar rcut_euclid = 2*m_R*fast::sin(rcut/(2*m_R));
        // store r_cut**2 for use internally
        ArrayHandle<Scalar> h_rcutsq(m_rcutsq, access_location::host, access_mode::readwrite);
        h_rcutsq.data[m_typpair_idx(typ1, typ2)] = rcut_euclid * rcut_euclid;
        h_rcutsq.data[m_typpair_idx(typ2, typ1)] = rcut_euclid * rcut_euclid;

        // store r_cut unmodified for so the neighbor list knows what particles to include
        ArrayHandle<Scalar> h_r_cut_nlist(*m_r_cut_nlist,
                                          access_location::host,
                                          access_mode::readwrite);
        h_r_cut_nlist.data[m_typpair_idx(typ1, typ2)] = rcut_euclid;
        h_r_cut_nlist.data[m_typpair_idx(typ2, typ1)] = rcut_euclid;
        }

    // notify the neighbor list that we have changed r_cut values
    m_nlist->notifyRCutMatrixChange();
    }

template<class evaluator>
Scalar GeodesicPotentialPair<evaluator>::getRCut(pybind11::tuple types)
    {
    auto typ1 = m_pdata->getTypeByName(types[0].cast<std::string>());
    auto typ2 = m_pdata->getTypeByName(types[1].cast<std::string>());
    validateTypes(typ1, typ2, "getting r_cut.");
    ArrayHandle<Scalar> h_rcutsq(m_rcutsq, access_location::host, access_mode::read);
    return 2*m_R*fast:arcsin(sqrt(h_rcutsq.data[m_typpair_idx(typ1, typ2)])/(2*m_R));
    }

/*! \post The pair forces are computed for the given timestep. The neighborlist's compute method is
   called to ensure that it is up to date before proceeding.

    \param timestep specifies the current time step of the simulation
*/
template<class evaluator>
void GeodesicPotentialPair<evaluator>::computeForces(uint64_t timestep)
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
    ArrayHandle<size_t> h_head_list(m_nlist->getHeadList(),
                                    access_location::host,
                                    access_mode::read);

    ArrayHandle<Scalar4> h_pos(m_pdata->getPositions(), access_location::host, access_mode::read);
    ArrayHandle<Scalar> h_charge(m_pdata->getCharges(), access_location::host, access_mode::read);
    ArrayHandle<Scalar4> h_orientation(m_pdata->getOrientationArray(),
                                       access_location::host,
                                       access_mode::read);
    ArrayHandle<unsigned int> h_tag(m_pdata->getTags(), access_location::host, access_mode::read);

    // force arrays
    ArrayHandle<Scalar4> h_force(m_force, access_location::host, access_mode::overwrite);
    ArrayHandle<Scalar4> h_torque(m_torque, access_location::host, access_mode::overwrite);
    ArrayHandle<Scalar> h_virial(m_virial, access_location::host, access_mode::overwrite);

    const BoxDim box = m_pdata->getBox();
    ArrayHandle<Scalar> h_rcutsq(m_rcutsq, access_location::host, access_mode::read);
        {
        // need to start from a zero force, energy and virial
        m_force.zeroFill();
        m_torque.zeroFill();
        m_virial.zeroFill();

        PDataFlags flags = this->m_pdata->getFlags();
        bool compute_virial = flags[pdata_flag::pressure_tensor];

        // for each particle
        for (int i = 0; i < (int)m_pdata->getN(); i++)
            {
            // access the particle's position and type (MEM TRANSFER: 4 scalars)
            Scalar3 pi = make_scalar3(h_pos.data[i].x, h_pos.data[i].y, h_pos.data[i].z);
            unsigned int typei = __scalar_as_int(h_pos.data[i].w);
            Scalar4 quat_i = h_orientation.data[i];

            // sanity check
            assert(typei < m_pdata->getNTypes());

            // access charge (if needed)
            Scalar qi = Scalar(0.0);
            if (evaluator::needsCharge())
                qi = h_charge.data[i];

            // initialize current particle force, torque, potential energy, and virial to 0
            Scalar fxi = Scalar(0.0);
            Scalar fyi = Scalar(0.0);
            Scalar fzi = Scalar(0.0);
            Scalar txi = Scalar(0.0);
            Scalar tyi = Scalar(0.0);
            Scalar tzi = Scalar(0.0);
            Scalar pei = Scalar(0.0);
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
                Scalar4 quat_j = h_orientation.data[j];

                // access the type of the neighbor particle (MEM TRANSFER: 1 scalar)
                unsigned int typej = __scalar_as_int(h_pos.data[j].w);
                assert(typej < m_pdata->getNTypes());

                // access charge (if needed)
                Scalar qj = Scalar(0.0);
                if (evaluator::needsCharge())
                    qj = h_charge.data[j];

                // apply periodic boundary conditions
                dx = box.minImage(dx);

                // get parameters for this type pair
                unsigned int typpair_idx = m_typpair_idx(typei, typej);
                const param_type& param = m_params[typpair_idx];
                Scalar rcutsq = h_rcutsq.data[typpair_idx];

                // design specifies that energies are shifted if
                // shift mode is set to shift
                bool energy_shift = false;
                if (m_shift_mode == shift)
                    energy_shift = true;

                // compute the force and potential energy
                Scalar3 force = make_scalar3(0.0, 0.0, 0.0);
                Scalar3 torque_i = make_scalar3(0.0, 0.0, 0.0);
                Scalar3 torque_j = make_scalar3(0.0, 0.0, 0.0);

                Scalar pair_eng = Scalar(0.0);

                evaluator eval(dx, quat_i, quat_j, rcutsq, param);

                if (evaluator::needsCharge())
                    eval.setCharge(qi, qj);
                if (evaluator::needsShape())
                    eval.setShape(&m_shape_params[typei], &m_shape_params[typej]);
                if (evaluator::needsTags())
                    eval.setTags(h_tag.data[i], h_tag.data[j]);

                bool evaluated = eval.evaluate(force, pair_eng, energy_shift, torque_i, torque_j);

                if (evaluated)
                    {
                    Scalar3 force2 = Scalar(0.5) * force;

                    // add the force, potential energy and virial to the particle i
                    // (FLOPS: 8)
                    fxi += force.x;
                    fyi += force.y;
                    fzi += force.z;
                    txi += torque_i.x;
                    tyi += torque_i.y;
                    tzi += torque_i.z;
                    pei += pair_eng * Scalar(0.5);

                    if (compute_virial)
                        {
                        virialxxi += dx.x * force2.x;
                        virialxyi += dx.y * force2.x;
                        virialxzi += dx.z * force2.x;
                        virialyyi += dx.y * force2.y;
                        virialyzi += dx.z * force2.y;
                        virialzzi += dx.z * force2.z;
                        }

                    // add the force to particle j if we are using the third law (MEM TRANSFER: 10
                    // scalars / FLOPS: 8)
                    if (third_law && j < m_pdata->getN())
                        {
                        h_force.data[j].x -= force.x;
                        h_force.data[j].y -= force.y;
                        h_force.data[j].z -= force.z;
                        h_torque.data[j].x += torque_j.x;
                        h_torque.data[j].y += torque_j.y;
                        h_torque.data[j].z += torque_j.z;
                        h_force.data[j].w += pair_eng * Scalar(0.5);
                        if (compute_virial)
                            {
                            h_virial.data[0 * m_virial_pitch + j] += dx.x * force2.x;
                            h_virial.data[1 * m_virial_pitch + j] += dx.y * force2.x;
                            h_virial.data[2 * m_virial_pitch + j] += dx.z * force2.x;
                            h_virial.data[3 * m_virial_pitch + j] += dx.y * force2.y;
                            h_virial.data[4 * m_virial_pitch + j] += dx.z * force2.y;
                            h_virial.data[5 * m_virial_pitch + j] += dx.z * force2.z;
                            }
                        }
                    }
                }

            // finally, increment the force, potential energy and virial for particle i
            h_force.data[i].x += fxi;
            h_force.data[i].y += fyi;
            h_force.data[i].z += fzi;
            h_torque.data[i].x += txi;
            h_torque.data[i].y += tyi;
            h_torque.data[i].z += tzi;
            h_force.data[i].w += pei;
            if (compute_virial)
                {
                h_virial.data[0 * m_virial_pitch + i] += virialxxi;
                h_virial.data[1 * m_virial_pitch + i] += virialxyi;
                h_virial.data[2 * m_virial_pitch + i] += virialxzi;
                h_virial.data[3 * m_virial_pitch + i] += virialyyi;
                h_virial.data[4 * m_virial_pitch + i] += virialyzi;
                h_virial.data[5 * m_virial_pitch + i] += virialzzi;
                }
            }
        }
    }


namespace detail
    {
//! Export this pair potential to python
/*! \param name Name of the class in the exported python module
    \tparam T Evaluator type to export.
*/
template<class T> void export_GeodesicPotentialPair(pybind11::module& m, const std::string& name)
    {
    pybind11::class_<GeodesicPotentialPair<T>, PotentialPair<T>, std::shared_ptr<GeodesicPotentialPair<T>>>
        geodesicpotentialpair(m, name.c_str());
    geodesicpotentialpair
        .def(pybind11::init<std::shared_ptr<SystemDefinition>, std::shared_ptr<NeighborList>,Scalar>())
        .def("setRCut", &GeodesicPotentialPair<T>::setRCutPython)
        .def("getRCut", &GeodesicPotentialPair<T>::getRCut)
    }

    } // end namespace detail
    } // end namespace md
    } // end namespace hoomd

#endif // __GEODESIC_POTENTIAL_PAIR_H__
