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
   only those particles  with the cutoff radius of each other. The computation of the actual V(r) is
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

    void setRadius(Scalar R)
	{m_R = R;}

    Scalar getRadius()
	{return m_R;}

    //! Set the rcut for a single type pair
    virtual void setRcut(unsigned int typ1, unsigned int typ2, Scalar rcut);

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
    : PotentialPair<evaluator>(sysdef, nlist), m_R(R)
    {
    this->m_exec_conf->msg->notice(5) << "Constructing GeodesicPotentialPair<" << evaluator::getName()
                                << ">" << std::endl;
    }

template<class evaluator> GeodesicPotentialPair<evaluator>::~GeodesicPotentialPair()
    {
    this->m_exec_conf->msg->notice(5) << "Destroying GeodesicPotentialPair<" << evaluator::getName()
                                << ">" << std::endl;

    if (this->m_attached)
        {
        this->m_nlist->removeRCutMatrix(this->m_r_cut_nlist);
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
        ArrayHandle<Scalar> h_rcutsq(this->m_rcutsq, access_location::host, access_mode::readwrite);
        h_rcutsq.data[this->m_typpair_idx(typ1, typ2)] = rcut * rcut;
        h_rcutsq.data[this->m_typpair_idx(typ2, typ1)] = rcut * rcut;

        // store r_cut unmodified for so the neighbor list knows what particles to include
        ArrayHandle<Scalar> h_r_cut_nlist(*this->m_r_cut_nlist,
                                          access_location::host,
                                          access_mode::readwrite);
        h_r_cut_nlist.data[this->m_typpair_idx(typ1, typ2)] = rcut_euclid;
        h_r_cut_nlist.data[this->m_typpair_idx(typ2, typ1)] = rcut_euclid;
        }

    // notify the neighbor list that we have changed r_cut values
    this->m_nlist->notifyRCutMatrixChange();
    }

/*! \post The pair forces are computed for the given timestep. The neighborlist's compute method is
   called to ensure that it is up to date before proceeding.

    \param timestep specifies the current time step of the simulation
*/
template<class evaluator>
void GeodesicPotentialPair<evaluator>::computeForces(uint64_t timestep)
    {
    // start by updating the neighborlist
    this->m_nlist->compute(timestep);

    // depending on the neighborlist settings, we can take advantage of newton's third law
    // to reduce computations at the cost of memory access complexity: set that flag now
    bool third_law = this->m_nlist->getStorageMode() == NeighborList::half;

    // access the neighbor list, particle data, and system box
    ArrayHandle<unsigned int> h_n_neigh(this->m_nlist->getNNeighArray(),
                                        access_location::host,
                                        access_mode::read);
    ArrayHandle<unsigned int> h_nlist(this->m_nlist->getNListArray(),
                                      access_location::host,
                                      access_mode::read);
    ArrayHandle<size_t> h_head_list(this->m_nlist->getHeadList(),
                                    access_location::host,
                                    access_mode::read);

    ArrayHandle<Scalar4> h_pos(this->m_pdata->getPositions(), access_location::host, access_mode::read);
    ArrayHandle<Scalar> h_charge(this->m_pdata->getCharges(), access_location::host, access_mode::read);

    // force arrays
    ArrayHandle<Scalar4> h_force(this->m_force, access_location::host, access_mode::overwrite);
    ArrayHandle<Scalar> h_virial(this->m_virial, access_location::host, access_mode::overwrite);

    ArrayHandle<Scalar> h_ronsq(this->m_ronsq, access_location::host, access_mode::read);
    ArrayHandle<Scalar> h_rcutsq(this->m_rcutsq, access_location::host, access_mode::read);
        {
        // need to start from a zero force, energy and virial
        this->m_force.zeroFill();
        this->m_virial.zeroFill();

        PDataFlags flags = this->m_pdata->getFlags();
        bool compute_virial = flags[pdata_flag::pressure_tensor];

        // for each particle
        for (int i = 0; i < (int)this->m_pdata->getN(); i++)
            {
            // access the particle's position and type (MEM TRANSFER: 4 scalars)
            Scalar3 pi = make_scalar3(h_pos.data[i].x, h_pos.data[i].y, h_pos.data[i].z);
            unsigned int typei = __scalar_as_int(h_pos.data[i].w);

	    Scalar normi = fast::rsqrt(pi.x*pi.x+pi.y*pi.y+pi.z*pi.z);
            Scalar3 normali = pi*normi;

            // sanity check
            assert(typei < this->m_pdata->getNTypes());

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
                assert(j < this->m_pdata->getN() + this->m_pdata->getNGhosts());

                // calculate dr_ji (MEM TRANSFER: 3 scalars / FLOPS: 3)
                Scalar3 pj = make_scalar3(h_pos.data[j].x, h_pos.data[j].y, h_pos.data[j].z);

	    	Scalar normj = fast::rsqrt(pj.x*pj.x+pj.y*pj.y+pj.z*pj.z);
            	Scalar3 normalj = pj*normj;

                Scalar3 dx = pi - pj;

                // access the type of the neighbor particle (MEM TRANSFER: 1 scalar)
                unsigned int typej = __scalar_as_int(h_pos.data[j].w);
                assert(typej < this->m_pdata->getNTypes());

                // access charge (if needed)
                Scalar qj = Scalar(0.0);
                if (evaluator::needsCharge())
                    qj = h_charge.data[j];

                // calculate r_ij squared (FLOPS: 5)
                Scalar rsq = dot(dx, dx);
		
		Scalar r_geodesic = 2*m_R*asin(sqrt(rsq)/(2*m_R));

		Scalar rsq_geodesic = r_geodesic*r_geodesic;

                // get parameters for this type pair
                unsigned int typpair_idx = this->m_typpair_idx(typei, typej);
                const param_type& param = this->m_params[typpair_idx];
                Scalar rcutsq = h_rcutsq.data[typpair_idx];
                Scalar ronsq = Scalar(0.0);
                if (this->m_shift_mode == this->xplor)
                    ronsq = h_ronsq.data[typpair_idx];

                // design specifies that energies are shifted if
                // 1) shift mode is set to shift
                // or 2) shift mode is explor and ron > rcut
                bool energy_shift = false;
                if (this->m_shift_mode == this->shift)
                    energy_shift = true;
                else if (this->m_shift_mode == this->xplor)
                    {
                    if (ronsq > rcutsq)
                        energy_shift = true;
                    }

                // compute the force and potential energy
                Scalar force_divr = Scalar(0.0);
                Scalar pair_eng = Scalar(0.0);
                evaluator eval(rsq_geodesic, rcutsq, param);
                if (evaluator::needsCharge())
                    eval.setCharge(qi, qj);

                bool evaluated = eval.evalForceAndEnergy(force_divr, pair_eng, energy_shift);

                if (evaluated)
                    {
                    // modify the potential for xplor shifting
                    if (this->m_shift_mode == this->xplor)
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
                            Scalar ds_dr_divr = Scalar(12.0) * (rsq - ronsq) * rsq_minus_r_cut_sq
                                                * xplor_denom_inv;

                            // make modifications to the old pair energy and force
                            pair_eng = old_pair_eng * s;
                            // note: I'm not sure why the minus sign needs to be there: my notes
                            // have a
                            // + But this is verified correct via plotting
                            force_divr = s * old_force_divr - ds_dr_divr * old_pair_eng;
                            }
                        }

                    force_divr = force_divr * r_geodesic;
                    Scalar force_div2r = force_divr * Scalar(0.5);
                    // add the force, potential energy and virial to the particle i
                    // (FLOPS: 8)

		    Scalar3 dxi = dx - (dx.x*normali.x+dx.y*normali.y+ dx.z*normali.z)*normali;
		    Scalar normdxi = fast::rsqrt(dxi.x*dxi.x+dxi.y*dxi.y+dxi.z*dxi.z);

		    dxi = dxi*normdxi;
		    
                    fi += dxi * force_divr;
                    pei += pair_eng * Scalar(0.5);
                    if (compute_virial)
                        {
			//maybe wrong
                        virialxxi += force_div2r * dxi.x * pi.x;
                        virialxyi += force_div2r * dxi.x * pi.y;
                        virialxzi += force_div2r * dxi.x * pi.z;
                        virialyyi += force_div2r * dxi.y * pi.y;
                        virialyzi += force_div2r * dxi.y * pi.z;
                        virialzzi += force_div2r * dxi.z * pi.z;
                        }

                    // add the force to particle j if we are using the third law (MEM TRANSFER: 10
                    // scalars / FLOPS: 8) only add force to local particles
                    if (third_law && j < this->m_pdata->getN())
                        {
                        unsigned int mem_idx = j;

			Scalar3 dxj = -dx + (dx.x*normalj.x+dx.y*normalj.y+ dx.z*normalj.z)*normalj;
			Scalar normdxj = fast::rsqrt(dxj.x*dxj.x+dxj.y*dxj.y+dxj.z*dxj.z);

			dxj = dxj*normdxj;
                        h_force.data[mem_idx].x += dxj.x * force_divr;
                        h_force.data[mem_idx].y += dxj.y * force_divr;
                        h_force.data[mem_idx].z += dxj.z * force_divr;
                        h_force.data[mem_idx].w += pair_eng * Scalar(0.5);
                        if (compute_virial)
                            {
                            h_virial.data[0 * this->m_virial_pitch + mem_idx]
                                += force_div2r * dxj.x * pj.x;
                            h_virial.data[1 * this->m_virial_pitch + mem_idx]
                                += force_div2r * dxj.x * pj.y;
                            h_virial.data[2 * this->m_virial_pitch + mem_idx]
                                += force_div2r * dxj.x * pj.z;
                            h_virial.data[3 * this->m_virial_pitch + mem_idx]
                                += force_div2r * dxj.y * pj.y;
                            h_virial.data[4 * this->m_virial_pitch + mem_idx]
                                += force_div2r * dxj.y * pj.z;
                            h_virial.data[5 * this->m_virial_pitch + mem_idx]
                                += force_div2r * dxj.z * pj.z;
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
                h_virial.data[0 * this->m_virial_pitch + mem_idx] += virialxxi;
                h_virial.data[1 * this->m_virial_pitch + mem_idx] += virialxyi;
                h_virial.data[2 * this->m_virial_pitch + mem_idx] += virialxzi;
                h_virial.data[3 * this->m_virial_pitch + mem_idx] += virialyyi;
                h_virial.data[4 * this->m_virial_pitch + mem_idx] += virialyzi;
                h_virial.data[5 * this->m_virial_pitch + mem_idx] += virialzzi;
                }
            }
        }

    this->computeTailCorrection();
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
	.def_property("R", &GeodesicPotentialPair<T>::getRadius, &GeodesicPotentialPair<T>::setRadius);
    }

    } // end namespace detail
    } // end namespace md
    } // end namespace hoomd

#endif // __GEODESIC_POTENTIAL_PAIR_H__
