// Copyright (c) 2009-2019 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.


// Maintainer: phillicl

#ifndef __POTENTIAL_PAIR_DPDTHERMO_H__
#define __POTENTIAL_PAIR_DPDTHERMO_H__

#include "PotentialPair.h"
#include "hoomd/Variant.h"


/*! \file PotentialPairDPDThermo.h
    \brief Defines the template class for a dpd thermostat and LJ pair potential
    \note This header cannot be compiled by nvcc
*/

#ifdef NVCC
#error This header cannot be compiled by nvcc
#endif

//! Template class for computing dpd thermostat and LJ pair potential
/*! <b>Overview:</b>
    TODO - Revise Documentation Below

    PotentialPairDPDThermo computes a dpd thermostat and Lennard Jones pair potentials (and forces) between all particle pairs in the simulation. It
    employs the use of a neighbor list to limit the number of computations done to only those particles with the
    cutoff radius of each other. The computation of the actual V(r) is not performed directly by this class, but
    by an evaluator class (e.g. EvaluatorPairDPDLJThermo) which is passed in as a template parameter so the computations
    are performed as efficiently as possible.

    PotentialPairDPDThermo handles most of the gory internal details common to all standard pair potentials.
     - A cutoff radius to be specified per particle type pair for the conservative and stochastic potential
     - A RNG seed is stored.
     - Per type pair parameters are stored and a set method is provided
     - Logging methods are provided for the energy
     - And all the details about looping through the particles, computing dr, computing the virial, etc. are handled

    \sa export_PotentialPairDPDThermo()
*/
template < class evaluator >
class PotentialPairDPDThermo : public PotentialPair<evaluator>
    {
    public:
        //! Param type from evaluator
        typedef typename evaluator::param_type param_type;

        //! Construct the pair potential
        PotentialPairDPDThermo(std::shared_ptr<SystemDefinition> sysdef,
                      std::shared_ptr<NeighborList> nlist,
                      const std::string& log_suffix="");
        //! Destructor
        virtual ~PotentialPairDPDThermo() { };


        //! Set the seed
        virtual void setSeed(unsigned int seed);

        //! Set the temperature
        virtual void setT(std::shared_ptr<Variant> T);

        #ifdef ENABLE_MPI
        //! Get ghost particle fields requested by this pair potential
        virtual CommFlags getRequestedCommFlags(unsigned int timestep);
        #endif

    protected:

        unsigned int m_seed;  //!< seed for PRNG for DPD thermostat
        std::shared_ptr<Variant> m_T;     //!< Temperature for the DPD thermostat

        //! Actually compute the forces (overwrites PotentialPair::computeForces())
        virtual void computeForces(unsigned int timestep);
    };

/*! \param sysdef System to compute forces on
    \param nlist Neighborlist to use for computing the forces
    \param log_suffix Name given to this instance of the force
*/
template < class evaluator >
PotentialPairDPDThermo< evaluator >::PotentialPairDPDThermo(std::shared_ptr<SystemDefinition> sysdef,
                                                std::shared_ptr<NeighborList> nlist,
                                                const std::string& log_suffix)
    : PotentialPair<evaluator>(sysdef,nlist, log_suffix)
    {
    }

/*! \param seed Stored seed for PRNG
 \note All ranks other than 0 ignore the seed input and use the value of ranke 0.
*/
template< class evaluator >
void PotentialPairDPDThermo< evaluator >::setSeed(unsigned int seed)
    {
    m_seed = seed;
    // In case of MPI run, every rank should be initialized with the same seed.
    // For simplicity we broadcast the seed of rank 0 to all ranks.
#ifdef ENABLE_MPI
    if( this->m_pdata->getDomainDecomposition() )
        bcast(m_seed,0,this->m_exec_conf->getMPICommunicator());
#endif//ENABLE_MPI

    // Hash the User's Seed to make it less likely to be a low positive integer
    m_seed = m_seed*0x12345677 + 0x12345 ; m_seed^=(m_seed>>16); m_seed*= 0x45679;
    }

/*! \param T the temperature the system is thermostated on this time step.
*/
template< class evaluator >
void PotentialPairDPDThermo< evaluator >::setT(std::shared_ptr<Variant> T)
    {
    m_T = T;
    }

/*! \post The pair forces are computed for the given timestep. The neighborlist's compute method is called to ensure
    that it is up to date before proceeding.

    \param timestep specifies the current time step of the simulation
*/
template< class evaluator >
void PotentialPairDPDThermo< evaluator >::computeForces(unsigned int timestep)
    {
    // start by updating the neighborlist
    this->m_nlist->compute(timestep);

    // start the profile for this compute
    if (this->m_prof) this->m_prof->push(this->m_prof_name);

    // depending on the neighborlist settings, we can take advantage of newton's third law
    // to reduce computations at the cost of memory access complexity: set that flag now
    bool third_law = this->m_nlist->getStorageMode() == NeighborList::half;

    // access the neighbor list, particle data, and system box
    ArrayHandle<unsigned int> h_n_neigh(this->m_nlist->getNNeighArray(), access_location::host, access_mode::read);
    ArrayHandle<unsigned int> h_nlist(this->m_nlist->getNListArray(), access_location::host, access_mode::read);
    ArrayHandle<unsigned int> h_head_list(this->m_nlist->getHeadList(), access_location::host, access_mode::read);

    ArrayHandle<Scalar4> h_pos(this->m_pdata->getPositions(), access_location::host, access_mode::read);
    ArrayHandle<Scalar4> h_vel(this->m_pdata->getVelocities(), access_location::host, access_mode::read);
    ArrayHandle<unsigned int> h_tag(this->m_pdata->getTags(), access_location::host, access_mode::read);

    //force arrays
    ArrayHandle<Scalar4> h_force(this->m_force,access_location::host, access_mode::overwrite);
    ArrayHandle<Scalar>  h_virial(this->m_virial,access_location::host, access_mode::overwrite);

    const BoxDim& box = this->m_pdata->getBox();
    ArrayHandle<Scalar> h_ronsq(this->m_ronsq, access_location::host, access_mode::read);
    ArrayHandle<Scalar> h_rcutsq(this->m_rcutsq, access_location::host, access_mode::read);
    ArrayHandle<param_type> h_params(this->m_params, access_location::host, access_mode::read);

    // need to start from a zero force, energy and virial
    memset((void*)h_force.data,0,sizeof(Scalar4)*this->m_force.getNumElements());
    memset((void*)h_virial.data,0,sizeof(Scalar)*this->m_virial.getNumElements());

    // for each particle
    for (int i = 0; i < (int)this->m_pdata->getN(); i++)
        {
        // access the particle's position, velocity, and type (MEM TRANSFER: 7 scalars)
        Scalar3 pi = make_scalar3(h_pos.data[i].x, h_pos.data[i].y, h_pos.data[i].z);
        Scalar3 vi = make_scalar3(h_vel.data[i].x, h_vel.data[i].y, h_vel.data[i].z);

        unsigned int typei = __scalar_as_int(h_pos.data[i].w);
        const unsigned int head_i = h_head_list.data[i];

        // sanity check
        assert(typei < this->m_pdata->getNTypes());

        // initialize current particle force, potential energy, and virial to 0
        Scalar3 fi = make_scalar3(0,0,0);
        Scalar pei = 0.0;
        Scalar viriali[6];
        for (unsigned int l = 0; l < 6; l++)
            viriali[l] = 0.0;

        // loop over all of the neighbors of this particle
        const unsigned int size = (unsigned int)h_n_neigh.data[i];
        for (unsigned int k = 0; k < size; k++)
            {
            // access the index of this neighbor (MEM TRANSFER: 1 scalar)
            unsigned int j = h_nlist.data[head_i + k];
            assert(j < this->m_pdata->getN() + this->m_pdata->getNGhosts() );

            // calculate dr_ji (MEM TRANSFER: 3 scalars / FLOPS: 3)
            Scalar3 pj = make_scalar3(h_pos.data[j].x, h_pos.data[j].y, h_pos.data[j].z);
            Scalar3 dx = pi - pj;

            // calculate dv_ji (MEM TRANSFER: 3 scalars / FLOPS: 3)
            Scalar3 vj = make_scalar3(h_vel.data[j].x, h_vel.data[j].y, h_vel.data[j].z);
            Scalar3 dv = vi - vj;

            // access the type of the neighbor particle (MEM TRANSFER: 1 scalar)
            unsigned int typej = __scalar_as_int(h_pos.data[j].w);
            assert(typej < this->m_pdata->getNTypes());

            // apply periodic boundary conditions
            dx = box.minImage(dx);

            // calculate r_ij squared (FLOPS: 5)
            Scalar rsq = dot(dx, dx);

            //calculate the drag term r \dot v
            Scalar rdotv = dot(dx, dv);

            // get parameters for this type pair
            unsigned int typpair_idx = this->m_typpair_idx(typei, typej);
            param_type param = h_params.data[typpair_idx];
            Scalar rcutsq = h_rcutsq.data[typpair_idx];

            // design specifies that energies are shifted if
            // 1) shift mode is set to shift
            bool energy_shift = false;
            if (this->m_shift_mode == this->shift)
                energy_shift = true;

            // compute the force and potential energy
            Scalar force_divr = Scalar(0.0);
            Scalar force_divr_cons = Scalar(0.0);
            Scalar pair_eng = Scalar(0.0);
            evaluator eval(rsq, rcutsq, param);

            // Special Potential Pair DPD Requirements
            const Scalar currentTemp = m_T->getValue(timestep);

            // set seed using global tags
            unsigned int tagi = h_tag.data[i];
            unsigned int tagj = h_tag.data[j];
            eval.set_seed_ij_timestep(m_seed,tagi,tagj,timestep);
            eval.setDeltaT(this->m_deltaT);
            eval.setRDotV(rdotv);
            eval.setT(currentTemp);

            bool evaluated = eval.evalForceEnergyThermo(force_divr, force_divr_cons, pair_eng, energy_shift);

            if (evaluated)
                {
                // compute the virial (FLOPS: 2)
                Scalar pair_virial[6];
                pair_virial[0] = Scalar(0.5) * dx.x * dx.x * force_divr_cons;
                pair_virial[1] = Scalar(0.5) * dx.x * dx.y * force_divr_cons;
                pair_virial[2] = Scalar(0.5) * dx.x * dx.z * force_divr_cons;
                pair_virial[3] = Scalar(0.5) * dx.y * dx.y * force_divr_cons;
                pair_virial[4] = Scalar(0.5) * dx.y * dx.z * force_divr_cons;
                pair_virial[5] = Scalar(0.5) * dx.z * dx.z * force_divr_cons;


                // add the force, potential energy and virial to the particle i
                // (FLOPS: 8)
                fi += dx*force_divr;
                pei += pair_eng * Scalar(0.5);
                for (unsigned int l = 0; l < 6; l++)
                    viriali[l] += pair_virial[l];

                // add the force to particle j if we are using the third law (MEM TRANSFER: 10 scalars / FLOPS: 8)
                if (third_law)
                    {
                    unsigned int mem_idx = j;
                    h_force.data[mem_idx].x -= dx.x*force_divr;
                    h_force.data[mem_idx].y -= dx.y*force_divr;
                    h_force.data[mem_idx].z -= dx.z*force_divr;
                    h_force.data[mem_idx].w += pair_eng * Scalar(0.5);
                    for (unsigned int l = 0; l < 6; l++)
                        h_virial.data[l * this->m_virial_pitch + mem_idx] += pair_virial[l];
                    }
                }
            }

        // finally, increment the force, potential energy and virial for particle i
        unsigned int mem_idx = i;
        h_force.data[mem_idx].x += fi.x;
        h_force.data[mem_idx].y += fi.y;
        h_force.data[mem_idx].z += fi.z;
        h_force.data[mem_idx].w += pei;
        for (unsigned int l = 0; l < 6; l++)
            h_virial.data[l * this->m_virial_pitch + mem_idx] += viriali[l];
        }

    if (this->m_prof) this->m_prof->pop();
    }

#ifdef ENABLE_MPI
/*! \param timestep Current time step
 */
template < class evaluator >
CommFlags PotentialPairDPDThermo< evaluator >::getRequestedCommFlags(unsigned int timestep)
    {
    CommFlags flags = CommFlags(0);

    // DPD needs ghost particle velocity
    flags[comm_flag::velocity] = 1;
    // DPD needs tags for RNG
    flags[comm_flag::tag] = 1;

    flags |= PotentialPair<evaluator>::getRequestedCommFlags(timestep);

    return flags;
    }
#endif


//! Export this pair potential to python
/*! \param name Name of the class in the exported python module
    \tparam T Class type to export. \b Must be an instantiated PotentialPairDPDThermo class template.
    \tparam Base Base class of \a T. \b Must be PotentialPair<evaluator> with the same evaluator as used in \a T.
*/
template < class T, class Base > void export_PotentialPairDPDThermo(pybind11::module& m, const std::string& name)
    {
    pybind11::class_<T, std::shared_ptr<T> >(m, name.c_str(), pybind11::base< Base >())
        .def(pybind11::init< std::shared_ptr<SystemDefinition>, std::shared_ptr<NeighborList>, const std::string& >())
        .def("setSeed", &T::setSeed)
        .def("setT", &T::setT)
              ;
    }


#endif // __POTENTIAL_PAIR_DPDTHERMO_H__
