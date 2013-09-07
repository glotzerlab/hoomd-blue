/*
Highly Optimized Object-oriented Many-particle Dynamics -- Blue Edition
(HOOMD-blue) Open Source Software License Copyright 2008-2011 Ames Laboratory
Iowa State University and The Regents of the University of Michigan All rights
reserved.

HOOMD-blue may contain modifications ("Contributions") provided, and to which
copyright is held, by various Contributors who have granted The Regents of the
University of Michigan the right to modify and/or distribute such Contributions.

You may redistribute, use, and create derivate works of HOOMD-blue, in source
and binary forms, provided you abide by the following conditions:

* Redistributions of source code must retain the above copyright notice, this
list of conditions, and the following disclaimer both in the code and
prominently in any materials provided with the distribution.

* Redistributions in binary form must reproduce the above copyright notice, this
list of conditions, and the following disclaimer in the documentation and/or
other materials provided with the distribution.

* All publications and presentations based on HOOMD-blue, including any reports
or published results obtained, in whole or in part, with HOOMD-blue, will
acknowledge its use according to the terms posted at the time of submission on:
http://codeblue.umich.edu/hoomd-blue/citations.html

* Any electronic documents citing HOOMD-Blue will link to the HOOMD-Blue website:
http://codeblue.umich.edu/hoomd-blue/

* Apart from the above required attributions, neither the name of the copyright
holder nor the names of HOOMD-blue's contributors may be used to endorse or
promote products derived from this software without specific prior written
permission.

Disclaimer

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDER AND CONTRIBUTORS ``AS IS'' AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, AND/OR ANY
WARRANTIES THAT THIS SOFTWARE IS FREE OF INFRINGEMENT ARE DISCLAIMED.

IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT,
INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE
OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF
ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

// Maintainer: phillicl

#ifndef __POTENTIAL_PAIR_DPDTHERMO_H__
#define __POTENTIAL_PAIR_DPDTHERMO_H__

#include "PotentialPair.h"
#include "Variant.h"

#ifdef ENABLE_OPENMP
#include <omp.h>
#endif

#ifdef WIN32
#pragma warning( push )
#pragma warning( disable : 4103 4244 )
#endif

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
    cuttoff radius of each other. The computation of the actual V(r) is not performed directly by this class, but
    by an evaluator class (e.g. EvaluatorPairDPDLJThermo) which is passed in as a template parameter so the compuations
    are performed as efficiently as possible.

    PotentialPairDPDThermo handles most of the gory internal details common to all standard pair potentials.
     - A cuttoff radius to be specified per particle type pair for the conservative and stochastic potential
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
        PotentialPairDPDThermo(boost::shared_ptr<SystemDefinition> sysdef,
                      boost::shared_ptr<NeighborList> nlist,
                      const std::string& log_suffix="");
        //! Destructor
        virtual ~PotentialPairDPDThermo() { };


        //! Set the seed
        virtual void setSeed(unsigned int seed);

        //! Set the temperature
        virtual void setT(boost::shared_ptr<Variant> T);

        #ifdef ENABLE_MPI
        //! Get ghost particle fields requested by this pair potential
        virtual CommFlags getRequestedCommFlags(unsigned int timestep);
        #endif

    protected:

        unsigned int m_seed;  //!< seed for PRNG for DPD thermostat
        boost::shared_ptr<Variant> m_T;     //!< Temperature for the DPD thermostat

        //! Actually compute the forces (overwrites PotentialPair::computeForces())
        virtual void computeForces(unsigned int timestep);
    };

/*! \param sysdef System to compute forces on
    \param nlist Neighborlist to use for computing the forces
    \param log_suffix Name given to this instance of the force
*/
template < class evaluator >
PotentialPairDPDThermo< evaluator >::PotentialPairDPDThermo(boost::shared_ptr<SystemDefinition> sysdef,
                                                boost::shared_ptr<NeighborList> nlist,
                                                const std::string& log_suffix)
    : PotentialPair<evaluator>(sysdef,nlist, log_suffix)
    {
    }

/*! \param seed Stored seed for PRNG
*/
template< class evaluator >
void PotentialPairDPDThermo< evaluator >::setSeed(unsigned int seed)
    {
    m_seed = seed;

    // Hash the User's Seed to make it less likely to be a low positive integer
    m_seed = m_seed*0x12345677 + 0x12345 ; m_seed^=(m_seed>>16); m_seed*= 0x45679;

    }

/*! \param T the temperature the system is thermostated on this time step.
*/
template< class evaluator >
void PotentialPairDPDThermo< evaluator >::setT(boost::shared_ptr<Variant> T)
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
    Index2D nli = this->m_nlist->getNListIndexer();

    ArrayHandle<Scalar4> h_pos(this->m_pdata->getPositions(), access_location::host, access_mode::read);
    ArrayHandle<Scalar4> h_vel(this->m_pdata->getVelocities(), access_location::host, access_mode::read);

    //force arrays
    ArrayHandle<Scalar4> h_force(this->m_force,access_location::host, access_mode::overwrite);
    ArrayHandle<Scalar>  h_virial(this->m_virial,access_location::host, access_mode::overwrite);

    const BoxDim& box = this->m_pdata->getBox();
    ArrayHandle<Scalar> h_ronsq(this->m_ronsq, access_location::host, access_mode::read);
    ArrayHandle<Scalar> h_rcutsq(this->m_rcutsq, access_location::host, access_mode::read);
    ArrayHandle<param_type> h_params(this->m_params, access_location::host, access_mode::read);

#pragma omp parallel
    {
    #ifdef ENABLE_OPENMP
    int tid = omp_get_thread_num();
    #else
    int tid = 0;
    #endif

    // need to start from a zero force, energy and virial
    memset(&(this->m_fdata_partial[this->m_index_thread_partial(0,tid)]) , 0, sizeof(Scalar4)*this->m_pdata->getN());
    memset(&(this->m_virial_partial[6*this->m_index_thread_partial(0,tid)]) , 0, 6*sizeof(Scalar)*this->m_pdata->getN());

    // for each particle
#pragma omp for schedule(guided)
    for (int i = 0; i < (int)this->m_pdata->getN(); i++)
        {
        // access the particle's position, velocity, and type (MEM TRANSFER: 7 scalars)
        Scalar3 pi = make_scalar3(h_pos.data[i].x, h_pos.data[i].y, h_pos.data[i].z);
        Scalar3 vi = make_scalar3(h_vel.data[i].x, h_vel.data[i].y, h_vel.data[i].z);

        unsigned int typei = __scalar_as_int(h_pos.data[i].w);

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
            unsigned int j = h_nlist.data[nli(i, k)];
            assert(j < this->m_pdata->getN());

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
            eval.set_seed_ij_timestep(m_seed,i,j,timestep);
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
                    unsigned int mem_idx = this->m_index_thread_partial(j,tid);
                    this->m_fdata_partial[mem_idx].x -= dx.x*force_divr;
                    this->m_fdata_partial[mem_idx].y -= dx.y*force_divr;
                    this->m_fdata_partial[mem_idx].z -= dx.z*force_divr;
                    this->m_fdata_partial[mem_idx].w += pair_eng * Scalar(0.5);
                    for (unsigned int l = 0; l < 6; l++)
                        this->m_virial_partial[l+6*mem_idx] += pair_virial[l];

                    }
                }
            }

        // finally, increment the force, potential energy and virial for particle i
        unsigned int mem_idx = this->m_index_thread_partial(i,tid);
        this->m_fdata_partial[mem_idx].x += fi.x;
        this->m_fdata_partial[mem_idx].y += fi.y;
        this->m_fdata_partial[mem_idx].z += fi.z;
        this->m_fdata_partial[mem_idx].w += pei;
        for (unsigned int l = 0; l < 6; l++)
            this->m_virial_partial[l+6*mem_idx] += viriali[l];
        }
#pragma omp barrier

    // now that the partial sums are complete, sum up the results in parallel
#pragma omp for
    for (int i = 0; i < (int)this->m_pdata->getN(); i++)
        {
        // assign result from thread 0
        h_force.data[i].x = this->m_fdata_partial[i].x;
        h_force.data[i].y = this->m_fdata_partial[i].y;
        h_force.data[i].z = this->m_fdata_partial[i].z;
        h_force.data[i].w = this->m_fdata_partial[i].w;
        for (unsigned int l = 0; l < 6; l++)
            h_virial.data[l*this->m_virial_pitch+i]  = this->m_virial_partial[l+6*i];

        #ifdef ENABLE_OPENMP
        // add results from other threads
        int nthreads = omp_get_num_threads();
        for (int thread = 1; thread < nthreads; thread++)
            {
            unsigned int mem_idx = this->m_index_thread_partial(i,thread);
            h_force.data[i].x += this->m_fdata_partial[mem_idx].x;
            h_force.data[i].y += this->m_fdata_partial[mem_idx].y;
            h_force.data[i].z += this->m_fdata_partial[mem_idx].z;
            h_force.data[i].w += this->m_fdata_partial[mem_idx].w;
            h_virial.data[i]  += this->m_virial_partial[mem_idx];
            for (unsigned int l = 0; l < 6; l++)
                 h_virial.data[l*this->m_virial_pitch+i]  = this->m_virial_partial[l+6*mem_idx];
            }
        #endif
        }
    } // end omp parallel

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

    flags |= PotentialPair<evaluator>::getRequestedCommFlags(timestep);

    return flags;
    } 
#endif


//! Export this pair potential to python
/*! \param name Name of the class in the exported python module
    \tparam T Class type to export. \b Must be an instantiated PotentialPairDPDThermo class template.
    \tparam Base Base class of \a T. \b Must be PotentialPair<evaluator> with the same evaluator as used in \a T.
*/

//NOTE - not sure this boost python export is set up correctly.
template < class T, class Base > void export_PotentialPairDPDThermo(const std::string& name)
    {
    boost::python::scope in_pair =
        boost::python::class_<T, boost::shared_ptr<T>, boost::python::bases< Base >, boost::noncopyable >
                  (name.c_str(), boost::python::init< boost::shared_ptr<SystemDefinition>, boost::shared_ptr<NeighborList>, const std::string& >())
                  .def("setSeed", &T::setSeed)
                  .def("setT", &T::setT)
                  ;
    }

#ifdef WIN32
#pragma warning( pop )
#endif

#endif // __POTENTIAL_PAIR_DPDTHERMO_H__

