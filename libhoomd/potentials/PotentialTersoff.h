/*
Highly Optimized Object-oriented Many-particle Dynamics -- Blue Edition
(HOOMD-blue) Open Source Software License Copyright 2008, 2009 Ames Laboratory
Iowa State University and The Regents of the University of Michigan All rights
reserved.

HOOMD-blue may contain modifications ("Contributions") provided, and to which
copyright is held, by various Contributors who have granted The Regents of the
University of Michigan the right to modify and/or distribute such Contributions.

Redistribution and use of HOOMD-blue, in source and binary forms, with or
without modification, are permitted, provided that the following conditions are
met:

* Redistributions of source code must retain the above copyright notice, this
list of conditions, and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright notice, this
list of conditions, and the following disclaimer in the documentation and/or
other materials provided with the distribution.

* Neither the name of the copyright holder nor the names of HOOMD-blue's
contributors may be used to endorse or promote products derived from this
software without specific prior written permission.

Disclaimer

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDER AND CONTRIBUTORS ``AS IS''
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, AND/OR
ANY WARRANTIES THAT THIS SOFTWARE IS FREE OF INFRINGEMENT ARE DISCLAIMED.

IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT,
INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE
OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF
ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

#ifndef __POTENTIAL_TERSOFF_H__
#define __POTENTIAL_TERSOFF_H__

#include <iostream>
#include <stdexcept>
#include <boost/shared_ptr.hpp>
#include <boost/python.hpp>
#include <fstream>

#include "HOOMDMath.h"
#include "Index1D.h"
#include "GPUArray.h"
#include "ForceCompute.h"
#include "NeighborList.h"

#ifdef ENABLE_OPENMP
#include <omp.h>
#endif

#ifdef WIN32
#pragma warning( push )
#pragma warning( disable : 4103 4244 )
#endif

/*! \file PotentialTersoff.h
    \brief Defines the template class for standard three-body potentials
    \details The heart of the code that computes three-body potentials is in this file.
    \note This header cannot be compiled by nvcc
*/

#ifdef NVCC
#error This header cannot be compiled by nvcc
#endif

//! Template class for computing three-body potentials
/*! <b>Overview:</b>
    PotentialTersoff computes standard three-body potentials and forces between all particles in the
    simulation.  It employs the use of a neighbor list to limit the number of comutations done to
    only those particles within the cutoff radius of each other.  the computation of the actual
    potential is not performed directly by this class, but by an evaluator class (e.g.
    EvaluatorTersoff) which is passed in as a template parameter so the computations are performed
    as efficiently as possible.

    PotentialTersoff handles most of the internal details common to all standard three-body potentials.
     - A cutoff radius to be specified per particle type-pair
     - Per type-pair parameters are stored and a set method is provided
     - Logging methods are provided for the energy
     - All the details about looping through the particles, computing dr, computing the virial, etc. are handled

    <b>Implementation details</b>

    Unlike the pair potentials, the three-body potentials offer two force directions: ij and ik.
    In addition, some three-body potentials (such as the Tersoff potential) compute unique forces on
    each of the three particles involved.  Three-body evaluators must thus return six force magnitudes:
    two for each particle.  These values are returned in the Scalar4 values \a force_divr_ij and
    \a force_divr_ik.  The x components refer to particle i, y to particle j, and z to particle k.
    If your particular three-body potential does not compute one of these forces, then the evaluator
    can simply return 0 for that force.  In addition, the potential energy is stored in the w component
    of force_divr_ij.  Scalar4 values are used instead of Scalar3's in order to
    maintain compatibility between the CPU and GPU codes.

    rcutsq, ronsq, and the params are stored per particle type-pair. It wastes a little bit of space, but benchmarks
    show that storing the symmetric type pairs and indexing with Index2D is faster than not storing redudant pairs
    and indexing with Index2DUpperTriangular. All of these values are stored in GPUArray
    for easy access on the GPU by a derived class. The type of the parameters is defined by \a param_type in the
    potential evaluator class passed in. See the appropriate documentation for the evaluator for the definition of each
    element of the parameters.

    For profiling and logging, PotentialTersoff needs to know the name of the potential. For now, that will be queried from
    the evaluator. Perhaps in the future we could allow users to change that so multiple pair potentials could be logged
    independently.

    \sa export_PotentialTersoff()
*/
template < class evaluator >
class PotentialTersoff : public ForceCompute
    {
    public:
        //! Param type from evaluator
        typedef typename evaluator::param_type param_type;

        //! Construct the potential
        PotentialTersoff(boost::shared_ptr<SystemDefinition> sysdef,
                         boost::shared_ptr<NeighborList> nlist,
                         const std::string& log_suffix="");
        //! Destructor
        virtual ~PotentialTersoff() { };

        //! Set the pair parameters for a single type pair
        virtual void setParams(unsigned int typ1, unsigned int typ2, const param_type& param);
        //! Set the rcut for a single type pair
        virtual void setRcut(unsigned int typ1, unsigned int typ2, Scalar rcut);
        //! Set ron for a single type pair
        virtual void setRon(unsigned int typ1, unsigned int typ2, Scalar ron);

        //! Returns a list of log quantities this compute calculates
        virtual std::vector< std::string > getProvidedLogQuantities();
        //! Calculates the requested log value and returns it
        virtual Scalar getLogValue(const std::string& quantity, unsigned int timestep);

        //! Shifting modes that can be applied to the energy
        enum energyShiftMode
            {
            no_shift = 0,
            shift,
            xplor
            };

        //! Set the mode to use for shifting the energy
        void setShiftMode(energyShiftMode mode)
            {
            m_shift_mode = mode;
            }
    protected:
        boost::shared_ptr<NeighborList> m_nlist;    //!< The neighborlist to use for the computation
        energyShiftMode m_shift_mode;               //!< Store the mode with which to handle the energy shift at r_cut
        Index2D m_typpair_idx;                      //!< Helper class for indexing per type pair arrays
        GPUArray<Scalar> m_rcutsq;                  //!< Cuttoff radius squared per type pair
        GPUArray<Scalar> m_ronsq;                   //!< ron squared per type pair
        GPUArray<param_type> m_params;   //!< Pair parameters per type pair
        std::string m_prof_name;                    //!< Cached profiler name
        std::string m_log_name;                     //!< Cached log name

        //! Actually compute the forces
        virtual void computeForces(unsigned int timestep);
    };

/*! \param sysdef System to compute forces on
    \param nlist Neighborlist to use for computing the forces
    \param log_suffix Name given to this instance of the force
*/
template < class evaluator >
PotentialTersoff< evaluator >::PotentialTersoff(boost::shared_ptr<SystemDefinition> sysdef,
                                                boost::shared_ptr<NeighborList> nlist,
                                                const std::string& log_suffix)
    : ForceCompute(sysdef), m_nlist(nlist), m_shift_mode(no_shift), m_typpair_idx(m_pdata->getNTypes())
    {
    assert(m_pdata);
    assert(m_nlist);

    GPUArray<Scalar> rcutsq(m_typpair_idx.getNumElements(), exec_conf);
    m_rcutsq.swap(rcutsq);
    GPUArray<Scalar> ronsq(m_typpair_idx.getNumElements(), exec_conf);
    m_ronsq.swap(ronsq);
    GPUArray<param_type> params(m_typpair_idx.getNumElements(), exec_conf);
    m_params.swap(params);

    // initialize name
    m_prof_name = std::string("Triplet ") + evaluator::getName();
    m_log_name = std::string("pair_") + evaluator::getName() + std::string("_energy") + log_suffix;

    // initialize memory for per thread reduction
    allocateThreadPartial();
    }

/*! \param typ1 First type index in the pair
    \param typ2 Second type index in the pair
    \param param Parameter to set
    \note When setting the value for (\a typ1, \a typ2), the parameter for (\a typ2, \a typ1) is automatically
          set.
*/
template< class evaluator >
void PotentialTersoff< evaluator >::setParams(unsigned int typ1, unsigned int typ2, const param_type& param)
    {
    if (typ1 >= m_pdata->getNTypes() || typ2 >= m_pdata->getNTypes())
        {
        this->m_exec_conf->msg->error() << "pair." << evaluator::getName() << ": Trying to set pair params for a non existant type! "
                  << typ1 << "," << typ2 << std::endl << std::endl;
        throw std::runtime_error("Error setting parameters in PotentialTersoff");
        }

    ArrayHandle<param_type> h_params(m_params, access_location::host, access_mode::readwrite);
    h_params.data[m_typpair_idx(typ1, typ2)] = param;
    h_params.data[m_typpair_idx(typ2, typ1)] = param;
    }

/*! \param typ1 First type index in the pair
    \param typ2 Second type index in the pair
    \param rcut Cuttoff radius to set
    \note When setting the value for (\a typ1, \a typ2), the parameter for (\a typ2, \a typ1) is automatically
          set.
*/
template< class evaluator >
void PotentialTersoff< evaluator >::setRcut(unsigned int typ1, unsigned int typ2, Scalar rcut)
    {
    if (typ1 >= m_pdata->getNTypes() || typ2 >= m_pdata->getNTypes())
        {
        std::cerr << std::endl << "***Error! Trying to set rcut for a non existant type! "
                  << typ1 << "," << typ2 << std::endl << std::endl;
        throw std::runtime_error("Error setting parameters in PotentialTersoff");
        }

    ArrayHandle<Scalar> h_rcutsq(m_rcutsq, access_location::host, access_mode::readwrite);
    h_rcutsq.data[m_typpair_idx(typ1, typ2)] = rcut * rcut;
    h_rcutsq.data[m_typpair_idx(typ2, typ1)] = rcut * rcut;
    }

/*! \param typ1 First type index in the pair
    \param typ2 Second type index in the pair
    \param ron XPLOR r_on radius to set
    \note When setting the value for (\a typ1, \a typ2), the parameter for (\a typ2, \a typ1) is automatically
          set.
*/
template< class evaluator >
void PotentialTersoff< evaluator >::setRon(unsigned int typ1, unsigned int typ2, Scalar ron)
    {
    if (typ1 >= m_pdata->getNTypes() || typ2 >= m_pdata->getNTypes())
        {
        std::cerr << std::endl << "***Error! Trying to set ron for a non existant type! "
                  << typ1 << "," << typ2 << std::endl << std::endl;
        throw std::runtime_error("Error setting parameters in PotentialTersoff");
        }

    ArrayHandle<Scalar> h_ronsq(m_ronsq, access_location::host, access_mode::readwrite);
    h_ronsq.data[m_typpair_idx(typ1, typ2)] = ron * ron;
    h_ronsq.data[m_typpair_idx(typ2, typ1)] = ron * ron;
    }

/*! PotentialTersoff provides:
     - \c pair_"name"_energy
    where "name" is replaced with evaluator::getName()
*/
template< class evaluator >
std::vector< std::string > PotentialTersoff< evaluator >::getProvidedLogQuantities()
    {
    vector<string> list;
    list.push_back(m_log_name);
    return list;
    }

/*! \param quantity Name of the log value to get
    \param timestep Current timestep of the simulation
*/
template< class evaluator >
Scalar PotentialTersoff< evaluator >::getLogValue(const std::string& quantity, unsigned int timestep)
    {
    if (quantity == m_log_name)
        {
        compute(timestep);
        return calcEnergySum();
        }
    else
        {
        this->m_exec_conf->msg->error() << "pair." << evaluator::getName() << ": " << quantity << " is not a valid log quantity" 
                  << std::endl << endl;
        throw std::runtime_error("Error getting log value");
        }
    }

/*! \post The forces are computed for the given timestep. The neighborlist's compute method is called to ensure
    that it is up to date before proceeding.

    \param timestep specifies the current time step of the simulation
*/
template< class evaluator >
void PotentialTersoff< evaluator >::computeForces(unsigned int timestep)
    {
    // start by updating the neighborlist
    m_nlist->compute(timestep);

    // start the profile for this compute
    if (m_prof) m_prof->push(m_prof_name);

    // The three-body potentials can't handle a half neighbor list, so check now.
    bool third_law = m_nlist->getStorageMode() == NeighborList::half;
    if (third_law)
        {
        std::cerr << std::endl << "***Error! PotentialTersoff cannot handle a half neighborlist"
                  << std::endl << std::endl;
        throw std::runtime_error("Error computing forces in PotentialTersoff");
        }

    // access the neighbor list, particle data, and system box
    ArrayHandle<unsigned int> h_n_neigh(m_nlist->getNNeighArray(), access_location::host, access_mode::read);
    ArrayHandle<unsigned int> h_nlist(m_nlist->getNListArray(), access_location::host, access_mode::read);
    Index2D nli = m_nlist->getNListIndexer();

    ArrayHandle<Scalar4> h_pos(m_pdata->getPositions(), access_location::host, access_mode::read);
    ArrayHandle<Scalar> h_diameter(m_pdata->getDiameters(), access_location::host, access_mode::read);
    ArrayHandle<Scalar> h_charge(m_pdata->getCharges(), access_location::host, access_mode::read);


    //force arrays
    ArrayHandle<Scalar4> h_force(m_force,access_location::host, access_mode::overwrite);


    const BoxDim& box = m_pdata->getBox();
    ArrayHandle<Scalar> h_ronsq(m_ronsq, access_location::host, access_mode::read);
    ArrayHandle<Scalar> h_rcutsq(m_rcutsq, access_location::host, access_mode::read);
    ArrayHandle<param_type> h_params(m_params, access_location::host, access_mode::read);

#pragma omp parallel
{
    #ifdef ENABLE_OPENMP
    int tid = omp_get_thread_num();
    #else
    int tid = 0;
    #endif

    // need to start from a zero force, energy
    memset(&m_fdata_partial[m_index_thread_partial(0,tid)] , 0, sizeof(Scalar4)*m_pdata->getN());

    // for each particle
#pragma omp for schedule(guided)
    for (int i = 0; i < (int)m_pdata->getN(); i++)
        {
        // access the particle's position and type (MEM TRANSFER: 4 scalars)
        Scalar3 posi = make_scalar3(h_pos.data[i].x, h_pos.data[i].y, h_pos.data[i].z);
        unsigned int typei = __scalar_as_int(h_pos.data[i].w);
        // sanity check
        assert(typei < m_pdata->getNTypes());

        // initialize current force and potential energy of particle i to 0
        Scalar3 fi = make_scalar3(0.0, 0.0, 0.0);
        Scalar pei = 0.0;

        // loop over all of the neighbors of this particle
        const unsigned int size = (unsigned int)h_n_neigh.data[i];
        for (unsigned int j = 0; j < size; j++)
            {
            // access the index of neighbor j (MEM TRANSFER: 1 scalar)
            unsigned int jj = h_nlist.data[nli(i, j)];
            assert(jj < m_pdata->getN());

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
            param_type param = h_params.data[typpair_idx];
            Scalar rcutsq = h_rcutsq.data[typpair_idx];

            // evaluate the base repulsive and attractive terms
            Scalar fR = 0.0;
            Scalar fA = 0.0;
            evaluator eval(rij_sq, rcutsq, param);
            bool evaluated = eval.evalRepulsiveAndAttractive(fR, fA);

            if (evaluated)
                {
                // evaluate chi
                Scalar chi = 0.0;
                for (unsigned int k = 0; k < size; k++)
                    {
                    // access the index of neighbor k
                    unsigned int kk = h_nlist.data[nli(i,k)];
                    assert(kk < m_pdata->getN());

                    // access the position and type of neighbor k
                    Scalar3 posk = make_scalar3(h_pos.data[kk].x, h_pos.data[kk].y, h_pos.data[kk].z);
                    unsigned int typek = __scalar_as_int(h_pos.data[kk].w);
                    assert(typek < m_pdata->getNTypes());

                    // access the type pair parameters for i and k
                    typpair_idx = m_typpair_idx(typei, typek);
                    param_type temp_param = h_params.data[typpair_idx];

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
                            cos_th = dot(dxij, dxik) / sqrt(rij_sq * rik_sq);

                        // evaluate the partial chi term
                        eval.setRik(rik_sq);
                        if (evaluator::needsAngle())
                            eval.setAngle(cos_th);

                        eval.evalChi(chi);
                        }
                    }

                // evaluate the force and energy from the ij interaction
                Scalar force_divr = Scalar(0.0);
                Scalar potential_eng = Scalar(0.0);
                Scalar bij = Scalar(0.0);
                eval.evalForceij(fR, fA, chi, bij, force_divr, potential_eng);

                // add this force to particle i
                fi += force_divr * dxij;
                pei += potential_eng * Scalar(0.5);

                // add this force to particle j
                fj += Scalar(-1.0) * force_divr * dxij;
                pej += potential_eng * Scalar(0.5);

                // evaluate the force from the ik interactions
                for (unsigned int k = 0; k < size; k++)
                    {
                    // access the index of neighbor k
                    unsigned int kk = h_nlist.data[nli(i, k)];
                    assert(kk < m_pdata->getN());

                    // access the position and type of neighbor k
                    Scalar3 posk = make_scalar3(h_pos.data[kk].x, h_pos.data[kk].y, h_pos.data[kk].z);
                    unsigned int typek = __scalar_as_int(h_pos.data[kk].w);
                    assert(typek < m_pdata->getNTypes());

                    // access the type pair parameters for i and k
                    typpair_idx = m_typpair_idx(typei, typek);
                    param_type temp_param = h_params.data[typpair_idx];

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
                        Scalar4 force_divr_ij = make_scalar4(0.0, 0.0, 0.0, 0.0);
                        Scalar4 force_divr_ik = make_scalar4(0.0, 0.0, 0.0, 0.0);
                        eval.evalForceik(fR, fA, chi, bij, force_divr_ij, force_divr_ik);

                        // add the force to particle i
                        // (FLOPS: 17)
                        fi.x += force_divr_ij.x * dxij.x + force_divr_ik.x * dxik.x;
                        fi.y += force_divr_ij.x * dxij.y + force_divr_ik.x * dxik.y;
                        fi.z += force_divr_ij.x * dxij.z + force_divr_ik.x * dxik.z;

                        // add the force to particle j (FLOPS: 17)
                        fj.x += force_divr_ij.y * dxij.x + force_divr_ik.y * dxik.x;
                        fj.y += force_divr_ij.y * dxij.y + force_divr_ik.y * dxik.y;
                        fj.z += force_divr_ij.y * dxij.z + force_divr_ik.y * dxik.z;

                        // add the force to particle k
                        fk.x += force_divr_ij.z * dxij.x + force_divr_ik.z * dxik.x;
                        fk.y += force_divr_ij.z * dxij.y + force_divr_ik.z * dxik.y;
                        fk.z += force_divr_ij.z * dxij.z + force_divr_ik.z * dxik.z;

                        // increment the force for particle k
                        unsigned int mem_idx = m_index_thread_partial(kk, tid);
                        m_fdata_partial[mem_idx].x += fk.x;
                        m_fdata_partial[mem_idx].y += fk.y;
                        m_fdata_partial[mem_idx].z += fk.z;
                        }
                    }
                }
            // increment the force and potential energy for particle j
            unsigned int mem_idx = m_index_thread_partial(jj, tid);
            m_fdata_partial[mem_idx].x += fj.x;
            m_fdata_partial[mem_idx].y += fj.y;
            m_fdata_partial[mem_idx].z += fj.z;
            m_fdata_partial[mem_idx].w += pej;
            }
        // finally, increment the force and potential energy for particle i
        unsigned int mem_idx = m_index_thread_partial(i,tid);
        m_fdata_partial[mem_idx].x += fi.x;
        m_fdata_partial[mem_idx].y += fi.y;
        m_fdata_partial[mem_idx].z += fi.z;
        m_fdata_partial[mem_idx].w += pei;
        }
#pragma omp barrier

    // now that the partial sums are complete, sum up the results in parallel
#pragma omp for
    for (int i = 0; i < (int)m_pdata->getN(); i++)
        {
        // assign result from thread 0
        h_force.data[i].x  = m_fdata_partial[i].x;
        h_force.data[i].y = m_fdata_partial[i].y;
        h_force.data[i].z = m_fdata_partial[i].z;
        h_force.data[i].w = m_fdata_partial[i].w;

        #ifdef ENABLE_OPENMP
        // add results from other threads
        int nthreads = omp_get_num_threads();
        for (int thread = 1; thread < nthreads; thread++)
            {
            unsigned int mem_idx = m_index_thread_partial(i,thread);
            h_force.data[i].x += m_fdata_partial[mem_idx].x;
            h_force.data[i].y += m_fdata_partial[mem_idx].y;
            h_force.data[i].z += m_fdata_partial[mem_idx].z;
            h_force.data[i].w += m_fdata_partial[mem_idx].w;
            }
        #endif
        }
    } // end omp parallel

    if (m_prof) m_prof->pop();
    }

//! Export this triplet potential to python
/*! \param name Name of the class in the exported python module
    \tparam T Class type to export. \b Must be an instantiated PotentialTersoff class template.
*/
template < class T > void export_PotentialTersoff(const std::string& name)
    {
    boost::python::scope in_pair =
        boost::python::class_<T, boost::shared_ptr<T>, boost::python::bases<ForceCompute>, boost::noncopyable >
                  (name.c_str(), boost::python::init< boost::shared_ptr<SystemDefinition>, boost::shared_ptr<NeighborList>, const std::string& >())
                  .def("setParams", &T::setParams)
                  .def("setRcut", &T::setRcut)
                  .def("setRon", &T::setRon)
                  .def("setShiftMode", &T::setShiftMode)
                  ;

    boost::python::enum_<typename T::energyShiftMode>("energyShiftMode")
        .value("no_shift", T::no_shift)
        .value("shift", T::shift)
        .value("xplor", T::xplor)
    ;
    }

#ifdef WIN32
#pragma warning( pop )
#endif

#endif

