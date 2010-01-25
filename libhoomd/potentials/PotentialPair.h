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

// $Id$
// $URL$
// Maintainer: joaander

#ifndef __POTENTIAL_PAIR_H__
#define __POTENTIAL_PAIR_H__

#include <iostream>
#include <stdexcept>
#include <boost/shared_ptr.hpp>
#include <boost/python.hpp>

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

/*! \file PotentialPair.h
    \brief Defines the template class for standard pair potentials
    \details The heart of the code that computes pair potentials is in this file.
    \note This header cannot be compiled by nvcc
*/

#ifdef NVCC
#error This header cannot be compiled by nvcc
#endif

//! Template class for computing pair potentials
/*! <b>Overview:</b>
    PotentialPair computes standard pair potentials (and forces) between all particle pairs in the simulation. It
    employs the use of a neighbor list to limit the number of computations done to only those particles with the 
    cuttoff radius of each other. The computation of the actual V(r) is not performed directly by this class, but
    by an evaluator class (e.g. EvaluatorPairLJ) which is passed in as a template parameter so the compuations
    are performed as efficiently as possible.
    
    PotentialPair handles most of the gory internal details common to all standard pair potentials.
     - A cuttoff radius to be specified per particle type pair
     - The energy can be globally shifted to 0 at the cuttoff
     - XPLOR switching can be enabled
     - Per type pair parameters are stored and a set method is provided
     - Logging methods are provided for the energy
     - And all the details about looping through the particles, computing dr, computing the virial, etc. are handled
    
    A note on the design of XPLOR switching:
    We need to be able to handle smooth XPLOR switching in systems of mixed LJ/WCA particles. There are three modes to
    enable all of the various use-cases:
     - Mode 1: No shifting. All pair potentials are computed as is and not shifted to 0 at the cuttoff.
     - Mode 2: Shift everything. All pair potentials (no matter what type pair) are shifted so they are 0 at the cuttoff
     - Mode 3: XPLOR switching enabled. A r_on value is specified per type pair. When r_on is less than r_cut, normal
       XPLOR switching will be applied to the unshifted potential. When r_on is greather than r_cut, the energy will
       be shifted. In this manner, a valid r_on value can be given for the LJ interactions and r_on > r_cut can be set
       for WCA (which will then be shifted).
    
    XPLOR switching gets significantly more complicated for all pair potentials when shifted potentials are used. Thus,
    the combination of XPLOR switching + shifted potentials will not be supported to avoid slowing down the calculation
    for everyone.
    
    <b>Implementation details</b>
    
    rcutsq, ronsq, and the params are stored per particle type pair. It wastes a little bit of space, but benchmarks
    show that storing the symmetric type pairs and indexing with Index2D is faster than not storing redudant pairs
    and indexing with Index2DUpperTriangular. All of these values are stored in GPUArray
    for easy access on the GPU by a derived class. The type of the parameters is defined by \a param_type in the
    potential evaluator class passed in. See the appropriate documentation for the evaluator for the definition of each
    element of the parameters.
    
    For profiling and logging, PotentialPair needs to know the name of the potential. For now, that will be queried from
    the evaluator. Perhaps in the future we could allow users to change that so multiple pair potentials could be logged
    independantly.
    
    \sa export_PotentialPair()
*/
template < class evaluator >
class PotentialPair : public ForceCompute
    {
    public:
        //! Param type from evaluator
        typedef typename evaluator::param_type param_type;
    
        //! Construct the pair potential
        PotentialPair(boost::shared_ptr<SystemDefinition> sysdef, boost::shared_ptr<NeighborList> nlist);
        //! Destructor
        virtual ~PotentialPair() { };

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
*/
template < class evaluator >
PotentialPair< evaluator >::PotentialPair(boost::shared_ptr<SystemDefinition> sysdef,
                                                boost::shared_ptr<NeighborList> nlist)
    : ForceCompute(sysdef), m_nlist(nlist), m_shift_mode(no_shift), m_typpair_idx(1)
    {
    assert(m_pdata);
    assert(m_nlist);
    
    // initialize the per type pair memory
    unsigned int ntypes = m_pdata->getNTypes();
    assert(ntypes > 0);
    m_typpair_idx = Index2D(ntypes);
    
    GPUArray<Scalar> rcutsq(m_typpair_idx.getNumElements(), m_pdata->getExecConf());
    m_rcutsq.swap(rcutsq);
    GPUArray<Scalar> ronsq(m_typpair_idx.getNumElements(), m_pdata->getExecConf());
    m_ronsq.swap(ronsq);
    GPUArray<param_type> params(m_typpair_idx.getNumElements(), m_pdata->getExecConf());
    m_params.swap(params);
    
    // initialize name
    m_prof_name = std::string("Pair ") + evaluator::getName();
    m_log_name = std::string("pair_") + evaluator::getName() + std::string("_energy");

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
void PotentialPair< evaluator >::setParams(unsigned int typ1, unsigned int typ2, const param_type& param)
    {
    if (typ1 >= m_pdata->getNTypes() || typ2 >= m_pdata->getNTypes())
        {
        std::cerr << std::endl << "***Error! Trying to set pair params for a non existant type! "
                  << typ1 << "," << typ2 << std::endl << std::endl;
        throw std::runtime_error("Error setting parameters in PotentialPair");
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
void PotentialPair< evaluator >::setRcut(unsigned int typ1, unsigned int typ2, Scalar rcut)
    {
    if (typ1 >= m_pdata->getNTypes() || typ2 >= m_pdata->getNTypes())
        {
        std::cerr << std::endl << "***Error! Trying to set rcut for a non existant type! "
                  << typ1 << "," << typ2 << std::endl << std::endl;
        throw std::runtime_error("Error setting parameters in PotentialPair");
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
void PotentialPair< evaluator >::setRon(unsigned int typ1, unsigned int typ2, Scalar ron)
    {
    if (typ1 >= m_pdata->getNTypes() || typ2 >= m_pdata->getNTypes())
        {
        std::cerr << std::endl << "***Error! Trying to set ron for a non existant type! "
                  << typ1 << "," << typ2 << std::endl << std::endl;
        throw std::runtime_error("Error setting parameters in PotentialPair");
        }
    
    ArrayHandle<Scalar> h_ronsq(m_ronsq, access_location::host, access_mode::readwrite);
    h_ronsq.data[m_typpair_idx(typ1, typ2)] = ron * ron;
    h_ronsq.data[m_typpair_idx(typ2, typ1)] = ron * ron;
    }

/*! PotentialPair provides:
     - \c pair_"name"_energy
    where "name" is replaced with evaluator::getName()
*/
template< class evaluator >
std::vector< std::string > PotentialPair< evaluator >::getProvidedLogQuantities()
    {
    vector<string> list;
    list.push_back(m_log_name);
    return list;
    }

/*! \param quantity Name of the log value to get
    \param timestep Current timestep of the simulation
*/
template< class evaluator >
Scalar PotentialPair< evaluator >::getLogValue(const std::string& quantity, unsigned int timestep)
    {
    if (quantity == m_log_name)
        {
        compute(timestep);
        return calcEnergySum();
        }
    else
        {
        std::cerr << std::endl << "***Error! " << quantity << " is not a valid log quantity for PotentialPair" 
                  << std::endl << endl;
        throw std::runtime_error("Error getting log value");
        }
    }

/*! \post The pair forces are computed for the given timestep. The neighborlist's compute method is called to ensure
    that it is up to date before proceeding.

    \param timestep specifies the current time step of the simulation
*/
template< class evaluator >
void PotentialPair< evaluator >::computeForces(unsigned int timestep)
    {
    // start by updating the neighborlist
    m_nlist->compute(timestep);
    
    // start the profile for this compute
    if (m_prof) m_prof->push(m_prof_name);
    
    // depending on the neighborlist settings, we can take advantage of newton's third law
    // to reduce computations at the cost of memory access complexity: set that flag now
    bool third_law = m_nlist->getStorageMode() == NeighborList::half;
    
    // access the neighbor list, particle data, and system box
    const vector< vector< unsigned int > >& full_list = m_nlist->getList();
    const ParticleDataArraysConst& arrays = m_pdata->acquireReadOnly();
    const BoxDim& box = m_pdata->getBox();
    ArrayHandle<Scalar> h_ronsq(m_ronsq, access_location::host, access_mode::read);
    ArrayHandle<Scalar> h_rcutsq(m_rcutsq, access_location::host, access_mode::read);
    ArrayHandle<param_type> h_params(m_params, access_location::host, access_mode::read);
    
    // precalculate box lengths for use in the periodic imaging
    Scalar Lx = box.xhi - box.xlo;
    Scalar Ly = box.yhi - box.ylo;
    Scalar Lz = box.zhi - box.zlo;
    
#pragma omp parallel
    {
    #ifdef ENABLE_OPENMP
    int tid = omp_get_thread_num();
    #else
    int tid = 0;
    #endif

    // need to start from a zero force, energy and virial
    memset(&m_fdata_partial[m_index_thread_partial(0,tid)] , 0, sizeof(Scalar4)*arrays.nparticles);
    memset(&m_virial_partial[m_index_thread_partial(0,tid)] , 0, sizeof(Scalar)*arrays.nparticles);
    
    // for each particle
#pragma omp for schedule(dynamic, 100)
    for (unsigned int i = 0; i < arrays.nparticles; i++)
        {
        // access the particle's position and type (MEM TRANSFER: 4 scalars)
        Scalar xi = arrays.x[i];
        Scalar yi = arrays.y[i];
        Scalar zi = arrays.z[i];
        unsigned int typei = arrays.type[i];
        // sanity check
        assert(typei < m_pdata->getNTypes());
        
        // access diameter and charge (if needed)
        Scalar di = Scalar(0.0);
        Scalar qi = Scalar(0.0);
        if (evaluator::needsDiameter())
            di = arrays.diameter[i];
        if (evaluator::needsCharge())
            qi = arrays.charge[i];
        
        // initialize current particle force, potential energy, and virial to 0
        Scalar fxi = 0.0;
        Scalar fyi = 0.0;
        Scalar fzi = 0.0;
        Scalar pei = 0.0;
        Scalar viriali = 0.0;
        
        // loop over all of the neighbors of this particle
        const vector< unsigned int >& list = full_list[i];
        const unsigned int size = (unsigned int)list.size();
        for (unsigned int k = 0; k < size; k++)
            {
            // access the index of this neighbor (MEM TRANSFER: 1 scalar)
            unsigned int j = list[k];
            assert(j < m_pdata->getN());
            
            // calculate dr_ji (MEM TRANSFER: 3 scalars / FLOPS: 3)
            Scalar dx = xi - arrays.x[j];
            Scalar dy = yi - arrays.y[j];
            Scalar dz = zi - arrays.z[j];
            
            // access the type of the neighbor particle (MEM TRANSFER: 1 scalar)
            unsigned int typej = arrays.type[j];
            assert(typej < m_pdata->getNTypes());
            
            // access diameter and charge (if needed)
            Scalar dj = Scalar(0.0);
            Scalar qj = Scalar(0.0);
            if (evaluator::needsDiameter())
                dj = arrays.diameter[j];
            if (evaluator::needsCharge())
                qj = arrays.charge[j];
            
            // apply periodic boundary conditions (FLOPS: 9)
            if (dx >= box.xhi)
                dx -= Lx;
            else if (dx < box.xlo)
                dx += Lx;
                
            if (dy >= box.yhi)
                dy -= Ly;
            else if (dy < box.ylo)
                dy += Ly;
                
            if (dz >= box.zhi)
                dz -= Lz;
            else if (dz < box.zlo)
                dz += Lz;
                
            // calculate r_ij squared (FLOPS: 5)
            Scalar rsq = dx*dx + dy*dy + dz*dz;
            
            // get parameters for this type pair
            unsigned int typpair_idx = m_typpair_idx(typei, typej);
            param_type param = h_params.data[typpair_idx];
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
            if (evaluator::needsDiameter())
                eval.setDiameter(di, dj);
            if (evaluator::needsCharge())
                eval.setCharge(qi, qj);
            
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
                        Scalar xplor_denom_inv =
                            Scalar(1.0) / ((rcutsq - ronsq) * (rcutsq - ronsq) * (rcutsq - ronsq));
                        
                        Scalar rsq_minus_r_cut_sq = rsq - rcutsq;
                        Scalar s = rsq_minus_r_cut_sq * rsq_minus_r_cut_sq *
                                   (rcutsq + Scalar(2.0) * rsq - Scalar(3.0) * ronsq) * xplor_denom_inv;
                        Scalar ds_dr_divr = Scalar(12.0) * (rsq - ronsq) * rsq_minus_r_cut_sq * xplor_denom_inv;
                        
                        // make modifications to the old pair energy and force
                        pair_eng = old_pair_eng * s;
                        // note: I'm not sure why the minus sign needs to be there: my notes have a +
                        // But this is verified correct via plotting
                        force_divr = s * old_force_divr - ds_dr_divr * old_pair_eng;
                        }
                    }
                    
                // compute the virial (FLOPS: 2)
                Scalar pair_virial = Scalar(1.0/6.0) * rsq * force_divr;
                
                // add the force, potential energy and virial to the particle i
                // (FLOPS: 8)
                fxi += dx*force_divr;
                fyi += dy*force_divr;
                fzi += dz*force_divr;
                pei += pair_eng * Scalar(0.5);
                viriali += pair_virial;
                
                // add the force to particle j if we are using the third law (MEM TRANSFER: 10 scalars / FLOPS: 8)
                if (third_law)
                    {
                    unsigned int mem_idx = m_index_thread_partial(j,tid);
                    m_fdata_partial[mem_idx].x -= dx*force_divr;
                    m_fdata_partial[mem_idx].y -= dy*force_divr;
                    m_fdata_partial[mem_idx].z -= dz*force_divr;
                    m_fdata_partial[mem_idx].w += pair_eng * Scalar(0.5);
                    m_virial_partial[mem_idx] += pair_virial;
                    }
                }
            }
            
        // finally, increment the force, potential energy and virial for particle i
        unsigned int mem_idx = m_index_thread_partial(i,tid);
        m_fdata_partial[mem_idx].x += fxi;
        m_fdata_partial[mem_idx].y += fyi;
        m_fdata_partial[mem_idx].z += fzi;
        m_fdata_partial[mem_idx].w += pei;
        m_virial_partial[mem_idx] += viriali;
        }
#pragma omp barrier
    
    // now that the partial sums are complete, sum up the results in parallel
#pragma omp for
    for (int i = 0; i < (int)arrays.nparticles; i++)
        {
        // assign result from thread 0
        m_fx[i] = m_fdata_partial[i].x;
        m_fy[i] = m_fdata_partial[i].y;
        m_fz[i] = m_fdata_partial[i].z;
        m_pe[i] = m_fdata_partial[i].w;
        m_virial[i] = m_virial_partial[i];

        #ifdef ENABLE_OPENMP
        // add results from other threads
        int nthreads = omp_get_num_threads();
        for (int thread = 1; thread < nthreads; thread++)
            {
            unsigned int mem_idx = m_index_thread_partial(i,thread);
            m_fx[i] += m_fdata_partial[mem_idx].x;
            m_fy[i] += m_fdata_partial[mem_idx].y;
            m_fz[i] += m_fdata_partial[mem_idx].z;
            m_pe[i] += m_fdata_partial[mem_idx].w;
            m_virial[i] += m_virial_partial[mem_idx];
            }
        #endif
        }
    } // end omp parallel

    m_pdata->release();
    
#ifdef ENABLE_CUDA
    // the force data is now only up to date on the cpu
    m_data_location = cpu;
#endif
    
    if (m_prof) m_prof->pop();
    }

//! Export this pair potential to python
/*! \param name Name of the class in the exported python module
    \tparam T Class type to export. \b Must be an instantiated PotentialPair class template.
*/
template < class T > void export_PotentialPair(const std::string& name)
    {
    boost::python::scope in_pair = 
        boost::python::class_<T, boost::shared_ptr<T>, boost::python::bases<ForceCompute>, boost::noncopyable >
                  (name.c_str(), boost::python::init< boost::shared_ptr<SystemDefinition>, boost::shared_ptr<NeighborList> >())
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

#endif // __POTENTIAL_PAIR_H__

