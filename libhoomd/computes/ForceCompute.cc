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
// Maintainer: joaander, grva, baschult

/*! \file ForceCompute.cc
    \brief Defines the ForceCompute class
*/

#ifdef WIN32
#pragma warning( push )
#pragma warning( disable : 4244 )
#endif

#include "ForceCompute.h"
#include <iostream>
using namespace std;

#include <boost/python.hpp>
using namespace boost::python;

#include <boost/shared_ptr.hpp>
#include <boost/bind.hpp>
using namespace boost;

/*! \post \c force, and \c virial are all initialized as empty arrays
*/
ForceDataArrays::ForceDataArrays()
    {
    }

/*! \param sysdef System to compute forces on
    \post The Compute is initialized and all memory needed for the forces is allocated
    \post \c force and \c virial GPUarrays are initialized
    \post All forces are initialized to 0
*/
ForceCompute::ForceCompute(boost::shared_ptr<SystemDefinition> sysdef) : Compute(sysdef), m_particles_sorted(false),
    m_index_thread_partial(0)
    {
    assert(m_pdata);
    assert(m_pdata->getN() > 0);
    
    // allocate data on the host
    unsigned int num_particles = m_pdata->getN();
    GPUArray<Scalar4>  force(num_particles,exec_conf);
    GPUArray<Scalar>  virial(num_particles,exec_conf);
    m_force.swap(force);
    m_virial.swap(virial);
    m_force.memclear();
    m_virial.memclear();

    m_fdata_partial = NULL;
    m_virial_partial = NULL;
  
    // connect to the ParticleData to recieve notifications when particles change order in memory
    m_sort_connection = m_pdata->connectParticleSort(bind(&ForceCompute::setParticlesSorted, this));
    }

/*! \post m_fdata and virial _partial are both allocated, and m_index_thread_partial is intiialized for indexing them
*/
void ForceCompute::allocateThreadPartial()
    {
    assert(exec_conf->n_cpu >= 1);
    m_index_thread_partial = Index2D(m_pdata->getN(), exec_conf->n_cpu);
    //Don't use GPU arrays here, *_partial's only used on CPU
    m_fdata_partial = new Scalar4[m_index_thread_partial.getNumElements()];
    m_virial_partial = new Scalar[m_index_thread_partial.getNumElements()];
   }

/*! Frees allocated memory
*/
ForceCompute::~ForceCompute()
    {
    if (m_fdata_partial)
        {
        delete[] m_fdata_partial;
        m_fdata_partial = NULL;
        delete[] m_virial_partial;
        m_virial_partial = NULL;
        }
    m_sort_connection.disconnect();
    }

/*! Sums the total potential energy calculated by the last call to compute() and returns it.
*/
Scalar ForceCompute::calcEnergySum()
    {
    const ForceDataArrays& arrays = acquire();
    
    // always perform the sum in double precision for better accuracy
    // this is cheating and is really just a temporary hack to get logging up and running
    // the potential accuracy loss in simulations needs to be evaluated here and a proper
    // summation algorithm put in place
    double pe_total = 0.0;
    for (unsigned int i=0; i < m_pdata->getN(); i++)
        {
        pe_total += (double)arrays.pe[i];
        }
        
    return Scalar(pe_total);
    }

/*! Access the computed forces on the CPU, this may require copying data from the GPU
    \returns Structure of arrays of the x,y,and z components of the forces on each particle
    calculated by the last call to compute()
    
    \note These are const pointers so the caller cannot muss with the data
 */
const ForceDataArrays& ForceCompute::acquire()
    {
    return m_arrays;
    }
/*! Performs the force computation.
    \param timestep Current Timestep
    \note If compute() has previously been called with a value of timestep equal to
        the current value, the forces are assumed to already have been computed and nothing will
        be done
*/

void ForceCompute::compute(unsigned int timestep)
    {
    // skip if we shouldn't compute this step
    if (!m_particles_sorted && !shouldCompute(timestep))
        return;
        
    computeForces(timestep);
    m_particles_sorted = false;
    }

/*! \param num_iters Number of iterations to average for the benchmark
    \returns Milliseconds of execution time per calculation

    Calls computeForces repeatedly to benchmark the force compute.
*/

double ForceCompute::benchmark(unsigned int num_iters)
    {
    ClockSource t;
    // warm up run
    computeForces(0);
    
#ifdef ENABLE_CUDA
    if (exec_conf->isCUDAEnabled())
        {
        cudaThreadSynchronize();
        CHECK_CUDA_ERROR();
        }
#endif
    
    // benchmark
    uint64_t start_time = t.getTime();
    for (unsigned int i = 0; i < num_iters; i++)
        computeForces(0);
        
#ifdef ENABLE_CUDA
    if (exec_conf->isCUDAEnabled())
        cudaThreadSynchronize();
#endif
    uint64_t total_time_ns = t.getTime() - start_time;
    
    // convert the run time to milliseconds
    return double(total_time_ns) / 1e6 / double(num_iters);
    }

//! Wrapper class for wrapping pure virtual methodos of ForceCompute in python
class ForceComputeWrap : public ForceCompute, public wrapper<ForceCompute>
    {
    public:
        //! Constructor
        /*! \param sysdef Particle data passed to the base class */
        ForceComputeWrap(shared_ptr<SystemDefinition> sysdef) : ForceCompute(sysdef) { }
    protected:
        //! Calls the overidden ForceCompute::computeForces()
        /*! \param timestep parameter to pass on to the overidden method */
        void computeForces(unsigned int timestep)
            {
            this->get_override("computeForces")(timestep);
            }
    };

void export_ForceCompute()
    {
    class_< ForceComputeWrap, boost::shared_ptr<ForceComputeWrap>, bases<Compute>, boost::noncopyable >
    ("ForceCompute", init< boost::shared_ptr<SystemDefinition> >())
    .def("acquire", &ForceCompute::acquire, return_value_policy<copy_const_reference>())
    .def("getForce", &ForceCompute::getForce)
    .def("getVirial", &ForceCompute::getVirial)
    .def("getEnergy", &ForceCompute::getEnergy)
    ;
    }

#ifdef WIN32
#pragma warning( pop )
#endif

