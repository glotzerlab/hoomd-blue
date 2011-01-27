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

/*! \post \c fx, \c fy, \c fz, \c pe, and \c virial are all set to NULL
*/
ForceDataArrays::ForceDataArrays()
    {
    }

/*! \param sysdef System to compute forces on
    \post The Compute is initialized and all memory needed for the forces is allocated
    \post \c fx, \c fy, \c fz pointers in m_arrays are set
    \post All forces are initialized to 0
*/
ForceCompute::ForceCompute(boost::shared_ptr<SystemDefinition> sysdef) : Compute(sysdef), m_particles_sorted(false),
    m_index_thread_partial(0)
    {
    assert(m_pdata);
    assert(m_pdata->getN() > 0);
    
    // allocate data on the host
    unsigned int num_particles = m_pdata->getN();
<<<<<<< .mine
		GPUArray<Scalar4> forces(num_particles,exec_conf);
		GPUArray<Scalar>  virial(num_particles,exec_conf);
		m_force.swap(force);
		m_virial.swap(virial);

		m_fdata_partial = NULL;
=======
    m_arrays.f = m_f = GPUArray<Scalar4>(num_particles,exec_conf);
    m_arrays.virial = m_virial = GPUArray<Scalar>(num_particles,exec_conf);
    m_fdata_partial = NULL;
>>>>>>> .r3651
    m_virial_partial = NULL;
    
#ifdef ENABLE_CUDA
    // setup ForceDataArrays the GPU
    if (exec_conf->isCUDAEnabled())
        {
        m_gpu_forces.allocate(m_pdata->getN());
        CHECK_CUDA_ERROR();
        
        hostToDeviceCopy();
        m_data_location = cpugpu;
        }
    else
        m_data_location = cpu;
#endif
        
    // connect to the ParticleData to recieve notifications when particles change order in memory
    m_sort_connection = m_pdata->connectParticleSort(bind(&ForceCompute::setParticlesSorted, this));
    }

/*! \post m_fdata and virial _partial are both allocated, and m_index_thread_partial is intiialized for indexing them
*/
void ForceCompute::allocateThreadPartial()
    {
    assert(exec_conf->n_cpu >= 1);
    m_index_thread_partial = Index2D(m_pdata->getN(), exec_conf->n_cpu);
    m_fdata_partial = GPUArray<Scalar4>(m_index_thread_partial.getNumElements(),exec_conf);
    m_virial_partial = GPUArray<Scalar>(m_index_thread_partial.getNumElements(),exec_conf);
    }

/*! Frees allocated memory
*/
ForceCompute::~ForceCompute()
    {
    // free the host data
    delete[] m_fx;
    m_arrays.fx = m_fx = NULL;
    delete[] m_fy;
    m_arrays.fy = m_fy = NULL;
    delete[] m_fz;
    m_arrays.fz = m_fz = NULL;
    delete[] m_pe;
    m_arrays.pe = m_pe = NULL;
    delete[] m_virial;
    m_arrays.virial = m_virial = NULL;
    
#ifdef ENABLE_CUDA
    if (exec_conf->isCUDAEnabled())
        {
        m_gpu_forces.deallocate();
        CHECK_CUDA_ERROR();
        }
#endif

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

#ifdef ENABLE_CUDA
/*! Access computed forces on the GPU. This may require copying data from the CPU if the forces
    were computed there.
    \returns Data pointer to the forces on the GPU

    \note For performance reasons, the returned pointers will \b not change
    from call to call. The call still must be made, however, to ensure that
    the data has been copied to the GPU.
*/
ForceDataArraysGPU& ForceCompute::acquireGPU()
    {
    if (!exec_conf->isCUDAEnabled())
        {
        cerr << endl << "***Error! Acquiring forces on GPU, but hoomd is running on the CPU" << endl << endl;
        throw runtime_error("Error acquiring GPU forces");
        }
        
    // this is the complicated graphics card version, need to do some work
    // switch based on the current location of the data
    switch (m_data_location)
        {
        case cpu:
            // if the data is on the cpu, we need to copy it over to the gpu
            hostToDeviceCopy();
            // now we are in the cpugpu state
            m_data_location = cpugpu;
            return m_gpu_forces;
            break;
        case cpugpu:
            // if the data is up to date on both the cpu and gpu, life is easy
            // state remains the same, and return it
            return m_gpu_forces;
            break;
        case gpu:
            // if the data resides on the gpu, life is easy
            // state remains the same, and return it
            return m_gpu_forces;
            break;
        default:
            // anything other than the above is an undefined state!
            assert(false);
            return m_gpu_forces;
            break;
        }
    }

/*! Force data from the host is copied to each GPU in the execution configuration.
*/
void ForceCompute::hostToDeviceCopy()
    {
    // commenting profiling: enable when benchmarking suspected slow portions of the code. This isn't needed all the time
    // if (m_prof) m_prof->push("ForceCompute - CPU->GPU");
    
    m_gpu_forces.hostToDeviceCopy(m_fx, m_fy, m_fz, m_pe, m_virial);
    if (exec_conf->isCUDAErrorCheckingEnabled())
        CHECK_CUDA_ERROR();

    //if (m_prof) m_prof->pop(exec_conf, 0, m_single_xarray_bytes*4);
    }

/*! \sa hostToDeviceCopy()
*/
void ForceCompute::deviceToHostCopy()
    {
    // commenting profiling: enable when benchmarking suspected slow portions of the code. This isn't needed all the time
    // if (m_prof) m_prof->push("ForceCompute - GPU->CPU");
    
    m_gpu_forces.deviceToHostCopy(m_fx, m_fy, m_fz, m_pe, m_virial);
    if (exec_conf->isCUDAErrorCheckingEnabled())
        CHECK_CUDA_ERROR();
    
    //if (m_prof) m_prof->pop(exec_conf, 0, m_single_xarray_bytes*4);
    }

#endif

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

