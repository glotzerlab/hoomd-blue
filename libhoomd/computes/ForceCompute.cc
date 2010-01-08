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
ForceDataArrays::ForceDataArrays() : fx(NULL), fy(NULL), fz(NULL), pe(NULL), virial(NULL)
    {
    }

#ifdef ENABLE_CUDA
/*! \post \a d_data.force, \a d_data.virial and \a h_staging are all set to NULL
    \post \a m_num_local and \a m_local_start are set to 0
*/
ForceDataArraysGPU::ForceDataArraysGPU()
    {
    // zero pointers
    d_data.force = NULL;
    d_data.virial = NULL;
    h_staging = NULL;
    // zero flags
    m_num_local = m_local_start = 0;
    }

/*! \param num_local Number of particles local to this GPU
    \param local_start Starting index of the particles local to this GPU

    \pre allocate() has not previously been called
    \post Memory is allocated on the GPU for the force data
    \note allocate() \b must be called on the GPU on which the data is going to be used
*/
cudaError_t ForceDataArraysGPU::allocate(unsigned int num_local, unsigned int local_start)
    {
    // sanity checks
    assert(h_staging == NULL);
    
    // allocate GPU data and check for errors
    cudaError_t error = d_data.allocate(num_local);
    if (error != cudaSuccess)
        return error;
        
    // allocate host staging memory and check for errors
    error = cudaMallocHost((void **)((void *)&h_staging), num_local*sizeof(float4));
    if (error != cudaSuccess)
        return error;
        
    // fill out variables
    m_num_local = num_local;
    m_local_start = local_start;
    
    // all done, return success
    return cudaSuccess;
    }

/*! \pre allocate() has previously been called
    \post All allocated memory is freed
    \note deallocate() \b must be called on the same GPU as allocate()
*/
cudaError_t ForceDataArraysGPU::deallocate()
    {
    // sanity checks
    assert(h_staging != NULL);
    
    // free the memory on the GPU and check for errors
    cudaError_t error = d_data.deallocate();
    if (error != cudaSuccess)
        return error;
        
    // free the staging memory and check for errors
    error = cudaFreeHost((void*)h_staging);
    if (error != cudaSuccess)
        return error;
        
    // all done, return success
    return cudaSuccess;
    }

/*! \pre All data has been allocated and initialized
    \post Data from \a h_data is copied to the GPU data in \a d_data
    \param fx source of fx
    \param fy source of fy
    \param fz source of fz
    \param pe source of pe
    \param virial source of virial
*/
cudaError_t ForceDataArraysGPU::hostToDeviceCopy(Scalar *fx, Scalar *fy, Scalar *fz, Scalar *pe, Scalar *virial)
    {
    // sanity checks
    assert(fx != NULL);
    assert(fy != NULL);
    assert(fz != NULL);
    assert(pe != NULL);
    assert(virial != NULL);
    assert(d_data.force != NULL);
    assert(d_data.virial != NULL);
    assert(m_num_local != 0);
    
    // start by filling out the staging array with interleaved forces
    for (unsigned int i = 0; i < m_num_local; i++)
        {
        unsigned int global_idx = i + m_local_start;
        h_staging[i] = make_float4(fx[global_idx],
                                   fy[global_idx],
                                   fz[global_idx],
                                   pe[global_idx]);
        }
        
    // copy it to the device and check for errors
    cudaError_t error = cudaMemcpy(d_data.force, h_staging, sizeof(float4)*m_num_local, cudaMemcpyHostToDevice);
    if (error != cudaSuccess)
        return error;
        
    // copy virial to the device and check for errors
    error = cudaMemcpy(d_data.virial, virial + m_local_start, sizeof(float)*m_num_local, cudaMemcpyHostToDevice);
    if (error != cudaSuccess)
        return error;
        
    // all done, return success
    return cudaSuccess;
    }

/*! \pre All data has been allocated and initialized
    \post Data from the GPU \a d_data is copied to the host in \a h_data
    \param fx desitnation for the fx
    \param fy desitnation for the fy
    \param fz desitnation for the fz
    \param pe desitnation for the pe
    \param virial desitnation for the virial
*/
cudaError_t ForceDataArraysGPU::deviceToHostCopy(Scalar *fx, Scalar *fy, Scalar *fz, Scalar *pe, Scalar *virial)
    {
    // sanity checks
    assert(fx != NULL);
    assert(fy != NULL);
    assert(fz != NULL);
    assert(pe != NULL);
    assert(virial != NULL);
    assert(d_data.force != NULL);
    assert(d_data.virial != NULL);
    assert(m_num_local != 0);
    
    // copy from the device to the staging area
    cudaError_t error = cudaMemcpy(h_staging, d_data.force, sizeof(float4)*m_num_local, cudaMemcpyDeviceToHost);
    if (error != cudaSuccess)
        return error;
        
    // start by filling out the staging array with interleaved forces
    for (unsigned int i = 0; i < m_num_local; i++)
        {
        unsigned int global_idx = i + m_local_start;
        float4 f = h_staging[i];
        fx[global_idx] = f.x;
        fy[global_idx] = f.y;
        fz[global_idx] = f.z;
        pe[global_idx] = f.w;
        }
        
    // copy virial to the device and check for errors
    error = cudaMemcpy(virial + m_local_start, d_data.virial, sizeof(float)*m_num_local, cudaMemcpyDeviceToHost);
    if (error != cudaSuccess)
        return error;
        
    // all done, return success
    return cudaSuccess;
    }

#endif

/*! \param sysdef System to compute forces on
    \post The Compute is initialized and all memory needed for the forces is allocated
    \post \c fx, \c fy, \c fz pointers in m_arrays are set
    \post All forces are initialized to 0
*/
ForceCompute::ForceCompute(boost::shared_ptr<SystemDefinition> sysdef) : Compute(sysdef), m_particles_sorted(false)
    {
    assert(m_pdata);
    assert(m_pdata->getN() > 0);
    
    // allocate data on the host
    unsigned int num_particles = m_pdata->getN();
    m_arrays.fx = m_fx = new Scalar[num_particles];
    m_arrays.fy = m_fy = new Scalar[num_particles];
    m_arrays.fz = m_fz = new Scalar[num_particles];
    m_arrays.pe = m_pe = new Scalar[num_particles];
    m_arrays.virial = m_virial = new Scalar[num_particles];
    
    // zero host data
    for (unsigned int i = 0; i < num_particles; i++)
        m_fx[i] = m_fy[i] = m_fz[i] = m_pe[i] = m_virial[i] = Scalar(0.0);
        
#ifdef ENABLE_CUDA
    // setup ForceDataArrays on each GPU
    m_gpu_forces.resize(exec_conf.gpu.size());
    
    if (!exec_conf.gpu.empty())
        {
        exec_conf.tagAll(__FILE__, __LINE__);
        
        for (unsigned int cur_gpu = 0; cur_gpu < exec_conf.gpu.size(); cur_gpu++)
            exec_conf.gpu[cur_gpu]->call(bind(&ForceDataArraysGPU::allocate, &m_gpu_forces[cur_gpu], m_pdata->getLocalNum(cur_gpu), m_pdata->getLocalBeg(cur_gpu)));
            
        hostToDeviceCopy();
        m_data_location = cpugpu;
        }
    else
        m_data_location = cpu;
#endif
        
    // connect to the ParticleData to recieve notifications when particles change order in memory
    m_sort_connection = m_pdata->connectParticleSort(bind(&ForceCompute::setParticlesSorted, this));
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
    for (unsigned int cur_gpu = 0; cur_gpu < exec_conf.gpu.size(); cur_gpu++)
        exec_conf.gpu[cur_gpu]->call(bind(&ForceDataArraysGPU::deallocate, &m_gpu_forces[cur_gpu]));
#endif
        
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
#ifdef ENABLE_CUDA
    
    // this is the complicated graphics card version, need to do some work
    // switch based on the current location of the data
    switch (m_data_location)
        {
        case cpu:
            // if the data is solely on the cpu, life is easy, return the data arrays
            // and stay in the same state
            return m_arrays;
            break;
        case cpugpu:
            // if the data is up to date on both the cpu and gpu, life is easy, return
            // the data arrays and stay in the same state
            return m_arrays;
            break;
        case gpu:
            // if the data resides on the gpu, it needs to be copied back to the cpu
            // this changes to the cpugpu state since the data is now fully up to date on
            // both
            deviceToHostCopy();
            m_data_location = cpugpu;
            return m_arrays;
            break;
        default:
            // anything other than the above is an undefined state!
            assert(false);
            return m_arrays;
            break;
        }
        
    // the apple compiler thinks we could get to here, make it happy
    // anything other than the above is an undefined state!
    assert(false);
    return m_arrays;
    
#else
    
    return m_arrays;
#endif
    }

#ifdef ENABLE_CUDA
/*! Access computed forces on the GPU. This may require copying data from the CPU if the forces
    were computed there.
    \returns Data pointer to the forces on the GPU

    \note For performance reasons, the returned pointers will \b not change
    from call to call. The call still must be made, however, to ensure that
    the data has been copied to the GPU.
*/
vector<ForceDataArraysGPU>& ForceCompute::acquireGPU()
    {
    if (exec_conf.gpu.empty())
        {
        cerr << endl << "***Error! Acquiring forces on GPU, but there is no GPU in the exection configuration" << endl << endl;
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
    
    exec_conf.tagAll(__FILE__, __LINE__);
    for (unsigned int cur_gpu = 0; cur_gpu < exec_conf.gpu.size(); cur_gpu++)
        exec_conf.gpu[cur_gpu]->callAsync(bind(&ForceDataArraysGPU::hostToDeviceCopy, &m_gpu_forces[cur_gpu], m_fx, m_fy, m_fz, m_pe, m_virial));
    exec_conf.syncAll();
    
    //if (m_prof) m_prof->pop(exec_conf, 0, m_single_xarray_bytes*4);
    }

/*! \sa hostToDeviceCopy()
*/
void ForceCompute::deviceToHostCopy()
    {
    // commenting profiling: enable when benchmarking suspected slow portions of the code. This isn't needed all the time
    // if (m_prof) m_prof->push("ForceCompute - GPU->CPU");
    
    exec_conf.tagAll(__FILE__, __LINE__);
    for (unsigned int cur_gpu = 0; cur_gpu < exec_conf.gpu.size(); cur_gpu++)
        exec_conf.gpu[cur_gpu]->callAsync(bind(&ForceDataArraysGPU::deviceToHostCopy, &m_gpu_forces[cur_gpu], m_fx, m_fy, m_fz, m_pe, m_virial));
    exec_conf.syncAll();
    
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
    exec_conf.callAll(bind(cudaThreadSynchronize));
#endif
    
    // benchmark
    uint64_t start_time = t.getTime();
    for (unsigned int i = 0; i < num_iters; i++)
        computeForces(0);
        
#ifdef ENABLE_CUDA
    exec_conf.callAll(bind(cudaThreadSynchronize));
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

