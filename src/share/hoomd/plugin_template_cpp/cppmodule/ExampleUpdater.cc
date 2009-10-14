// we need to include boost.python in order to export ExampleUpdater to python
#include <boost/python.hpp>
using namespace boost::python;

// we need to include boost.bind for GPUWorker execution
#include <boost/bind.hpp>
using namespace boost;

#include "ExampleUpdater.h"
#ifdef ENABLE_CUDA
#include "ExampleUpdater.cuh"
#endif

// ********************************
// here follows the code for ExampleUpdater on the CPU

/*! \param sysdef System to zero the velocities of
*/
ExampleUpdater::ExampleUpdater(boost::shared_ptr<SystemDefinition> sysdef)
        : Updater(sysdef)
    {
    }


/*! Perform the needed calculations to zero the system's velocity
    \param timestep Current time step of the simulation
*/
void ExampleUpdater::update(unsigned int timestep)
    {
    if (m_prof) m_prof->push("ExampleUpdater");
    
    // access the particle data for writing on the CPU
    assert(m_pdata);
    ParticleDataArrays arrays = m_pdata->acquireReadWrite();
    
    // zero the velocity of every particle
    for (unsigned int i = 0; i < arrays.nparticles; i++)
        {
        arrays.vx[i] = Scalar(0.0);
        arrays.vy[i] = Scalar(0.0);
        arrays.vz[i] = Scalar(0.0);
        }
        
    m_pdata->release();
    
    if (m_prof) m_prof->pop();
    }

void export_ExampleUpdater()
    {
    class_<ExampleUpdater, boost::shared_ptr<ExampleUpdater>, bases<Updater>, boost::noncopyable>
    ("ExampleUpdater", init< boost::shared_ptr<SystemDefinition> >())
    ;
    }

// ********************************
// here follows the code for ExampleUpdater on the GPU

#ifdef ENABLE_CUDA

/*! \param sysdef System to zero the velocities of
*/
ExampleUpdaterGPU::ExampleUpdaterGPU(boost::shared_ptr<SystemDefinition> sysdef)
        : ExampleUpdater(sysdef)
    {
    }


/*! Perform the needed calculations to zero the system's velocity
    \param timestep Current time step of the simulation
*/
void ExampleUpdaterGPU::update(unsigned int timestep)
    {
    if (m_prof) m_prof->push("ExampleUpdater");
    
    // access the particle data arrays for writing on the GPU
    vector<gpu_pdata_arrays>& d_pdata = m_pdata->acquireReadWriteGPU();
    
    exec_conf.tagAll(__FILE__, __LINE__);
    exec_conf.gpu[0]->call(bind(gpu_zero_velocities, d_pdata[0]));
    
    m_pdata->release();
    
    if (m_prof) m_prof->pop();
    }

void export_ExampleUpdaterGPU()
    {
    class_<ExampleUpdaterGPU, boost::shared_ptr<ExampleUpdaterGPU>, bases<ExampleUpdater>, boost::noncopyable>
    ("ExampleUpdaterGPU", init< boost::shared_ptr<SystemDefinition> >())
    ;
    }

#endif // ENABLE_CUDA

