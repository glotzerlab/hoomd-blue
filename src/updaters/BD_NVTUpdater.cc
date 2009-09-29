/*
Highly Optimized Object-Oriented Molecular Dynamics (HOOMD) Open
Source Software License
Copyright (c) 2008 Ames Laboratory Iowa State University
All rights reserved.

Redistribution and use of HOOMD, in source and binary forms, with or
without modification, are permitted, provided that the following
conditions are met:

* Redistributions of source code must retain the above copyright notice,
this list of conditions and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright
notice, this list of conditions and the following disclaimer in the
documentation and/or other materials provided with the distribution.

* Neither the name of the copyright holder nor the names HOOMD's
contributors may be used to endorse or promote products derived from this
software without specific prior written permission.

Disclaimer

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDER AND
CONTRIBUTORS ``AS IS''  AND ANY EXPRESS OR IMPLIED WARRANTIES,
INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY
AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED.

IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS  BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF
THE POSSIBILITY OF SUCH DAMAGE.
*/

// $Id: BD_NVTUpdater.cc 1206 2008-09-04 18:00:45Z phillicl $
// $URL: http://svn2.assembla.com/svn/hoomd/trunk/src/updaters/BD_NVTUpdater.cc $
// Maintainer: phillicl

/*! \file BD_NVTUpdater.cc
    \brief Defines the BD_NVTUpdater class
*/

#ifdef WIN32
#pragma warning( push )
#pragma warning( disable : 4103 4244 )
#endif

#include <boost/python.hpp>
using namespace boost::python;

#include "BD_NVTUpdater.h"
#include <math.h>

#include <boost/bind.hpp>

#ifdef ENABLE_CUDA
#include "Integrator.cuh"
#endif


using namespace std;

/*! \param sysdef System to update
    \param deltaT Time step to use
    \param Temp Temperature to set
    \param seed Random seed to use for the random force compuataion
*/
BD_NVTUpdater::BD_NVTUpdater(boost::shared_ptr<SystemDefinition> sysdef,
							 Scalar deltaT,
							 boost::shared_ptr<Variant> Temp,
							 unsigned int seed,
							 bool use_diam) 
	: NVEUpdater(sysdef, deltaT), m_T(Temp), m_seed(seed), m_bath(false), m_use_diam(use_diam)
    {
#ifdef ENABLE_CUDA
    // check the execution configuration
    if (exec_conf.exec_mode == ExecutionConfiguration::CPU )
        m_bdfc = boost::shared_ptr<StochasticForceCompute>(new StochasticForceCompute(m_sysdef, m_deltaT, m_T, m_seed, m_use_diam));
    else
        m_bdfc =  boost::shared_ptr<StochasticForceComputeGPU> (new StochasticForceComputeGPU(m_sysdef, m_deltaT, m_T, m_seed, m_use_diam));
        
#else
    m_bdfc = boost::shared_ptr<StochasticForceCompute>(new StochasticForceCompute(m_sysdef, m_deltaT, m_T, m_seed, m_use_diam));
#endif
        
    addStochasticBath();
    }

/*! The StochasticForceCompute is added to the list \a m_forces.
    The index to which it is added is tracked in a m_bath_index so that other calls
    can reference it to set coefficients.
*/
void BD_NVTUpdater::addStochasticBath()
    {
    if (m_bath)
        cout << "Stochastic Bath Already Added" << endl;
    else
        {
        addForceCompute(m_bdfc);
        m_bath = true;
        }
    }

/*! \param Temp Temperature of the Stochastic Bath
*/
void BD_NVTUpdater::setT(boost::shared_ptr<Variant> Temp)
    {
    m_T = Temp;
    m_bdfc->setT(m_T);
    }

/*! Disables the ForceComputes
    Since the base class removes all force computes, this class flags that the stochastic bath
    has been removed so it can be re-added when it is needed.
*/
void BD_NVTUpdater::removeForceComputes()
    {
    m_bath = false;
    Integrator::removeForceComputes();
    }

/*! Uses velocity verlet
    \param timestep Current time step of the simulation

    \pre Associated ParticleData is initialized, and particle positions and velocities
	are set for time timestep
    
	\post Forces and accelerations are computed and particle's positions, velocities
	and accelartions are updated to their values at timestep+1.
*/
void BD_NVTUpdater::update(unsigned int timestep)
    {
    // hack to get correct profiling
    m_bdfc->setProfiler(m_prof);
    
    if (!m_bath) addStochasticBath();
    NVEUpdater::update(timestep);
    }

//! Exports the BD_NVTUpdater class to python
void export_BD_NVTUpdater()
    {
    class_<BD_NVTUpdater, boost::shared_ptr<BD_NVTUpdater>, bases<NVEUpdater>, boost::noncopyable>
    ("BD_NVTUpdater", init< boost::shared_ptr<SystemDefinition>, Scalar, boost::shared_ptr<Variant>, unsigned int, bool >())
    .def("setGamma", &BD_NVTUpdater::setGamma)
    .def("setT", &BD_NVTUpdater::setT)
    ;
    }

#ifdef WIN32
#pragma warning( pop )
#endif
