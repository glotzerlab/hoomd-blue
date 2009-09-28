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

// $Id$
// $URL$

#ifdef WIN32
#pragma warning( push )
#pragma warning( disable : 4103 4244 )
#endif

#include <iostream>

//! name the boost unit test module
#define BOOST_TEST_MODULE NVERigidUpdaterTests
#include "boost_utf_configure.h"

#include <boost/bind.hpp>
#include <boost/function.hpp>
#include <boost/shared_ptr.hpp>

#include "NVTUpdater.h"
#ifdef ENABLE_CUDA
#include "NVTUpdaterGPU.h"
#endif

#include "BinnedNeighborList.h"
#include "Initializers.h"
#include "LJForceCompute.h"

#ifdef ENABLE_CUDA
#include "BinnedNeighborListGPU.h"
#include "LJForceComputeGPU.h"
#endif

#include "saruprng.h"
#include <math.h>
#include <time.h>

using namespace std;
using namespace boost;

/*! \file nve_rigid_updater_test.cc
	\brief Implements unit tests for NVERigidUpdater 
	\ingroup unit_tests
*/


//! Tolerance for floating point comparisons
#ifdef SINGLE_PRECISION
const Scalar tol = Scalar(1e-2);
#else
const Scalar tol = 1e-3;
#endif

//! Typedef'd NVEUpdator class factory
typedef boost::function<shared_ptr<NVTUpdater> (shared_ptr<SystemDefinition> sysdef, Scalar deltaT, Scalar Q, Scalar T)> nvtup_creator;

//! NVTUpdater creator
shared_ptr<NVTUpdater> base_class_nvt_creator(shared_ptr<SystemDefinition> sysdef, Scalar deltaT, Scalar Q, Scalar T)
	{
	shared_ptr<VariantConst> T_variant(new VariantConst(T));
	return shared_ptr<NVTUpdater>(new NVTUpdater(sysdef, deltaT, Q, T_variant));
	}
	
#ifdef ENABLE_CUDA
//! NVTUpdaterGPU factory for the unit tests
shared_ptr<NVTUpdater> gpu_nvt_creator(shared_ptr<SystemDefinition> sysdef, Scalar deltaT, Scalar Q, Scalar T)
	{
	shared_ptr<VariantConst> T_variant(new VariantConst(T));	
	return shared_ptr<NVTUpdater>(new NVTUpdaterGPU(sysdef, deltaT, Q, T_variant));
	}
#endif

void nvt_updater_energy_tests(nvtup_creator nvt_creator, const ExecutionConfiguration& exec_conf)
{
	#ifdef ENABLE_CUDA
	g_gpu_error_checking = true;
	#endif
	
	// check that the nve updater can actually integrate particle positions and velocities correctly
	// start with a 2 particle system to keep things simple: also put everything in a huge box so boundary conditions
	// don't come into play
	unsigned int nbodies = 3000;
	unsigned int nparticlesperbody = 5;
	unsigned int N = nbodies * nparticlesperbody;
	Scalar box_length = 60.0;
	shared_ptr<SystemDefinition> sysdef(new SystemDefinition(N, BoxDim(box_length), 1, 0, 0, 0, 0, exec_conf));
	shared_ptr<ParticleData> pdata = sysdef->getParticleData();
	BoxDim box = pdata->getBox();
	
	Scalar temperature = 2.5;
	unsigned int steps = 10000;
	unsigned int sampling = 1000;
	
	// setup a simple initial state
	unsigned int ibody = 0;
	unsigned int iparticle = 0;
	Scalar x0 = box.xlo + 0.01;
	Scalar y0 = box.ylo + 0.01;
	Scalar z0 = box.zlo + 0.01;
	Scalar xspacing = 6.0f;
	Scalar yspacing = 2.0f;
	Scalar zspacing = 2.0f;
	
	unsigned int seed = 10483;
	boost::shared_ptr<Saru> random = boost::shared_ptr<Saru>(new Saru(seed));
	Scalar KE = Scalar(0.0);
	
	ParticleDataArrays arrays = pdata->acquireReadWrite();
	
	// initialize bodies in a cubic lattice with some velocity profile
	for (unsigned int i = 0; i < nbodies; i++)
	{
		for (unsigned int j = 0; j < nparticlesperbody; j++)
		{
			arrays.x[iparticle] = x0 + 1.0 * j; 
			arrays.y[iparticle] = y0 + 0.0;
			arrays.z[iparticle] = z0 + 0.0;
			
			arrays.vx[iparticle] = random->d(); 
			arrays.vy[iparticle] = random->d();  
			arrays.vz[iparticle] = random->d();  
			
			KE += Scalar(0.5) * (arrays.vx[iparticle]*arrays.vx[iparticle] + arrays.vy[iparticle]*arrays.vy[iparticle] + arrays.vz[iparticle]*arrays.vz[iparticle]);
			
			arrays.body[iparticle] = ibody;
						
			iparticle++;
		}
		
		x0 += xspacing;
		if (x0 + xspacing >= box.xhi) 
		{
			x0 = box.xlo + 0.01;
		
			y0 += yspacing;
			if (y0 + yspacing >= box.yhi) 
			{
				y0 = box.ylo + 0.01;
				
				z0 += zspacing;
				if (z0 + zspacing >= box.zhi) 
					z0 = box.zlo + 0.01;
			}
		}
		
		ibody++;
	}
	
	assert(iparticle == N);
	
	pdata->release();

	Scalar deltaT = Scalar(0.001);
	Scalar Q = Scalar(2.0);
	Scalar tau = sqrt(Q / (Scalar(3.0) * temperature));

	shared_ptr<NVTUpdater> nvt_up = nvt_creator(sysdef, deltaT, tau, temperature);
	shared_ptr<BinnedNeighborListGPU> nlist(new BinnedNeighborListGPU(sysdef, Scalar(2.5), Scalar(0.3)));
	shared_ptr<LJForceComputeGPU> fc(new LJForceComputeGPU(sysdef, nlist, Scalar(2.5)));
//	shared_ptr<BinnedNeighborList> nlist(new BinnedNeighborList(sysdef, Scalar(2.5), Scalar(0.3)));
//	shared_ptr<LJForceCompute> fc(new LJForceCompute(sysdef, nlist, Scalar(2.5)));
	
	// setup some values for alpha and sigma
	Scalar epsilon = Scalar(1.0);
	Scalar sigma = Scalar(1.0);
	Scalar alpha = Scalar(1.0);
	Scalar lj1 = Scalar(4.0) * epsilon * pow(sigma, Scalar(12.0));
	Scalar lj2 = alpha * Scalar(4.0) * epsilon * pow(sigma, Scalar(6.0));
	
	// specify the force parameters
	fc->setParams(0,0,lj1,lj2);
	
	nvt_up->addForceCompute(fc);
	
	sysdef->init();

	Scalar PE;
	
	shared_ptr<RigidData> rdata = sysdef->getRigidData();
	unsigned int nrigid_dof = rdata->getNumDOF();
	
	// Rescale particle velocities to desired temperature:
	Scalar current_temp = 2.0 * KE / nrigid_dof;
	Scalar factor = sqrt(temperature / current_temp);
	
	arrays = pdata->acquireReadWrite();
	for (unsigned int j = 0; j < N; j++) 
	{
		arrays.vx[j] *= factor;
		arrays.vy[j] *= factor;
		arrays.vz[j] *= factor;
	}
	
	pdata->release();
	
	cout << "Number of particles = " << N << "; Number of rigid bodies = " << rdata->getNumBodies() << "\n";
	cout << "Step\tTemp\tPotEng\tKinEng\tTotalE\n";
		
	clock_t start = clock();
	
//	steps = 0;
	for (unsigned int i = 0; i <= steps; i++)
		{

		nvt_up->update(i);
			
		if (i % sampling == 0) 
			{			
			arrays = pdata->acquireReadWrite();
			KE = Scalar(0.0);
			for (unsigned int j = 0; j < N; j++) 
				KE += Scalar(0.5) * (arrays.vx[j]*arrays.vx[j] + arrays.vy[j]*arrays.vy[j] + arrays.vz[j]*arrays.vz[j]);
			PE = fc->calcEnergySum();
				
			current_temp = 2.0 * KE / nrigid_dof;
			printf("%8d\t%12.8g\t%12.8g\t%12.8g\t%12.8g\n", i, current_temp, PE / N, KE / N, (PE + KE) / N); 	
			
			pdata->release();
			}
		}
	
	clock_t end = clock();
	double elapsed = (double)(end - start) / (double)CLOCKS_PER_SEC;	
	printf("Elapased time: %f sec or %f TPS\n", elapsed, (double)steps / elapsed);

	// Output coordinates
	arrays = pdata->acquireReadWrite();
	
	FILE *fp = fopen("test_energy_nvt.xyz", "w");
	Scalar Lx = box.xhi - box.xlo;
	Scalar Ly = box.yhi - box.ylo;
	Scalar Lz = box.zhi - box.zlo;
	fprintf(fp, "%d\n%f\t%f\t%f\n", arrays.nparticles, Lx, Ly, Lz);
	for (unsigned int i = 0; i < arrays.nparticles; i++) 
		fprintf(fp, "N\t%f\t%f\t%f\n", arrays.x[i], arrays.y[i], arrays.z[i]);
	
	fclose(fp);
	pdata->release();

	}

/*
BOOST_AUTO_TEST_CASE( NVTUpdater_energy_tests )
{
	printf("\nTesting energy conservation on CPU...\n");
	nvtup_creator nvt_creator = bind(base_class_nvt_creator, _1, _2, _3, _4);
	nvt_updater_energy_tests(nvt_creator, ExecutionConfiguration(ExecutionConfiguration::CPU, 0));
}
*/
#ifdef ENABLE_CUDA

//! boost test case for base class integration tests
BOOST_AUTO_TEST_CASE( NVTUpdaterGPU_energy_tests )
{
	printf("\nTesting energy conservation on GPU...\n");
	nvtup_creator nvt_creator_gpu = bind(gpu_nvt_creator, _1, _2, _3, _4);
	nvt_updater_energy_tests(nvt_creator_gpu, ExecutionConfiguration());
}

#endif

#ifdef WIN32
#pragma warning( pop )
#endif
