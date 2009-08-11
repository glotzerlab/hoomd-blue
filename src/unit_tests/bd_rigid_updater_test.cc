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

// $Id: nve_updater_test.cc 1622 2009-01-28 22:51:01Z joaander $
// $URL: http://svn2.assembla.com/svn/hoomd/trunk/src/unit_tests/nve_updater_test.cc $

#ifdef WIN32
#pragma warning( push )
#pragma warning( disable : 4103 4244 )
#endif

#include <iostream>

//! name the boost unit test module
#define BOOST_TEST_MODULE BDRigidUpdaterTests
#include "boost_utf_configure.h"

#include <boost/bind.hpp>
#include <boost/function.hpp>
#include <boost/shared_ptr.hpp>

#include "BD_NVTUpdater.h"
#ifdef ENABLE_CUDA
#include "BD_NVTUpdaterGPU.h"
#endif

#include "BinnedNeighborList.h"
#include "Initializers.h"
#include "LJForceCompute.h"

#ifdef ENABLE_CUDA
#include "BinnedNeighborListGPU.h"
#include "LJForceComputeGPU.h"
#endif

#include "FENEBondForceCompute.h"

#ifdef ENABLE_CUDA
#include "FENEBondForceComputeGPU.h"
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
typedef boost::function<shared_ptr<BD_NVTUpdater> (shared_ptr<SystemDefinition> sysdef, Scalar deltaT, Scalar Temp, unsigned int seed)> bdnvtup_creator;

void bd_updater_lj_tests(bdnvtup_creator bdup_creator, ExecutionConfiguration exec_conf)
{
	#ifdef ENABLE_CUDA
	g_gpu_error_checking = true;
	#endif
	
	unsigned int nbodies = 800;
	unsigned int nparticlesperbuildingblock = 7;
	unsigned int body_size = 5;
	unsigned int natomtypes = 2;
	unsigned int nbondtypes = 1;
	
	unsigned int N = nbodies * nparticlesperbuildingblock;
	Scalar box_length = 24.0148;
	shared_ptr<SystemDefinition> sysdef(new SystemDefinition(N, BoxDim(box_length), natomtypes, nbondtypes, 0, 0, 0, exec_conf));
	shared_ptr<ParticleData> pdata = sysdef->getParticleData();
	BoxDim box = pdata->getBox();
	
	// setup a simple initial state
	unsigned int ibody = 0;
	unsigned int iparticle = 0;
	Scalar x0 = box.xlo + 0.01;
	Scalar y0 = box.ylo + 0.01;
	Scalar z0 = box.zlo + 0.01;
	Scalar xspacing = 7.0f;
	Scalar yspacing = 1.0f;
	Scalar zspacing = 2.0f;
	
	unsigned int seed = 10483;
	boost::shared_ptr<Saru> random = boost::shared_ptr<Saru>(new Saru(seed));
	Scalar temperature = 1.0;
	Scalar KE = Scalar(0.0);
	
	ParticleDataArrays arrays = pdata->acquireReadWrite();
	
	// initialize bodies in a cubic lattice with some velocity profile
	for (unsigned int i = 0; i < nbodies; i++)
	{
		for (unsigned int j = 0; j < nparticlesperbuildingblock; j++)
		{
			arrays.x[iparticle] = x0 + 1.0 * j;
			arrays.y[iparticle] = y0 + 0.0;
			arrays.z[iparticle] = z0 + 0.0;

			arrays.vx[iparticle] = random->d(); 
			arrays.vy[iparticle] = random->d();  
			arrays.vz[iparticle] = random->d();  
			
			KE += Scalar(0.5) * (arrays.vx[iparticle]*arrays.vx[iparticle] + arrays.vy[iparticle]*arrays.vy[iparticle] + arrays.vz[iparticle]*arrays.vz[iparticle]);
			
			if (j < body_size) 
				{
				arrays.body[iparticle] = ibody;
				arrays.type[iparticle] = 1;
				}
			else
				{
				arrays.type[iparticle] = 0;
					
				sysdef->getBondData()->addBond(Bond(0, iparticle, iparticle-1));
				}
				
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
	shared_ptr<BD_NVTUpdater> bdnvt_up = bdup_creator(sysdef, deltaT, temperature, 48103);
	shared_ptr<BinnedNeighborListGPU> nlist(new BinnedNeighborListGPU(sysdef, Scalar(2.5), Scalar(0.3)));
	shared_ptr<LJForceComputeGPU> fc(new LJForceComputeGPU(sysdef, nlist, Scalar(2.5)));
	
	// setup some values for alpha and sigma
	Scalar epsilon = Scalar(1.0);
	Scalar sigma = Scalar(1.0);
	Scalar alpha_lj = Scalar(1.0);  // alpha = 0.0: close to WCA
	Scalar alpha_wca = Scalar(0.0);  // alpha = 0.0: close to WCA
	Scalar lj1 = Scalar(4.0) * epsilon * pow(sigma, Scalar(12.0));
	Scalar lj2_lj = alpha_lj * Scalar(4.0) * epsilon * pow(sigma, Scalar(6.0));
	Scalar lj2_wca = alpha_wca * Scalar(4.0) * epsilon * pow(sigma, Scalar(6.0));
	
	// specify the force parameters
	fc->setParams(0,0,lj1,lj2_wca);
	fc->setParams(0,1,lj1,lj2_wca);
	fc->setParams(1,1,lj1,lj2_wca);
	
	bdnvt_up->addForceCompute(fc);
	
	shared_ptr<FENEBondForceComputeGPU> fenebond(new FENEBondForceComputeGPU(sysdef));
	fenebond->setParams(0, Scalar(30.0), Scalar(1.5), Scalar(1.0), Scalar(1.122));
	
	bdnvt_up->addForceCompute(fenebond);
	
	
	sysdef->init();

	Scalar PE;
	unsigned int equil_steps = 10000;
	unsigned int steps = 10000000;
	unsigned int sampling = 1000;
	
	shared_ptr<RigidData> rdata = sysdef->getRigidData();
	unsigned int nrigid_dof = rdata->getNumDOF();
	unsigned int nnonrigid_dof = 3 * N - 3 * body_size * nbodies;
	
	// Rescale particle velocities to desired temperature:
	Scalar current_temp = 2.0 * KE / (nrigid_dof + nnonrigid_dof);
	Scalar factor = sqrt(temperature / current_temp);
	
	arrays = pdata->acquireReadWrite();
	for (unsigned int j = 0; j < N; j++) 
	{
		arrays.vx[j] *= factor;
		arrays.vy[j] *= factor;
		arrays.vz[j] *= factor;
	}
	
	pdata->release();
	
	clock_t start = clock();
	
	// Mix with WCA interactions
	cout << "Equilibrating...\n";
	cout << "Number of particles = " << N << "; Number of rigid bodies = " << rdata->getNumBodies() << "\n";
	cout << "Step\tTemp\tPotEng\tKinEng\tTotalE\n";
	
	for (unsigned int i = 0; i <= equil_steps; i++)
		{

		bdnvt_up->update(i);
			
		if (i % sampling == 0) 
			{			
			arrays = pdata->acquireReadWrite();
			KE = Scalar(0.0);
			for (unsigned int j = 0; j < N; j++) 
				KE += Scalar(0.5) * (arrays.vx[j]*arrays.vx[j] + arrays.vy[j]*arrays.vy[j] + arrays.vz[j]*arrays.vz[j]);
			PE = fc->calcEnergySum();
				
			current_temp = 2.0 * KE / (nrigid_dof + nnonrigid_dof);
			printf("%8d\t%12.8g\t%12.8g\t%12.8g\t%12.8g\n", i, current_temp, PE / N, KE / N, (PE + KE) / N); 	
			
			pdata->release();
			}
		}
	
	// Production: turn on LJ interactions between rods
	fc->setParams(1,1,lj1,lj2_lj);
	
	cout << "Production...\n";
	cout << "Number of particles = " << N << "; Number of rigid bodies = " << rdata->getNumBodies() << "\n";
	cout << "Step\tTemp\tPotEng\tKinEng\tTotalE\n";
	for (unsigned int i = 0; i <= equil_steps; i++)
	{
		
		bdnvt_up->update(i);
		
		if (i % sampling == 0) 
		{			
			arrays = pdata->acquireReadWrite();
			KE = Scalar(0.0);
			for (unsigned int j = 0; j < N; j++) 
				KE += Scalar(0.5) * (arrays.vx[j]*arrays.vx[j] + arrays.vy[j]*arrays.vy[j] + arrays.vz[j]*arrays.vz[j]);
			PE = fc->calcEnergySum();
			
			current_temp = 2.0 * KE / (nrigid_dof + nnonrigid_dof);
			printf("%8d\t%12.8g\t%12.8g\t%12.8g\t%12.8g\n", i, current_temp, PE / N, KE / N, (PE + KE) / N); 	
			
			pdata->release();
		}
	}
	
	clock_t end = clock();
	double elapsed = (double)(end - start) / (double)CLOCKS_PER_SEC;	
	printf("Elapased time: %f sec or %f TPS\n", elapsed, (double)steps / elapsed);

	// Output coordinates
	arrays = pdata->acquireReadWrite();
	
	FILE *fp = fopen("test_bd_energy.xyz", "w");
	Scalar Lx = box.xhi - box.xlo;
	Scalar Ly = box.yhi - box.ylo;
	Scalar Lz = box.zhi - box.zlo;
	fprintf(fp, "%d\n%f\t%f\t%f\n", arrays.nparticles, Lx, Ly, Lz);
	for (unsigned int i = 0; i < arrays.nparticles; i++)
		{
		if (arrays.type[i] == 1)
			fprintf(fp, "N\t%f\t%f\t%f\n", arrays.x[i], arrays.y[i], arrays.z[i]);
		else
			fprintf(fp, "C\t%f\t%f\t%f\n", arrays.x[i], arrays.y[i], arrays.z[i]);
		}	
			
	fclose(fp);
	pdata->release();

	}

#ifdef ENABLE_CUDA
//! BD_NVTUpdaterGPU factory for the unit tests
shared_ptr<BD_NVTUpdater> gpu_bdnvt_creator(shared_ptr<SystemDefinition> sysdef, Scalar deltaT, Scalar Temp, unsigned int seed)
{
	shared_ptr<VariantConst> T_variant(new VariantConst(Temp));
	return shared_ptr<BD_NVTUpdater>(new BD_NVTUpdaterGPU(sysdef, deltaT, T_variant, seed, false));
}
#endif

#ifdef ENABLE_CUDA

//! extended LJ-liquid test for the GPU class
BOOST_AUTO_TEST_CASE( BDUpdaterGPU_LJ_tests )
{
	bdnvtup_creator bdnvt_creator_gpu = bind(gpu_bdnvt_creator, _1, _2, _3, _4);
	bd_updater_lj_tests(bdnvt_creator_gpu, ExecutionConfiguration(ExecutionConfiguration::GPU));
}
#endif

#ifdef WIN32
#pragma warning( pop )
#endif
