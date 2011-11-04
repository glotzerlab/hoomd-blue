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

* All publications based on HOOMD-blue, including any reports or published
results obtained, in whole or in part, with HOOMD-blue, will acknowledge its use
according to the terms posted at the time of submission on:
http://codeblue.umich.edu/hoomd-blue/citations.html

* Any electronic documents citing HOOMD-Blue will link to the HOOMD-Blue website
at: http://codeblue.umich.edu/hoomd-blue/.

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

#include "TwoStepBDNVT.h"
#include "TwoStepBDNVTRigid.h"
#ifdef ENABLE_CUDA
#include "TwoStepBDNVTRigidGPU.h"
#include "TwoStepBDNVTGPU.h"
#endif

#include "IntegratorTwoStep.h"

#include "BoxResizeUpdater.h"

#include "AllPairPotentials.h"
#include "NeighborList.h"
#include "Initializers.h"

#include "saruprng.h"
#include <math.h>
#include <time.h>
#include <fstream>
#include <vector>

using namespace std;
using namespace boost;

/*! \file nve_rigid_updater_test.cc
    \brief Implements unit tests for NVERigidUpdater
    \ingroup unit_tests
*/


//! Tolerance for floating point comparisons
#ifdef SINGLE_PRECISION
const Scalar tolerance = Scalar(1e-2);
#else
const Scalar tolerance = Scalar(1e-3);
#endif

struct AtomInfo
{
    int type, localidx, body;
    double mass;
    double x, y, z;
};

struct BondInfo
{
    char type[50];
    int localidxi, localidxj;
    double kappa, R0, sigma, epsilon;
};

struct BuildingBlock
{
    std::vector<AtomInfo> atoms;
    std::vector<BondInfo> bonds;
    double spacing_x, spacing_y, spacing_z;
};

//! Typedef'd TwoStepBDNVTRigid class factory
typedef boost::function<shared_ptr<TwoStepBDNVTRigid> (shared_ptr<SystemDefinition> sysdef, 
                            shared_ptr<ParticleGroup> group, Scalar T, unsigned int seed)> bdnvtup_creator;


void bd_updater_lj_tests(bdnvtup_creator bdup_creator, boost::shared_ptr<ExecutionConfiguration> exec_conf)
    {
    unsigned int nbodies = 800;
    unsigned int nparticlesperbuildingblock = 5;
    unsigned int nbondsperbuildingblock;
    unsigned int body_size = 5;
    unsigned int natomtypes = 2;
    unsigned int nbondtypes = 1;
    
    unsigned int N = nbodies * nparticlesperbuildingblock;
    Scalar box_length = 24.0814;
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
    
    BuildingBlock buildingBlock;
    buildingBlock.spacing_x = 6.0; 
    buildingBlock.spacing_y = 6.0;
    buildingBlock.spacing_z = 2.0;
   
    AtomInfo atomi;
    int num_atoms = 5;
    for (int atom_n = 0; atom_n < num_atoms; atom_n++)
        {
        //Add Atom
        atomi.localidx = atom_n;
        atomi.x = Scalar(atom_n);
        atomi.y = 0.0;
        atomi.z = 0.0;
        atomi.body = 1;
        atomi.type = 1;
        atomi.mass = 1.0;
        buildingBlock.atoms.push_back(atomi);
        }     
                
    nparticlesperbuildingblock = buildingBlock.atoms.size();
    nbondsperbuildingblock = buildingBlock.bonds.size();


    unsigned int seed = 258719;
    boost::shared_ptr<Saru> random = boost::shared_ptr<Saru>(new Saru(seed));
    Scalar temperature = 1.4;
    Scalar KE = Scalar(0.0);
    Scalar PE = Scalar(0.0);
    Scalar AvgT = Scalar(0);
    
    
    ParticleDataArrays arrays = pdata->acquireReadWrite();
    
    // initialize bodies in a cubic lattice with some velocity profile
    for (unsigned int i = 0; i < nbodies; i++)
        {
        for (unsigned int j = 0; j < nparticlesperbuildingblock; j++)
            {
            arrays.x[iparticle] = x0 + buildingBlock.atoms[j].x;
            arrays.y[iparticle] = y0 + buildingBlock.atoms[j].y;
            arrays.z[iparticle] = z0 + buildingBlock.atoms[j].z;

            arrays.vx[iparticle] = random->d();
            arrays.vy[iparticle] = random->d();
            arrays.vz[iparticle] = random->d();
            
            KE += Scalar(0.5) * (arrays.vx[iparticle]*arrays.vx[iparticle] + arrays.vy[iparticle]*arrays.vy[iparticle] + arrays.vz[iparticle]*arrays.vz[iparticle]);
            
            arrays.type[iparticle] = buildingBlock.atoms[j].type;
                    
            if (buildingBlock.atoms[j].body > 0)
                arrays.body[iparticle] = ibody;
                        
            unsigned int head = i * nparticlesperbuildingblock;
            for (unsigned int j = 0; j < nbondsperbuildingblock; j++)
                {
                unsigned int particlei = head + buildingBlock.bonds[j].localidxi;
                unsigned int particlej = head + buildingBlock.bonds[j].localidxj;
                    
                sysdef->getBondData()->addBond(Bond(0, particlei, particlej));
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
    
    shared_ptr<RigidData> rdata = sysdef->getRigidData();
    // Initialize rigid bodies
    rdata->initializeData();
    
    shared_ptr<ParticleSelector> selector_all(new ParticleSelectorTag(sysdef, 0, pdata->getN()-1));
    shared_ptr<ParticleGroup> group_all(new ParticleGroup(sysdef, selector_all));
    
    Scalar deltaT = Scalar(0.005);
    shared_ptr<TwoStepBDNVTRigid> two_step_bdnvt = bdup_creator(sysdef, group_all, temperature, 453034);
        
    shared_ptr<IntegratorTwoStep> bdnvt_up(new IntegratorTwoStep(sysdef, deltaT));
    bdnvt_up->addIntegrationMethod(two_step_bdnvt);

    shared_ptr<NeighborList> nlist(new NeighborList(sysdef, Scalar(2.5), Scalar(0.8)));
    shared_ptr<PotentialPairLJ> fc(new PotentialPairLJ(sysdef, nlist));
    fc->setRcut(0, 0, Scalar(1.122));
    fc->setRcut(0, 1, Scalar(1.122));
    fc->setRcut(1, 1, Scalar(1.122));
    
    // setup some values for alpha and sigma
    Scalar epsilon = Scalar(1.0);
    Scalar sigma = Scalar(1.0);
    Scalar alpha = Scalar(1.0);
    Scalar lj1 = Scalar(4.0) * epsilon * pow(sigma, Scalar(12.0));
    Scalar lj2 = alpha * Scalar(4.0) * epsilon * pow(sigma, Scalar(6.0));
    
    // specify the force parameters
    fc->setParams(0,0,make_scalar2(lj1,lj2));
    fc->setParams(0,1,make_scalar2(lj1,lj2));
    fc->setParams(1,1,make_scalar2(lj1,lj2));
    
    bdnvt_up->addForceCompute(fc);
      
    unsigned int nrigid_dof, nnonrigid_dof;
    Scalar current_temp;
    
    unsigned int start_step = 0;
    unsigned int steps = 5000;
    unsigned int sampling = 100;
    unsigned int averaging_delay = 1000;
    

    // CALLING SET RV
    rdata->setRV(true);       
        
    nrigid_dof = rdata->getNumDOF();
    nnonrigid_dof = 3 * (N - body_size * nbodies);
        
    // Production: turn on LJ interactions between rods
    fc->setRcut(1, 1, Scalar(2.5));
    
    bdnvt_up->prepRun(0);

 
    for (unsigned int i = start_step; i <= start_step + steps; i++)
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
            if (i > averaging_delay)
                AvgT += current_temp;
            
            pdata->release();
            
            }
        }

    AvgT /= Scalar((steps-averaging_delay)/sampling);    
    //Test to see if the temperature has equilibrated to where its been set.        
    MY_BOOST_CHECK_CLOSE(AvgT, 1.4, 2.5);    
    }

#ifdef ENABLE_CUDA
//! TwoStepBDNVTRigidGPU factory for the unit tests
shared_ptr<TwoStepBDNVTRigid> gpu_bdnvt_creator(shared_ptr<SystemDefinition> sysdef, shared_ptr<ParticleGroup> group, Scalar T, unsigned int seed)
    {
    shared_ptr<VariantConst> T_variant(new VariantConst(T));
    return shared_ptr<TwoStepBDNVTRigid>(new TwoStepBDNVTRigidGPU(sysdef, group, T_variant, seed, false));
    }
#endif

#ifdef ENABLE_CUDA

//! Test of Rigid Rods
BOOST_AUTO_TEST_CASE( BDRigidGPU_rod_tests )
    {
    bdnvtup_creator bdnvt_creator_gpu = bind(gpu_bdnvt_creator, _1, _2, _3, _4);
    bd_updater_lj_tests(bdnvt_creator_gpu, boost::shared_ptr<ExecutionConfiguration>(new ExecutionConfiguration(ExecutionConfiguration::GPU)));
    }
#endif

#ifdef WIN32
#pragma warning( pop )
#endif

