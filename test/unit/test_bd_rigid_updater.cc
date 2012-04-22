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

* All publications and presentations based on HOOMD-blue, including any reports
or published results obtained, in whole or in part, with HOOMD-blue, will
acknowledge its use according to the terms posted at the time of submission on:
http://codeblue.umich.edu/hoomd-blue/citations.html

* Any electronic documents citing HOOMD-Blue will link to the HOOMD-Blue website:
http://codeblue.umich.edu/hoomd-blue/

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

template < class BDRigid >
void bd_updater_lj_tests(boost::shared_ptr<ExecutionConfiguration> exec_conf)
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
    Scalar3 lo = box.getLo();
    Scalar3 hi = box.getHi();
    
    // setup a simple initial state
    unsigned int ibody = 0;
    unsigned int iparticle = 0;
    Scalar x0 = lo.x + 0.01;
    Scalar y0 = lo.y + 0.01;
    Scalar z0 = lo.z + 0.01;
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
    
    
    {
    ArrayHandle<Scalar4> h_pos(pdata->getPositions(), access_location::host, access_mode::readwrite);
    ArrayHandle<Scalar4> h_vel(pdata->getVelocities(), access_location::host, access_mode::readwrite);
    ArrayHandle<unsigned int> h_body(pdata->getBodies(), access_location::host, access_mode::readwrite);
    
    // initialize bodies in a cubic lattice with some velocity profile
    for (unsigned int i = 0; i < nbodies; i++)
        {
        for (unsigned int j = 0; j < nparticlesperbuildingblock; j++)
            {
            h_pos.data[iparticle].x = x0 + buildingBlock.atoms[j].x;
            h_pos.data[iparticle].y = y0 + buildingBlock.atoms[j].y;
            h_pos.data[iparticle].z = z0 + buildingBlock.atoms[j].z;

            h_vel.data[iparticle].x = random->d();
            h_vel.data[iparticle].y = random->d();
            h_vel.data[iparticle].z = random->d();
            
            KE += Scalar(0.5) * (h_vel.data[iparticle].x*h_vel.data[iparticle].x + h_vel.data[iparticle].y*h_vel.data[iparticle].y + h_vel.data[iparticle].z*h_vel.data[iparticle].z);
            
            h_pos.data[iparticle].w = __int_as_scalar(buildingBlock.atoms[j].type);
                    
            if (buildingBlock.atoms[j].body > 0)
                h_body.data[iparticle] = ibody;
                        
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
        if (x0 + xspacing >= hi.x)
            {
            x0 = lo.x + 0.01;
            
            y0 += yspacing;
            if (y0 + yspacing >= hi.y)
                {
                y0 = lo.y + 0.01;
                
                z0 += zspacing;
                if (z0 + zspacing >= hi.z)
                    z0 = lo.z + 0.01;
                }
            }
            
        ibody++;
        }
        
    assert(iparticle == N);
    
    }
    
    shared_ptr<RigidData> rdata = sysdef->getRigidData();
    // Initialize rigid bodies
    rdata->initializeData();
    
    shared_ptr<ParticleSelector> selector_all(new ParticleSelectorTag(sysdef, 0, pdata->getN()-1));
    shared_ptr<ParticleGroup> group_all(new ParticleGroup(sysdef, selector_all));
    
    Scalar deltaT = Scalar(0.005);
    boost::shared_ptr<Variant> T_variant(new VariantConst(temperature));
    shared_ptr<TwoStepBDNVTRigid> two_step_bdnvt = shared_ptr<TwoStepBDNVTRigid>(new BDRigid(sysdef, group_all, T_variant, 453034, false));
        
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

            ArrayHandle< Scalar4 > h_vel(pdata->getVelocities(), access_location::host, access_mode::read);
            KE = Scalar(0);
            for (unsigned int j = 0; j < N; j++)
                KE += Scalar(0.5) * (h_vel.data[j].x*h_vel.data[j].x +h_vel.data[j].y*h_vel.data[j].y + h_vel.data[j].z*h_vel.data[j].z);
            PE = fc->calcEnergySum();
            
            current_temp = 2.0 * KE / (nrigid_dof + nnonrigid_dof);
            if (i > averaging_delay)
                AvgT += current_temp;
            
            }
        }

    AvgT /= Scalar((steps-averaging_delay)/sampling);    
    //Test to see if the temperature has equilibrated to where its been set. Use a wide window because the test simulation is short.
    MY_BOOST_CHECK_CLOSE(AvgT, 1.4, 2);
    }

#ifdef ENABLE_CUDA

BOOST_AUTO_TEST_CASE( BDRigidGPU_rod_tests )
    {
    bd_updater_lj_tests<TwoStepBDNVTRigidGPU>(boost::shared_ptr<ExecutionConfiguration>(new ExecutionConfiguration(ExecutionConfiguration::GPU)));
    }
#endif

BOOST_AUTO_TEST_CASE( BDRigid_rod_tests )
    {
    bd_updater_lj_tests<TwoStepBDNVTRigid>(boost::shared_ptr<ExecutionConfiguration>(new ExecutionConfiguration(ExecutionConfiguration::CPU)));
    }
