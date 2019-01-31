// Copyright (c) 2009-2019 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.


// this include is necessary to get MPI included before anything else to support intel MPI
#include "hoomd/ExecutionConfiguration.h"

#include <iostream>

#include <functional>

#include "hoomd/md/FIREEnergyMinimizer.h"

#ifdef ENABLE_CUDA
#include "hoomd/md/FIREEnergyMinimizerGPU.h"
#include "hoomd/md/TwoStepNVEGPU.h"
#endif

#include "hoomd/md/AllPairPotentials.h"
#include "hoomd/md/NeighborListTree.h"
#include "hoomd/ComputeThermo.h"
#include "hoomd/md/TwoStepNVE.h"

#include <math.h>

using namespace std;


#include "hoomd/test/upp11_config.h"
HOOMD_UP_MAIN();

//! Typedef'd FIREEnergyMinimizer class factory
typedef std::function<std::shared_ptr<FIREEnergyMinimizer> (std::shared_ptr<SystemDefinition> sysdef, Scalar dT)> fire_creator;
typedef std::function<std::shared_ptr<TwoStepNVE> (std::shared_ptr<SystemDefinition> sysdef, std::shared_ptr<ParticleGroup> group)> nve_creator;

//! FIREEnergyMinimizer creator
std::shared_ptr<FIREEnergyMinimizer> base_class_fire_creator(std::shared_ptr<SystemDefinition> sysdef, Scalar dt)
    {
    return std::shared_ptr<FIREEnergyMinimizer>(new FIREEnergyMinimizer(sysdef, dt));
    }

//! TwoStepNVE factory for the unit tests
std::shared_ptr<TwoStepNVE> base_class_nve_creator(std::shared_ptr<SystemDefinition> sysdef, std::shared_ptr<ParticleGroup> group)
    {
    return std::shared_ptr<TwoStepNVE>(new TwoStepNVE(sysdef, group));
    }

#ifdef ENABLE_CUDA
//! TwoStepNVEGPU factory for the unit tests
std::shared_ptr<TwoStepNVE> gpu_nve_creator(std::shared_ptr<SystemDefinition> sysdef, std::shared_ptr<ParticleGroup> group)
    {
    return std::shared_ptr<TwoStepNVE>(new TwoStepNVEGPU(sysdef, group));
    }

//! FIREEnergyMinimizerGPU creator
std::shared_ptr<FIREEnergyMinimizer> gpu_fire_creator(std::shared_ptr<SystemDefinition> sysdef, Scalar dt)
    {
    return std::shared_ptr<FIREEnergyMinimizer>(new FIREEnergyMinimizerGPU(sysdef, dt));
    }
#endif


void randpts(vector<Scalar>& x, vector<Scalar>& y, vector<Scalar>& z, unsigned int N, Scalar box, Scalar rmin)
    {
    unsigned int i = 0;
    Scalar cut = rmin*rmin;

    while(i < N)
        {
        Scalar xi = rand()/(Scalar(RAND_MAX)) * box - 0.5 * box;
        Scalar yi = rand()/(Scalar(RAND_MAX)) * box - 0.5 * box;
        Scalar zi = rand()/(Scalar(RAND_MAX)) * box - 0.5 * box;

        int overlap = 0;
        for(unsigned int j=0; j<i; j++)
            {
            Scalar dx = xi - x[j];
            dx -= box * rint ( dx / box);
            Scalar dy = yi - y[j];
            dy -= box * rint ( dy / box);
            Scalar dz = zi - z[j];
            dz -= box * rint ( dz / box);
            Scalar rsq = dx*dx + dy*dy + dz*dz;
            if(rsq < cut)
                {
                overlap = 1;
                break;
                }
            }

        if(!overlap)
            {
            x[i] = xi;
            y[i] = yi;
            z[i] = zi;
            i++;
            }
        }
    }

double x_blj [] = {
-2.93783, 2.43118, -2.11152,
1.12526, 0.591193, 2.73596,
1.68581, 1.3712, 2.38201,
1.34308, -0.818018, -0.329553,
2.18825, -1.20129, 2.24475,
2.97397, -1.12544, -2.93335,
-2.28572, -0.43173, 2.51326,
2.18046, -0.847423, 0.243442,
2.01769, -1.6332, -0.512097,
1.90327, -2.65715, 1.96657,
-1.47332, -0.828616, -2.93905,
1.61233, 2.72168, -1.35492,
2.76166, 0.829623, 2.12365,
-0.49171, -0.399128, 0.401507,
-0.0957744, 0.528925, -0.260215,
-2.44636, 0.42354, -0.554317,
-0.32345, -2.9129, -1.98104,
-2.74642, 0.0142028, 1.13395,
-2.4737, 1.64851, -2.7288,
0.118203, 0.19057, -1.84057,
-0.996886, -2.97163, -2.90115,
0.0202117, 1.41245, -1.79951,
-2.8445, -2.48133, 1.28545,
1.46052, -0.338488, -2.14404,
0.0105135, 1.57872, -0.738221,
2.45553, 2.79862, -2.8356,
2.16427, 1.77394, -1.45113,
2.6297, -0.771198, -0.731814,
-0.565703, -1.96088, -1.352,
0.50169, -1.3073, 0.266587,
0.171501, 0.819745, -2.67406,
-1.63009, -1.48273, 1.49457,
-1.4022, 0.570543, -0.676188,
-0.254007, -0.809361, -1.31268,
-2.41836, 2.93192, -1.20549,
1.40474, -1.84894, 2.3983,
-1.00818, -0.0746554, -1.60277,
0.504458, 2.07894, 0.0852257,
-2.1506, 2.8929, 0.709505,
1.06261, -1.00655, 1.93092,
1.25547, -2.66289, 2.98677,
-1.75129, -0.829839, -1.87532,
2.78906, -0.176533, 2.7203,
1.50141, -1.62425, 0.420163,
1.75023, 0.736549, 0.253628,
-1.11172, 1.56711, -0.412517,
0.267144, -0.463353, -0.415419,
0.559279, -1.0004, -2.00577,
-1.67905, 1.28344, 0.852698,
1.41409, -0.0260375, 1.67606,
2.17422, 0.684098, -2.94604,
-0.530817, 0.908144, 1.83158,
1.89519, -2.60728, -2.11191,
-0.846385, 0.878043, -2.27423,
2.05121, -2.78021, -0.362579,
-2.9084, -1.48806, -0.129955,
-2.85202, -0.896483, 1.67016,
-1.133, 1.66653, -1.40534,
-2.94412, 0.657862, -2.22749,
-1.82906, -0.436751, 1.48202,
-1.87107, 0.926953, -2.22953,
-1.43352, 1.44928, 1.85823,
0.468959, 0.339916, 1.89037,
0.78693, 2.98253, 2.21983,
-2.32038, 2.33676, 1.593,
-2.62484, 1.79534, 0.634404,
-0.371511, -2.13419, 2.88128,
-1.34461, -2.16457, 2.5065,
-0.461189, -2.65169, -0.495375,
-0.0702259, -1.10621, -2.91872,
1.22265, -1.80406, -2.22713,
0.495092, -0.172305, -2.77125,
-0.550732, -0.218605, -2.47408,
2.61424, 2.35426, -0.0525821,
-1.27344, -1.2388, -0.956124,
0.14325, -1.52428, 1.2276,
1.85094, 0.849848, 1.47776,
-1.00867, -2.86448, 0.442076,
1.68472, -0.295716, 2.69833,
-2.76346, 1.9198, -1.12046,
2.57045, 2.69979, -1.13589,
0.555653, -2.18459, -0.273156,
-1.14185, -0.21081, 2.13253,
1.05182, 0.166293, -0.342186,
-1.15381, 0.375327, 1.26577,
2.01321, -0.880184, 1.26248,
-2.41605, -0.167059, -2.48278,
1.69888, 2.46122, 2.6077,
1.2232, -0.351493, 0.694047,
-1.64976, 2.31037, -2.69082,
0.942093, -2.83303, -2.03937,
-1.04396, 0.695922, 0.306374,
2.31164, -1.74298, -1.74883,
1.80264, 2.35857, 1.40427,
-2.0496, 2.02945, -1.79515,
2.77201, 2.77251, 1.95969,
0.342246, 1.42097, 1.17912,
1.07949, 2.90888, -0.451025,
1.33493, 1.55264, 0.958572,
-1.57837, 1.37597, 2.8674,
2.61912, 1.72204, 1.51797,
2.26794, 0.000775846, 0.783223,
2.47228, 1.64814, -2.40953,
-0.377457, 2.44633, -1.14871,
0.176026, -1.99759, -2.13526,
1.95313, -0.123408, -1.26195,
-2.69768, -2.36294, -2.87164,
2.87066, -2.62566, -1.91164,
1.18596, 1.79446, -0.658579,
2.2897, -2.14013, -2.95544,
2.18185, 1.49389, -0.403087,
1.56015, 2.41208, -2.37927,
0.990482, -1.06264, 2.9911,
0.292718, -2.98649, 0.296703,
-0.46849, 2.00453, 1.88049,
-2.99257, -0.0586179, -1.42608,
-2.5125, 2.65686, 2.78833,
-1.20169, -2.00866, -0.288253,
0.613859, 2.39919, -1.19455,
1.57302, 2.36801, 0.335058,
2.91973, -2.21754, -0.912482,
2.31347, 1.518, 0.576975,
2.82649, 0.930881, -1.14058,
0.650695, 0.837925, -1.06041,
-0.1734, 0.587371, 0.908956,
-2.04864, -2.04138, 0.692118,
0.278903, -2.88524, -2.81471,
-0.670093, 2.03739, -2.40545,
0.987121, 2.00136, 2.03404,
-0.428259, 2.96165, 2.2158,
-2.53766, 0.979667, 1.35922,
1.1066, -1.92851, 1.42731,
-2.79859, -1.0197, -1.92269,
-2.30045, -2.48222, 2.14238,
0.297804, 1.26451, 2.31177,
-0.707121, -0.905051, 1.3186,
0.676925, 1.04297, 0.0760722,
-2.3039, -1.90151, -1.60472,
-1.84456, -1.6931, -2.52636,
2.74169, -1.60591, 0.797613,
2.66374, 1.93304, 2.57328,
-2.82345, -0.352003, 0.0973264,
0.395908, -1.81283, 2.24452,
1.95741, -1.16891, -2.68156,
0.944032, 0.627509, 0.989534,
2.14857, -2.39009, 1.0552,
-0.30898, 0.235914, 2.58949,
1.32985, -2.69661, 0.511045,
-1.03852, -2.21755, -2.26323,
2.82422, 2.56861, 0.929538,
-0.590126, 1.45045, 2.7637,
-1.30585, 2.73375, -1.73275,
-0.438396, 0.749269, -1.15658,
-1.48006, -0.558668, 0.565043,
0.670178, -1.37956, -0.945792,
1.12256, 0.643856, -2.09556,
2.39953, 0.329059, -0.461881,
-1.92546, -1.35391, -0.131397,
-2.20042, 1.2813, -0.0785904,
-2.12804, -2.59019, -0.351315,
0.152696, 2.20495, 2.64721,
-0.865797, -1.23327, 2.30527,
-2.4676, 0.657711, 2.89547,
-2.03388, 1.17292, -1.20365,
1.13719, 1.62732, -1.74445,
2.97389, 0.728505, 0.307283,
-0.677584, 1.44842, 0.996511,
-1.75734, -0.341187, -0.41066,
2.10412, 0.72415, -1.91222,
-1.28628, 2.2951, 1.28564,
-0.214624, 2.39373, 0.939068,
0.327753, -2.54851, -1.20438,
0.9319, 1.65968, -2.82096,
1.56851, -1.12431, -1.34342,
2.91593, -2.55448, 0.268307,
-0.260879, -2.16396, 0.446135,
-1.48442, 2.18723, 0.30242,
0.217123, -2.62867, 1.30373,
-1.45877, -2.39588, -1.16702,
-2.00501, 0.145022, -1.4486,
-1.58911, 2.70197, 2.2773,
-2.347, -0.94721, -1.03615,
-2.39421, 2.40017, -0.195969,
-1.56637, -2.57549, 1.37687,
0.06305, -0.641461, 2.2498,
-2.39636, -1.0145, 0.761698,
1.65284, 0.80109, -0.95679,
-2.36283, 1.51304, 2.25911,
-1.48266, 0.235851, -2.97032,
-0.304155, -1.49552, -0.412279,
-1.17218, -1.55037, 0.570874,
1.33918, -2.28237, -1.27012,
2.43904, -0.07509, 1.7717,
0.882913, -0.22775, -1.29087,
0.262192, -0.461287, 1.11276,
2.50194, -0.237431, -2.35039,
-0.647379, -2.07323, 1.57034,
-0.448339, 2.30056, -0.0599073,
-1.31599, 2.60675, -0.693989,
0.802989, 2.51759, 1.15523,
2.79683, -2.01046, 2.10352,
-2.08517, -2.73006, -2.17432,
-0.715256, -0.253906, -0.58492,
-0.78755, -1.2594, -2.18874,
-1.88423, 0.513921, 2.07589,
0.393033, 2.30513, -2.27136,
-2.30585, -1.47812, 2.45097,
-2.01704, 0.369488, 0.52233,
-2.35289, -1.84459, -0.690954,
2.86038, 1.15677, -2.9749,
-0.694855, -0.517146, 2.75492,
2.69181, 0.604646, 1.25467,
-1.06447, -1.6233, -2.90962,
-1.93729, -2.69748, 2.87021,
2.25638, 2.60099, -1.94901,
0.721559, -2.07522, 0.658676,
-1.34054, -1.65729, -1.67143,
0.920017, -1.07667, 1.10233,
-0.699839, 2.96362, 1.33565,
-1.94973, 1.92222, -0.791896,
1.96329, 2.37188, -0.5568,
-1.06515, -1.03239, -0.170532,
-0.906606, 2.2557, 2.66,
0.19866, 2.50715, 1.7795,
1.44341, 1.5144, 0.0877154,
-2.26384, -0.972874, -2.67691,
0.902785, 2.54189, 2.95169,
0.0780872, 0.0489579, -1.01193,
1.93291, 0.38986, 2.28184,
1.8027, -0.126915, 0.0466883,
2.90314, -1.77622, -2.36521,
1.08208, 1.09343, 1.7228,
-2.52048, -1.69058, 1.28589,
2.12397, -2.51534, -1.21637,
0.389791, -0.0151411, 0.339518,
2.86584, -0.687583, 0.877299,
-2.96415, 1.56356, -0.307448,
-1.15103, 0.70601, 2.38851,
2.22005, 2.9927, 0.48915,
-0.416409, 0.00277854, 1.76995,
-0.307789, -1.23003, 0.476881,
1.40615, -2.2351, -0.296636,
-1.65568, -0.0685609, -2.17245,
0.90926, -0.267006, 2.4452,
0.684146, -1.99342, -2.9539,
-1.7105, -0.911605, 2.19462,
-1.32297, 0.709174, -1.52713,
-2.94637, -2.92223, -0.468926,
1.16225, -2.77801, 1.46172,
2.2096, -2.10778, 0.218746,
-2.79252, 1.51737, -1.95456,
-0.18541, 1.46031, 0.254424,
2.27254, -0.821918, -1.77266,
-1.41072, 1.56289, -2.30567,
0.0353821, 1.61798, -2.6834,
1.64871, 1.31223, -2.38637,
1.95249, -1.7397, 1.6701,
-2.70093, 0.229611, 2.19764,
2.976, -1.42114, -1.22056,
0.139363, 2.78531, -0.483974
};

//Calculated from Matlab file.  Changes to the way FIRE works may require recalculation of these values.
double x_two_lj [] = {
2,2,2,2,2,1.9999,1.9999,1.9998,1.9997,1.9996,1.9994,1.9992,1.999,1.9986,1.9982,1.9977,1.9971,1.9963,1.9953,1.994,1.9925,
1.9907,1.9883,1.9855,1.982,1.9778,1.9729,1.9676,1.9618,1.9554,1.9486,1.9411,1.9332,1.9246,1.9155,1.9057,1.8953,1.8843,
1.8726,1.8602,1.847,1.8331,1.8183,1.8027,1.7862,1.7687,1.7502,1.7305,1.7097,1.6875,1.664,1.6389,1.6122,1.5835,1.5528,
1.5196,1.4839,1.445,1.4025,1.356,1.3046,1.2476,1.1847,1.1167,1.1168,1.1171,1.1176,1.1183,1.1192,1.1202,1.1213,1.1227,
1.1227};

//! Compares the output from one NVEUpdater to another
void fire_smallsystem_test(fire_creator fire_creator1, nve_creator nve_creator1, std::shared_ptr<ExecutionConfiguration> exec_conf)
    {
    const unsigned int N = 260;
    Scalar rho(Scalar(1.2));
    Scalar L = Scalar(pow((double)(N/rho), 1.0/3.0));
    std::shared_ptr<SystemDefinition> sysdef(new SystemDefinition(N, BoxDim(L, L, L), 2, 0, 0, 0, 0, exec_conf));
    std::shared_ptr<ParticleData> pdata = sysdef->getParticleData();

    // enable the energy computation
    PDataFlags flags;
    flags[pdata_flag::potential_energy] = 1;
    pdata->setFlags(flags);

    for (unsigned int i=0; i<N; i++)
        {
        Scalar3 pos = make_scalar3(x_blj[i*3 + 0],x_blj[i*3 + 1],x_blj[i*3 + 2]);
        pdata->setPosition(i,pos);
        if (i<(unsigned int)N*0.8)
            pdata->setType(i,0);
        else
            pdata->setType(i,1);
        }

    std::shared_ptr<ParticleSelector> selector_all(new ParticleSelectorTag(sysdef, 0, pdata->getN()-1));
    std::shared_ptr<ParticleGroup> group_all(new ParticleGroup(sysdef, selector_all));

    std::shared_ptr<NeighborListTree> nlist(new NeighborListTree(sysdef, Scalar(2.5), Scalar(0.3)));
    std::shared_ptr<PotentialPairLJ> fc(new PotentialPairLJ(sysdef, nlist));
    fc->setRcut(0, 0, Scalar(2.5));
    fc->setRcut(0, 1, Scalar(2.5));
    fc->setRcut(1, 1, Scalar(2.5));

   // setup some values for alpha and sigma
    Scalar epsilon00 = Scalar(1.0);
    Scalar sigma00 = Scalar(1.0);
    Scalar epsilon01 = Scalar(1.5);
    Scalar sigma01 = Scalar(0.8);
    Scalar epsilon11 = Scalar(0.5);
    Scalar sigma11 = Scalar(0.88);
    Scalar alpha = Scalar(1.0);
    Scalar lj001 = Scalar(4.0) * epsilon00 * pow(sigma00,Scalar(12.0));
    Scalar lj002 = alpha * Scalar(4.0) * epsilon00 * pow(sigma00,Scalar(6.0));
    Scalar lj011 = Scalar(4.0) * epsilon01 * pow(sigma01,Scalar(12.0));
    Scalar lj012 = alpha * Scalar(4.0) * epsilon01 * pow(sigma01,Scalar(6.0));
    Scalar lj111 = Scalar(4.0) * epsilon11 * pow(sigma11,Scalar(12.0));
    Scalar lj112 = alpha * Scalar(4.0) * epsilon11 * pow(sigma11,Scalar(6.0));

    // specify the force parameters
    fc->setParams(0,0,make_scalar2(lj001,lj002));
    fc->setRcut(0,0,2.5);
    fc->setParams(0,1,make_scalar2(lj011,lj012));
    fc->setRcut(0,1,2.5);
    fc->setParams(1,1,make_scalar2(lj111,lj112));
    fc->setRcut(1,1,2.5);
    fc->setShiftMode(PotentialPairLJ::shift);

    std::shared_ptr<TwoStepNVE> nve = nve_creator1(sysdef,group_all);
    std::shared_ptr<FIREEnergyMinimizer> fire = fire_creator1(sysdef, Scalar(0.05));
    fire->addIntegrationMethod(nve);
    fire->setFtol(5.0);
    fire->addForceCompute(fc);
    fire->setMinSteps(10);
    fire->prepRun(0);

    int max_step = 1000;
    for (int i = 1; i<=max_step; i++) {
        fire->update(i);
        if (fire->hasConverged())  {break;}
        }

    ComputeThermo ct(sysdef, group_all);
    ct.compute(max_step);
    MY_CHECK_CLOSE(ct.getPotentialEnergy()/Scalar(N), -7.75, (0.01/7.75)*100);

    fire->reset();

    for (unsigned int i=0; i<N; i++)
        {
        Scalar3 pos = make_scalar3(x_blj[i*3 + 0],x_blj[i*3 + 1],x_blj[i*3 + 2]);
        pdata->setPosition(i,pos);
        }

    for (int i = max_step+1; i<=2*max_step; i++)
        fire->update(i);

    ct.compute(max_step+1);
    MY_CHECK_CLOSE(ct.getPotentialEnergy()/Scalar(N), -7.75, (0.01/7.75)*100);

    //cerr << fire->computePotentialEnergy(max_step)/Scalar(N) << endl;

    }

void fire_twoparticle_test(fire_creator fire_creator1, nve_creator nve_creator1, std::shared_ptr<ExecutionConfiguration> exec_conf)
    {
    const unsigned int N = 2;
    //Scalar rho(1.2);
    Scalar L = Scalar(20);
    std::shared_ptr<SystemDefinition> sysdef(new SystemDefinition(N, BoxDim(L, L, L), 1, 0, 0, 0, 0, exec_conf));
    std::shared_ptr<ParticleData> pdata = sysdef->getParticleData();

    // enable the energy computation
    PDataFlags flags;
    flags[pdata_flag::potential_energy] = 1;
    pdata->setFlags(flags);

    pdata->setPosition(0,make_scalar3(0.0,0.0,0.0));
    pdata->setType(0,0);
    pdata->setPosition(1,make_scalar3(2.0,0.0,0.0));
    pdata->setType(1,0);

    std::shared_ptr<ParticleSelector> selector_one(new ParticleSelectorTag(sysdef, 1, 1));
    std::shared_ptr<ParticleGroup> group_one(new ParticleGroup(sysdef, selector_one));

    std::shared_ptr<NeighborListTree> nlist(new NeighborListTree(sysdef, Scalar(3.0), Scalar(0.3)));
    std::shared_ptr<PotentialPairLJ> fc(new PotentialPairLJ(sysdef, nlist));
    fc->setRcut(0, 0, Scalar(3.0));


   // setup some values for alpha and sigma
    Scalar epsilon00 = Scalar(1.0);
    Scalar sigma00 = Scalar(1.0);
    Scalar alpha = Scalar(1.0);
    Scalar lj001 = Scalar(4.0) * epsilon00 * pow(sigma00,Scalar(12.0));
    Scalar lj002 = alpha * Scalar(4.0) * epsilon00 * pow(sigma00,Scalar(6.0));

    // specify the force parameters
    fc->setParams(0,0,make_scalar2(lj001,lj002));
    fc->setRcut(0,0,3.0);
    fc->setShiftMode(PotentialPairLJ::shift);

    std::shared_ptr<TwoStepNVE> nve = nve_creator1(sysdef,group_one);
    std::shared_ptr<FIREEnergyMinimizer> fire = fire_creator1(sysdef, Scalar(0.05));
    fire->addIntegrationMethod(nve);

    fire->addForceCompute(fc);
    fire->setFtol(Scalar(5.0));
    fire->setEtol(Scalar(1e-7));
    fire->setMinSteps(10);
    fire->prepRun(0);

    int max_step = 100;
    Scalar diff = Scalar(0.0);

    for (int i = 1; i<=max_step; i++)
        {
        fire->update(i);
        if (fire->hasConverged()) { break;}
        Scalar posx = pdata->getPosition(1).x;
        diff += (posx- Scalar(x_two_lj[i]))*(posx- Scalar(x_two_lj[i]));

        //MY_CHECK_CLOSE(arrays.x[1], x_two_lj[i], 0.01);   // Trajectory overkill test!
        }

    MY_CHECK_SMALL(diff, 0.001);

    }

//! Sees if a single particle's trajectory is being calculated correctly
UP_TEST( FIREEnergyMinimizer_twoparticle_test )
    {
    fire_twoparticle_test(base_class_fire_creator, base_class_nve_creator, std::shared_ptr<ExecutionConfiguration>(new ExecutionConfiguration(ExecutionConfiguration::CPU)));
    }

//! Compares the output of FIREEnergyMinimizer to the conjugate gradient method from LAMMPS
UP_TEST( FIREEnergyMinimizer_smallsystem_test )
    {
    fire_smallsystem_test(base_class_fire_creator, base_class_nve_creator, std::shared_ptr<ExecutionConfiguration>(new ExecutionConfiguration(ExecutionConfiguration::CPU)));
    }

#ifdef ENABLE_CUDA
//! Sees if a single particle's trajectory is being calculated correctly
UP_TEST( FIREEnergyMinimizerGPU_twoparticle_test )
    {
    fire_twoparticle_test(gpu_fire_creator, gpu_nve_creator, std::shared_ptr<ExecutionConfiguration>(new ExecutionConfiguration(ExecutionConfiguration::GPU)));
    }

//! Compares the output of FIREEnergyMinimizerGPU to the conjugate gradient method from LAMMPS
UP_TEST( FIREEnergyMinimizerGPU_smallsystem_test )
    {
    fire_smallsystem_test(gpu_fire_creator, gpu_nve_creator, std::shared_ptr<ExecutionConfiguration>(new ExecutionConfiguration(ExecutionConfiguration::GPU)));
    }
#endif
