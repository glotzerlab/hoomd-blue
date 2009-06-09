#ifdef WIN32
#pragma warning( push )
#pragma warning( disable : 4103 4244 )
#endif

#include <iostream>

//! Name the boost unit test module
#define BOOST_TEST_MODULE ImproperForceTests
#include "boost_utf_configure.h"

#include <boost/test/floating_point_comparison.hpp>
#include <boost/bind.hpp>
#include <boost/function.hpp>

#include "HarmonicImproperForceCompute.h"
#include "ConstForceCompute.h"
#ifdef ENABLE_CUDA
#include "HarmonicImproperForceComputeGPU.h"
#endif

#include "Initializers.h"

using namespace std;
using namespace boost;

//! Helper macro for testing if two numbers are close
#define MY_BOOST_CHECK_CLOSE(a,b,c) BOOST_CHECK_CLOSE(a,Scalar(b),Scalar(c))
//! Helper macro for testing if a number is small
#define MY_BOOST_CHECK_SMALL(a,c) BOOST_CHECK_SMALL(a,Scalar(c))

//! Global tolerance for floating point comparisons
#ifdef SINGLE_PRECISION
const Scalar tol = Scalar(1e-1);
#else
const Scalar tol = 1e-2;
#endif

//! Typedef to make using the boost::function factory easier
typedef boost::function<shared_ptr<HarmonicImproperForceCompute>  (shared_ptr<ParticleData> pdata)> improperforce_creator;

//! Perform some simple functionality tests of any BondForceCompute
void improper_force_basic_tests(improperforce_creator tf_creator, ExecutionConfiguration exec_conf)
	{
	#ifdef CUDA
	g_gpu_error_checking = true;
	#endif
	
	/////////////////////////////////////////////////////////
	// start with the simplest possible test: 4 particles in a huge box with only one improper type !!!! NO IMPROPERS
	shared_ptr<ParticleData> pdata_4(new ParticleData(4, BoxDim(1000.0), 1, 0, 0, 0, 1, exec_conf));
	ParticleDataArrays arrays = pdata_4->acquireReadWrite();
	arrays.x[0] = Scalar(10.0); // put atom a at (10,1,2)
	arrays.y[0] = Scalar(1.0);
	arrays.z[0] = Scalar(2.0);

	arrays.x[1] = arrays.y[1] = arrays.z[1] = Scalar(1.0); // put atom b at (1,1,1)


	arrays.x[2] = Scalar(6.0); // put atom c at (6,-7,8)
	arrays.y[2] = Scalar(-7.0);
	arrays.z[2] = Scalar(8.0);

	arrays.x[3] = Scalar(9.0); // put atom d at (9,50,11)
	arrays.y[3] = Scalar(50.0);
	arrays.z[3] = Scalar(11.0);

/*
	printf(" Particle 1: x = %f  y = %f  z = %f \n", arrays.x[0], arrays.y[0], arrays.z[0]);
	printf(" Particle 2: x = %f  y = %f  z = %f \n", arrays.x[1], arrays.y[1], arrays.z[1]);      
	printf(" Particle 3: x = %f  y = %f  z = %f \n", arrays.x[2], arrays.y[2], arrays.z[2]);   
	printf(" Particle 4: x = %f  y = %f  z = %f \n", arrays.x[3], arrays.y[3], arrays.z[3]);            
	printf("\n");
*/
	pdata_4->release();

	// create the improper force compute to check
	shared_ptr<HarmonicImproperForceCompute> fc_4 = tf_creator(pdata_4);
	fc_4->setParams(0, Scalar(2.0), Scalar(1.570796)); // type=0, K=2.0,chi=pi/2

	// compute the force and check the results
	fc_4->compute(0);
	ForceDataArrays force_arrays = fc_4->acquire();

	// check that the force is correct, it should be 0 since we haven't created any impropers yet
	MY_BOOST_CHECK_SMALL(force_arrays.fx[0], tol);
	MY_BOOST_CHECK_SMALL(force_arrays.fy[0], tol);
	MY_BOOST_CHECK_SMALL(force_arrays.fz[0], tol);
	MY_BOOST_CHECK_SMALL(force_arrays.pe[0], tol);
	MY_BOOST_CHECK_SMALL(force_arrays.virial[0], tol);
	
	// add an impropers and check again
	pdata_4->getImproperData()->addImproper(Improper(0,0,1,2,3)); // add type 0 improper bewtween atoms 0-1-2-3
	fc_4->compute(1);
/*
 FORCE 1: fx = 0.024609  fy = -0.178418  fz = -0.221484 
 FORCE 2: fx = 0.108934  fy = 0.109425  fz = 0.047247 
 FORCE 3: fx = -0.092712  fy = 0.068413  fz = 0.144409 
 FORCE 4: fx = -0.040832  fy = 0.000579  fz = 0.029827 
 Energy: 1 = 0.158927  2 = 0.158927  3 = 0.158927 4 = 0.158927 

*/

	// this time there should be a force
	force_arrays = fc_4->acquire();
	MY_BOOST_CHECK_CLOSE(force_arrays.fx[0], 0.024609, tol);
	MY_BOOST_CHECK_CLOSE(force_arrays.fy[0], -0.178418, tol);
	MY_BOOST_CHECK_CLOSE(force_arrays.fz[0], -0.221484, tol);
	MY_BOOST_CHECK_CLOSE(force_arrays.pe[0], 0.158927, tol);
	MY_BOOST_CHECK_SMALL(force_arrays.virial[0], tol);

	MY_BOOST_CHECK_CLOSE(force_arrays.fx[1], 0.108934, tol);
	MY_BOOST_CHECK_CLOSE(force_arrays.fy[1], 0.109425 , tol);
	MY_BOOST_CHECK_CLOSE(force_arrays.fz[1], 0.047247, tol);
	MY_BOOST_CHECK_CLOSE(force_arrays.pe[1], 0.158927, tol);
	MY_BOOST_CHECK_SMALL(force_arrays.virial[1], tol);

	MY_BOOST_CHECK_CLOSE(force_arrays.fx[2], -0.092712, tol);
	MY_BOOST_CHECK_CLOSE(force_arrays.fy[2], 0.068413, tol);
	MY_BOOST_CHECK_CLOSE(force_arrays.fz[2], 0.144409, tol);
	MY_BOOST_CHECK_CLOSE(force_arrays.pe[2], 0.158927, tol);
	MY_BOOST_CHECK_SMALL(force_arrays.virial[2], tol);

	MY_BOOST_CHECK_CLOSE(force_arrays.fx[3], -0.040832, tol);
	MY_BOOST_CHECK_CLOSE(force_arrays.fy[3], 0.000579, tol);
	MY_BOOST_CHECK_CLOSE(force_arrays.fz[3], 0.029827, tol);
	MY_BOOST_CHECK_CLOSE(force_arrays.pe[3], 0.158927, tol);
	MY_BOOST_CHECK_SMALL(force_arrays.virial[3], tol);
	
	// rearrange the two particles in memory and see if they are properly updated
	arrays = pdata_4->acquireReadWrite();

	arrays.x[1] = Scalar(10.0); // put atom b at (10,1,2)
	arrays.y[1] = Scalar(1.0);
	arrays.z[1] = Scalar(2.0);

	arrays.x[0] = arrays.y[0] = arrays.z[0] = Scalar(1.0); // put atom a at (1,1,1)

	arrays.tag[0] = 1;
	arrays.tag[1] = 0;
	arrays.rtag[0] = 1;
	arrays.rtag[1] = 0;
	pdata_4->release();

	// notify that we made the sort
	pdata_4->notifyParticleSort();
	// recompute at the same timestep, the forces should still be updated
	fc_4->compute(1);

	force_arrays = fc_4->acquire();

	MY_BOOST_CHECK_CLOSE(force_arrays.fx[1], 0.024609, tol);
	MY_BOOST_CHECK_CLOSE(force_arrays.fy[1], -0.178418, tol);
	MY_BOOST_CHECK_CLOSE(force_arrays.fz[1], -0.221484, tol);
	MY_BOOST_CHECK_CLOSE(force_arrays.pe[1], 0.158927, tol);
	MY_BOOST_CHECK_SMALL(force_arrays.virial[1], tol);

	MY_BOOST_CHECK_CLOSE(force_arrays.fx[0], 0.108934, tol);
	MY_BOOST_CHECK_CLOSE(force_arrays.fy[0], 0.109425 , tol);
	MY_BOOST_CHECK_CLOSE(force_arrays.fz[0], 0.047247, tol);
	MY_BOOST_CHECK_CLOSE(force_arrays.pe[0], 0.158927, tol);
	MY_BOOST_CHECK_SMALL(force_arrays.virial[0], tol);
 

	////////////////////////////////////////////////////////////////////
	// now, lets do a more thorough test and include boundary conditions
	// there are way too many permutations to test here, so I will simply
	// test +x, -x, +y, -y, +z, and -z independantly
	// build a 8 particle system with particles across each boundary
	// also test more than one type of impropers
 	shared_ptr<ParticleData> pdata_8(new ParticleData(8, BoxDim(60.0, 70.0, 80.0), 1, 0, 0, 0, 2, exec_conf));

	arrays = pdata_8->acquireReadWrite();
	arrays.x[0] = Scalar(-9.6); arrays.y[0] = -9.0; arrays.z[0] = 0.0;
	arrays.x[1] =  Scalar(9.6); arrays.y[1] = 1.0; arrays.z[1] = 0.0;
	arrays.x[2] = 0; arrays.y[2] = Scalar(-19.6); arrays.z[2] = 0.0;
	arrays.x[3] = 0; arrays.y[3] = Scalar(19.6); arrays.z[3] = 10.0;
	arrays.x[4] = 0; arrays.y[4] = 0; arrays.z[4] = Scalar(-29.6);
	arrays.x[5] = 0; arrays.y[5] = 0; arrays.z[5] =  Scalar(29.6);
	arrays.x[6] = 3; arrays.y[6] = 3; arrays.z[6] =  Scalar(29.6);
	arrays.x[7] = 3; arrays.y[7] = 0; arrays.z[7] =  Scalar(31.0);
	pdata_8->release();
	
	shared_ptr<HarmonicImproperForceCompute> fc_8 = tf_creator(pdata_8);
	fc_8->setParams(0, 2.0, 1.578);
	fc_8->setParams(1, 4.0, 1.444);
	
	pdata_8->getImproperData()->addImproper(Improper(0, 0,1,2,3));
	pdata_8->getImproperData()->addImproper(Improper(1, 4,5,6,7));
	
	fc_8->compute(0);
	// check that the forces are correctly computed
	force_arrays = fc_8->acquire();
/*
 FORCE 1: fx = -0.000000  fy = 0.000000  fz = 0.275672 
 FORCE 2: fx = -0.150230  fy = 0.070010  fz = 0.148276 
 FORCE 3: fx = 0.272530  fy = -0.127004  fz = -0.599490 
 FORCE 4: fx = -0.122300  fy = 0.056994  fz = 0.175541 
 Energy: 1 = 0.412477  2 = 0.412477  3 = 0.412477 4 = 0.412477 

 Virial: 1 = -0.000001  2 = -0.000001  3 = -0.000001 4 = -0.000001 
*/

	MY_BOOST_CHECK_SMALL(force_arrays.fx[0], tol);
	MY_BOOST_CHECK_SMALL(force_arrays.fy[0], tol);
	MY_BOOST_CHECK_CLOSE(force_arrays.fz[0], 0.275672,tol);
	MY_BOOST_CHECK_CLOSE(force_arrays.pe[0], 0.412477, tol);
	MY_BOOST_CHECK_SMALL(force_arrays.virial[0], tol);

	MY_BOOST_CHECK_CLOSE(force_arrays.fx[1], -0.150230, tol);
	MY_BOOST_CHECK_CLOSE(force_arrays.fy[1], 0.070010,tol);
	MY_BOOST_CHECK_CLOSE(force_arrays.fz[1], 0.148276,tol); 
	MY_BOOST_CHECK_CLOSE(force_arrays.pe[1], 0.412477, tol);
	MY_BOOST_CHECK_SMALL(force_arrays.virial[1], tol);

	MY_BOOST_CHECK_CLOSE(force_arrays.fx[2], 0.272530,tol);
	MY_BOOST_CHECK_CLOSE(force_arrays.fy[2], -0.127004, tol);
	MY_BOOST_CHECK_CLOSE(force_arrays.fz[2], -0.599490,tol); 
	MY_BOOST_CHECK_CLOSE(force_arrays.pe[2], 0.412477, tol);
	MY_BOOST_CHECK_SMALL(force_arrays.virial[2], tol);

	MY_BOOST_CHECK_CLOSE(force_arrays.fx[3], -0.122300,tol);
	MY_BOOST_CHECK_CLOSE(force_arrays.fy[3], 0.056994, tol);
	MY_BOOST_CHECK_CLOSE(force_arrays.fz[3], 0.175541,tol); 
	MY_BOOST_CHECK_CLOSE(force_arrays.pe[3], 0.412477, tol);
	MY_BOOST_CHECK_SMALL(force_arrays.virial[3], tol);

/*
 FORCE 1: fx = -0.000000  fy = 0.000000  fz = 0.275672 
 FORCE 2: fx = -0.150230  fy = 0.070010  fz = 0.148276 
 FORCE 3: fx = 0.272530  fy = -0.127004  fz = -0.599490 
 FORCE 4: fx = -0.122300  fy = 0.056994  fz = 0.175541 
 FORCE 5: fx = -0.124166  fy = 0.124166  fz = -0.000000 
 FORCE 6: fx = -0.155688  fy = 0.155688  fz = 0.599688 
 FORCE 7: fx = -0.279854  fy = 0.279854  fz = 0.599688 
 FORCE 8: fx = 0.559709  fy = -0.559709  fz = -1.199376 
 Energy: 1 = 0.412477  2 = 0.412477  3 = 0.412477 4 = 0.412477 
 Energy: 5 = 0.208441  6 = 0.208441  7 = 0.208441 8 = 0.208441 

*/
	MY_BOOST_CHECK_CLOSE(force_arrays.fx[4], -0.124166,tol);
	MY_BOOST_CHECK_CLOSE(force_arrays.fy[4], 0.124166,tol);
	MY_BOOST_CHECK_SMALL(force_arrays.fz[4], tol);
	MY_BOOST_CHECK_CLOSE(force_arrays.pe[4], 0.208441, tol);
	MY_BOOST_CHECK_SMALL(force_arrays.virial[4], tol);

	MY_BOOST_CHECK_CLOSE(force_arrays.fx[5], -0.155688,tol);
	MY_BOOST_CHECK_CLOSE(force_arrays.fy[5], 0.155688,tol);
	MY_BOOST_CHECK_CLOSE(force_arrays.fz[5], 0.599688,tol);
	MY_BOOST_CHECK_CLOSE(force_arrays.pe[5], 0.208441, tol);
	MY_BOOST_CHECK_SMALL(force_arrays.virial[5], tol);

	MY_BOOST_CHECK_CLOSE(force_arrays.fx[6], -0.279854,tol);
	MY_BOOST_CHECK_CLOSE(force_arrays.fy[6], 0.279854,tol);
	MY_BOOST_CHECK_CLOSE(force_arrays.fz[6], 0.599688,tol);
	MY_BOOST_CHECK_CLOSE(force_arrays.pe[6], 0.208441, tol);
	MY_BOOST_CHECK_SMALL(force_arrays.virial[6], tol);

	MY_BOOST_CHECK_CLOSE(force_arrays.fx[7], 0.559709,tol);
	MY_BOOST_CHECK_CLOSE(force_arrays.fy[7], -0.559709,tol);
	MY_BOOST_CHECK_CLOSE(force_arrays.fz[7], -1.199376,tol);
	MY_BOOST_CHECK_CLOSE(force_arrays.pe[7], 0.208441, tol);
	MY_BOOST_CHECK_SMALL(force_arrays.virial[7], tol);


	// one more test: this one will test two things:
	// 1) That the forces are computed correctly even if the particles are rearranged in memory
	// and 2) That two forces can add to the same particle
	shared_ptr<ParticleData> pdata_5(new ParticleData(5, BoxDim(100.0, 100.0, 100.0), 1, 0, 0, 0, 1, exec_conf));
	arrays = pdata_5->acquireReadWrite();

	arrays.x[0] = Scalar(-9.6); arrays.y[0] = -9.0; arrays.z[0] = 0.0;
	arrays.x[1] =  Scalar(9.6); arrays.y[1] = 1.0; arrays.z[1] = 0.0;
	arrays.x[2] = 0; arrays.y[2] = Scalar(-19.6); arrays.z[2] = 0.0;
	arrays.x[3] = 0; arrays.y[3] = Scalar(19.6); arrays.z[3] = 10.0;
	arrays.x[4] = 0; arrays.y[4] = 0; arrays.z[4] = Scalar(-29.6);

	arrays.tag[0] = 2;
	arrays.tag[1] = 3;
	arrays.tag[2] = 0;
	arrays.tag[3] = 1;
	arrays.rtag[arrays.tag[0]] = 0;
	arrays.rtag[arrays.tag[1]] = 1;
	arrays.rtag[arrays.tag[2]] = 2;
	arrays.rtag[arrays.tag[3]] = 3;
	pdata_5->release();

	// build the improper force compute and try it out
	shared_ptr<HarmonicImproperForceCompute> fc_5 = tf_creator(pdata_5);
	fc_5->setParams(0, 5.0, 1.33333);

	pdata_5->getImproperData()->addImproper(Improper(0, 0,1,2,3));
	pdata_5->getImproperData()->addImproper(Improper(0, 1,2,3,4));

	fc_5->compute(0);
	force_arrays = fc_5->acquire();

	MY_BOOST_CHECK_CLOSE(force_arrays.fx[0], 0.304428, tol);
	MY_BOOST_CHECK_CLOSE(force_arrays.fy[0], 0.014121,tol);
	MY_BOOST_CHECK_CLOSE(force_arrays.fz[0], -0.504956,tol);
	MY_BOOST_CHECK_CLOSE(force_arrays.pe[0], 1.285859, tol);
	MY_BOOST_CHECK_SMALL(force_arrays.virial[0], tol);

	MY_BOOST_CHECK_CLOSE(force_arrays.fx[1], -0.006890, tol);
	MY_BOOST_CHECK_CLOSE(force_arrays.fy[1], 0.013229,tol);
	MY_BOOST_CHECK_CLOSE(force_arrays.fz[1], -0.274493,tol);
	MY_BOOST_CHECK_CLOSE(force_arrays.pe[1], 1.285859, tol);
	MY_BOOST_CHECK_SMALL(force_arrays.virial[1], tol);

/*
 FORCE 1: fx = 0.304428  fy = 0.014121  fz = -0.504956 
 FORCE 2: fx = -0.006890  fy = 0.013229  fz = -0.274493 
 FORCE 3: fx = -0.175244  fy = -0.158713  fz = 0.622154 
 FORCE 4: fx = -0.035541  fy = -0.035200  fz = 0.134787 
 FORCE 5: fx = -0.086752  fy = 0.166564  fz = 0.022509 
 Energy: 1 = 1.285859  2 = 1.285859  3 = 0.888413 4 = 1.285859 

 Energy: 5 = 0.397447 

*/
	MY_BOOST_CHECK_CLOSE(force_arrays.fx[2], -0.175244, tol);
	MY_BOOST_CHECK_CLOSE(force_arrays.fy[2], -0.158713,tol);
	MY_BOOST_CHECK_CLOSE(force_arrays.fz[2], 0.622154,tol);
	MY_BOOST_CHECK_CLOSE(force_arrays.pe[2], 0.888413, tol);
	MY_BOOST_CHECK_SMALL(force_arrays.virial[2], tol);

	MY_BOOST_CHECK_CLOSE(force_arrays.fx[3], -0.035541, tol);
	MY_BOOST_CHECK_CLOSE(force_arrays.fy[3], -0.035200,tol);
	MY_BOOST_CHECK_CLOSE(force_arrays.fz[3], 0.134787,tol);
	MY_BOOST_CHECK_CLOSE(force_arrays.pe[3], 1.285859, tol);
	MY_BOOST_CHECK_SMALL(force_arrays.virial[3], tol);

	MY_BOOST_CHECK_CLOSE(force_arrays.fx[4], -0.086752, tol);
	MY_BOOST_CHECK_CLOSE(force_arrays.fy[4], 0.166564,tol);
	MY_BOOST_CHECK_CLOSE(force_arrays.fz[4], 0.022509,tol);
	MY_BOOST_CHECK_CLOSE(force_arrays.pe[4], 0.397447, tol);
	MY_BOOST_CHECK_SMALL(force_arrays.virial[4], tol);


	}





//! Compares the output of two HarmonicImproperForceComputes
void improper_force_comparison_tests(improperforce_creator tf_creator1, improperforce_creator tf_creator2, ExecutionConfiguration exec_conf)
	{
	#ifdef CUDA
	g_gpu_error_checking = true;
	#endif
	
	// INTERESTING NOTE: the code will depending on the number of ramdom particles
	// even 1000 will make the code blow up, 500 is used for safety... hope it works!
	const unsigned int N = 500;
	
	// create a particle system to sum forces on
	// just randomly place particles. We don't really care how huge the bond forces get: this is just a unit test
	RandomInitializer rand_init(N, Scalar(0.2), Scalar(0.9), "A");
	shared_ptr<ParticleData> pdata(new ParticleData(rand_init, exec_conf));
	
	shared_ptr<HarmonicImproperForceCompute> fc1 = tf_creator1(pdata);
	shared_ptr<HarmonicImproperForceCompute> fc2 = tf_creator2(pdata);
	fc1->setParams(0, Scalar(2.0), Scalar(3.0));
	fc2->setParams(0, Scalar(2.0), Scalar(3.0));

	// add improperrs
	for (unsigned int i = 0; i < N-3; i++)
		{
		pdata->getImproperData()->addImproper(Improper(0, i, i+1,i+2, i+3));
		}
		
	// compute the forces
	fc1->compute(0);
	fc2->compute(0);
	
	// verify that the forces are identical (within roundoff errors)
	ForceDataArrays arrays1 = fc1->acquire();
	ForceDataArrays arrays2 = fc2->acquire();

	Scalar rough_tol = Scalar(3.0);

	for (unsigned int i = 0; i < N; i++)
		{
		BOOST_CHECK_CLOSE(arrays1.fx[i], arrays2.fx[i], rough_tol);
		BOOST_CHECK_CLOSE(arrays1.fy[i], arrays2.fy[i], rough_tol);
		BOOST_CHECK_CLOSE(arrays1.fz[i], arrays2.fz[i], rough_tol);
		BOOST_CHECK_CLOSE(arrays1.pe[i], arrays2.pe[i], rough_tol);
		BOOST_CHECK_SMALL(arrays1.virial[i], rough_tol);
		BOOST_CHECK_SMALL(arrays2.virial[i], rough_tol); 
		}
	}

//! HarmonicImproperForceCompute creator for improper_force_basic_tests()
shared_ptr<HarmonicImproperForceCompute> base_class_tf_creator(shared_ptr<ParticleData> pdata)
	{
	return shared_ptr<HarmonicImproperForceCompute>(new HarmonicImproperForceCompute(pdata));
	}

#ifdef ENABLE_CUDA
//! ImproperForceCompute creator for bond_force_basic_tests()
shared_ptr<HarmonicImproperForceCompute> gpu_tf_creator(shared_ptr<ParticleData> pdata)
	{
	return shared_ptr<HarmonicImproperForceCompute>(new HarmonicImproperForceComputeGPU(pdata));
	}
#endif

//! boost test case for improper forces on the CPU
BOOST_AUTO_TEST_CASE( HarmonicImproperForceCompute_basic )
	{
	printf(" IN BOOST_AUTO_TEST_CASE: CPU \n");
	improperforce_creator tf_creator = bind(base_class_tf_creator, _1);
	improper_force_basic_tests(tf_creator, ExecutionConfiguration(ExecutionConfiguration::CPU, 0));
	}

#ifdef ENABLE_CUDA
//! boost test case for improper forces on the GPU
BOOST_AUTO_TEST_CASE( HarmonicImproperForceComputeGPU_basic )
	{
	printf(" IN BOOST_AUTO_TEST_CASE: GPU \n");
	improperforce_creator tf_creator = bind(gpu_tf_creator, _1);
	improper_force_basic_tests(tf_creator, ExecutionConfiguration(ExecutionConfiguration::GPU, 0));
	}

//! boost test case for comparing bond GPU and CPU BondForceComputes
BOOST_AUTO_TEST_CASE( HarmonicImproperForceComputeGPU_compare )
	{
	improperforce_creator tf_creator_gpu = bind(gpu_tf_creator, _1);
	improperforce_creator tf_creator = bind(base_class_tf_creator, _1);
	improper_force_comparison_tests(tf_creator, tf_creator_gpu, ExecutionConfiguration(ExecutionConfiguration::GPU, 0));
	}
	
//! boost test case for comparing calculation on the CPU to multi-gpu ones
BOOST_AUTO_TEST_CASE( HarmonicImproperForce_MultiGPU_compare)
	{
	vector<unsigned int> gpu_list;
	gpu_list.push_back(0);
	gpu_list.push_back(0);
	gpu_list.push_back(0);
	gpu_list.push_back(0);
	ExecutionConfiguration exec_conf(ExecutionConfiguration::GPU, gpu_list);
	
	improperforce_creator tf_creator_gpu = bind(gpu_tf_creator, _1);
	improperforce_creator tf_creator = bind(base_class_tf_creator, _1);
	improper_force_comparison_tests(tf_creator, tf_creator_gpu, exec_conf);
	}
#endif

