//! name the boost unit test module
#define BOOST_TEST_MODULE DomainDecompositionTests
#include "boost_utf_configure.h"

#include "HOOMDMath.h"
#include "ExecutionConfiguration.h"
#include "System.h"
#include "TwoStepNVE.h"
#include "IntegratorTwoStep.h"
#include "AllPairPotentials.h"
#include "NeighborListBinned.h"
#include "SFCPackUpdater.h"
#include "RandomGenerator.h"

#include <boost/python.hpp>
#include <boost/mpi.hpp>
#include <boost/shared_ptr.hpp>

#include <math.h>

#include "Communicator.h"
#include "NeighborListBinnedMPI.h"
#include "MPIInitializer.h"

#ifdef ENABLE_CUDA
#include "CellListGPU.h"
#include "NeighborListGPUBinned.h"
#include "TwoStepNVEGPU.h"
#include "ParticleGroupGPU.h"
#include "CommunicatorGPU.h"
#include "NeighborListBinnedMPIGPU.h"
#endif

using namespace boost;

void set_num_threads(int nthreads);

void test_domain_decomposition(boost::shared_ptr<ExecutionConfiguration> exec_conf)
{
    ClockSource clk;

    boost::mpi::environment env(boost::unit_test::framework::master_test_suite().argc,
                                boost::unit_test::framework::master_test_suite().argv);
    shared_ptr<boost::mpi::communicator> world(new boost::mpi::communicator);

    // initialize a random particle system
    Scalar phi_p = 0.2;
    unsigned int N = 200000;
    Scalar L = pow(M_PI/6.0/phi_p*Scalar(N),1.0/3.0);
    BoxDim box_g(L);
    RandomGenerator rand_init(box_g, 12345);
    std::vector<string> types;
    types.push_back("A");
    std::vector<uint> bonds;
    std::vector<string> bond_types;
    rand_init.addGenerator((int)N, boost::shared_ptr<PolymerParticleGenerator>(new PolymerParticleGenerator(1.0, types, bonds, bonds, bond_types, 100)));
    rand_init.setSeparationRadius("A", .4);

    if (world->rank() == 0) rand_init.generate();

    boost::shared_ptr<MPIInitializer> mpi_init(new MPIInitializer(rand_init, world));

    shared_ptr<SystemDefinition> sysdef(new SystemDefinition(*mpi_init, exec_conf));

    boost::shared_ptr<Communicator> comm;
    std::vector<unsigned int> neighbor_rank;
    for (unsigned int i = 0; i < 6; i++)
        neighbor_rank.push_back(mpi_init->getNeighborRank(i));

    int3 dim = make_int3(mpi_init->getDimension(0),
                         mpi_init->getDimension(1),
                         mpi_init->getDimension(2));
#ifdef ENABLE_CUDA
    if (exec_conf->isCUDAEnabled())
        comm = shared_ptr<Communicator>(new CommunicatorGPU(sysdef, world, neighbor_rank, dim, mpi_init->getGlobalBox()));
    else
#endif
        comm = boost::shared_ptr<Communicator>(new Communicator(sysdef,world,neighbor_rank, dim, mpi_init->getGlobalBox()));

    boost::shared_ptr<Profiler> prof(new Profiler());
    comm->setProfiler(prof);

    std::cout << "Processor " << world->rank() << " " << sysdef->getParticleData()->getN() << " particles. " << std::endl;
//    std::cout << world.rank() << " box xlo " << box.xlo << " xhi " << box.xhi << " ylo " << box.ylo << " yhi " << box.yhi << " zlo " << box.zlo << " zhi " << box.zhi << std::endl;

    shared_ptr<ParticleData> pdata = sysdef->getParticleData();
    std::cout << "global tags " << pdata->getNGlobal() << std::endl;
    shared_ptr<ParticleSelector> selector_all;
#ifdef ENABLE_CUDA
    if (exec_conf->isCUDAEnabled())
        selector_all = shared_ptr<ParticleSelector>(new ParticleSelectorTagGPU(sysdef, 0, pdata->getNGlobal()-1));
    else
#endif
    selector_all = shared_ptr<ParticleSelector>(new ParticleSelectorTag(sysdef, 0, pdata->getNGlobal()-1));


    shared_ptr<ParticleGroup> group_all(new ParticleGroup(sysdef, selector_all));

    shared_ptr<TwoStepNVE> two_step_nve;
#ifdef ENABLE_CUDA
    if (exec_conf->isCUDAEnabled())
        two_step_nve = shared_ptr<TwoStepNVE>(new TwoStepNVEGPU(sysdef, group_all));
    else
#endif
    two_step_nve = shared_ptr<TwoStepNVE>(new TwoStepNVE(sysdef, group_all));

    for (unsigned int i = 0; i < 3; i++)
        two_step_nve->setNoWrapParticles(i,(comm->getDimension(i)>1));

//    Scalar r_cut = pow(2.0,1./6.);
//    Scalar r_buff = 0.4;
    Scalar r_cut = Scalar(3.0);
    Scalar r_buff = Scalar(0.8);
    shared_ptr<NeighborList> nlist;
#ifdef ENABLE_CUDA
    if (exec_conf->isCUDAEnabled())
        nlist = shared_ptr<NeighborList>(new NeighborListBinnedMPIGPU(sysdef, r_cut, r_buff, comm, boost::shared_ptr<CellList>(new CellListGPU(sysdef))));
    else
#endif
    nlist = shared_ptr<NeighborList>(new NeighborListBinnedMPI(sysdef, r_cut, r_buff, comm ));
    nlist->setStorageMode(NeighborList::full);

    shared_ptr<PotentialPairLJ> fc;

#ifdef ENABLE_CUDA
    if (exec_conf->isCUDAEnabled())
        fc = shared_ptr<PotentialPairLJ>(new PotentialPairLJGPU(sysdef, nlist));
    else
#endif
    fc = shared_ptr<PotentialPairLJ>(new PotentialPairLJ(sysdef, nlist));
    fc->setRcut(0, 0, r_cut);

    // setup some values for alpha and sigma
    Scalar epsilon = Scalar(1.0);
    Scalar sigma = Scalar(1.0);
    Scalar alpha = Scalar(1.0);
    Scalar lj1 = Scalar(4.0) * epsilon * pow(sigma,Scalar(12.0));
    Scalar lj2 = alpha * Scalar(4.0) * epsilon * pow(sigma,Scalar(6.0));
    fc->setParams(0,0,make_scalar2(lj1,lj2));

    shared_ptr<IntegratorTwoStep> nve(new IntegratorTwoStep(sysdef, Scalar(0.005)));
    nve->addIntegrationMethod(two_step_nve);
    nve->addForceCompute(fc);

    nve->prepRun(0);

    int64_t initial_time = clk.getTime();
    boost::shared_ptr<SFCPackUpdater> sorter(new SFCPackUpdater(sysdef));

    for (int i=0; i< 5000; i++)
       {
      // if (!( i%100))
//       if (world.rank()==0)  std::cout << "step " << i << std::endl;
       if (i % 300 == 0) sorter->update(i);
       nve->update(i);
       comm->copyGhosts();
       Scalar TPS = i/Scalar(clk.getTime() - initial_time) * Scalar(1e9);
       if (i%10 == 0 && world->rank() == 0) std::cout << "step " << i << " TPS: " << TPS << std::endl;
       }

    if (world->rank() == 0)
        cout << *prof;
}

//! Tests MPI domain decomposition with NVE integrator
#if 0
BOOST_AUTO_TEST_CASE( DomainDecomposition_NVE_test )
    {
    set_num_threads(1);
    test_domain_decomposition(boost::shared_ptr<ExecutionConfiguration>(new ExecutionConfiguration(ExecutionConfiguration::CPU)));
    }
#endif
#ifdef ENABLE_CUDA
//! Tests MPI domain decomposition with NVE integrator on the GPU
BOOST_AUTO_TEST_CASE( DomainDecomposition_NVE_test_GPU )
    {
    boost::shared_ptr<ExecutionConfiguration> exec_conf(new ExecutionConfiguration(ExecutionConfiguration::GPU));
    test_domain_decomposition(exec_conf);
    }
#endif
