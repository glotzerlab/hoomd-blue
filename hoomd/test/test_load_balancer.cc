// Copyright (c) 2009-2024 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

#ifdef ENABLE_MPI

// this has to be included after naming the test module
#include "upp11_config.h"
HOOMD_UP_MAIN();

#include "hoomd/System.h"
#include "hoomd/Trigger.h"

#include <functional>
#include <memory>

#include "hoomd/Communicator.h"
#include "hoomd/ExecutionConfiguration.h"
#include "hoomd/LoadBalancer.h"
#ifdef ENABLE_HIP
#include "hoomd/LoadBalancerGPU.h"
#endif

#define TO_TRICLINIC(v) dest_box.makeCoordinates(ref_box.makeFraction(make_scalar3(v.x, v.y, v.z)))
#define TO_POS4(v) make_scalar4(v.x, v.y, v.z, h_pos.data[rtag].w)
#define FROM_TRICLINIC(v) \
    ref_box.makeCoordinates(dest_box.makeFraction(make_scalar3(v.x, v.y, v.z)))

using namespace std;
using namespace hoomd;

template<class LB>
void test_load_balancer_basic(std::shared_ptr<ExecutionConfiguration> exec_conf,
                              const BoxDim& dest_box)
    {
    // this test needs to be run on eight processors
    int size;
    MPI_Comm_size(exec_conf->getHOOMDWorldMPICommunicator(), &size);
    UP_ASSERT_EQUAL(size, 8);

    // create a system with eight particles
    BoxDim ref_box = BoxDim(2.0);
    std::shared_ptr<SystemDefinition> sysdef(new SystemDefinition(8,        // number of particles
                                                                  dest_box, // box dimensions
                                                                  1, // number of particle types
                                                                  0, // number of bond types
                                                                  0, // number of angle types
                                                                  0, // number of dihedral types
                                                                  0, // number of dihedral types
                                                                  exec_conf));

    std::shared_ptr<ParticleData> pdata(sysdef->getParticleData());

    pdata->setPosition(0, TO_TRICLINIC(make_scalar3(0.25, -0.25, 0.25)), false);
    pdata->setPosition(1, TO_TRICLINIC(make_scalar3(0.25, -0.25, 0.75)), false);
    pdata->setPosition(2, TO_TRICLINIC(make_scalar3(0.25, -0.75, 0.25)), false);
    pdata->setPosition(3, TO_TRICLINIC(make_scalar3(0.25, -0.75, 0.75)), false);
    pdata->setPosition(4, TO_TRICLINIC(make_scalar3(0.75, -0.25, 0.25)), false);
    pdata->setPosition(5, TO_TRICLINIC(make_scalar3(0.75, -0.25, 0.75)), false);
    pdata->setPosition(6, TO_TRICLINIC(make_scalar3(0.75, -0.75, 0.25)), false);
    pdata->setPosition(7, TO_TRICLINIC(make_scalar3(0.75, -0.75, 0.75)), false);

    SnapshotParticleData<Scalar> snap(8);
    pdata->takeSnapshot(snap);

    // initialize a 2x2x2 domain decomposition on processor with rank 0
    std::vector<Scalar> fxs(1), fys(1), fzs(1);
    fxs[0] = Scalar(0.5);
    fys[0] = Scalar(0.5);
    fzs[0] = Scalar(0.5);
    std::shared_ptr<DomainDecomposition> decomposition(
        new DomainDecomposition(exec_conf, pdata->getBox().getL(), fxs, fys, fzs));
    std::shared_ptr<Communicator> comm(new Communicator(sysdef, decomposition));
    pdata->setDomainDecomposition(decomposition);
    sysdef->setCommunicator(comm);

    pdata->initializeFromSnapshot(snap);

    auto trigger = std::make_shared<PeriodicTrigger>(1);
    std::shared_ptr<LoadBalancer> lb(new LB(sysdef, trigger));
    lb->setMaxIterations(2);

    // migrate atoms
    comm->migrateParticles();
    const Index3D& di = decomposition->getDomainIndexer();
    UP_ASSERT_EQUAL(pdata->getOwnerRank(0), di(1, 0, 1));
    UP_ASSERT_EQUAL(pdata->getOwnerRank(1), di(1, 0, 1));
    UP_ASSERT_EQUAL(pdata->getOwnerRank(2), di(1, 0, 1));
    UP_ASSERT_EQUAL(pdata->getOwnerRank(3), di(1, 0, 1));
    UP_ASSERT_EQUAL(pdata->getOwnerRank(4), di(1, 0, 1));
    UP_ASSERT_EQUAL(pdata->getOwnerRank(5), di(1, 0, 1));
    UP_ASSERT_EQUAL(pdata->getOwnerRank(6), di(1, 0, 1));
    UP_ASSERT_EQUAL(pdata->getOwnerRank(7), di(1, 0, 1));

    // adjust the domain boundaries
    for (unsigned int t = 0; t < 10; ++t)
        {
        lb->update(t);
        }

    // each rank should own one particle
    UP_ASSERT_EQUAL(pdata->getN(), 1);
    UP_ASSERT_EQUAL(pdata->getOwnerRank(0), di(0, 1, 0));
    UP_ASSERT_EQUAL(pdata->getOwnerRank(1), di(0, 1, 1));
    UP_ASSERT_EQUAL(pdata->getOwnerRank(2), di(0, 0, 0));
    UP_ASSERT_EQUAL(pdata->getOwnerRank(3), di(0, 0, 1));
    UP_ASSERT_EQUAL(pdata->getOwnerRank(4), di(1, 1, 0));
    UP_ASSERT_EQUAL(pdata->getOwnerRank(5), di(1, 1, 1));
    UP_ASSERT_EQUAL(pdata->getOwnerRank(6), di(1, 0, 0));
    UP_ASSERT_EQUAL(pdata->getOwnerRank(7), di(1, 0, 1));

    // flip the particle signs and see if the domains can realign correctly
    pdata->setPosition(0, TO_TRICLINIC(make_scalar3(-0.25, 0.25, -0.25)), false);
    pdata->setPosition(1, TO_TRICLINIC(make_scalar3(-0.25, 0.25, -0.75)), false);
    pdata->setPosition(2, TO_TRICLINIC(make_scalar3(-0.25, 0.75, -0.25)), false);
    pdata->setPosition(3, TO_TRICLINIC(make_scalar3(-0.25, 0.75, -0.75)), false);
    pdata->setPosition(4, TO_TRICLINIC(make_scalar3(-0.75, 0.25, -0.25)), false);
    pdata->setPosition(5, TO_TRICLINIC(make_scalar3(-0.75, 0.25, -0.75)), false);
    pdata->setPosition(6, TO_TRICLINIC(make_scalar3(-0.75, 0.75, -0.25)), false);
    pdata->setPosition(7, TO_TRICLINIC(make_scalar3(-0.75, 0.75, -0.75)), false);
    comm->migrateParticles();

    for (unsigned int t = 10; t < 20; ++t)
        {
        lb->update(t);
        }
    // each rank should own one particle
    UP_ASSERT_EQUAL(pdata->getN(), 1);
    UP_ASSERT_EQUAL(pdata->getOwnerRank(0), di(1, 0, 1));
    UP_ASSERT_EQUAL(pdata->getOwnerRank(1), di(1, 0, 0));
    UP_ASSERT_EQUAL(pdata->getOwnerRank(2), di(1, 1, 1));
    UP_ASSERT_EQUAL(pdata->getOwnerRank(3), di(1, 1, 0));
    UP_ASSERT_EQUAL(pdata->getOwnerRank(4), di(0, 0, 1));
    UP_ASSERT_EQUAL(pdata->getOwnerRank(5), di(0, 0, 0));
    UP_ASSERT_EQUAL(pdata->getOwnerRank(6), di(0, 1, 1));
    UP_ASSERT_EQUAL(pdata->getOwnerRank(7), di(0, 1, 0));
    }

template<class LB>
void test_load_balancer_multi(std::shared_ptr<ExecutionConfiguration> exec_conf,
                              const BoxDim& dest_box)
    {
    // this test needs to be run on eight processors
    int size;
    MPI_Comm_size(exec_conf->getHOOMDWorldMPICommunicator(), &size);
    UP_ASSERT_EQUAL(size, 8);

    // create a system with eight particles
    BoxDim ref_box = BoxDim(2.0);
    std::shared_ptr<SystemDefinition> sysdef(new SystemDefinition(8,        // number of particles
                                                                  dest_box, // box dimensions
                                                                  1, // number of particle types
                                                                  0, // number of bond types
                                                                  0, // number of angle types
                                                                  0, // number of dihedral types
                                                                  0, // number of dihedral types
                                                                  exec_conf));

    std::shared_ptr<ParticleData> pdata(sysdef->getParticleData());

    pdata->setPosition(0, TO_TRICLINIC(make_scalar3(0.1, -0.1, -0.4)), false);
    pdata->setPosition(1, TO_TRICLINIC(make_scalar3(0.1, -0.2, -0.4)), false);
    pdata->setPosition(2, TO_TRICLINIC(make_scalar3(0.1, -0.1, 0.2)), false);
    pdata->setPosition(3, TO_TRICLINIC(make_scalar3(0.1, -0.2, 0.2)), false);
    pdata->setPosition(4, TO_TRICLINIC(make_scalar3(0.2, -0.1, 0.55)), false);
    pdata->setPosition(5, TO_TRICLINIC(make_scalar3(0.2, -0.2, 0.55)), false);
    pdata->setPosition(6, TO_TRICLINIC(make_scalar3(0.2, -0.1, 0.9)), false);
    pdata->setPosition(7, TO_TRICLINIC(make_scalar3(0.2, -0.2, 0.9)), false);

    SnapshotParticleData<Scalar> snap(8);
    pdata->takeSnapshot(snap);

    // initialize a 1x2x4 domain decomposition on processor with rank 0
    std::vector<Scalar> fxs, fys(1), fzs(3);
    fys[0] = Scalar(0.5);
    fzs[0] = Scalar(0.25);
    fzs[1] = Scalar(0.25);
    fzs[2] = Scalar(0.25);
    std::shared_ptr<DomainDecomposition> decomposition(
        new DomainDecomposition(exec_conf, pdata->getBox().getL(), fxs, fys, fzs));
    std::shared_ptr<Communicator> comm(new Communicator(sysdef, decomposition));
    pdata->setDomainDecomposition(decomposition);
    sysdef->setCommunicator(comm);

    pdata->initializeFromSnapshot(snap);

    auto trigger = std::make_shared<PeriodicTrigger>(1);
    std::shared_ptr<LoadBalancer> lb(new LB(sysdef, trigger));
    lb->enableDimension(1, false);
    lb->setMaxIterations(100);

    // migrate atoms and check placement
    comm->migrateParticles();
    const Index3D& di = decomposition->getDomainIndexer();
    UP_ASSERT_EQUAL(pdata->getOwnerRank(0), di(0, 0, 1));
    UP_ASSERT_EQUAL(pdata->getOwnerRank(1), di(0, 0, 1));
    UP_ASSERT_EQUAL(pdata->getOwnerRank(2), di(0, 0, 2));
    UP_ASSERT_EQUAL(pdata->getOwnerRank(3), di(0, 0, 2));
    UP_ASSERT_EQUAL(pdata->getOwnerRank(4), di(0, 0, 3));
    UP_ASSERT_EQUAL(pdata->getOwnerRank(5), di(0, 0, 3));
    UP_ASSERT_EQUAL(pdata->getOwnerRank(6), di(0, 0, 3));
    UP_ASSERT_EQUAL(pdata->getOwnerRank(7), di(0, 0, 3));

    // balance particles along z only
    lb->update(0);
        {
        uint3 grid_pos = decomposition->getGridPos();
        if (grid_pos.y == 0)
            {
            UP_ASSERT_EQUAL(pdata->getN(), 2);
            }
        else
            {
            UP_ASSERT_EQUAL(pdata->getN(), 0);
            }

        // check that fractional cuts lie in the right ranges
        vector<Scalar> frac_y = decomposition->getCumulativeFractions(1);
        MY_CHECK_CLOSE(frac_y[1], 0.5, tol);
        vector<Scalar> frac_z = decomposition->getCumulativeFractions(2);
        UP_ASSERT(frac_z[1] > 0.3 && frac_z[1] <= 0.6);
        UP_ASSERT(frac_z[2] > 0.6 && frac_z[2] <= 0.775);
        UP_ASSERT(frac_z[3] > 0.775 && frac_z[3] <= 0.95);

        UP_ASSERT_EQUAL(pdata->getOwnerRank(0), di(0, 0, 0));
        UP_ASSERT_EQUAL(pdata->getOwnerRank(1), di(0, 0, 0));
        UP_ASSERT_EQUAL(pdata->getOwnerRank(2), di(0, 0, 1));
        UP_ASSERT_EQUAL(pdata->getOwnerRank(3), di(0, 0, 1));
        UP_ASSERT_EQUAL(pdata->getOwnerRank(4), di(0, 0, 2));
        UP_ASSERT_EQUAL(pdata->getOwnerRank(5), di(0, 0, 2));
        UP_ASSERT_EQUAL(pdata->getOwnerRank(6), di(0, 0, 3));
        UP_ASSERT_EQUAL(pdata->getOwnerRank(7), di(0, 0, 3));
        }

    // turn on balancing along y and check that this balances now
    lb->enableDimension(1, true);
    lb->update(10);
        {
        UP_ASSERT_EQUAL(pdata->getN(), 1);

        // check that fractional cuts lie in the right ranges
        vector<Scalar> frac_y = decomposition->getCumulativeFractions(1);
        UP_ASSERT(frac_y[1] > 0.4 && frac_y[1] <= 0.45);

        UP_ASSERT_EQUAL(pdata->getOwnerRank(0), di(0, 1, 0));
        UP_ASSERT_EQUAL(pdata->getOwnerRank(1), di(0, 0, 0));
        UP_ASSERT_EQUAL(pdata->getOwnerRank(2), di(0, 1, 1));
        UP_ASSERT_EQUAL(pdata->getOwnerRank(3), di(0, 0, 1));
        UP_ASSERT_EQUAL(pdata->getOwnerRank(4), di(0, 1, 2));
        UP_ASSERT_EQUAL(pdata->getOwnerRank(5), di(0, 0, 2));
        UP_ASSERT_EQUAL(pdata->getOwnerRank(6), di(0, 1, 3));
        UP_ASSERT_EQUAL(pdata->getOwnerRank(7), di(0, 0, 3));
        }
    }

//! Ghost layer subscriber
struct ghost_layer_width_request
    {
    //! Constructor
    /*!
     * \param r_ghost Ghost layer width
     */
    ghost_layer_width_request(Scalar r_ghost) : m_r_ghost(r_ghost) { }

    //! Get the ghost layer width
    /*!
     * \param type Type index
     * \returns Constant ghost layer width for all types
     */
    Scalar get(unsigned int type)
        {
        return m_r_ghost;
        }
    Scalar m_r_ghost; //!< Ghost layer width
    };

template<class LB>
void test_load_balancer_ghost(std::shared_ptr<ExecutionConfiguration> exec_conf,
                              const BoxDim& dest_box)
    {
    // this test needs to be run on eight processors
    int size;
    MPI_Comm_size(exec_conf->getHOOMDWorldMPICommunicator(), &size);
    UP_ASSERT_EQUAL(size, 8);

    // create a system with eight particles
    BoxDim ref_box = BoxDim(2.0);
    std::shared_ptr<SystemDefinition> sysdef(new SystemDefinition(8,        // number of particles
                                                                  dest_box, // box dimensions
                                                                  1, // number of particle types
                                                                  0, // number of bond types
                                                                  0, // number of angle types
                                                                  0, // number of dihedral types
                                                                  0, // number of dihedral types
                                                                  exec_conf));

    std::shared_ptr<ParticleData> pdata(sysdef->getParticleData());

    pdata->setPosition(0, TO_TRICLINIC(make_scalar3(0.25, -0.25, 0.9)), false);
    pdata->setPosition(1, TO_TRICLINIC(make_scalar3(0.25, -0.25, 0.99)), false);
    pdata->setPosition(2, TO_TRICLINIC(make_scalar3(0.25, -0.75, 0.9)), false);
    pdata->setPosition(3, TO_TRICLINIC(make_scalar3(0.25, -0.75, 0.99)), false);
    pdata->setPosition(4, TO_TRICLINIC(make_scalar3(0.75, -0.25, 0.9)), false);
    pdata->setPosition(5, TO_TRICLINIC(make_scalar3(0.75, -0.25, 0.99)), false);
    pdata->setPosition(6, TO_TRICLINIC(make_scalar3(0.75, -0.75, 0.9)), false);
    pdata->setPosition(7, TO_TRICLINIC(make_scalar3(0.75, -0.75, 0.99)), false);

    SnapshotParticleData<Scalar> snap(8);
    pdata->takeSnapshot(snap);

    // initialize a 2x2x2 domain decomposition on processor with rank 0
    std::vector<Scalar> fxs(1), fys(1), fzs(1);
    fxs[0] = Scalar(0.5);
    fys[0] = Scalar(0.5);
    fzs[0] = Scalar(0.5);
    std::shared_ptr<DomainDecomposition> decomposition(
        new DomainDecomposition(exec_conf, pdata->getBox().getL(), fxs, fys, fzs));
    std::shared_ptr<Communicator> comm(new Communicator(sysdef, decomposition));
    pdata->setDomainDecomposition(decomposition);
    sysdef->setCommunicator(comm);

    pdata->initializeFromSnapshot(snap);

    auto trigger = std::make_shared<PeriodicTrigger>(1);
    std::shared_ptr<LoadBalancer> lb(new LB(sysdef, trigger));

    // migrate atoms and check placement
    comm->migrateParticles();
    const Index3D& di = decomposition->getDomainIndexer();
    UP_ASSERT_EQUAL(pdata->getOwnerRank(0), di(1, 0, 1));
    UP_ASSERT_EQUAL(pdata->getOwnerRank(1), di(1, 0, 1));
    UP_ASSERT_EQUAL(pdata->getOwnerRank(2), di(1, 0, 1));
    UP_ASSERT_EQUAL(pdata->getOwnerRank(3), di(1, 0, 1));
    UP_ASSERT_EQUAL(pdata->getOwnerRank(4), di(1, 0, 1));
    UP_ASSERT_EQUAL(pdata->getOwnerRank(5), di(1, 0, 1));
    UP_ASSERT_EQUAL(pdata->getOwnerRank(6), di(1, 0, 1));
    UP_ASSERT_EQUAL(pdata->getOwnerRank(7), di(1, 0, 1));

    // add a ghost layer subscriber and exchange ghosts
    ghost_layer_width_request g(Scalar(0.05));
    comm->getGhostLayerWidthRequestSignal()
        .connect<ghost_layer_width_request, &ghost_layer_width_request::get>(g);
    comm->exchangeGhosts();

    for (unsigned int t = 0; t < 20; ++t)
        {
        lb->update(t);
        }

    // because of the ghost layer width, you shouldn't be able to get to a domain this small
    uint3 grid_pos = decomposition->getGridPos();
    if (grid_pos.z == 1) // top layer has 2 each because (x,y) balanced out
        {
        UP_ASSERT_EQUAL(pdata->getN(), 2);
        }
    else // bottom layer has none
        {
        UP_ASSERT_EQUAL(pdata->getN(), 0);
        }
    UP_ASSERT_EQUAL(pdata->getOwnerRank(0), di(0, 1, 1));
    UP_ASSERT_EQUAL(pdata->getOwnerRank(1), di(0, 1, 1));
    UP_ASSERT_EQUAL(pdata->getOwnerRank(2), di(0, 0, 1));
    UP_ASSERT_EQUAL(pdata->getOwnerRank(3), di(0, 0, 1));
    UP_ASSERT_EQUAL(pdata->getOwnerRank(4), di(1, 1, 1));
    UP_ASSERT_EQUAL(pdata->getOwnerRank(5), di(1, 1, 1));
    UP_ASSERT_EQUAL(pdata->getOwnerRank(6), di(1, 0, 1));
    UP_ASSERT_EQUAL(pdata->getOwnerRank(7), di(1, 0, 1));
    }

//! Tests basic particle redistribution
UP_TEST(LoadBalancer_test_basic)
    {
    std::shared_ptr<ExecutionConfiguration> exec_conf(
        new ExecutionConfiguration(ExecutionConfiguration::CPU));
    // cubic box
    test_load_balancer_basic<LoadBalancer>(exec_conf, BoxDim(2.0));
    // triclinic box 1
    test_load_balancer_basic<LoadBalancer>(exec_conf, BoxDim(1.0, .1, .2, .3));
    // triclinic box 2
    test_load_balancer_basic<LoadBalancer>(exec_conf, BoxDim(1.0, -.6, .7, .5));
    }

//! Tests particle redistribution with multiple domains and specific directions
UP_TEST(LoadBalancer_test_multi)
    {
    std::shared_ptr<ExecutionConfiguration> exec_conf(
        new ExecutionConfiguration(ExecutionConfiguration::CPU));
    // cubic box
    test_load_balancer_multi<LoadBalancer>(exec_conf, BoxDim(2.0));
    // triclinic box 1
    test_load_balancer_multi<LoadBalancer>(exec_conf, BoxDim(1.0, .1, .2, .3));
    // triclinic box 2
    test_load_balancer_multi<LoadBalancer>(exec_conf, BoxDim(1.0, -.6, .7, .5));
    }

//! Tests particle redistribution with ghost layer width minimum
UP_TEST(LoadBalancer_test_ghost)
    {
    std::shared_ptr<ExecutionConfiguration> exec_conf(
        new ExecutionConfiguration(ExecutionConfiguration::CPU));
    // cubic box
    test_load_balancer_ghost<LoadBalancer>(exec_conf, BoxDim(2.0));
    // triclinic box 1
    test_load_balancer_ghost<LoadBalancer>(exec_conf, BoxDim(1.0, .1, .2, .3));
    // triclinic box 2
    test_load_balancer_ghost<LoadBalancer>(exec_conf, BoxDim(1.0, -.6, .7, .5));
    }

#ifdef ENABLE_HIP
//! Tests basic particle redistribution on the GPU
UP_TEST(LoadBalancerGPU_test_basic)
    {
    std::shared_ptr<ExecutionConfiguration> exec_conf(
        new ExecutionConfiguration(ExecutionConfiguration::GPU));
    // cubic box
    test_load_balancer_basic<LoadBalancerGPU>(exec_conf, BoxDim(2.0));
    // triclinic box 1
    test_load_balancer_basic<LoadBalancerGPU>(exec_conf, BoxDim(1.0, .1, .2, .3));
    // triclinic box 2
    test_load_balancer_basic<LoadBalancerGPU>(exec_conf, BoxDim(1.0, -.6, .7, .5));
    }

//! Tests particle redistribution with multiple domains and specific directions on the GPU
UP_TEST(LoadBalancerGPU_test_multi)
    {
    std::shared_ptr<ExecutionConfiguration> exec_conf(
        new ExecutionConfiguration(ExecutionConfiguration::GPU));
    // cubic box
    test_load_balancer_multi<LoadBalancerGPU>(exec_conf, BoxDim(2.0));
    // triclinic box 1
    test_load_balancer_multi<LoadBalancerGPU>(exec_conf, BoxDim(1.0, .1, .2, .3));
    // triclinic box 2
    test_load_balancer_multi<LoadBalancerGPU>(exec_conf, BoxDim(1.0, -.6, .7, .5));
    }

//! Tests particle redistribution with ghost layer width minimum
UP_TEST(LoadBalancerGPU_test_ghost)
    {
    std::shared_ptr<ExecutionConfiguration> exec_conf(
        new ExecutionConfiguration(ExecutionConfiguration::GPU));
    // cubic box
    test_load_balancer_ghost<LoadBalancerGPU>(exec_conf, BoxDim(2.0));
    // triclinic box 1
    test_load_balancer_ghost<LoadBalancerGPU>(exec_conf, BoxDim(1.0, .1, .2, .3));
    // triclinic box 2
    test_load_balancer_ghost<LoadBalancerGPU>(exec_conf, BoxDim(1.0, -.6, .7, .5));
    }
#endif // ENABLE_HIP

#endif // ENABLE_MPI
