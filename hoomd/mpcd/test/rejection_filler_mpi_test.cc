// Copyright (c) 2009-2024 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

#include "hoomd/Communicator.h"
#include "hoomd/mpcd/RejectionVirtualParticleFiller.h"
#ifdef ENABLE_HIP
#include "hoomd/mpcd/RejectionVirtualParticleFillerGPU.h"
#endif // ENABLE_HIP
#include "hoomd/mpcd/SphereGeometry.h"

#include "hoomd/SnapshotSystemData.h"
#include "hoomd/test/upp11_config.h"

HOOMD_UP_MAIN()

using namespace hoomd;

template<class F>
void sphere_rejection_fill_mpi_test(std::shared_ptr<ExecutionConfiguration> exec_conf)
    {
    UP_ASSERT_EQUAL(exec_conf->getNRanks(), 8);

    std::shared_ptr<SnapshotSystemData<Scalar>> snap(new SnapshotSystemData<Scalar>());
    snap->global_box = std::make_shared<BoxDim>(20.0);
    snap->particle_data.type_mapping.push_back("A");
    snap->mpcd_data.resize(1);
    snap->mpcd_data.type_mapping.push_back("A");
    snap->mpcd_data.type_mapping.push_back("B");
    snap->mpcd_data.position[0] = vec3<Scalar>(1, 1, 1);
    snap->mpcd_data.velocity[0] = vec3<Scalar>(123, 456, 789);

    std::shared_ptr<DomainDecomposition> decomposition(
        new DomainDecomposition(exec_conf, snap->global_box->getL(), 2, 2, 2));
    std::shared_ptr<SystemDefinition> sysdef(new SystemDefinition(snap, exec_conf, decomposition));
    std::shared_ptr<Communicator> pdata_comm(new Communicator(sysdef, decomposition));
    sysdef->setCommunicator(pdata_comm);

    auto pdata = sysdef->getMPCDParticleData();
    auto cl = std::make_shared<mpcd::CellList>(sysdef, 2.0, false);
    // There should be no virtual particles in the system at this point
    UP_ASSERT_EQUAL(pdata->getNVirtual(), 0);
    UP_ASSERT_EQUAL(pdata->getNVirtualGlobal(), 0);

    // create a spherical confinement of radius 5.0
    Scalar r = 5.0;
    auto sphere = std::make_shared<const mpcd::SphereGeometry>(r, true);
    std::shared_ptr<Variant> kT = std::make_shared<VariantConstant>(1.5);
    std::shared_ptr<mpcd::RejectionVirtualParticleFiller<mpcd::SphereGeometry>> filler
        = std::make_shared<F>(sysdef, "B", 2.0, kT, sphere);
    filler->setCellList(cl);

    /*
     * Test basic filling up for this cell list
     */
    filler->fill(0);
        {
        ArrayHandle<Scalar4> h_pos(pdata->getPositions(), access_location::host, access_mode::read);
        ArrayHandle<Scalar4> h_vel(pdata->getVelocities(),
                                   access_location::host,
                                   access_mode::read);
        ArrayHandle<unsigned int> h_tag(pdata->getTags(), access_location::host, access_mode::read);
        const BoxDim& box = sysdef->getParticleData()->getBox();
        // check if the virtual particles are outside the sphere
        unsigned int N_out(0);
        for (unsigned int i = 0; i < pdata->getNVirtual(); ++i)
            {
            const unsigned int idx = pdata->getN() + i;
            // check if the virtual particles are in the box
            UP_ASSERT(h_pos.data[idx].x >= box.getLo().x && h_pos.data[idx].x < box.getHi().x);
            UP_ASSERT(h_pos.data[idx].y >= box.getLo().y && h_pos.data[idx].y < box.getHi().y);
            UP_ASSERT(h_pos.data[idx].z >= box.getLo().z && h_pos.data[idx].z < box.getHi().z);

            Scalar3 pos = make_scalar3(h_pos.data[idx].x, h_pos.data[idx].y, h_pos.data[idx].z);
            const Scalar r2 = dot(pos, pos);
            if (r2 > r * r)
                ++N_out;
            }
        UP_ASSERT_EQUAL(N_out, pdata->getNVirtual());
        }

    /*
     * Test avg. number of virtual particles on each rank by filling up the system
     */
    Scalar N_avg_rank(0);
    Scalar N_avg_global(0);
    unsigned int num_samples(500);
    for (unsigned int t = 0; t < num_samples; ++t)
        {
        pdata->removeVirtualParticles();
        UP_ASSERT_EQUAL(pdata->getNVirtualGlobal(), 0);
        filler->fill(1 + t);

        N_avg_rank += pdata->getNVirtual();
        N_avg_global += pdata->getNVirtualGlobal();
        }
    N_avg_rank /= num_samples;
    N_avg_global /= num_samples;

    /*
     * Expected number of virtual particles = int( density * volume outside sphere )
     * volume outside sphere = sim-box volume - sphere volume
     * N_exptd = int(density*(L^3 - 4*pi*r^3/3))
     *         = 14952
     * 8 CPUs -> each CPU has equal volume to fill with virtual particles
     * Therefore, N_exptd_rank = 14952/8 = 1869
     */
    UP_ASSERT_CLOSE(N_avg_rank, Scalar(1869), tol_small);
    UP_ASSERT_CLOSE(N_avg_global, Scalar(14952), tol_small);
    }

UP_TEST(sphere_rejection_fill_mpi)
    {
    sphere_rejection_fill_mpi_test<mpcd::RejectionVirtualParticleFiller<mpcd::SphereGeometry>>(
        std::make_shared<ExecutionConfiguration>(ExecutionConfiguration::CPU));
    }
#ifdef ENABLE_HIP
UP_TEST(sphere_rejection_fill_mpi_gpu)
    {
    sphere_rejection_fill_mpi_test<mpcd::RejectionVirtualParticleFillerGPU<mpcd::SphereGeometry>>(
        std::make_shared<ExecutionConfiguration>(ExecutionConfiguration::GPU));
    }
#endif // ENABLE_HIP
