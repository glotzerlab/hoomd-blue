// Copyright (c) 2009-2024 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

// this include is necessary to get MPI included before anything else to support intel MPI
#include "hoomd/ExecutionConfiguration.h"

#include "hoomd/RandomNumbers.h"
#include "hoomd/md/MolecularForceCompute.cc"

#include <set>

using namespace std;
using namespace std::placeholders;
using namespace hoomd;
using namespace hoomd::md;

/*! \file test_MolecularForceCompute.cc
    \brief Implements unit tests for MolecularForceCompute on CPU and GPU
    \ingroup unit_tests
*/

#include "hoomd/test/upp11_config.h"

HOOMD_UP_MAIN();

class MyMolecularForceCompute : public MolecularForceCompute
    {
    public:
    MyMolecularForceCompute(std::shared_ptr<SystemDefinition> sysdef,
                            std::vector<unsigned int>& molecule_tags,
                            unsigned int n_molecules)
        : MolecularForceCompute(sysdef)
        {
        m_molecule_tag.resize(molecule_tags.size());
        m_n_molecules_global = n_molecules;
        unsigned int i = 0;
        for (auto it = molecule_tags.begin(); it != molecule_tags.end(); ++it)
            {
            m_molecule_tag[i++] = *it;
            }
        }

    void setNMolecules(unsigned int nmol)
        {
        m_n_molecules_global = nmol;
        }

    void setMoleculeTags(std::vector<unsigned int>& molecule_tags)
        {
        unsigned int i = 0;
        for (auto it = molecule_tags.begin(); it != molecule_tags.end(); ++it)
            {
            m_molecule_tag[i++] = *it;
            }
        }
    };

//! Test if basic sorting of particles into molecules works
void basic_molecule_test(std::shared_ptr<ExecutionConfiguration> exec_conf)
    {
    std::shared_ptr<SystemDefinition> sysdef_5(
        new SystemDefinition(5, BoxDim(1000.0), 1, 0, 0, 0, 0, exec_conf));
    std::shared_ptr<ParticleData> pdata_5 = sysdef_5->getParticleData();

    // three molecules, consecutive in memory
    unsigned int nmol = 3;
    std::vector<unsigned int> molecule_tags(5);
    molecule_tags[0] = 1;
    molecule_tags[1] = 1;
    molecule_tags[2] = 3;
    molecule_tags[3] = 2;
    molecule_tags[4] = 2;

    MyMolecularForceCompute mfc(sysdef_5, molecule_tags, nmol);

        {
        // check molecule lists
        ArrayHandle<unsigned int> h_molecule_length(mfc.getMoleculeLengths(),
                                                    access_location::host,
                                                    access_mode::read);
        ArrayHandle<unsigned int> h_molecule_list(mfc.getMoleculeList(),
                                                  access_location::host,
                                                  access_mode::read);
        Index2D molecule_indexer = mfc.getMoleculeIndexer();

        UP_ASSERT_EQUAL(molecule_indexer.getW(), 2); // max length
        UP_ASSERT_EQUAL(molecule_indexer.getH(), 3);

        // molecule list is sorted by lowest molecule member index
        UP_ASSERT_EQUAL(h_molecule_length.data[0], 2);
        UP_ASSERT_EQUAL(h_molecule_length.data[1], 1);
        UP_ASSERT_EQUAL(h_molecule_length.data[2], 2);

        UP_ASSERT_EQUAL(h_molecule_list.data[molecule_indexer(0, 0)], 0);
        UP_ASSERT_EQUAL(h_molecule_list.data[molecule_indexer(1, 0)], 1);
        UP_ASSERT_EQUAL(h_molecule_list.data[molecule_indexer(0, 1)], 2);
        UP_ASSERT_EQUAL(h_molecule_list.data[molecule_indexer(0, 2)], 3);
        UP_ASSERT_EQUAL(h_molecule_list.data[molecule_indexer(1, 2)], 4);
        }

        {
        // jumble the tags
        ArrayHandle<unsigned int> h_tag(pdata_5->getTags(),
                                        access_location::host,
                                        access_mode::readwrite);
        ArrayHandle<unsigned int> h_rtag(pdata_5->getRTags(),
                                         access_location::host,
                                         access_mode::readwrite);
        h_tag.data[0] = 4;
        h_tag.data[1] = 3;
        h_tag.data[2] = 2;
        h_tag.data[3] = 1;
        h_tag.data[4] = 0;

        h_rtag.data[0] = 4;
        h_rtag.data[1] = 3;
        h_rtag.data[2] = 2;
        h_rtag.data[3] = 1;
        h_rtag.data[4] = 0;
        }
    pdata_5->notifyParticleSort();

        {
        // check molecule lists
        ArrayHandle<unsigned int> h_molecule_length(mfc.getMoleculeLengths(),
                                                    access_location::host,
                                                    access_mode::read);
        ArrayHandle<unsigned int> h_molecule_list(mfc.getMoleculeList(),
                                                  access_location::host,
                                                  access_mode::read);
        Index2D molecule_indexer = mfc.getMoleculeIndexer();

        UP_ASSERT_EQUAL(molecule_indexer.getW(), 2); // max length
        UP_ASSERT_EQUAL(molecule_indexer.getH(), 3);

        // molecule list is sorted by lowest molecule member index
        UP_ASSERT_EQUAL(h_molecule_length.data[0], 2);
        UP_ASSERT_EQUAL(h_molecule_length.data[1], 1);
        UP_ASSERT_EQUAL(h_molecule_length.data[2], 2);

        UP_ASSERT_EQUAL(h_molecule_list.data[molecule_indexer(0, 0)], 1);
        UP_ASSERT_EQUAL(h_molecule_list.data[molecule_indexer(1, 0)], 0);
        UP_ASSERT_EQUAL(h_molecule_list.data[molecule_indexer(0, 1)], 2);
        UP_ASSERT_EQUAL(h_molecule_list.data[molecule_indexer(0, 2)], 4);
        UP_ASSERT_EQUAL(h_molecule_list.data[molecule_indexer(1, 2)], 3);
        }
    }

//! Test if the CPU and the GPU implementation give consistent results
void comparison_test(std::shared_ptr<ExecutionConfiguration> exec_conf_cpu,
                     std::shared_ptr<ExecutionConfiguration> exec_conf_gpu)
    {
    unsigned int nptl = 100;

    std::shared_ptr<SystemDefinition> sysdef_cpu(
        new SystemDefinition(nptl, BoxDim(1000.0), 1, 0, 0, 0, 0, exec_conf_cpu));
    std::shared_ptr<ParticleData> pdata_cpu = sysdef_cpu->getParticleData();

    std::shared_ptr<SystemDefinition> sysdef_gpu(
        new SystemDefinition(nptl, BoxDim(1000.0), 1, 0, 0, 0, 0, exec_conf_gpu));
    std::shared_ptr<ParticleData> pdata_gpu = sysdef_gpu->getParticleData();

    unsigned int niter = 100;

    std::vector<unsigned int> molecule_tags(nptl, NO_MOLECULE);
    hoomd::RandomGenerator rng(hoomd::Seed(0, 1, 2), hoomd::Counter(4, 5, 6));

    MyMolecularForceCompute mfc_cpu(sysdef_cpu, molecule_tags, 0);
    MyMolecularForceCompute mfc_gpu(sysdef_gpu, molecule_tags, 0);

    for (unsigned i = 0; i < niter; ++i)
        {
        // randomly assign molecule tags

        for (unsigned int j = 0; j < nptl; ++j)
            {
            // choose a molecule tag 0 <= mol_tag <= nptl
            unsigned int t = hoomd::UniformIntDistribution(nptl)(rng);
            if (t == nptl)
                t = NO_MOLECULE;

            molecule_tags[j] = t;
            }
        // count number of unique molecules
        std::set<unsigned int> unique_tags;
        for (auto it = molecule_tags.begin(); it != molecule_tags.end(); ++it)
            {
            if (*it != NO_MOLECULE)
                unique_tags.insert(*it);
            }

        mfc_cpu.setNMolecules((unsigned int)unique_tags.size());
        mfc_gpu.setNMolecules((unsigned int)unique_tags.size());

        mfc_cpu.setMoleculeTags(molecule_tags);
        mfc_gpu.setMoleculeTags(molecule_tags);

        pdata_cpu->notifyParticleSort();
        pdata_gpu->notifyParticleSort();

            {
            // check molecule lists for consistency
            ArrayHandle<unsigned int> h_molecule_length_cpu(mfc_cpu.getMoleculeLengths(),
                                                            access_location::host,
                                                            access_mode::read);
            ArrayHandle<unsigned int> h_molecule_list_cpu(mfc_cpu.getMoleculeList(),
                                                          access_location::host,
                                                          access_mode::read);
            Index2D molecule_indexer_cpu = mfc_cpu.getMoleculeIndexer();

            ArrayHandle<unsigned int> h_molecule_length_gpu(mfc_gpu.getMoleculeLengths(),
                                                            access_location::host,
                                                            access_mode::read);
            ArrayHandle<unsigned int> h_molecule_list_gpu(mfc_gpu.getMoleculeList(),
                                                          access_location::host,
                                                          access_mode::read);
            Index2D molecule_indexer_gpu = mfc_gpu.getMoleculeIndexer();

            UP_ASSERT_EQUAL(molecule_indexer_gpu.getW(), molecule_indexer_cpu.getW());
            UP_ASSERT_EQUAL(molecule_indexer_gpu.getH(), molecule_indexer_cpu.getH());

            for (unsigned int j = 0; j < molecule_indexer_cpu.getW(); ++j)
                {
                UP_ASSERT_EQUAL(h_molecule_length_cpu.data[j], h_molecule_length_gpu.data[j]);

                for (unsigned int k = 0; k < h_molecule_length_cpu.data[j]; ++k)
                    {
                    UP_ASSERT_EQUAL(h_molecule_list_cpu.data[molecule_indexer_cpu(k, j)],
                                    h_molecule_list_gpu.data[molecule_indexer_gpu(k, j)]);
                    }
                }
            }
        }
    }

//! test case for particle test on CPU
UP_TEST(MolecularForceCompute_basic)
    {
    basic_molecule_test(std::shared_ptr<ExecutionConfiguration>(
        new ExecutionConfiguration(ExecutionConfiguration::CPU)));
    }

#ifdef ENABLE_HIP
//! test case for particle test on GPU
UP_TEST(MolecularForceCompute_basic_GPU)
    {
    basic_molecule_test(std::shared_ptr<ExecutionConfiguration>(
        new ExecutionConfiguration(ExecutionConfiguration::GPU)));
    }

//! test case for comparing GPU output to base class output
UP_TEST(MolecularForceCompute_compare)
    {
    comparison_test(std::shared_ptr<ExecutionConfiguration>(
                        new ExecutionConfiguration(ExecutionConfiguration::CPU)),
                    std::shared_ptr<ExecutionConfiguration>(
                        new ExecutionConfiguration(ExecutionConfiguration::GPU)));
    }

#endif
