// Copyright (c) 2009-2019 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.


// Maintainer: jglaser

#include "MolecularForceCompute.h"

#include "hoomd/CachedAllocator.h"
#include "hoomd/Autotuner.h"

#ifdef ENABLE_CUDA
#include "MolecularForceCompute.cuh"
#endif

#include <string.h>
#include <map>

namespace py = pybind11;

/*! \file MolecularForceCompute.cc
    \brief Contains code for the MolecularForceCompute class
*/

/*! \param sysdef SystemDefinition containing the ParticleData to compute forces on
*/
MolecularForceCompute::MolecularForceCompute(std::shared_ptr<SystemDefinition> sysdef)
    : ForceConstraint(sysdef), m_molecule_tag(m_exec_conf), m_n_molecules_global(0), m_dirty(true),
      m_molecule_list(m_exec_conf), m_molecule_length(m_exec_conf), m_molecule_order(m_exec_conf),
      m_molecule_idx(m_exec_conf)
    {
    // connect to the ParticleData to receive notifications when particles change order in memory
    m_pdata->getParticleSortSignal().connect<MolecularForceCompute, &MolecularForceCompute::setDirty>(this);

    TAG_ALLOCATION(m_molecule_tag);
    TAG_ALLOCATION(m_molecule_list);
    TAG_ALLOCATION(m_molecule_length);
    TAG_ALLOCATION(m_molecule_order);
    TAG_ALLOCATION(m_molecule_idx);

    #ifdef ENABLE_CUDA
    if (m_exec_conf->isCUDAEnabled())
        {
        // initialize autotuner
        std::vector<unsigned int> valid_params;
        for (unsigned int block_size = 32; block_size <= 1024; block_size += 32)
            valid_params.push_back(block_size);

        m_tuner_fill.reset(new Autotuner(valid_params, 5, 100000, "fill_molecule_table", this->m_exec_conf));
        }
    #endif
    }

//! Destructor
MolecularForceCompute::~MolecularForceCompute()
    {
    m_pdata->getParticleSortSignal().disconnect<MolecularForceCompute, &MolecularForceCompute::setDirty>(this);
    }

#ifdef ENABLE_CUDA
void MolecularForceCompute::initMoleculesGPU()
    {
    if (m_prof) m_prof->push(m_exec_conf,"init molecules");

    unsigned int nptl_local = m_pdata->getN() + m_pdata->getNGhosts();

    unsigned int n_local_molecules = 0;

    // maximum molecule length
    unsigned int nmax = 0;

    // number of local particles that are part of molecules
    unsigned int n_local_ptls_in_molecules = 0;

    // resize to maximum possible number of local molecules
    m_molecule_length.resize(nptl_local);
    m_molecule_idx.resize(nptl_local);

    ScopedAllocation<unsigned int> d_idx_sorted_by_tag(m_exec_conf->getCachedAllocator(), nptl_local);
    ScopedAllocation<unsigned int> d_local_molecules_lowest_idx(m_exec_conf->getCachedAllocator(), nptl_local);

        {
        ArrayHandle<unsigned int> d_molecule_tag(m_molecule_tag, access_location::device, access_mode::read);
        ArrayHandle<unsigned int> d_tag(m_pdata->getTags(), access_location::device, access_mode::read);
        ArrayHandle<unsigned int> d_molecule_length(m_molecule_length, access_location::device, access_mode::overwrite);
        ArrayHandle<unsigned int> d_molecule_idx(m_molecule_idx, access_location::device, access_mode::overwrite);

        // temporary buffers
        ScopedAllocation<unsigned int> d_local_molecule_tags(m_exec_conf->getCachedAllocator(), nptl_local);
        ScopedAllocation<unsigned int> d_local_unique_molecule_tags(m_exec_conf->getCachedAllocator(), m_n_molecules_global);
        ScopedAllocation<unsigned int> d_sorted_by_tag(m_exec_conf->getCachedAllocator(), nptl_local);
        ScopedAllocation<unsigned int> d_idx_sorted_by_molecule_and_tag(m_exec_conf->getCachedAllocator(), nptl_local);

        ScopedAllocation<unsigned int> d_lowest_idx(m_exec_conf->getCachedAllocator(), m_n_molecules_global);
        ScopedAllocation<unsigned int> d_lowest_idx_sort(m_exec_conf->getCachedAllocator(), m_n_molecules_global);
        ScopedAllocation<unsigned int> d_lowest_idx_in_molecules(m_exec_conf->getCachedAllocator(), m_n_molecules_global);
        ScopedAllocation<unsigned int> d_lowest_idx_by_molecule_tag(m_exec_conf->getCachedAllocator(), m_molecule_tag.getNumElements());

        gpu_sort_by_molecule(nptl_local,
            d_tag.data,
            d_molecule_tag.data,
            d_local_molecule_tags.data,
            d_local_molecules_lowest_idx.data,
            d_local_unique_molecule_tags.data,
            d_molecule_idx.data,
            d_sorted_by_tag.data,
            d_idx_sorted_by_tag.data,
            d_idx_sorted_by_molecule_and_tag.data,
            d_lowest_idx.data,
            d_lowest_idx_sort.data,
            d_lowest_idx_in_molecules.data,
            d_lowest_idx_by_molecule_tag.data,
            d_molecule_length.data,
            n_local_molecules,
            nmax,
            n_local_ptls_in_molecules,
            m_exec_conf->getCachedAllocator(),
            m_exec_conf->isCUDAErrorCheckingEnabled());

        if (m_exec_conf->isCUDAErrorCheckingEnabled())
            CHECK_CUDA_ERROR();
        }

    // set up indexer
    m_molecule_indexer = Index2D(nmax, n_local_molecules);

    m_exec_conf->msg->notice(7) << "MolecularForceCompute: " << n_local_molecules << " molecules, "
        << n_local_ptls_in_molecules << " particles in molecules " << std::endl;

    // resize molecule list
    m_molecule_list.resize(m_molecule_indexer.getNumElements());

    // resize molecule lookup to size of local particle data
    m_molecule_order.resize(m_pdata->getMaxN());

        {
        // write out molecule list and order
        ArrayHandle<unsigned int> d_molecule_list(m_molecule_list, access_location::device, access_mode::overwrite);
        ArrayHandle<unsigned int> d_molecule_order(m_molecule_order, access_location::device, access_mode::overwrite);
        ArrayHandle<unsigned int> d_molecule_idx(m_molecule_idx, access_location::device, access_mode::read);

        m_tuner_fill->begin();
        unsigned int block_size = m_tuner_fill->getParam();

        gpu_fill_molecule_table(nptl_local,
            n_local_ptls_in_molecules,
            m_molecule_indexer,
            d_molecule_idx.data,
            d_local_molecules_lowest_idx.data,
            d_idx_sorted_by_tag.data,
            d_molecule_list.data,
            d_molecule_order.data,
            block_size,
            m_exec_conf->getCachedAllocator());

        if (m_exec_conf->isCUDAErrorCheckingEnabled())
            CHECK_CUDA_ERROR();

        m_tuner_fill->end();
        }

    // distribute molecules evenly over GPUs
    // NOTE: going forward we could slave the GPU partition of the molecules
    // to that of the local particles in the ParticleData
    m_gpu_partition = GPUPartition(m_exec_conf->getGPUIds());
    m_gpu_partition.setN(n_local_molecules);

    if (m_exec_conf->allConcurrentManagedAccess())
        {
        auto gpu_map = m_exec_conf->getGPUIds();

        for (unsigned int idev = 0; idev < m_exec_conf->getNumActiveGPUs(); ++idev)
            {
            std::pair<unsigned int, unsigned int> range = m_gpu_partition.getRange(idev);
            unsigned int nelem = range.second - range.first;

            if (nelem == 0)
                continue;

            cudaMemAdvise(m_molecule_length.get()+range.first,
                sizeof(unsigned int)*nelem,
                cudaMemAdviseSetPreferredLocation,
                gpu_map[idev]);
            cudaMemPrefetchAsync(m_molecule_length.get()+range.first,
                sizeof(unsigned int)*nelem,
                gpu_map[idev]);

            if (m_molecule_indexer.getW() == 0)
                continue;

            cudaMemAdvise(m_molecule_list.get()+m_molecule_indexer(0,range.first),
                sizeof(unsigned int)*nelem*m_molecule_indexer.getW(),
                cudaMemAdviseSetPreferredLocation, gpu_map[idev]);
            cudaMemPrefetchAsync(m_molecule_list.get()+m_molecule_indexer(0,range.first),
                sizeof(unsigned int)*nelem*m_molecule_indexer.getW(),
                gpu_map[idev]);
            }

        for (unsigned int idev = 0; idev < m_exec_conf->getNumActiveGPUs(); ++idev)
            {
            auto range = m_pdata->getGPUPartition().getRange(idev);
            unsigned int nelem =  range.second - range.first;

            // skip if no hint set
            if (!nelem)
                continue;

            cudaMemAdvise(m_molecule_idx.get()+range.first, sizeof(unsigned int)*nelem, cudaMemAdviseSetPreferredLocation, gpu_map[idev]);
            cudaMemPrefetchAsync(m_molecule_idx.get()+range.first, sizeof(unsigned int)*nelem, gpu_map[idev]);
            }

        CHECK_CUDA_ERROR();
        }

    if (m_prof) m_prof->pop(m_exec_conf);
    }
#endif

void MolecularForceCompute::initMolecules()
    {
    // return early if no molecules are defined
    if (!m_n_molecules_global) return;

    m_exec_conf->msg->notice(7) << "MolecularForceCompute initializing molecule table" << std::endl;

    #ifdef ENABLE_CUDA
    if (m_exec_conf->isCUDAEnabled())
        {
        initMoleculesGPU();
        return;
        }
    #endif

    if (m_prof) m_prof->push("init molecules");

    // construct local molecule table
    unsigned int nptl_local = m_pdata->getN() + m_pdata->getNGhosts();

    ArrayHandle<unsigned int> h_molecule_tag(m_molecule_tag, access_location::host, access_mode::read);
    ArrayHandle<unsigned int> h_tag(m_pdata->getTags(), access_location::host, access_mode::read);
    ArrayHandle<unsigned int> h_rtag(m_pdata->getRTags(), access_location::host, access_mode::read);

    std::set<unsigned int> local_molecule_tags;

    unsigned int n_local_molecules = 0;

    std::vector<unsigned int> local_molecule_idx(nptl_local, NO_MOLECULE);

    // resize molecule lookup to size of local particle data
    m_molecule_order.resize(m_pdata->getMaxN());

    // keep track of particle with lowest tag
    std::map<unsigned int, unsigned int> lowest_tag_by_molecule;

    for (unsigned int iptl = 0; iptl < nptl_local; ++iptl)
        {
        unsigned int tag = h_tag.data[iptl];
        assert(tag < m_molecule_tag.getNumElements());

        unsigned int mol_tag = h_molecule_tag.data[tag];
        if (mol_tag == NO_MOLECULE)
            continue;

        auto it = lowest_tag_by_molecule.find(mol_tag);
        unsigned int min_tag = tag;
        if (it != lowest_tag_by_molecule.end())
            min_tag = std::min(it->second, tag);

        lowest_tag_by_molecule[mol_tag] = min_tag;
        }

    // sort local molecules by index of lowest tag, and inside the molecule by ptl tag
    std::map<unsigned int, std::set<unsigned int> > local_molecules_sorted;

    for (unsigned int iptl = 0; iptl < nptl_local; ++iptl)
        {
        unsigned int tag = h_tag.data[iptl];
        assert(tag < m_molecule_tag.getNumElements());

        unsigned int mol_tag = h_molecule_tag.data[tag];
        if (mol_tag == NO_MOLECULE) continue;

        unsigned int lowest_tag = lowest_tag_by_molecule[mol_tag];
        unsigned int lowest_idx = h_rtag.data[lowest_tag];
        assert(lowest_idx < m_pdata->getN() + m_pdata->getNGhosts());

        local_molecules_sorted[lowest_idx].insert(tag);
        }

    n_local_molecules = local_molecules_sorted.size();

    m_exec_conf->msg->notice(7) << "MolecularForceCompute: " << n_local_molecules << " molecules" << std::endl;

    m_molecule_length.resize(n_local_molecules);

    ArrayHandle<unsigned int> h_molecule_length(m_molecule_length, access_location::host, access_mode::overwrite);

    // reset lengths
    for (unsigned int imol = 0; imol < n_local_molecules; ++imol)
        {
        h_molecule_length.data[imol] = 0;
        }

    // count molecule lengths
    unsigned int i = 0;
    for (auto it = local_molecules_sorted.begin(); it != local_molecules_sorted.end(); ++it)
        {
        h_molecule_length.data[i++] = it->second.size();
        }

    // find maximum length
    unsigned nmax = 0;
    for (unsigned int imol = 0; imol < n_local_molecules; ++imol)
        {
        if (h_molecule_length.data[imol] > nmax)
            {
            nmax = h_molecule_length.data[imol];
            }
        }

    // set up indexer
    m_molecule_indexer = Index2D(nmax, n_local_molecules);

    // resize molecule list
    m_molecule_list.resize(m_molecule_indexer.getNumElements());

    // reset lengths again
    for (unsigned int imol = 0; imol < n_local_molecules; ++imol)
        {
        h_molecule_length.data[imol] = 0;
        }

    // reset molecule order
    ArrayHandle<unsigned int> h_molecule_order(m_molecule_order, access_location::host, access_mode::overwrite);
    memset(h_molecule_order.data, 0, sizeof(unsigned int)*(m_pdata->getN() + m_pdata->getNGhosts()));

    // resize reverse-lookup
    m_molecule_idx.resize(nptl_local);

    // fill molecule list
    ArrayHandle<unsigned int> h_molecule_list(m_molecule_list, access_location::host, access_mode::overwrite);
    ArrayHandle<unsigned int> h_molecule_idx(m_molecule_idx, access_location::host, access_mode::overwrite);

    // reset reverse lookup
    memset(h_molecule_idx.data, 0, sizeof(unsigned int)*nptl_local);

    unsigned int i_mol = 0;
    for (auto it_mol = local_molecules_sorted.begin(); it_mol != local_molecules_sorted.end(); ++it_mol)
        {
        for (std::set<unsigned int>::iterator it_tag = it_mol->second.begin(); it_tag != it_mol->second.end(); ++it_tag)
            {
            unsigned int n = h_molecule_length.data[i_mol]++;
            unsigned int ptl_idx = h_rtag.data[*it_tag];
            assert(ptl_idx < m_pdata->getN() + m_pdata->getNGhosts());
            h_molecule_list.data[m_molecule_indexer(n, i_mol)] = ptl_idx;
            h_molecule_idx.data[ptl_idx] = i_mol;
            h_molecule_order.data[ptl_idx] = n;
            }
        i_mol ++;
        }

    if (m_prof) m_prof->pop(m_exec_conf);
    }

void export_MolecularForceCompute(py::module& m)
    {
    py::class_< MolecularForceCompute, std::shared_ptr<MolecularForceCompute> >(m, "MolecularForceCompute", py::base<ForceConstraint>())
    .def(py::init< std::shared_ptr<SystemDefinition> >())
    ;
    }
