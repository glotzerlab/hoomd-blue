// Copyright (c) 2009-2019 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.


// Maintainer: jglaser

/*! \file MolecularForceCompute.cuh
    \brief Contains GPU kernel code used by MolecularForceCompute
*/
#ifndef __MOLECULAR_FORCE_COMPUTE_CUH__
#define __MOLECULAR_FORCE_COMPUTE_CUH__

#include "hoomd/Index1D.h"
#include "hoomd/CachedAllocator.h"

#ifdef NVCC
const unsigned int NO_MOLECULE = (unsigned int)0xffffffff;
#endif

cudaError_t gpu_sort_by_molecule(unsigned int nptl,
    const unsigned int *d_tag,
    const unsigned int *d_molecule_tag,
    unsigned int *d_local_molecule_tags,
    unsigned int *d_local_molecules_lowest_idx,
    unsigned int *d_local_unique_molecule_tags,
    unsigned int *d_local_molecule_idx,
    unsigned int *d_sorted_by_tag,
    unsigned int *d_idx_sorted_by_tag,
    unsigned int *d_idx_sorted_by_molecule_and_tag,
    unsigned int *d_lowest_idx,
    unsigned int *d_lowest_idx_sort,
    unsigned int *d_lowest_idx_in_molecules,
    unsigned int *d_lowest_idx_by_molecule_tag,
    unsigned int *d_molecule_length,
    unsigned int &n_local_molecules,
    unsigned int &max_len,
    unsigned int &n_local_ptls_in_molecules,
    CachedAllocator& alloc,
    bool check_cuda);

cudaError_t gpu_fill_molecule_table(
    unsigned int nptl,
    unsigned int n_local_ptls_in_molecules,
    Index2D molecule_idx,
    const unsigned int *d_molecule_idx,
    const unsigned int *d_local_molecule_tags,
    const unsigned int *d_idx_sorted_by_tag,
    unsigned int *d_molecule_list,
    unsigned int *d_molecule_order,
    unsigned int block_size,
    CachedAllocator& alloc);

#endif
