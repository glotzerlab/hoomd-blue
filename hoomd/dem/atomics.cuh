// Copyright (c) 2009-2019 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.

// Maintainer: mspells

#ifndef __DEM_ATOMICS_CUH__
#define __DEM_ATOMICS_CUH__

template<typename Real>
__device__ inline Real genAtomicAdd(Real *address, Real val)
    {
    return atomicAdd(address, val);
    }

//! Atomic add for doubles, taken from the CUDA manual
template<>
__device__ inline double genAtomicAdd(double* address, double val)
    {
    unsigned long long int* address_as_ull =
        (unsigned long long int*)address;
    unsigned long long int old = *address_as_ull, assumed;
    do {
        assumed = old;
        old = atomicCAS(address_as_ull, assumed,
            __double_as_longlong(val +
                __longlong_as_double(assumed)));
        } while (assumed != old);
    return __longlong_as_double(old);
    }

#endif
