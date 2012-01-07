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

// Maintainer: joaander

#include "ParticleData.cuh"

#ifdef WIN32
#include <cassert>
#else
#include <assert.h>
#endif

/*! \file ParticleData.cu
    \brief Contains GPU kernel code and data structure functions used by ParticleData
*/

//! Kernel for un-interleaving float4 input into float output
/*! \param d_out Device pointer to write un-interleaved output
    \param d_in Device pointer to read interleaved input
    \param N Number of elements in input
    \param pitch Spacing of arrays through the output

    \pre N/block_size + 1 blocks are run on the device
*/
extern "C" __global__ void uninterleave_float4_kernel(float *d_out, float4 *d_in, int N, int pitch)
    {
    int pidx  = blockDim.x * blockIdx.x + threadIdx.x;
    
    if (pidx < N)
        {
        float4 in = d_in[pidx];
        
        d_out[pidx] = in.x;
        d_out[pidx+pitch] = in.y;
        d_out[pidx+pitch+pitch] = in.z;
        d_out[pidx+pitch+pitch+pitch] = in.w;
        }
    }


/*! The most efficient data storage on the device is to put x,y,z,type into a float4
    data structure. The most efficient storage on the CPU is x,y,z,type each as
    separate arrays. Translation between the two is best done on the device, and
    memory transfers done with one big cudaMemcpy. This function, and its sister
    gpu_interleave_float4() perform the transformation between a float* with x,y,z,type
    packed non-interleaved to a float4* storing the same values interleaved.

    Performance is best when pitch is a multiple of 64.

    \param d_out Device pointer to write output to
    \param d_in Device pointer to read input from
    \param N Number of elements to interleave
    \param pitch Spacing between \c x[0] and \c y[0] in \a d_out

    \post A code snipped best describes what is done:
    \verbatim
    d_out[i] = d_in[i].x
    d_out[i+pitch] = d_in[i].y
    d_out[i+pitch*2] = d_in[i].z
    d_out[i+pitch*3] = d_in[i].w
    \endverbatim

    \returns Any error code from the kernel call retrieved via cudaGetLastError()
    \note Always returns cudaSuccess in release builds for performance reasons
*/
cudaError_t gpu_uninterleave_float4(float *d_out, float4 *d_in, int N, int pitch)
    {
    assert(pitch >= N);
    assert(d_out);
    assert(d_in);
    assert(N > 0);
    
    const int M = 64;
    uninterleave_float4_kernel<<< N/M+1, M >>>(d_out, d_in, N, pitch);
    
    return cudaSuccess;
    }

//! Kernel for interleaving float input into float4 output
/*! \param d_out Device pointer to write interleaved output
    \param d_in Device pointer to read non-interleaved input
    \param N Number of elements in output
    \param pitch Spacing of arrays through the input

    \pre N/block_size + 1 blocks are run on the device
*/
extern "C" __global__ void interleave_float4_kernel(float4 *d_out, float *d_in, int N, int pitch)
    {
    int pidx  = blockDim.x * blockIdx.x + threadIdx.x;
    
    if (pidx < N)
        {
        float x = d_in[pidx];
        float y = d_in[pidx+pitch];
        float z = d_in[pidx+pitch+pitch];
        float w = d_in[pidx+pitch+pitch+pitch];
        
        float4 out;
        out.x = x;
        out.y = y;
        out.z = z;
        out.w = w;
        d_out[pidx] = out;
        }
    }

/*! See gpu_uninterleave_float4() for details.
    \param d_out Device pointer to write output to
    \param d_in Device pointer to read input from
    \param N Number of elements to interleave
    \param pitch Spacing between \c x[0] and \c y[0] in \a d_in

    \returns Any error code from the kernel call retrieved via cudaGetLastError()
    \note Always returns cudaSuccess in release builds for performance reasons
*/
cudaError_t gpu_interleave_float4(float4 *d_out, float *d_in, int N, int pitch)
    {
    assert(pitch >= N);
    assert(d_out);
    assert(d_in);
    assert(N > 0);
    
    const int M = 64;
    interleave_float4_kernel<<< N/M+1, M >>>(d_out, d_in, N, pitch);
    
    return cudaSuccess;
    }


