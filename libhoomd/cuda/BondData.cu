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

#include "BondData.cuh"
#include "ParticleData.cuh"

#ifdef WIN32
#include <cassert>
#else
#include <assert.h>
#endif

/*! \file BondData.cu
    \brief Implements the helper functions (GPU version) for updating the GPU bond table
*/

#define MAX(i,j) (i > j ? i : j)

//! Kernel to find the maximum number of angles per particle
__global__ void gpu_find_max_bond_number_kernel(const uint2 *bonds,
                                             const unsigned int *d_rtag,
                                             unsigned int *d_n_bonds,
                                             unsigned int num_bonds,
                                             unsigned int N,
                                             unsigned int n_ghosts,
                                             const unsigned int cur_max,
                                             unsigned int *condition)
    {
    int bond_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (bond_idx >= num_bonds)
        return;

    uint2 bond = bonds[bond_idx];
    unsigned int tag1 = bond.x;
    unsigned int tag2 = bond.y;
    unsigned int idx1 = d_rtag[tag1];
    unsigned int idx2 = d_rtag[tag2];

    bool bond_needed = false;
    bool bond_valid = true;
    if (idx1 < N)
        {
        unsigned int n = atomicInc(&d_n_bonds[idx1], 0xffffffff);
        bond_valid &= (idx2 < N + n_ghosts);
        if (n >= cur_max) bond_needed = true;
        }
    if (idx2 < N)
        {
        unsigned int n = atomicInc(&d_n_bonds[idx2], 0xffffffff);
        bond_valid &= (idx1 < N + n_ghosts);
        if (n >= cur_max) bond_needed = true;
        }

    if (bond_needed)
        atomicOr(condition, 1);
    if (!bond_valid)
        atomicOr(condition, 2);
    }

//! Kernel to fill the GPU bond table
__global__ void gpu_fill_gpu_bond_table(const uint2 *bonds,
                                        const unsigned int *bond_type,
                                        uint2 *gpu_btable,
                                        const unsigned int pitch,
                                        const unsigned int *d_rtag,
                                        unsigned int *d_n_bonds,
                                        unsigned int num_bonds,
                                        unsigned int N)
    {
    int bond_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (bond_idx >= num_bonds)
        return;

    uint2 bond = bonds[bond_idx];
    unsigned int tag1 = bond.x;
    unsigned int tag2 = bond.y;
    unsigned int type = bond_type[bond_idx];
    unsigned int idx1 = d_rtag[tag1];
    unsigned int idx2 = d_rtag[tag2];

    if (idx1 < N)
        {
        unsigned int num1 = atomicInc(&d_n_bonds[idx1],0xffffffff);
        gpu_btable[num1*pitch+idx1] = make_uint2(idx2,type);
        }
    if (idx2 < N)
        {
        unsigned int num2 = atomicInc(&d_n_bonds[idx2],0xffffffff);
        gpu_btable[num2*pitch+idx2] = make_uint2(idx1,type);
        }
    }


//! Find the maximum number of bonds per particle
/*! \param d_n_bonds Number of bonds per particle (return array)
    \param d_bonds Array of bonds
    \param num_bonds Size of bond array
    \param N Number of particles in the system
    \param d_rtag Array of reverse-lookup particle tag . particle index
    \param cur_max Current maximum bonded particle number
    \param d_condition Condition variable, set to unequal zero if we exceed the maximum numbers
 */
cudaError_t gpu_find_max_bond_number(unsigned int *d_n_bonds,
                                     const uint2 *d_bonds,
                                     const unsigned int num_bonds,
                                     const unsigned int N,
                                     const unsigned int n_ghosts,
                                     const unsigned int *d_rtag,
                                     const unsigned int cur_max,
                                     unsigned int *d_condition)
    {
    assert(d_bonds);
    assert(d_rtag);
    assert(d_n_bonds);

    unsigned int block_size = 512;

    // clear n_bonds array
    cudaMemset(d_n_bonds, 0, sizeof(unsigned int) * N);

    gpu_find_max_bond_number_kernel<<<num_bonds/block_size + 1, block_size>>>(d_bonds,
                                                                              d_rtag,
                                                                              d_n_bonds,
                                                                              num_bonds,
                                                                              N,
                                                                              n_ghosts,
                                                                              cur_max,
                                                                              d_condition);

    return cudaSuccess;
    }

//! Construct the GPU bond table
/*! \param d_gpu_bondtable Pointer to the bond table on the GPU
    \param d_n_bonds Number of bonds per particle (return array)
    \param d_bonds Bonds array
    \param d_bond_type Array of bond types
    \param d_rtag Reverse-lookup tag->index
    \param num_bonds Number of bonds in bond list
    \param pitch Pitch of 2D bondtable array
    \param N Number of particles
 */
cudaError_t gpu_create_bondtable(uint2 *d_gpu_bondtable,
                                 unsigned int *d_n_bonds,
                                 const uint2 *d_bonds,
                                 const unsigned int *d_bond_type,
                                 const unsigned int *d_rtag,
                                 const unsigned int num_bonds,
                                 unsigned int pitch,
                                 unsigned int N)
    {
    unsigned int block_size = 512;

    // clear n_bonds array
    cudaMemset(d_n_bonds, 0, sizeof(unsigned int) * N);

    gpu_fill_gpu_bond_table<<<num_bonds/block_size + 1, block_size>>>(d_bonds,
                                                                      d_bond_type,
                                                                      d_gpu_bondtable,
                                                                      pitch,
                                                                      d_rtag,
                                                                      d_n_bonds,
                                                                      num_bonds,
                                                                      N);
    return cudaSuccess;
    }

//! Kernel to mark duplicate received bonds
__global__ void gpu_mark_recv_bond_duplicates_kernel(const unsigned int n_bonds,
                                         const bond_element *recv_bonds,
                                         unsigned int *bond_remove_mask,
                                         const unsigned int n_recv_bonds,
                                         unsigned int *bond_rtag,
                                         unsigned char *recv_bond_active,
                                         unsigned int *n_duplicate_recv_bonds)
    {
    unsigned int recv_idx = blockIdx.x *blockDim.x + threadIdx.x;

    if (recv_idx >= n_recv_bonds) return;

    const bond_element& el = recv_bonds[recv_idx];
    unsigned int tag = el.tag;
   
    // stage the bond
    unsigned int rtag = atomicMin(&bond_rtag[tag], (unsigned int) BOND_NOT_LOCAL-1);

    bool duplicate = false;

    if (rtag != BOND_NOT_LOCAL)
        {
        bool remove = false;
        if (rtag < n_bonds)
            remove = bond_remove_mask[rtag];

        // if the bond is a duplicate of a local bond which is not removed, mark it
        if (! remove)
            {
            duplicate = true;
            atomicInc(n_duplicate_recv_bonds, 0xffffffff);
            }
        }

    recv_bond_active[recv_idx] = duplicate ? 0 : 1;
    }

//! Mark duplicate bonds received
/*! \param n_bonds Number of bonds in local bond table
    \param d_recv_bonds Buffer of received bonds
    \param d_bond_remove_mask Flags for every local bond to indicate removal
    \param n_recv_bonds Number of bonds received
    \param d_bond_rtag Bond tag->idx lookup
    \param d_recv_bond_active Per-received bond flag, 1 if unique, 0 if duplicate (return values)
    \param d_n_duplicate_recv_bonds Number of duplicates found (return value)
 */
void gpu_mark_recv_bond_duplicates(const unsigned int n_bonds,
                                   const bond_element *d_recv_bonds,
                                   unsigned int *d_bond_remove_mask,
                                   const unsigned int n_recv_bonds,
                                   unsigned int *d_bond_rtag,
                                   unsigned char *d_recv_bond_active,
                                   unsigned int *d_n_duplicate_recv_bonds)
    {
    cudaMemsetAsync(d_n_duplicate_recv_bonds, 0, sizeof(unsigned int));

    unsigned int block_size = 512;

    gpu_mark_recv_bond_duplicates_kernel<<<n_recv_bonds/block_size+1,block_size>>>(
        n_bonds,
        d_recv_bonds,
        d_bond_remove_mask,
        n_recv_bonds,
        d_bond_rtag,
        d_recv_bond_active,
        d_n_duplicate_recv_bonds);
    }

//! Kernel to backfill the local bond table with received bonds and remove non-local bonds
__global__ void gpu_fill_bondtable_kernel(const unsigned int old_n_bonds,
                                               const unsigned int n_recv_bonds,
                                               const unsigned int n_unique_recv_bonds,
                                               const unsigned int n_remove_bonds,
                                               const unsigned int *remove_mask,
                                               const unsigned char *recv_bond_active,
                                               const bond_element *recv_buf,
                                               uint2 *bonds,
                                               unsigned int *bond_type,
                                               unsigned int *bond_tag,
                                               unsigned int *bond_rtag,
                                               unsigned int *n_fetch_bond)
    {
    unsigned int bond_idx = blockDim.x * blockIdx.x + threadIdx.x;

    unsigned int new_nbonds = old_n_bonds - n_remove_bonds + n_unique_recv_bonds;
    
    if (bond_idx >= old_n_bonds + n_unique_recv_bonds) return;

    bool replace = true;

    if (bond_idx < old_n_bonds)
        {
        replace = remove_mask[bond_idx];

        // reset rtag
        if (replace) bond_rtag[bond_tag[bond_idx]] = BOND_NOT_LOCAL;
        }
    
    if (replace && bond_idx < new_nbonds)
        {
        // try to atomically fetch a bond from the received list, ignore duplicates
        bool active = false;
        unsigned int n;
        while (!active)
            {
            n = atomicInc(n_fetch_bond, 0xffffffff);
            if (n < n_recv_bonds)
                active = recv_bond_active[n];
            else
                active = true;
            }

        if (n < n_recv_bonds) 
            {
            // copy over receive buffer data
            const bond_element &el= recv_buf[n];

            bonds[bond_idx] = el.bond;
            bond_type[bond_idx] = el.type;
            bond_tag[bond_idx] = el.tag;
            }
        else
            {
            unsigned int fetch_idx = new_nbonds + (n - n_recv_bonds);
            bool remove = remove_mask[fetch_idx];

            // we should not normally read past the end of the array, if the number
            // of removed particles correctly reflects the number of remove flags set
            while (remove) {
                // reset rtags as we go
                bond_rtag[bond_tag[fetch_idx]] = BOND_NOT_LOCAL;

                n = atomicInc(n_fetch_bond, 0xffffffff);

                fetch_idx = new_nbonds + (n - n_recv_bonds);
                remove = remove_mask[fetch_idx];
                };

            // backfill with a bond from the end
            bonds[bond_idx] = bonds[fetch_idx];
            bond_type[bond_idx] = bond_type[fetch_idx];
            bond_tag[bond_idx] = bond_tag[fetch_idx];
            }
         } // if replace
    }

//! Backfill local bond table with received bonds and remove non-local bonds
/*! \param old_n_bonds Current size of bond table
    \param n_recv_bonds Size of bond receive buffer
    \param n_unique_recv_bonds Number of unique received bonds
    \param n_remove_bonds Number of bonds to be removed from local bond table
    \param d_remove_mask Flag for every bond, 1 if bond is to be removed, 0 otherwise
    \param d_recv_bond_active Flag for every received bond, 1 if unique, 0 if duplicate
    \param d_recv_buf Buffer of received bonds
    \param d_bonds Local bond table
    \param d_bond_type Local list of bond types
    \param d_bond_tag Local list of bond tags
    \param d_bond_rtag Bond tag->idx lookup table
    \param d_n_fetch_bond Temporary counter for backfilling of bonds
*/
void gpu_fill_bond_bondtable(const unsigned int old_n_bonds,
                             const unsigned int n_recv_bonds,
                             const unsigned int n_unique_recv_bonds,
                             const unsigned int n_remove_bonds,
                             const unsigned int *d_remove_mask,
                             const unsigned char *d_recv_bond_active,
                             const bond_element *d_recv_buf,
                             uint2 *d_bonds,
                             unsigned int *d_bond_type,
                             unsigned int *d_bond_tag,
                             unsigned int *d_bond_rtag,
                             unsigned int *d_n_fetch_bond)
    {
    unsigned int block_size = 512;
    
    cudaMemsetAsync(d_n_fetch_bond, 0, sizeof(unsigned int));

    unsigned int end = old_n_bonds + n_unique_recv_bonds;

    gpu_fill_bondtable_kernel<<<end/block_size+1,block_size>>>(
        old_n_bonds,
        n_recv_bonds,
        n_unique_recv_bonds,
        n_remove_bonds,
        d_remove_mask,
        d_recv_bond_active,
        d_recv_buf,
        d_bonds,
        d_bond_type,
        d_bond_tag,
        d_bond_rtag,
        d_n_fetch_bond);
    }

//! Kernel to update reverse-lookup tags for bonds
__global__ void gpu_update_bond_rtags_kernel(unsigned int *bond_rtag,
                                      const unsigned int *bond_tag,
                                      const unsigned int num_bonds)
    {
    unsigned int bond_idx = blockIdx.x*blockDim.x + threadIdx.x;

    if (bond_idx >= num_bonds) return;

    bond_rtag[bond_tag[bond_idx]] = bond_idx;
    }

//! Update the bond tag ->idx lookup table
/*! \param d_bond_rtag Reverse-lookup table
    \param d_bond_tag Local list of bond tags
    \param num_bonds Number of local bonds
 */
void gpu_update_bond_rtags(unsigned int *d_bond_rtag,
                           const unsigned int *d_bond_tag,
                           const unsigned int num_bonds)
    {
    unsigned int block_size = 512;

    gpu_update_bond_rtags_kernel<<<num_bonds/block_size+1, block_size>>>(d_bond_rtag,
                                                                         d_bond_tag,
                                                                         num_bonds);
    }
