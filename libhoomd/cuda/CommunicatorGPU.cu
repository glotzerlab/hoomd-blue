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

// Maintainer: jglaser

/*! \file CommunicatorGPU.cu
    \brief Implementation of communication algorithms on the GPU
*/

#ifdef ENABLE_MPI
#include "CommunicatorGPU.cuh"
#include "ParticleData.cuh"

#include <thrust/device_vector.h>
#include <thrust/binary_search.h>
#include <thrust/iterator/permutation_iterator.h>
#include <thrust/gather.h>
#include <thrust/scatter.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/copy.h>
#include <thrust/partition.h>
#include <thrust/count.h>

#ifdef WIN32
#include <cassert>
#else
#include <assert.h>
#endif

using namespace thrust;

//! Select local particles that within a boundary layer of the neighboring domain in a given direction
struct make_nonbonded_plan : thrust::unary_function<thrust::tuple<float4, unsigned char>, unsigned char>
    {
    float3 lo;
    float3 hi;
    const float r_ghost; //!< Width of boundary layer

    //! Constructor
    /*! \param _box Local box dimensions
     * \param _r_ghost Width of boundary layer
     */
    make_nonbonded_plan(const BoxDim _box, float _r_ghost)
        : r_ghost(_r_ghost)
        {
        lo = _box.getLo();
        hi = _box.getHi();
        }

    //! Make exchange plan
    /*! \param t Tuple of Particle position to check and current plan
        \returns The updated plan for this particle
     */
    __host__ __device__ unsigned char operator()(const thrust::tuple<float4, unsigned char>& t)
        {
        float4 pos = thrust::get<0>(t);

        unsigned char plan = thrust::get<1>(t);
        if (pos.x >= hi.x  - r_ghost)
            plan |= send_east;

        if (pos.x < lo.x + r_ghost)
            plan |= send_west;

        if (pos.y >= hi.y  - r_ghost)
            plan |= send_north;

        if (pos.y < lo.y + r_ghost)
            plan |= send_south;

        if (pos.z >= hi.z  - r_ghost)
            plan |= send_up;

        if (pos.z < lo.z + r_ghost)
            plan |= send_down;

        return plan;
        }
     };

thrust::device_vector<unsigned int> *keys;       //!< Temporary vector of sort keys

unsigned int *d_n_send_particles;  //! Counter for construction of atom send lists
unsigned int *d_n_copy_ghosts;     //! Counter for ghost list construction
unsigned int *d_n_copy_ghosts_r;     //! Counter for ghost list construction (reverse direction)

void gpu_allocate_tmp_storage()
    {
    keys = new thrust::device_vector<unsigned int>;

    cudaMalloc(&d_n_send_particles,sizeof(unsigned int));
    cudaMalloc(&d_n_copy_ghosts, sizeof(unsigned int));
    cudaMalloc(&d_n_copy_ghosts_r, sizeof(unsigned int));
    }

void gpu_deallocate_tmp_storage()
    {
    delete keys;

    cudaFree(d_n_send_particles);
    cudaFree(d_n_copy_ghosts);
    cudaFree(d_n_copy_ghosts_r);
    }

//! GPU Kernel to find incomplete bonds
/*! \param btable Bond table
 * \param plan Plan array
 * \param d_pos Array of particle positions
 * \param d_rtag Array of global reverse-lookup tags
 * \param box The local box dimensions
 * \param N number of (local) particles
 * \param n_bonds Number of bonds in bond table
 */
__global__ void gpu_mark_particles_in_incomplete_bonds_kernel(const uint2 *btable,
                                                         unsigned char *plan,
                                                         const float4 *pos,
                                                         const unsigned int *d_rtag,
                                                         const unsigned int N,
                                                         const unsigned int n_bonds,
                                                         const Scalar3 lo,
                                                         const Scalar3 L2)
    {
    unsigned int bond_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (bond_idx >= n_bonds)
        return;

    uint2 bond = btable[bond_idx];

    unsigned int tag1 = bond.x;
    unsigned int tag2 = bond.y;
    unsigned int idx1 = d_rtag[tag1];
    unsigned int idx2 = d_rtag[tag2];

    if ((idx1 >= N) && (idx2 < N))
        {
        // send particle with index idx2 to neighboring domains
        Scalar4 postype = pos[idx2];
        // Multiple threads may update the plan simultaneously, but this should
        // be safe, since they store the same result
        unsigned char p = plan[idx2];
        p |= (postype.x > lo.x + L2.x) ? send_east : send_west;
        p |= (postype.y > lo.y + L2.y) ? send_north : send_south;
        p |= (postype.z > lo.z + L2.z) ? send_up : send_down;
        plan[idx2] = p;
        }
    else if ((idx1 < N) && (idx2 >= N))
        {
        // send particle with index idx1 to neighboring domains
        Scalar4 postype = pos[idx1];
        // Multiple threads may update the plan simultaneously, but this should
        // be safe, since they store the same result
        unsigned char p = plan[idx1];
        p |= (postype.x > lo.x + L2.x) ? send_east : send_west;
        p |= (postype.y > lo.y + L2.y) ? send_north : send_south;
        p |= (postype.z > lo.z + L2.z) ? send_up : send_down;
        plan[idx1] = p;
       }
    }

//! Mark particles in incomplete bonds for sending
/* \param d_btable GPU bond table
 * \param d_plan Plan array
 * \param d_rtag Array of global reverse-lookup tags
 * \param N number of (local) particles
 * \param n_bonds Total number of bonds in bond table
 */
void gpu_mark_particles_in_incomplete_bonds(const uint2 *d_btable,
                                          unsigned char *d_plan,
                                          const float4 *d_pos,
                                          const unsigned int *d_rtag,
                                          const unsigned int N,
                                          const unsigned int n_bonds,
                                          const BoxDim box)
    {
    assert(d_btable);
    assert(d_plan);
    assert(N>0);

    unsigned int block_size = 512;
    Scalar3 lo = box.getLo();
    Scalar3 L2 = box.getL()/2.0f;
    gpu_mark_particles_in_incomplete_bonds_kernel<<<n_bonds/block_size + 1, block_size>>>(d_btable,
                                                                                    d_plan,
                                                                                    d_pos,
                                                                                    d_rtag,
                                                                                    N,
                                                                                    n_bonds,
                                                                                    lo,
                                                                                    L2);
    }

//! Helper kernel to reorder particle data, step one
template<int boundary>
__global__ void gpu_select_send_particles_kernel(const float4 *d_pos,
                                         float4 *d_pos_tmp,
                                         const float4 *d_vel,
                                         float4 *d_vel_tmp,
                                         const float3 *d_accel,
                                         float3 *d_accel_tmp,
                                         const int3 *d_image,
                                         int3 *d_image_tmp,
                                         const float *d_charge,
                                         float *d_charge_tmp,
                                         const float *d_diameter,
                                         float *d_diameter_tmp,
                                         const unsigned int *d_body,
                                         unsigned int *d_body_tmp,
                                         const float4  *d_orientation,
                                         float4 *d_orientation_tmp,
                                         const unsigned int *d_tag,
                                         unsigned int *d_tag_tmp,
                                         unsigned char *remove_mask,
                                         unsigned int *n_send_ptls,
                                         unsigned int N,
                                         const Scalar3 lo,
                                         const Scalar3 hi,
                                         const Scalar3 L,
                                         const unsigned int dir)
    {
    unsigned int idx = blockIdx.x*blockDim.x + threadIdx.x;

    if (idx >= N)
        return;

    Scalar4 pos = d_pos[idx];
    bool send = ((dir == 0 && pos.x >= hi.x)||  // send east
                (dir == 1 && pos.x < lo.x)  ||  // send west
                (dir == 2 && pos.y >= hi.y) ||  // send north
                (dir == 3 && pos.y < lo.y)  ||  // send south
                (dir == 4 && pos.z >= hi.z) ||  // send up
                (dir == 5 && pos.z < lo.z));    // send down

    // do not send particles twice
    bool remove = (remove_mask[idx] == 1);

    if (send && !remove)
        {
        unsigned int n = atomicInc(n_send_ptls,0xffffffff);

        int3 image = d_image[idx];

        switch (boundary)
            {
            case 0:
                pos.x -= L.x; // wrap across western boundary
                image.x++;
                break;
            case 1:
                pos.x += L.x; // eastern boundary
                image.x--;
                break;
            case 2:
                pos.y -= L.y; // northern boundary
                image.y++;
                break;
            case 3:
                pos.y += L.y; // southern boundary
                image.y--;
                break;
            case 4:
                pos.z -= L.z; // upper boundary
                image.z++;
                break;
            case 5:
                pos.z += L.z; // lower boundary
                image.z--;
                break;
            default:
                break;            // no wrap
            }

        d_pos_tmp[n] = pos;
        d_vel_tmp[n] = d_vel[idx];
        d_accel_tmp[n] = d_accel[idx];
        d_image_tmp[n] = image;
        d_charge_tmp[n] = d_charge[idx];
        d_diameter_tmp[n] = d_diameter[idx];
        d_body_tmp[n] = d_body[idx];
        d_orientation_tmp[n] = d_orientation[idx];
        d_tag_tmp[n] = d_tag[idx];

        // mark particle for removal
        remove_mask[idx] = 1;
        }

    }

/*! Reorder the particles according to a migration criterium
 *  Particles that remain in the simulation box come first, followed by the particles that are sent in the
 *  specified direction
 *
 *  \param N Number of particles in local simulation box
 *  \param n_send_ptls Number of particles that are sent (return value)
 *  \param d_remove_mask Per-particle flag if particle has been sent
 *  \param d_pos Array of particle positions
 *  \param d_pos_tmp Array of particle positions to write to
 *  \param d_vel Array of particle velocities
 *  \param d_vel_tmp Array of particle velocities to write to
 *  \param d_accel Array of particle accelerations
 *  \param d_accel_tmp Array of particle accelerations to write to
 *  \param d_image Array of particle images
 *  \param d_image_tmp Array of particle images
 *  \param d_charge Array of particle charges
 *  \param d_charge_tmp Array of particle charges
 *  \param d_diameter Array of particle diameter
 *  \param d_diameter_tmp Array of particle diameter
 *  \param d_body Array of particle body ids
 *  \param d_body_tmp Array of particle body ids
 *  \param d_orientation Array of particle orientations
 *  \param d_orientation_tmp Array of particle orientations
 *  \param d_tag Array of particle global tags
 *  \param d_tag_tmp Array of particle global tags
 *  \param box Dimensions of local simulation box
 *  \param dir Direction to send particles to
 */
void gpu_migrate_select_particles(unsigned int N,
                        unsigned int &n_send_ptls,
                        unsigned char *d_remove_mask,
                        float4 *d_pos,
                        float4 *d_pos_tmp,
                        float4 *d_vel,
                        float4 *d_vel_tmp,
                        float3 *d_accel,
                        float3 *d_accel_tmp,
                        int3 *d_image,
                        int3 *d_image_tmp,
                        float *d_charge,
                        float *d_charge_tmp,
                        float *d_diameter,
                        float *d_diameter_tmp,
                        unsigned int *d_body,
                        unsigned int *d_body_tmp,
                        float4 *d_orientation,
                        float4 *d_orientation_tmp,
                        unsigned int *d_tag,
                        unsigned int *d_tag_tmp,
                        const BoxDim& box,
                        const BoxDim& global_box,
                        unsigned int dir,
                        const bool is_at_boundary[])
    {
    n_send_ptls = 0;
    cudaMemcpy(d_n_send_particles, &n_send_ptls, sizeof(unsigned int), cudaMemcpyHostToDevice);

    unsigned int block_size = 512;

    if (dir == 0 && is_at_boundary[0])
        gpu_select_send_particles_kernel<0><<<N/block_size+1,block_size>>>(d_pos,
                                                                    d_pos_tmp,
                                                                    d_vel,
                                                                    d_vel_tmp,
                                                                    d_accel, 
                                                                    d_accel_tmp, 
                                                                    d_image, 
                                                                    d_image_tmp, 
                                                                    d_charge, 
                                                                    d_charge_tmp, 
                                                                    d_diameter, 
                                                                    d_diameter_tmp,
                                                                    d_body,
                                                                    d_body_tmp, 
                                                                    d_orientation, 
                                                                    d_orientation_tmp, 
                                                                    d_tag,
                                                                    d_tag_tmp,
                                                                    d_remove_mask,
                                                                    d_n_send_particles,
                                                                    N,
                                                                    box.getLo(), 
                                                                    box.getHi(),
                                                                    global_box.getL(),
                                                                    dir);
     else if (dir == 1 && is_at_boundary[1])
        gpu_select_send_particles_kernel<1><<<N/block_size+1,block_size>>>(d_pos,
                                                                    d_pos_tmp,
                                                                    d_vel,
                                                                    d_vel_tmp,
                                                                    d_accel, 
                                                                    d_accel_tmp, 
                                                                    d_image, 
                                                                    d_image_tmp, 
                                                                    d_charge, 
                                                                    d_charge_tmp, 
                                                                    d_diameter, 
                                                                    d_diameter_tmp,
                                                                    d_body,
                                                                    d_body_tmp, 
                                                                    d_orientation, 
                                                                    d_orientation_tmp, 
                                                                    d_tag,
                                                                    d_tag_tmp,
                                                                    d_remove_mask,
                                                                    d_n_send_particles,
                                                                    N,
                                                                    box.getLo(), 
                                                                    box.getHi(),
                                                                    global_box.getL(),
                                                                    dir);
    else if (dir == 2 && is_at_boundary[2])
        gpu_select_send_particles_kernel<2><<<N/block_size+1,block_size>>>(d_pos,
                                                                    d_pos_tmp,
                                                                    d_vel,
                                                                    d_vel_tmp,
                                                                    d_accel, 
                                                                    d_accel_tmp, 
                                                                    d_image, 
                                                                    d_image_tmp, 
                                                                    d_charge, 
                                                                    d_charge_tmp, 
                                                                    d_diameter, 
                                                                    d_diameter_tmp,
                                                                    d_body,
                                                                    d_body_tmp, 
                                                                    d_orientation, 
                                                                    d_orientation_tmp, 
                                                                    d_tag,
                                                                    d_tag_tmp,
                                                                    d_remove_mask,
                                                                    d_n_send_particles,
                                                                    N,
                                                                    box.getLo(), 
                                                                    box.getHi(),
                                                                    global_box.getL(),
                                                                    dir);
    else if (dir == 3 && is_at_boundary[3])
        gpu_select_send_particles_kernel<3><<<N/block_size+1,block_size>>>(d_pos,
                                                                    d_pos_tmp,
                                                                    d_vel,
                                                                    d_vel_tmp,
                                                                    d_accel, 
                                                                    d_accel_tmp, 
                                                                    d_image, 
                                                                    d_image_tmp, 
                                                                    d_charge, 
                                                                    d_charge_tmp, 
                                                                    d_diameter, 
                                                                    d_diameter_tmp,
                                                                    d_body,
                                                                    d_body_tmp, 
                                                                    d_orientation, 
                                                                    d_orientation_tmp, 
                                                                    d_tag,
                                                                    d_tag_tmp,
                                                                    d_remove_mask,
                                                                    d_n_send_particles,
                                                                    N,
                                                                    box.getLo(), 
                                                                    box.getHi(),
                                                                    global_box.getL(),
                                                                    dir);
    else if (dir == 4 && is_at_boundary[4])
        gpu_select_send_particles_kernel<4><<<N/block_size+1,block_size>>>(d_pos,
                                                                    d_pos_tmp,
                                                                    d_vel,
                                                                    d_vel_tmp,
                                                                    d_accel, 
                                                                    d_accel_tmp, 
                                                                    d_image, 
                                                                    d_image_tmp, 
                                                                    d_charge, 
                                                                    d_charge_tmp, 
                                                                    d_diameter, 
                                                                    d_diameter_tmp,
                                                                    d_body,
                                                                    d_body_tmp, 
                                                                    d_orientation, 
                                                                    d_orientation_tmp, 
                                                                    d_tag,
                                                                    d_tag_tmp,
                                                                    d_remove_mask,
                                                                    d_n_send_particles,
                                                                    N,
                                                                    box.getLo(), 
                                                                    box.getHi(),
                                                                    global_box.getL(),
                                                                    dir);
    else if (dir == 5 && is_at_boundary[5])
        gpu_select_send_particles_kernel<5><<<N/block_size+1,block_size>>>(d_pos,
                                                                    d_pos_tmp,
                                                                    d_vel,
                                                                    d_vel_tmp,
                                                                    d_accel, 
                                                                    d_accel_tmp, 
                                                                    d_image, 
                                                                    d_image_tmp, 
                                                                    d_charge, 
                                                                    d_charge_tmp, 
                                                                    d_diameter, 
                                                                    d_diameter_tmp,
                                                                    d_body,
                                                                    d_body_tmp, 
                                                                    d_orientation, 
                                                                    d_orientation_tmp, 
                                                                    d_tag,
                                                                    d_tag_tmp,
                                                                    d_remove_mask,
                                                                    d_n_send_particles,
                                                                    N,
                                                                    box.getLo(), 
                                                                    box.getHi(),
                                                                    global_box.getL(),
                                                                    dir);
    else // no wrap
        gpu_select_send_particles_kernel<-1><<<N/block_size+1,block_size>>>(d_pos,
                                                                    d_pos_tmp,
                                                                    d_vel,
                                                                    d_vel_tmp,
                                                                    d_accel, 
                                                                    d_accel_tmp, 
                                                                    d_image, 
                                                                    d_image_tmp, 
                                                                    d_charge, 
                                                                    d_charge_tmp, 
                                                                    d_diameter, 
                                                                    d_diameter_tmp,
                                                                    d_body,
                                                                    d_body_tmp, 
                                                                    d_orientation, 
                                                                    d_orientation_tmp, 
                                                                    d_tag,
                                                                    d_tag_tmp,
                                                                    d_remove_mask,
                                                                    d_n_send_particles,
                                                                    N,
                                                                    box.getLo(), 
                                                                    box.getHi(),
                                                                    global_box.getL(),
                                                                    dir); 

    cudaMemcpy(&n_send_ptls, d_n_send_particles, sizeof(unsigned int), cudaMemcpyDeviceToHost);
    }

void gpu_migrate_compact_particles(unsigned int N,
                        unsigned char *d_remove_mask,
                        unsigned int &n_remove_ptls,
                        float4 *d_pos,
                        float4 *d_pos_tmp,
                        float4 *d_vel,
                        float4 *d_vel_tmp,
                        float3 *d_accel,
                        float3 *d_accel_tmp,
                        int3 *d_image,
                        int3 *d_image_tmp,
                        float *d_charge,
                        float *d_charge_tmp,
                        float *d_diameter,
                        float *d_diameter_tmp,
                        unsigned int *d_body,
                        unsigned int *d_body_tmp,
                        float4 *d_orientation,
                        float4 *d_orientation_tmp,
                        unsigned int *d_tag,
                        unsigned int *d_tag_tmp)
    {
    keys->resize(N);

    thrust::counting_iterator<unsigned int> count(0);
    thrust::copy(count, count + N, keys->begin());

    thrust::device_ptr<unsigned char> remove_mask_ptr(d_remove_mask);
    thrust::device_vector<unsigned int>::iterator keys_middle;

    keys_middle = thrust::remove_if(keys->begin(),
                             keys->end(),
                             remove_mask_ptr, 
                             thrust::identity<unsigned char>());

    n_remove_ptls = keys->end()- keys_middle;
 
    thrust::device_ptr<float4> pos_ptr(d_pos);
    thrust::device_ptr<float4> pos_tmp_ptr(d_pos_tmp);
    thrust::device_ptr<float4> vel_ptr(d_vel);
    thrust::device_ptr<float4> vel_tmp_ptr(d_vel_tmp);
    thrust::device_ptr<float3> accel_ptr(d_accel);
    thrust::device_ptr<float3> accel_tmp_ptr(d_accel_tmp);
    thrust::device_ptr<int3> image_ptr(d_image);
    thrust::device_ptr<int3> image_tmp_ptr(d_image_tmp);
    thrust::device_ptr<float> charge_ptr(d_charge);
    thrust::device_ptr<float> charge_tmp_ptr(d_charge_tmp);
    thrust::device_ptr<float> diameter_ptr(d_diameter);
    thrust::device_ptr<float> diameter_tmp_ptr(d_diameter_tmp);
    thrust::device_ptr<unsigned int> body_ptr(d_body);
    thrust::device_ptr<unsigned int> body_tmp_ptr(d_body_tmp);
    thrust::device_ptr<float4> orientation_ptr(d_orientation);
    thrust::device_ptr<float4> orientation_tmp_ptr(d_orientation_tmp);
    thrust::device_ptr<unsigned int> tag_ptr(d_tag);
    thrust::device_ptr<unsigned int> tag_tmp_ptr(d_tag_tmp);

    // reorder particle data, write into temporary arrays
    thrust::gather(keys->begin(), keys_middle, pos_ptr, pos_tmp_ptr);
    thrust::gather(keys->begin(), keys_middle, vel_ptr, vel_tmp_ptr);
    thrust::gather(keys->begin(), keys_middle, accel_ptr, accel_tmp_ptr);
    thrust::gather(keys->begin(), keys_middle, image_ptr, image_tmp_ptr);
    thrust::gather(keys->begin(), keys_middle, charge_ptr, charge_tmp_ptr);
    thrust::gather(keys->begin(), keys_middle, diameter_ptr, diameter_tmp_ptr);
    thrust::gather(keys->begin(), keys_middle, body_ptr, body_tmp_ptr);
    thrust::gather(keys->begin(), keys_middle, orientation_ptr, orientation_tmp_ptr);
    thrust::gather(keys->begin(), keys_middle, tag_ptr, tag_tmp_ptr);
    }

 
//! Reset reverse lookup tags of particles we are removing
/* \param n_delete_ptls Number of particles to delete
 * \param d_delete_tags Array of particle tags to delete
 * \param d_rtag Array for tag->idx lookup
 */
void gpu_reset_rtags(unsigned int n_delete_ptls,
                     unsigned int *d_delete_tags,
                     unsigned int *d_rtag)
    {
    thrust::device_ptr<unsigned int> delete_tags_ptr(d_delete_tags);
    thrust::device_ptr<unsigned int> rtag_ptr(d_rtag);

    thrust::constant_iterator<unsigned int> not_local(NOT_LOCAL);
    thrust::scatter(not_local,
                    not_local + n_delete_ptls,
                    delete_tags_ptr,
                    rtag_ptr);
    }

__global__ void gpu_reset_rtags_by_mask_kernel(const unsigned int N,
                                        const unsigned char *remove_mask,
                                        unsigned int *tag,
                                        unsigned int *rtag)
    {
    unsigned int idx = blockIdx.x*blockDim.x + threadIdx.x;
    
    if (idx >= N) return;

    if (remove_mask[idx])
        rtag[tag[idx]] = NOT_LOCAL;
    }

//! Reset reverse lookup tags of particles by the remove mask
/* \param n_delete_ptls Number of particles to check
 * \param d_remove_mask Mask indicating which particles are to be removed
 * \param d_tag Array of particle tags
 * \param d_rtag Array for tag->idx lookup
 */
void gpu_reset_rtags_by_mask(unsigned int N,
                     unsigned char *d_remove_mask,
                     unsigned int *d_tag,
                     unsigned int *d_rtag)
    {
    unsigned int block_size = 512;

    gpu_reset_rtags_by_mask_kernel<<<N/block_size+1,block_size>>>(N, d_remove_mask,d_tag,d_rtag);
    }


//! Construct plans for sending non-bonded ghost particles
/*! \param d_plan Array of ghost particle plans
 * \param N number of particles to check
 * \param d_pos Array of particle positions
 * \param box Dimensions of local simulation box
 * \param r_ghost Width of boundary layer
 */
void gpu_make_nonbonded_exchange_plan(unsigned char *d_plan,
                                      unsigned int N,
                                      float4 *d_pos,
                                      const BoxDim &box,
                                      float r_ghost)
    {
    thrust::device_ptr<float4> pos_ptr(d_pos);
    thrust::device_ptr<unsigned char> plan_ptr(d_plan);

    thrust::transform(
        thrust::make_zip_iterator(thrust::make_tuple(
            pos_ptr,
            plan_ptr)),
        thrust::make_zip_iterator(thrust::make_tuple(
            pos_ptr,
            plan_ptr)) + N,
        plan_ptr,
        make_nonbonded_plan(box, r_ghost));
    }

//! Kernel to construct list of ghosts to send
template<unsigned int boundary>
__global__ void gpu_exchange_ghosts_kernel(const unsigned int n_total,
                                         const unsigned int dir,
                                         const unsigned char *plan,
                                         const unsigned int *tag,
                                         unsigned int *copy_ghosts,
                                         unsigned int *copy_ghosts_r,
                                         float4 *pos,
                                         float4 *pos_copybuf,
                                         float4 *pos_copybuf_r,
                                         float *charge,
                                         float *charge_copybuf,
                                         float *charge_copybuf_r,
                                         float *diameter,
                                         float *diameter_copybuf,
                                         float *diameter_copybuf_r,
                                         unsigned char *plan_copybuf,
                                         unsigned char *plan_copybuf_r,
                                         unsigned int *tag_copybuf,
                                         unsigned int *tag_copybuf_r,
                                         unsigned int *n_copy_ghosts,
                                         unsigned int *n_copy_ghosts_r,
                                         const bool is_at_boundary,
                                         const bool is_at_boundary_reverse,
                                         Scalar3 L)
    {
    unsigned int idx = blockIdx.x*blockDim.x + threadIdx.x;
    
    if (idx >= n_total) return;

    unsigned char p = plan[idx];
    bool do_send = (p & (1 << dir));
    bool do_send_r = (p & (1 << (dir+1)));

    if (do_send)
        {
        unsigned int n = atomicInc(n_copy_ghosts, 0xffffffff);
        Scalar4 postype = pos[idx]; 

        switch(boundary)
            {
            case 0:
                if (is_at_boundary) postype.x -= L.x;
                break;
            case 1:
                if (is_at_boundary) postype.y -= L.y;
                break;
            case 2:
                if (is_at_boundary) postype.z -= L.z;
                break;
            }

        pos_copybuf[n] = postype;
        tag_copybuf[n] = tag[idx];
        plan_copybuf[n] = plan[idx];
        charge_copybuf[n] = charge[idx];
        diameter_copybuf[n] = diameter[idx];

        copy_ghosts[n] = idx;
        }

    if (do_send_r)
        {
        unsigned int n = atomicInc(n_copy_ghosts_r, 0xffffffff);
        Scalar4 postype = pos[idx]; 

        switch(boundary)
            {
            case 0:
                if (is_at_boundary_reverse) postype.x += L.x;
                break;
            case 1:
                if (is_at_boundary_reverse) postype.y += L.y;
                break;
            case 2:
                if (is_at_boundary_reverse) postype.z += L.z;
                break;
            }

        pos_copybuf_r[n] = postype;
        tag_copybuf_r[n] = tag[idx];
        plan_copybuf_r[n] = plan[idx];
        charge_copybuf_r[n] = charge[idx];
        diameter_copybuf_r[n] = diameter[idx];
        copy_ghosts_r[n] = idx;
        }
       
    }

    
//! Construct a list of particle tags to send as ghost particles
/*! \param n_total Total number of particles to check
 * \param N number of local particles
 * \param dir Direction in which ghost particles are sent
 * \param d_plan Array of particle exchange plans
 * \param d_global_tag Array of particle global tags
 * \param d_copy_ghosts Array to be fillled with indices of particles that are to be sent as ghosts
 * \param d_ghost_tag Array of ghost particle tags to be sent
 * \param n_copy_ghosts Number of local particles that are sent in the given direction as ghosts (return value)
 */
void gpu_exchange_ghosts(unsigned int n_total,
                         unsigned char *d_plan,
                         unsigned int *d_copy_ghosts,
                         unsigned int *d_copy_ghosts_r,
                         float4 *d_pos,
                         float4 *d_pos_copybuf,
                         float4 *d_pos_copybuf_r,
                         float *d_charge,
                         float *d_charge_copybuf,
                         float *d_charge_copybuf_r,
                         float *d_diameter,
                         float *d_diameter_copybuf,
                         float *d_diameter_copybuf_r,
                         unsigned char *d_plan_copybuf,
                         unsigned char *d_plan_copybuf_r,
                         unsigned int *d_tag,
                         unsigned int *d_tag_copybuf,
                         unsigned int *d_tag_copybuf_r,
                         unsigned int &n_copy_ghosts,
                         unsigned int &n_copy_ghosts_r,
                         unsigned int dir,
                         const bool is_at_boundary[],
                         const BoxDim& global_box)
    {
    n_copy_ghosts = 0;
    cudaMemcpy(d_n_copy_ghosts, &n_copy_ghosts, sizeof(unsigned int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_n_copy_ghosts_r, &n_copy_ghosts, sizeof(unsigned int), cudaMemcpyHostToDevice);

    unsigned int block_size = 512;
    if (dir == 0)
        gpu_exchange_ghosts_kernel<0><<<n_total/block_size+1, block_size>>>(n_total,
                                         dir,
                                         d_plan,
                                         d_tag,
                                         d_copy_ghosts,
                                         d_copy_ghosts_r,
                                         d_pos,
                                         d_pos_copybuf,
                                         d_pos_copybuf_r,
                                         d_charge,
                                         d_charge_copybuf,
                                         d_charge_copybuf_r,
                                         d_diameter,
                                         d_diameter_copybuf,
                                         d_diameter_copybuf_r,
                                         d_plan_copybuf,
                                         d_plan_copybuf_r,
                                         d_tag_copybuf,
                                         d_tag_copybuf_r,
                                         d_n_copy_ghosts,
                                         d_n_copy_ghosts_r,
                                         is_at_boundary[dir],
                                         is_at_boundary[dir+1],
                                         global_box.getL());
    else if (dir == 2)
        gpu_exchange_ghosts_kernel<1><<<n_total/block_size+1, block_size>>>(n_total,
                                         dir,
                                         d_plan,
                                         d_tag,
                                         d_copy_ghosts,
                                         d_copy_ghosts_r,
                                         d_pos,
                                         d_pos_copybuf,
                                         d_pos_copybuf_r,
                                         d_charge,
                                         d_charge_copybuf,
                                         d_charge_copybuf_r,
                                         d_diameter,
                                         d_diameter_copybuf,
                                         d_diameter_copybuf_r,
                                         d_plan_copybuf,
                                         d_plan_copybuf_r,
                                         d_tag_copybuf,
                                         d_tag_copybuf_r,
                                         d_n_copy_ghosts,
                                         d_n_copy_ghosts_r,
                                         is_at_boundary[dir],
                                         is_at_boundary[dir+1],
                                         global_box.getL());
     else if (dir == 4)
        gpu_exchange_ghosts_kernel<2><<<n_total/block_size+1, block_size>>>(n_total,
                                         dir,
                                         d_plan,
                                         d_tag,
                                         d_copy_ghosts,
                                         d_copy_ghosts_r,
                                         d_pos,
                                         d_pos_copybuf,
                                         d_pos_copybuf_r,
                                         d_charge,
                                         d_charge_copybuf,
                                         d_charge_copybuf_r,
                                         d_diameter,
                                         d_diameter_copybuf,
                                         d_diameter_copybuf_r,
                                         d_plan_copybuf,
                                         d_plan_copybuf_r,
                                         d_tag_copybuf,
                                         d_tag_copybuf_r,
                                         d_n_copy_ghosts,
                                         d_n_copy_ghosts_r,
                                         is_at_boundary[dir],
                                         is_at_boundary[dir+1],
                                         global_box.getL());

    cudaMemcpy(&n_copy_ghosts, d_n_copy_ghosts, sizeof(unsigned int), cudaMemcpyDeviceToHost);
    cudaMemcpy(&n_copy_ghosts_r, d_n_copy_ghosts_r, sizeof(unsigned int), cudaMemcpyDeviceToHost);
    }

//! Update global tag <-> local particle index reverse lookup array
/*! \param nptl Number of particles for which we are updating the reverse lookup tags
 * \param start_idx starting index of first particle in local particle data arrays
 * \param d_tag array of particle tags
 * \param d_rtag array of particle reverse lookup tags to store information to
 */
void gpu_update_rtag(unsigned int nptl, unsigned int start_idx, unsigned int *d_tag, unsigned int *d_rtag)
    {
    thrust::device_ptr<unsigned int> tag_ptr(d_tag);
    thrust::device_ptr<unsigned int> rtag_ptr(d_rtag);

    thrust::counting_iterator<unsigned int> first(start_idx);
    thrust::counting_iterator<unsigned int> last = first + nptl;
    thrust::scatter(first, last, tag_ptr, rtag_ptr);
    }

//! Fill ghost copy buffer & apply periodic boundary conditions to a ghost particle before sending
template<int boundary>
__global__ void gpu_copy_ghost_particles_kernel(float4 *pos,
                                      unsigned int *copy_ghosts,
                                      unsigned int *copy_ghosts_reverse,
                                      float4 *pos_copybuf,
                                      float4 *pos_copybuf_reverse,
                                      unsigned int nghost,
                                      unsigned int nghost_reverse,
                                      Scalar3 L,
                                      bool is_at_boundary,
                                      bool is_at_boundary_reverse)
    {
    unsigned int ghost_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (ghost_idx < nghost)
        {
        Scalar4 postype = pos[copy_ghosts[ghost_idx]];

        switch(boundary)
            {
            case 0:
                if (is_at_boundary) postype.x -= L.x;
                break;
            case 1: 
                if (is_at_boundary) postype.y -= L.y;
                break;
            case 2:
                if (is_at_boundary) postype.z -= L.z;
                break;
            }
        pos_copybuf[ghost_idx] = postype;
        }

    if (ghost_idx < nghost_reverse)
        {
        Scalar4 postype = pos[copy_ghosts_reverse[ghost_idx]];

        switch(boundary)
            {
            case 0:
                if (is_at_boundary_reverse) postype.x += L.x;
                break;
            case 1:
                if (is_at_boundary_reverse) postype.y += L.y;
                break;
            case 2:
                if (is_at_boundary_reverse) postype.z += L.z;
                break;
            }
        pos_copybuf_reverse[ghost_idx] = postype;
        } 
    } 


//! Copy ghost particle positions into send buffer
/*! This method copies two send buffers at a time, in direction dir
 *  and in the opposite direction dir+1. The underlying assumption is that
 *  particles are never sent back in the opposite direction from which they
 *  were received from.

 * \param nghost Number of ghost particles to copy in direction dir
 * \param nghost_r Number of ghost particles to copy in direction dir+1
 * \param d_pos Array of particle positions
 * \param d_copy_ghosts Global particle tags of particles to copy
 * \param d_copy_ghosts Global particle tags of particles to copy in reverse direction
 * \param d_pos_copybuf Send buffer of ghost particle positions
 * \param d_pos_copybuf Send buffer of ghost particle positions in reverse direction
 * \param dir Current send direction (this method only called for even directions 0,2,4)
 * \param is_at_boundary Per-direction flags whether we share a boundary with the global box
 * \paramm global_box Global boundaries
 */
void gpu_copy_ghosts(unsigned int nghost,
                     unsigned int nghost_r,
                     float4 *d_pos,
                     unsigned int *d_copy_ghosts,
                     unsigned int *d_copy_ghosts_r,
                     float4 *d_pos_copybuf,
                     float4 *d_pos_copybuf_r,
                     unsigned int dir,
                     const bool is_at_boundary[],
                     const BoxDim& global_box)
    {

    unsigned int block_size = 192;
    unsigned int n = (nghost > nghost_r) ? nghost : nghost_r;

    if (dir == 0)
        gpu_copy_ghost_particles_kernel<0><<<n/block_size+1, block_size>>>(d_pos, d_copy_ghosts, d_copy_ghosts_r, d_pos_copybuf, d_pos_copybuf_r, nghost, nghost_r, global_box.getL(), is_at_boundary[dir], is_at_boundary[dir+1]);
    else if (dir == 2)
        gpu_copy_ghost_particles_kernel<1><<<n/block_size+1, block_size>>>(d_pos, d_copy_ghosts, d_copy_ghosts_r, d_pos_copybuf, d_pos_copybuf_r, nghost, nghost_r, global_box.getL(), is_at_boundary[dir], is_at_boundary[dir+1]);
    else if (dir == 4)
        gpu_copy_ghost_particles_kernel<2><<<n/block_size+1, block_size>>>(d_pos, d_copy_ghosts, d_copy_ghosts_r, d_pos_copybuf, d_pos_copybuf_r, nghost, nghost_r, global_box.getL(), is_at_boundary[dir], is_at_boundary[dir+1]);
 
    } 

//! Reset reverse lookup tags of removed ghost particles to NOT_LOCAL
/*! \param nghost Number of ghost particles for which the tags are to be reset
 * \param d_gloal_rtag Pointer to reverse-lookup tags to reset
 */
void gpu_reset_ghost_rtag(unsigned int nghost,
                          unsigned int *d_global_rtag)
     {
     thrust::device_ptr<unsigned int> global_rtag_ptr(d_global_rtag);
     thrust::fill(global_rtag_ptr, global_rtag_ptr + nghost, NOT_LOCAL);
     }
#endif
