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

//! Wrap a received particle across global box boundaries
struct wrap_received_particle
    {
    const Scalar3 L;          //!< Lengths of global simulation box
    const unsigned int dir;   //!< Current direction of particle migration
    bool is_at_boundary[6];   //!< Flags to indicate whether this box share a boundary with the global box

    //! Constructor
    /*! \param _global_box Dimensions of global simulation box
        \param _dir Direction along which the particle was received
        \param _is_at_boundary Flags to indicate whether the local box shares a boundary with the global box
     */
    wrap_received_particle(const BoxDim _global_box, const unsigned int _dir,  const bool _is_at_boundary[])
        : L(_global_box.getL()), dir(_dir)
        {
        for (unsigned int i = 0; i < 6; i++)
            is_at_boundary[i] = _is_at_boundary[i];
        }

   //! Wrap particle across boundaries
   /*! \param el particle data element to transform
    * \return transformed particle data element
    */
    __host__ __device__ thrust::tuple<float4, int3> operator()(const thrust::tuple<float4, int3> t)
        {
        float4 postype = thrust::get<0>(t);
        int3 image = thrust::get<1>(t);

        // wrap particles received across a global boundary back into global box
        if (dir==0 && is_at_boundary[1])
            {
            postype.x -= L.x;
            image.x++;
            }
        else if (dir==1 && is_at_boundary[0])
            {
            postype.x += L.x;
            image.x--;
            }
        else if (dir==2 && is_at_boundary[3])
            {
            postype.y -= L.y;
            image.y++;
            }
        else if (dir==3 && is_at_boundary[2])
            {
            postype.y += L.y;
            image.y--;
            }
        else if (dir==4 && is_at_boundary[5])
            {
            postype.z -= L.z;
            image.z++;
            }
        else if (dir==5 && is_at_boundary[4])
            {
            postype.z += L.z;
            image.z--;
            }
        return make_tuple(postype,image);
        }

     };


thrust::device_vector<unsigned int> *keys;       //!< Temporary vector of sort keys

unsigned int *d_n_send_particles;  //! Counter for construction of atom send lists
unsigned int *d_n_copy_ghosts;     //! Counter for ghost list construction

void gpu_allocate_tmp_storage()
    {
    keys = new thrust::device_vector<unsigned int>;

    cudaMalloc(&d_n_send_particles,sizeof(unsigned int));
    cudaMalloc(&d_n_copy_ghosts, sizeof(unsigned int));
    }

void gpu_deallocate_tmp_storage()
    {
    delete keys;

    cudaFree(d_n_send_particles);
    cudaFree(d_n_copy_ghosts);
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

        d_pos_tmp[n] = d_pos[idx];
        d_vel_tmp[n] = d_vel[idx];
        d_accel_tmp[n] = d_accel[idx];
        d_image_tmp[n] = d_image[idx];
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
                        unsigned int dir)
    {
    n_send_ptls = 0;
    cudaMemcpy(d_n_send_particles, &n_send_ptls, sizeof(unsigned int), cudaMemcpyHostToDevice);

    unsigned int block_size = 512;

    gpu_select_send_particles_kernel<<<N/block_size+1,block_size>>>(d_pos,
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


//! Wrap received particles across global box boundaries
/*! \param d_pos Particle positions array
 * \param d_image Particle images array
 * \param n_recv_ptl Number of received particles (return value)
 * \param global_box Dimensions of global box
 * \param dir Direction along which particles where received
 * \param is_at_boundary Array of per-direction flags to indicate whether this box lies at a global boundary
 */
void gpu_migrate_wrap_received_particles(float4 *d_pos,
                                 int3 *d_image,
                                 unsigned int n_recv_ptl,
                                 const BoxDim& global_box,
                                 const unsigned int dir,
                                 const bool is_at_boundary[])
    {
    thrust::device_ptr<float4> pos_ptr(d_pos);
    thrust::device_ptr<int3> image_ptr(d_image);

    thrust::transform(thrust::make_zip_iterator(thrust::make_tuple(
                        pos_ptr,
                        image_ptr)),
                      thrust::make_zip_iterator(thrust::make_tuple(
                        pos_ptr,
                        image_ptr)) + n_recv_ptl,
                      thrust::make_zip_iterator(thrust::make_tuple(
                        pos_ptr,
                        image_ptr)),
                      wrap_received_particle(global_box, dir, is_at_boundary));
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
__global__ void gpu_make_exchange_ghost_list_kernel(const unsigned int n_total,
                                         const unsigned int N,
                                         const unsigned int dir,
                                         const unsigned char *plan,
                                         const unsigned int *tag,
                                         unsigned int *copy_ghosts,
                                         unsigned int *ghost_tag,
                                         unsigned int *n_copy_ghosts)
    {
    unsigned int idx = blockIdx.x*blockDim.x + threadIdx.x;
    
    if (idx >= n_total) return;

    unsigned char p = plan[idx];
    bool do_send = (p & (1 << dir));

    if (do_send)
        {
        unsigned int n = atomicInc(n_copy_ghosts, 0xffffffff);
        ghost_tag[n] = tag[idx];
        copy_ghosts[n] = idx;
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
void gpu_make_exchange_ghost_list(unsigned int n_total,
                                  unsigned int N,
                                  unsigned int dir,
                                  unsigned char *d_plan,
                                  unsigned int *d_global_tag,
                                  unsigned int *d_copy_ghosts,
                                  unsigned int *d_ghost_tag,
                                  unsigned int &n_copy_ghosts)
    {
    n_copy_ghosts = 0;
    cudaMemcpy(d_n_copy_ghosts, &n_copy_ghosts, sizeof(unsigned int), cudaMemcpyHostToDevice);

    unsigned int block_size = 512;
    gpu_make_exchange_ghost_list_kernel<<<n_total/block_size + 1, block_size>>>(n_total,
                                                                                N,
                                                                                dir,
                                                                                d_plan,
                                                                                d_global_tag,
                                                                                d_copy_ghosts,
                                                                                d_ghost_tag,
                                                                                d_n_copy_ghosts);
    cudaMemcpy(&n_copy_ghosts, d_n_copy_ghosts, sizeof(unsigned int), cudaMemcpyDeviceToHost);
    }

//! Fill ghost copy buffer & apply periodic boundary conditions to a ghost particle before sending
template<int boundary>
__global__ void gpu_exchange_ghosts_kernel(float4 *pos,
                                      unsigned int *copy_ghosts,
                                      float4 *pos_copybuf,
                                      float *charge,
                                      float *charge_copybuf,
                                      float *diameter,
                                      float *diameter_copybuf,
                                      unsigned char *plan,
                                      unsigned char *plan_copybuf,
                                      unsigned int nghost,
                                      Scalar3 L)
    {
    unsigned int ghost_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (ghost_idx >= nghost) return;
    unsigned int pidx = copy_ghosts[ghost_idx];
    Scalar4 postype = pos[pidx];

    // wrap particles global boundary back into global box before sending
    switch(boundary)
        {
        case 0: // west boundary 
            postype.x -= L.x;
            break;
        case 1: // east boundary
            postype.x += L.x;
            break;
        case 2: // north boundary
            postype.y -= L.y;
            break;
        case 3: // south boundary
            postype.y += L.y;
            break;
        case 4: // upper boundary
            postype.z -= L.z;
            break;
        case 5: // lower boundary
            postype.z += L.z;
            break;
        case -1: // do not wrap
            break;
        }
    pos_copybuf[ghost_idx] = postype;
    charge_copybuf[ghost_idx] = charge[pidx];
    diameter_copybuf[ghost_idx] = diameter[pidx];
    plan_copybuf[ghost_idx] = plan[pidx];
    } 


//! Fill send buffers of particles we are sending as ghost particles with partial particle data
/*! \param nghost Number of ghost particles to copy into send buffers
 * \param d_copy_ghosts Array of particle tags to copy as ghost particles
 * \param d_rtag Inverse look-up array for global tags <-> local indices
 * \param d_pos Array of particle positions
 * \param d_pos_copybuf Send buffer for particle positions
 * \param d_charge Array of particle charges
 * \param d_charge_copybuf Send buffer for particle charges
 * \param d_diameter Array of particle diameters
 * \param d_diameter_copybuf Send buffer for particle diameters
 * \param d_plan Array of particle plans
 * \param d_plan_copybuf Send buffer for particle plans
 * \param dir Current send direction
 * \param is_at_boundary Per-direction flag if we share a boundary with the global box
 * \param global_box The global box
 */
void gpu_exchange_ghosts(unsigned int nghost,
                         unsigned int *d_copy_ghosts,
                         float4 *d_pos,
                         float4 *d_pos_copybuf,
                         float *d_charge,
                         float *d_charge_copybuf,
                         float *d_diameter,
                         float *d_diameter_copybuf,
                         unsigned char *d_plan,
                         unsigned char *d_plan_copybuf,
                         const unsigned int dir,
                         const bool is_at_boundary[],
                         const BoxDim& global_box)
    {
    unsigned int block_size = 512;
    if (dir == 0 && is_at_boundary[0])
        gpu_exchange_ghosts_kernel<0><<<nghost/block_size+1, block_size>>>(d_pos, d_copy_ghosts, d_pos_copybuf, d_charge, d_charge_copybuf, d_diameter, d_diameter_copybuf, d_plan, d_plan_copybuf, nghost, global_box.getL());
    else if (dir == 1 && is_at_boundary[1])
        gpu_exchange_ghosts_kernel<1><<<nghost/block_size+1, block_size>>>(d_pos, d_copy_ghosts, d_pos_copybuf, d_charge, d_charge_copybuf, d_diameter, d_diameter_copybuf, d_plan, d_plan_copybuf, nghost, global_box.getL());
    else if (dir == 2 && is_at_boundary[2])
        gpu_exchange_ghosts_kernel<2><<<nghost/block_size+1, block_size>>>(d_pos, d_copy_ghosts, d_pos_copybuf, d_charge, d_charge_copybuf, d_diameter, d_diameter_copybuf, d_plan, d_plan_copybuf, nghost, global_box.getL());
    else if (dir == 3 && is_at_boundary[3])
        gpu_exchange_ghosts_kernel<3><<<nghost/block_size+1, block_size>>>(d_pos, d_copy_ghosts, d_pos_copybuf, d_charge, d_charge_copybuf, d_diameter, d_diameter_copybuf, d_plan, d_plan_copybuf, nghost, global_box.getL());
    else if (dir == 4 && is_at_boundary[4])
        gpu_exchange_ghosts_kernel<4><<<nghost/block_size+1, block_size>>>(d_pos, d_copy_ghosts, d_pos_copybuf, d_charge, d_charge_copybuf, d_diameter, d_diameter_copybuf, d_plan, d_plan_copybuf, nghost, global_box.getL());
    else if (dir == 5 && is_at_boundary[5])
        gpu_exchange_ghosts_kernel<5><<<nghost/block_size+1, block_size>>>(d_pos, d_copy_ghosts, d_pos_copybuf, d_charge, d_charge_copybuf, d_diameter, d_diameter_copybuf, d_plan, d_plan_copybuf, nghost, global_box.getL());
    else
        gpu_exchange_ghosts_kernel<-1><<<nghost/block_size+1, block_size>>>(d_pos, d_copy_ghosts, d_pos_copybuf, d_charge, d_charge_copybuf, d_diameter, d_diameter_copybuf, d_plan, d_plan_copybuf, nghost, global_box.getL());
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
                                      float4 *pos_copybuf,
                                      unsigned int nghost,
                                      Scalar3 L)
    {
    unsigned int ghost_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (ghost_idx >= nghost) return;
    Scalar4 postype = pos[copy_ghosts[ghost_idx]];

    // wrap particles global boundary back into global box before sending
    switch(boundary)
        {
        case 0: // west boundary
            postype.x -= L.x;
            break;
        case 1: // east boundary
            postype.x += L.x;
            break;
        case 2: // north boundary
            postype.y -= L.y;
            break;
        case 3: // south boundary
            postype.y += L.y;
            break;
        case 4: // upper boundary
            postype.z -= L.z;
            break;
        case 5: // lower boundary
            postype.z += L.z;
            break;
        case -1: // do not wrap
            break;
        }
    pos_copybuf[ghost_idx] = postype;
    } 


//! Copy ghost particle positions into send buffer
/*! \param nghost Number of ghost particles to copy
 * \param d_pos Array of particle positions
 * \param d_copy_ghosts Global particle tags of particles to copy
 * \param d_pos_copybuf Send buffer of ghost particle positions
 * \param dir Current send direction
 * \param is_at_boundary Per-direction flags whether we share a boundary with the global box
 * \paramm global_box Global boundaries
 */
void gpu_copy_ghosts(unsigned int nghost,
                     float4 *d_pos,
                     unsigned int *d_copy_ghosts,
                     float4 *d_pos_copybuf,
                     unsigned int dir,
                     const bool is_at_boundary[],
                     const BoxDim& global_box)
    {

    unsigned int block_size = 512;
    if (dir == 0 && is_at_boundary[0])
        gpu_copy_ghost_particles_kernel<0><<<nghost/block_size+1, block_size>>>(d_pos, d_copy_ghosts, d_pos_copybuf, nghost, global_box.getL());
    else if (dir == 1 && is_at_boundary[1])
        gpu_copy_ghost_particles_kernel<1><<<nghost/block_size+1, block_size>>>(d_pos, d_copy_ghosts, d_pos_copybuf, nghost, global_box.getL());
    else if (dir == 2 && is_at_boundary[2])
        gpu_copy_ghost_particles_kernel<2><<<nghost/block_size+1, block_size>>>(d_pos, d_copy_ghosts, d_pos_copybuf, nghost, global_box.getL());
    else if (dir == 3 && is_at_boundary[3])
        gpu_copy_ghost_particles_kernel<3><<<nghost/block_size+1, block_size>>>(d_pos, d_copy_ghosts, d_pos_copybuf, nghost, global_box.getL());
    else if (dir == 4 && is_at_boundary[4])
        gpu_copy_ghost_particles_kernel<4><<<nghost/block_size+1, block_size>>>(d_pos, d_copy_ghosts, d_pos_copybuf, nghost, global_box.getL());
    else if (dir == 5 && is_at_boundary[5])
        gpu_copy_ghost_particles_kernel<5><<<nghost/block_size+1, block_size>>>(d_pos, d_copy_ghosts, d_pos_copybuf, nghost, global_box.getL());
    else
        gpu_copy_ghost_particles_kernel<-1><<<nghost/block_size+1, block_size>>>(d_pos, d_copy_ghosts, d_pos_copybuf, nghost, global_box.getL());
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
