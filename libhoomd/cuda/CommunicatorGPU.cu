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

#include <stdio.h>

#ifdef WIN32
#include <cassert>
#else
#include <assert.h>
#endif

using namespace thrust;

unsigned int gpu_pdata_element_size()
    {
    return sizeof(pdata_element_gpu);
    }

unsigned int gpu_ghost_element_size()
    {
    return sizeof(ghost_element_gpu);
    }

unsigned int gpu_update_element_size()
    {
    return sizeof(update_element_gpu);
    }

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

unsigned int *d_n_send_particles_corner;  //!< Number of particles sent over a corner
unsigned int *d_n_send_particles_edge;    //!< Number of particles sent over an edge
unsigned int *d_n_send_particles_face;    //!< Number of particles sent over a face
unsigned int *d_n_remove_ptls;            //!< Number of particles that will be removed
unsigned int *d_n_fetch_ptl;              //!< Index of fetched particle from received ptl list

__constant__ unsigned int d_corner_plan_lookup[8];
__constant__ unsigned int d_edge_plan_lookup[12];
__constant__ unsigned int d_face_plan_lookup[6];

__constant__ unsigned int d_is_communicating[6]; //!< Per-direction flag indicating whether we are communicating in that direction
__constant__ unsigned int d_is_at_boundary[6]; //!< Per-direction flag indicating whether the box has a global boundary

extern unsigned int *corner_plan_lookup[];
extern unsigned int *edge_plan_lookup[];
extern unsigned int *face_plan_lookup[];

void gpu_allocate_tmp_storage(const unsigned int *is_communicating,
                              const unsigned int *is_at_boundary)
    {
    cudaMalloc(&d_n_send_particles_corner,8*sizeof(unsigned int));
    cudaMalloc(&d_n_send_particles_edge,12*sizeof(unsigned int));
    cudaMalloc(&d_n_send_particles_face,6*sizeof(unsigned int));
    cudaMalloc(&d_n_remove_ptls,sizeof(unsigned int));
    cudaMalloc(&d_n_fetch_ptl,sizeof(unsigned int));

    cudaMemcpyToSymbol(d_corner_plan_lookup, corner_plan_lookup, sizeof(unsigned int)*8);
    cudaMemcpyToSymbol(d_edge_plan_lookup, edge_plan_lookup, sizeof(unsigned int)*12);
    cudaMemcpyToSymbol(d_face_plan_lookup, face_plan_lookup, sizeof(unsigned int)*6);

    cudaMemcpyToSymbol(d_is_communicating, is_communicating, sizeof(unsigned int)*6);
    cudaMemcpyToSymbol(d_is_at_boundary, is_at_boundary, sizeof(unsigned int)*6);
    }

void gpu_deallocate_tmp_storage()
    {
    cudaFree(d_n_send_particles_corner);
    cudaFree(d_n_send_particles_edge);
    cudaFree(d_n_send_particles_face);
    cudaFree(d_n_remove_ptls);
    cudaFree(d_n_fetch_ptl);
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
__global__ void gpu_select_send_particles_kernel(const Scalar4 *d_pos,
                                                 const Scalar4 *d_vel,
                                                 const Scalar3 *d_accel,
                                                 const int3 *d_image,
                                                 const Scalar *d_charge,
                                                 const Scalar *d_diameter,
                                                 const unsigned int *d_body,
                                                 const Scalar4 *d_orientation,
                                                 const unsigned int *d_tag,
                                                 unsigned int *d_rtag,
                                                 char *corner_buf,
                                                 const unsigned int corner_buf_pitch,
                                                 char *edge_buf,
                                                 const unsigned int edge_buf_pitch,
                                                 char *face_buf,
                                                 const unsigned int face_buf_pitch,
                                                 unsigned char *remove_mask,
                                                 unsigned int *n_send_ptls_corner,
                                                 unsigned int *n_send_ptls_edge,
                                                 unsigned int *n_send_ptls_face,
                                                 unsigned int *n_remove_ptls,
                                                 unsigned int max_send_ptls_corner,
                                                 unsigned int max_send_ptls_edge,
                                                 unsigned int max_send_ptls_face,
                                                 unsigned int *condition,
                                                 unsigned int N,
                                                 const Scalar3 lo,
                                                 const Scalar3 hi,
                                                 const Scalar3 L)
    {
    unsigned int idx = blockIdx.x*blockDim.x + threadIdx.x;

    if (idx >= N)
        return;

    Scalar4 pos = d_pos[idx];

    unsigned int plan = 0;
    unsigned int count = 0;

    if ((pos.x >= hi.x) && d_is_communicating[face_east])
        {
        plan |= send_east;
        count++;
        }
    else if ((pos.x < lo.x) && d_is_communicating[face_west])
        {
        plan |= send_west;
        count++;
        }

    if ((pos.y >= hi.y) && d_is_communicating[face_north])
        {
        plan |= send_north;
        count++;
        }
    else if ((pos.y < lo.y) && d_is_communicating[face_south])
        {
        plan |= send_south;
        count++;
        }

    if ((pos.z >= hi.z) && d_is_communicating[face_up])
        {
        plan |= send_up; 
        count++;
        }
    else if ((pos.z < lo.z) && d_is_communicating[face_down])
        {
        plan |= send_down; 
        count++;
        }

    if (count)
        {
        const unsigned int pdata_size = sizeof(pdata_element_gpu);

        int3 image = d_image[idx];

        // apply global boundary conditions
        if ((plan & send_east) && d_is_at_boundary[0])
            {
            pos.x -= L.x;
            image.x++;
            }
        if ((plan & send_west) && d_is_at_boundary[1])
            {
            pos.x += L.x; 
            image.x--;
            }
        if ((plan & send_north) && d_is_at_boundary[2])
            {
            pos.y -= L.y;
            image.y++;
            }
        if ((plan & send_south) && d_is_at_boundary[3])
            {
            pos.y += L.y;
            image.y--;
            }
        if ((plan & send_up) && d_is_at_boundary[4])
            {
            pos.z -= L.z;
            image.z++;
            }
        if ((plan & send_down) && d_is_at_boundary[5])
            {
            pos.z += L.z; 
            image.z--;
            }

        // fill up buffer element
        pdata_element_gpu el;
        el.pos = pos;
        el.vel = d_vel[idx];
        el.accel = d_accel[idx];
        el.image = image;
        el.charge = d_charge[idx];
        el.diameter = d_diameter[idx];
        el.body = d_body[idx];
        el.orientation = d_orientation[idx];

        unsigned int tag = d_tag[idx];
        el.tag = tag;

        // mark particle for removal
        remove_mask[idx] = 1;

        // reset rtag
        d_rtag[tag] = NOT_LOCAL;

        atomicInc(n_remove_ptls, 0xffffffff);

        if (count == 1)
            {
            // face ptl
            unsigned int face = 0;
            for (unsigned int i = 0; i < 6; ++i)
                if (d_face_plan_lookup[i] == plan) face = i;

            unsigned int n = atomicInc(&n_send_ptls_face[face],0xffffffff);
            if (n < max_send_ptls_face)
                *((pdata_element_gpu *) &face_buf[n*pdata_size+face*face_buf_pitch]) = el;
            else
                atomicOr(condition,1);
            }
        else if (count == 2)
            {
            // edge ptl
            unsigned int edge = 0;
            for (unsigned int i = 0; i < 12; ++i)
                if (d_edge_plan_lookup[i] == plan) edge = i;

            unsigned int n = atomicInc(&n_send_ptls_edge[edge],0xffffffff);
            if (n < max_send_ptls_edge)
                *((pdata_element_gpu *) &edge_buf[n*pdata_size+edge*edge_buf_pitch]) = el;
            else
                atomicOr(condition,2);
            }
        else if (count == 3)
            {
            // corner ptl
            unsigned int corner;
            for (unsigned int i = 0; i < 8; ++i)
                if (d_corner_plan_lookup[i] == plan) corner = i;
 
            unsigned int n = atomicInc(&n_send_ptls_corner[corner],0xffffffff);
            if (n < max_send_ptls_corner)
                *((pdata_element_gpu *) &corner_buf[n*pdata_size+corner*corner_buf_pitch]) = el;
            else
                atomicOr(condition,4);
            }
        else
            {
            // invalid box
            atomicOr(condition,8);
            }
        }
    else
        remove_mask[idx] = 0;

    }

/*! Reorder the particles according to a migration criterium
 *  Particles that remain in the simulation box come first, followed by the particles that are sent in the
 *  specified direction
 *
 *  \param N Number of particles in local simulation box
 *  \param n_send_ptls_corner Number of particles that are sent over a corner (per corner)
 *  \param n_send_ptls_edge Number of particles that are sent over an edge (per edge)
 *  \param n_send_ptls_face Number of particles that are sent over a face (per face)
 *  \param n_remove_ptls Number of particles that will be removed
 *  \param n_max_send_ptls_corner Maximum size of corner send buf
 *  \param n_max_send_ptls_edge Maximum size of edge send buf
 *  \param n_max_send_ptls_face Maximum size of face send buf
 *  \param d_remove_mask Per-particle flag if particle has been sent
 *  \param d_corner_buf 2D Array of particle data elements that are sent over a corner
 *  \param corner_buf_pitch Pitch of 2D corner send buf
 *  \param d_edge_buf 2D Array of particle data elements that are sent over an edge
 *  \param edge_buf_pitch Pitch of 2D edge send buf
 *  \param d_face_buf 2D Array of particle data elements that are sent over a face
 *  \param face_buf_pitch Pitch of 2D face send buf
 *  \param tag_pitch
 *  \param box Dimensions of local simulation box
 *  \param dir Direction to send particles to
 */
void gpu_migrate_select_particles(unsigned int N,
                                  const Scalar4 *d_pos,
                                  const Scalar4 *d_vel,
                                  const Scalar3 *d_accel,
                                  const int3 *d_image,
                                  const Scalar *d_charge,
                                  const Scalar *d_diameter,
                                  const unsigned int *d_body,
                                  const Scalar4 *d_orientation,
                                  const unsigned int *d_tag,
                                  unsigned int *d_rtag,
                                  unsigned int *n_send_ptls_corner,
                                  unsigned int *n_send_ptls_edge,
                                  unsigned int *n_send_ptls_face,
                                  unsigned int &n_remove_ptls,
                                  unsigned n_max_send_ptls_corner,
                                  unsigned n_max_send_ptls_edge,
                                  unsigned n_max_send_ptls_face,
                                  unsigned char *d_remove_mask,
                                  char *d_corner_buf,
                                  unsigned int corner_buf_pitch,
                                  char *d_edge_buf,
                                  unsigned int edge_buf_pitch,
                                  char *d_face_buf,
                                  unsigned int face_buf_pitch,
                                  const BoxDim& box,
                                  const BoxDim& global_box,
                                  unsigned int *d_condition)
    {
    cudaMemset(d_n_send_particles_corner, 0, sizeof(unsigned int)*8);
    cudaMemset(d_n_send_particles_edge, 0, sizeof(unsigned int)*12);
    cudaMemset(d_n_send_particles_face, 0, sizeof(unsigned int)*6);
    cudaMemset(d_n_remove_ptls, 0, sizeof(unsigned int));

    unsigned int block_size = 512;

    gpu_select_send_particles_kernel<<<N/block_size+1,block_size>>>(d_pos,
                                                                    d_vel,
                                                                    d_accel,
                                                                    d_image,
                                                                    d_charge,
                                                                    d_diameter,
                                                                    d_body,
                                                                    d_orientation,
                                                                    d_tag,
                                                                    d_rtag,
                                                                    d_corner_buf,
                                                                    corner_buf_pitch,
                                                                    d_edge_buf,
                                                                    edge_buf_pitch,
                                                                    d_face_buf,
                                                                    face_buf_pitch,
                                                                    d_remove_mask,
                                                                    d_n_send_particles_corner,
                                                                    d_n_send_particles_edge,
                                                                    d_n_send_particles_face,
                                                                    d_n_remove_ptls,
                                                                    n_max_send_ptls_corner,
                                                                    n_max_send_ptls_edge,
                                                                    n_max_send_ptls_face,
                                                                    d_condition,
                                                                    N,
                                                                    box.getLo(), 
                                                                    box.getHi(),
                                                                    global_box.getL());

    cudaMemcpy(n_send_ptls_corner, d_n_send_particles_corner, 8*sizeof(unsigned int), cudaMemcpyDeviceToHost);
    cudaMemcpy(n_send_ptls_edge, d_n_send_particles_edge, 12*sizeof(unsigned int), cudaMemcpyDeviceToHost);
    cudaMemcpy(n_send_ptls_face, d_n_send_particles_face, 6*sizeof(unsigned int), cudaMemcpyDeviceToHost);
    cudaMemcpy(&n_remove_ptls, d_n_remove_ptls, sizeof(unsigned int), cudaMemcpyDeviceToHost);
    }

__global__ void gpu_migrate_fill_particle_arrays_kernel(unsigned int old_nparticles,
                                             unsigned int n_recv_ptls,
                                             unsigned int n_remove_ptls,
                                             unsigned int *n_fetch_ptl,
                                             unsigned char *remove_mask,
                                             char *recv_buf,
                                             float4 *d_pos,
                                             float4 *d_vel,
                                             float3 *d_accel,
                                             int3 *d_image,
                                             float *d_charge,
                                             float *d_diameter,
                                             unsigned int *d_body,
                                             float4 *d_orientation,
                                             unsigned int *d_tag,
                                             unsigned int *d_rtag)
    {
    unsigned int idx = blockDim.x * blockIdx.x + threadIdx.x;

    unsigned int new_nparticles = old_nparticles - n_remove_ptls + n_recv_ptls;

    if (idx >= new_nparticles) return;

    unsigned char replace = 1;

    if (idx < old_nparticles)
        replace = remove_mask[idx];

    if (replace)
        {
        // try to atomically fetch a particle from the received list
        unsigned int n = atomicInc(n_fetch_ptl, 0xffffffff);
       
        if (n < n_recv_ptls) 
            {
            // copy over receive buffer data
            pdata_element_gpu &el= ((pdata_element_gpu *) recv_buf)[n];

            d_pos[idx] = el.pos;
            d_vel[idx] = el.vel;
            d_accel[idx] = el.accel;
            d_image[idx] = el.image;
            d_charge[idx] = el.charge;
            d_diameter[idx] = el.diameter;
            d_body[idx] = el.body;
            d_orientation[idx] = el.orientation;

            unsigned int tag = el.tag;
            d_tag[idx] = tag;
            d_rtag[tag] = idx;
            }
        else
            {
            unsigned int fetch_idx = new_nparticles + (n - n_recv_ptls);
            unsigned char remove = remove_mask[fetch_idx];

            while (remove)  {
                n = atomicInc(n_fetch_ptl, 0xffffffff);
                fetch_idx = new_nparticles + (n - n_recv_ptls);
                remove = remove_mask[fetch_idx];
                }

            // backfill with a particle from the end
            d_pos[idx] = d_pos[fetch_idx];
            d_vel[idx] = d_vel[fetch_idx];
            d_accel[idx] = d_accel[fetch_idx];
            d_image[idx] = d_image[fetch_idx];
            d_charge[idx] = d_charge[fetch_idx];
            d_diameter[idx] = d_diameter[fetch_idx];
            d_body[idx] = d_body[fetch_idx];
            d_orientation[idx] = d_orientation[fetch_idx];

            unsigned int tag = d_tag[fetch_idx];
            d_tag[idx] = tag;
            d_rtag[tag] = idx;
            }
        } // if replace
    }

void gpu_migrate_fill_particle_arrays(unsigned int old_nparticles,
                        unsigned int n_recv_ptls,
                        unsigned int n_remove_ptls,
                        unsigned char *d_remove_mask,
                        char *d_recv_buf,
                        float4 *d_pos,
                        float4 *d_vel,
                        float3 *d_accel,
                        int3 *d_image,
                        float *d_charge,
                        float *d_diameter,
                        unsigned int *d_body,
                        float4 *d_orientation,
                        unsigned int *d_tag,
                        unsigned int *d_rtag)
    {
    cudaMemset(d_n_fetch_ptl, 0, sizeof(unsigned int));

    unsigned int block_size = 512;
    unsigned int new_end = old_nparticles + n_recv_ptls - n_remove_ptls;
    gpu_migrate_fill_particle_arrays_kernel<<<new_end/block_size+1,block_size>>>(old_nparticles,
                                             n_recv_ptls,
                                             n_remove_ptls,
                                             d_n_fetch_ptl,
                                             d_remove_mask,
                                             d_recv_buf,
                                             d_pos,
                                             d_vel,
                                             d_accel,
                                             d_image,
                                             d_charge,
                                             d_diameter,
                                             d_body,
                                             d_orientation,
                                             d_tag,
                                             d_rtag);
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

//! Kernel to pack local particle data into ghost send buffers
__global__ void gpu_exchange_ghosts_kernel(const unsigned int N,
                                         const unsigned char *d_plan,
                                         const unsigned int *d_tag,
                                         unsigned int *d_ghost_idx_face,
                                         unsigned int ghost_idx_face_pitch,
                                         unsigned int *d_ghost_idx_edge,
                                         unsigned int ghost_idx_edge_pitch,
                                         unsigned int *d_ghost_idx_corner,
                                         unsigned int ghost_idx_corner_pitch,
                                         float4 *d_pos,
                                         float *d_charge,
                                         float *d_diameter,
                                         char *d_ghost_corner_buf,
                                         unsigned int corner_buf_pitch,
                                         char *d_ghost_edge_buf,
                                         unsigned int edge_buf_pitch,
                                         char *d_ghost_face_buf,
                                         unsigned int face_buf_pitch,
                                         unsigned int *n_copy_ghosts_corner,
                                         unsigned int *n_copy_ghosts_edge,
                                         unsigned int *n_copy_ghosts_face,
                                         unsigned int max_copy_ghosts_corner,
                                         unsigned int max_copy_ghosts_edge,
                                         unsigned int max_copy_ghosts_face,
                                         unsigned int *condition)
    {
    unsigned int idx = blockIdx.x*blockDim.x + threadIdx.x;
    
    if (idx >= N) return;

    unsigned char plan = d_plan[idx];

    // if we are not communicating in a direction, discard it from plan
    for (unsigned int face = 0; face < 6; ++face)
        plan &= (d_is_communicating[face]) ? ~0 : ~d_face_plan_lookup[face];

    if (plan)
        {
        ghost_element_gpu el;
        el.pos = d_pos[idx];
        el.charge = d_charge[idx];
        el.diameter = d_diameter[idx];
        el.tag = d_tag[idx];

        // the boundary plan indicates whether the particle will cross a particle
        // in a specific direction
        unsigned int boundary_plan = 0;
        for (unsigned int face = 0; face < 6; ++face)
            if (d_is_at_boundary[face] && (plan & d_face_plan_lookup[face]))
                boundary_plan |= d_face_plan_lookup[face];

        el.plan = boundary_plan;

        unsigned int count = 0;
        if (plan & send_east) count++;
        if (plan & send_west) count++;
        if (plan & send_north) count++;
        if (plan & send_south) count++;
        if (plan & send_up) count++;
        if (plan & send_down) count++;

        const unsigned int ghost_size = sizeof(ghost_element_gpu);

       // determine corner to send ptl to
       if (count == 3)
            {
            // corner ptl
            unsigned int corner = 0;
            bool has_corner = false;
            for (unsigned int i = 0; i < 8; ++i)
                {
                if ((plan & d_corner_plan_lookup[i]) == d_corner_plan_lookup[i])
                    {
                    corner = i;
                    has_corner = true;
                    }
                }
            if (!has_corner)
                atomicOr(condition, 8); // invalid plan

            unsigned int n = atomicInc(&n_copy_ghosts_corner[corner],0xffffffff);
            if (n < max_copy_ghosts_corner)
                {
                *((ghost_element_gpu *) &d_ghost_corner_buf[n*ghost_size+corner*corner_buf_pitch]) = el;
                d_ghost_idx_corner[n+corner*ghost_idx_corner_pitch] = idx;
                }
            else
                // overflow
                atomicOr(condition,4);
            }
 
        // determine box edge to copy ptl to
        if (count == 2)
            {
            bool has_edge = false;
            unsigned edge = 0;
            for (unsigned int i = 0; i < 12; ++i)
                if ((plan & d_edge_plan_lookup[i]) == d_edge_plan_lookup[i])
                    {
                    has_edge = true;
                    edge = i;
                    break;
                    }

            if (!has_edge)
                atomicOr(condition,8); // invalid plan

            unsigned int n = atomicInc(&n_copy_ghosts_edge[edge],0xffffffff);
            if (n < max_copy_ghosts_edge)
                {
                *((ghost_element_gpu *) &d_ghost_edge_buf[n*ghost_size+edge*edge_buf_pitch]) = el;

                // store particle index in ghost copying lists
                d_ghost_idx_edge[n+edge*ghost_idx_edge_pitch] = idx;
                }
            else
                // overflow
                atomicOr(condition,2);
            }

        // determine box face to copy ptl
        if (count == 1)
            {
            bool has_face = false;
            unsigned int face = 0;
            for (unsigned int i = 0; i < 6; ++i)
                if ((plan & d_face_plan_lookup[i]) == d_face_plan_lookup[i])
                    {
                    face = i;
                    has_face = true;
                    break;
                    }

            if (!has_face)
                atomicOr(condition,8); // invalid plan
            
            unsigned int n = atomicInc(&n_copy_ghosts_face[face],0xffffffff);
            if (n < max_copy_ghosts_face)
                {
                *((ghost_element_gpu *) &d_ghost_face_buf[n*ghost_size+face*face_buf_pitch]) = el;

                // store particle index in ghost copying lists
                d_ghost_idx_face[n+face*ghost_idx_face_pitch] = idx;
                }
            else
                // overflow
                atomicOr(condition,1);

            }
        
        if (count > 3)
            atomicOr(condition,8); // invalid plan
        } 
    }

//! Construct a list of particle tags to send as ghost particles
/*! \param n_total Total number of particles to check
 * \param N number of local particles
 * \param dir Direction in which ghost particles are sent
 * \param d_plan Array of particle exchange plans
 * \param d_tag Array of particle global tags
 * \param d_copy_ghosts Array to be fillled with indices of particles that are to be sent as ghosts
 * \param d_ghost_tag Array of ghost particle tags to be sent
 * \param n_copy_ghosts Number of local particles that are sent in the given direction as ghosts (return value)
 */
void gpu_exchange_ghosts(const unsigned int N,
                         const unsigned char *d_plan,
                         const unsigned int *d_tag,
                         unsigned int *d_ghost_idx_face,
                         unsigned int ghost_idx_face_pitch,
                         unsigned int *d_ghost_idx_edge,
                         unsigned int ghost_idx_edge_pitch,
                         unsigned int *d_ghost_idx_corner,
                         unsigned int ghost_idx_corner_pitch,
                         float4 *d_pos,
                         float *d_charge,
                         float *d_diameter,
                         char *d_ghost_corner_buf,
                         unsigned int corner_buf_pitch,
                         char *d_ghost_edge_buf,
                         unsigned int edge_buf_pitch,
                         char *d_ghost_face_buf,
                         unsigned int face_buf_pitch,
                         unsigned int *d_n_copy_ghosts_corner,
                         unsigned int *d_n_copy_ghosts_edge,
                         unsigned int *d_n_copy_ghosts_face,
                         unsigned int max_copy_ghosts_corner,
                         unsigned int max_copy_ghosts_edge,
                         unsigned int max_copy_ghosts_face,
                         unsigned int *d_condition)
    {
    cudaMemsetAsync(d_n_copy_ghosts_corner, 0, sizeof(unsigned int)*8,0);
    cudaMemsetAsync(d_n_copy_ghosts_edge, 0, sizeof(unsigned int)*12,0);
    cudaMemsetAsync(d_n_copy_ghosts_face, 0, sizeof(unsigned int)*6,0);


    unsigned int block_size = 512;
    gpu_exchange_ghosts_kernel<<<N/block_size+1, block_size>>>(N,
                         d_plan,
                         d_tag,
                         d_ghost_idx_face,
                         ghost_idx_face_pitch,
                         d_ghost_idx_edge,
                         ghost_idx_edge_pitch,
                         d_ghost_idx_corner,
                         ghost_idx_corner_pitch,
                         d_pos,
                         d_charge,
                         d_diameter,
                         d_ghost_corner_buf,
                         corner_buf_pitch,
                         d_ghost_edge_buf,
                         edge_buf_pitch,
                         d_ghost_face_buf,
                         face_buf_pitch,
                         d_n_copy_ghosts_corner,
                         d_n_copy_ghosts_edge,
                         d_n_copy_ghosts_face,
                         max_copy_ghosts_corner,
                         max_copy_ghosts_edge,
                         max_copy_ghosts_face,
                         d_condition);
    }

template<class element_type, bool update>
__global__ void gpu_exchange_ghosts_unpack_kernel(unsigned int N,
                                                  unsigned int n_tot_recv_ghosts,
                                                  const unsigned int *n_local_ghosts_face,
                                                  const unsigned int *n_local_ghosts_edge,
                                                  const unsigned int n_tot_recv_ghosts_local,
                                                  const unsigned int *n_recv_ghosts_local,
                                                  const unsigned int *n_forward_ghosts_face,
                                                  const unsigned int *n_forward_ghosts_edge,
                                                  const char *d_face_ghosts,
                                                  const unsigned int face_pitch,
                                                  const char *d_edge_ghosts,
                                                  const unsigned int edge_pitch,
                                                  const char *d_recv_ghosts,
                                                  Scalar4 *d_pos,
                                                  Scalar *d_charge,
                                                  Scalar *d_diameter,
                                                  unsigned int *d_tag,
                                                  unsigned int *d_rtag,
                                                  unsigned int *d_ghost_plan,
                                                  const Scalar3 L)
    {
    unsigned int ghost_idx = blockIdx.x*blockDim.x+threadIdx.x;

    if (ghost_idx >= n_tot_recv_ghosts) return;

    element_type *el_ptr;
    const unsigned int ghost_size = sizeof(element_type);

    // decide for each ghost particle from which buffer we are fetching it from
    unsigned int offset = 0;

    // first ghosts that are received for the local box only
    bool done = false;
    unsigned int local_idx = ghost_idx;
    int recv_dir = -1;
    if (local_idx < n_tot_recv_ghosts_local)
        {
        unsigned int local_offset = 0;
        for (recv_dir = 0; recv_dir < 6; ++recv_dir)
            {
            local_offset +=n_recv_ghosts_local[recv_dir];
            if (local_idx < local_offset) break;
            }

        el_ptr = (element_type *) &d_recv_ghosts[local_idx*ghost_size];
        done = true;
        }
    else
        offset += n_tot_recv_ghosts_local;
    
    if (! done)
        {
        unsigned int n_tot_forward_ghosts_face[6];

        for (unsigned int i = 0; i < 6; ++i)
            {
            n_tot_forward_ghosts_face[i] = 0;
            for (unsigned int j = 0; j < 6; ++j)
                n_tot_forward_ghosts_face[i] += n_forward_ghosts_face[6*i+j];
            }

        // ghosts we have forwarded over a face of our box
        for (unsigned int i=0; i < 6; ++i)
            {
            local_idx = ghost_idx - offset;

            if (local_idx < n_tot_forward_ghosts_face[i])
                {
                unsigned int local_offset = 0;
                for (recv_dir = 0; recv_dir < 6; ++recv_dir)
                    {
                    local_offset += n_forward_ghosts_face[6*recv_dir+i];
                    if (local_idx < local_offset) break;
                    }

                unsigned int n = n_local_ghosts_face[i]+local_idx;
                el_ptr = (element_type *) &d_face_ghosts[n*ghost_size + i*face_pitch];
                done = true;
                break;
                }
            else
                offset += n_tot_forward_ghosts_face[i];
            }
        }

    if (! done)
        {
        unsigned int n_tot_forward_ghosts_edge[6];

        for (unsigned int i = 0; i < 6; ++i)
            {
            n_tot_forward_ghosts_edge[i] = 0;
            for (unsigned int j = 0; j < 12; ++j)
                n_tot_forward_ghosts_edge[i] += n_forward_ghosts_edge[12*i+j];
            }

        // ghosts we have forwared over an edge of our box
        for (unsigned int i=0; i < 12; ++i)
            {
            local_idx = ghost_idx - offset;

            if (local_idx < n_tot_forward_ghosts_edge[i])
                {
                unsigned int local_offset = 0;
                for (recv_dir = 0; recv_dir < 6; ++recv_dir)
                    {
                    local_offset += n_forward_ghosts_edge[12*recv_dir+i];
                    if (local_idx < local_offset) break;
                    }

                unsigned int n = n_local_ghosts_edge[i]+local_idx;
                el_ptr = (element_type *) &d_edge_ghosts[n*ghost_size + i*edge_pitch];
                done = true;
                break;
                }
            else
                offset += n_tot_forward_ghosts_edge[i];
            }
        }

    // we have a pointer to the data element to be unpacked, now unpack
    Scalar4 postype;
    unsigned int boundary_plan;
    if (update)
        {
        // only update position
        update_element_gpu &el = *(update_element_gpu *) el_ptr;
        postype = el.pos;

        // fetch previously saved plan
        boundary_plan = d_ghost_plan[ghost_idx];
        }
    else
        {
        // unpack all data needed for force computation
        ghost_element_gpu &el = *(ghost_element_gpu *) el_ptr;
        postype = el.pos;

        d_charge[N+ghost_idx] = el.charge;
        d_diameter[N+ghost_idx] = el.diameter;
        unsigned int tag = el.tag;
        d_tag[N+ghost_idx] = tag;
        d_rtag[tag] = N+ghost_idx;
        
        // save plan
        boundary_plan = el.plan;
        d_ghost_plan[ghost_idx] = boundary_plan;
        }

    // apply global boundary conditions for received particle

    // if the plan indicates the particle has crossed a boundary prior to arriving here,
    // apply appropriate boundary conditions

    if ((boundary_plan & send_east) && (face_east <= recv_dir) && d_is_at_boundary[face_west])
        postype.x -= L.x;
    if ((boundary_plan & send_west) && (face_west <= recv_dir) && d_is_at_boundary[face_east])
        postype.x += L.x;
    if ((boundary_plan & send_north) && (face_north <= recv_dir) && d_is_at_boundary[face_south])
        postype.y -= L.y;
    if ((boundary_plan & send_south) && (face_south <= recv_dir) && d_is_at_boundary[face_north])
        postype.y += L.y;
    if ((boundary_plan & send_up) && (face_up <= recv_dir) && d_is_at_boundary[face_down])
        postype.z -= L.z;
    if ((boundary_plan & send_down) && (face_down <= recv_dir) && d_is_at_boundary[face_up])
        postype.z += L.z;

    d_pos[N+ghost_idx] = postype;
    }

void gpu_exchange_ghosts_unpack(unsigned int N,
                                unsigned int n_tot_recv_ghosts,
                                const unsigned int *d_n_local_ghosts_face,
                                const unsigned int *d_n_local_ghosts_edge,
                                const unsigned int n_tot_recv_ghosts_local,
                                const unsigned int *d_n_recv_ghosts_local,
                                const unsigned int *d_n_recv_ghosts_face,
                                const unsigned int *d_n_recv_ghosts_edge,
                                const char *d_face_ghosts,
                                const unsigned int face_pitch,
                                const char *d_edge_ghosts,
                                const unsigned int edge_pitch,
                                const char *d_recv_ghosts,
                                Scalar4 *d_pos,
                                Scalar *d_charge,
                                Scalar *d_diameter,
                                unsigned int *d_tag,
                                unsigned int *d_rtag,
                                unsigned int *d_ghost_plan,
                                const BoxDim& global_box)
    {
    unsigned int block_size = 512;
    gpu_exchange_ghosts_unpack_kernel<ghost_element_gpu, false><<<n_tot_recv_ghosts/block_size+1, block_size>>>(N,
                                                                           n_tot_recv_ghosts,
                                                                           d_n_local_ghosts_face,
                                                                           d_n_local_ghosts_edge,
                                                                           n_tot_recv_ghosts_local,
                                                                           d_n_recv_ghosts_local,
                                                                           d_n_recv_ghosts_face,
                                                                           d_n_recv_ghosts_edge,
                                                                           d_face_ghosts,
                                                                           face_pitch,
                                                                           d_edge_ghosts,
                                                                           edge_pitch,
                                                                           d_recv_ghosts,
                                                                           d_pos,
                                                                           d_charge,
                                                                           d_diameter,
                                                                           d_tag,
                                                                           d_rtag,
                                                                           d_ghost_plan,
                                                                           global_box.getL());
    } 

//! Kernel to pack local particle data into ghost send buffers
__global__ void gpu_update_ghosts_pack_kernel(const unsigned int n_copy_ghosts,
                                         const unsigned int *d_ghost_idx_face,
                                         const unsigned int ghost_idx_face_pitch,
                                         const unsigned int *d_ghost_idx_edge,
                                         const unsigned int ghost_idx_edge_pitch,
                                         const unsigned int *d_ghost_idx_corner,
                                         const unsigned int ghost_idx_corner_pitch,
                                         const float4 *d_pos,
                                         char *d_update_corner_buf,
                                         unsigned int corner_buf_pitch,
                                         char *d_update_edge_buf,
                                         unsigned int edge_buf_pitch,
                                         char *d_update_face_buf,
                                         unsigned int face_buf_pitch,
                                         const unsigned int *n_copy_ghosts_corner,
                                         const unsigned int *n_copy_ghosts_edge,
                                         const unsigned int *n_copy_ghosts_face)
    {
    unsigned int ghost_idx = blockIdx.x*blockDim.x+threadIdx.x;

    if (ghost_idx >= n_copy_ghosts) return;

    // this kernel traverses 26 index buffers at once by mapping
    // a continuous local ghost particle index onto a single element in a single index buffer.
    // Then it fetches the particle data for the particle index stored in that buffer
    // and writes it to the send buffer

    // order does matter, inside the individual buffers the particle order must be the same as the one
    // in the ghost index buffer (created by exchange_ghosts kernel)
    unsigned int offset = 0;

    // the particle index we are going to fetch (initialized with a dummy value)
    unsigned int idx = NOT_LOCAL;

    bool done = false;

    update_element_gpu *buf_ptr = NULL; 
    const unsigned int update_size = sizeof(update_element_gpu);

    // first, ghosts that are sent over a corner
    for (unsigned int corner=0; corner < 8; ++corner)
        {
        unsigned int local_idx = ghost_idx - offset;

        if (local_idx < n_copy_ghosts_corner[corner])
            {
            idx = d_ghost_idx_corner[local_idx + corner * ghost_idx_corner_pitch];
            buf_ptr = (update_element_gpu *) &d_update_corner_buf[local_idx*update_size + corner*corner_buf_pitch];
            done = true;
            break;
            }
        else
            offset += n_copy_ghosts_corner[corner];
        }
  
    if (! done)
        {
        // second, ghosts that are sent over an edge
        for (unsigned int edge=0; edge < 12; ++edge)
            {
            unsigned int local_idx = ghost_idx - offset;

            if (local_idx < n_copy_ghosts_edge[edge])
                {
                idx = d_ghost_idx_edge[local_idx + edge * ghost_idx_edge_pitch];
                buf_ptr = (update_element_gpu *) &d_update_edge_buf[local_idx*update_size + edge*edge_buf_pitch];
                done = true;
                break;
                }
            else
                offset += n_copy_ghosts_edge[edge];
            }
        }

    if (!done)
        {
        // third, ghosts that are sent through a face of the box
        for (unsigned int face=0; face < 6; ++face)
            {
            unsigned int local_idx = ghost_idx - offset;

            if (local_idx < n_copy_ghosts_face[face])
                {
                idx = d_ghost_idx_face[local_idx + face * ghost_idx_face_pitch];
                buf_ptr = (update_element_gpu *) &d_update_face_buf[local_idx*update_size + face*face_buf_pitch];
                done = true;
                break;
                }
            else
                offset += n_copy_ghosts_face[face];
            }
        }

    // we have found a ghost index to be updated
    // store data in buffer element
    update_element_gpu &el = *buf_ptr;
    el.pos = d_pos[idx];
    }

//! Pack local particle data into ghost send buffers
void gpu_update_ghosts_pack(const unsigned int n_copy_ghosts,
                                     const unsigned int *d_ghost_idx_face,
                                     const unsigned int ghost_idx_face_pitch,
                                     const unsigned int *d_ghost_idx_edge,
                                     const unsigned int ghost_idx_edge_pitch,
                                     const unsigned int *d_ghost_idx_corner,
                                     const unsigned int ghost_idx_corner_pitch,
                                     const float4 *d_pos,
                                     char *d_update_corner_buf,
                                     unsigned int corner_buf_pitch,
                                     char *d_update_edge_buf,
                                     unsigned int edge_buf_pitch,
                                     char *d_update_face_buf,
                                     unsigned int face_buf_pitch,
                                     const unsigned int *d_n_local_ghosts_corner,
                                     const unsigned int *d_n_local_ghosts_edge,
                                     const unsigned int *d_n_local_ghosts_face,
                                     cudaStream_t stream)
    {
    unsigned int block_size = 512;
    gpu_update_ghosts_pack_kernel<<<n_copy_ghosts/block_size+1,block_size,0,stream>>>(n_copy_ghosts,
                                                                             d_ghost_idx_face,
                                                                             ghost_idx_face_pitch,
                                                                             d_ghost_idx_edge,
                                                                             ghost_idx_edge_pitch,
                                                                             d_ghost_idx_corner,
                                                                             ghost_idx_corner_pitch,
                                                                             d_pos,
                                                                             d_update_corner_buf,
                                                                             corner_buf_pitch,
                                                                             d_update_edge_buf,
                                                                             edge_buf_pitch,
                                                                             d_update_face_buf,
                                                                             face_buf_pitch,
                                                                             d_n_local_ghosts_corner,
                                                                             d_n_local_ghosts_edge,
                                                                             d_n_local_ghosts_face);
    }

void gpu_update_ghosts_unpack(unsigned int N,
                                unsigned int n_tot_recv_ghosts,
                                const unsigned int *d_n_local_ghosts_face,
                                const unsigned int *d_n_local_ghosts_edge,
                                const unsigned int n_tot_recv_ghosts_local,
                                const unsigned int *d_n_recv_ghosts_local,
                                const unsigned int *d_n_recv_ghosts_face,
                                const unsigned int *d_n_recv_ghosts_edge,
                                const char *d_face_ghosts,
                                const unsigned int face_pitch,
                                const char *d_edge_ghosts,
                                const unsigned int edge_pitch,
                                const char *d_recv_ghosts,
                                Scalar4 *d_pos,
                                unsigned int *d_ghost_plan,
                                const BoxDim& global_box,
                                cudaStream_t stream)
    {
    unsigned int block_size = 512;
    gpu_exchange_ghosts_unpack_kernel<update_element_gpu, true><<<n_tot_recv_ghosts/block_size+1, block_size,0,stream>>>
                                                                          (N,
                                                                           n_tot_recv_ghosts,
                                                                           d_n_local_ghosts_face,
                                                                           d_n_local_ghosts_edge,
                                                                           n_tot_recv_ghosts_local,
                                                                           d_n_recv_ghosts_local,
                                                                           d_n_recv_ghosts_face,
                                                                           d_n_recv_ghosts_edge,
                                                                           d_face_ghosts,
                                                                           face_pitch,
                                                                           d_edge_ghosts,
                                                                           edge_pitch,
                                                                           d_recv_ghosts,
                                                                           d_pos,
                                                                           NULL,
                                                                           NULL,
                                                                           NULL,
                                                                           NULL,
                                                                           d_ghost_plan,
                                                                           global_box.getL());
    } 

#endif
