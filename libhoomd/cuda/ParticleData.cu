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

#include "ParticleData.cuh"

/*! \file ParticleData.cu
    \brief ImplementsGPU kernel code and data structure functions used by ParticleData
*/

#ifdef ENABLE_MPI

#include <thrust/iterator/zip_iterator.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/permutation_iterator.h>
#include <thrust/device_ptr.h>

#include "moderngpu/kernels/scan.cuh"

//! A tuple of pdata pointers
typedef thrust::tuple <
    thrust::device_ptr<unsigned int>,  // tag
    thrust::device_ptr<Scalar4>,       // pos
    thrust::device_ptr<Scalar4>,       // vel
    thrust::device_ptr<Scalar3>,       // accel
    thrust::device_ptr<Scalar>,        // charge
    thrust::device_ptr<Scalar>,        // diameter
    thrust::device_ptr<int3>,          // image
    thrust::device_ptr<unsigned int>,  // body
    thrust::device_ptr<Scalar4>,       // orientation
    thrust::device_ptr<unsigned int>   // communication flags
    > pdata_it_tuple_gpu;

//! A zip iterator for filtering particle data
typedef thrust::zip_iterator<pdata_it_tuple_gpu> pdata_zip_gpu;

//! A tuple of pdata fields
typedef thrust::tuple <
    unsigned int,  // tag
    Scalar4,       // pos
    Scalar4,       // vel
    Scalar3,       // accel
    Scalar,        // charge
    Scalar,        // diameter
    int3,          // image
    unsigned int,  // body
    Scalar4,       // orientation
    unsigned int   // communication flags
    > pdata_tuple_gpu;

//! Kernel to partition particle data
__global__ void gpu_scatter_particle_data_kernel(
    const unsigned int N,
    const Scalar4 *d_pos,
    const Scalar4 *d_vel,
    const Scalar3 *d_accel,
    const Scalar *d_charge,
    const Scalar *d_diameter,
    const int3 *d_image,
    const unsigned int *d_body,
    const Scalar4 *d_orientation,
    const unsigned int *d_tag,
    unsigned int *d_rtag,
    Scalar4 *d_pos_alt,
    Scalar4 *d_vel_alt,
    Scalar3 *d_accel_alt,
    Scalar *d_charge_alt,
    Scalar *d_diameter_alt,
    int3 *d_image_alt,
    unsigned int *d_body_alt,
    Scalar4 *d_orientation_alt,
    unsigned int *d_tag_alt,
    pdata_element *d_out,
    unsigned int *d_comm_flags,
    unsigned int *d_comm_flags_out,
    const unsigned int *d_scan)
    {
    unsigned int idx = blockIdx.x*blockDim.x + threadIdx.x;

    if (idx >= N) return;

    bool remove = d_comm_flags[idx];

    unsigned int scan_remove = d_scan[idx];
    unsigned int scan_keep = idx - scan_remove;

    if (remove)
        {
        pdata_element p;
        p.pos = d_pos[idx];
        p.vel = d_vel[idx];
        p.accel = d_accel[idx];
        p.charge = d_charge[idx];
        p.diameter = d_diameter[idx];
        p.image = d_image[idx];
        p.body = d_body[idx];
        p.orientation = d_orientation[idx];
        p.tag = d_tag[idx];
        d_out[scan_remove] = p;
        d_comm_flags_out[scan_remove] = d_comm_flags[idx];

        // reset communication flags
        d_comm_flags[idx] = 0;

        // reset rtag
        d_rtag[p.tag] = NOT_LOCAL;
        }
    else
        {
        d_pos_alt[scan_keep] = d_pos[idx];
        d_vel_alt[scan_keep] = d_vel[idx];
        d_accel_alt[scan_keep] = d_accel[idx];
        d_charge_alt[scan_keep] = d_charge[idx];
        d_diameter_alt[scan_keep] = d_diameter[idx];
        d_image_alt[scan_keep] = d_image[idx];
        d_body_alt[scan_keep] = d_body[idx];
        d_orientation_alt[scan_keep] = d_orientation[idx];
        unsigned int tag = d_tag[idx];
        d_tag_alt[scan_keep] = tag;

        // update rtag
        d_rtag[tag] = scan_keep;
        }

    }

struct gpu_comm_flag_set : thrust::unary_function<unsigned int, unsigned int>
    {
    __device__ unsigned int operator() (unsigned int comm_flag)
        {
        return comm_flag ? 1 : 0;
        }
    };

/*! \param N Number of local particles
    \param d_pos Device array of particle positions
    \param d_vel Device array of particle velocities
    \param d_accel Device array of particle accelerations
    \param d_charge Device array of particle charges
    \param d_diameter Device array of particle diameters
    \param d_image Device array of particle images
    \param d_body Device array of particle body tags
    \param d_orientation Device array of particle orientations
    \param d_tag Device array of particle tags
    \param d_rtag Device array for reverse-lookup table
    \param d_pos_alt Device array of particle positions (output)
    \param d_vel_alt Device array of particle velocities (output)
    \param d_accel_alt Device array of particle accelerations (output)
    \param d_charge_alt Device array of particle charges (output)
    \param d_diameter_alt Device array of particle diameters (output)
    \param d_image_alt Device array of particle images (output)
    \param d_body_alt Device array of particle body tags (output)
    \param d_orientation_alt Device array of particle orientations (output)
    \param d_out Output array for packed particle data
    \param max_n_out Maximum number of elements to write to output array

    \returns Number of elements marked for removal
 */
unsigned int gpu_pdata_remove(const unsigned int N,
                    const Scalar4 *d_pos,
                    const Scalar4 *d_vel,
                    const Scalar3 *d_accel,
                    const Scalar *d_charge,
                    const Scalar *d_diameter,
                    const int3 *d_image,
                    const unsigned int *d_body,
                    const Scalar4 *d_orientation,
                    const unsigned int *d_tag,
                    unsigned int *d_rtag,
                    Scalar4 *d_pos_alt,
                    Scalar4 *d_vel_alt,
                    Scalar3 *d_accel_alt,
                    Scalar *d_charge_alt,
                    Scalar *d_diameter_alt,
                    int3 *d_image_alt,
                    unsigned int *d_body_alt,
                    Scalar4 *d_orientation_alt,
                    unsigned int *d_tag_alt,
                    pdata_element *d_out,
                    unsigned int *d_comm_flags,
                    unsigned int *d_comm_flags_out,
                    unsigned int max_n_out,
                    mgpu::ContextPtr mgpu_context,
                    cached_allocator& alloc)
    {
    unsigned int n_out;

    // allocate temp array for scan results
    unsigned int *d_tmp = (unsigned int *)alloc.allocate(N*sizeof(unsigned int));

    thrust::device_ptr<const unsigned int> comm_flags_ptr(d_comm_flags);

    mgpu::Scan<mgpu::MgpuScanTypeExc>(
        thrust::make_transform_iterator(comm_flags_ptr, gpu_comm_flag_set()),
        N, (unsigned int) 0, mgpu::plus<unsigned int>(), (unsigned int *)NULL, &n_out, d_tmp, *mgpu_context);

    // Don't write past end of buffer
    if (n_out <= max_n_out)
        {
        // partition particle data into local and removed particles
        unsigned int block_size =512;
        unsigned int n_blocks = N/block_size+1;

        gpu_scatter_particle_data_kernel<<<n_blocks, block_size>>>(
            N,
            d_pos,
            d_vel,
            d_accel,
            d_charge,
            d_diameter,
            d_image,
            d_body,
            d_orientation,
            d_tag,
            d_rtag,
            d_pos_alt,
            d_vel_alt,
            d_accel_alt,
            d_charge_alt,
            d_diameter_alt,
            d_image_alt,
            d_body_alt,
            d_orientation_alt,
            d_tag_alt,
            d_out,
            d_comm_flags,
            d_comm_flags_out,
            d_tmp);
        }

    // deallocate tmp array
    alloc.deallocate((char *)d_tmp,0);

    // return elements written to output stream
    return n_out;
    }

//! A converter from pdata_element to a tuple of tag and a tuple of pdata entries
/*! Writes the tag into the rtag table at the same time
 */
struct to_pdata_tuple_gpu : public thrust::unary_function<const pdata_element, pdata_tuple_gpu>
    {
    __device__ const pdata_tuple_gpu operator() (const pdata_element p)
        {
        // make tuple
        return thrust::make_tuple(
            p.tag,
            p.pos,
            p.vel,
            p.accel,
            p.charge,
            p.diameter,
            p.image,
            p.body,
            p.orientation,
            0 // communication flags
            );
        }
    };


/*! \param old_nparticles old local particle count
    \param num_add_ptls Number of particles in input array
    \param d_pos Device array of particle positions
    \param d_vel Device iarray of particle velocities
    \param d_accel Device array of particle accelerations
    \param d_charge Device array of particle charges
    \param d_diameter Device array of particle diameters
    \param d_image Device array of particle images
    \param d_body Device array of particle body tags
    \param d_orientation Device array of particle orientations
    \param d_tag Device array of particle tags
    \param d_rtag Device array for reverse-lookup table
    \param d_in Device array of packed input particle data
    \param d_comm_flags Device array of communication flags (pdata)
*/
void gpu_pdata_add_particles(const unsigned int old_nparticles,
                    const unsigned int num_add_ptls,
                    Scalar4 *d_pos,
                    Scalar4 *d_vel,
                    Scalar3 *d_accel,
                    Scalar *d_charge,
                    Scalar *d_diameter,
                    int3 *d_image,
                    unsigned int *d_body,
                    Scalar4 *d_orientation,
                    unsigned int *d_tag,
                    unsigned int *d_rtag,
                    const pdata_element *d_in,
                    unsigned int *d_comm_flags,
                    cached_allocator& alloc)
    {
    // wrap device arrays into thrust ptr
    thrust::device_ptr<Scalar4> pos_ptr(d_pos);
    thrust::device_ptr<Scalar4> vel_ptr(d_vel);
    thrust::device_ptr<Scalar3> accel_ptr(d_accel);
    thrust::device_ptr<Scalar> charge_ptr(d_charge);
    thrust::device_ptr<Scalar> diameter_ptr(d_diameter);
    thrust::device_ptr<int3> image_ptr(d_image);
    thrust::device_ptr<unsigned int> body_ptr(d_body);
    thrust::device_ptr<Scalar4> orientation_ptr(d_orientation);
    thrust::device_ptr<unsigned int> tag_ptr(d_tag);
    thrust::device_ptr<unsigned int> comm_flags_ptr(d_comm_flags);

    // wrap input array
    thrust::device_ptr<const pdata_element> in_ptr(d_in);

    // Construct zip iterator
    pdata_zip_gpu pdata_begin(
       thrust::make_tuple(
            tag_ptr,
            pos_ptr,
            vel_ptr,
            accel_ptr,
            charge_ptr,
            diameter_ptr,
            image_ptr,
            body_ptr,
            orientation_ptr,
            comm_flags_ptr
            )
        );
    pdata_zip_gpu pdata_end = pdata_begin + old_nparticles;

    // wrap reverse-lookup table
    thrust::device_ptr<unsigned int> rtag_ptr(d_rtag);

    typedef thrust::counting_iterator<unsigned int> count_it;

    // add new particles at the end, writing rtags at the same time
    thrust::transform(in_ptr, in_ptr + num_add_ptls, pdata_end, to_pdata_tuple_gpu());

    // update rtags
    thrust::counting_iterator<unsigned int> idx(old_nparticles);
    thrust::scatter(idx, idx + num_add_ptls, tag_ptr+old_nparticles, rtag_ptr);
    }

#endif // ENABLE_MPI
