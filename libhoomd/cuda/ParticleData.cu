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

#include "clipped_range.h"

/*! \file ParticleData.cu
    \brief ImplementsGPU kernel code and data structure functions used by ParticleData
*/

#ifdef ENABLE_MPI

#include <thrust/iterator/zip_iterator.h>
#include <thrust/iterator/retag.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/functional.h>
#include <thrust/device_ptr.h>
#include <thrust/partition.h>

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
    thrust::device_ptr<Scalar4>        // orientation
    > pdata_it_tuple_gpu;

//! A tuple of pdata pointers (const version)
typedef thrust::tuple <
    thrust::device_ptr<const unsigned int>,  // tag
    thrust::device_ptr<const Scalar4>,       // pos
    thrust::device_ptr<const Scalar4>,       // vel
    thrust::device_ptr<const Scalar3>,       // accel
    thrust::device_ptr<const Scalar>,        // charge
    thrust::device_ptr<const Scalar>,        // diameter
    thrust::device_ptr<const int3>,          // image
    thrust::device_ptr<const unsigned int>,  // body
    thrust::device_ptr<const Scalar4>        // orientation
    > pdata_it_tuple_gpu_const;

//! A zip iterator for filtering particle data
typedef thrust::zip_iterator<pdata_it_tuple_gpu> pdata_zip_gpu;

//! A zip iterator for filtering particle data (const version)
typedef thrust::zip_iterator<pdata_it_tuple_gpu_const> pdata_zip_gpu_const;


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
    Scalar4        // orientation
    > pdata_tuple_gpu;

//! A predicate to select pdata tuples by rtag
struct combined_tuple_select_rtag_gpu
    {
    //! Constructor
    combined_tuple_select_rtag_gpu(thrust::device_ptr<unsigned int> _rtag_ptr, const unsigned int _compare)
        :  rtag_ptr(_rtag_ptr), compare(_compare)
        { }

    //! Returns true if the remove flag is set for a particle
    __device__ bool operator() (const thrust::tuple<pdata_tuple_gpu,pdata_element> t) const
        {
        unsigned int tag = thrust::get<0>(thrust::get<0>(t));
        return rtag_ptr[tag] == compare;
        }

    thrust::device_ptr<unsigned int> rtag_ptr; //!< Reverse-lookup table
    const unsigned int compare;                //!< rtag value to compare to
    };

//! A converter from a tuple of pdata entries to a pdata_element
struct to_pdata_element_gpu : public thrust::unary_function<const pdata_tuple_gpu,const pdata_element>
    {
    __device__ const pdata_element operator() (const pdata_tuple_gpu t)
        {
        pdata_element p;

        p.tag = thrust::get<0>(t);
        p.pos = thrust::get<1>(t);
        p.vel = thrust::get<2>(t);
        p.accel = thrust::get<3>(t);
        p.charge = thrust::get<4>(t);
        p.diameter = thrust::get<5>(t);
        p.image = thrust::get<6>(t);
        p.body = thrust::get<7>(t);
        p.orientation = thrust::get<8>(t);

        return p;
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
                    unsigned int max_n_out,
                    cached_allocator& alloc)
    {
    // wrap device arrays into thrust ptr
    thrust::device_ptr<const Scalar4> pos_ptr(d_pos);
    thrust::device_ptr<const Scalar4> vel_ptr(d_vel);
    thrust::device_ptr<const Scalar3> accel_ptr(d_accel);
    thrust::device_ptr<const Scalar> charge_ptr(d_charge);
    thrust::device_ptr<const Scalar> diameter_ptr(d_diameter);
    thrust::device_ptr<const int3> image_ptr(d_image);
    thrust::device_ptr<const unsigned int> body_ptr(d_body);
    thrust::device_ptr<const Scalar4> orientation_ptr(d_orientation);
    thrust::device_ptr<const unsigned int> tag_ptr(d_tag);

    // wrap output device arrays into thrust ptr
    thrust::device_ptr<Scalar4> pos_alt_ptr(d_pos_alt);
    thrust::device_ptr<Scalar4> vel_alt_ptr(d_vel_alt);
    thrust::device_ptr<Scalar3> accel_alt_ptr(d_accel_alt);
    thrust::device_ptr<Scalar> charge_alt_ptr(d_charge_alt);
    thrust::device_ptr<Scalar> diameter_alt_ptr(d_diameter_alt);
    thrust::device_ptr<int3> image_alt_ptr(d_image_alt);
    thrust::device_ptr<unsigned int> body_alt_ptr(d_body_alt);
    thrust::device_ptr<Scalar4> orientation_alt_ptr(d_orientation_alt);
    thrust::device_ptr<unsigned int> tag_alt_ptr(d_tag_alt);

    // wrap output array
    thrust::device_ptr<pdata_element> out_ptr(d_out);

    // wrap reverse-lookup table
    thrust::device_ptr<unsigned int> rtag_ptr(d_rtag);

    // Construct zip iterator for input
    pdata_zip_gpu_const pdata_begin(
       thrust::make_tuple(
            tag_ptr,
            pos_ptr,
            vel_ptr,
            accel_ptr,
            charge_ptr,
            diameter_ptr,
            image_ptr,
            body_ptr,
            orientation_ptr
            )
        );

    // Construct zip iterator for output
    pdata_zip_gpu pdata_alt_begin(
       thrust::make_tuple(
            tag_alt_ptr,
            pos_alt_ptr,
            vel_alt_ptr,
            accel_alt_ptr,
            charge_alt_ptr,
            diameter_alt_ptr,
            image_alt_ptr,
            body_alt_ptr,
            orientation_alt_ptr
            )
        );

    // set up transform iterator to compact particle data into records
    typedef thrust::transform_iterator<to_pdata_element_gpu, pdata_zip_gpu_const > transform_it;
    transform_it in_transform(pdata_begin);

    // Combine two input streams
    thrust::zip_iterator<thrust::tuple<pdata_zip_gpu_const, transform_it> >
        in(thrust::make_tuple(pdata_begin, in_transform));

    // Output stream 1
    typedef thrust::zip_iterator<thrust::tuple< thrust::discard_iterator< >, thrust::device_ptr<pdata_element> > >
        out_it_1;
    out_it_1 out_1(thrust::make_tuple(thrust::make_discard_iterator(), out_ptr));

    // Output stream 2
    typedef thrust::zip_iterator<thrust::tuple< pdata_zip_gpu, thrust::discard_iterator<> > > out_it_2;
    out_it_2 out_2(thrust::make_tuple(pdata_alt_begin, thrust::make_discard_iterator()));

    // Clip output stream 1
    clipped_range<out_it_1> clip(out_1, out_1 + max_n_out);

    // partition input into two outputs (local particles and removed particles)
    thrust::pair<clipped_range<out_it_1>::iterator, out_it_2> res =
        thrust::stable_partition_copy(thrust::cuda::par(alloc), in, in+N, clip.begin(),
        out_2, combined_tuple_select_rtag_gpu(rtag_ptr,NOT_LOCAL));

    // return elements written to output stream
    return res.first - clip.begin();
    }

//! A tuple combining tag and a pdata element
typedef thrust::tuple<
    unsigned int,
    const pdata_element
    > idx_pdata_element_gpu;

void gpu_pdata_update_rtags(
    const unsigned int *d_tag,
    unsigned int *d_rtag,
    const unsigned int N,
    cached_allocator& alloc)
    {
    thrust::device_ptr<const unsigned int> tag_ptr(d_tag);
    thrust::device_ptr<unsigned int> rtag_ptr(d_rtag);

    // update rtags
    thrust::counting_iterator<unsigned int> idx(0);
    thrust::scatter(thrust::cuda::par(alloc), idx, idx + N, tag_ptr, rtag_ptr);
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
            p.orientation
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
            orientation_ptr
            )
        );
    pdata_zip_gpu pdata_end = pdata_begin + old_nparticles;

    // wrap reverse-lookup table
    thrust::device_ptr<unsigned int> rtag_ptr(d_rtag);

    typedef thrust::counting_iterator<unsigned int> count_it;

    // add new particles at the end, writing rtags at the same time
    thrust::transform(thrust::cuda::par(alloc),in_ptr, in_ptr + num_add_ptls, pdata_end, to_pdata_tuple_gpu());

    // update rtags
    thrust::counting_iterator<unsigned int> idx(old_nparticles);
    thrust::scatter(thrust::cuda::par(alloc), idx, idx + num_add_ptls, tag_ptr+old_nparticles, rtag_ptr);
    }

#endif // ENABLE_MPI
