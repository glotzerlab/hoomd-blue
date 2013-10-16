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

/*! \file ParticleData.cu
    \brief ImplementsGPU kernel code and data structure functions used by ParticleData
*/

#ifdef ENABLE_MPI

#include <thrust/iterator/zip_iterator.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/functional.h>
#include <thrust/device_ptr.h>
#include <thrust/copy.h>
#include <thrust/count.h>
#include <thrust/remove.h>

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

//! A predicate to select particles by rtag
struct pdata_element_select_gpu : public thrust::unary_function<pdata_element, bool>
    {
    //! Constructor
    pdata_element_select_gpu(const unsigned int *_d_rtag, unsigned int _compare)
        : d_rtag(_d_rtag), compare(_compare)
        { }

    //! Returns true if the remove flag is set for a particle
    __device__ bool operator() (pdata_element const p) const
        {
        return d_rtag[p.tag] == compare;
        }

    const unsigned int *d_rtag; //!< The reverse-lookup tag array
    const unsigned int compare; //!< rtag value to compare to
    };

//! A predicate to select rtags by value
struct rtag_select_gpu
    {
    //! Constructor
    rtag_select_gpu(const unsigned int _compare)
        :  compare(_compare)
        { }

    //! Returns true if the remove flag is set for a particle
    __device__ bool operator() (const unsigned int rtag) const
        {
        return rtag == compare;
        }

    const unsigned int compare; //!< rtag value to compare to
    };

//! A predicate to select pdata tuples by rtag
struct pdata_tuple_select_rtag_gpu
    {
    //! Constructor
    pdata_tuple_select_rtag_gpu(const unsigned int *_d_rtag, const unsigned int _compare)
        :  d_rtag(_d_rtag), compare(_compare)
        { }

    //! Returns true if the remove flag is set for a particle
    __device__ bool operator() (const pdata_tuple_gpu& t) const
        {
        unsigned int tag = thrust::get<0>(t);
        return d_rtag[tag] == compare;
        }

    const unsigned int *d_rtag; //!< Reverse-lookup table
    const unsigned int compare; //!< rtag value to compare to
    };


//! A converter from pdata_element to a tuple of pdata entries
struct to_pdata_tuple_gpu : public thrust::unary_function<const pdata_element, pdata_tuple_gpu>
    {
    __device__ const pdata_tuple_gpu operator() (const pdata_element p)
        {
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
    \param d_vel Device iarray of particle velocities
    \param d_accel Device array of particle accelerations
    \param d_charge Device array of particle charges
    \param d_diameter Device array of particle diameters
    \param d_image Device array of particle images
    \param d_body Device array of particle body tags
    \param d_orientation Device array of particle orientations
    \param d_tag Device array of particle tags
    \param d_rtag Device array for reverse-lookup table
    \param d_out Output array for packed particle ata
 */
void gpu_pdata_pack(const unsigned int N,
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
                    pdata_element *d_out,
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

    // wrap output array
    thrust::device_ptr<pdata_element> out_ptr(d_out);

    // Construct zip iterator
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

    // set up transform iterator to compact particle data into records
    thrust::transform_iterator<to_pdata_element_gpu, pdata_zip_gpu_const> pdata_transform(pdata_begin);

    // compact selected particle elements into output array
    thrust::copy_if(thrust::cuda::par(alloc),
        pdata_transform, pdata_transform+N, out_ptr, pdata_element_select_gpu(d_rtag,STAGED));

    // wrap rtag array
    thrust::device_ptr<unsigned int> rtag_ptr(d_rtag);

    // set up permutation iterator to point into rtags
    thrust::permutation_iterator<
        thrust::device_ptr<unsigned int>, thrust::device_ptr<const unsigned int> > rtag_prm(rtag_ptr, tag_ptr);

    // set all STAGED tags to NOT_LOCAL
    thrust::replace_if(thrust::cuda::par(alloc), rtag_prm, rtag_prm + N, rtag_select_gpu(STAGED), NOT_LOCAL);
    }

/*! \param N Number of local particles
    \param d_tag Device array of particle tags
    \param d_rtag Device array of reverse-lookup tags
    \param compare rtag value to compare to
 */
unsigned int gpu_pdata_count_rtag_equals(const unsigned int N,
    const unsigned int *d_tag,
    const unsigned int *d_rtag,
    const unsigned int compare,
    cached_allocator& alloc)
    {
    thrust::device_ptr<const unsigned int> tag_ptr(d_tag);
    thrust::device_ptr<const unsigned int> rtag_ptr(d_rtag);

    // set up permutation iterator to point into rtags
    thrust::permutation_iterator<
        thrust::device_ptr<const unsigned int>, thrust::device_ptr<const unsigned int> > rtag_prm(rtag_ptr, tag_ptr);

    return thrust::count_if(thrust::cuda::par(alloc), rtag_prm, rtag_prm + N, rtag_select_gpu(compare));
    }

/*! \param old_nparticles old local particle count
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

    \returns number of particles removed
*/
unsigned int gpu_pdata_remove_particles(const unsigned int old_nparticles,
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

    // erase all elements for which rtag == NOT_LOCAL
    // the array remains contiguous
    pdata_zip_gpu new_pdata_end;
    new_pdata_end = thrust::remove_if(thrust::cuda::par(alloc),
        pdata_begin, pdata_end, pdata_tuple_select_rtag_gpu(d_rtag, NOT_LOCAL));

    return pdata_end - new_pdata_end;
    }

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

    // add new particles at the end
    thrust::transform(thrust::cuda::par(alloc), in_ptr, in_ptr + num_add_ptls, pdata_end, to_pdata_tuple_gpu());

    unsigned int new_n_particles = old_nparticles + num_add_ptls;

    // recompute rtags
    thrust::counting_iterator<unsigned int> idx(0);
    thrust::scatter(thrust::cuda::par(alloc), idx, idx+new_n_particles, tag_ptr, rtag_ptr);
    }
#endif // ENABLE_MPI
