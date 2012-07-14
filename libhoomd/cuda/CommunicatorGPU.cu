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

//! Apply (global) periodic boundary conditions to a ghost particle
struct wrap_ghost_particle
    {
    const Scalar3 L;              //!< Lengths of global simulation box
    const unsigned int dir;       //!< Current direction of ghost exchange
    bool is_at_boundary[6];       //!< Flags to indicate whether this box share a boundary with the global box

    //! Constructor
    /*! \param _global_box Dimensions of global simulation box
     *! \param _dir
     */
    wrap_ghost_particle(const BoxDim _global_box, const unsigned int _dir, const bool _is_at_boundary[] )
        : L(_global_box.getL()), dir(_dir)
        {
        for (unsigned int i = 0; i < 6; i++)
            is_at_boundary[i] = _is_at_boundary[i];
        }

    //! Apply periodic boundary conditions
    /*! \param postype Position and type  to apply boundary conditions to
     * \return the Position and type with boundary conditions applied
     */
    __host__ __device__ float4 operator()(float4 postype)
        {
        // wrap particles received across a global boundary back into global box
        if (dir==0 && is_at_boundary[1])
            postype.x -= L.x;
        else if (dir==1 && is_at_boundary[0])
            postype.x += L.x;
        else if (dir==2 && is_at_boundary[3])
            postype.y -= L.y;
        else if (dir==3 && is_at_boundary[2])
            postype.y += L.y;
        else if (dir==4 && is_at_boundary[5])
            postype.z -= L.z;
        else if (dir==5 && is_at_boundary[4])
            postype.z += L.z;

        return postype;
        }
     };

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

//! Select ghost particles for sending in one direction
struct select_particle_ghost
    {
    const unsigned int dir; //!< Current direction

    //! Constructor
    /*! \param _dir Direction of the neighboring domain
     */
    select_particle_ghost(unsigned int _dir)
        : dir(_dir)
        {
        }

    //! Select particles for sending
    /*! \param plan Particle exchange plan
        \returns true if particle is selected for sending
     */
    __host__ __device__ bool operator()(const unsigned char plan)
        {
        return (plan & (1 << dir));
        }
     };


//! Structure to pack a particle data element into
struct __align__(128) pdata_element_gpu
    {
    float4 pos;               //!< Position
    float4 vel;               //!< Velocity
    float3 accel;             //!< Acceleration
    float charge;             //!< Charge
    float diameter;           //!< Diameter
    int3 image;               //!< Image
    unsigned int body;        //!< Body id
    float4 orientation;       //!< Orientation
    unsigned int global_tag;  //!< global tag
    };

//! Get the size of a \c pdata_element_gpu
/*! The CUDA compiler aligns structure members differently than the C++ compiler. This function is used
    to return the actual size as returned by the CUDA compiler.

    \returns the size of a pdata_element_gpu (in bytes)
 */
unsigned int gpu_pdata_element_size()
    {
    return sizeof(pdata_element_gpu);
    }

//! Define a thrust tuple for a particle data element
typedef thrust::tuple<float4,
                      float4,
                      float3,
                      float,
                      float,
                      int3,
                      unsigned int,
                      float4,
                      unsigned int> pdata_tuple_gpu;

//! Select particles to be sent in a specified direction
struct select_particle_migrate_gpu : public thrust::unary_function<const pdata_tuple_gpu&, bool>
    {
    const unsigned int dir; //!< Direction to send particles to
    const float4 *d_pos;    //!< Device array of particle positions
    float3 lo;              //!< Lower box boundary
    float3 hi;              //!< Upper box boundary

    //! Constructor
    /*!
     */
    select_particle_migrate_gpu(const BoxDim _box,
                            const unsigned int _dir,
                            const float4 *_d_pos)
        : dir(_dir), d_pos(_d_pos)
        {
        lo = _box.getLo();
        hi = _box.getHi();
        }

    //! Select a particle
    /*! t particle data to consider for sending
     * \return true if particle stays in the box
     */
    __host__ __device__ bool operator()(const unsigned int& idx)
        {
        const float4& pos = d_pos[idx];
        // we return true if the particle stays in our box,
        // false otherwise
        return !((dir == 0 && pos.x >= hi.x)||  // send east
                (dir == 1 && pos.x < lo.x)  ||  // send west
                (dir == 2 && pos.y >= hi.y) ||  // send north
                (dir == 3 && pos.y < lo.y)  ||  // send south
                (dir == 4 && pos.z >= hi.z) ||  // send up
                (dir == 5 && pos.z < lo.z));    // send down
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
    __host__ __device__ pdata_element_gpu operator()(pdata_element_gpu el)
        {
        float4& postype = el.pos;
        int3& image = el.image;

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
        return el;
        }

     };


//! Pack a particle data tuple
struct pack_pdata : public thrust::unary_function<pdata_tuple_gpu, pdata_element_gpu>
    {
    //! Transform operator
    /*! \param t Particle data tuple to pack
     * \return Packed particle data element
     */
    __host__ __device__ pdata_element_gpu operator()(const pdata_tuple_gpu& t)
        {
        pdata_element_gpu el;
        el.pos  = thrust::get<0>(t);
        el.vel  = thrust::get<1>(t);
        el.accel= thrust::get<2>(t);
        el.charge = thrust::get<3>(t);
        el.diameter = thrust::get<4>(t);
        el.image = thrust::get<5>(t);
        el.body = thrust::get<6>(t);
        el.orientation = thrust::get<7>(t);
        el.global_tag = thrust::get<8>(t);
        return el;
        }
    };

//! Unpack a particle data element
struct unpack_pdata : public thrust::unary_function<pdata_element_gpu, pdata_tuple_gpu>
    {
    //! Transform operator
    /*! \param el Particle data element to unpack
     */
    __host__ __device__ pdata_tuple_gpu operator()(const pdata_element_gpu & el)
        {
        return pdata_tuple_gpu(el.pos,
                           el.vel,
                           el.accel,
                           el.charge,
                           el.diameter,
                           el.image,
                           el.body,
                           el.orientation,
                           el.global_tag);
        }
    };

thrust::device_vector<unsigned int> *keys;       //!< Temporary vector of sort keys

void gpu_allocate_tmp_storage()
    {
    keys = new thrust::device_vector<unsigned int>;
    }

void gpu_deallocate_tmp_storage()
    {
    delete keys;
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
                                                         const float4 *d_pos,
                                                         const unsigned int *d_rtag,
                                                         const BoxDim box,
                                                         const unsigned int N,
                                                         const unsigned int n_bonds)
    {
    unsigned int bond_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (bond_idx >= n_bonds)
        return;

    uint2 bond = btable[bond_idx];

    unsigned int tag1 = bond.x;
    unsigned int tag2 = bond.y;
    unsigned int idx1 = d_rtag[tag1];
    unsigned int idx2 = d_rtag[tag2];

    float3 L2 = box.getL() / 2.0f;
    float3 lo = box.getLo();

    if ((idx1 >= N) && (idx2 < N))
        {
        // send particle with index idx2 to neighboring domains
        float4 pos = d_pos[idx2];
        unsigned char p = plan[idx2];
        p |= (pos.x > lo.x + L2.x) ? send_east : send_west;
        p |= (pos.y > lo.y + L2.y) ? send_north : send_south;
        p |= (pos.z > lo.z + L2.z) ? send_up : send_down;

        // Multiple threads may update the plan simultaneously, but this should
        // be safe, since they store the same result
        plan[idx2] = p;
        }
    else if ((idx1 < N) && (idx2 >= N))
        {
        // send particle with index idx1 to neighboring domains
        float4 pos = d_pos[idx1];
        unsigned char p = plan[idx1];
        p |= (pos.x > lo.x + L2.x) ? send_east : send_west;
        p |= (pos.y > lo.y + L2.y) ? send_north : send_south;
        p |= (pos.z > lo.z + L2.z) ? send_up : send_down;

        // Multiple threads may update the plan simultaneously, but this should
        // be safe, since they store the same result
        plan[idx1] = p;
        }
    }

//! Mark particles in incomplete bonds for sending
/* \param d_btable GPU bond table
 * \param d_plan Plan array
 * \param d_pos Array of particle positions
 * \param d_rtag Array of global reverse-lookup tags
 * \param box The local box dimensions
 * \param N number of (local) particles
 * \param n_bonds Total number of bonds in bond table
 */
void gpu_mark_particles_in_incomplete_bonds(const uint2 *d_btable,
                                          unsigned char *d_plan,
                                          const float4 *d_pos,
                                          const unsigned int *d_rtag,
                                          const BoxDim& box,
                                          const unsigned int N,
                                          const unsigned int n_bonds)
    {
    assert(d_gpu_btable);
    assert(d_plan);
    assert(N>0);

    unsigned int block_size = 512;
    gpu_mark_particles_in_incomplete_bonds_kernel<<<n_bonds/block_size + 1, block_size>>>(d_btable,
                                                                                    d_plan,
                                                                                    d_pos,
                                                                                    d_rtag,
                                                                                    box,
                                                                                    N,
                                                                                    n_bonds);
    }

//! Helper kernel to reorder particle data, step one
__global__ void gpu_reorder_pdata_step_one_kernel(const float4 *d_pos,
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
                                         unsigned int *keys,
                                         unsigned int N)
    {
    unsigned int idx = blockIdx.x*blockDim.x + threadIdx.x;

    if (idx >= N)
        return;

    unsigned int key = keys[idx];
    d_pos_tmp[idx] = d_pos[key];
    d_vel_tmp[idx] = d_vel[key];
    d_accel_tmp[idx] = d_accel[key];
    d_image_tmp[idx] = d_image[key];
    d_charge_tmp[idx] = d_charge[key];
    d_diameter_tmp[idx] = d_diameter[key];
    d_body_tmp[idx] = d_body[key];
    d_orientation_tmp[idx] = d_orientation[key];
    d_tag_tmp[idx] = d_tag[key];
    }

/*! Reorder the particles according to a migration criterium
 *  Particles that remain in the simulation box come first, followed by the particles that are sent in the
 *  specified direction
 *
 *  \param N Number of particles in local simulation box
 *  \param n_send_ptls Number of particles that are sent (return value)
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
    if (keys->size() < N)
        {
        unsigned int cur_size = keys->size() ? keys->size() : N;
        while (cur_size < N) cur_size *= 2;
        keys->resize(cur_size);
        }

    thrust::counting_iterator<unsigned int> count(0);
    thrust::copy(count, count + N, keys->begin());

    thrust::device_vector<unsigned int>::iterator keys_middle;

    keys_middle = thrust::stable_partition(keys->begin(),
                             keys->begin() + N,
                             select_particle_migrate_gpu(box, dir, d_pos));

    n_send_ptls = (keys->begin() + N) - keys_middle;

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
    thrust::gather(keys->begin(), keys->begin() + N, pos_ptr, pos_tmp_ptr);
    thrust::gather(keys->begin(), keys->begin() + N, vel_ptr, vel_tmp_ptr);
    thrust::gather(keys->begin(), keys->begin() + N, accel_ptr, accel_tmp_ptr);
    thrust::gather(keys->begin(), keys->begin() + N, image_ptr, image_tmp_ptr);
    thrust::gather(keys->begin(), keys->begin() + N, charge_ptr, charge_tmp_ptr);
    thrust::gather(keys->begin(), keys->begin() + N, diameter_ptr, diameter_tmp_ptr);
    thrust::gather(keys->begin(), keys->begin() + N, body_ptr, body_tmp_ptr);
    thrust::gather(keys->begin(), keys->begin() + N, orientation_ptr, orientation_tmp_ptr);
    thrust::gather(keys->begin(), keys->begin() + N, tag_ptr, tag_tmp_ptr);
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

//! Pack particle data into send buffer
/*! \param N number of particles to check for sending
   \param d_pos Array of particle positions
   \param d_vel Array of particle velocities
   \param d_accel Array of particle accelerations
   \param d_image Array of particle images
   \param d_charge Array of particle charges
   \param d_diameter Array of particle diameter
   \param d_body Array of particle body ids
   \param d_orientation Array of particle orientations
   \param d_tag Array of particle global tags
   \param d_send_buf Send buffer (has to be large enough, i.e. maxium size = number of local particles )
   \param d_send_buf_end Pointer to end of send buffer (return value)
*/
void gpu_migrate_pack_send_buffer(unsigned int N,
                           float4 *d_pos,
                           float4 *d_vel,
                           float3 *d_accel,
                           int3 *d_image,
                           float *d_charge,
                           float *d_diameter,
                           unsigned int *d_body,
                           float4  *d_orientation,
                           unsigned int *d_tag,
                           char *d_send_buf,
                           char *&d_send_buf_end)
    {
    thrust::device_ptr<float4> pos_ptr(d_pos);
    thrust::device_ptr<float4> vel_ptr(d_vel);
    thrust::device_ptr<float3> accel_ptr(d_accel);
    thrust::device_ptr<int3> image_ptr(d_image);
    thrust::device_ptr<float> charge_ptr(d_charge);
    thrust::device_ptr<float> diameter_ptr(d_diameter);
    thrust::device_ptr<unsigned int> body_ptr(d_body);
    thrust::device_ptr<float4> orientation_ptr(d_orientation);
    thrust::device_ptr<unsigned int> tag_ptr(d_tag);
    thrust::device_ptr<pdata_element_gpu> send_buf_ptr((pdata_element_gpu *) d_send_buf);

    // we perform operations on the whole particle data
    typedef thrust::tuple<thrust::device_ptr<float4>,
                          thrust::device_ptr<float4>,
                          thrust::device_ptr<float3>,
                          thrust::device_ptr<float>,
                          thrust::device_ptr<float>,
                          thrust::device_ptr<int3>,
                          thrust::device_ptr<unsigned int>,
                          thrust::device_ptr<float4>,
                          thrust::device_ptr<unsigned int> > pdata_iterator_tuple;

    thrust::zip_iterator<pdata_iterator_tuple> pdata_first = thrust::make_tuple( pos_ptr,
                                               vel_ptr,
                                               accel_ptr,
                                               charge_ptr,
                                               diameter_ptr,
                                               image_ptr,
                                               body_ptr,
                                               orientation_ptr,
                                               tag_ptr);
    thrust::zip_iterator<pdata_iterator_tuple> pdata_end = pdata_first + N;


    // pack the particles into the send buffer
    thrust::device_ptr<pdata_element_gpu> send_buf_end_ptr =
        thrust::copy(thrust::make_transform_iterator(pdata_first, pack_pdata()),
                     thrust::make_transform_iterator(pdata_end, pack_pdata()),
                     send_buf_ptr);

    d_send_buf_end = (char *) thrust::raw_pointer_cast(send_buf_end_ptr);
    }

//! Wrap received particles across global box boundaries
/*! \param d_recv_buf Received particle data
 * \param d_recv_buf_end End of received particle data
 * \param n_recv_ptl Number of received particles (return value)
 * \param global_box Dimensions of global box
 * \param dir Direction along which particles where received
 * \param is_at_boundary Array of per-direction flags to indicate whether this box lies at a global boundary
 */
void gpu_migrate_wrap_received_particles(char *d_recv_buf,
                                 char *d_recv_buf_end,
                                 unsigned int &n_recv_ptl,
                                 const BoxDim& global_box,
                                 const unsigned int dir,
                                 const bool is_at_boundary[])
    {
    thrust::device_ptr<pdata_element_gpu> recv_buf_ptr((pdata_element_gpu *) d_recv_buf);
    thrust::device_ptr<pdata_element_gpu> recv_buf_end_ptr((pdata_element_gpu *) d_recv_buf_end);
    thrust::transform(recv_buf_ptr,
                      recv_buf_end_ptr,
                      recv_buf_ptr,
                      wrap_received_particle(global_box, dir, is_at_boundary));
    n_recv_ptl = recv_buf_end_ptr - recv_buf_ptr;
    }

//! Add received particles to local box 
/*! \param d_recv_buf Buffer of received particle data
 * \param d_recv_buf_end Pointer to end of receive buffer
 * \param d_pos Array to store particle positions
 * \param d_vel Array to store particle velocities
 * \param d_accel Array to store particle accelerations
 * \param d_image Array to store particle images
 * \param d_charge Array to store particle charges
 * \param d_diameter Array to store particle diameters
 * \param d_body Array to store particle body ids
 * \param d_orientation Array to store particle body orientations
 * \param d_tag Array to store particle global tags
 */
void gpu_migrate_add_particles(  char *d_recv_buf,
                                 char *d_recv_buf_end,
                                 float4 *d_pos,
                                 float4 *d_vel,
                                 float3 *d_accel,
                                 int3 *d_image,
                                 float *d_charge,
                                 float *d_diameter,
                                 unsigned int *d_body,
                                 float4  *d_orientation,
                                 unsigned int *d_tag)
    {
    thrust::device_ptr<pdata_element_gpu> recv_buf_ptr((pdata_element_gpu *) d_recv_buf);
    thrust::device_ptr<pdata_element_gpu> recv_buf_end_ptr((pdata_element_gpu *) d_recv_buf_end);
    thrust::device_ptr<float4> pos_ptr(d_pos);
    thrust::device_ptr<float4> vel_ptr(d_vel);
    thrust::device_ptr<float3> accel_ptr(d_accel);
    thrust::device_ptr<int3> image_ptr(d_image);
    thrust::device_ptr<float> charge_ptr(d_charge);
    thrust::device_ptr<float> diameter_ptr(d_diameter);
    thrust::device_ptr<unsigned int> body_ptr(d_body);
    thrust::device_ptr<float4> orientation_ptr(d_orientation);
    thrust::device_ptr<unsigned int> tag_ptr(d_tag);

    thrust::copy(thrust::make_transform_iterator(recv_buf_ptr, unpack_pdata()),
                    thrust::make_transform_iterator(recv_buf_end_ptr, unpack_pdata()),
                    make_zip_iterator( thrust::make_tuple( pos_ptr,
                                               vel_ptr,
                                               accel_ptr,
                                               charge_ptr,
                                               diameter_ptr,
                                               image_ptr,
                                               body_ptr,
                                               orientation_ptr,
                                               tag_ptr) )) -
                    make_zip_iterator( thrust::make_tuple( pos_ptr,
                                               vel_ptr,
                                               accel_ptr,
                                               charge_ptr,
                                               diameter_ptr,
                                               image_ptr,
                                               body_ptr,
                                               orientation_ptr,
                                               tag_ptr) );
    }

//! Wrap received ghost particles across global box
/*! \param dir Direction along which particles were received
 * \param n Number of particles to apply periodic boundary conditions to
 * \param d_pos Array of particle positions to apply periodic boundary conditions to
 * \param global_box Dimensions of global simulation box
 * \param is_at_boundary Array of flags to indicate whether this box is a boundary box
 */
void gpu_wrap_ghost_particles(unsigned int dir,
                              unsigned int n,
                              float4 *d_pos,
                              const BoxDim& global_box,
                              const bool is_at_boundary[])
    {
    thrust::device_ptr<float4> pos_ptr(d_pos);
    thrust::transform(pos_ptr, pos_ptr +n, pos_ptr, wrap_ghost_particle(global_box, dir, is_at_boundary ));
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

//! Construct a list of particle tags to send as ghost particles
/*! \param N number of particles to check
 * \param dir Direction in which ghost particles are sent
 * \param d_plan Array of particle exchange plans
 * \param d_global_tag Array of particle global tags
 * \param d_copy_ghosts Array to be fillled x with global tags of particles that are to be send as ghosts
 * \param n_copy_ghosts Number of local particles that are sent in the given direction as ghosts (return value)
 */
void gpu_make_exchange_ghost_list(unsigned int N,
                                  unsigned int dir,
                                  unsigned char *d_plan,
                                  unsigned int *d_global_tag,
                                  unsigned int* d_copy_ghosts,
                                  unsigned int &n_copy_ghosts)
    {
    thrust::device_ptr<unsigned char> plan_ptr(d_plan);
    thrust::device_ptr<unsigned int> global_tag_ptr(d_global_tag);
    thrust::device_ptr<unsigned int> copy_ghosts_ptr(d_copy_ghosts);

    thrust::device_ptr<unsigned int> copy_ghosts_end_ptr;

    copy_ghosts_end_ptr = thrust::copy_if(global_tag_ptr,
                                          global_tag_ptr+N,
                                          plan_ptr,
                                          copy_ghosts_ptr,
                                          select_particle_ghost(dir));

    n_copy_ghosts =  copy_ghosts_end_ptr - copy_ghosts_ptr;
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
 */
void gpu_exchange_ghosts(unsigned int nghost,
                         unsigned int *d_copy_ghosts,
                         unsigned int *d_rtag,
                         float4 *d_pos,
                         float4 *d_pos_copybuf,
                         float *d_charge,
                         float *d_charge_copybuf,
                         float *d_diameter,
                         float *d_diameter_copybuf,
                         unsigned char *d_plan,
                         unsigned char *d_plan_copybuf)
    {
    thrust::device_ptr<unsigned int> copy_ghosts_ptr(d_copy_ghosts);
    thrust::device_ptr<unsigned int> rtag_ptr(d_rtag);
    thrust::device_ptr<float4> pos_ptr(d_pos);
    thrust::device_ptr<float4> pos_copybuf_ptr(d_pos_copybuf);
    thrust::device_ptr<float> charge_ptr(d_charge);
    thrust::device_ptr<float> charge_copybuf_ptr(d_charge_copybuf);
    thrust::device_ptr<float> diameter_ptr(d_diameter);
    thrust::device_ptr<float> diameter_copybuf_ptr(d_diameter_copybuf);
    thrust::device_ptr<unsigned char> plan_ptr(d_plan);
    thrust::device_ptr<unsigned char> plan_copybuf_ptr(d_plan_copybuf);

    permutation_iterator<device_ptr<unsigned int>, device_ptr<unsigned int> > ghost_rtag(rtag_ptr, copy_ghosts_ptr);
    gather(ghost_rtag, ghost_rtag + nghost, pos_ptr, pos_copybuf_ptr);
    gather(ghost_rtag, ghost_rtag + nghost, charge_ptr, charge_copybuf_ptr);
    gather(ghost_rtag, ghost_rtag + nghost, diameter_ptr, diameter_copybuf_ptr);
    gather(ghost_rtag, ghost_rtag + nghost, plan_ptr, plan_copybuf_ptr);
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

//! Copy ghost particle positions into send buffer
/*! \param nghost Number of ghost particles to copy
 * \param d_pos Array of particle positions
 * \param d_copy_ghosts Global particle tags of particles to copy
 * \param d_pos_copybuf Send buffer of ghost particle positions
 * \param d_rtag Global tag <-> local particle index reverse lookup array
 */
void gpu_copy_ghosts(unsigned int nghost,
                     float4 *d_pos,
                     unsigned int *d_copy_ghosts,
                     float4 *d_pos_copybuf,
                     unsigned int *d_rtag)
    {
    thrust::device_ptr<float4> pos_ptr(d_pos);
    thrust::device_ptr<unsigned int> rtag_ptr(d_rtag);
    thrust::device_ptr<unsigned int> copy_ghosts_ptr(d_copy_ghosts);
    thrust::device_ptr<float4> copybuf_ptr(d_pos_copybuf);

    permutation_iterator<device_ptr<unsigned int>, device_ptr<unsigned int> > ghost_rtag(rtag_ptr, copy_ghosts_ptr);
    gather(ghost_rtag, ghost_rtag + nghost, pos_ptr, copybuf_ptr);

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
