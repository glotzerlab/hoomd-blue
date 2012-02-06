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

/*! \file Communicator.cu
    \brief Implementation of communication algorithms on the GPU
*/

#ifdef ENABLE_MPI
#include "CommunicatorGPU.cuh"

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

using namespace thrust;

//! Apply (global) periodic boundary conditions to a ghost particle
struct wrap_ghost_particle
    {
    const gpu_boxsize box;  //!< Dimensions of global simulation box
    const float rghost;     //!< Width of ghost layer
    const unsigned int dir; //!< Direction along which particle was received

    //! Constructor
    /*! \param _box Dimensions of global simulation box
     * \param _rghost Width of ghost layer
     * \param _dir Direction along which particle was received
     */
    wrap_ghost_particle(gpu_boxsize _box, float _rghost, unsigned int _dir)
        : box(_box), rghost(_rghost), dir(_dir)
        {
        }

    //! Apply peridoic boundary conditions
    /*! \param pos position element to apply boundary conditions to
     * \return the position element with boundary conditions applied
     */
    __host__ __device__ float4 operator()(const float4 &pos)
        {
            // wrap particles received across a global boundary back into global box
            float4 pos2 = pos;
            if (dir==0 && pos2.x >= box.xhi - rghost)
                pos2.x -= box.xhi - box.xlo;
            else if (dir==1 && pos2.x < box.xlo + rghost)
                pos2.x += box.xhi - box.xlo;
            else if (dir==2 && pos2.y >= box.yhi - rghost)
                pos2.y -= box.yhi - box.ylo;
            else if (dir==3 && pos2.y < box.ylo + rghost)
                pos2.y += box.yhi - box.ylo;
            else if (dir==4 && pos2.z >= box.zhi - rghost)
                pos2.z -= box.zhi - box.zlo;
            else if (dir==5 && pos2.z < box.zlo + rghost)
                pos2.z += box.zhi - box.zlo;
            return pos2;
        }
     };

//! Select local particles that within a boundary layer of the neighboring domain in a given direction
struct select_particle_ghost
    {
    const gpu_boxsize box;    //!< Local box dimensions
    const float r_ghost;      //!< Width of boundary layer
    const unsigned int dir;   //!< Direction of the neighboring domain

    //! Constructor
    /*! \param _box Local box dimensions
     * \param _r_ghost Width of boundary layer
     * \param _dir Direction of the neighboring domain
     */
    select_particle_ghost(const gpu_boxsize _box, float _r_ghost, unsigned int _dir)
        : box(_box), r_ghost(_r_ghost), dir(_dir)
        {
        }

    //! Apply selection criterium
    /*! \param pos the position of the particle to apply the criterium to
        \returns true if particle lies within the boundary layer
     */
    __host__ __device__ bool operator()(const float4 &pos)
        {
        return ((dir==0 && (pos.x >= box.xhi - r_ghost)) ||                  // send east
            (dir==1 && (pos.x < box.xlo + r_ghost) && (pos.x >= box.xlo)) || // send west
            (dir==2 && (pos.y >= box.yhi - r_ghost)) ||                      // send north
            (dir==3 && (pos.y < box.ylo + r_ghost) && (pos.y >= box.ylo)) || // send south
            (dir==4 && (pos.z >= box.zhi - r_ghost)) ||                      // send up
            (dir==5 && (pos.z < box.zlo + r_ghost) && (pos.z >= box.zlo)));  // send down

        }
     };

//! Structure to pack a particle data element into
struct pdata_element
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

//! Get the size of a \c pdata_element
/*! The CUDA compiler aligns structure members differently than the C++ compiler. This function is used
    to return the actual size as returned by the CUDA compiler.

    \returns the size of a pdata_element (in bytes)
 */
unsigned int gpu_pdata_element_size()
    {
    return sizeof(pdata_element);
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
                      unsigned int> pdata_tuple;

//! Determine whether a particle is found inside the box boundaries along a given axis
struct select_particle_keep
    {
    const gpu_boxsize box;  //!< Local box dimensions
    const bool send_x;      //!< True if we have neighbors in the x direction
    const bool send_y;      //!< True if we have neighbors in the y direction
    const bool send_z;      //!< True if we have neighbors in the z direction

    //! Constructor
    /*! \param _box Dimensions of the local box
     * \param _send_x True if we have neighbors in the x direction
     * \param _send_y True if we have neighbors in the y direction
     * \param _send_z True if we have neighbors in the z direction
     */
    select_particle_keep(const gpu_boxsize _box, bool _send_x, bool _send_y, bool _send_z)
        : box(_box), send_x(_send_x), send_y(_send_y), send_z(_send_z)
        {
        }

    //! Determine whether we want to keep a particle (i.e. not consider it for sending)
    /*! \param t the input particle data tuple
     * \returns true if the particle is inside the boundaries on an axis along which we have neighboring domains
     */
    __host__ __device__ bool operator()(const pdata_tuple& t)
        {
        float4 pos = thrust::get<0>(t);
        return !((send_x && pos.x >= box.xhi) || // send east
                (send_x && pos.x < box.xlo)  ||  // send west
                (send_y && pos.y >= box.yhi) ||  // send north
                (send_y && pos.y < box.ylo)  ||  // send south
                (send_z && pos.z >= box.zhi) ||  // send up
                (send_z && pos.z < box.zlo));    // send down
        }
     };

//! Select particles to be sent in a specified direction
struct select_particle_migrate
    {
//    const gpu_boxsize box;     //!< Local box dimensions
    const float xlo;
    const float xhi;
    const float ylo;
    const float yhi;
    const float zlo;
    const float zhi;
    const unsigned int dir;    //!< Direction to send particles to

    //! Constructor
    /*! \param _box Dimensions of local box
        \param _dir Direction to send particles to
     */
    select_particle_migrate(const float _xlo, const float _xhi, const float _ylo, const float _yhi, const float _zlo, const float _zhi,
                            unsigned int &_dir)
        : xlo(_xlo), xhi(_xhi), ylo(_ylo), yhi(_yhi), zlo(_zlo), zhi(_zhi), dir(_dir)
        {
        }

    //! Select a particle
    /*! element particle data of the particle to consider for sending
     * \return true if the particle is selected
     */
    __host__ __device__ bool operator()(const pdata_element& element)
        {
        const float4& pos = element.pos;
        return ((dir==0 && pos.x >= xhi) ||  // send east
                (dir==1 && pos.x < xlo)  ||  // send west
                (dir==2 && pos.y >= yhi) ||  // send north
                (dir==3 && pos.y < ylo)  ||  // send south
                (dir==4 && pos.z >= zhi) ||  // send up
                (dir==5 && pos.z < zlo));    // send down

        }

     };

//! Wrap a received particle across global box boundaries
struct wrap_received_particle
    {
    const gpu_boxsize box;   //!< Dimensions of global simulation box
    const unsigned int dir;  //!< Direction along which the particle was received

    //! Constructor
    /*! \param _box Dimensions of global simulation box
        \param _dir Direciton along whic the particle was received
     */
    wrap_received_particle(const gpu_boxsize _box, unsigned int _dir)
        : box(_box), dir(_dir)
        {
        }

   //! Wrap particle across boundaries
   /*! \param el particle data element to transform
    * \return transformed particle data element
    */
    __host__ __device__ pdata_element operator()(const pdata_element & el)
        {
        pdata_element el2 = el;
        float4& pos = el2.pos;
        int3& image = el2.image;

        if (dir == 0 && pos.x >= box.xhi)
            {
            pos.x -= box.xhi - box.xlo;
            image.x++;
            }
        else if (dir == 1 && pos.x < box.xlo)
            {
            pos.x += box.xhi - box.xlo;
            image.x--;
            }

        if (dir == 2 && pos.y >= box.yhi)
            {
            pos.y -= box.yhi - box.ylo;
            image.y++;
            }
        else if (dir == 3 && pos.y < box.ylo)
            {
            pos.y += box.yhi - box.ylo;
            image.y--;
            }

        if (dir == 4 && pos.z >= box.zhi)
            {
            pos.z -= box.zhi - box.zlo;
            image.z++;
            }
        else if (dir == 5 && pos.z < box.zlo)
            {
            pos.z += box.zhi - box.zlo;
            image.z--;
            }
        return el2;
        }

     };


//! Determine whether a received particle is to be added to the local box
struct isInBox
    {
    const gpu_boxsize box;  //!< Local box dimensions

    //! Constructor
    /* \param _box Local box dimensions
     */
    isInBox(const gpu_boxsize _box)
        : box(_box)
        {
        }

    //! Determine whether particle is in local box
    /*! \param pos Position of the particle to check
     * \return true if position is in local box
     */
    __host__ __device__ bool check_ptl(const float4& pos)
        {
        return (box.xlo <= pos.x  && pos.x < box.xhi) &&
               (box.ylo <= pos.y  && pos.y < box.yhi) &&
               (box.zlo <= pos.z  && pos.z < box.zhi);
        }

    //! Determine whether particle is in local box
    /*! \param el the particle data element to apply the criterium to
     * \return true if the particle is to be added to the local particle data
     */
    __host__ __device__ bool operator()(const pdata_element & el)
        {
        return check_ptl(el.pos);
        }

    //! Determine whether particle is in local box
    /*! \param t the particle data tuple to apply the criterium to
     * \return true if the particle is to be added to the local particle data
     */
    __host__ __device__ bool operator()(const pdata_tuple & t)
        {
        return check_ptl(thrust::get<0>(t));
        }
     };

//! Pack a particle data tuple
struct pack_pdata : public thrust::unary_function<pdata_tuple, pdata_element>
    {
    //! Transform operator
    /*! \param t Particle data tuple to pack
     * \return Packed particle data element
     */
    __host__ __device__ pdata_element operator()(const pdata_tuple& t)
        {
        pdata_element el;
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
struct unpack_pdata : public thrust::unary_function<pdata_element, pdata_tuple>
    {
    //! Transform operator
    /*! \param el Particle data element to unpack
        \param Tuple of particle data fields
     */
    __host__ __device__ pdata_tuple operator()(const pdata_element & el)
        {
        return pdata_tuple(el.pos,
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

/*! Selects all particles which are no longer inside the local box boundaries and move them to the end of the particle data arrays.
   The number of particles that have been selected and moved to the end is returned.

   \post The particle data arrays are divided into two consecutive arrays:
   Local particles, and particles that have left the simulation box.
   The relative order of the local particles is preserved. The overall size of the particle
   data arrays is unchanged.

   \param N Number of particles in local simulation box
   \param n_delete_ptls Number of particles that have been moved to the end (return value)
   \param d_pos Array of particle positions
   \param d_vel Array of particle velocities
   \param d_accel Array of particle accelerations
   \param d_image Array of particle images
   \param d_charge Array of particle charges
   \param d_diameter Array of particle diameter
   \param d_body Array of particle body ids
   \param d_orientation Array of particle orientations
   \param d_tag Array of particle global tags
   \param box Dimensions of local simulation box
   \param send_x Flag to indicate if we have neighbor domains in the x direction
   \param send_x Flag to indicate if we have neighbor domains in the x direction
   \param send_x Flag to indicate if we have neighbor domains in the y direction
*/
void gpu_migrate_compact_pdata(unsigned int N,
                        unsigned int &n_delete_ptls,
                        float4 *d_pos,
                        float4 *d_vel,
                        float3 *d_accel,
                        int3 *d_image,
                        float *d_charge,
                        float *d_diameter,
                        unsigned int *d_body,
                        float4  *d_orientation,
                        unsigned int *d_tag,
                        gpu_boxsize box,
                        bool send_x,
                        bool send_y,
                        bool send_z)
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

    // move all particles we are going to send to the end
    // we use a stable partition here because we don't want to destroy
    // the sort order of the particles
    thrust::zip_iterator<pdata_iterator_tuple> pdata_middle =
        thrust::stable_partition(pdata_first,
                                 pdata_end,
                                 select_particle_keep(box,send_x, send_y, send_z));

    n_delete_ptls = pdata_end - pdata_middle;
    }

//! Determine particles to be sent in a given direction and fill send buffer
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
   \param box Dimensions of the local simulation box
   \param dir Direction particles are sent to
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
                           char *&d_send_buf_end,
                           gpu_boxsize box,
                           unsigned int dir)
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
    thrust::device_ptr<pdata_element> send_buf_ptr((pdata_element *) d_send_buf);

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


    // now pack the particles we want to send into a buffer
    thrust::device_ptr<pdata_element> send_buf_end_ptr =
        thrust::copy_if(thrust::make_transform_iterator(pdata_first, pack_pdata()),
                                  thrust::make_transform_iterator(pdata_end, pack_pdata()),
                                  send_buf_ptr,
                                  select_particle_migrate(box.xlo, box.xhi, box.ylo, box.yhi, box.zlo, box.zhi,dir));
    d_send_buf_end = (char *) thrust::raw_pointer_cast(send_buf_end_ptr);
    }

//! Select particles to forward in a given direction and pack them into a send buffer
/*! \param d_recv_buf Received particle data to check
 * \param d_recv_buf_end Pointer to end of received particle data
 * \param d_send_buf Send buffer to store particle data in
 * \param d_send_buf_end Pointer to end of send buffer (return value)
 * \param box Local box dimensions
 * \param dir Direction in which particles are to be forwarded
 */
void gpu_migrate_forward_particles(char *d_recv_buf,
                                   char *d_recv_buf_end,
                                   char *d_send_buf,
                                   char *&d_send_buf_end,
                                   gpu_boxsize box,
                                   unsigned int dir)
    {
    thrust::device_ptr<pdata_element> recv_buf_ptr((pdata_element *) d_recv_buf);
    thrust::device_ptr<pdata_element> recv_buf_end_ptr((pdata_element *) d_recv_buf_end);
    thrust::device_ptr<pdata_element> send_buf_ptr((pdata_element *) d_send_buf);

    device_ptr<pdata_element> send_buf_end_ptr =
        thrust::copy_if(recv_buf_ptr,
                        recv_buf_end_ptr,
                        send_buf_ptr,
                        select_particle_migrate(box.xlo, box.xhi, box.ylo, box.yhi, box.zlo, box.zhi, dir));
    d_send_buf_end = (char *) thrust::raw_pointer_cast(send_buf_end_ptr);
    }

//! Wrap received particles across global box boundaries
/*! \param d_recv_buf Received particle data
 * \param d_recv_buf_end End of received particle data
 * \param global_box Dimensions of global box
 * \param dir Direction along which particles where received
 */
void gpu_migrate_wrap_received_particles(char *d_recv_buf,
                                 char *d_recv_buf_end,
                                 const gpu_boxsize& global_box,
                                 unsigned int dir)
    {
    thrust::device_ptr<pdata_element> recv_buf_ptr((pdata_element *) d_recv_buf);
    thrust::device_ptr<pdata_element> recv_buf_end_ptr((pdata_element *) d_recv_buf_end);
    thrust::transform(recv_buf_ptr, recv_buf_end_ptr, recv_buf_ptr, wrap_received_particle(global_box, dir));
    }

//! Count received particles that are to be added to the local simulation box
/*!\param num_ptls_in_box Number of particles that are to be added to the local box (return value)
 * \param d_recv_buf Buffer of received particles to chek
 * \param d_recv_buf_end Pointer to end of received particle data
 * \param box Dimensions of local simulation box
 */
void gpu_migrate_count_particles_in_box(unsigned int &num_ptls_in_box,
                                char *d_recv_buf,
                                char *d_recv_buf_end,
                                const gpu_boxsize& box)
    {
    thrust::device_ptr<pdata_element> recv_buf_ptr((pdata_element *) d_recv_buf);
    thrust::device_ptr<pdata_element> recv_buf_end_ptr((pdata_element *) d_recv_buf_end);
    num_ptls_in_box = thrust::count_if(recv_buf_ptr, recv_buf_end_ptr, isInBox(box));
    }

//! Add received particles to local box if their positions are inside the local boundaries
/*! \param n_recv_ptls Number of received particles to check
 * \param d_recv_buf Buffer of received particle data
 * \param d_pos Array to store particle positions
 * \param d_vel Array to store particle velocities
 * \param d_accel Array to store particle accelerations
 * \param d_image Array to store particle images
 * \param d_charge Array to store particle charges
 * \param d_diameter Array to store particle diameters
 * \param d_body Array to store particle body ids
 * \param d_orientation Array to store particle body orientations
 * \param d_tag Array to store particle global tags
 * \param box Local box dimensions
 */
void gpu_migrate_move_particles_into_box(unsigned int &num_added_ptls,
                                 char *d_recv_buf,
                                 char *d_recv_buf_end,
                                 float4 *d_pos,
                                 float4 *d_vel,
                                 float3 *d_accel,
                                 int3 *d_image,
                                 float *d_charge,
                                 float *d_diameter,
                                 unsigned int *d_body,
                                 float4  *d_orientation,
                                 unsigned int *d_tag,
                                 const gpu_boxsize &box)
    {
    thrust::device_ptr<pdata_element> recv_buf_ptr((pdata_element *) d_recv_buf);
    thrust::device_ptr<pdata_element> recv_buf_end_ptr((pdata_element *) d_recv_buf_end);
    thrust::device_ptr<float4> pos_ptr(d_pos);
    thrust::device_ptr<float4> vel_ptr(d_vel);
    thrust::device_ptr<float3> accel_ptr(d_accel);
    thrust::device_ptr<int3> image_ptr(d_image);
    thrust::device_ptr<float> charge_ptr(d_charge);
    thrust::device_ptr<float> diameter_ptr(d_diameter);
    thrust::device_ptr<unsigned int> body_ptr(d_body);
    thrust::device_ptr<float4> orientation_ptr(d_orientation);
    thrust::device_ptr<unsigned int> tag_ptr(d_tag);
    num_added_ptls =
        thrust::copy_if(thrust::make_transform_iterator(recv_buf_ptr, unpack_pdata()),
                    thrust::make_transform_iterator(recv_buf_end_ptr, unpack_pdata()),
                    make_zip_iterator( thrust::make_tuple( pos_ptr,
                                               vel_ptr,
                                               accel_ptr,
                                               charge_ptr,
                                               diameter_ptr,
                                               image_ptr,
                                               body_ptr,
                                               orientation_ptr,
                                               tag_ptr) ),
                    isInBox(box)) - make_zip_iterator( thrust::make_tuple( pos_ptr,
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
 * \param rghost Boundary layer width
 */
void gpu_wrap_ghost_particles(unsigned int dir,
                              unsigned int n,
                              float4 *d_pos,
                              gpu_boxsize global_box,
                              float rghost)
    {
    thrust::device_ptr<float4> pos_ptr(d_pos);
    thrust::transform(pos_ptr, pos_ptr +n, pos_ptr, wrap_ghost_particle(global_box, rghost, dir));
    }

//! Construct a list of particle tags to send as ghost particles
/*! \param N number of particles to check
 * \param dir Direction in which ghost particles are sent
 * \param d_pos Array of particle positions
 * \param d_global_tag Array of particle global tags
 * \param d_copy_ghosts Array to be fillled x with global tags of particles that are to be send as ghosts
 * \param n_copy_ghosts Number of local particles that are sent in the given direction as ghosts (return value)
 * \param box Dimensions of local simulation box
 * \param r_ghost Width of boundary layer
 */
void gpu_make_exchange_ghost_list(unsigned int N,
                                  unsigned int dir,
                                  float4 *d_pos,
                                  unsigned int *d_global_tag,
                                  unsigned int* d_copy_ghosts,
                                  unsigned int &n_copy_ghosts,
                                  gpu_boxsize box,
                                  float r_ghost)
    {
    thrust::device_ptr<float4> pos_ptr(d_pos);
    thrust::device_ptr<unsigned int> global_tag_ptr(d_global_tag);
    thrust::device_ptr<unsigned int> copy_ghosts_ptr(d_copy_ghosts);

    thrust::device_ptr<unsigned int> copy_ghosts_end_ptr;

    copy_ghosts_end_ptr = thrust::copy_if(global_tag_ptr, global_tag_ptr+N, pos_ptr, copy_ghosts_ptr, select_particle_ghost(box, r_ghost, dir));

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
 */
void gpu_exchange_ghosts(unsigned int nghost,
                         unsigned int *d_copy_ghosts,
                         unsigned int *d_rtag,
                         float4 *d_pos,
                         float4 *d_pos_copybuf,
                         float *d_charge,
                         float *d_charge_copybuf,
                         float *d_diameter,
                         float *d_diameter_copybuf)
    {
    thrust::device_ptr<unsigned int> copy_ghosts_ptr(d_copy_ghosts);
    thrust::device_ptr<unsigned int> rtag_ptr(d_rtag);
    thrust::device_ptr<float4> pos_ptr(d_pos);
    thrust::device_ptr<float4> pos_copybuf_ptr(d_pos_copybuf);
    thrust::device_ptr<float> charge_ptr(d_charge);
    thrust::device_ptr<float> charge_copybuf_ptr(d_charge_copybuf);
    thrust::device_ptr<float> diameter_ptr(d_diameter);
    thrust::device_ptr<float> diameter_copybuf_ptr(d_diameter_copybuf);

    permutation_iterator<device_ptr<unsigned int>, device_ptr<unsigned int> > ghost_rtag(rtag_ptr, copy_ghosts_ptr);
    gather(ghost_rtag, ghost_rtag + nghost, pos_ptr, pos_copybuf_ptr);
    gather(ghost_rtag, ghost_rtag + nghost, charge_ptr, charge_copybuf_ptr);
    gather(ghost_rtag, ghost_rtag + nghost, diameter_ptr, diameter_copybuf_ptr);
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
#endif
