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
#include "BondData.cuh"

#include <thrust/replace.h>
#include <thrust/device_ptr.h>
#include <thrust/scatter.h>
#include <thrust/count.h>
#include <thrust/transform.h>
#include <thrust/copy.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/iterator/zip_iterator.h>

using namespace thrust;

unsigned int *d_n_fetch_ptl;              //!< Index of fetched particle from received ptl list

__constant__ unsigned int d_corner_plan_lookup[8]; //!< Lookup-table corner -> plan flags
__constant__ unsigned int d_edge_plan_lookup[12];  //!< Lookup-table edges -> plan flags
__constant__ unsigned int d_face_plan_lookup[6];   //!< Lookup-table faces -> plan falgs

__constant__ unsigned int d_is_communicating[6]; //!< Per-direction flag indicating whether we are communicating in that direction
__constant__ unsigned int d_is_at_boundary[6]; //!< Per-direction flag indicating whether the box has a global boundary

void gpu_allocate_tmp_storage(const unsigned int *is_communicating,
                              const unsigned int *is_at_boundary,
                              const unsigned int *corner_plan_lookup,
                              const unsigned int *edge_plan_lookup,
                              const unsigned int *face_plan_lookup)
    {
    cudaMalloc(&d_n_fetch_ptl,sizeof(unsigned int));

    cudaMemcpyToSymbol(d_corner_plan_lookup, corner_plan_lookup, sizeof(unsigned int)*8);
    cudaMemcpyToSymbol(d_edge_plan_lookup, edge_plan_lookup, sizeof(unsigned int)*12);
    cudaMemcpyToSymbol(d_face_plan_lookup, face_plan_lookup, sizeof(unsigned int)*6);

    cudaMemcpyToSymbol(d_is_communicating, is_communicating, sizeof(unsigned int)*6);
    cudaMemcpyToSymbol(d_is_at_boundary, is_at_boundary, sizeof(unsigned int)*6);
    }

void gpu_deallocate_tmp_storage()
    {
    cudaFree(d_n_fetch_ptl);
    }


//! GPU Kernel to find incomplete bonds
__global__ void gpu_mark_particles_in_incomplete_bonds_kernel(const uint2 *btable,
                                                         unsigned char *plan,
                                                         const Scalar4 *d_postype,
                                                         const unsigned int *d_rtag,
                                                         const unsigned int N,
                                                         const unsigned int n_bonds,
                                                         const BoxDim box)
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
        const Scalar4& postype = d_postype[idx2];
        Scalar3 pos = make_scalar3(postype.x,postype.y,postype.z);
        // Multiple threads may update the plan simultaneously, but this should
        // be safe, since they store the same result
        unsigned char p = plan[idx2];
        Scalar3 f = box.makeFraction(pos);
        p |= (f.x > Scalar(0.5)) ? send_east : send_west;
        p |= (f.y > Scalar(0.5)) ? send_north : send_south;
        p |= (f.z > Scalar(0.5)) ? send_up : send_down;
        plan[idx2] = p;
        }
    else if ((idx1 < N) && (idx2 >= N))
        {
        // send particle with index idx1 to neighboring domains
        const Scalar4& postype = d_postype[idx1];
        Scalar3 pos = make_scalar3(postype.x,postype.y,postype.z);
        // Multiple threads may update the plan simultaneously, but this should
        // be safe, since they store the same result
        unsigned char p = plan[idx1];
        Scalar3 f = box.makeFraction(pos);
        p |= (f.x > Scalar(0.5)) ? send_east : send_west;
        p |= (f.y > Scalar(0.5)) ? send_north : send_south;
        p |= (f.z > Scalar(0.5)) ? send_up : send_down;
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
                                          const Scalar4 *d_pos,
                                          const unsigned int *d_rtag,
                                          const unsigned int N,
                                          const unsigned int n_bonds,
                                          const BoxDim& box)
    {
    unsigned int block_size = 512;
    gpu_mark_particles_in_incomplete_bonds_kernel<<<n_bonds/block_size + 1, block_size>>>(d_btable,
                                                                                    d_plan,
                                                                                    d_pos,
                                                                                    d_rtag,
                                                                                    N,
                                                                                    n_bonds,
                                                                                    box);
    }

//! Select a particle for migration
struct select_particle_migrate_gpu : public thrust::unary_function<const unsigned int, bool>
    {
    const BoxDim box;       //!< Local simulation box dimensions
    const unsigned int dir; //!< Direction to send particles to

    //! Constructor
    /*!
     */
    select_particle_migrate_gpu(const BoxDim & _box,
                            const unsigned int _dir)
        : box(_box), dir(_dir)
        { }

    //! Select a particle
    /*! t particle data to consider for sending
     * \return true if particle stays in the box
     */
    __host__ __device__ bool operator()(const Scalar4 postype)
        {
        Scalar3 pos = make_scalar3(postype.x, postype.y, postype.z);
        Scalar3 f = box.makeFraction(pos);

        // return true if the particle stays leaves the box
        return ((dir == face_east && f.x >= Scalar(1.0)) ||   // send east
                (dir == face_west && f.x < Scalar(0.0))  ||   // send west
                (dir == face_north && f.y >= Scalar(1.0)) ||  // send north
                (dir == face_south && f.y < Scalar(0.0))  ||  // send south
                (dir == face_up && f.z >= Scalar(1.0)) ||     // send up
                (dir == face_down && f.z < Scalar(0.0) ));    // send down
        }

     };

/*! \param N Number of local particles
    \param d_pos Device array of particle positions
    \param d_tag Device array of particle tags
    \param d_rtag Device array for reverse-lookup table
    \param dir Current direction
    \param box Local box
 */
void gpu_stage_particles(const unsigned int N,
                         const Scalar4 *d_pos,
                         const unsigned int *d_tag,
                         unsigned int *d_rtag,
                         const unsigned int dir,
                         const BoxDim& box,
                         cached_allocator& alloc)
    {
    // Wrap particle data arrays
    thrust::device_ptr<const Scalar4> pos_ptr(d_pos);
    thrust::device_ptr<const unsigned int> tag_ptr(d_tag);

    // Wrap rtag array
    thrust::device_ptr<unsigned int> rtag_ptr(d_rtag);

    // pointer from tag into rtag
    thrust::permutation_iterator<
        thrust::device_ptr<unsigned int>, thrust::device_ptr<const unsigned int> > rtag_prm(rtag_ptr, tag_ptr);

    // set flag for particles that are to be sent
    thrust::replace_if(thrust::cuda::par(alloc),
        rtag_prm, rtag_prm + N, pos_ptr, select_particle_migrate_gpu(box, dir), STAGED);
    }

//! Select a bond for migration
struct wrap_particle_op_gpu : public thrust::unary_function<const pdata_element, pdata_element>
    {
    const BoxDim box; //!< The box for which we are applying boundary conditions

    //! Constructor
    /*!
     */
    wrap_particle_op_gpu(const BoxDim _box)
        : box(_box)
        {
        }

    //! Wrap position information inside particle data element
    /*! \param p Particle data element
     * \returns The particle data element with wrapped coordinates
     */
    __device__ pdata_element operator()(const pdata_element p)
        {
        pdata_element ret = p;
        box.wrap(ret.pos, ret.image);
        return ret;
        }
     };


/*! \param n_recv Number of particles in buffer
    \param d_in Buffer of particle data elements
    \param box Box for which to apply boundary conditions
 */
void gpu_wrap_particles(const unsigned int n_recv,
                        pdata_element *d_in,
                        const BoxDim& box)
    {
    // Wrap device ptr
    thrust::device_ptr<pdata_element> in_ptr(d_in);

    // Apply box wrap to input buffer
    thrust::transform(in_ptr, in_ptr + n_recv, in_ptr, wrap_particle_op_gpu(box));
    }

//! A tuple of bond data pointers (const version)
typedef thrust::tuple <
    thrust::device_ptr<const unsigned int>,  // tag
    thrust::device_ptr<const uint2>,         // bond
    thrust::device_ptr<const unsigned int>   // type
    > bdata_it_tuple_gpu_const;

//! A zip iterator for accessing bond data (const version)
typedef thrust::zip_iterator<bdata_it_tuple_gpu_const> bdata_zip_gpu_const;

//! A tuple of bond data fields
typedef thrust::tuple <
    const unsigned int,  // tag
    const uint2 ,        // bond
    const unsigned int   // type
    > bdata_tuple_gpu;

//! A converter from a tuple of bond data fields to a bond_element
struct to_bond_element_gpu : public thrust::unary_function<const bdata_tuple_gpu,const bond_element>
    {
    __device__ const bond_element operator() (const bdata_tuple_gpu t)
        {
        bond_element b;
        b.tag = thrust::get<0>(t);
        b.bond = thrust::get<1>(t);
        b.type = thrust::get<2>(t);
        return b;
        }
    };

//! Select bonds that leave the box
struct select_bond_migrate_gpu : public thrust::unary_function<const unsigned int, bool>
    {
    const unsigned int *d_rtag;       //!< Particle r-lookup table

    //! Constructor
    /*!
     */
    select_bond_migrate_gpu(const unsigned int *_d_rtag)
        : d_rtag(_d_rtag)
        {
        }

    //! Select bonds for sending
    /*! \param b bond to consider for sending
     * \return true if bond is leaving this box
     */
    __device__ bool operator()(const uint2 b)
        {
        unsigned int rtag_a = d_rtag[b.x];
        unsigned int rtag_b = d_rtag[b.y];

        // number of particles that remain local
        unsigned num_local = 2;
        if (rtag_a == NOT_LOCAL || rtag_a == STAGED) num_local--;
        if (rtag_b == NOT_LOCAL || rtag_b == STAGED) num_local--;

        // if no particle is local anymore, bond is sent and removed
        return !num_local;
        }
     };

//! Select bonds that are split (a copy is sent to another domain)
struct select_bond_split_gpu : public thrust::unary_function<const uint2, bool>
    {
    const unsigned int *d_rtag;       //!< Particle r-lookup table

    //! Constructor
    /*!
     */
    select_bond_split_gpu(const unsigned int *_d_rtag)
        : d_rtag(_d_rtag)
        {
        }

    //! Select bonds for sending
    /*! \param b bond to consider for sending
     * \return true if bond is leaving this box
     */
    __device__ bool operator()(const uint2 b)
        {
        unsigned int rtag_a = d_rtag[b.x];
        unsigned int rtag_b = d_rtag[b.y];

        // number of particles that remain local
        unsigned num_local = 2;
        if (rtag_a == NOT_LOCAL || rtag_a == STAGED) num_local--;
        if (rtag_b == NOT_LOCAL || rtag_b == STAGED) num_local--;

        // number of particles that leave the domain
        unsigned int num_leave = 0;
        if (rtag_a == STAGED) num_leave++;
        if (rtag_b == STAGED) num_leave++;

        return num_local && num_leave;
        }
     };

/*! \param n_bonds Number of local bonds
    \param d_bonds Array of bonds
    \param d_bond_tag Array of bond tags
    \param d_bond_rtag Reverse-lookup table for bond tags
    \param d_rtag Particle data reverse-lookup table
    \param d_out Output array for packed bond data
 */
void gpu_select_bonds(unsigned int n_bonds,
                      const uint2 *d_bonds,
                      const unsigned int *d_bond_tag,
                      unsigned int *d_bond_rtag,
                      const unsigned int *d_rtag,
                      cached_allocator& alloc)
    {
    // Wrap bond data pointers
    thrust::device_ptr<const uint2> bonds_ptr(d_bonds);
    thrust::device_ptr<const unsigned int> bond_tag_ptr(d_bond_tag);

    thrust::device_ptr<unsigned int> bond_rtag_ptr(d_bond_rtag);

    // Wrap reverse-lookup for particles
    thrust::device_ptr<const unsigned int> rtag_ptr(d_rtag);

    // pointer from bond tag into bond rtag
    thrust::permutation_iterator<
        thrust::device_ptr<unsigned int>, thrust::device_ptr<const unsigned int> >
        bond_rtag_prm(bond_rtag_ptr, bond_tag_ptr);

    // set bond rtags for bonds that leave the domain to BOND_STAGED
    thrust::replace_if(thrust::cuda::par(alloc),
        bond_rtag_prm, bond_rtag_prm+n_bonds, bonds_ptr, select_bond_migrate_gpu(d_rtag), BOND_STAGED);

    // set bond rtags for bonds that are replicated to BOND_SPLIT
    thrust::replace_if(thrust::cuda::par(alloc),
        bond_rtag_prm, bond_rtag_prm+n_bonds, bonds_ptr, select_bond_split_gpu(d_rtag), BOND_SPLIT);
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


//! Kernel to select ghost atoms due to non-bonded interactions
__global__ void gpu_nonbonded_plan_kernel(unsigned char *plan,
                               const unsigned int N,
                               Scalar4 *d_postype,
                               const BoxDim box,
                               Scalar3 ghost_fraction)
    {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= N) return;

    Scalar4 postype = d_postype[idx];
    Scalar3 pos = make_scalar3(postype.x,postype.y,postype.z);
    Scalar3 f = box.makeFraction(pos);

    unsigned char p = plan[idx];

    // is particle inside ghost layer? set plan accordingly.
    if (f.x >= Scalar(1.0) - ghost_fraction.x)
        p |= send_east;
    if (f.x < ghost_fraction.x)
        p |= send_west;
    if (f.y >= Scalar(1.0) - ghost_fraction.y)
        p |= send_north;
    if (f.y < ghost_fraction.y)
        p |= send_south;
    if (f.z >= Scalar(1.0) - ghost_fraction.z)
        p |= send_up;
    if (f.z < ghost_fraction.z)
        p |= send_down;

    // write out plan
    plan[idx] = p;
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
                                      Scalar4 *d_pos,
                                      const BoxDim &box,
                                      Scalar3 ghost_fraction)
    {
    unsigned int block_size = 512;

    gpu_nonbonded_plan_kernel<<<N/block_size+1, block_size>>>(d_plan,
                                                              N,
                                                              d_pos,
                                                              box,
                                                              ghost_fraction);
    }

//! Write data to send buffer
__device__ void write_to_buf(char *buf,
                             const Scalar4 postype,
                             const Scalar4 vel,
                             const Scalar charge,
                             const Scalar diameter,
                             const Scalar4 orientation,
                             const unsigned int tag,
                             unsigned char exch_pos,
                             unsigned char exch_vel,
                             unsigned char exch_charge,
                             unsigned char exch_diameter,
                             unsigned char exch_orientation)
    {
    unsigned int offs = 0;

    // Scalar4's first, then Scalars/unsigned ints
    // for proper alignment
    if (exch_pos)
        {
        *((Scalar4 *) &buf[offs]) = postype;
        offs += sizeof(Scalar4);
        }
    if (exch_vel)
        {
        *((Scalar4 *) &buf[offs]) = vel;
        offs += sizeof(Scalar4);
        }
    if (exch_orientation)
        {
        *((Scalar4 *) &buf[offs]) = orientation;
        offs += sizeof(Scalar4);
        }
    if (exch_charge)
        {
        *((Scalar *) &buf[offs]) = charge;
        offs += sizeof(Scalar);
        }
    if (exch_diameter)
        {
        *((Scalar *) &buf[offs]) = diameter;
        offs += sizeof(Scalar);
        }

    *((unsigned int *) &buf[offs]) = tag;
    offs += sizeof(Scalar);
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
                                         Scalar4 *d_pos,
                                         Scalar *d_charge,
                                         Scalar *d_diameter,
                                         Scalar4 *d_vel,
                                         Scalar4 *d_orientation,
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
                                         unsigned int *condition,
                                         unsigned int sz,
                                         unsigned char exch_pos,
                                         unsigned char exch_vel,
                                         unsigned char exch_charge,
                                         unsigned char exch_diameter,
                                         unsigned char exch_orientation)

    {
    unsigned int idx = blockIdx.x*blockDim.x + threadIdx.x;

    if (idx >= N) return;

    unsigned char plan = d_plan[idx];

    // if we are not communicating in a direction, discard it from plan
    for (unsigned int face = 0; face < 6; ++face)
        plan &= (d_is_communicating[face]) ? ~0 : ~d_face_plan_lookup[face];

    if (plan)
        {
        Scalar4 postype;
        Scalar4 vel;
        Scalar charge;
        Scalar diameter;
        Scalar4 orientation;
        unsigned int tag;
        if (exch_pos)
            postype = d_pos[idx];
        if (exch_vel)
            vel = d_vel[idx];
        if (exch_charge)
            charge = d_charge[idx];
        if (exch_diameter)
            diameter = d_diameter[idx];
        if (exch_orientation)
            orientation =  d_orientation[idx];

        tag = d_tag[idx];

        // the boundary plan indicates whether the particle will cross a particle
        // in a specific direction
        unsigned int boundary_plan = 0;
        for (unsigned int face = 0; face < 6; ++face)
            if (d_is_at_boundary[face] && (plan & d_face_plan_lookup[face]))
                boundary_plan |= d_face_plan_lookup[face];

        unsigned int count = 0;
        if (plan & send_east) count++;
        if (plan & send_west) count++;
        if (plan & send_north) count++;
        if (plan & send_south) count++;
        if (plan & send_up) count++;
        if (plan & send_down) count++;

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
                {
                atomicOr(condition, 8); // invalid plan
                return;
                }

            unsigned int n = atomicInc(&n_copy_ghosts_corner[corner],0xffffffff);
            if (n < max_copy_ghosts_corner)
                {
                // copy ghost data to send buffer
                write_to_buf(&d_ghost_corner_buf[n*sz+corner*corner_buf_pitch],
                    postype, vel, charge, diameter, orientation, tag,
                    exch_pos, exch_vel, exch_charge, exch_diameter, exch_orientation);

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
            unsigned int edge = 0;
            for (unsigned int i = 0; i < 12; ++i)
                if ((plan & d_edge_plan_lookup[i]) == d_edge_plan_lookup[i])
                    {
                    has_edge = true;
                    edge = i;
                    break;
                    }

            if (!has_edge)
                {
                atomicOr(condition,8); // invalid plan
                return;
                }

            unsigned int n = atomicInc(&n_copy_ghosts_edge[edge],0xffffffff);
            if (n < max_copy_ghosts_edge)
                {
                // copy ghost data to send buffer
                write_to_buf(&d_ghost_edge_buf[n*sz+edge*edge_buf_pitch],
                    postype, vel, charge, diameter, orientation, tag,
                    exch_pos, exch_vel, exch_charge, exch_diameter, exch_orientation);

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
                {
                atomicOr(condition,8); // invalid plan
                return;
                }

            unsigned int n = atomicInc(&n_copy_ghosts_face[face],0xffffffff);
            if (n < max_copy_ghosts_face)
                {
                // copy ghost data to send buffer
                write_to_buf(&d_ghost_face_buf[n*sz+face*face_buf_pitch],
                    postype, vel, charge, diameter, orientation, tag,
                    exch_pos, exch_vel, exch_charge, exch_diameter, exch_orientation);

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
/*! \param N number of local particles
 * \param d_plan Array of particle exchange plans
 * \param d_tag Array of particle global tags
 * \param d_ghost_idx_face List of particle indices sent as ghosts through a face (return array)
 * \param ghost_idx_face_pitch Offset of different faces in ghost index list
 * \param d_ghost_idx_edge List of particle indices sent as ghosts over an edge (return array)
 * \param ghost_idx_edge_pitch Offset of different edges in ghost index list
 * \param d_ghost_idx_corner List of particle indices sent as ghosts via an corner (return array)
 * \param ghost_idx_corner_pitch Offset of different corners in ghost index list
 * \param d_pos Array of particle positions
 * \param d_charge Array of particle charges
 * \param d_diameter Array of particle diameters
 * \param d_vel Particle data array of velocities
 * \param d_orientation Particle data array of orientations
 * \param d_ghost_corner_buf Buffer for ghosts sent via a corner (return array)
 * \param corner_buf_pitch Offsets of different corners in send buffer
 * \param d_ghost_edge_buf Buffer for ghosts sent over an edge (return array)
 * \param edge_buf_pitch Offsets of different edges in send buffer
 * \param d_ghost_face_buf Buffer for ghosts sent through a face (return array)
 * \param face_buf_pitch Offsets of different faces in send buffer
 * \param d_n_copy_ghosts_corner Number of ghosts sent via a corner (return array)
 * \param d_n_copy_ghosts_edge Number of ghosts sent over an edge (return array)
 * \param d_n_copy_ghosts_face Number of ghosts sent via a corner (return array)
 * \param max_copy_ghosts_corner Size of corner ghost send buffer
 * \param max_copy_ghosts_edge Size of edge ghost send buffer
 * \param max_copy_ghosts_face Size of face ghost send buffer
 * \param d_condition Return value, unequal zero if buffer overflow
 * \param sz Size of exchange element
 * \param exch_pos >0 if exchanging positions
 * \param exch_vel >0 if exchanging positions
 * \param exch_charge >0 if exchanging positions
 * \param exch_diameter >0 if exchanging positions
 * \param exch_orientation >0 if exchanging positions
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
                         Scalar4 *d_pos,
                         Scalar *d_charge,
                         Scalar *d_diameter,
                         Scalar4 *d_vel,
                         Scalar4 *d_orientation,
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
                         unsigned int *d_condition,
                         unsigned int sz,
                         unsigned char exch_pos,
                         unsigned char exch_vel,
                         unsigned char exch_charge,
                         unsigned char exch_diameter,
                         unsigned char exch_orientation)
    {
    cudaMemset(d_n_copy_ghosts_corner, 0, sizeof(unsigned int)*8);
    cudaMemset(d_n_copy_ghosts_edge, 0, sizeof(unsigned int)*12);
    cudaMemset(d_n_copy_ghosts_face, 0, sizeof(unsigned int)*6);

    unsigned int block_size = 256;
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
                         d_vel,
                         d_orientation,
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
                         d_condition,
                         sz,
                         exch_pos,
                         exch_vel,
                         exch_charge,
                         exch_diameter,
                         exch_orientation);
    }

//! Unpack a buffer of received ghost particles into local particle data arrays
/*! \tparam element_type Type of ghost element
    \tparam update True if only updating positions, false if copying all fields needed for force calculation
 */
template<bool update>
__global__ void gpu_exchange_ghosts_unpack_kernel(unsigned int N,
                                                  unsigned int n_tot_recv_ghosts,
                                                  const unsigned int *n_local_ghosts_face,
                                                  const unsigned int *n_local_ghosts_edge,
                                                  const unsigned int n_tot_recv_ghosts_local,
                                                  const unsigned int *n_recv_ghosts_local,
                                                  const unsigned int *n_recv_ghosts_face,
                                                  const unsigned int *n_recv_ghosts_edge,
                                                  const char *d_face_ghosts,
                                                  const unsigned int face_pitch,
                                                  const char *d_edge_ghosts,
                                                  const unsigned int edge_pitch,
                                                  const char *d_recv_ghosts,
                                                  Scalar4 *d_pos,
                                                  Scalar *d_charge,
                                                  Scalar *d_diameter,
                                                  Scalar4 *d_vel,
                                                  Scalar4 *d_orientation,
                                                  unsigned int *d_tag,
                                                  unsigned int *d_rtag,
                                                  const BoxDim shifted_global_box,
                                                  const unsigned int sz,
                                                  unsigned char exch_pos,
                                                  unsigned char exch_vel,
                                                  unsigned char exch_charge,
                                                  unsigned char exch_diameter,
                                                  unsigned char exch_orientation
                                                  )
    {
    unsigned int ghost_idx = blockIdx.x*blockDim.x+threadIdx.x;

    if (ghost_idx >= n_tot_recv_ghosts) return;

    char *el_ptr;

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

        el_ptr = (char *) &d_recv_ghosts[local_idx*sz];
        done = true;
        }
    else
        offset += n_tot_recv_ghosts_local;

    if (! done)
        {
        unsigned int n_tot_recv_ghosts_face[6];

        for (unsigned int i = 0; i < 6; ++i)
            {
            n_tot_recv_ghosts_face[i] = 0;
            for (unsigned int j = 0; j < 6; ++j)
                n_tot_recv_ghosts_face[i] += n_recv_ghosts_face[6*j+i];
            }

        // ghosts we have forwarded over a face of our box
        for (unsigned int i=0; i < 6; ++i)
            {
            local_idx = ghost_idx - offset;

            if (local_idx < n_tot_recv_ghosts_face[i])
                {
                unsigned int local_offset = 0;
                for (recv_dir = 0; recv_dir < 6; ++recv_dir)
                    {
                    local_offset += n_recv_ghosts_face[6*recv_dir+i];
                    if (local_idx < local_offset) break;
                    }

                unsigned int n = n_local_ghosts_face[i]+local_idx;
                el_ptr = (char *) &d_face_ghosts[n*sz + i*face_pitch];
                done = true;
                break;
                }
            else
                offset += n_tot_recv_ghosts_face[i];
            }
        }

    if (! done)
        {
        unsigned int n_tot_recv_ghosts_edge[12];

        for (unsigned int i = 0; i < 12; ++i)
            {
            n_tot_recv_ghosts_edge[i] = 0;
            for (unsigned int j = 0; j < 6; ++j)
                n_tot_recv_ghosts_edge[i] += n_recv_ghosts_edge[12*j+i];
            }

        // ghosts we have forwared over an edge of our box
        for (unsigned int i=0; i < 12; ++i)
            {
            local_idx = ghost_idx - offset;

            if (local_idx < n_tot_recv_ghosts_edge[i])
                {
                unsigned int local_offset = 0;
                for (recv_dir = 0; recv_dir < 6; ++recv_dir)
                    {
                    local_offset += n_recv_ghosts_edge[12*recv_dir+i];
                    if (local_idx < local_offset) break;
                    }

                unsigned int n = n_local_ghosts_edge[i]+local_idx;
                el_ptr = (char *) &d_edge_ghosts[n*sz + i*edge_pitch];
                done = true;
                break;
                }
            else
                offset += n_tot_recv_ghosts_edge[i];
            }
        }

    // we have a pointer to the data element to be unpacked, now unpack
    // unpack in the order it was packed
    Scalar4 postype;
    if (update)
        {
        unsigned int offs = 0;
        if (exch_pos)
            {
            Scalar4 &pos = *((Scalar4 *) (el_ptr+offs));
            offs += sizeof(Scalar4);
            postype = pos;
            }
        if (exch_vel)
            {
            Scalar4 &vel = *((Scalar4 *) (el_ptr+offs));
            offs += sizeof(Scalar4);
            d_vel[N+ghost_idx] = vel;
            }
        if (exch_orientation)
            {
            Scalar4 &orientation = *((Scalar4 *) (el_ptr+offs));
            offs += sizeof(Scalar4);
            d_orientation[N+ghost_idx] = orientation;
            }
        }
    else
        {
        unsigned int offs = 0;
        if (exch_pos)
            {
            Scalar4 &pos = *((Scalar4 *) (el_ptr+offs));
            offs += sizeof(Scalar4);
            postype = pos;
            }
        if (exch_vel)
            {
            Scalar4 &vel = *((Scalar4 *) (el_ptr+offs));
            offs += sizeof(Scalar4);
            d_vel[N+ghost_idx] = vel;
            }
        if (exch_orientation)
            {
            Scalar4 &orientation = *((Scalar4 *) (el_ptr+offs));
            offs += sizeof(Scalar4);
            d_orientation[N+ghost_idx] = orientation;
            }
        if (exch_charge)
            {
            Scalar charge = *((Scalar *) (el_ptr+offs));
            offs += sizeof(Scalar);
            d_charge[N+ghost_idx] = charge;
            }
       if (exch_diameter)
            {
            Scalar diameter = *((Scalar *) (el_ptr+offs));
            offs += sizeof(Scalar);
            d_diameter[N+ghost_idx] = diameter;
            }

        unsigned int tag = *((unsigned int *) (el_ptr+offs));
        offs += sizeof(unsigned int);

        d_tag[N+ghost_idx] = tag;
        d_rtag[tag] = N+ghost_idx;
        }

    // apply global boundary conditions for received particle
    if (exch_pos)
        {
        int3 img = make_int3(0,0,0);
        shifted_global_box.wrap(postype,img);

        d_pos[N+ghost_idx] = postype;
        }
    }

//! Unpack received ghosts
/*! \param N Number of local particles
    \param n_tot_recv_ghosts Number of received ghosts
    \param d_n_local_ghosts_face Number of local particles SENT as ghosts across a face
    \param d_n_local_ghosts_edge Number of local particle SENT as ghosts over an edge
    \param n_tot_recv_ghosts_local Number of ghosts received for local box
    \param d_n_recv_ghosts_local Number of ghosts received for local box (per send direction)
    \param d_n_recv_ghosts_face Number of ghosts received for forwarding across a face (per send-direction and face)
    \param d_n_recv_ghosts_edge Number of ghosts received for forwarding over an edge (per send-direction an edge)
    \param d_face_ghosts Buffer of ghosts sent/forwarded across a face
    \param face_pitch Offsets of different faces in face ghost buffer
    \param d_edge_ghosts Buffer of ghosts sent/forwarded over an edge
    \param edge_pitch Offsets of different edges in edge ghost buffer
    \param d_recv_ghosts Buffer of ghosts received for the local box
    \param d_pos Array of particle positions
    \param d_charge Array of particle charges
    \param d_diameter Array of particle diameters
    \param d_vel Array of particle velocities
    \param d_orientation Array of particle orientations
    \param d_tag Array of particle tags
    \param d_rtag Lookup table particle tag->idx
    \param shifted_global_box Global simulation box, shifted by one local box if local box has a global boundary
    \param sz Size of exchange element
    \param exch_pos >0 if exchanging positions
    \param exch_vel >0 if exchanging positions
    \param exch_charge >0 if exchanging positions
    \param exch_diameter >0 if exchanging positions
    \param exch_orientation >0 if exchanging positions
*/
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
                                Scalar4 *d_vel,
                                Scalar4 *d_orientation,
                                unsigned int *d_tag,
                                unsigned int *d_rtag,
                                const BoxDim& shifted_global_box,
                                unsigned int sz,
                                unsigned char exch_pos,
                                unsigned char exch_vel,
                                unsigned char exch_charge,
                                unsigned char exch_diameter,
                                unsigned char exch_orientation
                                )
    {
    unsigned int block_size = 512;

    gpu_exchange_ghosts_unpack_kernel<false><<<n_tot_recv_ghosts/block_size+1, block_size>>>(N,
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
                                                                           d_vel,
                                                                           d_orientation,
                                                                           d_tag,
                                                                           d_rtag,
                                                                           shifted_global_box,
                                                                           sz,
                                                                           exch_pos,
                                                                           exch_vel,
                                                                           exch_charge,
                                                                           exch_diameter,
                                                                           exch_orientation);
    }

//! Kernel to pack local particle data into ghost send buffers
__global__ void gpu_update_ghosts_pack_kernel(const unsigned int n_copy_ghosts,
                                         const unsigned int *d_ghost_idx_face,
                                         const unsigned int ghost_idx_face_pitch,
                                         const unsigned int *d_ghost_idx_edge,
                                         const unsigned int ghost_idx_edge_pitch,
                                         const unsigned int *d_ghost_idx_corner,
                                         const unsigned int ghost_idx_corner_pitch,
                                         const Scalar4 *d_pos,
                                         const Scalar4 *d_vel,
                                         const Scalar4 *d_orientation,
                                         char *d_update_corner_buf,
                                         unsigned int corner_buf_pitch,
                                         char *d_update_edge_buf,
                                         unsigned int edge_buf_pitch,
                                         char *d_update_face_buf,
                                         unsigned int face_buf_pitch,
                                         const unsigned int *n_copy_ghosts_corner,
                                         const unsigned int *n_copy_ghosts_edge,
                                         const unsigned int *n_copy_ghosts_face,
                                         unsigned int sz,
                                         unsigned char exch_pos,
                                         unsigned char exch_vel,
                                         unsigned char exch_orientation)
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

    char *buf_ptr = NULL;

    // the particle index we are going to fetch (initialized with a dummy value)
    unsigned int idx = NOT_LOCAL;

    bool done = false;

    // first, ghosts that are sent over a corner
    for (unsigned int corner=0; corner < 8; ++corner)
        {
        unsigned int local_idx = ghost_idx - offset;

        if (local_idx < n_copy_ghosts_corner[corner])
            {
            idx = d_ghost_idx_corner[local_idx + corner * ghost_idx_corner_pitch];
            buf_ptr = (char *) &d_update_corner_buf[local_idx*sz + corner*corner_buf_pitch];
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
                buf_ptr = (char *) &d_update_edge_buf[local_idx*sz + edge*edge_buf_pitch];
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
                buf_ptr = (char *) &d_update_face_buf[local_idx*sz + face*face_buf_pitch];
                done = true;
                break;
                }
            else
                offset += n_copy_ghosts_face[face];
            }
        }

    // we have found a ghost index to be updated
    // store data in buffer element
    unsigned int offs = 0;
    if (exch_pos)
        {
        *((Scalar4 *) (buf_ptr+offs)) = d_pos[idx];
        offs += sizeof(Scalar4);
        }
    if (exch_vel)
        {
        *((Scalar4 *) (buf_ptr+offs)) = d_vel[idx];
        offs += sizeof(Scalar4);
        }
    if (exch_orientation)
        {
        *((Scalar4 *) (buf_ptr+offs)) = d_orientation[idx];
        offs += sizeof(Scalar4);
        }
    }

//! Pack local particle data into ghost send buffers
/*! \param n_copy_ghosts Number of ghosts to be packed
    \param d_ghost_idx_face List of particle indices sent as ghosts across a face
    \param ghost_idx_face_pitch Offsets of different box faces in ghost index list
    \param d_ghost_idx_edge List of particle indices sent as ghosts across a edge
    \param ghost_idx_edge_pitch Offsets of different box edges in ghost index list
    \param d_ghost_idx_corner List of particle indices sent as ghosts across a corner
    \param ghost_idx_corner_pitch Offsets of different box corners in ghost index list
    \param d_pos Array of particle positions
    \param d_update_corner_buf Buffer of ghost particle positions sent via corner (return array)
    \param corner_buf_pitch Offsets of different corners in update buffer
    \param d_update_edge_buf Buffer of ghost particle positions sent over an edge (return array)
    \param edge_buf_pitch Offsets of different edges in update buffer
    \param d_update_face_buf Buffer of ghost particle positions sent over an face (return array)
    \param face_buf_pitch Buffer of different faces in update buffer
    \param d_n_local_ghosts_corner Number of ghosts sent in every corner
    \param d_n_local_ghosts_edge Number of ghosts sent over every edge
    \param d_n_local_ghosts_face Number of ghosts sent across every face
    \param sz Size of update element
    \param exch_pos >0 if exchanging positions
    \param exch_vel >0 if exchanging positions
    \param exch_orientation >0 if exchanging positions
*/
void gpu_update_ghosts_pack(const unsigned int n_copy_ghosts,
                                     const unsigned int *d_ghost_idx_face,
                                     const unsigned int ghost_idx_face_pitch,
                                     const unsigned int *d_ghost_idx_edge,
                                     const unsigned int ghost_idx_edge_pitch,
                                     const unsigned int *d_ghost_idx_corner,
                                     const unsigned int ghost_idx_corner_pitch,
                                     const Scalar4 *d_pos,
                                     const Scalar4 *d_vel,
                                     const Scalar4 *d_orientation,
                                     char *d_update_corner_buf,
                                     unsigned int corner_buf_pitch,
                                     char *d_update_edge_buf,
                                     unsigned int edge_buf_pitch,
                                     char *d_update_face_buf,
                                     unsigned int face_buf_pitch,
                                     const unsigned int *d_n_local_ghosts_corner,
                                     const unsigned int *d_n_local_ghosts_edge,
                                     const unsigned int *d_n_local_ghosts_face,
                                     unsigned int sz,
                                     unsigned char exch_pos,
                                     unsigned char exch_vel,
                                     unsigned char exch_orientation)
    {
    unsigned int block_size = 512;
    gpu_update_ghosts_pack_kernel<<<n_copy_ghosts/block_size+1,block_size>>>(n_copy_ghosts,
                                                                             d_ghost_idx_face,
                                                                             ghost_idx_face_pitch,
                                                                             d_ghost_idx_edge,
                                                                             ghost_idx_edge_pitch,
                                                                             d_ghost_idx_corner,
                                                                             ghost_idx_corner_pitch,
                                                                             d_pos,
                                                                             d_vel,
                                                                             d_orientation,
                                                                             d_update_corner_buf,
                                                                             corner_buf_pitch,
                                                                             d_update_edge_buf,
                                                                             edge_buf_pitch,
                                                                             d_update_face_buf,
                                                                             face_buf_pitch,
                                                                             d_n_local_ghosts_corner,
                                                                             d_n_local_ghosts_edge,
                                                                             d_n_local_ghosts_face,
                                                                             sz,
                                                                             exch_pos,
                                                                             exch_vel,
                                                                             exch_orientation);
    }

//! Unpack received ghosts
/*! \param N Number of local particles
    \param n_tot_recv_ghosts Number of received ghosts
    \param d_n_local_ghosts_face Number of local particles SENT as ghosts across a face
    \param d_n_local_ghosts_edge Number of local particle SENT as ghosts over an edge
    \param n_tot_recv_ghosts_local Number of ghosts received for local box
    \param d_n_recv_ghosts_local Number of ghosts received for local box (per send direction)
    \param d_n_recv_ghosts_face Number of ghosts received for forwarding across a face (per send-direction and face)
    \param d_n_recv_ghosts_edge Number of ghosts received for forwarding over an edge (per send-direction an edge)
    \param d_face_ghosts Buffer of ghosts sent/forwarded across a face
    \param face_pitch Offsets of different faces in face ghost buffer
    \param d_edge_ghosts Buffer of ghosts sent/forwarded over an edge
    \param edge_pitch Offsets of different edges in edge ghost buffer
    \param d_recv_ghosts Buffer of ghosts received for the local box
    \param d_pos Array of particle positions
    \param d_vel Array of particle velocities
    \param d_orientation Array of particle orientations
    \param global_box Global simulation box
    \param sz Size of update element
    \param exch_pos >0 if exchanging positions
    \param exch_vel >0 if exchanging positions
    \param exch_orientation >0 if exchanging positions
*/
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
                                Scalar4 *d_vel,
                                Scalar4 *d_orientation,
                                const BoxDim& shifted_global_box,
                                unsigned int sz,
                                unsigned char exch_pos,
                                unsigned char exch_vel,
                                unsigned char exch_orientation)
    {
    unsigned int block_size = 512;
    gpu_exchange_ghosts_unpack_kernel<true><<<n_tot_recv_ghosts/block_size+1, block_size>>>
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
                                                                           d_vel,
                                                                           d_orientation,
                                                                           NULL,
                                                                           NULL,
                                                                           shifted_global_box,
                                                                           sz,
                                                                           exch_pos,
                                                                           exch_vel,
                                                                           0,
                                                                           0,
                                                                           exch_orientation);
    }


#endif
