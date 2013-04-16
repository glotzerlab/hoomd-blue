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

// Maintainer: ndtrung

/*! \file RigidData.cc
    \brief Defines RigidData and related classes.
*/

#ifdef WIN32
#pragma warning( push )
#pragma warning( disable : 4103 4244 )
#endif

#include <cassert>
#include <math.h>
#include <boost/python.hpp>
#include <algorithm>
using namespace boost::python;

#include "RigidData.h"
#include "QuaternionMath.h"

using namespace boost;
using namespace std;

// Maximum number of iterations for Jacobi rotations
#define MAXJACOBI 50
// Maximum value macro
#define MAX(A,B) ((A) > (B)) ? (A) : (B)

/*! \param particle_data ParticleData this use in initializing this RigidData

    \pre \a particle_data has been completeley initialized with all arrays filled out
    \post All data members in RigidData are completely initialized from the given info in \a particle_data
*/
RigidData::RigidData(boost::shared_ptr<ParticleData> particle_data)
    : m_pdata(particle_data), m_n_bodies(0), m_ndof(0)
    {
    // leave arrays initialized to NULL. There are currently 0 bodies and their
    // initialization is delayed because we cannot reasonably determine when that initialization
    // must be done

    // connect the sort signal
    m_sort_connection = m_pdata->connectParticleSort(bind(&RigidData::recalcIndices, this));

    // save the execution configuration
    m_exec_conf = m_pdata->getExecConf();
    }

RigidData::~RigidData()
    {
    m_sort_connection.disconnect();
    }


/*! \pre m_body_size has been filled with values
    \pre m_particle_tags has been filled with values
    \pre m_particle_indices has been allocated
    \post m_particle_indices is updated to match the current sorting of the particle data
*/
void RigidData::recalcIndices()
    {
    if (m_n_bodies == 0)
        return;
        
    // sanity check
    assert(m_pdata);
    assert(!m_particle_tags.isNull());
    assert(!m_particle_indices.isNull());
//  assert(m_n_bodies <= m_particle_tags.getPitch());
//  assert(m_n_bodies <= m_particle_indices.getPitch());
//  printf("m_n_bodies = %d; particle tags pitch = %d\n", m_n_bodies, m_particle_tags.getPitch());
     
    assert(m_n_bodies == m_body_size.getNumElements());
    
    // get the particle data
    ArrayHandle< unsigned int > h_rtag(m_pdata->getRTags(), access_location::host, access_mode::read);

    // get all the rigid data we need
    ArrayHandle<unsigned int> tags(m_particle_tags, access_location::host, access_mode::read);
    unsigned int tags_pitch = m_particle_tags.getPitch();
    
    ArrayHandle<unsigned int> indices(m_particle_indices, access_location::host, access_mode::readwrite);
    unsigned int indices_pitch = m_particle_indices.getPitch();
    ArrayHandle<unsigned int> rigid_particle_indices(m_rigid_particle_indices, access_location::host, access_mode::readwrite);
    
    
    ArrayHandle<unsigned int> body_size(m_body_size, access_location::host, access_mode::read);
    ArrayHandle<unsigned int> h_particle_offset(m_particle_offset, access_location::host, access_mode::readwrite);
    
    // for each body
    unsigned int ridx = 0;
    for (unsigned int body = 0; body < m_n_bodies; body++)
        {
        // for each particle in this body
        unsigned int len = body_size.data[body];
        assert(body <= m_particle_tags.getHeight() && body <= m_particle_indices.getHeight());
        assert(len <= tags_pitch && len <= indices_pitch);
        
        for (unsigned int i = 0; i < len; i++)
            {
            // translate the tag to the current index
            unsigned int tag = tags.data[body*tags_pitch + i];
            unsigned int pidx = h_rtag.data[tag];
            indices.data[body*indices_pitch + i] = pidx;
            h_particle_offset.data[pidx] = i;
            
            rigid_particle_indices.data[ridx++] = pidx;
            }
        }
        
    #ifdef ENABLE_CUDA
    //Sort them so they are ordered
    sort(rigid_particle_indices.data, rigid_particle_indices.data + ridx); 
    #endif
    }

//! Internal single use matrix multiply
inline static void mat_multiply(Scalar a[3][3], Scalar b[3][3], Scalar c[3][3])
    {
    c[0][0] = a[0][0] * b[0][0] + a[0][1] * b[1][0] + a[0][2] * b[2][0];
    c[0][1] = a[0][0] * b[0][1] + a[0][1] * b[1][1] + a[0][2] * b[2][1];
    c[0][2] = a[0][0] * b[0][2] + a[0][1] * b[1][2] + a[0][2] * b[2][2];
    
    c[1][0] = a[1][0] * b[0][0] + a[1][1] * b[1][0] + a[1][2] * b[2][0];
    c[1][1] = a[1][0] * b[0][1] + a[1][1] * b[1][1] + a[1][2] * b[2][1];
    c[1][2] = a[1][0] * b[0][2] + a[1][1] * b[1][2] + a[1][2] * b[2][2];
    
    c[2][0] = a[2][0] * b[0][0] + a[2][1] * b[1][0] + a[2][2] * b[2][0];
    c[2][1] = a[2][0] * b[0][1] + a[2][1] * b[1][1] + a[2][2] * b[2][1];
    c[2][2] = a[2][0] * b[0][2] + a[2][1] * b[1][2] + a[2][2] * b[2][2];
    }

/*! \pre all data members have been allocated
    \post all data members are initialized with data from the particle data
*/
void RigidData::initializeData()
    {
    
    // get the particle data
    ArrayHandle< unsigned int > h_body(m_pdata->getBodies(), access_location::host, access_mode::read);

    ArrayHandle<Scalar4> h_p_orientation(m_pdata->getOrientationArray(), access_location::host, access_mode::read);
    BoxDim box = m_pdata->getBox();
    
    // determine the number of rigid bodies
    unsigned int maxbody = 0;
    unsigned int minbody = NO_BODY;
    bool found_body = false;
    unsigned int nparticles = m_pdata->getN();
    for (unsigned int j = 0; j < nparticles; j++)
        {
        if (h_body.data[j] != NO_BODY)
            {
            found_body = true;
            if (maxbody < h_body.data[j])
                maxbody = h_body.data[j];
            if (minbody > h_body.data[j])
                minbody = h_body.data[j];
            }
        }
    
    if (found_body)
        {
        m_n_bodies = maxbody + 1;   // h_body.data[j] is numbered from 0
        if (minbody != 0)
            {
            m_exec_conf->msg->error() << "rigid data: Body indices do not start at 0\n";
            throw runtime_error("Error initializing rigid data");
            }
        }
    else
        m_n_bodies = 0;
        
    if (m_n_bodies <= 0)
        {
        return;
        }
        
    // allocate nbodies-size arrays
    GPUArray<unsigned int> body_dof(m_n_bodies, m_pdata->getExecConf());
    GPUArray<Scalar> body_mass(m_n_bodies, m_pdata->getExecConf());
    GPUArray<unsigned int> body_size(m_n_bodies, m_pdata->getExecConf());
    GPUArray<Scalar4> moment_inertia(m_n_bodies, m_pdata->getExecConf());
    GPUArray<Scalar4> orientation(m_n_bodies, m_pdata->getExecConf());
    GPUArray<Scalar4> ex_space(m_n_bodies, m_pdata->getExecConf());
    GPUArray<Scalar4> ey_space(m_n_bodies, m_pdata->getExecConf());
    GPUArray<Scalar4> ez_space(m_n_bodies, m_pdata->getExecConf());
    GPUArray<int3> body_image(m_n_bodies, m_pdata->getExecConf());
    GPUArray<Scalar4> conjqm_alloc(m_n_bodies, m_pdata->getExecConf());

    GPUArray<Scalar4> com(m_n_bodies, m_pdata->getExecConf());
    GPUArray<Scalar4> vel(m_n_bodies, m_pdata->getExecConf());
    GPUArray<Scalar4> angmom(m_n_bodies, m_pdata->getExecConf());
    GPUArray<Scalar4> angvel(m_n_bodies, m_pdata->getExecConf());
    GPUArray<Scalar4> force(m_n_bodies, m_pdata->getExecConf());
    GPUArray<Scalar4> torque(m_n_bodies, m_pdata->getExecConf());
    
    GPUArray<unsigned int> particle_offset(m_pdata->getN(), m_pdata->getExecConf());
    
    m_body_dof.swap(body_dof);
    m_body_mass.swap(body_mass);
    m_body_size.swap(body_size);
    m_moment_inertia.swap(moment_inertia);
    m_orientation.swap(orientation);
    m_ex_space.swap(ex_space);
    m_ey_space.swap(ey_space);
    m_ez_space.swap(ez_space);
    m_body_image.swap(body_image);
    m_conjqm.swap(conjqm_alloc);

    m_com.swap(com);
    m_vel.swap(vel);
    m_angmom.swap(angmom);
    m_angvel.swap(angvel);
    m_force.swap(force);
    m_torque.swap(torque);
    
    m_particle_offset.swap(particle_offset);
    
    {
    // determine the largest size of rigid bodies (nmax)
    ArrayHandle<unsigned int> body_size_handle(m_body_size, access_location::host, access_mode::readwrite);
    for (unsigned int body = 0; body < m_n_bodies; body++)
        body_size_handle.data[body] = 0;
        
    for (unsigned int j = 0; j < nparticles; j++)
        {
        unsigned int body = h_body.data[j];
        if (body != NO_BODY)
            body_size_handle.data[body]++;
        }
        
    // determine the maximum number of particles in a rigid body
    m_nmax = 0;
    for (unsigned int body = 0; body < m_n_bodies; body++)
        if (m_nmax < body_size_handle.data[body])
            m_nmax = body_size_handle.data[body];
    
    // determine body_mass, inertia tensor, com and vel
    GPUArray<Scalar> inertia(6, m_n_bodies, m_pdata->getExecConf()); // the inertia tensor is symmetric, therefore we only need to store 6 elements
    ArrayHandle<Scalar> inertia_handle(inertia, access_location::host, access_mode::readwrite);
    unsigned int inertia_pitch = inertia.getPitch();
    
    ArrayHandle<unsigned int> body_dof_handle(m_body_dof, access_location::host, access_mode::readwrite);
    ArrayHandle<Scalar> body_mass_handle(m_body_mass, access_location::host, access_mode::readwrite);
    ArrayHandle<Scalar4> moment_inertia_handle(m_moment_inertia, access_location::host, access_mode::readwrite);
    ArrayHandle<Scalar4> orientation_handle(m_orientation, access_location::host, access_mode::readwrite);
    ArrayHandle<Scalar4> ex_space_handle(m_ex_space, access_location::host, access_mode::readwrite);
    ArrayHandle<Scalar4> ey_space_handle(m_ey_space, access_location::host, access_mode::readwrite);
    ArrayHandle<Scalar4> ez_space_handle(m_ez_space, access_location::host, access_mode::readwrite);
    ArrayHandle<int3> body_image_handle(m_body_image, access_location::host, access_mode::readwrite);
    ArrayHandle<Scalar4> com_handle(m_com, access_location::host, access_mode::readwrite);
    
    for (unsigned int body = 0; body < m_n_bodies; body++)
        {
        body_mass_handle.data[body] = 0.0;
        com_handle.data[body].x = 0.0;
        com_handle.data[body].y = 0.0;
        com_handle.data[body].z = 0.0;
        
        inertia_handle.data[inertia_pitch * body] = 0.0;
        inertia_handle.data[inertia_pitch * body + 1] = 0.0;
        inertia_handle.data[inertia_pitch * body + 2] = 0.0;
        inertia_handle.data[inertia_pitch * body + 3] = 0.0;
        inertia_handle.data[inertia_pitch * body + 4] = 0.0;
        inertia_handle.data[inertia_pitch * body + 5] = 0.0;
        }
    
    // determine a nominal box image vector for each body
    // this is done so that bodies may be "unwrapped" from around the box dimensions in a numerically
    // stable way by bringing all particles unwrapped coords to being at most slightly outside of the box.
    std::vector<int3> nominal_body_image(m_n_bodies);

    ArrayHandle< int3 > h_image(m_pdata->getImages(), access_location::host, access_mode::read);
    ArrayHandle< Scalar4 > h_pos(m_pdata->getPositions(), access_location::host, access_mode::read);
    for (unsigned int j = 0; j < nparticles; j++)
        {
        unsigned int body = h_body.data[j];
        if (body != NO_BODY)
            nominal_body_image[body] = h_image.data[j];
        }
    
    // compute the center of mass for each body by summing up mass * \vec{r} for each particle in the body    
    ArrayHandle< Scalar4 > h_vel(m_pdata->getVelocities(), access_location::host, access_mode::read);
    for (unsigned int j = 0; j < m_pdata->getN(); j++)
        {
        if (h_body.data[j] == NO_BODY) continue;
        
        unsigned int body = h_body.data[j];
        Scalar mass_one = h_vel.data[j].w;
        body_mass_handle.data[body] += mass_one;
        // unwrap all particles in a body to the same image
        int3 shift = make_int3(h_image.data[j].x - nominal_body_image[body].x,
                               h_image.data[j].y - nominal_body_image[body].y,
                               h_image.data[j].z - nominal_body_image[body].z);
        Scalar3 wrapped = make_scalar3(h_pos.data[j].x, h_pos.data[j].y, h_pos.data[j].z);
        Scalar3 unwrapped = box.shift(wrapped, shift);
        
        com_handle.data[body].x += mass_one * unwrapped.x;
        com_handle.data[body].y += mass_one * unwrapped.y;
        com_handle.data[body].z += mass_one * unwrapped.z;
        }

    // complete the COM calculation by dividing by the mass of the body
    // for the moment, this is left in nominal unwrapped coordinates (it may be slightly outside the box) to enable
    // computation of the moment of inertia. This will be corrected after the moment of inertia is computed.
    for (unsigned int body = 0; body < m_n_bodies; body++)
        {
        Scalar mass_body = body_mass_handle.data[body];
        com_handle.data[body].x /= mass_body;
        com_handle.data[body].y /= mass_body;
        com_handle.data[body].z /= mass_body;
        
        body_image_handle.data[body].x = nominal_body_image[body].x;
        body_image_handle.data[body].y = nominal_body_image[body].y;
        body_image_handle.data[body].z = nominal_body_image[body].z;
        }
    
    Scalar4 porientation;
    Scalar4 ex, ey, ez;
    InertiaTensor pinertia_tensor;
    Scalar rot_mat[3][3], rot_mat_trans[3][3], Ibody[3][3], Ispace[3][3], tmp[3][3];

    ArrayHandle< unsigned int > h_tag(m_pdata->getTags(), access_location::host, access_mode::read);

    // determine the inertia tensor then diagonalize it
    for (unsigned int j = 0; j < m_pdata->getN(); j++)
        {
        if (h_body.data[j] == NO_BODY) continue;
        
        unsigned int body = h_body.data[j];
        Scalar mass_one = h_vel.data[j].w;
        unsigned int tag = h_tag.data[j];
        
        // unwrap all particles in a body to the same image
        int3 shift = make_int3(h_image.data[j].x - nominal_body_image[body].x,
                               h_image.data[j].y - nominal_body_image[body].y,
                               h_image.data[j].z - nominal_body_image[body].z);
        Scalar3 wrapped = make_scalar3(h_pos.data[j].x, h_pos.data[j].y, h_pos.data[j].z);
        Scalar3 unwrapped = box.shift(wrapped, shift);
        
        Scalar dx = unwrapped.x - com_handle.data[body].x;
        Scalar dy = unwrapped.y - com_handle.data[body].y;
        Scalar dz = unwrapped.z - com_handle.data[body].z;
            
        inertia_handle.data[inertia_pitch * body + 0] += mass_one * (dy * dy + dz * dz);
        inertia_handle.data[inertia_pitch * body + 1] += mass_one * (dz * dz + dx * dx);
        inertia_handle.data[inertia_pitch * body + 2] += mass_one * (dx * dx + dy * dy);
        inertia_handle.data[inertia_pitch * body + 3] -= mass_one * dx * dy;
        inertia_handle.data[inertia_pitch * body + 4] -= mass_one * dy * dz;
        inertia_handle.data[inertia_pitch * body + 5] -= mass_one * dx * dz;
        
        // take into account the partile inertia moments
        // get the original particle orientation and inertia tensor from input
        porientation = h_p_orientation.data[j];
        pinertia_tensor = m_pdata->getInertiaTensor(tag);
        
        exyzFromQuaternion(porientation, ex, ey, ez);
        
        rot_mat[0][0] = rot_mat_trans[0][0] = ex.x;
        rot_mat[1][0] = rot_mat_trans[0][1] = ex.y;
        rot_mat[2][0] = rot_mat_trans[0][2] = ex.z;
        
        rot_mat[0][1] = rot_mat_trans[1][0] = ey.x;
        rot_mat[1][1] = rot_mat_trans[1][1] = ey.y;
        rot_mat[2][1] = rot_mat_trans[1][2] = ey.z;
        
        rot_mat[0][2] = rot_mat_trans[2][0] = ez.x;
        rot_mat[1][2] = rot_mat_trans[2][1] = ez.y;
        rot_mat[2][2] = rot_mat_trans[2][2] = ez.z;
        
        Ibody[0][0] = pinertia_tensor.components[0];
        Ibody[0][1] = Ibody[1][0] = pinertia_tensor.components[1];
        Ibody[0][2] = Ibody[2][0] = pinertia_tensor.components[2];
        Ibody[1][1] = pinertia_tensor.components[3];
        Ibody[1][2] = Ibody[2][1] = pinertia_tensor.components[4];
        Ibody[2][2] = pinertia_tensor.components[5];
        
        // convert the particle inertia tensor to the space fixed frame 
        mat_multiply(Ibody, rot_mat_trans, tmp);
        mat_multiply(rot_mat, tmp, Ispace);
        
        inertia_handle.data[inertia_pitch * body + 0] += Ispace[0][0];
        inertia_handle.data[inertia_pitch * body + 1] += Ispace[1][1];
        inertia_handle.data[inertia_pitch * body + 2] += Ispace[2][2];
        inertia_handle.data[inertia_pitch * body + 3] += Ispace[0][1];
        inertia_handle.data[inertia_pitch * body + 4] += Ispace[1][2];
        inertia_handle.data[inertia_pitch * body + 5] += Ispace[0][2];
        }
    
    // allocate temporary arrays: revision needed!
    Scalar **matrix, *evalues, **evectors;
    matrix = new Scalar*[3];
    evectors = new Scalar*[3];
    evalues = new Scalar[3];
    for (unsigned int j = 0; j < 3; j++)
        {
        matrix[j] = new Scalar[3];
        evectors[j] = new Scalar[3];
        }
        
    unsigned int dof_one;
    for (unsigned int body = 0; body < m_n_bodies; body++)
        {
        matrix[0][0] = inertia_handle.data[inertia_pitch * body + 0];
        matrix[1][1] = inertia_handle.data[inertia_pitch * body + 1];
        matrix[2][2] = inertia_handle.data[inertia_pitch * body + 2];
        matrix[0][1] = matrix[1][0] = inertia_handle.data[inertia_pitch * body + 3];
        matrix[1][2] = matrix[2][1] = inertia_handle.data[inertia_pitch * body + 4];
        matrix[2][0] = matrix[0][2] = inertia_handle.data[inertia_pitch * body + 5];
        
        int error = diagonalize(matrix, evalues, evectors);
        if (error) 
            m_exec_conf->msg->warning() << "rigid data: Insufficient Jacobi iterations for diagonalization!\n";
        
        // obtain the moment inertia from eigen values
        moment_inertia_handle.data[body].x = evalues[0];
        moment_inertia_handle.data[body].y = evalues[1];
        moment_inertia_handle.data[body].z = evalues[2];
            
        // set tiny moment of inertia component to be zero, count the number of degrees of freedom
        // the actual DOF for temperature calculation is computed in the integrator (TwoStepNVERigid)
        // where the number of system dimensions is available
        // The counting below is only for book-keeping
        dof_one = 6;
    
        Scalar max = MAX(moment_inertia_handle.data[body].x, moment_inertia_handle.data[body].y);
        max = MAX(max, moment_inertia_handle.data[body].z);
        
        if (moment_inertia_handle.data[body].x < EPSILON * max)
            {
            dof_one--;
            moment_inertia_handle.data[body].x = Scalar(0.0);
            }
            
        if (moment_inertia_handle.data[body].y < EPSILON * max)
            {
            dof_one--;
            moment_inertia_handle.data[body].y = Scalar(0.0);
            }
            
        if (moment_inertia_handle.data[body].z < EPSILON * max)
            {
            dof_one--;
            moment_inertia_handle.data[body].z = Scalar(0.0);
            }
        
        body_dof_handle.data[body] = dof_one;    
        m_ndof += dof_one;
        
        // obtain the principle axes from eigen vectors
        ex_space_handle.data[body].x = evectors[0][0];
        ex_space_handle.data[body].y = evectors[1][0];
        ex_space_handle.data[body].z = evectors[2][0];
        
        ey_space_handle.data[body].x = evectors[0][1];
        ey_space_handle.data[body].y = evectors[1][1];
        ey_space_handle.data[body].z = evectors[2][1];
        
        ez_space_handle.data[body].x = evectors[0][2];
        ez_space_handle.data[body].y = evectors[1][2];
        ez_space_handle.data[body].z = evectors[2][2];
        
        // create the initial quaternion from the new body frame
        quaternionFromExyz(ex_space_handle.data[body], ey_space_handle.data[body], ez_space_handle.data[body],
                           orientation_handle.data[body]);
        }
        
    // deallocate temporary memory
    delete [] evalues;
    
    for (unsigned int j = 0; j < 3; j++)
        {
        delete [] matrix[j];
        delete [] evectors[j];
        }
        
    delete [] evectors;
    delete [] matrix;
    
        
    // allocate nmax by m_n_bodies arrays, swap to member variables then use array handles to access
    GPUArray<unsigned int> particle_tags(m_nmax, m_n_bodies,  m_pdata->getExecConf());
    m_particle_tags.swap(particle_tags);
    ArrayHandle<unsigned int> particle_tags_handle(m_particle_tags, access_location::host, access_mode::readwrite);
    unsigned int particle_tags_pitch = m_particle_tags.getPitch();
    
    GPUArray<unsigned int> particle_indices(m_nmax, m_n_bodies, m_pdata->getExecConf());
    m_particle_indices.swap(particle_indices);
    ArrayHandle<unsigned int> particle_indices_handle(m_particle_indices, access_location::host, access_mode::readwrite);
    unsigned int particle_indices_pitch = m_particle_indices.getPitch();
    
    for (unsigned int j = 0; j < m_n_bodies; j++) 
        for (unsigned int local = 0; local < particle_indices_pitch; local++)
            particle_indices_handle.data[j * particle_indices_pitch + local] = NO_INDEX; // initialize with a sentinel value
    
    GPUArray<Scalar4> particle_pos(m_nmax, m_n_bodies, m_pdata->getExecConf());
    m_particle_pos.swap(particle_pos);
    ArrayHandle<Scalar4> particle_pos_handle(m_particle_pos, access_location::host, access_mode::readwrite);
    unsigned int particle_pos_pitch = m_particle_pos.getPitch();
    
    GPUArray<Scalar4> particle_orientation(m_nmax, m_n_bodies, m_pdata->getExecConf());
    m_particle_orientation.swap(particle_orientation);
    ArrayHandle<Scalar4> h_particle_orientation(m_particle_orientation, access_location::host, access_mode::readwrite);

    GPUArray<unsigned int> local_indices(m_n_bodies, m_pdata->getExecConf());
    ArrayHandle<unsigned int> local_indices_handle(local_indices, access_location::host, access_mode::readwrite);
    for (unsigned int body = 0; body < m_n_bodies; body++)
        local_indices_handle.data[body] = 0;
    
    // Now set the m_nmax according to the actual pitches to avoid dublicating rounding up (e.g. if m_nmax is rounded up to 16 here, 
    // then in the GPUArray constructor the pitch is rounded up once more to be 32.
    m_nmax = particle_tags_pitch;
    
    //tally up how many particles belong to rigid bodies
    unsigned int rigid_particle_count = 0;
    
    // determine the particle indices and particle tags
    for (unsigned int j = 0; j < m_pdata->getN(); j++)
        {
        if (h_body.data[j] == NO_BODY) continue;
        
        rigid_particle_count++;
        
        // get the corresponding body
        unsigned int body = h_body.data[j];
        // get the current index in the body
        unsigned int current_localidx = local_indices_handle.data[body];
        // set the particle index to be this value
        particle_indices_handle.data[body * particle_indices_pitch + current_localidx] = j;
        // set the particle tag to be the tag of this particle
        particle_tags_handle.data[body * particle_tags_pitch + current_localidx] = h_tag.data[j];
        
        // determine the particle position in the body frame
        // with ex_space, ey_space and ex_space vectors computed from the diagonalization
        // unwrap all particles in a body to the same image
        int3 shift = make_int3(h_image.data[j].x - nominal_body_image[body].x,
                               h_image.data[j].y - nominal_body_image[body].y,
                               h_image.data[j].z - nominal_body_image[body].z);
        Scalar3 wrapped = make_scalar3(h_pos.data[j].x, h_pos.data[j].y, h_pos.data[j].z);
        Scalar3 unwrapped = box.shift(wrapped, shift);
        
        Scalar dx = unwrapped.x - com_handle.data[body].x;
        Scalar dy = unwrapped.y - com_handle.data[body].y;
        Scalar dz = unwrapped.z - com_handle.data[body].z;
                    
        unsigned int idx = body * particle_pos_pitch + current_localidx;
        particle_pos_handle.data[idx].x = dx * ex_space_handle.data[body].x + dy * ex_space_handle.data[body].y +
                dz * ex_space_handle.data[body].z;
        particle_pos_handle.data[idx].y = dx * ey_space_handle.data[body].x + dy * ey_space_handle.data[body].y +
                dz * ey_space_handle.data[body].z;
        particle_pos_handle.data[idx].z = dx * ez_space_handle.data[body].x + dy * ez_space_handle.data[body].y +
                dz * ez_space_handle.data[body].z;
        
        // initialize h_particle_orientation.data[idx] here from the initial particle orientation. This means
        // reading the intial particle orientation from ParticleData and translating it backwards into the body frame
        Scalar4 qc;
        quatconj(orientation_handle.data[body], qc);
        
        porientation = h_p_orientation.data[j];
        quatquat(qc, porientation, h_particle_orientation.data[idx]);
        normalize(h_particle_orientation.data[idx]);
        
        // increment the current index by one
        local_indices_handle.data[body]++;
        }

    // now that all computations using nominally unwrapped coordinates are done, put the COM into the simulation box
    for (unsigned int body = 0; body < m_n_bodies; body++)
        {
        box.wrap(com_handle.data[body], body_image_handle.data[body]);
        }
    
    //initialize rigid_particle_indices
    GPUArray<unsigned int> rigid_particle_indices(rigid_particle_count, m_pdata->getExecConf());
    m_rigid_particle_indices.swap(rigid_particle_indices);
    m_num_particles = rigid_particle_count;

    GPUArray<Scalar4> particle_oldpos(m_pdata->getN(), m_pdata->getExecConf());
    m_particle_oldpos.swap(particle_oldpos);
    
    GPUArray<Scalar4> particle_oldvel(m_pdata->getN(), m_pdata->getExecConf());
    m_particle_oldvel.swap(particle_oldvel);
                                          
    // release particle data for later access
    }   // out of scope for handles
        
    // finish up by initializing the indices
    recalcIndices();
    }
 
/* Set position and velocity of constituent particles in rigid bodies in the 1st or second half of integration
    based on the body center of mass and particle relative position in each body frame.
    \param set_x if true, positions are updated too.  Else just velocities.
   
*/       
void RigidData::setRV(bool set_x)
   {
    #ifdef ENABLE_CUDA
        if (m_pdata->getExecConf()->exec_mode == ExecutionConfiguration::GPU)
            setRVGPU(true);
        else
    #endif
         setRVCPU(true);
   }    

    
/* Set position and velocity of constituent particles in rigid bodies in the 1st or second half of integration on the CPU
    based on the body center of mass and particle relative position in each body frame.
    \param set_x if true, positions are updated too.  Else just velocities.
*/

void RigidData::setRVCPU(bool set_x)
    {
    // get box
    const BoxDim& box = m_pdata->getBox();

    // access to the force
    const GPUArray< Scalar4 >& net_force = m_pdata->getNetForce();
    ArrayHandle<Scalar4> h_net_force(net_force, access_location::host, access_mode::read);
    
    // rigid body handles
    ArrayHandle<unsigned int> body_size_handle(m_body_size, access_location::host, access_mode::read);
    ArrayHandle<Scalar4> com(m_com, access_location::host, access_mode::read);
    ArrayHandle<Scalar4> vel_handle(m_vel, access_location::host, access_mode::read);
    ArrayHandle<Scalar4> angvel_handle(m_angvel, access_location::host, access_mode::read);
    ArrayHandle<Scalar4> orientation_handle(m_orientation, access_location::host, access_mode::read);
    ArrayHandle<int3> body_image_handle(m_body_image, access_location::host, access_mode::read);
        
    ArrayHandle<unsigned int> particle_indices_handle(m_particle_indices, access_location::host, access_mode::read);
    unsigned int indices_pitch = m_particle_indices.getPitch();
    ArrayHandle<Scalar4> particle_pos_handle(m_particle_pos, access_location::host, access_mode::read);
    unsigned int particle_pos_pitch = m_particle_pos.getPitch();
    ArrayHandle<Scalar4> particle_orientation(m_particle_orientation, access_location::host, access_mode::read);

    // access the particle data arrays
    ArrayHandle<Scalar4> h_pos(m_pdata->getPositions(), access_location::host, access_mode::readwrite);
    ArrayHandle<Scalar4> h_vel(m_pdata->getVelocities(), access_location::host, access_mode::readwrite);
    ArrayHandle<int3> h_image(m_pdata->getImages(), access_location::host, access_mode::readwrite);
    ArrayHandle<Scalar4> h_p_orientation(m_pdata->getOrientationArray(), access_location::host, access_mode::readwrite);
    
    Scalar4 ex_space, ey_space, ez_space;
    
    // for each body of all the bodies
    for (unsigned int body = 0; body < m_n_bodies; body++)
        {
        exyzFromQuaternion(orientation_handle.data[body], ex_space, ey_space, ez_space);
        
        unsigned int len = body_size_handle.data[body];
        // for each particle
        for (unsigned int j = 0; j < len; j++)
            {
            // get the actual index of particle in the particle arrays
            unsigned int pidx = particle_indices_handle.data[body * indices_pitch + j];
            // get the index of particle in the current rigid body in the particle_pos array
            unsigned int localidx = body * particle_pos_pitch + j;
            
            // project the position in the body frame to the space frame: xr = rotation_matrix * particle_pos
            Scalar xr = ex_space.x * particle_pos_handle.data[localidx].x
                        + ey_space.x * particle_pos_handle.data[localidx].y
                        + ez_space.x * particle_pos_handle.data[localidx].z;
            Scalar yr = ex_space.y * particle_pos_handle.data[localidx].x
                        + ey_space.y * particle_pos_handle.data[localidx].y
                        + ez_space.y * particle_pos_handle.data[localidx].z;
            Scalar zr = ex_space.z * particle_pos_handle.data[localidx].x
                        + ey_space.z * particle_pos_handle.data[localidx].y
                        + ez_space.z * particle_pos_handle.data[localidx].z;
                        
            if (set_x) 
                {
                // x_particle = x_com + xr
                Scalar3 pos = make_scalar3(com.data[body].x + xr,
                                           com.data[body].y + yr,
                                           com.data[body].z + zr);
                
                // adjust particle images based on body images
                h_image.data[pidx] = body_image_handle.data[body];
                
                box.wrap(pos, h_image.data[pidx]);
                h_pos.data[pidx].x = pos.x;
                h_pos.data[pidx].y = pos.y;
                h_pos.data[pidx].z = pos.z;

                // update the particle orientation: q_i = quat[body] * particle_quat
                Scalar4 porientation; 
                quatquat(orientation_handle.data[body], particle_orientation.data[localidx], porientation);
                normalize(porientation);
                h_p_orientation.data[pidx] = porientation;
                }
            
            // v_particle = v_com + angvel x xr
            h_vel.data[pidx].x = vel_handle.data[body].x + angvel_handle.data[body].y * zr - angvel_handle.data[body].z * yr;
            h_vel.data[pidx].y = vel_handle.data[body].y + angvel_handle.data[body].z * xr - angvel_handle.data[body].x * zr;
            h_vel.data[pidx].z = vel_handle.data[body].z + angvel_handle.data[body].x * yr - angvel_handle.data[body].y * xr;
            }
        }
        
    }

/* Helper GPU function to set position and velocity of constituent particles in rigid bodies in the 1st or second half of integration
    based on the body center of mass and particle relative position in each body frame.
    \param set_x if true, positions are updated too.  Else just velocities.
   
*/
#ifdef ENABLE_CUDA
void RigidData::setRVGPU(bool set_x)
    {
        
    // sanity check
    if (m_n_bodies <= 0)
        return;

    ArrayHandle<Scalar4> d_pos(m_pdata->getPositions(), access_location::device, access_mode::readwrite);
    ArrayHandle<Scalar4> d_vel(m_pdata->getVelocities(), access_location::device, access_mode::readwrite);
    ArrayHandle<int3> d_image(m_pdata->getImages(), access_location::device, access_mode::readwrite);
    ArrayHandle<unsigned int> d_body(m_pdata->getBodies(), access_location::device, access_mode::read);

    // Acquire handles
    ArrayHandle<unsigned int> rigid_particle_indices(m_rigid_particle_indices, access_location::device, access_mode::read);
    
    // access all the needed data
    ArrayHandle<Scalar4> d_porientation(m_pdata->getOrientationArray(),access_location::device,access_mode::readwrite);
    
    BoxDim box = m_pdata->getBox();
    
    ArrayHandle<Scalar> body_mass_handle(m_body_mass, access_location::device, access_mode::read);
    ArrayHandle<Scalar4> moment_inertia_handle(m_moment_inertia, access_location::device, access_mode::read);
    ArrayHandle<Scalar4> com_handle(m_com, access_location::device, access_mode::readwrite);
    ArrayHandle<Scalar4> vel_handle(m_vel, access_location::device, access_mode::readwrite);
    ArrayHandle<Scalar4> angvel_handle(m_angvel, access_location::device, access_mode::readwrite);
    ArrayHandle<Scalar4> angmom_handle(m_angmom, access_location::device, access_mode::readwrite);
    ArrayHandle<Scalar4> orientation_handle(m_orientation, access_location::device, access_mode::readwrite);
    ArrayHandle<int3> body_image_handle(m_body_image, access_location::device, access_mode::readwrite);
    ArrayHandle<Scalar4> particle_pos_handle(m_particle_pos, access_location::device, access_mode::read);
    ArrayHandle<unsigned int> particle_indices_handle(m_particle_indices, access_location::device, access_mode::read);
    ArrayHandle<Scalar4> force_handle(m_force, access_location::device, access_mode::read);
    ArrayHandle<Scalar4> torque_handle(m_torque, access_location::device, access_mode::read);
    
    ArrayHandle<unsigned int> d_particle_offset(m_particle_offset, access_location::device, access_mode::read);
    ArrayHandle<Scalar4> d_particle_orientation(m_particle_orientation, access_location::device, access_mode::readwrite);


    // More data is filled in here than I use.  
    gpu_rigid_data_arrays d_rdata;
    d_rdata.n_bodies = m_n_bodies;
    d_rdata.n_group_bodies = m_n_bodies;
    d_rdata.nmax = m_nmax;
    d_rdata.local_beg = 0;
    d_rdata.local_num = m_n_bodies;
    
    d_rdata.body_mass = body_mass_handle.data;
    d_rdata.moment_inertia = moment_inertia_handle.data;
    d_rdata.com = com_handle.data;
    d_rdata.vel = vel_handle.data;
    d_rdata.angvel = angvel_handle.data;
    d_rdata.angmom = angmom_handle.data;
    d_rdata.orientation = orientation_handle.data;
    d_rdata.body_image = body_image_handle.data;
    d_rdata.particle_pos = particle_pos_handle.data;
    d_rdata.particle_indices = particle_indices_handle.data;
    d_rdata.force = force_handle.data;
    d_rdata.torque = torque_handle.data;
    d_rdata.particle_offset = d_particle_offset.data;
    d_rdata.particle_orientation = d_particle_orientation.data;
    
    
    gpu_rigid_setRV(               d_pos.data,
                                   d_vel.data,
                                   d_image.data,
                                   d_body.data,
                                   d_rdata,
                                   d_porientation.data,
                                   rigid_particle_indices.data,
                                   m_num_particles,
                                   box,
                                   set_x);    
                                       
        
    }
#endif    

/*! Compute eigenvalues and eigenvectors of 3x3 real symmetric matrix based on Jacobi rotations adapted from Numerical Recipes jacobi() function (LAMMPS)
    \param matrix Matrix to be diagonalized
    \param evalues Eigen-values obtained after diagonalized
    \param evectors Eigen-vectors obtained after diagonalized in columns

*/

int RigidData::diagonalize(Scalar **matrix, Scalar *evalues, Scalar **evectors)
    {
    int i,j,k;
    Scalar tresh, theta, tau, t, sm, s, h, g, c, b[3], z[3];
    
    for (i = 0; i < 3; i++)
        {
        for (j = 0; j < 3; j++) evectors[i][j] = 0.0;
        evectors[i][i] = 1.0;
        }
        
    for (i = 0; i < 3; i++)
        {
        b[i] = evalues[i] = matrix[i][i];
        z[i] = 0.0;
        }
        
    for (int iter = 1; iter <= MAXJACOBI; iter++)
        {
        sm = 0.0;
        for (i = 0; i < 2; i++)
            for (j = i+1; j < 3; j++)
                sm += fabs(matrix[i][j]);
                
        if (sm == 0.0) return 0;
        
        if (iter < 4) tresh = 0.2*sm/(3*3);
        else tresh = 0.0;
        
        for (i = 0; i < 2; i++)
            {
            for (j = i+1; j < 3; j++)
                {
                g = 100.0 * fabs(matrix[i][j]);
                if (iter > 4 && fabs(evalues[i]) + g == fabs(evalues[i])
                        && fabs(evalues[j]) + g == fabs(evalues[j]))
                    matrix[i][j] = 0.0;
                else if (fabs(matrix[i][j]) > tresh)
                    {
                    h = evalues[j]-evalues[i];
                    if (fabs(h)+g == fabs(h)) t = (matrix[i][j])/h;
                    else
                        {
                        theta = 0.5 * h / (matrix[i][j]);
                        t = 1.0/(fabs(theta)+sqrt(1.0+theta*theta));
                        if (theta < 0.0) t = -t;
                        }
                        
                    c = 1.0/sqrt(1.0+t*t);
                    s = t*c;
                    tau = s/(1.0+c);
                    h = t*matrix[i][j];
                    z[i] -= h;
                    z[j] += h;
                    evalues[i] -= h;
                    evalues[j] += h;
                    matrix[i][j] = 0.0;
                    for (k = 0; k < i; k++) rotate(matrix,k,i,k,j,s,tau);
                    for (k = i+1; k < j; k++) rotate(matrix,i,k,k,j,s,tau);
                    for (k = j+1; k < 3; k++) rotate(matrix,i,k,j,k,s,tau);
                    for (k = 0; k < 3; k++) rotate(evectors,k,i,k,j,s,tau);
                    }
                }
            }
            
        for (i = 0; i < 3; i++)
            {
            evalues[i] = b[i] += z[i];
            z[i] = 0.0;
            }
        }
        
    return 1;
    }

/*! Perform a single Jacobi rotation
    \param matrix Matrix to be diagonalized
    \param i 
    \param j 
    \param k
    \param l
    \param s
    \param tau
*/

void RigidData::rotate(Scalar **matrix, int i, int j, int k, int l, Scalar s, Scalar tau)
    {
    Scalar g = matrix[i][j];
    Scalar h = matrix[k][l];
    matrix[i][j] = g - s * (h + g * tau);
    matrix[k][l] = h + s * (g - h * tau);
    }

/*! Calculate the quaternion from three axes
    \param ex_space x-axis unit vector
    \param ey_space y-axis unit vector
    \param ez_space z-axis unit vector
    \param quat returned quaternion
*/

void RigidData::quaternionFromExyz(Scalar4 &ex_space, Scalar4 &ey_space, Scalar4 &ez_space, Scalar4 &quat)
    {
    
    // enforce 3 evectors as a right-handed coordinate system
    // flip 3rd evector if needed
    Scalar ez0, ez1, ez2; // Cross product of first two vectors
    ez0 = ex_space.y * ey_space.z - ex_space.z * ey_space.y;
    ez1 = ex_space.z * ey_space.x - ex_space.x * ey_space.z;
    ez2 = ex_space.x * ey_space.y - ex_space.y * ey_space.x;
    
    // then dot product with the third one
    if (ez0 * ez_space.x + ez1 * ez_space.y + ez2 * ez_space.z < 0.0)
        {
        ez_space.x = -ez_space.x;
        ez_space.y = -ez_space.y;
        ez_space.z = -ez_space.z;
        }
        
    // squares of quaternion components
    Scalar q0sq = 0.25 * (ex_space.x + ey_space.y + ez_space.z + 1.0);
    Scalar q1sq = q0sq - 0.5 * (ey_space.y + ez_space.z);
    Scalar q2sq = q0sq - 0.5 * (ex_space.x + ez_space.z);
    Scalar q3sq = q0sq - 0.5 * (ex_space.x + ey_space.y);
    
    // some component must be greater than 1/4 since they sum to 1
    // compute other components from it
    if (q0sq >= 0.25)
        {
        quat.x = sqrt(q0sq);
        quat.y = (ey_space.z - ez_space.y) / (4.0 * quat.x);
        quat.z = (ez_space.x - ex_space.z) / (4.0 * quat.x);
        quat.w = (ex_space.y - ey_space.x) / (4.0 * quat.x);
        }
    else if (q1sq >= 0.25)
        {
        quat.y = sqrt(q1sq);
        quat.x = (ey_space.z - ez_space.y) / (4.0 * quat.y);
        quat.z = (ey_space.x + ex_space.y) / (4.0 * quat.y);
        quat.w = (ex_space.z + ez_space.x) / (4.0 * quat.y);
        }
    else if (q2sq >= 0.25)
        {
        quat.z = sqrt(q2sq);
        quat.x = (ez_space.x - ex_space.z) / (4.0 * quat.z);
        quat.y = (ey_space.x + ex_space.y) / (4.0 * quat.z);
        quat.w = (ez_space.y + ey_space.z) / (4.0 * quat.z);
        }
    else if (q3sq >= 0.25)
        {
        quat.w = sqrt(q3sq);
        quat.x = (ex_space.y - ey_space.x) / (4.0 * quat.w);
        quat.y = (ez_space.x + ex_space.z) / (4.0 * quat.w);
        quat.z = (ez_space.y + ey_space.z) / (4.0 * quat.w);
        }
        
    // Normalize
    Scalar norm = 1.0 / sqrt(quat.x * quat.x + quat.y * quat.y + quat.z * quat.z + quat.w * quat.w);
    quat.x *= norm;
    quat.y *= norm;
    quat.z *= norm;
    quat.w *= norm;
    
    }

/*! Calculate the axes from quaternion
    \param quat returned quaternion
    \param ex_space x-axis unit vector
    \param ey_space y-axis unit vector
    \param ez_space z-axis unit vector
    
*/
void RigidData::exyzFromQuaternion(Scalar4 &quat, Scalar4 &ex_space, Scalar4 &ey_space, Scalar4 &ez_space)
    {
    ex_space.x = quat.x * quat.x + quat.y * quat.y - quat.z * quat.z - quat.w * quat.w;
    ex_space.y = 2.0 * (quat.y * quat.z + quat.x * quat.w);
    ex_space.z = 2.0 * (quat.y * quat.w - quat.x * quat.z);
    
    ey_space.x = 2.0 * (quat.y * quat.z - quat.x * quat.w);
    ey_space.y = quat.x * quat.x - quat.y * quat.y + quat.z * quat.z - quat.w * quat.w;
    ey_space.z = 2.0 * (quat.z * quat.w + quat.x * quat.y);
  
    ez_space.x = 2.0 * (quat.y * quat.w + quat.x * quat.z);
    ez_space.y = 2.0 * (quat.z * quat.w - quat.x * quat.y);
    ez_space.z = quat.x * quat.x - quat.y * quat.y - quat.z * quat.z + quat.w * quat.w;
    }
    
/*!
    \param body Body index to set angular momentum
    \param angmom Angular momentum
*/
void RigidData::setAngMom(unsigned int body, Scalar3 angmom)
    {
    if (body < 0 || body >= m_n_bodies) 
        {
        m_exec_conf->msg->error() << "Error setting angular momentum for body " << body << "\n";
        return;
        }
    
    ArrayHandle<Scalar4> angmom_handle(m_angmom, access_location::host, access_mode::readwrite);
    
    angmom_handle.data[body].x = angmom.x;
    angmom_handle.data[body].y = angmom.y;
    angmom_handle.data[body].z = angmom.z;
    angmom_handle.data[body].w = 0;
    }

/*! computeVirialCorrectionStart() must be called at the start of any time step update when there are rigid bodies
    present in the system and the virial needs to be computed. It only peforms part of the virial correction. The other
    part is completed by calling computeVirialCorrectionEnd()
*/
void RigidData::computeVirialCorrectionStart()
    {
    #ifdef ENABLE_CUDA
        if (m_pdata->getExecConf()->isCUDAEnabled())
            computeVirialCorrectionStartGPU();
        else
    #endif
        computeVirialCorrectionStartCPU();
    }
        

/*! computeVirialCorrectionEnd() must be called at the end of any time step update when there are rigid bodies
    present in the system and the virial needs to be computed. And computeVirialCorrectionStart() must have been
    called at the beginning of the step.
*/
void RigidData::computeVirialCorrectionEnd(Scalar deltaT)
    {
    #ifdef ENABLE_CUDA
        if (m_pdata->getExecConf()->isCUDAEnabled())
            computeVirialCorrectionEndGPU(deltaT);
        else
    #endif
        computeVirialCorrectionEndCPU(deltaT);
    }

/*! Helper function that perform the first part necessary to compute the rigid body virial correction on the CPU.
*/
void RigidData::computeVirialCorrectionStartCPU()
    {
    // get access to the particle data
    ArrayHandle<Scalar4> h_pos(m_pdata->getPositions(), access_location::host, access_mode::read);
    ArrayHandle<Scalar4> h_vel(m_pdata->getVelocities(), access_location::host, access_mode::read);
    ArrayHandle<Scalar4> h_oldpos(m_particle_oldpos, access_location::host, access_mode::overwrite);
    ArrayHandle<Scalar4> h_oldvel(m_particle_oldvel, access_location::host, access_mode::overwrite);    

    // loop through the particles and save the current position and velocity of each one
    for (unsigned int i = 0; i < m_pdata->getN(); i++)
        {
        h_oldpos.data[i] = make_scalar4(h_pos.data[i].x, h_pos.data[i].y, h_pos.data[i].z, 0.0);
        h_oldvel.data[i] = make_scalar4(h_vel.data[i].y, h_vel.data[i].y, h_vel.data[i].z, 0.0);
        }
    }

/*! Helper function that perform the second part necessary to compute the rigid body virial correction on the CPU.
*/
void RigidData::computeVirialCorrectionEndCPU(Scalar deltaT)
    {
    // get access to the particle data
    ArrayHandle<Scalar4> h_vel(m_pdata->getVelocities(), access_location::host, access_mode::read);
    ArrayHandle<unsigned int> h_body(m_pdata->getBodies(), access_location::host, access_mode::read);
    ArrayHandle<Scalar4> h_oldpos(m_particle_oldpos, access_location::host, access_mode::read);
    ArrayHandle<Scalar4> h_oldvel(m_particle_oldvel, access_location::host, access_mode::read); 

    ArrayHandle<Scalar> h_net_virial( m_pdata->getNetVirial(), access_location::host, access_mode::readwrite);
    unsigned int virial_pitch = m_pdata->getNetVirial().getPitch();
    ArrayHandle<Scalar4> h_net_force( m_pdata->getNetForce(), access_location::host, access_mode::read);

    // loop through all the particles and compute the virial correction to each one
    for (unsigned int i = 0; i < m_pdata->getN(); i++)
        {
        // only correct the virial for body particles
        if (h_body.data[i] != NO_BODY)
            {
            // calculate the virial from the position and velocity from the previous step
            Scalar mass = h_vel.data[i].w;
            Scalar4 old_vel = h_oldvel.data[i];
            Scalar4 old_pos = h_oldpos.data[i];
            Scalar3 fc;
            fc.x = mass * (h_vel.data[i].x - old_vel.x) / deltaT - h_net_force.data[i].x;
            fc.y = mass * (h_vel.data[i].y - old_vel.y) / deltaT - h_net_force.data[i].y;
            fc.z = mass * (h_vel.data[i].z - old_vel.z) / deltaT - h_net_force.data[i].z;

            h_net_virial.data[0*virial_pitch+i] += old_pos.x * fc.x;
            h_net_virial.data[1*virial_pitch+i] += old_pos.x * fc.y;
            h_net_virial.data[2*virial_pitch+i] += old_pos.x * fc.z;
            h_net_virial.data[3*virial_pitch+i] += old_pos.y * fc.y;
            h_net_virial.data[4*virial_pitch+i] += old_pos.y * fc.z;
            h_net_virial.data[5*virial_pitch+i] += old_pos.z * fc.z;
            }
        }
    }


#ifdef ENABLE_CUDA
/*! Helper function that perform the first part necessary to compute the rigid body virial correction on the GPU.
*/
void RigidData::computeVirialCorrectionStartGPU()
    {
    // get access to the particle data
    ArrayHandle<Scalar4> d_pos(m_pdata->getPositions(), access_location::device, access_mode::read);
    ArrayHandle<Scalar4> d_vel(m_pdata->getVelocities(), access_location::device, access_mode::read);
    ArrayHandle<Scalar4> d_oldpos(m_particle_oldpos, access_location::device, access_mode::overwrite);
    ArrayHandle<Scalar4> d_oldvel(m_particle_oldvel, access_location::device, access_mode::overwrite);    

    // copy the existing position and velocity over to the oldpos arrays
    cudaMemcpy(d_oldpos.data, d_pos.data, sizeof(Scalar4)*m_pdata->getN(), cudaMemcpyDeviceToDevice);
    cudaMemcpy(d_oldvel.data, d_vel.data, sizeof(Scalar4)*m_pdata->getN(), cudaMemcpyDeviceToDevice);

    CHECK_CUDA_ERROR();

    }

/*! Helper function that perform the second part necessary to compute the rigid body virial correction on the GPU.
*/
void RigidData::computeVirialCorrectionEndGPU(Scalar deltaT)
    {
    // get access to the particle data
    ArrayHandle<Scalar4> d_vel(m_pdata->getVelocities(), access_location::device, access_mode::read);
    ArrayHandle<unsigned int> d_body(m_pdata->getBodies(), access_location::device, access_mode::read);
    ArrayHandle<Scalar4> d_oldpos(m_particle_oldpos, access_location::device, access_mode::read);
    ArrayHandle<Scalar4> d_oldvel(m_particle_oldvel, access_location::device, access_mode::read); 

    ArrayHandle<Scalar> d_net_virial( m_pdata->getNetVirial(), access_location::device, access_mode::readwrite);
    unsigned int virial_pitch = m_pdata->getNetVirial().getPitch();
    ArrayHandle<Scalar4> d_net_force( m_pdata->getNetForce(), access_location::device, access_mode::read);

    gpu_compute_virial_correction_end(d_net_virial.data,
                                      virial_pitch,
                                      d_net_force.data,
                                      d_oldpos.data,
                                      d_oldvel.data,
                                      d_vel.data,
                                      d_body.data,
                                      deltaT,
                                      m_pdata->getN());

    CHECK_CUDA_ERROR();

    }
#endif

/*! \param snapshot SnapshotRigidData to initialize from
 */
void RigidData::initializeFromSnapshot(const SnapshotRigidData& snapshot)
    {
    // check that all fields in the snapshot have correct length
    if (m_exec_conf->getRank() == 0 && !snapshot.validate())
        {
        m_exec_conf->msg->error() << "init.*: inconsistent size of rigid body snapshot."
                                << std::endl << std::endl;
        throw std::runtime_error("Error initializing rigid bodies.");
        }

    ArrayHandle<Scalar4> h_com(getCOM(), access_location::host, access_mode::overwrite);
    ArrayHandle<Scalar4> h_vel(getVel(), access_location::host, access_mode::overwrite);
    ArrayHandle<Scalar4> h_angmom(getAngMom(), access_location::host, access_mode::overwrite);
    ArrayHandle<int3> h_body_image(getBodyImage(), access_location::host, access_mode::overwrite);
  
    // Error out if snapshot contains a different number of bodies
    if (getNumBodies() != snapshot.size)
        {
        m_exec_conf->msg->error() << "SnapshotRigidData has mismatched size." << std::endl << std::endl;
        throw std::runtime_error("Error initializing RigidData.");
        }

    // We don't need to restore force, torque and orientation because the setup will do the rest,
    // and simulation still resumes smoothly.
    // NOTE: this may not be true if re-initialized in the middle of the simulation
    unsigned int n_bodies = snapshot.size;
    for (unsigned int body = 0; body < n_bodies; body++)
        {
        h_com.data[body] = make_scalar4(snapshot.com[body].x, snapshot.com[body].y, snapshot.com[body].z,0.0);
        h_vel.data[body] = make_scalar4(snapshot.vel[body].x, snapshot.vel[body].y, snapshot.vel[body].z,0.0);
        h_angmom.data[body] = make_scalar4(snapshot.angmom[body].x, snapshot.angmom[body].y, snapshot.angmom[body].z, 0.0); 
        h_body_image.data[body] = snapshot.body_image[body];
        }
 
    }

/*! \param snapshot The snapshot to fill with the rigid body data
 */
void RigidData::takeSnapshot(SnapshotRigidData& snapshot) const
    {
    ArrayHandle<Scalar4> h_com(getCOM(), access_location::host, access_mode::read);
    ArrayHandle<Scalar4> h_vel(getVel(), access_location::host, access_mode::read);
    ArrayHandle<Scalar4> h_angmom(getAngMom(), access_location::host, access_mode::read);
    ArrayHandle<int3> h_body_image(getBodyImage(), access_location::host, access_mode::read);

    // allocate memory in snapshot
    unsigned int n_bodies = getNumBodies();
    snapshot.resize(n_bodies);

    for (unsigned int i = 0; i < n_bodies; ++i)
        {
        snapshot.com[i] = make_scalar3(h_com.data[i].x,h_com.data[i].y,h_com.data[i].z);
        snapshot.vel[i] = make_scalar3(h_vel.data[i].x,h_vel.data[i].y,h_vel.data[i].z);
        snapshot.angmom[i] = make_scalar3(h_angmom.data[i].x,h_angmom.data[i].y,h_angmom.data[i].z);
        snapshot.body_image[i] = h_body_image.data[i];
        }
    }

void export_SnapshotRigidData()
    {
    class_<SnapshotRigidData, boost::shared_ptr<SnapshotRigidData> >
        ("SnapshotRigidData", init<>())
        .def_readwrite("size", &SnapshotRigidData::size)
        .def_readwrite("com", &SnapshotRigidData::com)
        .def_readwrite("vel", &SnapshotRigidData::vel)
        .def_readwrite("angmom", &SnapshotRigidData::angmom)
        .def_readwrite("body_image", &SnapshotRigidData::body_image)
        ;
    }

void export_RigidData()
    {
    class_<RigidData, boost::shared_ptr<RigidData>, boost::noncopyable>("RigidData", init< boost::shared_ptr<ParticleData> >())
    .def("initializeData", &RigidData::initializeData)
    .def("getNumBodies", &RigidData::getNumBodies)    
    .def("getBodyCOM", &RigidData::getBodyCOM)    
    .def("setBodyCOM", &RigidData::setBodyCOM)        
    .def("getBodyVel", &RigidData::getBodyVel)  
    .def("setBodyVel", &RigidData::setBodyVel)      
    .def("getBodyOrientation", &RigidData::getBodyOrientation)    
    .def("setBodyOrientation", &RigidData::setBodyOrientation)        
    .def("getBodyNSize", &RigidData::getBodyNSize)   
    .def("getMass", &RigidData::getMass)   
    .def("setMass", &RigidData::setMass)                   
    .def("getBodyAngMom", &RigidData::getBodyAngMom) 
    .def("setAngMom", &RigidData::setAngMom)                                                                                           
    .def("getBodyMomInertia", &RigidData::getBodyMomInertia) 
    .def("setBodyMomInertia", &RigidData::setBodyMomInertia)         
    .def("getParticleTag", &RigidData::getParticleTag)   
    .def("getParticleDisp", &RigidData::getParticleDisp)
    .def("setParticleDisp", &RigidData::setParticleDisp)
    .def("getBodyNetForce", &RigidData::getBodyNetForce)
    .def("getBodyNetTorque", &RigidData::getBodyNetTorque)
    .def("setRV", &RigidData::setRV)
    ;
    }

#ifdef WIN32
#pragma warning( pop )
#endif

