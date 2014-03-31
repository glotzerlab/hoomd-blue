/*
Highly Optimized Object-oriented Many-particle Dynamics -- Blue Edition
(HOOMD-blue) Open Source Software License Copyright 2009-2014 The Regents of
the University of Michigan All rights reserved.

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

/*! \file RigidData.h
    \brief Contains declarations for RigidData and related classes.
*/

#ifdef NVCC
#error This header cannot be compiled by nvcc
#endif

#ifndef __RIGID_DATA_H__
#define __RIGID_DATA_H__

#include "ParticleData.h"
#include "GPUArray.h"
#include "RigidBodyGroup.h"


#ifdef ENABLE_CUDA
#include "RigidData.cuh"
#endif

//! Flag for invalid particle index
const unsigned int NO_INDEX = 0xffffffff;

//! Handy structure for passing around rigid body data
/*! \ingroup data_structs
 */
struct SnapshotRigidData
    {
    std::vector<Scalar3> com;       //!< Centers of masses
    std::vector<Scalar3> vel;       //!< Rigid bodies velocities
    std::vector<Scalar3> angmom;    //!< Angular momenta
    std::vector<int3> body_image;   //!< The body images
    unsigned int size;              //!< Number of bodies in this snapshot

    //! Default constructor
    SnapshotRigidData()
        {
        size = 0;
        }

    //! Resize the snapshot
    /*! \param n_bodies number of rigid bodies to reserve memory for
     */
    void resize(unsigned int n_bodies)
        {
        com.resize(n_bodies);
        vel.resize(n_bodies);
        angmom.resize(n_bodies);
        body_image.resize(n_bodies);
        size = n_bodies;
        }

    //! Validate the snapshot
    /* \returns true if number of elements in snapshot is consistent
     */
    bool validate() const
        {
        if (! com.size() == size) return false;
        if (! vel.size() == size) return false;
        if (! angmom.size() == size) return false;
        if (! body_image.size() == size) return false;
        return true;
        }

    //! Replicate the snapshot
    /*! \param n Number of times to replicate the snapshot
     */
    void replicate(unsigned int n);
    };

//! Stores all per rigid body values
/*! All rigid body data (except for the per-particle body value) is stored in RigidData
    which can be accessed from SystemDefinition. On construction, RigidData will read the body
    tag from the passed in ParticleData and initialize all rigid body data structures.

    The 2D arrays in this class bear a little explanation. They are arranged as \a n_max by
    \b n_bodies arrays (where \a n_max is the size of the largest rigid body in the system). Bodies
    are listed across the column and the quantity for each particle in that body is listed down the
    rows of the corresponding column. Thus, to access the index of particle \a p in body \b b, one
    would access array element \c particle_indices_handle.data[b*pitch+p] . This will set us up for
    the fastest GPU implementation of tasks like summing the force/torque on each body as we will
    be able to process 1 body in each block with one particle in each thread, performing any sums as
    reductions.

    \ingroup data_structs
*/
class RigidData
    {
    public:
        //! Initializes all rigid body data from the given particle data
        RigidData(boost::shared_ptr<ParticleData> particle_data);
        //! Destructor
        ~RigidData();

        //! Get the number of bodies in the rigid data
        unsigned int getNumBodies() const
            {
            return m_n_bodies;
            }
        //! Get the maximum number of particles in a rigid body
        unsigned int getNmax()
            {
            return m_nmax;
            }
        //! Get the total degrees of freedom
        unsigned int getNumDOF()
            {
            return m_ndof;
            }

        //! Set the body angular momentum for a rigid body
        void setAngMom(unsigned int body, Scalar3 angmom);

        //! \name getter methods (static data)
        //@{
        //! Get the m_moment_inertial
        const GPUArray<Scalar4>& getMomentInertia()
            {
            return m_moment_inertia;
            }
        //! Get m_body_size
        const GPUArray<unsigned int>& getBodySize()
            {
            return m_body_size;
            }
        //! Get the m_particle_tags
        const GPUArray<unsigned int>& getParticleTags()
            {
            return m_particle_tags;
            }
        //! Get m_particle_indices
        const GPUArray<unsigned int>& getParticleIndices()
            {
            return m_particle_indices;
            }
        //! Get m_particle_pos
        const GPUArray<Scalar4>& getParticlePos()
            {
            return m_particle_pos;
            }
        //! Get m_particle_orientation
        const GPUArray<Scalar4>& getParticleOrientation()
            {
            return m_particle_orientation;
            }
        //! Get m_mass
        const GPUArray<Scalar>& getBodyMass()
            {
            return m_body_mass;
            }
        //! Get m_body_dof
        const GPUArray<unsigned int>& getBodyDOF()
            {
            return m_body_dof;
            }
        //! Get m_particle_pos
        const GPUArray<Scalar4>& getParticleOldPos()
            {
            return m_particle_oldpos;
            }
        //! Get m_particle_pos
        const GPUArray<Scalar4>& getParticleOldVel()
            {
            return m_particle_oldvel;
            }
        //! Get m_particle_offset
        const GPUArray<unsigned int>& getParticleOffset()
            {
            return m_particle_offset;
            }
        //@}

        //! \name getter methods (integrated data)
        //@{
        //! Get m_com
        const GPUArray<Scalar4>& getCOM() const
            {
            return m_com;
            }
        //! Get m_vel
        const GPUArray<Scalar4>& getVel() const
            {
            return m_vel;
            }
        //! Get m_orientation
        const GPUArray<Scalar4>& getOrientation() const
            {
            return m_orientation;
            }
        //! Get m_conjqm
        const GPUArray<Scalar4>& getConjqm() const
            {
            return m_conjqm;
            }
        //! Get m_angmom
        const GPUArray<Scalar4>& getAngMom() const
            {
            return m_angmom;
            }
        //! Get m_angvel
        const GPUArray<Scalar4>& getAngVel() const
            {
            return m_angvel;
            }
        //! Get m_body_image
        const GPUArray<int3>& getBodyImage() const
            {
            return m_body_image;
            }
        //! Get m_ex_space
        const GPUArray<Scalar4>& getExSpace()
            {
            return m_ex_space;
            }
        //! Get m_ey_space
        const GPUArray<Scalar4>& getEySpace()
            {
            return m_ey_space;
            }
        //! Get m_ez_space
        const GPUArray<Scalar4>& getEzSpace()
            {
            return m_ez_space;
            }
        //! Get m_force
        const GPUArray<Scalar4>& getForce()
            {
            return m_force;
            }
        //! Get m_torque
        const GPUArray<Scalar4>& getTorque()
            {
            return m_torque;
            }
         //! Get m_rigid_particle_index
        const GPUArray<unsigned int>& getAllIndexRigid()
            {
            return m_rigid_particle_indices;
            }

         //! Get m_rigid_particle_index
        unsigned int getNumIndexRigid()
            {
            return m_num_particles;
            }

         //! Get the number of particles of a body
        unsigned int getBodyNSize(unsigned int body)
            {
            assert(body >= 0 && body < getNumBodies());
            ArrayHandle<unsigned int> size_handle(m_body_size, access_location::host, access_mode::read);
            unsigned int result = size_handle.data[body];
            return result;
            }
         //! Get the mass of a body
        Scalar getMass(unsigned int body)
            {
            assert(body >= 0 && body < getNumBodies());
            ArrayHandle<Scalar> mass_handle(m_body_mass, access_location::host, access_mode::read);
            Scalar result = mass_handle.data[body];
            return result;
            }
        //! set the mass of a body
        void setMass(unsigned int body, const Scalar mass)
            {
            assert(body >= 0 && body < getNumBodies());
            ArrayHandle<Scalar> mass_handle(m_body_mass, access_location::host, access_mode::readwrite);
            mass_handle.data[body] = mass;
            }
        //! Get the particle tags of a body
        unsigned int getParticleTag(unsigned int body, unsigned int index)
            {
            assert(body >= 0 && body < getNumBodies());
            assert(index >= 0 && index < getBodyNSize(body));
            ArrayHandle<unsigned int> tags(m_particle_tags, access_location::host, access_mode::read);
            unsigned int tags_pitch = m_particle_tags.getPitch();
            unsigned int result = tags.data[body*tags_pitch + index];
            return result;
            }
        //! Get the particle displacement relative to COM of a body
        Scalar3 getParticleDisp(unsigned int body, unsigned int index)
            {
            assert(body >= 0 && body < getNumBodies());
            assert(index >= 0 && index < getBodyNSize(body));
            ArrayHandle<Scalar4> pos(m_particle_pos, access_location::host, access_mode::read);
            unsigned int particle_pos_pitch = m_particle_pos.getPitch();
            unsigned int idx = body * particle_pos_pitch + index;
            Scalar3 result = make_scalar3(pos.data[idx].x, pos.data[idx].y, pos.data[idx].z) ;
            return result;
            }
        //! Set the particle displacement relative to COM of a body
        void setParticleDisp(unsigned int body, unsigned int index, const Scalar3& new_pos)
            {
            assert(body >= 0 && body < getNumBodies());
            assert(index >= 0 && index < getBodyNSize(body));
            ArrayHandle<Scalar4> pos(m_particle_pos, access_location::host, access_mode::readwrite);
            unsigned int particle_pos_pitch = m_particle_pos.getPitch();
            unsigned int idx = body * particle_pos_pitch + index;
            pos.data[idx].x= new_pos.x;
            pos.data[idx].y= new_pos.y;
            pos.data[idx].z= new_pos.z;
            }
        //! Get the current COM of a body
        Scalar3 getBodyCOM(unsigned int body)
            {
            assert(body >= 0 && body < getNumBodies());
            ArrayHandle<Scalar4> com_handle(m_com, access_location::host, access_mode::read);
            Scalar3 result = make_scalar3(com_handle.data[body].x, com_handle.data[body].y, com_handle.data[body].z);
            return result;
            }
        //! Set the current COM of a body
        void setBodyCOM(unsigned int body,const Scalar3& pos)
            {
            assert(body >= 0 && body < getNumBodies());
            ArrayHandle<Scalar4> com_handle(m_com, access_location::host, access_mode::readwrite);
            com_handle.data[body].x = pos.x;
            com_handle.data[body].y = pos.y;
            com_handle.data[body].z = pos.z;
            }
         //! Get the current velocity of a body's COM
        Scalar3 getBodyVel(unsigned int body)
            {
            assert(body >= 0 && body < getNumBodies());
            ArrayHandle<Scalar4> vel_handle(m_vel, access_location::host, access_mode::read);
            Scalar3 result = make_scalar3(vel_handle.data[body].x, vel_handle.data[body].y, vel_handle.data[body].z);
            return result;
            }
         //! Set the current velocity of a body's COM
        void setBodyVel(unsigned int body, const Scalar3& vel)
            {
            assert(body >= 0 && body < getNumBodies());
            ArrayHandle<Scalar4> vel_handle(m_vel, access_location::host, access_mode::readwrite);
            vel_handle.data[body].x = vel.x;
            vel_handle.data[body].y = vel.y;
            vel_handle.data[body].z = vel.z;
            }
         //! Get the current orientation (quaternion) of a body
        Scalar4 getBodyOrientation(unsigned int body)
            {
            assert(body >= 0 && body < getNumBodies());
            ArrayHandle<Scalar4> orientation_handle(m_orientation, access_location::host, access_mode::read);
            Scalar4 result = make_scalar4(orientation_handle.data[body].x, orientation_handle.data[body].y, orientation_handle.data[body].z, orientation_handle.data[body].w);
            return result;
            }
        //! Set the current orientation (quaternion) of a body
        void setBodyOrientation(unsigned int body, const Scalar4& in_quat)
            {
            assert(body >= 0 && body < getNumBodies());
            ArrayHandle<Scalar4> orientation_handle(m_orientation, access_location::host, access_mode::readwrite);

            Scalar4 quat = in_quat;
            // Normalize
            Scalar norm = 1.0 / sqrt(quat.x * quat.x + quat.y * quat.y + quat.z * quat.z + quat.w * quat.w);
            quat.x *= norm;
            quat.y *= norm;
            quat.z *= norm;
            quat.w *= norm;

            orientation_handle.data[body].x = quat.x;
            orientation_handle.data[body].y = quat.y;
            orientation_handle.data[body].z = quat.z;
            orientation_handle.data[body].w = quat.w;
            }
         //! Get the current angular momentum of a body
        Scalar3 getBodyAngMom(unsigned int body)
            {
            assert(body >= 0 && body < getNumBodies());
            ArrayHandle<Scalar4> angmom_handle(m_angmom, access_location::host, access_mode::read);
            Scalar3 result = make_scalar3(angmom_handle.data[body].x, angmom_handle.data[body].y, angmom_handle.data[body].z);
            return result;
            }
         //! Get the diagonalized moment of inertia of a body
        Scalar3 getBodyMomInertia(unsigned int body)
            {
            assert(body >= 0 && body < getNumBodies());
            ArrayHandle<Scalar4> mom_inertia_handle(m_moment_inertia, access_location::host, access_mode::read);
            Scalar3 result = make_scalar3(mom_inertia_handle.data[body].x, mom_inertia_handle.data[body].y, mom_inertia_handle.data[body].z);
            return result;
            }
         //! set the diagonalized moment of inertia of a body
        void setBodyMomInertia(unsigned int body, const Scalar3& mom)
            {
            assert(body >= 0 && body < getNumBodies());
            ArrayHandle<Scalar4> mom_inertia_handle(m_moment_inertia, access_location::host, access_mode::readwrite);
            mom_inertia_handle.data[body].x= mom.x;
            mom_inertia_handle.data[body].y= mom.y;
            mom_inertia_handle.data[body].z= mom.z;
            }
        //! Get the current net force on a body
        Scalar3 getBodyNetForce(unsigned int body)
            {
            assert(body >= 0 && body < getNumBodies());
            ArrayHandle<Scalar4> h_force(m_force, access_location::host, access_mode::read);
            Scalar3 result = make_scalar3(h_force.data[body].x, h_force.data[body].y, h_force.data[body].z);
            return result;
            }
        //! Get the current net torque on a body
        Scalar3 getBodyNetTorque(unsigned int body)
            {
            assert(body >= 0 && body < getNumBodies());
            ArrayHandle<Scalar4> h_torque(m_torque, access_location::host, access_mode::read);
            Scalar3 result = make_scalar3(h_torque.data[body].x, h_torque.data[body].y, h_torque.data[body].z);
            return result;
            }
        //@}

        //! Update x and v of rigid body data and virial
        void setRV(bool set_x);

        //! Performs steps needed to compute the rigid body virial correction at the start of the step
        void computeVirialCorrectionStart();

        //! Performs steps needed to compute the rigid body virial correction at the end of the step
        void computeVirialCorrectionEnd(Scalar deltaT);

        //! Intitialize and fill out all data members: public to be called from NVEUpdater when the body information of particles wss already set.
        void initializeData();

        //! Compute the axes from quaternion, used when reading from restart files
        void exyzFromQuaternion(Scalar4 &quat, Scalar4 &ex_space, Scalar4 &ey_space, Scalar4 &ez_space);

        //! Functions used to diagonalize the inertia tensor for moment inertia and principle axes
        int diagonalize(Scalar **matrix, Scalar *evalues, Scalar **evectors);
        void rotate(Scalar **matrix, int i, int j, int k, int l, Scalar s, Scalar tau);

        //! Initialize from a snapshot
        void initializeFromSnapshot(const SnapshotRigidData& snapshot);

        //! Take a snapshot of the current rigid body data
        void takeSnapshot(SnapshotRigidData& snapshot) const;

    private:
        boost::shared_ptr<ParticleData> m_pdata;        //!< The particle data with which this RigidData is associated
        boost::shared_ptr<const ExecutionConfiguration> m_exec_conf; //!< Stored shared ptr to the execution configuration
        boost::signals2::connection m_sort_connection;   //!< Connection to the resort signal from ParticleData

        //! \name static data members (set on initialization)
        //@{
        unsigned int m_n_bodies;                    //!< Number of rigid bodies in the data structure
        unsigned int m_nmax;                        //!< Maximum number of particles in a rigid body
        unsigned int m_ndof;                        //!< Total number degrees of freedom of rigid bodies
        GPUArray<unsigned int> m_body_dof;          //!< n_bodies length 1D array of body DOF
        GPUArray<Scalar> m_body_mass;               //!< n_bodies length 1D array of body mass
        GPUArray<Scalar4> m_moment_inertia;         //!< n_bodies length 1D array of moments of inertia in the body frame
        GPUArray<unsigned int> m_body_size;         //!< n_bodies length 1D array listing the size of each rigid body
        GPUArray<unsigned int> m_particle_tags;     //!< n_max by n_bodies 2D array listing particle tags belonging to bodies
        GPUArray<Scalar4> m_particle_pos;           //!< n_max by n_bodies 2D array listing particle positions relative to the COM for this body in which body-fixed frame
        GPUArray<Scalar4> m_particle_orientation;   //!< n_max by n_bodies 2D array listing native particle orientations in the body frame
        GPUArray<unsigned int> m_particle_indices;  //!< n_max by n_bodies 2D array listing particle indices belonging to bodies (updated when particles are resorted)
        GPUArray<unsigned int> m_particle_offset;   //!< n_particles by 1 array listing the offset of each particle in its body
        //@}

        //! \name dynamic data members (updated via integration)
        //@{
        GPUArray<Scalar4> m_com;            //!< n_bodies length 1D array of center of mass positions
        GPUArray<Scalar4> m_vel;            //!< n_bodies length 1D array of body velocities
        GPUArray<Scalar4> m_angmom;         //!< n_bodies length 1D array of angular momentum in the space frame
        GPUArray<Scalar4> m_angvel;         //!< n_bodies length 1D array of angular velocity in the space frame
        GPUArray<Scalar4> m_orientation;    //!< n_bodies length 1D array of orientation quaternions
        GPUArray<Scalar4> m_conjqm;         //!< n_bodies length 1D array of conjugate momenta of orientation quaternion
        GPUArray<Scalar4> m_ex_space;       //!< n_bodies length 1D array of the x axis of the body frame in the space frame
        GPUArray<Scalar4> m_ey_space;       //!< n_bodies length 1D array of the y axis of the body frame in the space frame
        GPUArray<Scalar4> m_ez_space;       //!< n_bodies length 1D array of the z axis of the body frame in the space frame
        GPUArray<int3> m_body_image;        //!< n_bodies length 1D array of the body image

        // Body forces and torques are stored here instead of the rigid body integrator because of GPU implementation
        // since the body forces and torques in the shared memory of thread blocks become invalid after the kernel finishes.
        GPUArray<Scalar4> m_force;          //!< n_bodies length 1D array of the body force
        GPUArray<Scalar4> m_torque;         //!< n_bodies length 1D array of the body torque

        GPUArray<Scalar4> m_particle_oldpos;        //!< nparticles long array storing the old position of the particles (for virial computation)
        GPUArray<Scalar4> m_particle_oldvel;        //!< npartilces long array storing the old velocity of the particles (for virial computation)

        GPUArray<unsigned int> m_rigid_particle_indices; //!< All particle indices that are in rigid bodies, only needed for GPU code
        unsigned int m_num_particles;                    //!< length of m_rigid_particle_indices
        //@}

        //! Recalculate the cached indices from the stored tags after a particle sort
        void recalcIndices();

        //! Compute quaternion from the axes
        void quaternionFromExyz(Scalar4 &ex_space, Scalar4 &ey_space, Scalar4 &ez_space, Scalar4 &quat);

        //! Update x and v of rigid body data and virial
        void setRVCPU(bool set_x);

        //! Performs steps needed to compute the rigid body virial correction at the start of the step on the CPU
        void computeVirialCorrectionStartCPU();

        //! Performs steps needed to compute the rigid body virial correction at the end of the step on the CPU
        void computeVirialCorrectionEndCPU(Scalar deltaT);

#ifdef ENABLE_CUDA
        //! Helper funciton to update x and v of rigid body data and virial on the GPU
        void setRVGPU(bool set_x);

        //! Performs steps needed to compute the rigid body virial correction at the start of the step on the GPU
        void computeVirialCorrectionStartGPU();

        //! Performs steps needed to compute the rigid body virial correction at the end of the step on the GPU
        void computeVirialCorrectionEndGPU(Scalar deltaT);
#endif
    };

//! Export the SnapshotRigidData class to python
void export_SnapshotRigidData();

//! Export the RigidData class to python
void export_RigidData();

#endif
