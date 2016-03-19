/*
Highly Optimized Object-oriented Many-particle Dynamics -- Blue Edition
(HOOMD-blue) Open Source Software License Copyright 2009-2015 The Regents of
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

// Maintainer: joaander


#include "ActiveForceCompute.h"

#include <boost/python.hpp>
#include <vector>
using namespace boost::python;

using namespace std;

/*! \file ActiveForceCompute.cc
    \brief Contains code for the ActiveForceCompute class
*/

/*! \param seed required user-specified seed number for random number generator.
    \param f_list An array of (x,y,z) tuples for the active force vector for each individual particle.
    \param orientation_link if True then particle orientation is coupled to the active force vector. Only
    relevant for non-point-like anisotropic particles.
    \param rotation_diff rotational diffusion constant for all particles.
    \param constraint specifies a constraint surface, to which particles are confined,
    such as update.constraint_ellipsoid.
*/
ActiveForceCompute::ActiveForceCompute(boost::shared_ptr<SystemDefinition> sysdef,
                                        boost::shared_ptr<ParticleGroup> group,
                                        int seed,
                                        boost::python::list f_lst,
                                        bool orientation_link,
                                        Scalar rotation_diff,
                                        Scalar3 P,
                                        Scalar rx,
                                        Scalar ry,
                                        Scalar rz)
        : ForceCompute(sysdef), m_group(group), m_orientationLink(orientation_link), m_rotationDiff(rotation_diff),
            m_P(P), m_rx(rx), m_ry(ry), m_rz(rz)
    {
    m_exec_conf->msg->notice(5) << "Constructing ActiveForceCompute" << endl;

    unsigned int group_size = m_group->getNumMembersGlobal();
    if (group_size == 0)
        {
        m_exec_conf->msg->error() << "Creating a ActiveForceCompute with an empty group" << endl;
        throw std::runtime_error("Error initializing ActiveForceCompute");
        }

    vector<Scalar3> c_f_lst;
    tuple tmp_force;
    for (unsigned int i = 0; i < len(f_lst); i++)
        {
        tmp_force = extract<tuple>(f_lst[i]);
        if (len(tmp_force) !=3)
            throw runtime_error("Non-3D force given for ActiveForceCompute");
        c_f_lst.push_back( make_scalar3(extract<Scalar>(tmp_force[0]), extract<Scalar>(tmp_force[1]), extract<Scalar>(tmp_force[2])));
        }

    if (c_f_lst.size() != group_size) { throw runtime_error("Force given for ActiveForceCompute doesn't match particle number."); }
    if (m_orientationLink == true && m_rotationDiff != 0)
        {
        throw runtime_error("Non-spherical particles and rotational diffusion of the active force vector is ill defined. Instead implement rotational diffusion through the integrator.");
        }

    m_activeVec.resize(group_size);
    m_activeMag.resize(group_size);

    ArrayHandle<Scalar3> h_activeVec(m_activeVec, access_location::host);
    ArrayHandle<Scalar> h_activeMag(m_activeMag, access_location::host);

    // for each of the particles in the group
    for (unsigned int i = 0; i < group_size; i++)
        {
        h_activeMag.data[i] = slow::sqrt(c_f_lst[i].x*c_f_lst[i].x + c_f_lst[i].y*c_f_lst[i].y + c_f_lst[i].z*c_f_lst[i].z);
        if(h_activeMag.data[i] == 0.0) // fixes divide by 0 case if magnitude of active force vector is 0
            {
            h_activeMag.data[i] = 0.000000000001;
            }

        h_activeVec.data[i] = make_scalar3(0, 0, 0);
        h_activeVec.data[i].x = c_f_lst[i].x / h_activeMag.data[i];
        h_activeVec.data[i].y = c_f_lst[i].y / h_activeMag.data[i];
        h_activeVec.data[i].z = c_f_lst[i].z / h_activeMag.data[i];
        }

    last_computed = 10;

    // Hash the User's Seed to make it less likely to be a low positive integer
    m_seed = seed*0x12345677 + 0x12345; seed^=(seed>>16); seed*= 0x45679;
    }

ActiveForceCompute::~ActiveForceCompute()
    {
    m_exec_conf->msg->notice(5) << "Destroying ActiveForceCompute" << endl;
    }

/*! This function sets appropriate active forces on all active particles.
*/
void ActiveForceCompute::setForces()
    {
	//  array handles
    ArrayHandle<Scalar3> h_actVec(m_activeVec, access_location::host, access_mode::read);
    ArrayHandle<Scalar> h_actMag(m_activeMag, access_location::host, access_mode::read);
    ArrayHandle<Scalar4> h_force(m_force,access_location::host,access_mode::overwrite);
    ArrayHandle<Scalar4> h_orientation(m_pdata->getOrientationArray(), access_location::host, access_mode::read);
    ArrayHandle<unsigned int> h_rtag(m_pdata->getRTags(), access_location::host, access_mode::read);

    // sanity check
    assert(h_actVec.data != NULL);
    assert(h_actMag.data != NULL);
    assert(h_orientation.data != NULL);

    // zero forces so we don't leave any forces set for indices that are no longer part of our group
    memset(h_force.data, 0, sizeof(Scalar4) * m_force.getNumElements());

    for (unsigned int i = 0; i < m_group->getNumMembers(); i++)
        {
        unsigned int tag = m_group->getMemberTag(i);
        unsigned int idx = h_rtag.data[tag];

        Scalar3 f;
        // rotate force according to particle orientation only if orientation is linked to active force vector
        if (m_orientationLink == true)
            {
            vec3<Scalar> fi;
            f = make_scalar3(h_actMag.data[i]*h_actVec.data[i].x, h_actMag.data[i]*h_actVec.data[i].y, h_actMag.data[i]*h_actVec.data[i].z);
            quat<Scalar> quati(h_orientation.data[idx]);
            fi = rotate(quati, vec3<Scalar>(f));
            h_force.data[idx].x = fi.x;
            h_force.data[idx].y = fi.y;
            h_force.data[idx].z = fi.z;
            }
        else // no orientation link
            {
            f = make_scalar3(h_actMag.data[i]*h_actVec.data[i].x, h_actMag.data[i]*h_actVec.data[i].y, h_actMag.data[i]*h_actVec.data[i].z);
            h_force.data[idx].x = f.x;
            h_force.data[idx].y = f.y;
            h_force.data[idx].z = f.z;
            }
        }
    }

/*! This function applies rotational diffusion to all active particles
    \param timestep Current timestep
*/
void ActiveForceCompute::rotationalDiffusion(unsigned int timestep)
    {
	//  array handles
    ArrayHandle<Scalar3> h_actVec(m_activeVec, access_location::host, access_mode::readwrite);
    ArrayHandle<Scalar4> h_pos(m_pdata -> getPositions(), access_location::host, access_mode::read);
    ArrayHandle<unsigned int> h_rtag(m_pdata->getRTags(), access_location::host, access_mode::read);
    assert(h_pos.data != NULL);

    for (unsigned int i = 0; i < m_group->getNumMembers(); i++)
        {
        unsigned int tag = m_group->getMemberTag(i);
        unsigned int idx = h_rtag.data[tag];

        if (m_sysdef->getNDimensions() == 2) // 2D
            {
            Saru saru(idx, timestep, m_seed);
            Scalar delta_theta; // rotational diffusion angle
            delta_theta = m_rotationConst * gaussian_rng(saru, 1.0);
            Scalar theta; // angle on plane defining orientation of active force vector
            theta = atan2(h_actVec.data[i].y, h_actVec.data[i].x);
            theta += delta_theta;
            h_actVec.data[i].x = slow::cos(theta);
            h_actVec.data[i].y = slow::sin(theta);
            }
        else // 3D: Following Stenhammar, Soft Matter, 2014
            {
            if (m_rx == 0) // if no constraint
                {
                Saru saru(idx, timestep, m_seed);
                Scalar u = saru.s(Scalar(0), Scalar(1.0)); // generates an even distribution of random unit vectors in 3D
                Scalar v = saru.s(Scalar(0), Scalar(1.0));
                Scalar theta = 2.0 * M_PI * u;
                Scalar phi = slow::acos(2.0 * v - 1.0);

                vec3<Scalar> rand_vec;
                rand_vec.x = slow::sin(phi) * slow::cos(theta);
                rand_vec.y = slow::sin(phi) * slow::sin(theta);
                rand_vec.z = slow::cos(phi);

                vec3<Scalar> aux_vec;
                aux_vec.x = h_actVec.data[i].y * rand_vec.z - h_actVec.data[i].z * rand_vec.y;
                aux_vec.y = h_actVec.data[i].z * rand_vec.x - h_actVec.data[i].x * rand_vec.z;
                aux_vec.z = h_actVec.data[i].x * rand_vec.y - h_actVec.data[i].y * rand_vec.x;
                Scalar aux_vec_mag = slow::sqrt(aux_vec.x*aux_vec.x + aux_vec.y*aux_vec.y + aux_vec.z*aux_vec.z);
                aux_vec.x /= aux_vec_mag;
                aux_vec.y /= aux_vec_mag;
                aux_vec.z /= aux_vec_mag;

                vec3<Scalar> current_vec;
                current_vec.x = h_actVec.data[i].x;
                current_vec.y = h_actVec.data[i].y;
                current_vec.z = h_actVec.data[i].z;

                Scalar delta_theta = m_rotationConst * gaussian_rng(saru, 1.0);
                h_actVec.data[i].x = slow::cos(delta_theta)*current_vec.x + slow::sin(delta_theta)*aux_vec.x;
                h_actVec.data[i].y = slow::cos(delta_theta)*current_vec.y + slow::sin(delta_theta)*aux_vec.y;
                h_actVec.data[i].z = slow::cos(delta_theta)*current_vec.z + slow::sin(delta_theta)*aux_vec.z;
                }
            else // if constraint exists
                {
                EvaluatorConstraintEllipsoid Ellipsoid(m_P, m_rx, m_ry, m_rz);
                Saru saru(idx, timestep, m_seed);

                Scalar3 current_pos = make_scalar3(h_pos.data[idx].x, h_pos.data[idx].y, h_pos.data[idx].z);
                Scalar3 norm_scalar3 = Ellipsoid.evalNormal(current_pos); // the normal vector to which the particles are confined.

                vec3<Scalar> norm;
                norm = vec3<Scalar> (norm_scalar3);

                vec3<Scalar> current_vec;
                current_vec.x = h_actVec.data[i].x;
                current_vec.y = h_actVec.data[i].y;
                current_vec.z = h_actVec.data[i].z;
                vec3<Scalar> aux_vec = cross(current_vec, norm); // aux vec for defining direction that active force vector rotates towards.

                Scalar delta_theta; // rotational diffusion angle
                delta_theta = m_rotationConst * gaussian_rng(saru, 1.0);

                h_actVec.data[i].x = slow::cos(delta_theta)*current_vec.x + slow::sin(delta_theta)*aux_vec.x;
                h_actVec.data[i].y = slow::cos(delta_theta)*current_vec.y + slow::sin(delta_theta)*aux_vec.y;
                h_actVec.data[i].z = slow::cos(delta_theta)*current_vec.z + slow::sin(delta_theta)*aux_vec.z;
                }
            }
        }
    }

/*! This function sets an ellipsoid surface constraint for all active particles
*/
void ActiveForceCompute::setConstraint()
    {
    EvaluatorConstraintEllipsoid Ellipsoid(m_P, m_rx, m_ry, m_rz);

    //  array handles
    ArrayHandle<Scalar3> h_actVec(m_activeVec, access_location::host, access_mode::readwrite);
    ArrayHandle <Scalar4> h_pos(m_pdata -> getPositions(), access_location::host, access_mode::read);
    ArrayHandle<unsigned int> h_rtag(m_pdata->getRTags(), access_location::host, access_mode::read);
    assert(h_pos.data != NULL);

    for (unsigned int i = 0; i < m_group->getNumMembers(); i++)
        {
        unsigned int tag = m_group->getMemberTag(i);
        unsigned int idx = h_rtag.data[tag];

        Scalar3 current_pos = make_scalar3(h_pos.data[idx].x, h_pos.data[idx].y, h_pos.data[idx].z);

        Scalar3 norm_scalar3 = Ellipsoid.evalNormal(current_pos); // the normal vector to which the particles are confined.
        vec3<Scalar> norm;
        norm = vec3<Scalar>(norm_scalar3);
        Scalar dot_prod = h_actVec.data[i].x * norm.x + h_actVec.data[i].y * norm.y + h_actVec.data[i].z * norm.z;

        h_actVec.data[i].x -= norm.x * dot_prod;
        h_actVec.data[i].y -= norm.y * dot_prod;
        h_actVec.data[i].z -= norm.z * dot_prod;

        Scalar new_norm = slow::sqrt(h_actVec.data[i].x*h_actVec.data[i].x
                                     + h_actVec.data[i].y*h_actVec.data[i].y
                                     + h_actVec.data[i].z*h_actVec.data[i].z);

        h_actVec.data[i].x /= new_norm;
        h_actVec.data[i].y /= new_norm;
        h_actVec.data[i].z /= new_norm;
        }
    }

/*! This function applies constraints, rotational diffusion, and sets forces for all active particles
    \param timestep Current timestep
*/
void ActiveForceCompute::computeForces(unsigned int timestep)
    {
    if (m_prof) m_prof->push(m_exec_conf, "ActiveForceCompute");

    if (last_computed != timestep)
        {
        m_rotationConst = slow::sqrt(2.0 * m_rotationDiff * m_deltaT);

        last_computed = timestep;

        if (m_rx != 0)
            {
            setConstraint(); // apply surface constraints to active particles active force vectors
            }
        if (m_rotationDiff != 0)
            {
            rotationalDiffusion(timestep); // apply rotational diffusion to active particles
            }
	    setForces(); // set forces for particles
	    }

    #ifdef ENABLE_CUDA
    if(m_exec_conf->isCUDAErrorCheckingEnabled())
        CHECK_CUDA_ERROR();
    #endif

    if (m_prof)
        m_prof->pop(m_exec_conf);
    }


void export_ActiveForceCompute()
    {
    class_< ActiveForceCompute, boost::shared_ptr<ActiveForceCompute>, bases<ForceCompute>, boost::noncopyable >
    ("ActiveForceCompute", init< boost::shared_ptr<SystemDefinition>,
                                    boost::shared_ptr<ParticleGroup>,
                                    int,
                                    boost::python::list,
                                    bool,
                                    Scalar,
                                    Scalar3,
                                    Scalar,
                                    Scalar,
                                    Scalar >())
    ;
    }
