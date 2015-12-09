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

/*! \param blah this does blah
*/   
ActiveForceCompute::ActiveForceCompute(boost::shared_ptr<SystemDefinition> sysdef, int seed, boost::python::list f_lst, bool orientation_link, Scalar rotation_diff)
        : ForceCompute(sysdef)
{
    m_exec_conf->msg->notice(5) << "Constructing ActiveForceCompute" << endl;
    
    vector<Scalar3> m_f_lst;
    tuple tmp_force;
    for (unsigned int i = 0; i < len(f_lst); i++)
    {
        tmp_force = extract<tuple>(f_lst[i]);
        if (len(tmp_force) !=3) { throw runtime_error("Non-3D force given for ActiveForceCompute"); }
        m_f_lst.push_back( make_scalar3(extract<Scalar>(tmp_force[0]), extract<Scalar>(tmp_force[1]), extract<Scalar>(tmp_force[2])));
    }
    
    if (m_f_lst.size() != m_pdata->getN()) { throw runtime_error("Force given for ActiveForceCompute doesn't match particle number."); }
    
    act_force_vec.resize(m_pdata->getN());
    act_force_mag.resize(m_pdata->getN());
    for (unsigned int i = 0; i < m_pdata->getN(); i++) //set active force vector to array from python
    {
        act_force_mag[i] = sqrt(m_f_lst[i].x*m_f_lst[i].x + m_f_lst[i].y*m_f_lst[i].y + m_f_lst[i].z*m_f_lst[i].z);
        act_force_vec[i].x = m_f_lst[i].x/act_force_mag[i];
        act_force_vec[i].y = m_f_lst[i].y/act_force_mag[i];
        act_force_vec[i].z = m_f_lst[i].z/act_force_mag[i];
    }
    
    orientationLink = orientation_link;
    rotationDiff = rotation_diff;
    // Hash the User's Seed to make it less likely to be a low positive integer
    seed = seed*0x12345677 + 0x12345; seed^=(seed>>16); seed*= 0x45679;
}

ActiveForceCompute::~ActiveForceCompute()
{
        m_exec_conf->msg->notice(5) << "Destroying ActiveForceCompute" << endl;
}

/*! \param blah this does blah
*/
void ActiveForceCompute::setForces()
{
    ArrayHandle<Scalar4> h_pos(m_pdata->getPositions(), access_location::host, access_mode::read);
    ArrayHandle< unsigned int > h_rtag(m_pdata->getRTags(), access_location::host, access_mode::read);
    ArrayHandle<Scalar4> h_orientation(m_pdata->getOrientationArray(), access_location::host, access_mode::read);
    ArrayHandle<Scalar4> h_force(m_force,access_location::host,access_mode::overwrite);

    // sanity check
    assert(h_pos.data != NULL);
    assert(h_rtag != NULL);
    assert(h_orientation.data != NULL);
    assert(h_force.data != NULL);

    Scalar3 f;
    // rotate force according to particle orientation only if orientation is linked to active force vector and there are rigid bodies
    if (orientationLink == true && m_sysdef->getRigidData()->getNumBodies() > 0)
    {
        vec3<Scalar> fi;
        for (unsigned int i = 0; i < m_pdata->getN(); i++)
        {
            unsigned int idx = h_rtag.data[i]; // recover original tag for particle indexing
            f = make_scalar3(act_force_mag[i]*act_force_vec[i].x, act_force_mag[i]*act_force_vec[i].y, act_force_mag[i]*act_force_vec[i].z);
            quat<Scalar> quati(h_orientation.data[idx]);
            fi = rotate(quati, vec3<Scalar>(f));
            h_force.data[idx].x = fi.x;
            h_force.data[idx].y = fi.y;
            h_force.data[idx].z = fi.z;
        }
    } else
    {
        for (unsigned int i = 0; i < m_pdata->getN(); i++)
        {
            unsigned int idx = h_rtag.data[i]; // recover original tag for particle indexing
            f = make_scalar3(act_force_mag[i]*act_force_vec[i].x, act_force_mag[i]*act_force_vec[i].y, act_force_mag[i]*act_force_vec[i].z);
            h_force.data[idx].x = f.x;
            h_force.data[idx].y = f.y;
            h_force.data[idx].z = f.z;
        }
    }






}

/*! \param blah this does blah
*/
void ActiveForceCompute::rotationalDiffusion(unsigned int timestep)
{
    Saru saru(m_seed, timestep);

    if (m_sysdef->getNDimensions() == 2) // 2D ADD OR STATEMENT TO CHECK IF CONSTRAINT IS BEING USED
    {

    } else // 3D: Following Stenhammar, Soft Matter, 2014
    {
        //USE VECTOR MATH TO SIMPLIFY THINGS?
        for (unsigned int i = 0; i < m_pdata->getN(); i++)
        {
            Scalar u = saru.d(0, 1.0); //generates an even distribution of random unit vectors in 3D
            Scalar v = saru.d(0, 1.0);
            Scalar theta = 6.283185307179586476925286766559 * u;
            Scalar phi = acos(2.0 * v - 1.0) ;
            vec3<Scalar> rand_vec;
            rand_vec.x = sin(phi) * cos(theta);
            rand_vec.y = sin(phi) * sin(theta);
            rand_vec.z = cos(phi);
            Scalar diffusion_mag = rotationDiff * gaussian_rng(saru, 1.0);
            vec3<Scalar> delta_vec;
            delta_vec.x = act_force_vec[i].y * rand_vec.z - act_force_vec[i].z * rand_vec.y;
            delta_vec.y = act_force_vec[i].z * rand_vec.x - act_force_vec[i].x * rand_vec.z;
            delta_vec.z = act_force_vec[i].x * rand_vec.y - act_force_vec[i].y * rand_vec.x;
            act_force_vec[i].x += delta_vec.x * diffusion_mag * m_deltaT;
            act_force_vec[i].y += delta_vec.y * diffusion_mag * m_deltaT;
            act_force_vec[i].z += delta_vec.z * diffusion_mag * m_deltaT;
            Scalar new_mag = sqrt(act_force_vec[i].x*act_force_vec[i].x + act_force_vec[i].y*act_force_vec[i].y + act_force_vec[i].z*act_force_vec[i].z);
            act_force_vec[i].x *= act_force_mag[i] / new_mag;
            act_force_vec[i].y *= act_force_mag[i] / new_mag;
            act_force_vec[i].z *= act_force_mag[i] / new_mag;
        }
    }
}

/*! This function calls setForces()
    \param timestep Current timestep
*/
void ActiveForceCompute::computeForces(unsigned int timestep)
{
    // Orientational Diffusion, check to make sure hasn't already been computed this timestep
    if (shouldCompute(timestep) && rotationDiff != 0)
    {
        rotationalDiffusion(timestep);
    }

    // set force for particles
    setForces();
}


void export_ActiveForceCompute()
{
    class_< ActiveForceCompute, boost::shared_ptr<ActiveForceCompute>, bases<ForceCompute>, boost::noncopyable >
    ("ActiveForceCompute", init< boost::shared_ptr<SystemDefinition>, int, boost::python::list, bool, Scalar >())
    ;
}
