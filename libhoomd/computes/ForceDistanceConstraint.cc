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

// Maintainer: jglaser

#include "ForceDistanceConstraint.h"

#include <string.h>

#include <boost/python.hpp>

#include <Eigen/Dense>
using namespace Eigen;

/*! \file ForceDistanceConstraint.cc
    \brief Contains code for the ForceDistanceConstraint class
*/

/*! \param sysdef SystemDefinition containing the ParticleData to compute forces on
*/
ForceDistanceConstraint::ForceDistanceConstraint(boost::shared_ptr<SystemDefinition> sysdef)
        : ForceConstraint(sysdef), m_cdata(m_sysdef->getConstraintData()), m_cmatrix(m_exec_conf), m_cvec(m_exec_conf), m_lagrange(m_exec_conf)
    {
    }

/*! Does nothing in the base class
    \param timestep Current timestep
*/
void ForceDistanceConstraint::computeForces(unsigned int timestep)
    {
    if (m_prof)
        m_prof->push("Dist constraint");

    if (m_cdata->getNGlobal() == 0)
        {
        m_exec_conf->msg->error() << "constrain.distance() called with no constraints defined!\n" << std::endl;
        throw std::runtime_error("Error computing constraints.\n");
        }

    // reallocate through amortized resizin
    unsigned int n_constraint = m_cdata->getN();
    m_cmatrix.resize(n_constraint*n_constraint);
    m_cvec.resize(n_constraint);

    // populate the terms in the matrix vector equation
    fillMatrixVector(timestep);

    // solve the matrix vector equation
    computeConstraintForces(timestep);

    if (m_prof)
        m_prof->pop();
    }

void ForceDistanceConstraint::fillMatrixVector(unsigned int timestep)
    {
    // fill the matrix in row-major order
    unsigned int n_constraint = m_cdata->getN();

    // access particle data
    ArrayHandle<Scalar4> h_pos(m_pdata->getPositions(), access_location::host, access_mode::read);
    ArrayHandle<Scalar4> h_vel(m_pdata->getVelocities(), access_location::host, access_mode::read);
    ArrayHandle<unsigned int> h_rtag(m_pdata->getRTags(), access_location::host, access_mode::read);
    ArrayHandle<Scalar4> h_netforce(m_pdata->getNetForce(), access_location::host, access_mode::read);

    // access matrix elements
    ArrayHandle<Scalar> h_cmatrix(m_cmatrix, access_location::host, access_mode::overwrite);
    ArrayHandle<Scalar> h_cvec(m_cvec, access_location::host, access_mode::overwrite);

    // clear matrix
    memset(h_cmatrix.data, 0, sizeof(Scalar)*m_cmatrix.size());

    const BoxDim& box = m_pdata->getBox();

    for (unsigned int n = 0; n < n_constraint; ++n)
        {
        // lookup the tag of each of the particles participating in the constraint
        const BondData::members_t constraint = m_cdata->getMembersByIndex(n);
        assert(constraint.tag[0] < m_pdata->getMaximumTag());
        assert(constraint.tag[1] < m_pdata->getMaximumTag());

        // transform a and b into indicies into the particle data arrays
        // (MEM TRANSFER: 4 integers)
        unsigned int idx_a = h_rtag.data[constraint.tag[0]];
        unsigned int idx_b = h_rtag.data[constraint.tag[1]];
        assert(idx_a <= m_pdata->getN()+m_pdata->getNGhosts());
        assert(idx_b <= m_pdata->getN()+m_pdata->getNGhosts());

        vec3<Scalar> ra(h_pos.data[idx_a]);
        vec3<Scalar> rb(h_pos.data[idx_b]);
        vec3<Scalar> rn(ra-rb);

        // apply minimum image
        rn = box.minImage(rn);

        vec3<Scalar> va(h_vel.data[idx_a]);
        Scalar ma(h_vel.data[idx_a].w);
        vec3<Scalar> vb(h_vel.data[idx_b]);
        Scalar mb(h_vel.data[idx_b].w);

        vec3<Scalar> rndot(va-vb);
        vec3<Scalar> qn(rn+rndot*m_deltaT);

        // fill matrix row
        for (unsigned int m = 0; m < n_constraint; ++m)
            {
            // lookup the tag of each of the particles participating in the constraint
            const BondData::members_t constraint_m = m_cdata->getMembersByIndex(m);
            assert(constraint_m.tag[0] < m_pdata->getMaximumTag());
            assert(constraint_m.tag[1] < m_pdata->getMaximumTag());

            // transform a and b into indicies into the particle data arrays
            // (MEM TRANSFER: 4 integers)
            unsigned int idx_m_a = h_rtag.data[constraint_m.tag[0]];
            unsigned int idx_m_b = h_rtag.data[constraint_m.tag[1]];
            assert(idx_m_a <= m_pdata->getN()+m_pdata->getNGhosts());
            assert(idx_m_b <= m_pdata->getN()+m_pdata->getNGhosts());

            vec3<Scalar> rm_a(h_pos.data[idx_m_a]);
            vec3<Scalar> rm_b(h_pos.data[idx_m_b]);
            vec3<Scalar> rm(rm_a-rm_b);

            // apply minimum image
            rm = box.minImage(rm);

            if (idx_m_a == idx_a)
                {
                h_cmatrix.data[n*n_constraint+m] += Scalar(4.0)*dot(qn,rm)/ma;
                }
            if (idx_m_b == idx_a)
                {
                h_cmatrix.data[n*n_constraint+m] -= Scalar(4.0)*dot(qn,rm)/ma;
                }
            if (idx_m_a == idx_b)
                {
                h_cmatrix.data[n*n_constraint+m] -= Scalar(4.0)*dot(qn,rm)/mb;
                }
            if (idx_m_b == idx_b)
                {
                h_cmatrix.data[n*n_constraint+m] += Scalar(4.0)*dot(qn,rm)/mb;
                }
            }

        // get constraint distance
        Scalar d = __int_as_scalar(m_cdata->getTypeByIndex(n));

        // fill vector component
        h_cvec.data[n] = (dot(qn,qn)-d*d)/m_deltaT/m_deltaT;
        h_cvec.data[n] += Scalar(2.0)*dot(qn,vec3<Scalar>(h_netforce.data[idx_a])/ma
              -vec3<Scalar>(h_netforce.data[idx_b])/mb);
        }
    }

void ForceDistanceConstraint::computeConstraintForces(unsigned int timestep)
    {
    typedef Matrix<Scalar, Dynamic, Dynamic, RowMajor> matrix_t;
    typedef Matrix<Scalar, Dynamic, 1> vec_t;
    typedef Map<matrix_t> matrix_map_t;
    typedef Map<vec_t> vec_map_t;

    unsigned int n_constraint = m_cdata->getN();

    // reallocate array of constraint forces
    m_lagrange.resize(n_constraint);

    // access matrix
    ArrayHandle<Scalar> h_cmatrix(m_cmatrix, access_location::host, access_mode::read);
    ArrayHandle<Scalar> h_cvec(m_cvec, access_location::host, access_mode::read);
    ArrayHandle<Scalar> h_lagrange(m_lagrange, access_location::host, access_mode::overwrite);

    matrix_map_t map_matrix(h_cmatrix.data, n_constraint,n_constraint);
    vec_map_t map_vec(h_cvec.data, n_constraint, 1);
    vec_map_t map_lagrange(h_lagrange.data,n_constraint, 1);

    // solve Ax = b
    map_lagrange = map_matrix.colPivHouseholderQr().solve(map_vec);

    // access particle data arrays
    ArrayHandle<Scalar4> h_pos(m_pdata->getPositions(), access_location::host, access_mode::read);
    ArrayHandle<unsigned int> h_rtag(m_pdata->getRTags(), access_location::host, access_mode::read);

    // access force array
    ArrayHandle<Scalar4> h_force(m_force, access_location::host, access_mode::overwrite);

    const BoxDim& box = m_pdata->getBox();

    unsigned int n_ptl = m_pdata->getN();

    // reset force arrray
    memset(h_force.data,0,sizeof(Scalar4)*n_ptl);

    // copy output to force array
    for (unsigned int n = 0; n < n_constraint; ++n)
        {
        // lookup the tag of each of the particles participating in the constraint
        const BondData::members_t constraint = m_cdata->getMembersByIndex(n);
        assert(constraint.tag[0] < m_pdata->getMaximumTag());
        assert(constraint.tag[1] < m_pdata->getMaximumTag());

        // transform a and b into indicies into the particle data arrays
        // (MEM TRANSFER: 4 integers)
        unsigned int idx_a = h_rtag.data[constraint.tag[0]];
        unsigned int idx_b = h_rtag.data[constraint.tag[1]];
        assert(idx_a <= m_pdata->getN()+m_pdata->getNGhosts());
        assert(idx_b <= m_pdata->getN()+m_pdata->getNGhosts());

        vec3<Scalar> ra(h_pos.data[idx_a]);
        vec3<Scalar> rb(h_pos.data[idx_b]);
        vec3<Scalar> rn(ra-rb);

        // apply minimum image
        rn = box.minImage(rn);

        // if idx is local
        if (idx_a < n_ptl)
            {
            vec3<Scalar> f(h_force.data[idx_a]);
            f -= Scalar(2.0)*h_lagrange.data[n]*rn;
            h_force.data[idx_a] = make_scalar4(f.x,f.y,f.z,Scalar(0.0));
            }
        if (idx_b < n_ptl)
            {
            vec3<Scalar> f(h_force.data[idx_b]);
            f += Scalar(2.0)*h_lagrange.data[n]*rn;
            h_force.data[idx_b] = make_scalar4(f.x,f.y,f.z,Scalar(0.0));
            }
        }
    }

#ifdef ENABLE_MPI
/*! \param timestep Current time step
 */
CommFlags ForceDistanceConstraint::getRequestedCommFlags(unsigned int timestep)
    {
    CommFlags flags = CommFlags(0);

    // we need the velocity and the net force in addition to the position
    flags[comm_flag::velocity] = 1;

    // FIXME
    //flags[comm_flag::net_force] = 1;

    flags |= ForceCompute::getRequestedCommFlags(timestep);

    return flags;
    }
#endif


void export_ForceDistanceConstraint()
    {
    class_< ForceDistanceConstraint, boost::shared_ptr<ForceDistanceConstraint>, bases<ForceConstraint>, boost::noncopyable >
    ("ForceDistanceConstraint", init< boost::shared_ptr<SystemDefinition> >())
    ;
    }
