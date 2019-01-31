// Copyright (c) 2009-2019 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.


// Maintainer: jglaser

#include "ForceDistanceConstraint.h"

#include <string.h>
using namespace Eigen;
namespace py = pybind11;

/*! \file ForceDistanceConstraint.cc
    \brief Contains code for the ForceDistanceConstraint class
*/

/*! \param sysdef SystemDefinition containing the ParticleData to compute forces on
*/
ForceDistanceConstraint::ForceDistanceConstraint(std::shared_ptr<SystemDefinition> sysdef)
        : MolecularForceCompute(sysdef), m_cdata(m_sysdef->getConstraintData()),
          m_cmatrix(m_exec_conf), m_cvec(m_exec_conf), m_lagrange(m_exec_conf),
          m_rel_tol(1e-3), m_constraint_violated(m_exec_conf), m_condition(m_exec_conf),
          m_sparse_idxlookup(m_exec_conf), m_constraint_reorder(true), m_constraints_added_removed(true),
          m_d_max(0.0)
    {
    m_constraint_violated.resetFlags(0);

    // connect to the ConstraintData to receive notifications when constraints change order in memory
    m_cdata->getGroupReorderSignal().connect<ForceDistanceConstraint, &ForceDistanceConstraint::slotConstraintReorder>(this);

    // connect to ConstraintData to receive notifications when global constraint topology changes
    m_cdata->getGroupNumChangeSignal().connect<ForceDistanceConstraint, &ForceDistanceConstraint::slotConstraintsAddedRemoved>(this);

    // reset condition
    m_condition.resetFlags(0);
    }

//! Destructor
ForceDistanceConstraint::~ForceDistanceConstraint()
    {
    // disconnect from signal in ConstraintData
    m_cdata->getGroupReorderSignal().disconnect<ForceDistanceConstraint, &ForceDistanceConstraint::slotConstraintReorder>(this);
    m_cdata->getGroupNumChangeSignal().disconnect<ForceDistanceConstraint, &ForceDistanceConstraint::slotConstraintsAddedRemoved>(this);
    #ifdef ENABLE_MPI
    if (m_comm_ghost_layer_connected)
        m_comm->getGhostLayerWidthRequestSignal().disconnect<ForceDistanceConstraint, &ForceDistanceConstraint::askGhostLayerWidth>(this);
    #endif
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
    unsigned int n_constraint = m_cdata->getN()+m_cdata->getNGhosts();
    m_cmatrix.resize(n_constraint*n_constraint);
    m_cvec.resize(n_constraint);

    // populate the terms in the matrix vector equation
    fillMatrixVector(timestep);

    // check violations
    checkConstraints(timestep);

    // solve the matrix vector equation
    solveConstraints(timestep);

    // compute forces
    computeConstraintForces(timestep);

    if (m_prof)
        m_prof->pop();
    }

void ForceDistanceConstraint::fillMatrixVector(unsigned int timestep)
    {
    // fill the matrix in column-major order
    unsigned int n_constraint = m_cdata->getN()+m_cdata->getNGhosts();

    if (m_constraint_reorder)
        {
        // reset flag
        m_constraint_reorder = false;

        // resize lookup matrix
        m_sparse_idxlookup.resize(n_constraint*n_constraint);

        ArrayHandle<int> h_sparse_idxlookup(m_sparse_idxlookup, access_location::host, access_mode::overwrite);

        // reset lookup matrix values to -1
        for (unsigned int i = 0; i < n_constraint*n_constraint; ++i)
            {
            h_sparse_idxlookup.data[i] = -1;
            }
        }

    // access particle data
    ArrayHandle<Scalar4> h_pos(m_pdata->getPositions(), access_location::host, access_mode::read);
    ArrayHandle<Scalar4> h_vel(m_pdata->getVelocities(), access_location::host, access_mode::read);
    ArrayHandle<unsigned int> h_rtag(m_pdata->getRTags(), access_location::host, access_mode::read);
    ArrayHandle<Scalar4> h_netforce(m_pdata->getNetForce(), access_location::host, access_mode::read);

    // access matrix elements
    ArrayHandle<double> h_cmatrix(m_cmatrix, access_location::host, access_mode::overwrite);
    ArrayHandle<double> h_cvec(m_cvec, access_location::host, access_mode::overwrite);

    // clear matrix
    memset(h_cmatrix.data, 0, sizeof(double)*m_cmatrix.size());

    const BoxDim& box = m_pdata->getBox();

    unsigned int max_local = m_pdata->getN() + m_pdata->getNGhosts();
    for (unsigned int n = 0; n < n_constraint; ++n)
        {
        // lookup the tag of each of the particles participating in the constraint
        const ConstraintData::members_t constraint = m_cdata->getMembersByIndex(n);
        assert(constraint.tag[0] <= m_pdata->getMaximumTag());
        assert(constraint.tag[1] <= m_pdata->getMaximumTag());

        // transform a and b into indices into the particle data arrays
        // (MEM TRANSFER: 4 integers)
        unsigned int idx_a = h_rtag.data[constraint.tag[0]];
        unsigned int idx_b = h_rtag.data[constraint.tag[1]];

        if (idx_a >= max_local || idx_b >= max_local)
            {
            this->m_exec_conf->msg->error() << "constrain.distance(): constraint " <<
                constraint.tag[0] << " " << constraint.tag[1] << " incomplete." << std::endl << std::endl;
            throw std::runtime_error("Error in constraint calculation");
            }


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
            const ConstraintData::members_t constraint_m = m_cdata->getMembersByIndex(m);
            assert(constraint_m.tag[0] <= m_pdata->getMaximumTag());
            assert(constraint_m.tag[1] <= m_pdata->getMaximumTag());

            // transform a and b into indices into the particle data arrays
            // (MEM TRANSFER: 4 integers)
            unsigned int idx_m_a = h_rtag.data[constraint_m.tag[0]];
            unsigned int idx_m_b = h_rtag.data[constraint_m.tag[1]];
            assert(idx_m_a <= m_pdata->getN()+m_pdata->getNGhosts());
            assert(idx_m_b <= m_pdata->getN()+m_pdata->getNGhosts());

            if (idx_m_a >= max_local || idx_m_b >= max_local)
                {
                this->m_exec_conf->msg->error() << "constrain.distance(): constraint " <<
                    constraint_m.tag[0] << " " << constraint_m.tag[1] << " incomplete." << std::endl << std::endl;
                throw std::runtime_error("Error in constraint calculation");
                }

            vec3<Scalar> rm_a(h_pos.data[idx_m_a]);
            vec3<Scalar> rm_b(h_pos.data[idx_m_b]);
            vec3<Scalar> rm(rm_a-rm_b);

            // apply minimum image
            rm = box.minImage(rm);

            double delta(0.0);
            if (idx_m_a == idx_a)
                {
                delta += double(4.0)*dot(qn,rm)/ma;
                }
            if (idx_m_b == idx_a)
                {
                delta -= double(4.0)*dot(qn,rm)/ma;
                }
            if (idx_m_a == idx_b)
                {
                delta -= double(4.0)*dot(qn,rm)/mb;
                }
            if (idx_m_b == idx_b)
                {
                delta += double(4.0)*dot(qn,rm)/mb;
                }

            h_cmatrix.data[m*n_constraint+n] += delta;

            // update sparse matrix
            int k = m_sparse_idxlookup[m*n_constraint+n];

            if ( (k == -1 && delta != double(0.0))
                || (k != -1 && delta == double(0.0)))
                {
                m_condition.resetFlags(1);
                }

            if (k != -1)
                {
                // update sparse matrix value directly
                m_sparse.valuePtr()[k] = delta;
                }
            }

        // get constraint distance
        Scalar d = m_cdata->getValueByIndex(n);

        // check distance violation
        if (fast::sqrt(dot(rn,rn))-d >= m_rel_tol*d || std::isnan(dot(rn,rn)))
            {
            m_constraint_violated.resetFlags(n+1);
            }

        // fill vector component
        h_cvec.data[n] = (dot(qn,qn)-d*d)/m_deltaT/m_deltaT;
        h_cvec.data[n] += double(2.0)*dot(qn,vec3<Scalar>(h_netforce.data[idx_a])/ma
              -vec3<Scalar>(h_netforce.data[idx_b])/mb);
        }
    }

void ForceDistanceConstraint::checkConstraints(unsigned int timestep)
    {
    unsigned int n = m_constraint_violated.readFlags();
    if (n > 0)
        {
        ArrayHandle<unsigned int> h_group_tag(m_cdata->getTags(), access_location::host, access_mode::read);

        ConstraintData::members_t m = m_cdata->getMembersByIndex(n-1);
        unsigned int tag_a = m.tag[0];
        unsigned int tag_b = m.tag[1];
        Scalar d = m_cdata->getValueByIndex(n-1);

        ArrayHandle<Scalar4> h_pos(m_pdata->getPositions(), access_location::host, access_mode::read);
        ArrayHandle<unsigned int> h_rtag(m_pdata->getRTags(), access_location::host, access_mode::read);
        Scalar4 pos_a = h_pos.data[h_rtag.data[tag_a]];
        Scalar4 pos_b = h_pos.data[h_rtag.data[tag_b]];

        vec3<Scalar> rn = m_pdata->getBox().minImage(vec3<Scalar>(pos_a)-vec3<Scalar>(pos_b));
        m_exec_conf->msg->warning() << "Constraint " << h_group_tag.data[n-1] << " between particles "
            << tag_a << " and " << tag_b << " violated!" << std::endl
            << "(distance " << sqrt(dot(rn,rn)) << " exceeds " << d
            << " within relative tolerance " << m_rel_tol << ")" << std::endl;
        m_constraint_violated.resetFlags(0);
        }
    }

void ForceDistanceConstraint::solveConstraints(unsigned int timestep)
    {
    // use Eigen dense matrix algebra (slow for large matrices)
    typedef Matrix<double, Dynamic, Dynamic, ColMajor> matrix_t;
    typedef Matrix<double, Dynamic, 1> vec_t;
    typedef Map<matrix_t> matrix_map_t;
    typedef Map<vec_t> vec_map_t;

    unsigned int n_constraint = m_cdata->getN()+m_cdata->getNGhosts();

    // skip if zero constraints
    if (n_constraint == 0) return;

    if (m_prof)
        m_prof->push("solve");

    // reallocate array of constraint forces
    m_lagrange.resize(n_constraint);

    unsigned int sparsity_pattern_changed = m_condition.readFlags();

    if (sparsity_pattern_changed)
        {
        m_exec_conf->msg->notice(6) << "ForceDistanceConstraint: sparsity pattern changed. Solving on CPU" << std::endl;

        // reset flags
        m_condition.resetFlags(0);

        if (m_prof)
            m_prof->push("LU");

        // access matrix
        ArrayHandle<double> h_cmatrix(m_cmatrix, access_location::host, access_mode::read);

        // wrap array
        matrix_map_t map_matrix(h_cmatrix.data, n_constraint,n_constraint);

        // sparsity pattern changed
        m_sparse = map_matrix.sparseView();

            {
            ArrayHandle<int> h_sparse_idxlookup(m_sparse_idxlookup, access_location::host, access_mode::overwrite);

            // reset lookup matrix values to -1
            for (unsigned int i = 0; i < n_constraint*n_constraint; ++i)
                {
                h_sparse_idxlookup.data[i] = -1;
                }

            // construct lookup table
            int *inner_non_zeros = m_sparse.innerNonZeroPtr();
            int *outer = m_sparse.outerIndexPtr();
            int *inner = m_sparse.innerIndexPtr();
            for (int i = 0; i < m_sparse.outerSize(); ++i)
                {
                int id = outer[i];
                int end;

                if(m_sparse.isCompressed())
                    end = outer[i+1];
                else
                    end = id + inner_non_zeros[i];

                for (; id < end; ++id)
                    {
                    unsigned int col = i;
                    unsigned int row = inner[id];

                    // set pointer to index in sparse_val
                    h_sparse_idxlookup.data[col*n_constraint+row] = id;
                    }
                }
            }

        // Compute the ordering permutation vector from the structural pattern of A
        m_sparse_solver.analyzePattern(m_sparse);

        if (m_prof)
            m_prof->pop();
        }


    if (m_prof)
        m_prof->push("refactor/solve");

    // Compute the numerical factorization
    m_sparse_solver.factorize(m_sparse);

    if (m_sparse_solver.info())
        {
        m_exec_conf->msg->error() << "Could not solve linear system of constraint equations." << std::endl;
        throw std::runtime_error("Error evaluating constraint forces.\n");
        }

    // access RHS and solution vector
    ArrayHandle<double> h_cvec(m_cvec, access_location::host, access_mode::read);
    ArrayHandle<double> h_lagrange(m_lagrange, access_location::host, access_mode::overwrite);
    vec_map_t map_vec(h_cvec.data, n_constraint, 1);
    vec_map_t map_lagrange(h_lagrange.data,n_constraint, 1);

    //Use the factors to solve the linear system
    map_lagrange = m_sparse_solver.solve(map_vec);

    if (m_prof)
        m_prof->pop();

    if (m_prof)
        m_prof->pop();
    }

void ForceDistanceConstraint::computeConstraintForces(unsigned int timestep)
    {
    ArrayHandle<double> h_lagrange(m_lagrange, access_location::host, access_mode::read);

    // access particle data arrays
    ArrayHandle<Scalar4> h_pos(m_pdata->getPositions(), access_location::host, access_mode::read);
    ArrayHandle<unsigned int> h_rtag(m_pdata->getRTags(), access_location::host, access_mode::read);

    // access force and virial arrays
    ArrayHandle<Scalar4> h_force(m_force, access_location::host, access_mode::overwrite);
    ArrayHandle<Scalar> h_virial(m_virial, access_location::host, access_mode::overwrite);

    const BoxDim& box = m_pdata->getBox();

    unsigned int n_ptl = m_pdata->getN();

    // reset force array
    memset(h_force.data,0,sizeof(Scalar4)*n_ptl);
    memset(h_virial.data,0,sizeof(Scalar)*6*m_virial_pitch);

    unsigned int n_constraint = m_cdata->getN()+m_cdata->getNGhosts();

    // copy output to force array
    for (unsigned int n = 0; n < n_constraint; ++n)
        {
        // lookup the tag of each of the particles participating in the constraint
        const ConstraintData::members_t constraint = m_cdata->getMembersByIndex(n);
        assert(constraint.tag[0] <= m_pdata->getMaximumTag());
        assert(constraint.tag[1] <= m_pdata->getMaximumTag());

        // transform a and b into indices into the particle data arrays
        unsigned int idx_a = h_rtag.data[constraint.tag[0]];
        unsigned int idx_b = h_rtag.data[constraint.tag[1]];
        assert(idx_a < m_pdata->getN()+m_pdata->getNGhosts());
        assert(idx_b < m_pdata->getN()+m_pdata->getNGhosts());

        vec3<Scalar> ra(h_pos.data[idx_a]);
        vec3<Scalar> rb(h_pos.data[idx_b]);
        vec3<Scalar> rn(ra-rb);

        // apply minimum image
        rn = box.minImage(rn);

        // virial
        Scalar virialxx = -(Scalar) h_lagrange.data[n]*rn.x*rn.x;
        Scalar virialxy =- (Scalar) h_lagrange.data[n]*rn.x*rn.y;
        Scalar virialxz = -(Scalar) h_lagrange.data[n]*rn.x*rn.z;
        Scalar virialyy = -(Scalar) h_lagrange.data[n]*rn.y*rn.y;
        Scalar virialyz = -(Scalar) h_lagrange.data[n]*rn.y*rn.z;
        Scalar virialzz = -(Scalar) h_lagrange.data[n]*rn.z*rn.z;

        // if idx is local
        if (idx_a < n_ptl)
            {
            vec3<Scalar> f(h_force.data[idx_a]);
            f -= Scalar(2.0)*(Scalar)h_lagrange.data[n]*rn;
            h_force.data[idx_a] = make_scalar4(f.x,f.y,f.z,Scalar(0.0));

            h_virial.data[0*m_virial_pitch+idx_a] += virialxx;
            h_virial.data[1*m_virial_pitch+idx_a] += virialxy;
            h_virial.data[2*m_virial_pitch+idx_a] += virialxz;
            h_virial.data[3*m_virial_pitch+idx_a] += virialyy;
            h_virial.data[4*m_virial_pitch+idx_a] += virialyz;
            h_virial.data[5*m_virial_pitch+idx_a] += virialzz;
            }
        if (idx_b < n_ptl)
            {
            vec3<Scalar> f(h_force.data[idx_b]);
            f += Scalar(2.0)*(Scalar)h_lagrange.data[n]*rn;
            h_force.data[idx_b] = make_scalar4(f.x,f.y,f.z,Scalar(0.0));

            h_virial.data[0*m_virial_pitch+idx_b] += virialxx;
            h_virial.data[1*m_virial_pitch+idx_b] += virialxy;
            h_virial.data[2*m_virial_pitch+idx_b] += virialxz;
            h_virial.data[3*m_virial_pitch+idx_b] += virialyy;
            h_virial.data[4*m_virial_pitch+idx_b] += virialyz;
            h_virial.data[5*m_virial_pitch+idx_b] += virialzz;

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

    // request communication of particle forces
    flags[comm_flag::net_force] = 1;

    flags |= MolecularForceCompute::getRequestedCommFlags(timestep);

    return flags;
    }
#endif

//! Return maximum extent of molecule
Scalar ForceDistanceConstraint::dfs(unsigned int iconstraint, unsigned int molecule, std::vector<int>& visited,
    unsigned int *label, std::vector<ConstraintData::members_t>& groups, std::vector<Scalar>& length)
    {
    assert(iconstraint < groups.size());

    // don't mark constraints already visited
    assert(visited.size() > iconstraint);
    if (visited[iconstraint])
        {
        return Scalar(0.0);
        }

    // mark this constraint as visited
    visited[iconstraint] = 1;

    const ConstraintData::members_t constraint = groups[iconstraint];
    assert(constraint.tag[0] <= m_pdata->getMaximumTag());
    assert(constraint.tag[1] <= m_pdata->getMaximumTag());

    label[constraint.tag[0]] = molecule;
    label[constraint.tag[1]] = molecule;

    // NOTE: this loop could be optimized with a reverse-lookup table ptl idx -> constraint
    assert(iconstraint < length.size());
    Scalar dmax = length[iconstraint];

    for (unsigned int jconstraint = 0; jconstraint < groups.size(); ++jconstraint)
        {
        ConstraintData::members_t tags_j = groups[jconstraint];

        if (iconstraint == jconstraint) continue;

        if (tags_j.tag[0] == constraint.tag[0] || tags_j.tag[1] == constraint.tag[0] ||
            tags_j.tag[0] == constraint.tag[1] || tags_j.tag[1] == constraint.tag[1])
            {
            // recursively mark connected constraint with current label
            dmax += dfs(jconstraint, molecule, visited, label, groups, length);
            }
        }

    return dmax;
    }

Scalar ForceDistanceConstraint::askGhostLayerWidth(unsigned int type)
    {
    // only rebuild global tag list if necessary
    if (m_constraints_added_removed)
        {
        assignMoleculeTags();
        m_constraints_added_removed = false;
        }

    return m_d_max;
    }

void ForceDistanceConstraint::assignMoleculeTags()
    {
    ConstraintData::Snapshot snap;

    // take a global constraints snapshot
    m_cdata->takeSnapshot(snap);

    // broadcast constraint information
    std::vector<ConstraintData::members_t> groups = snap.groups;
    std::vector<Scalar> length = snap.val;

    #ifdef ENABLE_MPI
    if (m_comm)
        {
        bcast(groups, 0, m_exec_conf->getMPICommunicator());
        bcast(length, 0, m_exec_conf->getMPICommunicator());
        }
    #endif

    // walk through the global constraints and connect molecules

    unsigned int nconstraint_global = snap.size;
    std::vector<int> visited(nconstraint_global,0);

    // label per ptl (-1 == no label)
    m_molecule_tag.resize(m_pdata->getNGlobal());

    ArrayHandle<unsigned int> h_molecule_tag(m_molecule_tag, access_location::host, access_mode::overwrite);

    // reset labels
    unsigned int nptl = m_pdata->getNGlobal();
    for (unsigned int i = 0; i < nptl; ++i)
        {
        h_molecule_tag.data[i] = NO_MOLECULE;
        }

    int molecule = 0;

    // maximum molecule diameter
    m_d_max = Scalar(0.0);

        {
        // label ptls by connected component index
        for (unsigned int iconstraint = 0; iconstraint < nconstraint_global; ++iconstraint)
            {
            if (! visited[iconstraint])
                {
                // depth first search
                Scalar d = dfs(iconstraint, molecule++, visited, h_molecule_tag.data, groups, length);
                if (d > m_d_max)
                    {
                    m_d_max = d;
                    }
                }
            }
        }

    m_exec_conf->msg->notice(6) << "Maximum constraint length: " << m_d_max << std::endl;
    m_n_molecules_global = molecule;
    }

void export_ForceDistanceConstraint(py::module& m)
    {
    py::class_< ForceDistanceConstraint, std::shared_ptr<ForceDistanceConstraint> >(m, "ForceDistanceConstraint", py::base<MolecularForceCompute>())
        .def(py::init< std::shared_ptr<SystemDefinition> >())
        .def("setRelativeTolerance", &ForceDistanceConstraint::setRelativeTolerance)
    ;
    }
