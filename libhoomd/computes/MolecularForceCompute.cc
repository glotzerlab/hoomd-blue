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

#include "MolecularForceCompute.h"

#include <string.h>
#include <map>

#include <boost/python.hpp>

/*! \file MolecularForceCompute.cc
    \brief Contains code for the MolecularForceCompute class
*/

/*! \param sysdef SystemDefinition containing the ParticleData to compute forces on
*/
MolecularForceCompute::MolecularForceCompute(boost::shared_ptr<SystemDefinition> sysdef,
    boost::shared_ptr<NeighborList> nlist)
    : ForceConstraint(sysdef), m_nlist(nlist), m_molecule_list(m_exec_conf),
      m_molecule_length(m_exec_conf), m_molecule_tag(m_exec_conf),
      m_molecule_idx(m_exec_conf), m_d_max(0.0), m_last_d_max(0.0), m_n_molecules_global(0),
      m_is_first_step(true)
    {
    }

//! Destructor
MolecularForceCompute::~MolecularForceCompute()
    {
    }

#ifdef ENABLE_MPI
bool MolecularForceCompute::askMigrateRequest(unsigned int timestep)
    {
    Scalar r_buff = m_nlist->getRBuff();

    if (m_is_first_step)
        {
        // only on the first time, initialize molecules BEFORE communication
        initMolecules();
        m_is_first_step = false;
        }

    // get maximum diameter among local molecules
    m_d_max = getMaxDiameter();

    bool result = false;
    if (m_d_max - m_last_d_max > r_buff/Scalar(2.0))
        {
        result = true;
        }

   return result;
   }
#endif

Scalar MolecularForceCompute::getMaxDiameter()
    {

    // iterate over molecules
    Scalar d_max(0.0);

    ArrayHandle<unsigned int> h_molecule_length(m_molecule_length, access_location::host, access_mode::read);
    ArrayHandle<unsigned int> h_molecule_list(m_molecule_list, access_location::host, access_mode::read);
    ArrayHandle<Scalar4> h_postype(m_pdata->getPositions(), access_location::host, access_mode::read);
    ArrayHandle<unsigned int> h_rtag(m_pdata->getRTags(), access_location::host, access_mode::read);

    const BoxDim& box = m_pdata->getBox();

    unsigned int n_local = m_pdata->getN();

    for (unsigned int i = 0; i < m_molecule_indexer.getW(); ++i)
        {
        Scalar d_i(0.0);
        for (unsigned int j = 0; j < h_molecule_length.data[i]; ++j)
            {
            unsigned int idx_j = h_molecule_list.data[m_molecule_indexer(i,j)];
            assert(idx_j < m_pdata->getN() + m_pdata->getNGhosts());

            // only take into account local molecule (fragments)
            if (idx_j >= n_local) continue;

            vec3<Scalar> r_j(h_postype.data[idx_j]);
            for (unsigned int k = j+1; k < h_molecule_length.data[i]; ++k)
                {
                unsigned int idx_k = h_molecule_list.data[m_molecule_indexer(i,k)];
                assert(idx_k < m_pdata->getN() + m_pdata->getNGhosts());

                if (idx_k >= n_local) continue;

                vec3<Scalar> r_k(h_postype.data[idx_k]);
                vec3<Scalar> r_jk = r_k - r_j;

                // apply minimum image
                r_jk = box.minImage(r_jk);

                Scalar d_jk = sqrt(dot(r_jk,r_jk));
                if (d_jk > d_i)
                    {
                    d_i = d_jk;
                    }
                }
            }
        if (d_i > d_max)
            {
            d_max = d_i;
            }
        }

    #ifdef ENABLE_MPI
    if (m_comm)
        {
        // reduce over all ranks
        MPI_Allreduce(MPI_IN_PLACE, &d_max, 1, MPI_HOOMD_SCALAR, MPI_MAX, m_exec_conf->getMPICommunicator());
        }
    #endif

    return d_max;
    }


void MolecularForceCompute::initMolecules()
    {
    // construct local molecule table
    unsigned int nptl_local = m_pdata->getN();

    ArrayHandle<unsigned int> h_molecule_tag(m_molecule_tag, access_location::host, access_mode::read);
    ArrayHandle<unsigned int> h_tag(m_pdata->getTags(), access_location::host, access_mode::read);

    std::map<unsigned int,unsigned int> local_molecule_tags;

    unsigned int n_local_molecules = 0;

    std::vector<unsigned int> local_molecule_idx(nptl_local, NO_MOLECULE);

    // resize molecule lookup
    m_molecule_idx.resize(m_n_molecules_global);

    // identify local molecules and assign local indices to global molecule tags
    ArrayHandle<unsigned int> h_molecule_idx(m_molecule_idx, access_location::host, access_mode::overwrite);
    for (unsigned int i = 0; i < nptl_local; ++i)
        {
        unsigned int tag = h_tag.data[i];
        assert(tag <= m_pdata->getMaximumTag());

        unsigned int mol_tag = h_molecule_tag.data[tag];
        if (mol_tag == NO_MOLECULE) continue;

        std::map<unsigned int,unsigned int>::iterator it = local_molecule_tags.find(mol_tag);
        if (it == local_molecule_tags.end())
            {
            // insert element
            it = local_molecule_tags.insert(std::make_pair(mol_tag,n_local_molecules++)).first;
            unsigned int mol_idx = it->second;
            h_molecule_idx.data[mol_tag] = mol_idx;
            }

        local_molecule_idx[i] = it->second;
        }

    m_molecule_length.resize(n_local_molecules);

    ArrayHandle<unsigned int> h_molecule_length(m_molecule_length, access_location::host, access_mode::overwrite);

    // reset lengths
    for (unsigned int imol = 0; imol < n_local_molecules; ++imol)
        {
        h_molecule_length.data[imol] = 0;
        }

    // count molecule lengths
    for (unsigned int i = 0; i < nptl_local; ++i)
        {
        assert(i < label.size());
        unsigned int molecule_i = local_molecule_idx[i];
        if (molecule_i != NO_MOLECULE)
            {
            h_molecule_length.data[molecule_i]++;
            }
        }

    // find maximum length
    unsigned nmax = 0;
    for (unsigned int imol = 0; imol < n_local_molecules; ++imol)
        {
        if (h_molecule_length.data[imol] > nmax)
            {
            nmax = h_molecule_length.data[imol];
            }
        }

    // set up indexer
    m_molecule_indexer = Index2D(n_local_molecules, nmax);

    // resize molecule list
    m_molecule_list.resize(m_molecule_indexer.getNumElements());

    // reset lengths again
    for (unsigned int imol = 0; imol < n_local_molecules; ++imol)
        {
        h_molecule_length.data[imol] = 0;
        }

    // fill molecule list
    ArrayHandle<unsigned int> h_molecule_list(m_molecule_list, access_location::host, access_mode::overwrite);

    for (unsigned int iptl = 0; iptl < nptl_local; ++iptl)
        {
        assert(iptl < label.size());
        unsigned int i_mol = local_molecule_idx[iptl];

        if (i_mol != NO_MOLECULE)
            {
            unsigned int n = h_molecule_length.data[i_mol]++;
            h_molecule_list.data[m_molecule_indexer(i_mol,n)] = iptl;
            }
        }
    }

#ifdef ENABLE_MPI
void MolecularForceCompute::addGhostParticles(const GPUArray<unsigned int>& plans)
    {
    // store current value of maximum diameter at every ghost exchange
    m_last_d_max = m_d_max;

    // init local molecules
    initMolecules();

    ArrayHandle<unsigned int> h_tag(m_pdata->getTags(), access_location::host, access_mode::read);
    ArrayHandle<unsigned int> h_molecule_tag(m_molecule_tag, access_location::host, access_mode::read);

    ArrayHandle<unsigned int> h_plans(plans, access_location::host, access_mode::readwrite);

    unsigned int nptl_local = m_pdata->getN();

    std::set<unsigned int> ghost_molecules;

    ArrayHandle<unsigned int> h_molecule_idx(m_molecule_idx, access_location::host, access_mode::read);

    std::vector<unsigned int> ghost_molecule_plans(m_molecule_list.size(),0);
    std::vector<bool> not_completely_in_ghost_layer(m_molecule_list.size(), false);

    // identify ghost molecules and combine plans
    for (unsigned int i = 0; i < nptl_local; ++i)
        {
        // only consider particles sent as ghosts
        unsigned int plan = h_plans.data[i];

        unsigned int tag = h_tag.data[i];
        assert(tag <= m_pdata->getMaximumTag());

        unsigned int molecule_tag = h_molecule_tag.data[tag];
        assert(molecule_tag < m_n_molecules_global);

        if (molecule_tag != NO_MOLECULE)
            {
            unsigned int mol_idx = h_molecule_idx.data[molecule_tag];

            if (plan != 0)
                {
                ghost_molecules.insert(mol_idx);
                }
            else
                {
                // molecules have to reside completely in the ghost layer
                not_completely_in_ghost_layer[mol_idx] = true;
                }

            // combine plan
            ghost_molecule_plans[mol_idx] |= plan;
            }
        }


    ArrayHandle<unsigned int> h_molecule_list(m_molecule_list, access_location::host, access_mode::read);
    ArrayHandle<unsigned int> h_molecule_length(m_molecule_length, access_location::host, access_mode::read);

    // for every ghost molecule, combine plans
    for (std::set<unsigned int>::iterator it = ghost_molecules.begin(); it != ghost_molecules.end(); ++it)
        {
        unsigned int mol_idx = *it;

        for (unsigned int j = 0; j < h_molecule_length.data[mol_idx]; ++j)
            {
            unsigned int member_idx = h_molecule_list.data[m_molecule_indexer(mol_idx, j)];

            assert(member_idx <= m_pdata->getN());

            // update plan
            h_plans.data[member_idx] |= ghost_molecule_plans[mol_idx];
            }
        }
    }
#endif

void export_MolecularForceCompute()
    {
    class_< MolecularForceCompute, boost::shared_ptr<MolecularForceCompute>, bases<ForceConstraint>, boost::noncopyable >
    ("MolecularForceCompute", init< boost::shared_ptr<SystemDefinition>, boost::shared_ptr<NeighborList> >())
    ;
    }
