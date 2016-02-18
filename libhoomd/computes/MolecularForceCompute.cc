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
#include <boost/bind.hpp>

/*! \file MolecularForceCompute.cc
    \brief Contains code for the MolecularForceCompute class
*/

/*! \param sysdef SystemDefinition containing the ParticleData to compute forces on
*/
MolecularForceCompute::MolecularForceCompute(boost::shared_ptr<SystemDefinition> sysdef)
    : ForceConstraint(sysdef), m_molecule_list(m_exec_conf),
      m_molecule_length(m_exec_conf), m_molecule_tag(m_exec_conf),
      m_molecule_idx(m_exec_conf), m_n_molecules_global(0)
    {
    }

//! Destructor
MolecularForceCompute::~MolecularForceCompute()
    {
    }

void MolecularForceCompute::initMolecules()
    {
    // return early if no molecules are defined
    if (!m_n_molecules_global) return;

    m_exec_conf->msg->notice(7) << "MolecularForceCompute initializing molecule table" << std::endl;

    // construct local molecule table
    unsigned int nptl_local = m_pdata->getN() + m_pdata->getNGhosts();

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
        assert(tag < m_molecule_tag.getNumElements());

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

    // sort local molecules by ptl tag
    std::vector< std::set<unsigned int> > local_molecules_sorted_by_tag(n_local_molecules);

    for (unsigned int iptl = 0; iptl < nptl_local; ++iptl)
        {
        unsigned int i_mol = local_molecule_idx[iptl];

        if (i_mol != NO_MOLECULE)
            {
            local_molecules_sorted_by_tag[i_mol].insert(h_tag.data[iptl]);
            }
        }

    // fill molecule list
    ArrayHandle<unsigned int> h_molecule_list(m_molecule_list, access_location::host, access_mode::overwrite);
    ArrayHandle<unsigned int> h_rtag(m_pdata->getRTags(), access_location::host, access_mode::read);

    unsigned int i_mol = 0;
    for (std::vector< std::set<unsigned int> >::iterator it_mol = local_molecules_sorted_by_tag.begin();
        it_mol != local_molecules_sorted_by_tag.end(); ++it_mol)
        {
        for (std::set<unsigned int>::iterator it_tag = it_mol->begin(); it_tag != it_mol->end(); ++it_tag)
            {
            unsigned int n = h_molecule_length.data[i_mol]++;
            unsigned int ptl_idx = h_rtag.data[*it_tag];
            assert(ptl_idx < m_pdata->getN() + m_pdata->getNGhosts());
            h_molecule_list.data[m_molecule_indexer(i_mol, n)] = ptl_idx;
            }
        i_mol ++;
        }
    }

#ifdef ENABLE_MPI
/* This method adds particles that belong to the same molecule as a particle that is being
   communicated to the communication list. This is necessary for rigid bodies to ensure
   that the central particle always gets communicated.
*/
void MolecularForceCompute::addGhostParticles(const GPUArray<unsigned int>& plans)
    {
    return;
    if (!m_n_molecules_global) return;

    ArrayHandle<unsigned int> h_tag(m_pdata->getTags(), access_location::host, access_mode::read);
    ArrayHandle<unsigned int> h_rtag(m_pdata->getRTags(), access_location::host, access_mode::read);
    ArrayHandle<unsigned int> h_molecule_tag(m_molecule_tag, access_location::host, access_mode::read);

    ArrayHandle<unsigned int> h_plans(plans, access_location::host, access_mode::readwrite);

    unsigned int nptl_local = m_pdata->getN();

    std::multimap<unsigned int, unsigned int> ghost_molecules;
    std::map<unsigned int, unsigned int> ghost_molecule_plans;

    // identify ghost molecules and combine plans
    for (unsigned int i = 0; i < nptl_local; ++i)
        {
        // only consider particles sent as ghosts
        unsigned int plan = h_plans.data[i];

        unsigned int tag = h_tag.data[i];
        assert(tag <= m_pdata->getMaximumTag());

        unsigned int molecule_tag = h_molecule_tag.data[tag];

        if (molecule_tag != NO_MOLECULE)
            {
            assert(molecule_tag < m_n_molecules_global);

            unsigned int mol_plan = 0;
            std::map<unsigned int, unsigned int>::iterator it = ghost_molecule_plans.find(molecule_tag);
            bool update = false;
            if (it != ghost_molecule_plans.end())
                {
                mol_plan = it->second;
                update = true;
                }

            if (plan != 0)
                {
                ghost_molecules.insert(std::make_pair(molecule_tag, tag));

                // combine plan
                if (update)
                    {
                    it->second = mol_plan | plan;
                    }
                else
                    {
                    ghost_molecule_plans.insert(std::make_pair(molecule_tag, mol_plan | plan));
                    }
                }
            }
        }

    // complete ghost molecules by adding those ptls without own plans
    for (unsigned int i = 0; i < nptl_local; ++i)
        {
        // only consider particles sent as ghosts
        unsigned int plan = h_plans.data[i];

        unsigned int tag = h_tag.data[i];
        assert(tag <= m_pdata->getMaximumTag());

        unsigned int molecule_tag = h_molecule_tag.data[tag];

        if (molecule_tag != NO_MOLECULE)
            {
            assert(molecule_tag < m_n_molecules_global);

            unsigned int mol_plan = 0;
            std::map<unsigned int, unsigned int>::iterator it = ghost_molecule_plans.find(molecule_tag);
            if (it != ghost_molecule_plans.end())
                {
                mol_plan = it->second;
                }

            if (mol_plan == 0) continue;

            if (plan == 0)
                {
                ghost_molecules.insert(std::make_pair(molecule_tag, tag));
                }
            }
        }

    // for every ghost molecule, combine plans
    for (std::multimap<unsigned int, unsigned int>::iterator it = ghost_molecules.begin(); it != ghost_molecules.end(); ++it)
        {
        unsigned int mol_tag = it->first;
        unsigned int member_tag = it->second;
        assert(member_tag <= m_pdata->getMaximumTag());
        unsigned int member_idx = h_rtag.data[member_tag];
        assert(member_idx < m_pdata->getN());

        // get plan for molecule
        std::map<unsigned int, unsigned int>::iterator it_plan = ghost_molecule_plans.find(mol_tag);
        assert(it_plan != ghost_molecule_plans.end());

        // update particle plan
        h_plans.data[member_idx] |= it_plan->second;
        }
    }
#endif

void export_MolecularForceCompute()
    {
    class_< MolecularForceCompute, boost::shared_ptr<MolecularForceCompute>, bases<ForceConstraint>, boost::noncopyable >
    ("MolecularForceCompute", init< boost::shared_ptr<SystemDefinition> >())
    ;
    }
