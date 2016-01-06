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

#include <boost/python.hpp>

/*! \file MolecularForceCompute.cc
    \brief Contains code for the MolecularForceCompute class
*/

/*! \param sysdef SystemDefinition containing the ParticleData to compute forces on
*/
MolecularForceCompute::MolecularForceCompute(boost::shared_ptr<SystemDefinition> sysdef,
    boost::shared_ptr<NeighborList> nlist)
    : ForceConstraint(sysdef), m_nlist(nlist), m_molecule_list(m_exec_conf),
      m_molecule_length(m_exec_conf), m_molecule_ridx(m_exec_conf),
      m_d_max(0.0), m_last_d_max(0.0)
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

    m_d_max = getMaxDiameter();

    int result = 0;
    if (m_d_max - m_last_d_max > r_buff/Scalar(2.0))
        {
        result = 1;
        }

    // reduce over all ranks
    MPI_Allreduce(MPI_IN_PLACE, &result, 1, MPI_INT, MPI_MAX, m_exec_conf->getMPICommunicator());

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

    const BoxDim& box = m_pdata->getBox();

    for (unsigned int i = 0; i < m_molecule_indexer.getW(); ++i)
        {
        Scalar d_i(0.0);
        for (unsigned int j = 0; j < h_molecule_length.data[i]; ++j)
            {
            unsigned int idx_j = h_molecule_list.data[m_molecule_indexer(i,j)];

            assert(idx_j < m_pdata->getN() + m_pdata->getNGhosts());

            vec3<Scalar> r_j(h_postype.data[idx_j]);
            for (unsigned int k = 0; k < h_molecule_length.data[i]; ++k)
                {
                unsigned int idx_k = h_molecule_list.data[m_molecule_indexer(i,k)];

                assert(idx_k < m_pdata->getN() + m_pdata->getNGhosts());

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

    return d_max;
    }

void export_MolecularForceCompute()
    {
    class_< MolecularForceCompute, boost::shared_ptr<MolecularForceCompute>, bases<ForceConstraint>, boost::noncopyable >
    ("MolecularForceCompute", init< boost::shared_ptr<SystemDefinition>, boost::shared_ptr<NeighborList> >())
    ;
    }
