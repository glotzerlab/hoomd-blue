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

// Maintainer: joaander

/*! \file NeighborListBinned.cc
    \brief Defines NeighborListBinned
*/

#include "NeighborListBinned.h"

#include <boost/python.hpp>
using namespace boost::python;

#ifdef ENABLE_MPI
#include "Communicator.h"
#endif

NeighborListBinned::NeighborListBinned(boost::shared_ptr<SystemDefinition> sysdef,
                                       Scalar r_cut,
                                       Scalar r_buff,
                                       boost::shared_ptr<CellList> cl)
    : NeighborList(sysdef, r_cut, r_buff), m_cl(cl)
    {
    m_exec_conf->msg->notice(5) << "Constructing NeighborListBinned" << endl;

    // create a default cell list if one was not specified
    if (!m_cl)
        m_cl = boost::shared_ptr<CellList>(new CellList(sysdef));

    m_cl->setNominalWidth(r_cut + r_buff + m_d_max - Scalar(1.0));
    m_cl->setRadius(1);
    m_cl->setComputeTDB(false);
    m_cl->setFlagIndex();
    }

NeighborListBinned::~NeighborListBinned()
    {
    m_exec_conf->msg->notice(5) << "Destroying NeighborListBinned" << endl;

    }

void NeighborListBinned::setRCut(Scalar r_cut, Scalar r_buff)
    {
    NeighborList::setRCut(r_cut, r_buff);

    m_cl->setNominalWidth(r_cut + r_buff + m_d_max - Scalar(1.0));
    }

void NeighborListBinned::setMaximumDiameter(Scalar d_max)
    {
    NeighborList::setMaximumDiameter(d_max);

    // need to update the cell list settings appropriately
    m_cl->setNominalWidth(m_r_cut + m_r_buff + m_d_max - Scalar(1.0));
    }

void NeighborListBinned::buildNlist(unsigned int timestep)
    {
    m_cl->compute(timestep);

    uint3 dim = m_cl->getDim();
    Scalar3 ghost_width = m_cl->getGhostWidth();

    if (m_prof)
        m_prof->push(exec_conf, "compute");


    // acquire the particle data and box dimension
    ArrayHandle<Scalar4> h_pos(m_pdata->getPositions(), access_location::host, access_mode::read);
    ArrayHandle<unsigned int> h_body(m_pdata->getBodies(), access_location::host, access_mode::read);
    ArrayHandle<Scalar> h_diameter(m_pdata->getDiameters(), access_location::host, access_mode::read);

    const BoxDim& box = m_pdata->getBox();

    // start by creating a temporary copy of r_cut sqaured
    Scalar rmax = m_r_cut + m_r_buff;
    // add d_max - 1.0, if diameter filtering is not already taking care of it
    if (!m_filter_diameter)
        rmax += m_d_max - Scalar(1.0);
    Scalar rmaxsq = rmax*rmax;

    // access the cell list data arrays
    ArrayHandle<unsigned int> h_cell_size(m_cl->getCellSizeArray(), access_location::host, access_mode::read);
    ArrayHandle<Scalar4> h_cell_xyzf(m_cl->getXYZFArray(), access_location::host, access_mode::read);
    ArrayHandle<unsigned int> h_cell_adj(m_cl->getCellAdjArray(), access_location::host, access_mode::read);

    ArrayHandle<unsigned int> h_nlist(m_nlist, access_location::host, access_mode::overwrite);
    ArrayHandle<unsigned int> h_n_neigh(m_n_neigh, access_location::host, access_mode::overwrite);

    unsigned int conditions = 0;

    // access indexers
    Index3D ci = m_cl->getCellIndexer();
    Index2D cli = m_cl->getCellListIndexer();
    Index2D cadji = m_cl->getCellAdjIndexer();

    // get periodic flags
    uchar3 periodic = box.getPeriodic();

    // for each local particle
    unsigned int nparticles = m_pdata->getN();

#pragma omp parallel for schedule(dynamic, 100)
    for (int i = 0; i < (int)nparticles; i++)
        {
        unsigned int cur_n_neigh = 0;

        Scalar3 my_pos = make_scalar3(h_pos.data[i].x, h_pos.data[i].y, h_pos.data[i].z);
        unsigned int bodyi = h_body.data[i];
        Scalar di = h_diameter.data[i];

        // find the bin each particle belongs in
        Scalar3 f = box.makeFraction(my_pos,ghost_width);
        int ib = (unsigned int)(f.x * dim.x);
        int jb = (unsigned int)(f.y * dim.y);
        int kb = (unsigned int)(f.z * dim.z);

        // need to handle the case where the particle is exactly at the box hi
        if (ib == (int)dim.x && periodic.x)
            ib = 0;
        if (jb == (int)dim.y && periodic.y)
            jb = 0;
        if (kb == (int)dim.z && periodic.z)
            kb = 0;

        // identify the bin
        unsigned int my_cell = ci(ib,jb,kb);

        // loop through all neighboring bins
        for (unsigned int cur_adj = 0; cur_adj < cadji.getW(); cur_adj++)
            {
            unsigned int neigh_cell = h_cell_adj.data[cadji(cur_adj, my_cell)];

            // check against all the particles in that neighboring bin to see if it is a neighbor
            unsigned int size = h_cell_size.data[neigh_cell];
            for (unsigned int cur_offset = 0; cur_offset < size; cur_offset++)
                {
                Scalar4& cur_xyzf = h_cell_xyzf.data[cli(cur_offset, neigh_cell)];
                unsigned int cur_neigh = __scalar_as_int(cur_xyzf.w);

                Scalar3 neigh_pos = make_scalar3(cur_xyzf.x, cur_xyzf.y, cur_xyzf.z);

                Scalar3 dx = my_pos - neigh_pos;

                dx = box.minImage(dx);

                bool excluded = (i == (int)cur_neigh);

                if (m_filter_body && bodyi != NO_BODY)
                    excluded = excluded | (bodyi == h_body.data[cur_neigh]);

                Scalar sqshift = Scalar(0.0);
                if (m_filter_diameter)
                    {
                    // compute the shift in radius to accept neighbors based on their diameters
                    Scalar delta = (di + h_diameter.data[cur_neigh]) * Scalar(0.5) - Scalar(1.0);
                    // r^2 < (r_max + delta)^2
                    // r^2 < r_maxsq + delta^2 + 2*r_max*delta
                    sqshift = (delta + Scalar(2.0) * rmax) * delta;
                    }

                Scalar dr_sq = dot(dx,dx);

                if (dr_sq <= (rmaxsq + sqshift) && !excluded)
                    {
                    if (m_storage_mode == full || i < (int)cur_neigh)
                        {
                        // local neighbor
                        if (cur_n_neigh < m_nlist_indexer.getH())
                            h_nlist.data[m_nlist_indexer(i, cur_n_neigh)] = cur_neigh;
                        else
                            conditions = max(conditions, cur_n_neigh+1);

                        cur_n_neigh++;
                        }
                    }
                }
            }

        h_n_neigh.data[i] = cur_n_neigh;
        }

    // write out conditions
    m_conditions.resetFlags(conditions);

    if (m_prof)
        m_prof->pop(exec_conf);
    }

void export_NeighborListBinned()
    {
    class_<NeighborListBinned, boost::shared_ptr<NeighborListBinned>, bases<NeighborList>, boost::noncopyable >
                     ("NeighborListBinned", init< boost::shared_ptr<SystemDefinition>, Scalar, Scalar, boost::shared_ptr<CellList> >())
                     ;
    }
