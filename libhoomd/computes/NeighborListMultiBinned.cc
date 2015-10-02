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

// Maintainer: mphoward

/*! \file NeighborListMultiBinned.cc
    \brief Defines NeighborListMultiBinned
*/

#include "NeighborListMultiBinned.h"

#ifdef ENABLE_MPI
#include "Communicator.h"
#endif

#include <boost/python.hpp>
#include <boost/bind.hpp>

using namespace std;
using namespace boost::python;

NeighborListMultiBinned::NeighborListMultiBinned(boost::shared_ptr<SystemDefinition> sysdef,
                                                 Scalar r_cut,
                                                 Scalar r_buff,
                                                 boost::shared_ptr<CellList> cl)
    : NeighborList(sysdef, r_cut, r_buff), m_cl(cl), m_rcut_change(true)
    {
    m_exec_conf->msg->notice(5) << "Constructing NeighborListMultiBinned" << endl;

    // create a default cell list if one was not specified
    if (!m_cl)
        m_cl = boost::shared_ptr<CellList>(new CellList(sysdef));

    m_cl->setRadius(1);
    m_cl->setComputeTDB(true);
    m_cl->setFlagIndex();
    
    // call this class's special setRCut
    setRCut(r_cut, r_buff);

    // construct the cell list stencil generator for the current cell list
    m_cls = boost::shared_ptr<CellListStencil>(new CellListStencil(m_sysdef, m_cl));

    m_rcut_change_conn = connectRCutChange(boost::bind(&NeighborListMultiBinned::slotRCutChange, this));
    }

NeighborListMultiBinned::~NeighborListMultiBinned()
    {
    m_exec_conf->msg->notice(5) << "Destroying NeighborListMultiBinned" << endl;
    m_rcut_change_conn.disconnect();
    }

void NeighborListMultiBinned::setRCut(Scalar r_cut, Scalar r_buff)
    {
    NeighborList::setRCut(r_cut, r_buff);
    Scalar rmin = getMinRCut() + m_r_buff;
    if (m_diameter_shift)
        rmin += m_d_max - Scalar(1.0);
        
    m_cl->setNominalWidth(Scalar(0.5)*rmin);
    }

void NeighborListMultiBinned::setRCutPair(unsigned int typ1, unsigned int typ2, Scalar r_cut)
    {
    NeighborList::setRCutPair(typ1,typ2,r_cut);
    
    Scalar rmin = getMinRCut() + m_r_buff;
    if (m_diameter_shift)
        rmin += m_d_max - Scalar(1.0);
        
    m_cl->setNominalWidth(Scalar(0.5)*rmin);
    }

void NeighborListMultiBinned::setMaximumDiameter(Scalar d_max)
    {
    NeighborList::setMaximumDiameter(d_max);

    Scalar rmin = getMinRCut() + m_r_buff;
    if (m_diameter_shift)
        rmin += m_d_max - Scalar(1.0);
        
    m_cl->setNominalWidth(Scalar(0.5)*rmin);
    }

void NeighborListMultiBinned::updateRStencil()
    {
    ArrayHandle<Scalar> h_rcut_max(m_rcut_max, access_location::host, access_mode::read);
    std::vector<Scalar> rstencil(m_pdata->getNTypes(), -1.0);
    for (unsigned int cur_type=0; cur_type < m_pdata->getNTypes(); ++cur_type)
        {
        Scalar rcut = h_rcut_max.data[cur_type];
        if (rcut > Scalar(0.0))
            {
            Scalar rlist = rcut + m_r_buff;
            if (m_diameter_shift)
                rlist += m_d_max - Scalar(1.0);
            rstencil[cur_type] = rlist;
            }
        }
    m_cls->setRStencil(rstencil);
    }

void NeighborListMultiBinned::buildNlist(unsigned int timestep)
    {
    m_cl->compute(timestep);

    // update the stencil radii if there was a change
    if (m_rcut_change)
        {
        updateRStencil();
        m_rcut_change = false;
        }
    m_cls->compute(timestep);

    uint3 dim = m_cl->getDim();
    Scalar3 ghost_width = m_cl->getGhostWidth();

    if (m_prof)
        m_prof->push(m_exec_conf, "compute");

    // acquire the particle data and box dimension
    ArrayHandle<Scalar4> h_pos(m_pdata->getPositions(), access_location::host, access_mode::read);
    ArrayHandle<unsigned int> h_body(m_pdata->getBodies(), access_location::host, access_mode::read);
    ArrayHandle<Scalar> h_diameter(m_pdata->getDiameters(), access_location::host, access_mode::read);

    const BoxDim& box = m_pdata->getBox();
    Scalar3 nearest_plane_distance = box.getNearestPlaneDistance();

    // validate that the cutoff fits inside the box
    Scalar rmax = getMaxRCut() + m_r_buff;
    if (m_diameter_shift)
        rmax += m_d_max - Scalar(1.0);

    // get periodic flags
    uchar3 periodic = box.getPeriodic();

    if ((periodic.x && nearest_plane_distance.x <= rmax * 2.0) ||
        (periodic.y && nearest_plane_distance.y <= rmax * 2.0) ||
        (this->m_sysdef->getNDimensions() == 3 && periodic.z && nearest_plane_distance.z <= rmax * 2.0))
        {
        m_exec_conf->msg->error() << "nlist: Simulation box is too small! Particles would be interacting with themselves." << endl;
        throw runtime_error("Error updating neighborlist bins");
        }
        
    // access the rlist data
    ArrayHandle<Scalar> h_r_cut(m_r_cut, access_location::host, access_mode::read);

    // access the cell list data arrays
    ArrayHandle<unsigned int> h_cell_size(m_cl->getCellSizeArray(), access_location::host, access_mode::read);
    ArrayHandle<Scalar4> h_cell_xyzf(m_cl->getXYZFArray(), access_location::host, access_mode::read);
    ArrayHandle<Scalar4> h_cell_tdb(m_cl->getTDBArray(), access_location::host, access_mode::read);
    ArrayHandle<Scalar4> h_stencil(m_cls->getStencils(), access_location::host, access_mode::read);
    ArrayHandle<unsigned int> h_n_stencil(m_cls->getStencilSizes(), access_location::host, access_mode::read);
    const Index2D& stencil_idx = m_cls->getStencilIndexer();

    // access the neighbor list data
    ArrayHandle<unsigned int> h_head_list(m_head_list, access_location::host, access_mode::read);
    ArrayHandle<unsigned int> h_Nmax(m_Nmax, access_location::host, access_mode::read);
    ArrayHandle<unsigned int> h_conditions(m_conditions, access_location::host, access_mode::readwrite);
    ArrayHandle<unsigned int> h_nlist(m_nlist, access_location::host, access_mode::overwrite);
    ArrayHandle<unsigned int> h_n_neigh(m_n_neigh, access_location::host, access_mode::overwrite);
    
    // access indexers
    Index3D ci = m_cl->getCellIndexer();
    Index2D cli = m_cl->getCellListIndexer();

    // for each local particle
    unsigned int nparticles = m_pdata->getN();

    for (int i = 0; i < (int)nparticles; i++)
        {
        unsigned int cur_n_neigh = 0;
     
        const Scalar3 my_pos = make_scalar3(h_pos.data[i].x, h_pos.data[i].y, h_pos.data[i].z);
        const unsigned int type_i = __scalar_as_int(h_pos.data[i].w);  
        const unsigned int body_i = h_body.data[i];
        const Scalar diam_i = h_diameter.data[i];
        
        const unsigned int Nmax_i = h_Nmax.data[type_i];
        const unsigned int head_idx_i = h_head_list.data[i];

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

        // loop through all neighboring bins
        unsigned int n_stencil = h_n_stencil.data[type_i];
        for (unsigned int cur_stencil = 0; cur_stencil < n_stencil; ++cur_stencil)
            {
            // compute the stenciled cell cartesian coordinates
            Scalar4 stencil = h_stencil.data[stencil_idx(cur_stencil, type_i)];
            int sib = ib + __scalar_as_int(stencil.x);
            int sjb = jb + __scalar_as_int(stencil.y);
            int skb = kb + __scalar_as_int(stencil.z);
            Scalar cell_dist2 = stencil.w;
            // wrap through the boundary
            if (periodic.x)
                {
                if (sib >= (int)dim.x) sib -= dim.x;
                else if (sib < 0) sib += dim.x;

                // wrapping and the stencil construction should ensure this is in bounds
                assert(sib >= 0 && sib < (int)dim.x);
                }
            else if (sib < 0 || sib >= (int)dim.x)
                {
                // in aperiodic systems the stencil could maybe extend out of the grid
                continue;
                }

            if (periodic.y)
                {
                if (sjb >= (int)dim.y) sjb -= dim.y;
                else if (sjb < 0) sjb += dim.y;

                assert(sjb >= 0 && sjb < (int)dim.y);
                }
            else if (sjb < 0 || sjb >= (int)dim.y)
                {
                continue;
                }

            if (periodic.z)
                {
                if (skb >= (int)dim.z) skb -= dim.z;
                else if (skb < 0) skb += dim.z;

                assert(skb >= 0 && skb < (int)dim.z);
                }
            else if (skb < 0 || skb >= (int)dim.z)
                {
                continue;
                }

            unsigned int neigh_cell = ci(sib, sjb, skb);

            // check against all the particles in that neighboring bin to see if it is a neighbor
            unsigned int size = h_cell_size.data[neigh_cell];
            for (unsigned int cur_offset = 0; cur_offset < size; cur_offset++)
                {
                // read in the particle type (diameter and body as well while we've got the Scalar4 in)
                const Scalar4& neigh_tdb = h_cell_tdb.data[cli(cur_offset, neigh_cell)];
                const unsigned int type_j = __scalar_as_int(neigh_tdb.x);
                const Scalar diam_j = neigh_tdb.y;
                const unsigned int body_j = __scalar_as_int(neigh_tdb.z);
    
                // skip any particles belonging to the same rigid body if requested
                if (m_filter_body && body_i != NO_BODY && body_i == body_j) continue;

                // read cutoff and skip if pair is inactive
                Scalar r_cut = h_r_cut.data[m_typpair_idx(type_i,type_j)];
                if (r_cut <= Scalar(0.0)) continue;

                // compute the rlist based on the particle type we're interacting with
                Scalar r_list = r_cut + m_r_buff;
                Scalar sqshift = Scalar(0.0);
                if (m_diameter_shift)
                    {
                    const Scalar delta = (diam_i + diam_j) * Scalar(0.5) - Scalar(1.0);
                    // r^2 < (r_list + delta)^2
                    // r^2 < r_listsq + delta^2 + 2*r_list*delta
                    sqshift = (delta + Scalar(2.0) * r_list) * delta;
                    }
                Scalar r_listsq = r_list*r_list + sqshift;

                // compare the check distance to the minimum cell distance, and pass without distance check if unnecessary
                if (cell_dist2 > r_listsq) continue;

                // only load in the particle position and id if distance check is satisfied
                const Scalar4& neigh_xyzf = h_cell_xyzf.data[cli(cur_offset, neigh_cell)];
                unsigned int cur_neigh = __scalar_as_int(neigh_xyzf.w);

                // a particle cannot neighbor itself
                if (i == (int)cur_neigh) continue;

                Scalar3 neigh_pos = make_scalar3(neigh_xyzf.x, neigh_xyzf.y, neigh_xyzf.z);
                Scalar3 dx = my_pos - neigh_pos;
                dx = box.minImage(dx);
                    
                Scalar dr_sq = dot(dx,dx);

                if (dr_sq <= r_listsq)
                    {
                    if (m_storage_mode == full || i < (int)cur_neigh)
                        {
                        // local neighbor
                        if (cur_n_neigh < Nmax_i)
                            {
                            h_nlist.data[head_idx_i + cur_n_neigh] = cur_neigh;
                            }
                        else
                            h_conditions.data[type_i] = max(h_conditions.data[type_i], cur_n_neigh+1);

                        ++cur_n_neigh;
                        }
                    }
                }
            }

        h_n_neigh.data[i] = cur_n_neigh;
        }

    if (m_prof)
        m_prof->pop(m_exec_conf);
    }

void export_NeighborListMultiBinned()
    {
    class_<NeighborListMultiBinned, boost::shared_ptr<NeighborListMultiBinned>, bases<NeighborList>, boost::noncopyable >
        ("NeighborListMultiBinned", init< boost::shared_ptr<SystemDefinition>, Scalar, Scalar, boost::shared_ptr<CellList> >())
        ;
    }
