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
    : NeighborList(sysdef, r_cut, r_buff), m_cl(cl), m_compute_stencil(true)
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

    m_num_type_change_conn = m_pdata->connectNumTypesChange(boost::bind(&NeighborListMultiBinned::slotComputeStencil, this));
    m_box_change_conn = m_pdata->connectBoxChange(boost::bind(&NeighborListMultiBinned::slotComputeStencil, this));
    m_rcut_change_conn = connectRCutChange(boost::bind(&NeighborListMultiBinned::slotComputeStencil, this));
    }

NeighborListMultiBinned::~NeighborListMultiBinned()
    {
    m_exec_conf->msg->notice(5) << "Destroying NeighborListMultiBinned" << endl;
    m_num_type_change_conn.disconnect();
    m_box_change_conn.disconnect();
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

void NeighborListMultiBinned::calcStencil()
    {
    // compute the size of the bins in each dimension so that we know how big each is
    const uint3 dim = m_cl->getDim();
    const Scalar3 ghost_width = m_cl->getGhostWidth();
    const BoxDim& box = m_pdata->getBox();
    const Scalar3 L = box.getNearestPlaneDistance();

    // TODO: remove this requirement
    if (dim.x <= 2 || dim.y <= 2 || dim.z <= 2)
        {
        m_exec_conf->msg->error() << "nlist: multi requires 3 cells in each direction" << std::endl;
        throw std::runtime_error("Multi must have 3 cells in each direction");
        }

    const Scalar3 cell_size = make_scalar3((L.x + ghost_width.x) / Scalar(dim.x),
                                           (L.y + ghost_width.y) / Scalar(dim.y),
                                           (L.z + ghost_width.z) / Scalar(dim.z));

    // we have to have at least three cells
    Scalar r_list_max_max = getMaxRCut() + m_r_buff;
    if (m_diameter_shift)
        r_list_max_max += m_d_max - Scalar(1.0);
    int3 max_stencil_size = make_int3(static_cast<int>(ceil(r_list_max_max / cell_size.x)),
                                      static_cast<int>(ceil(r_list_max_max / cell_size.y)),
                                      static_cast<int>(ceil(r_list_max_max / cell_size.z)));
    if (max_stencil_size.x == 0) ++max_stencil_size.x;
    if (max_stencil_size.y == 0) ++max_stencil_size.y;
    if (max_stencil_size.z == 0) ++max_stencil_size.z;  
    if (m_sysdef->getNDimensions() == 2) max_stencil_size.z = 0; // this check is simple

    // compute the maximum number of bins in the stencil
    unsigned int max_n_stencil = (2*max_stencil_size.x+1)*(2*max_stencil_size.y+1)*(2*max_stencil_size.z+1);
    
    m_stencil_idx = Index2D(max_n_stencil, m_pdata->getNTypes());
    // reallocate the stencil memory
        {
        GPUArray<unsigned int> n_stencil(m_pdata->getNTypes(), m_exec_conf);
        m_n_stencil.swap(n_stencil);

        GPUArray<Scalar4> stencil(max_n_stencil*m_pdata->getNTypes(), m_exec_conf);
        m_stencil.swap(stencil);
        }

    ArrayHandle<Scalar> h_rcut_max(m_rcut_max, access_location::host, access_mode::read);
    ArrayHandle<Scalar4> h_stencil(m_stencil, access_location::host, access_mode::overwrite);
    ArrayHandle<unsigned int> h_n_stencil(m_n_stencil, access_location::host, access_mode::overwrite);
    for (unsigned int cur_type=0; cur_type < m_pdata->getNTypes(); ++cur_type)
        {
        // compute the maximum distance for inclusion in the neighbor list for this type
        Scalar r_list_max = h_rcut_max.data[cur_type] + m_r_buff;
        if (m_diameter_shift)
            r_list_max += m_d_max - Scalar(1.0);
        Scalar r_listsq_max = r_list_max*r_list_max;

        // get the stencil size
        // TODO: abstract this as an inline function
        int3 stencil_size = make_int3(static_cast<int>(ceil(r_list_max / cell_size.x)),
                                      static_cast<int>(ceil(r_list_max / cell_size.y)),
                                      static_cast<int>(ceil(r_list_max / cell_size.z)));
        if (stencil_size.x == 0) ++stencil_size.x;
        if (stencil_size.y == 0) ++stencil_size.y;
        if (stencil_size.z == 0) ++stencil_size.z;  
        if (m_sysdef->getNDimensions() == 2) stencil_size.z = 0; // this check is simple

        // loop through the possible stencils
        unsigned int n_stencil_i = 0;
        for (int k=-stencil_size.z; k <= stencil_size.z; ++k)
            {
            for (int j=-stencil_size.y; j <= stencil_size.y; ++j)
                {
                for (int i=-stencil_size.x; i <= stencil_size.x; ++i)
                    {
                    // compute the distance to the closest point in the bin
                    Scalar3 dr = make_scalar3(0.0,0.0,0.0);
                    if (i > 0) dr.x = (i-1) * cell_size.x;
                    else if (i < 0) dr.x = (i+1) * cell_size.x;

                    if (j > 0) dr.y = (j-1) * cell_size.y;
                    else if (j < 0) dr.y = (j+1) * cell_size.y;

                    if (k > 0) dr.z = (k-1) * cell_size.z;
                    else if (k < 0) dr.z = (k+1) * cell_size.z;

                    Scalar dr2 = dot(dr, dr);

                    if (dr2 < r_listsq_max)
                        {
                        // could this be packed as a Scalar2 using a single int for the stencil?
                        // maybe with shift by +1 so that you never get negative numbers (that should work)
                        // still need to unwrap later
                        h_stencil.data[m_stencil_idx(n_stencil_i, cur_type)] = make_scalar4(__int_as_scalar(i),
                                                                                            __int_as_scalar(j),
                                                                                            __int_as_scalar(k),
                                                                                            dr2);
                        ++n_stencil_i;
                        }
                    }
                }
            }
        if (n_stencil_i > max_n_stencil)
            {
            m_exec_conf->msg->error() << "nlist: stencil size too bug, report bug" << std::endl;
            throw std::runtime_error("bug: stencil size too big");
            }
        h_n_stencil.data[cur_type] = n_stencil_i;
        }

    m_compute_stencil = false;
    }

void NeighborListMultiBinned::buildNlist(unsigned int timestep)
    {
    m_cl->compute(timestep);
    if (m_compute_stencil)
        {
        calcStencil();
        }

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

    if ((box.getPeriodic().x && nearest_plane_distance.x <= rmax * 2.0) ||
        (box.getPeriodic().y && nearest_plane_distance.y <= rmax * 2.0) ||
        (this->m_sysdef->getNDimensions() == 3 && box.getPeriodic().z && nearest_plane_distance.z <= rmax * 2.0))
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
    ArrayHandle<Scalar4> h_stencil(m_stencil, access_location::host, access_mode::read);
    ArrayHandle<unsigned int> h_n_stencil(m_n_stencil, access_location::host, access_mode::read);

    // access the neighbor list data
    ArrayHandle<unsigned int> h_head_list(m_head_list, access_location::host, access_mode::read);
    ArrayHandle<unsigned int> h_Nmax(m_Nmax, access_location::host, access_mode::read);
    ArrayHandle<unsigned int> h_conditions(m_conditions, access_location::host, access_mode::readwrite);
    ArrayHandle<unsigned int> h_nlist(m_nlist, access_location::host, access_mode::overwrite);
    ArrayHandle<unsigned int> h_n_neigh(m_n_neigh, access_location::host, access_mode::overwrite);
    
    // access indexers
    Index3D ci = m_cl->getCellIndexer();
    Index2D cli = m_cl->getCellListIndexer();

    // get periodic flags
    uchar3 periodic = box.getPeriodic();

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
            Scalar4 stencil = h_stencil.data[m_stencil_idx(cur_stencil, type_i)];
            int sib = ib + __scalar_as_int(stencil.x);
            int sjb = jb + __scalar_as_int(stencil.y);
            int skb = kb + __scalar_as_int(stencil.z);
            Scalar cell_dist2 = stencil.w;
            // wrap through the boundary
            if (periodic.x)
                {
                if (sib >= (int)dim.x) sib -= dim.x;
                else if (sib < 0) sib += dim.x;
                }
            if (periodic.y)
                {
                if (sjb >= (int)dim.y) sjb -= dim.y;
                else if (sjb < 0) sjb += dim.y;
                }
            if (periodic.z)
                {
                if (skb >= (int)dim.z) skb -= dim.z;
                else if (skb < 0) skb += dim.z;
                }
            unsigned int neigh_cell = ci(sib, sjb, skb);

            // check against all the particles in that neighboring bin to see if it is a neighbor
            unsigned int size = h_cell_size.data[neigh_cell];
            for (unsigned int cur_offset = 0; cur_offset < size; cur_offset++)
                {
                // read in the particle type (diameter and body as well while we've got the Scalar4 in)
                const Scalar4& cur_tdb = h_cell_tdb.data[cli(cur_offset, neigh_cell)];
                const unsigned int type_j = __scalar_as_int(cur_tdb.x);
                const Scalar diam_j = cur_tdb.y;
                const unsigned int body_j = __scalar_as_int(cur_tdb.z);
    
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
                const Scalar4& cur_xyzf = h_cell_xyzf.data[cli(cur_offset, neigh_cell)];
                unsigned int cur_neigh = __scalar_as_int(cur_xyzf.w);

                // a particle cannot neighbor itself
                if (i == (int)cur_neigh) continue;

                Scalar3 neigh_pos = make_scalar3(cur_xyzf.x, cur_xyzf.y, cur_xyzf.z);
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

                        cur_n_neigh++;
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
