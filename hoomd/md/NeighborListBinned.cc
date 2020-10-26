// Copyright (c) 2009-2019 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.


// Maintainer: joaander

/*! \file NeighborListBinned.cc
    \brief Defines NeighborListBinned
*/

#include "NeighborListBinned.h"

#ifdef ENABLE_MPI
#include "hoomd/Communicator.h"
#endif


using namespace std;
namespace py = pybind11;

NeighborListBinned::NeighborListBinned(std::shared_ptr<SystemDefinition> sysdef,
                                       Scalar r_cut,
                                       Scalar r_buff,
                                       std::shared_ptr<CellList> cl)
    : NeighborList(sysdef, r_cut, r_buff), m_cl(cl)
    {
    m_exec_conf->msg->notice(5) << "Constructing NeighborListBinned" << endl;

    // create a default cell list if one was not specified
    if (!m_cl)
        m_cl = std::shared_ptr<CellList>(new CellList(sysdef));

    m_cl->setRadius(1);
    m_cl->setComputeXYZF(true);
    m_cl->setComputeTDB(false);
    m_cl->setFlagIndex();

    // call this class's special setRCut
    setRCut(r_cut, r_buff);
    }

NeighborListBinned::~NeighborListBinned()
    {
    m_exec_conf->msg->notice(5) << "Destroying NeighborListBinned" << endl;
    }

void NeighborListBinned::setRCut(Scalar r_cut, Scalar r_buff)
    {
    NeighborList::setRCut(r_cut, r_buff);
    Scalar rmax = getMaxRCut() + m_r_buff;
    if (m_diameter_shift)
        rmax += m_d_max - Scalar(1.0);

    m_cl->setNominalWidth(rmax);
    }

void NeighborListBinned::setRCutPair(unsigned int typ1, unsigned int typ2, Scalar r_cut)
    {
    NeighborList::setRCutPair(typ1,typ2,r_cut);

    Scalar rmax = getMaxRCut() + m_r_buff;
    if (m_diameter_shift)
        rmax += m_d_max - Scalar(1.0);

    m_cl->setNominalWidth(rmax);
    }

void NeighborListBinned::setMaximumDiameter(Scalar d_max)
    {
    NeighborList::setMaximumDiameter(d_max);

    // need to update the cell list settings appropriately
    Scalar rmax = getMaxRCut() + m_r_buff;
    if (m_diameter_shift)
        rmax += m_d_max - Scalar(1.0);

    m_cl->setNominalWidth(rmax);
    }

void NeighborListBinned::buildNlist(unsigned int timestep)
    {
    m_cl->compute(timestep);

    uint3 dim = m_cl->getDim();
    Scalar3 ghost_width = m_cl->getGhostWidth();

    if (m_prof)
        m_prof->push(m_exec_conf, "compute");

    // acquire the particle data and box dimension
    ArrayHandle<Scalar4> h_pos(m_pdata->getPositions(), access_location::host, access_mode::read);
    ArrayHandle<unsigned int> h_body(m_pdata->getBodies(), access_location::host, access_mode::read);
    ArrayHandle<Scalar> h_diameter(m_pdata->getDiameters(), access_location::host, access_mode::read);

    const BoxDim& box = m_pdata->getBox();

    // validate that the cutoff fits inside the box
    Scalar rmax = getMaxRCut() + m_r_buff;
    if (m_diameter_shift)
        rmax += m_d_max - Scalar(1.0);

    if (m_filter_body)
        {
        // add the maximum diameter of all composite particles
        Scalar max_d_comp = m_pdata->getMaxCompositeParticleDiameter();
        rmax += 0.5*max_d_comp;
        }

    // access the rlist data
    ArrayHandle<Scalar> h_r_cut(m_r_cut, access_location::host, access_mode::read);
    ArrayHandle<Scalar> h_r_listsq(m_r_listsq, access_location::host, access_mode::read);

    // access the cell list data arrays
    ArrayHandle<unsigned int> h_cell_size(m_cl->getCellSizeArray(), access_location::host, access_mode::read);
    ArrayHandle<Scalar4> h_cell_xyzf(m_cl->getXYZFArray(), access_location::host, access_mode::read);
    ArrayHandle<unsigned int> h_cell_adj(m_cl->getCellAdjArray(), access_location::host, access_mode::read);

    // access the neighbor list data
    ArrayHandle<unsigned int> h_head_list(m_head_list, access_location::host, access_mode::read);
    ArrayHandle<unsigned int> h_Nmax(m_Nmax, access_location::host, access_mode::read);
    ArrayHandle<unsigned int> h_conditions(m_conditions, access_location::host, access_mode::readwrite);
    ArrayHandle<unsigned int> h_nlist(m_nlist, access_location::host, access_mode::overwrite);
    ArrayHandle<unsigned int> h_n_neigh(m_n_neigh, access_location::host, access_mode::overwrite);

    // access indexers
    Index3D ci = m_cl->getCellIndexer();
    Index2D cli = m_cl->getCellListIndexer();
    Index2D cadji = m_cl->getCellAdjIndexer();

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

                // get the current neighbor type from the position data (will use tdb on the GPU)
                unsigned int cur_neigh_type = __scalar_as_int(h_pos.data[cur_neigh].w);
                Scalar r_cut = h_r_cut.data[m_typpair_idx(type_i,cur_neigh_type)];

                // automatically exclude particles without a distance check when:
                // (1) they are the same particle, or
                // (2) the r_cut(i,j) indicates to skip, or
                // (3) they are in the same body
                bool excluded = ((i == (int)cur_neigh) || (r_cut <= Scalar(0.0)));
                if (m_filter_body && body_i != NO_BODY)
                    excluded = excluded | (body_i == h_body.data[cur_neigh]);
                if (excluded)
                    continue;

                Scalar3 neigh_pos = make_scalar3(cur_xyzf.x, cur_xyzf.y, cur_xyzf.z);
                Scalar3 dx = my_pos - neigh_pos;
                dx = box.minImage(dx);

                Scalar r_list = r_cut + m_r_buff;
                Scalar sqshift = Scalar(0.0);
                if (m_diameter_shift)
                    {
                    const Scalar delta = (diam_i + h_diameter.data[cur_neigh]) * Scalar(0.5) - Scalar(1.0);
                    // r^2 < (r_list + delta)^2
                    // r^2 < r_listsq + delta^2 + 2*r_list*delta
                    sqshift = (delta + Scalar(2.0) * r_list) * delta;
                    }

                Scalar dr_sq = dot(dx,dx);

                // move the squared rlist by the diameter shift if necessary
                Scalar r_listsq = h_r_listsq.data[m_typpair_idx(type_i,cur_neigh_type)];
                if (dr_sq <= (r_listsq + sqshift) && !excluded)
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

void export_NeighborListBinned(py::module& m)
    {
    py::class_<NeighborListBinned, std::shared_ptr<NeighborListBinned> >(m, "NeighborListBinned", py::base<NeighborList>())
    .def(py::init< std::shared_ptr<SystemDefinition>, Scalar, Scalar, std::shared_ptr<CellList> >())
                     ;
    }
