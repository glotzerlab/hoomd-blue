// Copyright (c) 2009-2024 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

/*! \file NeighborListBinned.cc
    \brief Defines NeighborListBinned
*/

#include "NeighborListBinned.h"

#ifdef ENABLE_MPI
#include "hoomd/Communicator.h"
#endif

using namespace std;

namespace hoomd
    {
namespace md
    {
NeighborListBinned::NeighborListBinned(std::shared_ptr<SystemDefinition> sysdef, Scalar r_buff)
    : NeighborList(sysdef, r_buff), m_cl(std::make_shared<CellList>(sysdef))
    {
    m_exec_conf->msg->notice(5) << "Constructing NeighborListBinned" << endl;

    m_cl->setRadius(1);
    m_cl->setComputeXYZF(true);
    m_cl->setComputeTypeBody(false);
    m_cl->setFlagIndex();
    }

NeighborListBinned::~NeighborListBinned()
    {
    m_exec_conf->msg->notice(5) << "Destroying NeighborListBinned" << endl;
    }

void NeighborListBinned::buildNlist(uint64_t timestep)
    {
    // update the cell list size if needed
    if (m_update_cell_size)
        {
        Scalar rmax = getMaxRCut() + m_r_buff;

        m_cl->setNominalWidth(rmax);
        m_update_cell_size = false;
        }

    m_cl->compute(timestep);

    uint3 dim = m_cl->getDim();
    Scalar3 ghost_width = m_cl->getGhostWidth();

    // acquire the particle data and box dimension
    ArrayHandle<Scalar4> h_pos(m_pdata->getPositions(), access_location::host, access_mode::read);
    ArrayHandle<unsigned int> h_body(m_pdata->getBodies(),
                                     access_location::host,
                                     access_mode::read);

    const BoxDim& box = m_pdata->getBox();

    // access the rlist data
    ArrayHandle<Scalar> h_r_cut(m_r_cut, access_location::host, access_mode::read);
    ArrayHandle<Scalar> h_r_listsq(m_r_listsq, access_location::host, access_mode::read);

    // access the cell list data arrays
    ArrayHandle<unsigned int> h_cell_size(m_cl->getCellSizeArray(),
                                          access_location::host,
                                          access_mode::read);
    ArrayHandle<Scalar4> h_cell_xyzf(m_cl->getXYZFArray(),
                                     access_location::host,
                                     access_mode::read);
    ArrayHandle<unsigned int> h_cell_adj(m_cl->getCellAdjArray(),
                                         access_location::host,
                                         access_mode::read);

    // access the neighbor list data
    ArrayHandle<size_t> h_head_list(m_head_list, access_location::host, access_mode::read);
    ArrayHandle<unsigned int> h_Nmax(m_Nmax, access_location::host, access_mode::read);
    ArrayHandle<unsigned int> h_conditions(m_conditions,
                                           access_location::host,
                                           access_mode::readwrite);
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

        const unsigned int Nmax_i = h_Nmax.data[type_i];
        const size_t head_idx_i = h_head_list.data[i];

        // find the bin each particle belongs in
        Scalar3 f = box.makeFraction(my_pos, ghost_width);
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
        unsigned int my_cell = ci(ib, jb, kb);

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

                // get the current neighbor type from the position data (will use TypeBody on the
                // GPU)
                unsigned int cur_neigh_type = __scalar_as_int(h_pos.data[cur_neigh].w);
                Scalar r_cut = h_r_cut.data[m_typpair_idx(type_i, cur_neigh_type)];

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

                Scalar dr_sq = dot(dx, dx);

                Scalar r_listsq = h_r_listsq.data[m_typpair_idx(type_i, cur_neigh_type)];
                if (dr_sq <= r_listsq && !excluded)
                    {
                    // Add the neighbor index to the list.
                    if (m_storage_mode == full || i < (int)cur_neigh)
                        {
                        // local neighbor
                        if (cur_n_neigh < Nmax_i)
                            {
                            h_nlist.data[head_idx_i + cur_n_neigh] = cur_neigh;
                            }
                        else
                            h_conditions.data[type_i]
                                = max(h_conditions.data[type_i], cur_n_neigh + 1);

                        cur_n_neigh++;
                        }
                    }
                }
            }

        h_n_neigh.data[i] = cur_n_neigh;
        }
    }

namespace detail
    {
void export_NeighborListBinned(pybind11::module& m)
    {
    pybind11::class_<NeighborListBinned, NeighborList, std::shared_ptr<NeighborListBinned>>(
        m,
        "NeighborListBinned")
        .def(pybind11::init<std::shared_ptr<SystemDefinition>, Scalar>())
        .def_property("deterministic",
                      &NeighborListBinned::getDeterministic,
                      &NeighborListBinned::setDeterministic)
        .def("getDim",
             &NeighborListBinned::getDim,
             pybind11::return_value_policy::reference_internal)
        .def("getNmax", &NeighborListBinned::getNmax);
    }

    } // end namespace detail
    } // end namespace md
    } // end namespace hoomd
