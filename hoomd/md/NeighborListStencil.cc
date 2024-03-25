// Copyright (c) 2009-2024 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

/*! \file NeighborListStencil.cc
    \brief Defines NeighborListStencil
*/

#include "NeighborListStencil.h"

#ifdef ENABLE_MPI
#include "hoomd/Communicator.h"
#endif

using namespace std;

namespace hoomd
    {
namespace md
    {
/*!
 * \param sysdef System definition
 * \param r_cut Default cutoff radius
 * \param r_buff Neighbor list buffer width
 * \param cl Cell list
 * \param cls Cell list stencil
 *
 * A default cell list and stencil will be constructed if \a cl or \a cls are not instantiated.
 */
NeighborListStencil::NeighborListStencil(std::shared_ptr<SystemDefinition> sysdef, Scalar r_buff)
    : NeighborList(sysdef, r_buff), m_cl(std::make_shared<CellList>(sysdef)),
      m_cls(std::make_shared<CellListStencil>(sysdef, m_cl))
    {
    m_exec_conf->msg->notice(5) << "Constructing NeighborListStencil" << endl;

    m_cl->setRadius(1);
    m_cl->setComputeTypeBody(true);
    m_cl->setFlagIndex();
    m_cl->setComputeAdjList(false);
    }

NeighborListStencil::~NeighborListStencil()
    {
    m_exec_conf->msg->notice(5) << "Destroying NeighborListStencil" << endl;
    }

void NeighborListStencil::updateRStencil()
    {
    ArrayHandle<Scalar> h_rcut_max(m_rcut_max, access_location::host, access_mode::read);
    std::vector<Scalar> rstencil(m_pdata->getNTypes(), -1.0);
    for (unsigned int cur_type = 0; cur_type < m_pdata->getNTypes(); ++cur_type)
        {
        Scalar rcut = h_rcut_max.data[cur_type];
        if (rcut > Scalar(0.0))
            {
            Scalar rlist = rcut + m_r_buff;
            rstencil[cur_type] = rlist;
            }
        }
    m_cls->setRStencil(rstencil);
    }

void NeighborListStencil::buildNlist(uint64_t timestep)
    {
    if (m_update_cell_size)
        {
        // update the cell size if the user has not forced a specific size
        if (!m_override_cell_width)
            {
            Scalar rmin = getMinRCut() + m_r_buff;

            m_cl->setNominalWidth(rmin);
            }

        m_update_cell_size = false;
        }

    m_cl->compute(timestep);

    // update the stencil radii if there was a change
    if (m_needs_restencil)
        {
        updateRStencil();
        m_needs_restencil = false;
        }
    m_cls->compute(timestep);

    uint3 dim = m_cl->getDim();
    Scalar3 ghost_width = m_cl->getGhostWidth();

    // acquire the particle data and box dimension
    ArrayHandle<Scalar4> h_pos(m_pdata->getPositions(), access_location::host, access_mode::read);
    ArrayHandle<unsigned int> h_body(m_pdata->getBodies(),
                                     access_location::host,
                                     access_mode::read);

    const BoxDim& box = m_pdata->getBox();
    Scalar3 nearest_plane_distance = box.getNearestPlaneDistance();

    // validate that the cutoff fits inside the box
    Scalar rmax = getMaxRCut() + m_r_buff;

    // get periodic flags
    uchar3 periodic = box.getPeriodic();

    if ((periodic.x && nearest_plane_distance.x <= rmax * 2.0)
        || (periodic.y && nearest_plane_distance.y <= rmax * 2.0)
        || (this->m_sysdef->getNDimensions() == 3 && periodic.z
            && nearest_plane_distance.z <= rmax * 2.0))
        {
        std::ostringstream oss;
        oss << "nlist: Simulation box is too small! Particles would be interacting with themselves."
            << "rmax=" << rmax << std::endl;

        if (box.getPeriodic().x)
            oss << "nearest_plane_distance.x=" << nearest_plane_distance.x << std::endl;
        if (box.getPeriodic().y)
            oss << "nearest_plane_distance.y=" << nearest_plane_distance.y << std::endl;
        if (this->m_sysdef->getNDimensions() == 3 && box.getPeriodic().z)
            oss << "nearest_plane_distance.z=" << nearest_plane_distance.z << std::endl;
        throw std::runtime_error(oss.str());
        }

    // access the rlist data
    ArrayHandle<Scalar> h_r_cut(m_r_cut, access_location::host, access_mode::read);

    // access the cell list data arrays
    ArrayHandle<unsigned int> h_cell_size(m_cl->getCellSizeArray(),
                                          access_location::host,
                                          access_mode::read);
    ArrayHandle<Scalar4> h_cell_xyzf(m_cl->getXYZFArray(),
                                     access_location::host,
                                     access_mode::read);
    ArrayHandle<uint2> h_cell_type_body(m_cl->getTypeBodyArray(),
                                        access_location::host,
                                        access_mode::read);
    ArrayHandle<Scalar4> h_stencil(m_cls->getStencils(), access_location::host, access_mode::read);
    ArrayHandle<unsigned int> h_n_stencil(m_cls->getStencilSizes(),
                                          access_location::host,
                                          access_mode::read);
    const Index2D& stencil_idx = m_cls->getStencilIndexer();

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
                if (sib >= (int)dim.x)
                    sib -= dim.x;
                else if (sib < 0)
                    sib += dim.x;

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
                if (sjb >= (int)dim.y)
                    sjb -= dim.y;
                else if (sjb < 0)
                    sjb += dim.y;

                assert(sjb >= 0 && sjb < (int)dim.y);
                }
            else if (sjb < 0 || sjb >= (int)dim.y)
                {
                continue;
                }

            if (periodic.z)
                {
                if (skb >= (int)dim.z)
                    skb -= dim.z;
                else if (skb < 0)
                    skb += dim.z;

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
                // read in the particle type (diameter and body as well while we've got the Scalar4
                // in)
                const uint2& neigh_type_body = h_cell_type_body.data[cli(cur_offset, neigh_cell)];
                const unsigned int type_j = neigh_type_body.x;
                const unsigned int body_j = neigh_type_body.y;

                // skip any particles belonging to the same body if requested
                if (m_filter_body && body_i != NO_BODY && body_i == body_j)
                    continue;

                // read cutoff and skip if pair is inactive
                Scalar r_cut = h_r_cut.data[m_typpair_idx(type_i, type_j)];
                if (r_cut <= Scalar(0.0))
                    continue;

                // compute the rlist based on the particle type we're interacting with
                Scalar r_list = r_cut + m_r_buff;
                Scalar r_listsq = r_list * r_list;

                // compare the check distance to the minimum cell distance, and pass without
                // distance check if unnecessary
                if (cell_dist2 > r_listsq)
                    continue;

                // only load in the particle position and id if distance check is satisfied
                const Scalar4& neigh_xyzf = h_cell_xyzf.data[cli(cur_offset, neigh_cell)];
                unsigned int cur_neigh = __scalar_as_int(neigh_xyzf.w);

                // a particle cannot neighbor itself
                if (i == (int)cur_neigh)
                    continue;

                Scalar3 neigh_pos = make_scalar3(neigh_xyzf.x, neigh_xyzf.y, neigh_xyzf.z);
                Scalar3 dx = my_pos - neigh_pos;
                dx = box.minImage(dx);

                Scalar dr_sq = dot(dx, dx);

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
                            h_conditions.data[type_i]
                                = max(h_conditions.data[type_i], cur_n_neigh + 1);

                        ++cur_n_neigh;
                        }
                    }
                }
            }

        h_n_neigh.data[i] = cur_n_neigh;
        }
    }

namespace detail
    {
void export_NeighborListStencil(pybind11::module& m)
    {
    pybind11::class_<NeighborListStencil, NeighborList, std::shared_ptr<NeighborListStencil>>(
        m,
        "NeighborListStencil")
        .def(pybind11::init<std::shared_ptr<SystemDefinition>, Scalar>())
        .def_property("cell_width",
                      &NeighborListStencil::getCellWidth,
                      &NeighborListStencil::setCellWidth)
        .def_property("deterministic",
                      &NeighborListStencil::getDeterministic,
                      &NeighborListStencil::setDeterministic);
    }

    } // end namespace detail
    } // end namespace md
    } // end namespace hoomd
