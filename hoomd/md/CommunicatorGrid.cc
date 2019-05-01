// Copyright (c) 2009-2019 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.


// Maintainer: jglaser

#ifdef ENABLE_MPI

#include "CommunicatorGrid.h"

#include "hoomd/extern/kiss_fft.h"

#include <map>

#ifdef ENABLE_CUDA
#include <cufft.h>
#endif

/*! \param sysdef The system definition
 *  \param dim Dimensions of 3dim grid
 *  \param embed Embedding dimensions
 *  \param offset Start offset of inner grid in array
 *  \param add_outer_layer_to_inner True if outer ghost layer should be added to inner cells
 */
template<typename T>
CommunicatorGrid<T>::CommunicatorGrid(std::shared_ptr<SystemDefinition> sysdef, uint3 dim,
            uint3 embed, uint3 offset, bool add_outer_layer_to_inner)
    : m_pdata(sysdef->getParticleData()),
      m_exec_conf(m_pdata->getExecConf()),
      m_dim(dim),
      m_embed(embed),
      m_offset(offset),
      m_add_outer(add_outer_layer_to_inner)
    {
    m_exec_conf->msg->notice(5) << "Constructing CommunicatorGrid" << std::endl;

    initGridComm();
    }

template<typename T>
void CommunicatorGrid<T>::initGridComm()
    {
    typedef std::multimap<unsigned int, unsigned int> map_t;
    map_t idx_map;

    m_neighbors.clear();

    unsigned int n = 0;
    Index3D di = m_pdata->getDomainDecomposition()->getDomainIndexer();
    ArrayHandle<unsigned int> h_cart_ranks(m_pdata->getDomainDecomposition()->getCartRanks(),
        access_location::host, access_mode::read);

    uint3 my_pos = m_pdata->getDomainDecomposition()->getGridPos();

    std::vector<unsigned int> send_idx;
    std::vector<unsigned int> recv_idx;

    for (int nx = 0; nx < (int)m_embed.x; nx++)
        for (int ny = 0; ny < (int)m_embed.y; ny++)
            for (int nz = 0; nz < (int)m_embed.z; nz++)
                {
                if (nx >= (int)m_offset.x && nx < (int)(m_dim.x+m_offset.x) &&
                    ny >= (int)m_offset.y && ny < (int)(m_dim.y+m_offset.y) &&
                    nz >= (int)m_offset.z && nz < (int)(m_dim.z+m_offset.z))
                    continue; // inner cell;

                int ix = 0;
                int iy = 0;
                int iz = 0;
                if (nx < (int)m_offset.x)
                    ix = -1;
                else if (nx >= (int)(m_dim.x+m_offset.x))
                    ix = 1;

                if (ny < (int)m_offset.y)
                    iy = -1;
                else if (ny >= (int)(m_dim.y+m_offset.y))
                    iy = 1;

                if (nz < (int)m_offset.z)
                    iz = -1;
                else if (nz >= (int)(m_dim.z+m_offset.z))
                    iz = 1;

                assert(ix || iy || iz);

                int i = my_pos.x + ix;
                int j = my_pos.y + iy;
                int k = my_pos.z + iz;

                // wrap across boundaries
                if (i < 0)
                    i += di.getW();
                else if (i >= (int)di.getW())
                    i -= di.getW();

                if (j < 0)
                    j += di.getH();
                else if (j >= (int)di.getH())
                    j -= di.getH();

                if (k < 0)
                    k += di.getD();
                else if (k >= (int)di.getD())
                    k -= di.getD();

                unsigned int neigh_rank = h_cart_ranks.data[di(i,j,k)];

                // add to neighbor set
                m_neighbors.insert(neigh_rank);

                // corresponding inner cell
                unsigned int inner_nx = nx - ix * m_offset.x;
                if (di.getW() <= 2) inner_nx -= ix*(m_dim.x-m_offset.x);
                unsigned int inner_ny = ny - iy * m_offset.y;
                if (di.getH() <= 2) inner_ny -= iy*(m_dim.y-m_offset.y);
                unsigned int inner_nz = nz - iz * m_offset.z;
                if (di.getD() <= 2) inner_nz -= iz*(m_dim.z-m_offset.z);

                // index of receiving cell
                unsigned int ridx,sidx;
                if (m_add_outer)
                    {
                    ridx = inner_nx + m_embed.x * (inner_ny + m_embed.y * inner_nz);
                    sidx = nx + m_embed.x * (ny + m_embed.y* nz);
                    }
                else
                    {
                    ridx = nx + m_embed.x * (ny + m_embed.y* nz);
                    sidx = inner_nx + m_embed.x * (inner_ny + m_embed.y * inner_nz);
                    }

                recv_idx.push_back(ridx);
                send_idx.push_back(sidx);

                idx_map.insert(std::make_pair(neigh_rank, n++));
                }

    // write out send and recv indices

    // allocate memory
    GlobalArray<unsigned int> send_idx_array(idx_map.size(), m_exec_conf);
    m_send_idx.swap(send_idx_array);
    GlobalArray<unsigned int> recv_idx_array(idx_map.size(), m_exec_conf);
    m_recv_idx.swap(recv_idx_array);

    ArrayHandle<unsigned int> h_send_idx(m_send_idx, access_location::host, access_mode::overwrite);
    ArrayHandle<unsigned int> h_recv_idx(m_recv_idx, access_location::host, access_mode::overwrite);

    n = 0;
    for (map_t::iterator it = idx_map.begin(); it != idx_map.end(); it++)
        {
        unsigned int i = it->second;
        h_send_idx.data[n] = send_idx[i];
        h_recv_idx.data[n] = recv_idx[i];
        n++;
        }

    m_begin.clear();
    m_end.clear();
    for (std::set<unsigned int>::iterator it = m_neighbors.begin(); it != m_neighbors.end(); ++it)
        {
        map_t::iterator lower = idx_map.lower_bound(*it);
        map_t::iterator upper = idx_map.upper_bound(*it);
        m_begin.insert(std::make_pair(*it,std::distance(idx_map.begin(),lower)));
        m_end.insert(std::make_pair(*it,std::distance(idx_map.begin(),upper)));
        }

    // resize recv and send buffers
    GlobalArray<T> send_buf(m_send_idx.getNumElements(), m_exec_conf);
    m_send_buf.swap(send_buf);

    GlobalArray<T> recv_buf(m_recv_idx.getNumElements(), m_exec_conf);
    m_recv_buf.swap(recv_buf);
    }

template<typename T>
void CommunicatorGrid<T>::communicate(const GlobalArray<T>& grid)
    {
    assert(grid.getNumElements() >= m_embed.x*m_embed.y*m_embed.z);

        {
        ArrayHandle<T> h_send_buf(m_send_buf, access_location::host, access_mode::overwrite);
        ArrayHandle<unsigned int> h_send_idx(m_send_idx, access_location::host, access_mode::read);
        ArrayHandle<T> h_grid(grid, access_location::host, access_mode::read);

        // gather grid elements into send buf
        unsigned int n = m_send_buf.getNumElements();
        for (unsigned int i = 0; i < n; ++i)
            h_send_buf.data[i] = h_grid.data[h_send_idx.data[i]];
        }

        {
        // access send and recv buffers
        ArrayHandle<T> h_send_buf(m_send_buf, access_location::host, access_mode::read);
        ArrayHandle<T> h_recv_buf(m_recv_buf, access_location::host, access_mode::overwrite);

        typedef std::map<unsigned int, unsigned int>::iterator it_t;
        std::vector<MPI_Request> reqs(2*m_neighbors.size());

        unsigned int n = 0;
        for (std::set<unsigned int>::iterator it = m_neighbors.begin(); it != m_neighbors.end(); it++)
            {
            it_t b = m_begin.find(*it);
            assert(b != m_begin.end());
            it_t e = m_end.find(*it);
            assert(e != m_end.end());

            unsigned int offs = b->second;
            unsigned int n_elem = e->second - b->second;

            MPI_Isend(&h_send_buf.data[offs], n_elem*sizeof(T), MPI_BYTE, *it, 0,
                m_exec_conf->getMPICommunicator(), &reqs[n++]);
            MPI_Irecv(&h_recv_buf.data[offs], n_elem*sizeof(T), MPI_BYTE, *it, 0,
                m_exec_conf->getMPICommunicator(), &reqs[n++]);
            }

        std::vector<MPI_Status> stat(reqs.size());
        MPI_Waitall(reqs.size(), &reqs.front(), &stat.front());
        }

        {
        ArrayHandle<T> h_recv_buf(m_recv_buf, access_location::host, access_mode::read);
        ArrayHandle<unsigned int> h_recv_idx(m_recv_idx, access_location::host, access_mode::overwrite);
        ArrayHandle<T> h_grid(grid, access_location::host, access_mode::readwrite);

        // scatter recv buf into grid
        unsigned int n = m_send_buf.getNumElements();
        if (m_add_outer)
            for (unsigned int i = 0; i < n; ++i)
                h_grid.data[h_recv_idx.data[i]] = h_grid.data[h_recv_idx.data[i]] + h_recv_buf.data[i];
        else
            for (unsigned int i = 0; i < n; ++i)
                h_grid.data[h_recv_idx.data[i]] = h_recv_buf.data[i];
        }
    }

//! Explicit template instantiations
template class PYBIND11_EXPORT CommunicatorGrid<Scalar>;
template class PYBIND11_EXPORT CommunicatorGrid<unsigned int>;

//! Define plus operator for complex data type (needed by CommunicatorMesh)
inline kiss_fft_cpx operator + (kiss_fft_cpx& lhs, kiss_fft_cpx& rhs)
    {
    kiss_fft_cpx res;
    res.r = lhs.r + rhs.r;
    res.i = lhs.i + rhs.i;
    return res;
    }

template class PYBIND11_EXPORT CommunicatorGrid<kiss_fft_cpx>;

#ifdef ENABLE_CUDA
//! Define plus operator for complex data type (needed by CommunicatorMesh)
inline cufftComplex operator + (cufftComplex& lhs, cufftComplex& rhs)
    {
    cufftComplex res;
    res.x = lhs.x + rhs.x;
    res.y = lhs.y + rhs.y;
    return res;
    }

template class PYBIND11_EXPORT CommunicatorGrid<cufftComplex>;
#endif

#endif //ENABLE_MPI
