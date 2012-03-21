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

// Maintainer: jglaser

/*! \file MPIInitializer.cc
    \brief Implements the MPInitializer class
*/

#ifdef ENABLE_MPI
#include "MPIInitializer.h"

#include "SystemDefinition.h"
#include "ParticleData.h"

#include <boost/mpi.hpp>
#include <boost/python.hpp>
#include <boost/serialization/map.hpp>

using namespace boost::python;

// Define some of our types as fixed-size MPI datatypes for performance optimization
BOOST_IS_MPI_DATATYPE(Scalar4)
BOOST_IS_MPI_DATATYPE(Scalar3)
BOOST_IS_MPI_DATATYPE(Scalar)
BOOST_IS_MPI_DATATYPE(uint3)
BOOST_IS_MPI_DATATYPE(int3)

//! Constructor
/*! The constructor performs a spatial domain decomposition of the simulation box of processor with rank \b root.
 * The domain dimensions are distributed on the other processors.
 */
MPIInitializer::MPIInitializer(boost::shared_ptr<SystemDefinition> sysdef,
                               boost::shared_ptr<boost::mpi::communicator> comm,
                               unsigned int root,
                               unsigned int nx,
                               unsigned int ny,
                               unsigned int nz
                               )
      : m_sysdef(sysdef),
        m_pdata(sysdef->getParticleData()),
        m_mpi_comm(comm),
        m_global_box(1.0,1.0,1.0)
    {
    unsigned int rank = m_mpi_comm->rank();

    if (rank == root)
        {
        // get global box dimensions
        m_global_box = m_pdata->getBox();

        bool found_decomposition = findDecomposition(nx, ny, nz);
        if (! found_decomposition)
            {
            cerr << endl << "***Warning! Unable to find a decomposition of total number of domains == "
                 << m_mpi_comm->size()
                 << endl << "            with requested dimensions. Choosing default decomposition."
                 << endl << endl;

            nx = ny = nz = 0;
            findDecomposition(nx,ny,nz);
            }
        
        m_decomposition.nx = nx;
        m_decomposition.ny = ny;
        m_decomposition.nz = nz;

        // Print out information about the domain decomposition
        std::cout << "Domain decomposition: n_x = " << nx << " n_y = " << ny << " n_z = " << nz << std::endl;
        }

    // calculate physical box dimensions of every processor

    std::vector<BoxDim> box_proc(m_mpi_comm->size(),BoxDim(1.0));
    std::vector<uint3> grid_pos_proc(m_mpi_comm->size());
    if (rank == root)
        {
        for (unsigned int rank = 0; rank < (unsigned int) m_mpi_comm->size(); rank++)
            {
            BoxDim box(1.0,1.0,1.0);
            double Lx = (m_global_box.xhi-m_global_box.xlo)/(double)m_decomposition.nx;
            double Ly = (m_global_box.yhi-m_global_box.ylo)/(double)m_decomposition.ny;
            double Lz = (m_global_box.zhi-m_global_box.zlo)/(double)m_decomposition.nz;

            // position of this domain in the grid
            unsigned int k = rank/(m_decomposition.nx*m_decomposition.ny);
            unsigned int j = (rank % (m_decomposition.nx*m_decomposition.ny)) / m_decomposition.nx;
            unsigned int i = (rank % (m_decomposition.nx*m_decomposition.ny)) % m_decomposition.nx;

            box.xlo = m_global_box.xlo + (double)i * Lx;
            box.xhi = box.xlo + Lx;

            box.ylo = m_global_box.ylo + (double)j * Ly;
            box.yhi = box.ylo + Ly;

            box.zlo = m_global_box.zlo + (double)k * Lz;
            box.zhi = box.zlo + Lz;

            grid_pos_proc[rank] = make_uint3(i,j,k);
            box_proc[rank] = box;
            }
        }

    // broadcast global box dimensions
    boost::mpi::broadcast(*m_mpi_comm, m_global_box, root);

    // distribute local box dimensions
    BoxDim box(1.0);
    boost::mpi::scatter(*m_mpi_comm, box_proc, box, root);

    // broadcast grid dimensions
    boost::mpi::broadcast(*m_mpi_comm, m_decomposition.nx, root);
    boost::mpi::broadcast(*m_mpi_comm, m_decomposition.ny, root);
    boost::mpi::broadcast(*m_mpi_comm, m_decomposition.nz, root);

    // distribute grid positions
    boost::mpi::scatter(*m_mpi_comm, grid_pos_proc, m_decomposition.grid_pos, root);

    // Set up neighbor ranks & boundary information
    for (unsigned int dir=0; dir < 6; dir++)
        {
        m_decomposition.neighbors[dir] = getNeighborRank(dir);
        m_decomposition.is_at_boundary[dir] = isAtBoundary(dir);
        }

    m_decomposition.root = root;

    // set simulation box
    m_pdata->setBox(box);

    // set global simulation box
    m_pdata->setGlobalBox(m_global_box);
    }

//! Find a domain decomposition with given parameters
bool MPIInitializer::findDecomposition(unsigned int& nx, unsigned int& ny, unsigned int& nz)
    {
    Scalar Lx_g = m_global_box.xhi - m_global_box.xlo;
    Scalar Ly_g = m_global_box.yhi - m_global_box.ylo;
    Scalar Lz_g = m_global_box.zhi - m_global_box.zlo;
    assert(Lx_g > 0);
    assert(Ly_g > 0);
    assert(Lz_g > 0);

    // Calulate the number of sub-domains in every direction
    // by minimizing the surface area between domains at constant number of domains
    double min_surface_area = Lx_g*Ly_g*m_mpi_comm->size()+Lx_g*Lz_g+Ly_g*Lz_g;

    unsigned int nx_in = nx;
    unsigned int ny_in = ny;
    unsigned int nz_in = nz;

    bool found_decomposition = (nx_in == 0 && ny_in == 0 && nz_in == 0);

    // initial guess
    nx = 1;
    ny = 1;
    nz = m_mpi_comm->size();


    for (unsigned int nx_try = 1; nx_try <= (unsigned int) m_mpi_comm->size(); nx_try++)
        {
        if (nx_in != 0 && nx_try != nx_in)
            continue;
        for (unsigned int ny_try = 1; nx_try*ny_try <= (unsigned int) m_mpi_comm->size(); ny_try++)
            {
            if (ny_in != 0 && ny_try != ny_in)
                continue;
            for (unsigned int nz_try = 1; nx_try*ny_try*nz_try <= (unsigned int) m_mpi_comm->size(); nz_try++)
                {
                if (nz_in != 0 && nz_try != nz_in)
                    continue;
                if (nx_try*ny_try*nz_try != (unsigned int) m_mpi_comm->size()) continue;
                double surface_area = Lx_g*Ly_g*nz_try + Lx_g*Lz_g*ny_try + Ly_g*Lz_g*nx_try;
                if (surface_area < min_surface_area || !found_decomposition)
                    {
                    nx = nx_try;
                    ny = ny_try;
                    nz = nz_try;
                    min_surface_area = surface_area;
                    found_decomposition = true;
                    }
                }
            }
        }

    return found_decomposition;
    }

//! Calculate MPI ranks of neighboring domain.
unsigned int MPIInitializer::getNeighborRank(unsigned int dir)
    {
    assert(0<= dir && dir < 6);

    int adj[6][3] = {{1,0,0},{-1,0,0},{0,1,0},{0,-1,0},{0,0,1},{0,0,-1}};

    // determine neighbor position
    int ineigh = (int)m_decomposition.grid_pos.x + adj[dir][0];
    int jneigh = (int)m_decomposition.grid_pos.y + adj[dir][1];
    int kneigh = (int)m_decomposition.grid_pos.z + adj[dir][2];

    // wrap across boundaries
    if (ineigh < 0)
        ineigh += m_decomposition.nx;
    else if (ineigh == (int) m_decomposition.nx)
        ineigh -= m_decomposition.nx;

    if (jneigh < 0)
        jneigh += m_decomposition.ny;
    else if (jneigh == (int) m_decomposition.ny)
        jneigh -= m_decomposition.ny;

    if (kneigh < 0)
        kneigh += m_decomposition.nz;
    else if (kneigh == (int) m_decomposition.nz)
        kneigh -= m_decomposition.nz;

    return kneigh*m_decomposition.nx*m_decomposition.ny + jneigh * m_decomposition.nx + ineigh;
    }

//! Determines whether the local box shares a boundary with the global box
bool MPIInitializer::isAtBoundary(unsigned int dir) 
    {
        return ( (dir == 0 && m_decomposition.grid_pos.x == m_decomposition.nx - 1) ||
                 (dir == 1 && m_decomposition.grid_pos.x == 0)        ||
                 (dir == 2 && m_decomposition.grid_pos.y == m_decomposition.ny - 1) ||
                 (dir == 3 && m_decomposition.grid_pos.y == 0)        ||
                 (dir == 4 && m_decomposition.grid_pos.z == m_decomposition.nz - 1) ||
                 (dir == 5 && m_decomposition.grid_pos.z == 0));
    }

//! Export the domain decomposition information
void export_DomainDecomposition()
    {
    class_<DomainDecomposition>("DomainDecomposition")
        .def_readonly("grid_pos", &DomainDecomposition::grid_pos)
        .def_readonly("nx", &DomainDecomposition::nx)
        .def_readonly("ny", &DomainDecomposition::ny)
        .def_readonly("nz", &DomainDecomposition::nz)
        .def_readonly("root", &DomainDecomposition::root)
        ;
    }

//! Export MPIInitializer class to python
void export_MPIInitializer()
    {
    class_<MPIInitializer, boost::noncopyable >("MPIInitializer",
           init< boost::shared_ptr<SystemDefinition>, boost::shared_ptr<boost::mpi::communicator>,
           unsigned int, unsigned int, unsigned int, unsigned int>())
    .def("getDomainDecomposition", &MPIInitializer::getDomainDecomposition)
    ;
    }
#endif // ENABLE_MPI
