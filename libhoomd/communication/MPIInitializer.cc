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
                               unsigned int root)
      : m_sysdef(sysdef),
        m_pdata(sysdef->getParticleData()),
        m_mpi_comm(comm),
        m_global_box(1.0,1.0,1.0),
        m_box(1.0,1.0,1.0)
    {
    m_rank = m_mpi_comm->rank();

    if (m_rank == root)
        {
        // get global box dimensions
        m_global_box = m_pdata->getBox();

        double Lx_g = m_global_box.xhi - m_global_box.xlo;
        double Ly_g = m_global_box.yhi - m_global_box.ylo;
        double Lz_g = m_global_box.zhi - m_global_box.zlo;

        // Calulate the number of sub-domains in every direction
        // by minimizing the surface area between domains at constant number of domains
        double min_surface_area = Lx_g*Ly_g*m_mpi_comm->size()+Lx_g*Lz_g+Ly_g*Lz_g;

        // initial guess
        m_nx = 1; m_ny = 1; m_nz = m_mpi_comm->size();

        for (unsigned int nx = 1; nx <= (unsigned int) m_mpi_comm->size(); nx++)
            {
            for (unsigned int ny = 1; nx*ny <= (unsigned int) m_mpi_comm->size(); ny++)
                {
                for (unsigned int nz = 1; nx*ny*nz <= (unsigned int) m_mpi_comm->size(); nz++)
                    {
                    if (nx*ny*nz != (unsigned int) m_mpi_comm->size()) continue;
                    double surface_area = Lx_g*Ly_g*nz + Lx_g*Lz_g*ny + Ly_g*Lz_g*nx;
                    if (surface_area < min_surface_area)
                        {
                        m_nx = nx;
                        m_ny = ny;
                        m_nz = nz;
                        min_surface_area = surface_area;
                        }
                    }
                }
            }

        // Print out information about the domain decomposition
        std::cout << "Domain decomposition: n_x = " << m_nx << " n_y = " << m_ny << " n_z = " << m_nz << std::endl;
        }

    // calculate physical box dimensions of every processor

    m_box_proc.resize(m_mpi_comm->size(),BoxDim(1.0,1.0,1.0));
    m_grid_pos_proc.resize(m_mpi_comm->size());
    if (m_rank == root)
        {
        for (unsigned int rank = 0; rank < (unsigned int) m_mpi_comm->size(); rank++)
            {
            BoxDim box(1.0,1.0,1.0);
            m_Lx = (m_global_box.xhi-m_global_box.xlo)/(double)m_nx;
            m_Ly = (m_global_box.yhi-m_global_box.ylo)/(double)m_ny;
            m_Lz = (m_global_box.zhi-m_global_box.zlo)/(double)m_nz;

            // position of this domain in the grid
            unsigned int k = rank/(m_nx*m_ny);
            unsigned int j = (rank % (m_nx*m_ny)) / m_nx;
            unsigned int i = (rank % (m_nx*m_ny)) % m_nx;

            box.xlo = m_global_box.xlo + (double)i * m_Lx;
            box.xhi = box.xlo + m_Lx;

            box.ylo = m_global_box.ylo + (double)j * m_Ly;
            box.yhi = box.ylo + m_Ly;

            box.zlo = m_global_box.zlo + (double)k * m_Lz;
            box.zhi = box.zlo + m_Lz;

            m_grid_pos_proc[rank] = make_uint3(i,j,k);
            m_box_proc[rank] = box;
            }
        }

    // broadcast global box dimensions
    boost::mpi::broadcast(*m_mpi_comm, m_global_box, root);

    // distribute local box dimensions
    boost::mpi::scatter(*m_mpi_comm, m_box_proc, m_box, root);

    // broadcast grid dimensions
    boost::mpi::broadcast(*m_mpi_comm, m_nx, root);
    boost::mpi::broadcast(*m_mpi_comm, m_ny, root);
    boost::mpi::broadcast(*m_mpi_comm, m_nz, root);

    // distribute grid positions
    boost::mpi::scatter(*m_mpi_comm, m_grid_pos_proc, m_grid_pos, root);
    }

//! Distribute particle data onto processors
void MPIInitializer::scatter(unsigned int root)
    {
    if (m_rank == root)
        {
        // get number of particle types
        m_num_particle_types = m_pdata->getNTypes();

        // get particle type mapping
        m_type_mapping.clear();
        for (unsigned int i = 0; i < m_num_particle_types; i++)
            m_type_mapping.push_back(m_pdata->getNameByType(i));

        // get global simulation box dimensions
        m_global_box = m_pdata->getBox();

        // get global number of particles
        m_nglobal = m_pdata->getN();

        // first clear per-processor particle data
        m_pos_proc.clear();
        m_vel_proc.clear();
        m_accel_proc.clear();
        m_type_proc.clear();
        m_mass_proc.clear();
        m_charge_proc.clear();
        m_diameter_proc.clear();
        m_image_proc.clear();
        m_body_proc.clear();
        m_global_tag_proc.clear();
        m_N_proc.clear();

        // resize to number of ranks in communicator
        m_pos_proc.resize(m_mpi_comm->size());
        m_vel_proc.resize(m_mpi_comm->size());
        m_accel_proc.resize(m_mpi_comm->size());
        m_type_proc.resize(m_mpi_comm->size());
        m_mass_proc.resize(m_mpi_comm->size());
        m_charge_proc.resize(m_mpi_comm->size());
        m_diameter_proc.resize(m_mpi_comm->size());
        m_image_proc.resize(m_mpi_comm->size());
        m_body_proc.resize(m_mpi_comm->size());
        m_global_tag_proc.resize(m_mpi_comm->size());

        m_N_proc.resize(m_mpi_comm->size(),0);

        SnapshotParticleData global_snapshot(m_pdata->getN());
        m_pdata->takeSnapshot(global_snapshot);

        for (std::vector<Scalar3>::iterator it=global_snapshot.pos.begin(); it != global_snapshot.pos.end(); it++)
            {
            // determine domain the particle is placed into
            int i= (it->x - m_global_box.zlo)/m_Lx;
            int j= (it->y - m_global_box.ylo)/m_Ly;
            int k= (it->z - m_global_box.zlo)/m_Lz;

            // treat particles lying exactly on the boundary
            if (i == (int) m_nx)
                {
                i = 0;
                it->x = m_global_box.xlo;
                }
            if (j == (int) m_ny)
                {
                j = 0;
                it->y = m_global_box.ylo;
                }
            if (k == (int) m_nz)
                {
                k = 0;
                it->z = m_global_box.zlo;
                }


            unsigned int idx = it - global_snapshot.pos.begin();

            unsigned int rank = k*m_nx*m_ny + j * m_nx + i;
            assert(rank <= (unsigned int) m_mpi_comm->size());

            // fill up per-processor data structures
            m_pos_proc[rank].push_back(global_snapshot.pos[idx]);
            m_vel_proc[rank].push_back(global_snapshot.vel[idx]);
            m_accel_proc[rank].push_back(global_snapshot.accel[idx]);
            m_type_proc[rank].push_back(global_snapshot.type[idx]);
            m_mass_proc[rank].push_back(global_snapshot.mass[idx]);
            m_charge_proc[rank].push_back(global_snapshot.charge[idx]);
            m_diameter_proc[rank].push_back(global_snapshot.diameter[idx]);
            m_image_proc[rank].push_back(global_snapshot.image[idx]);
            m_body_proc[rank].push_back(global_snapshot.body[idx]);
            m_global_tag_proc[rank].push_back(global_snapshot.global_tag[idx]);
            m_N_proc[rank]++;

            }
        }


    // broadcast number of particle types
    boost::mpi::broadcast(*m_mpi_comm, m_num_particle_types, root);

    // broadcast type mapping
    boost::mpi::broadcast(*m_mpi_comm, m_type_mapping, root);

    // distribute number of particles
    unsigned int N = 0;
    boost::mpi::scatter(*m_mpi_comm, m_N_proc, N,root);

    // initialize snapshot
    SnapshotParticleData snap(N);

    // distribute positions
    boost::mpi::scatter(*m_mpi_comm, m_pos_proc,snap.pos,root);

    // distribute velocities
    boost::mpi::scatter(*m_mpi_comm, m_vel_proc,snap.vel,root);

    // distribute accelerations
    boost::mpi::scatter(*m_mpi_comm, m_accel_proc, snap.accel, root);

    // distribute particle types
    boost::mpi::scatter(*m_mpi_comm, m_type_proc, snap.type, root);

    // distribute particle masses
    boost::mpi::scatter(*m_mpi_comm, m_mass_proc, snap.mass, root);

    // distribute particle charges
    boost::mpi::scatter(*m_mpi_comm, m_charge_proc, snap.charge, root);

    // distribute particle diameters`
    boost::mpi::scatter(*m_mpi_comm, m_diameter_proc, snap.diameter, root);

    // distribute particle images
    boost::mpi::scatter(*m_mpi_comm, m_image_proc, snap.image, root);

    // distribute body ids
    boost::mpi::scatter(*m_mpi_comm, m_body_proc, snap.body, root);

    // distribute global tags
    boost::mpi::scatter(*m_mpi_comm, m_global_tag_proc, snap.global_tag, root);

    // broadcast global number of particles
    boost::mpi::broadcast(*m_mpi_comm, m_nglobal, root);

    snap.type_mapping = m_type_mapping;
    snap.num_particle_types = m_num_particle_types;

    // set global number of particles
    m_pdata->setNGlobal(m_nglobal);

    // reset all reverse lookup tags to NOT_LOCAL flag
        {
        ArrayHandle<unsigned int> h_global_rtag(m_pdata->getGlobalRTags(), access_location::host, access_mode::overwrite);
        for (unsigned int tag = 0; tag < m_nglobal; tag++)
            h_global_rtag.data[tag] = NOT_LOCAL;
        }

    // initialize local simulation box with snapshot
    m_pdata->initializeFromSnapshot(snap);

    // Notify about addition of particles
    m_pdata->notifyLocalParticleNumChange();

    // set simulation box
    m_pdata->setBox(m_box);

    // set global simulation box
    m_pdata->setGlobalBox(m_global_box);
    }

//! Gather particle data from all processors into a snapshot on a single processor
void MPIInitializer::gatherSnapshot(SnapshotParticleData& global_snapshot, unsigned int root)
    {
    // take a snapshot of the current configuration
    SnapshotParticleData snap(m_pdata->getN());
    global_snapshot.type_mapping.clear();
    for (unsigned int i = 0; i < m_num_particle_types; i++)
        global_snapshot.type_mapping.push_back(m_pdata->getNameByType(i));

    m_pdata->takeSnapshot(snap);

    // collect all particle data on the root processor
    boost::mpi::gather(*m_mpi_comm, snap.pos, m_pos_proc, root);
    boost::mpi::gather(*m_mpi_comm, snap.vel, m_vel_proc, root);
    boost::mpi::gather(*m_mpi_comm, snap.accel, m_accel_proc, root);
    boost::mpi::gather(*m_mpi_comm, snap.type, m_type_proc, root);
    boost::mpi::gather(*m_mpi_comm, snap.mass, m_mass_proc, root);
    boost::mpi::gather(*m_mpi_comm, snap.charge, m_charge_proc, root);
    boost::mpi::gather(*m_mpi_comm, snap.diameter, m_diameter_proc, root);
    boost::mpi::gather(*m_mpi_comm, snap.image, m_image_proc, root);
    boost::mpi::gather(*m_mpi_comm, snap.body, m_body_proc, root);
    boost::mpi::gather(*m_mpi_comm, snap.global_tag, m_global_tag_proc, root);
    boost::mpi::gather(*m_mpi_comm, m_pdata->getN(), m_N_proc, root);

    std::vector< std::map<unsigned int, unsigned int> > global_rtag_proc;
    global_rtag_proc.resize(m_mpi_comm->size());

    // gather the reverse-lookup maps
    boost::mpi::gather(*m_mpi_comm, snap.global_rtag, global_rtag_proc, root);

    // construct global snapshot
    if (global_snapshot.size != m_pdata->getNGlobal())
        {
        cerr << endl << "***Error! Number of particles in snapshot must be equal to global number of particles." << endl << endl;
        throw std::runtime_error("Error gathering ParticleData");
        }

    if (m_rank == root)
        {
        std::map<unsigned int, unsigned int>::iterator it;
        for (unsigned int tag = 0; tag < m_pdata->getNGlobal(); tag++)
            {
            bool found = false;
            unsigned int rank;
            for (rank = 0; rank < (unsigned int) m_mpi_comm->size(); rank ++)
                {
                it = global_rtag_proc[rank].find(tag);
                if (it != global_rtag_proc[rank].end())
                    {
                    found = true;
                    break;
                    }
                }
            if (! found)
                {
                cerr << endl << "***Error! Could not find particle " << tag << " on any processor. " << endl << endl;
                throw std::runtime_error("Error gathering ParticleData");
                }

            // rank contains the processor rank on which the particle was found
            unsigned int idx = it->second;
            global_snapshot.pos[tag] = m_pos_proc[rank][idx];
            global_snapshot.vel[tag] = m_vel_proc[rank][idx];
            global_snapshot.accel[tag] = m_accel_proc[rank][idx];
            global_snapshot.type[tag] = m_type_proc[rank][idx];
            global_snapshot.mass[tag] = m_mass_proc[rank][idx];
            global_snapshot.charge[tag] = m_charge_proc[rank][idx];
            global_snapshot.diameter[tag] = m_diameter_proc[rank][idx];
            global_snapshot.image[tag] = m_image_proc[rank][idx];
            global_snapshot.body[tag] = m_body_proc[rank][idx];
            global_snapshot.global_tag[tag] = tag;
            global_snapshot.global_rtag.insert(std::pair<unsigned int, unsigned int>(tag, tag));
            }
        }
    }

//! Calculate MPI ranks of neighboring domain.
unsigned int MPIInitializer::getNeighborRank(unsigned int dir)
    {
    assert(0<= dir && dir < 6);

    int adj[6][3] = {{1,0,0},{-1,0,0},{0,1,0},{0,-1,0},{0,0,1},{0,0,-1}};

    // determine neighbor position
    int ineigh = (int)m_grid_pos.x + adj[dir][0];
    int jneigh = (int)m_grid_pos.y + adj[dir][1];
    int kneigh = (int)m_grid_pos.z + adj[dir][2];

    // wrap across boundaries
    if (ineigh < 0)
        ineigh += m_nx;
    else if (ineigh == (int) m_nx)
        ineigh -= m_nx;

    if (jneigh < 0)
        jneigh += m_ny;
    else if (jneigh == (int) m_ny)
        jneigh -= m_ny;

    if (kneigh < 0)
        kneigh += m_nz;
    else if (kneigh == (int) m_nz)
        kneigh -= m_nz;

    return kneigh*m_nx*m_ny + jneigh * m_nx + ineigh;
    }

//! Get global box dimensions along a specified direction
unsigned int MPIInitializer::getDimension(unsigned int dir) const
    {
    assert(dir < 3);
    unsigned int dim = 0;

    if (dir ==0)
        {
        dim = m_nx;
        }
    else if (dir == 1)
        {
        dim = m_ny;
        }
    else if (dir == 2)
        {
        dim = m_nz;
        }

    return dim;
    }

//! Determine whether this box shares a boundary with the global simulation box
bool MPIInitializer::isAtBoundary(unsigned int dir) const
    {
        return ( (dir == 0 && m_grid_pos.x == m_nx - 1) ||
                 (dir == 1 && m_grid_pos.x == 0)        ||
                 (dir == 2 && m_grid_pos.y == m_ny - 1) ||
                 (dir == 3 && m_grid_pos.y == 0)        ||
                 (dir == 4 && m_grid_pos.z == m_nz - 1) ||
                 (dir == 5 && m_grid_pos.z == 0));
    }

//! Export MPIInitializer class to python
void export_MPIInitializer()
    {
    class_<MPIInitializer, bases<ParticleDataInitializer>, boost::noncopyable >("MPIInitializer",
           init< boost::shared_ptr<SystemDefinition>, boost::shared_ptr<boost::mpi::communicator>, unsigned int>())
    .def("getNeighborRank", &MPIInitializer::getNeighborRank)
    .def("getGlobalBox", &MPIInitializer::getGlobalBox)
    .def("getDimension", &MPIInitializer::getDimension)
    .def("scatter", &MPIInitializer::scatter)
    .def("gatherSnapshot", &MPIInitializer::gatherSnapshot)
    .def("isAtBoundary", &MPIInitializer::isAtBoundary)
    ;
    }
#endif // ENABLE_MPI
