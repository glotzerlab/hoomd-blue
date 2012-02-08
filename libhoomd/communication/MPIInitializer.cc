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

#include <boost/mpi.hpp>
#include <boost/python.hpp>

using namespace boost::python;

//! Define some of our types as fixed-size MPI datatypes for performance optimization
BOOST_IS_MPI_DATATYPE(Scalar4)
BOOST_IS_MPI_DATATYPE(Scalar3)
BOOST_IS_MPI_DATATYPE(Scalar)
BOOST_IS_MPI_DATATYPE(uint3)
BOOST_IS_MPI_DATATYPE(int3)

//! Constructor
/*! The constructor reads in a ParticleDataInitializer on the processor with rank 0
    and performs the necessary communications to initialize all processors with
    their local particle data.

    \post After construction of an instance of this class, its internal data structures
          have been initialized on all processors and the class accessor methods can be used
          to query information about local particle data.
 */
MPIInitializer::MPIInitializer(ParticleDataInitializer& init, boost::shared_ptr<boost::mpi::communicator> comm)
      : m_global_box(1.0,1.0,1.0), m_box(1.0,1.0,1.0)
    {
    m_pos_proc.resize(comm->size());
    m_rank = comm->rank();

    if (m_rank ==0)
    {
    m_global_box = init.getBox();
    // get global box dimensions
    double Lx_g = m_global_box.xhi - m_global_box.xlo;
    double Ly_g = m_global_box.yhi - m_global_box.ylo;
    double Lz_g = m_global_box.zhi - m_global_box.zlo;

        // Calulate the number of sub-domains in every direction
    // by minimizing the surface area between domains at constant number of domains
    double min_surface_area = Lx_g*Ly_g*comm->size()+Lx_g*Lz_g+Ly_g*Lz_g;

        // initial guess
    m_nx = 1; m_ny = 1; m_nz = comm->size();

    for (unsigned int nx = 1; nx <= (unsigned int) comm->size(); nx++)
           {
       for (unsigned int ny = 1; nx*ny <= (unsigned int) comm->size(); ny++)
               {
           for (unsigned int nz = 1; nx*ny*nz <= (unsigned int) comm->size(); nz++)
           {
           if (nx*ny*nz != (unsigned int) comm->size()) continue;
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

    // calculate physical box dimensions
    m_box_proc.resize(comm->size(),BoxDim(1.0,1.0,1.0));
    m_grid_pos_proc.resize(comm->size());
    for (unsigned int rank = 0; rank < (unsigned int) comm->size(); rank++)
        {
        BoxDim box(Lx_g,Ly_g,Lz_g);
        m_Lx = Lx_g/(double)m_nx;
        m_Ly = Ly_g/(double)m_ny;
        m_Lz = Lz_g/(double)m_nz;

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
    broadcast(*comm, m_global_box, 0);

    // broadcast grid dimensions
    broadcast(*comm, m_nx, 0);
    broadcast(*comm, m_ny, 0);
    broadcast(*comm, m_nz, 0);

    // distribute local box dimensions
    scatter(*comm, m_box_proc, m_box, 0);

    // distribute grid positions
    scatter(*comm, m_grid_pos_proc, m_grid_pos, 0);

    // broadcast number of particle types
    m_num_particle_types = init.getNumParticleTypes();
    broadcast(*comm, m_num_particle_types, 0);

    // broadcast particle type mapping
    m_type_mapping = init.getTypeMapping();
    broadcast(*comm, m_type_mapping, 0);

    if (m_rank == 0)
    {
    // distribute particles on processors
    m_nglobal = init.getNumParticles();
    SnapshotParticleData snap(m_nglobal);

    // initialize with default values
    for (unsigned int i = 0; i < m_nglobal; i++)
       {
       snap.mass[i] = 1.0;
       snap.diameter[i] = 1.0;
       snap.body[i] = NO_BODY;
       snap.global_tag[i] = i;
       }

    // use the Initializer to initialize the global particle data snapshot
    init.initSnapshot(snap);

    m_pos_proc.resize(comm->size());
    m_vel_proc.resize(comm->size());
    m_accel_proc.resize(comm->size());
    m_type_proc.resize(comm->size());
    m_mass_proc.resize(comm->size());
    m_charge_proc.resize(comm->size());
    m_diameter_proc.resize(comm->size());
    m_image_proc.resize(comm->size());
    m_rtag_proc.resize(comm->size());
    m_body_proc.resize(comm->size());
    m_global_tag_proc.resize(comm->size());


    std::vector< unsigned int > N_proc;
    N_proc.resize(comm->size(),0);

    for (std::vector<Scalar3>::iterator it=snap.pos.begin(); it != snap.pos.end(); it++)
        {
        // determine domain the particle lies in
        unsigned int i= (it->x-m_global_box.zlo)/m_Lx;
        unsigned int j= (it->y-m_global_box.ylo)/m_Ly;
        unsigned int k= (it->z-m_global_box.zlo)/m_Lz;

        // treat particles lying exactly on the boundary
        if (i == m_nx)
        {
        i = 0;
        it->x = m_global_box.xlo;
        }
        if (j == m_ny)
        {
        j = 0;
        it->y = m_global_box.ylo;
        }
        if (k == m_nz)
        {
        k = 0;
        it->z = m_global_box.zlo;
        }

        unsigned int rank = k*m_nx*m_ny + j * m_nx + i;

        // fill up per-processor data structures
        unsigned int idx = it - snap.pos.begin();
        m_pos_proc[rank].push_back(snap.pos[idx]);
        m_vel_proc[rank].push_back(snap.vel[idx]);
        m_accel_proc[rank].push_back(snap.accel[idx]);
        m_type_proc[rank].push_back(snap.type[idx]);
        m_mass_proc[rank].push_back(snap.mass[idx]);
        m_charge_proc[rank].push_back(snap.charge[idx]);
        m_diameter_proc[rank].push_back(snap.diameter[idx]);
        m_image_proc[rank].push_back(snap.image[idx]);
        m_body_proc[rank].push_back(snap.body[idx]);
        m_global_tag_proc[rank].push_back(snap.global_tag[idx]);

        // the particles are pushed to the processors in (local) tag order
        m_rtag_proc[rank].push_back(N_proc[rank]++);

        }
     }

     // distribute positions
     scatter(*comm, m_pos_proc,m_pos,0);

     // distribute velocities
     scatter(*comm, m_vel_proc,m_vel,0);

     // distribute accelerations
     scatter(*comm, m_accel_proc, m_accel, 0);

     // distribute particle types
     scatter(*comm, m_type_proc, m_type, 0);

     // distribute particle masses
     scatter(*comm, m_mass_proc, m_mass, 0);

     // distribute particle charges
     scatter(*comm, m_charge_proc, m_charge, 0);

     // distribute particle diameters`
     scatter(*comm, m_diameter_proc, m_diameter, 0);

     // distribute particle images
     scatter(*comm, m_image_proc, m_image, 0);

     // distribute particle reverse-lookup tags
     scatter(*comm, m_rtag_proc, m_rtag, 0);

     // distribute body ids
     scatter(*comm, m_body_proc, m_body, 0);

     // distribute global tags
     scatter(*comm, m_global_tag_proc, m_global_tag, 0);

     // broadcast global number of particles
     broadcast(*comm, m_nglobal, 0);

     std::cout << "rank " << comm->rank() << " " << m_pos.size() << " ptls." << std::endl;

     m_N = m_pos.size();

     }

//! Get rank of neighboring domain
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
    if (dir ==0 )
    dim = m_nx;
    else
    if (dir == 1) dim = m_ny;
    else
    if (dir == 2) dim = m_nz;

    return dim;
    }

//! Initialize a snapshot with local particle data
void MPIInitializer::initSnapshot(SnapshotParticleData& snapshot) const
    {
    // copy over vectors into snapshot
    snapshot.pos = m_pos;
    snapshot.vel = m_vel;
    snapshot.accel = m_accel;
    snapshot.type = m_type;
    snapshot.mass = m_mass;
    snapshot.charge = m_charge;
    snapshot.diameter = m_diameter;
    snapshot.image = m_image;
    snapshot.rtag = m_rtag;
    snapshot.body = m_body;
    snapshot.global_tag = m_global_tag;
    }

//! Export MPIInitializer class to python
void export_MPIInitializer()
    {
    class_<MPIInitializer, bases<ParticleDataInitializer>, boost::noncopyable >("MPIInitializer",
           init< ParticleDataInitializer&, boost::shared_ptr<boost::mpi::communicator> >())
    .def("getNeighborRank", &MPIInitializer::getNeighborRank)
    .def("getGlobalBox", &MPIInitializer::getGlobalBox)
    .def("getDimension", &MPIInitializer::getDimension)
    ;
    }
#endif // ENABLE_MPI
