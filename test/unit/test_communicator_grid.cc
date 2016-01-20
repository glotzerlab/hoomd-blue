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

#ifdef ENABLE_MPI

//! name the boost unit test module
#define BOOST_TEST_MODULE CommunicatorGridTests

#include <boost/test/unit_test.hpp>

// this has to be included after naming the test module
#include "MPITestSetup.h"

#include "System.h"

#include <boost/shared_ptr.hpp>
#include <boost/bind.hpp>

#include "CommunicatorGrid.h"

#ifdef ENABLE_CUDA
#include "CommunicatorGridGPU.h"
#endif

#include <algorithm>

// first test, to ensure that all the ghost cells are updated from the correct neighbors
template< class CG_uint >
void test_communicate_grid_basic(boost::shared_ptr<ExecutionConfiguration> exec_conf)
    {
    // create a system with eight particles
    boost::shared_ptr<SystemDefinition> sysdef(new SystemDefinition(8,           // number of particles
                                                             BoxDim(2.0), // box dimensions
                                                             1,           // number of particle types
                                                             0,           // number of bond types
                                                             0,           // number of angle types
                                                             0,           // number of dihedral types
                                                             0,           // number of dihedral types
                                                             exec_conf));



    boost::shared_ptr<ParticleData> pdata(sysdef->getParticleData());

    // initialize domain decomposition on processor with rank 0
    boost::shared_ptr<DomainDecomposition> decomposition(new DomainDecomposition(exec_conf, pdata->getBox().getL()));
    pdata->setDomainDecomposition(decomposition);

    unsigned int nx = 6;
    unsigned int ny = 6;
    unsigned int nz = 6;

    uint3 n_ghost_offset=make_uint3(1,1,1);

    // test inner-to-outer communication
    uint3 dim = make_uint3(nx-2, ny-2, nz-2);
    uint3 embed = make_uint3(nx, ny, nz);
    CG_uint grid_comm(sysdef, dim, embed, n_ghost_offset,false);
    GPUArray<unsigned int> grid(nx*ny*nz, exec_conf);

    unsigned int rank = exec_conf->getRank();
        {
        ArrayHandle<unsigned int> h_grid(grid, access_location::host, access_mode::overwrite);
        // set inner ghost layer to the current rank
        memset(h_grid.data,0, grid.getNumElements()*sizeof(unsigned int));
        for (unsigned int x = 0; x < nx; x++)
            for (unsigned int y = 0; y < ny; y++)
                for (unsigned int z = 0; z < nz; z++)
                    {
                    if (x == 1 || x == nx-2 || y == 1 || y==ny-2 || z ==1 || z==nz-2)
                        h_grid.data[x+embed.x*(y+embed.y*z)] = rank;
                    }
        }

    // perform communication
    grid_comm.communicate(grid);

    Index3D didx = decomposition->getDomainIndexer();
    uint3 grid_pos = didx.getTriple(rank);

        {
        ArrayHandle<unsigned int> h_grid(grid, access_location::host, access_mode::read);

        for (unsigned int x = 0; x < nx; x++)
            for (unsigned int y = 0; y < ny; y++)
                for (unsigned int z = 0; z < nz; z++)
                    {
                    unsigned int val = h_grid.data[x+embed.x*(y+embed.y*z)];
                    // check outer ghost cells 
                    int3 neighbor_dir = make_int3(0,0,0);
                    if (x == 0)
                        neighbor_dir.x = -1;
                    else if (x==nx -1)
                        neighbor_dir.x = 1;

                    if (y == 0)
                        neighbor_dir.y = -1;
                    else if (y==ny-1)
                        neighbor_dir.y = 1;

                    if (z == 0)
                        neighbor_dir.z = -1;
                    else if (z==nz-1)
                        neighbor_dir.z = 1;

                    if (! neighbor_dir.x && ! neighbor_dir.y && !neighbor_dir.z)
                        {
                        // check inner ghost cells
                        if (x == 1 || x == nx-2 || y == 1 || y==ny-2 || z ==1 || z==nz-2)
                            BOOST_CHECK_EQUAL(val, rank);
                        else
                            BOOST_CHECK_EQUAL(val,0);

                        continue;
                        }

                    int3 grid_idx = make_int3(grid_pos.x + neighbor_dir.x,
                                              grid_pos.y + neighbor_dir.y,
                                              grid_pos.z + neighbor_dir.z);
                    if (grid_idx.x == (int)didx.getW())
                        grid_idx.x = 0;
                    else if (grid_idx.x < 0)
                        grid_idx.x += (int)didx.getW();

                    if (grid_idx.y == (int)didx.getH())
                        grid_idx.y = 0;
                    else if (grid_idx.y < 0)
                        grid_idx.y += (int)didx.getH();

                    if (grid_idx.z == (int)didx.getD())
                        grid_idx.z = 0;
                    else if (grid_idx.z < 0)
                        grid_idx.z +=(int) didx.getD();

                    unsigned int neighbor_rank = didx(grid_idx.x,grid_idx.y,grid_idx.z);
                    BOOST_CHECK_EQUAL(val, neighbor_rank);
                    }
        } //end ArrayHandle scope

    // test outer-to-inner communication
    CG_uint grid_comm_3(sysdef, dim, embed, n_ghost_offset, true);
    GPUArray<unsigned int> grid_3(nx*ny*nz, exec_conf);

        {
        ArrayHandle<unsigned int> h_grid_3(grid_3, access_location::host, access_mode::overwrite);
        // set inner ghost layer to the current rank
        memset(h_grid_3.data,0, grid_3.getNumElements()*sizeof(unsigned int));
        for (unsigned int x = 0; x < nx; x++)
            for (unsigned int y = 0; y < ny; y++)
                for (unsigned int z = 0; z < nz; z++)
                    {
                    if (x == 0 || x == nx-1 || y == 0 || y == ny-1 || z == 0 || z == nz-1)
                        h_grid_3.data[x+embed.x*(y+embed.y*z)] = rank;
                    }
        }


    // perform communication
    grid_comm_3.communicate(grid_3);

        {
        ArrayHandle<unsigned int> h_grid_3(grid_3, access_location::host, access_mode::read);

        for (unsigned int x = 1; x < nx-1; x++)
            for (unsigned int y = 1; y < ny-1; y++)
                for (unsigned int z = 1; z < nz-1; z++)
                    {
                    unsigned int val = h_grid_3.data[x+embed.x*(y+embed.y*z)];
                    unsigned int sum = 0;
                    // check inner ghost cells
                    int3 grid_idx = make_int3(grid_pos.x,
                                              grid_pos.y,
                                              grid_pos.z);
                    uint3 count = make_uint3(0,0,0);
                    if (x == 1)
                        {
                        grid_idx.x -= 1;
                        if (grid_idx.x < 0)
                            grid_idx.x += (int)didx.getW();
                        count.x++;
                       }
                    else if (x==nx -2)
                        {
                        grid_idx.x += 1;
                        if (grid_idx.x == (int)didx.getW())
                            grid_idx.x = 0;
                        count.x++;
                        }

                    if (y == 1)
                        {
                        grid_idx.y -= 1;
                        if (grid_idx.y < 0)
                            grid_idx.y += (int)didx.getH();
                        count.y++;
                        }
                    else if (y==ny-2)
                        {
                        grid_idx.y += 1;
                        if (grid_idx.y == (int)didx.getH())
                            grid_idx.y = 0;
                        count.y++;
                        }

                    if (z == 1)
                        {
                        grid_idx.z -= 1;
                        if (grid_idx.z < 0)
                            grid_idx.z += (int)didx.getD();
                        count.z++;
                        }
                    else if (z==nz-2)
                        {
                        grid_idx.z += 1;
                        if (grid_idx.z == (int)didx.getD())
                            grid_idx.z = 0;
                        count.z++;
                        }

                    // inner cells
                    if (! count.x && ! count.y && ! count.z)
                        {
                        BOOST_CHECK_EQUAL(val,0);
                        continue;
                        }

                    for (unsigned int i = 0; i <= count.x; ++i)
                        for (unsigned int j = 0; j <= count.y; ++j)
                            for (unsigned int k = 0; k <= count.z; ++k)
                                {
                                uint3 grid_idx_2 = grid_pos;
                                if (i) grid_idx_2.x = grid_idx.x;
                                if (j) grid_idx_2.y = grid_idx.y;
                                if (k) grid_idx_2.z = grid_idx.z;
                                if (i || j || k)
                                    sum += didx(grid_idx_2.x,grid_idx_2.y,grid_idx_2.z);
                                }

                    BOOST_CHECK_EQUAL(val, sum);
                    }
        } //end ArrayHandle scope

    }

//! Test to check that all elements are received and updated in correct order
template<class CG_uint >
void test_communicate_grid_positions(boost::shared_ptr<ExecutionConfiguration> exec_conf)
    {
    // create a system with eight particles
    boost::shared_ptr<SystemDefinition> sysdef(new SystemDefinition(8,           // number of particles
                                                             BoxDim(2.0), // box dimensions
                                                             1,           // number of particle types
                                                             0,           // number of bond types
                                                             0,           // number of angle types
                                                             0,           // number of dihedral types
                                                             0,           // number of dihedral types
                                                             exec_conf));



    boost::shared_ptr<ParticleData> pdata(sysdef->getParticleData());

    boost::shared_ptr<DomainDecomposition> decomposition(new DomainDecomposition(exec_conf, pdata->getBox().getL()));
    pdata->setDomainDecomposition(decomposition);

    unsigned int nx = 6;
    unsigned int ny = 6;
    unsigned int nz = 6;

    uint3 n_ghost_offset=make_uint3(1,1,1);
    uint3 dim = make_uint3(nx-2, ny-2, nz-2);
    uint3 embed = make_uint3(nx, ny, nz);

    // second test, ensure that ghost cells are updated in correct order
    CG_uint grid_comm_2(sysdef, dim, embed, n_ghost_offset,false);
    GPUArray<unsigned int> grid_2(nx*ny*nz, exec_conf);

        {
        ArrayHandle<unsigned int> h_grid_2(grid_2, access_location::host, access_mode::overwrite);
        // set inner ghost layer to grid coordinates
        memset(h_grid_2.data,0, grid_2.getNumElements()*sizeof(unsigned int));
        for (unsigned int x = 1; x < nx-1; x++)
            for (unsigned int y = 1; y < ny-1; y++)
                for (unsigned int z = 1; z < nz-1; z++)
                    {
                    h_grid_2.data[x+embed.x*(y+embed.y*z)] = x+embed.x*(y+embed.y*z);
                    }
        }

    // perform communication
    grid_comm_2.communicate(grid_2);

        {
        ArrayHandle<unsigned int> h_grid_2(grid_2, access_location::host, access_mode::read);
        for (unsigned int x = 0; x < nx; x++)
            for (unsigned int y = 0; y < ny; y++)
                for (unsigned int z = 0; z < nz; z++)
                    {
                    unsigned int val = h_grid_2.data[x+embed.x*(y+embed.y*z)];

                    uint3 compare_xyz = make_uint3(x,y,z);

                    if (x == 0)
                        compare_xyz.x = nx-2;
                    else if (x == nx-1)
                        compare_xyz.x = 1;

                    if (y == 0)
                        compare_xyz.y = ny-2;
                    else if (y == ny-1)
                        compare_xyz.y = 1;

                    if (z == 0)
                        compare_xyz.z = nz-2;
                    else if (z == nz-1)
                        compare_xyz.z = 1;

                    unsigned int compare_val = compare_xyz.x +
                        embed.x * (compare_xyz.y + embed.y*compare_xyz.z);
                    BOOST_CHECK_EQUAL(compare_val, val);
                    }
        }
    }

//! Basic ghost grid exchange test
BOOST_AUTO_TEST_CASE( CommunicateGrid_test_basic )
    {
    test_communicate_grid_basic<CommunicatorGrid<unsigned int> >(
        boost::shared_ptr<ExecutionConfiguration>(new ExecutionConfiguration(ExecutionConfiguration::CPU)));
    }

//! Ghost grid exchange positions test
BOOST_AUTO_TEST_CASE( CommunicateGrid_test_positions )
    {
    test_communicate_grid_positions<CommunicatorGrid<unsigned int> >(
        boost::shared_ptr<ExecutionConfiguration>(new ExecutionConfiguration(ExecutionConfiguration::CPU)));
    }


#ifdef ENABLE_CUDA
//! Basic ghost grid exchange test on GPU
BOOST_AUTO_TEST_CASE( CommunicateGrid_test_basic_GPU )
    {
    test_communicate_grid_basic<CommunicatorGridGPU<unsigned int> >(
        boost::shared_ptr<ExecutionConfiguration>(new ExecutionConfiguration(ExecutionConfiguration::GPU)));
    }

//! Ghost grid exchange positions test on GPU
BOOST_AUTO_TEST_CASE( CommunicateGrid_test_positions_GPU )
    {
    test_communicate_grid_positions<CommunicatorGridGPU<unsigned int> >(
        boost::shared_ptr<ExecutionConfiguration>(new ExecutionConfiguration(ExecutionConfiguration::GPU)));
    }
#endif

#endif //ENABLE_MPI
