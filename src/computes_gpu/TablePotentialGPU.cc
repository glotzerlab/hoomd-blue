/*
Highly Optimized Object-oriented Many-particle Dynamics -- Blue Edition
(HOOMD-blue) Open Source Software License Copyright 2008, 2009 Ames Laboratory
Iowa State University and The Regents of the University of Michigan All rights
reserved.

HOOMD-blue may contain modifications ("Contributions") provided, and to which
copyright is held, by various Contributors who have granted The Regents of the
University of Michigan the right to modify and/or distribute such Contributions.

Redistribution and use of HOOMD-blue, in source and binary forms, with or
without modification, are permitted, provided that the following conditions are
met:

* Redistributions of source code must retain the above copyright notice, this
list of conditions, and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright notice, this
list of conditions, and the following disclaimer in the documentation and/or
other materials provided with the distribution.

* Neither the name of the copyright holder nor the names of HOOMD-blue's
contributors may be used to endorse or promote products derived from this
software without specific prior written permission.

Disclaimer

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDER AND CONTRIBUTORS ``AS IS''
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, AND/OR
ANY WARRANTIES THAT THIS SOFTWARE IS FREE OF INFRINGEMENT ARE DISCLAIMED.

IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT,
INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE
OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF
ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

// $Id$
// $URL$
// Maintainer: joaander

#ifdef WIN32
#pragma warning( push )
#pragma warning( disable : 4103 4244 )
#endif

#include <boost/python.hpp>
using namespace boost::python;
#include <boost/bind.hpp>
using namespace boost;

#include "TablePotentialGPU.h"
#include <stdexcept>

/*! \file TablePotentialGPU.cc
    \brief Defines the TablePotentialGPU class
*/

using namespace std;

/*! \param sysdef System to compute forces on
    \param nlist Neighborlist to use for computing the forces
    \param table_width Width the tables will be in memory
*/
TablePotentialGPU::TablePotentialGPU(boost::shared_ptr<SystemDefinition> sysdef,
                                     boost::shared_ptr<NeighborList> nlist,
                                     unsigned int table_width)
	: TablePotential(sysdef, nlist, table_width), m_block_size(64)
    {
    // can't run on the GPU if there aren't any GPUs in the execution configuration
    if (exec_conf.gpu.size() == 0)
        {
        cerr << endl << "***Error! Creating a LJForceComputeGPU with no GPU in the execution configuration" << endl << endl;
        throw std::runtime_error("Error initializing LJForceComputeGPU");
        }
    }

/*! \param block_size Block size to set
*/
void TablePotentialGPU::setBlockSize(int block_size)
    {
    m_block_size = block_size;
    }

/*! \post The table based forces are computed for the given timestep. The neighborlist's
compute method is called to ensure that it is up to date.

\param timestep specifies the current time step of the simulation

Calls gpu_compute_table_forces to do the leg work
*/
void TablePotentialGPU::computeForces(unsigned int timestep)
    {
    // start by updating the neighborlist
    m_nlist->compute(timestep);
    
    // start the profile
    if (m_prof) m_prof->push(exec_conf, "Table pair");
    
    // The GPU implementation CANNOT handle a half neighborlist, error out now
    bool third_law = m_nlist->getStorageMode() == NeighborList::half;
    if (third_law)
        {
        cerr << endl << "***Error! TablePotentialGPU cannot handle a half neighborlist" << endl << endl;
        throw runtime_error("Error computing forces in TablePotentialGPU");
        }
        
    // access the neighbor list, which just selects the neighborlist into the device's memory, copying
    // it there if needed
    vector<gpu_nlist_array>& nlist = m_nlist->getListGPU();
    
    // access the particle data
    vector<gpu_pdata_arrays>& pdata = m_pdata->acquireReadOnlyGPU();
    gpu_boxsize box = m_pdata->getBoxGPU();
    
    // access the table data
    ArrayHandle<Scalar2> d_tables(m_tables, access_location::device, access_mode::read);
    ArrayHandle<Scalar4> d_params(m_params, access_location::device, access_mode::read);
    
    // run the kernel on all GPUs in parallel
    exec_conf.tagAll(__FILE__, __LINE__);
    
    for (unsigned int cur_gpu = 0; cur_gpu < exec_conf.gpu.size(); cur_gpu++)
        exec_conf.gpu[cur_gpu]->callAsync(bind(gpu_compute_table_forces, m_gpu_forces[cur_gpu].d_data, pdata[cur_gpu], box, nlist[cur_gpu], d_tables.data, d_params.data, m_ntypes, m_table_width, m_block_size));
    exec_conf.syncAll();
    
    m_pdata->release();
    
    // the force data is now only up to date on the gpu
    m_data_location = gpu;
    
    if (m_prof) m_prof->pop(exec_conf);
    }

void export_TablePotentialGPU()
    {
    class_<TablePotentialGPU, boost::shared_ptr<TablePotentialGPU>, bases<TablePotential>, boost::noncopyable >
    ("TablePotentialGPU",
     init< boost::shared_ptr<SystemDefinition>,
     boost::shared_ptr<NeighborList>,
     unsigned int >())
    .def("setBlockSize", &TablePotentialGPU::setBlockSize)
    ;
    }

#ifdef WIN32
#pragma warning( pop )
#endif