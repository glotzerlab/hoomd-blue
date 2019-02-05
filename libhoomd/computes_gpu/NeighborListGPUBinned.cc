/*
Highly Optimized Object-oriented Many-particle Dynamics -- Blue Edition
(HOOMD-blue) Open Source Software License Copyright 2009-2016 The Regents of
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

// Maintainer: joaander

/*! \file NeighborListGPUBinned.cc
    \brief Defines NeighborListGPUBinned
*/

#include "NeighborListGPUBinned.h"
#include "NeighborListGPUBinned.cuh"

#include <boost/python.hpp>
using namespace boost::python;

#ifdef ENABLE_MPI
#include "Communicator.h"
#endif

NeighborListGPUBinned::NeighborListGPUBinned(boost::shared_ptr<SystemDefinition> sysdef,
                                             Scalar r_cut,
                                             Scalar r_buff,
                                             boost::shared_ptr<CellList> cl)
    : NeighborListGPU(sysdef, r_cut, r_buff), m_cl(cl), m_param(0)
    {
    // create a default cell list if one was not specified
    if (!m_cl)
        m_cl = boost::shared_ptr<CellList>(new CellList(sysdef));

    m_cl->setRadius(1);
    // types are always required now
    m_cl->setComputeTDB(true);
    m_cl->setFlagIndex();

    CHECK_CUDA_ERROR();

    // initialize autotuner
    // the full block size and threads_per_particle matrix is searched,
    // encoded as block_size*10000 + threads_per_particle
    std::vector<unsigned int> valid_params;

    unsigned int max_tpp = m_exec_conf->dev_prop.warpSize;
    if (m_exec_conf->getComputeCapability() < 300)
        {
        // no wide parallelism on Fermi
        max_tpp = 1;
        }

    for (unsigned int block_size = 32; block_size <= 1024; block_size += 32)
        {
        unsigned int s=1;

        while (s <= max_tpp)
            {
            valid_params.push_back(block_size*10000 + s);
            s = s * 2;
            }
        }

    m_tuner.reset(new Autotuner(valid_params, 5, 100000, "nlist_binned", this->m_exec_conf));
    m_last_tuned_timestep = 0;

    #ifdef ENABLE_MPI
    // synchronize over MPI
    m_tuner->setSync(m_pdata->getDomainDecomposition() != 0);
    #endif

    // call this class's special setRCut
    setRCut(r_cut, r_buff);
    }

NeighborListGPUBinned::~NeighborListGPUBinned()
    {
    }

void NeighborListGPUBinned::setRCut(Scalar r_cut, Scalar r_buff)
    {
    NeighborListGPU::setRCut(r_cut, r_buff);

    Scalar rmax = getMaxRCut() + m_r_buff;
    if (m_diameter_shift)
        rmax += m_d_max - Scalar(1.0);

    m_cl->setNominalWidth(rmax);
    }

void NeighborListGPUBinned::setRCutPair(unsigned int typ1, unsigned int typ2, Scalar r_cut)
    {
    NeighborListGPU::setRCutPair(typ1,typ2,r_cut);

    Scalar rmax = getMaxRCut() + m_r_buff;
    if (m_diameter_shift)
        rmax += m_d_max - Scalar(1.0);

    m_cl->setNominalWidth(rmax);
    }

void NeighborListGPUBinned::setMaximumDiameter(Scalar d_max)
    {
    NeighborListGPU::setMaximumDiameter(d_max);

    // need to update the cell list settings appropriately
    Scalar rmax = getMaxRCut() + m_r_buff;
    if (m_diameter_shift)
        rmax += m_d_max - Scalar(1.0);

    m_cl->setNominalWidth(rmax);
    }

void NeighborListGPUBinned::buildNlist(unsigned int timestep)
    {
    if (m_storage_mode != full)
        {
        m_exec_conf->msg->error() << "Only full mode nlists can be generated on the GPU" << std::endl;
        throw std::runtime_error("Error computing neighbor list");
        }

    m_cl->compute(timestep);

    if (m_prof)
        m_prof->push(m_exec_conf, "compute");

    // acquire the particle data
    ArrayHandle<Scalar4> d_pos(m_pdata->getPositions(), access_location::device, access_mode::read);
    ArrayHandle<Scalar> d_diameter(m_pdata->getDiameters(), access_location::device, access_mode::read);
    ArrayHandle<unsigned int> d_body(m_pdata->getBodies(), access_location::device, access_mode::read);

    const BoxDim& box = m_pdata->getBox();
    Scalar3 nearest_plane_distance = box.getNearestPlaneDistance();

    // access the cell list data arrays
    ArrayHandle<unsigned int> d_cell_size(m_cl->getCellSizeArray(), access_location::device, access_mode::read);
    ArrayHandle<Scalar4> d_cell_xyzf(m_cl->getXYZFArray(), access_location::device, access_mode::read);
    ArrayHandle<Scalar4> d_cell_tdb(m_cl->getTDBArray(), access_location::device, access_mode::read);
    ArrayHandle<unsigned int> d_cell_adj(m_cl->getCellAdjArray(), access_location::device, access_mode::read);

    ArrayHandle<unsigned int> d_head_list(m_head_list, access_location::device, access_mode::read);
    ArrayHandle<unsigned int> d_Nmax(m_Nmax, access_location::device, access_mode::read);
    ArrayHandle<unsigned int> d_conditions(m_conditions, access_location::device, access_mode::readwrite);
    ArrayHandle<unsigned int> d_nlist(m_nlist, access_location::device, access_mode::overwrite);
    ArrayHandle<unsigned int> d_n_neigh(m_n_neigh, access_location::device, access_mode::overwrite);
    ArrayHandle<Scalar4> d_last_pos(m_last_pos, access_location::device, access_mode::overwrite);

    // the maximum cutoff that any particle can participate in
    Scalar rmax = getMaxRCut() + m_r_buff;
    if (m_diameter_shift)
        rmax += m_d_max - Scalar(1.0);

    ArrayHandle<Scalar> d_r_cut(m_r_cut, access_location::device, access_mode::read);
    ArrayHandle<Scalar> d_r_listsq(m_r_listsq, access_location::device, access_mode::read);

    if ((box.getPeriodic().x && nearest_plane_distance.x <= rmax * 2.0) ||
        (box.getPeriodic().y && nearest_plane_distance.y <= rmax * 2.0) ||
        (this->m_sysdef->getNDimensions() == 3 && box.getPeriodic().z && nearest_plane_distance.z <= rmax * 2.0))
        {
        m_exec_conf->msg->error() << "nlist: Simulation box is too small! Particles would be interacting with themselves." << std::endl;
        throw std::runtime_error("Error updating neighborlist bins");
        }

    // we should not call the tuner with MPI sync enabled
    // if the kernel is launched more than once in the same timestep,
    // since those kernel launches may occur only on some, not all MPI ranks
    bool tune = !m_param && m_last_tuned_timestep != timestep;

    if (tune) this->m_tuner->begin();
    unsigned int param = !m_param ? this->m_tuner->getParam() : m_param;
    unsigned int block_size = param / 10000;
    unsigned int threads_per_particle = param % 10000;

    gpu_compute_nlist_binned(d_nlist.data,
                             d_n_neigh.data,
                             d_last_pos.data,
                             d_conditions.data,
                             d_Nmax.data,
                             d_head_list.data,
                             d_pos.data,
                             d_body.data,
                             d_diameter.data,
                             m_pdata->getN(),
                             d_cell_size.data,
                             d_cell_xyzf.data,
                             d_cell_tdb.data,
                             d_cell_adj.data,
                             m_cl->getCellIndexer(),
                             m_cl->getCellListIndexer(),
                             m_cl->getCellAdjIndexer(),
                             box,
                             d_r_cut.data,
                             m_r_buff,
                             m_pdata->getNTypes(),
                             threads_per_particle,
                             block_size,
                             m_filter_body,
                             m_diameter_shift,
                             m_cl->getGhostWidth(),
                             m_exec_conf->getComputeCapability()/10);
    if(m_exec_conf->isCUDAErrorCheckingEnabled()) CHECK_CUDA_ERROR();
    if (tune) this->m_tuner->end();

    m_last_tuned_timestep = timestep;

    if (m_prof)
        m_prof->pop(m_exec_conf);
    }

void export_NeighborListGPUBinned()
    {
    class_<NeighborListGPUBinned, boost::shared_ptr<NeighborListGPUBinned>, bases<NeighborListGPU>, boost::noncopyable >
                     ("NeighborListGPUBinned", init< boost::shared_ptr<SystemDefinition>, Scalar, Scalar, boost::shared_ptr<CellList> >())
                    .def("setTuningParam", &NeighborListGPUBinned::setTuningParam)
                     ;
    }
