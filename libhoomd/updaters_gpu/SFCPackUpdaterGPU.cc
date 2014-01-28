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

/*! \file SFCPackUpdaterGPU.cc
    \brief Defines the SFCPackUpdaterGPU class
*/

#ifdef ENABLE_CUDA

#ifdef WIN32
#pragma warning( push )
#pragma warning( disable : 4103 4244 )
#endif

#include <boost/python.hpp>
using namespace boost::python;
using namespace boost;

#include <math.h>
#include <stdexcept>
#include <algorithm>
#include <fstream>
#include <iostream>

#include "SFCPackUpdaterGPU.h"
#include "SFCPackUpdaterGPU.cuh"

using namespace std;

//! Constructor
/*! \param sysdef System to perform sorts on
 */
SFCPackUpdaterGPU::SFCPackUpdaterGPU(boost::shared_ptr<SystemDefinition> sysdef)
        : SFCPackUpdater(sysdef)
    {
    m_exec_conf->msg->notice(5) << "Constructing SFCPackUpdaterGPU" << endl;

    // perform lots of sanity checks
    assert(m_pdata);

    GPUArray<unsigned int> gpu_sort_order(m_pdata->getMaxN(), m_exec_conf);
    m_gpu_sort_order.swap(gpu_sort_order);

    GPUArray<unsigned int> gpu_particle_bins(m_pdata->getMaxN(), m_exec_conf);
    m_gpu_particle_bins.swap(gpu_particle_bins);

    // create at ModernGPU context
    m_mgpu_context = mgpu::CreateCudaDeviceAttachStream(0);
    }

/*! reallocate the internal arrays
 */
void SFCPackUpdaterGPU::reallocate()
    {
    m_gpu_sort_order.resize(m_pdata->getMaxN());
    m_gpu_particle_bins.resize(m_pdata->getMaxN());
    }

/*! Destructor
 */
SFCPackUpdaterGPU::~SFCPackUpdaterGPU()
    {
    m_exec_conf->msg->notice(5) << "Destroying SFCPackUpdaterGPU" << endl;
    }

void SFCPackUpdaterGPU::getSortedOrder2D()
    {
    // on the GPU, getSortedOrder3D handles both cases
    getSortedOrder3D();
    }

void SFCPackUpdaterGPU::getSortedOrder3D()
    {
    // start by checking the saneness of some member variables
    assert(m_pdata);
    assert(m_gpu_sort_order.getNumElements() >= m_pdata->getN());

    // make even bin dimensions
    const BoxDim& box = m_pdata->getBox();

    // reallocate memory arrays if m_grid changed
    // also regenerate the traversal order
    if ((m_last_grid != m_grid || m_last_dim != 3) && m_sysdef->getNDimensions() == 3)
        {
        if (m_grid > 256)
            {
            unsigned int mb = m_grid*m_grid*m_grid*4 / 1024 / 1024;
            m_exec_conf->msg->warning() << "sorter is about to allocate a very large amount of memory (" << mb << "MB)"
                 << " and may crash." << endl;
            m_exec_conf->msg->warning() << "            Reduce the amount of memory allocated to prevent this by decreasing the " << endl;
            m_exec_conf->msg->warning() << "            grid dimension (i.e. sorter.set_params(grid=128) ) or by disabling it " << endl;
            m_exec_conf->msg->warning() << "            ( sorter.disable() ) before beginning the run()." << endl;
            }

        // generate the traversal order
        GPUArray<unsigned int> traversal_order(m_grid*m_grid*m_grid,m_exec_conf);
        m_traversal_order.swap(traversal_order);

        vector< unsigned int > reverse_order(m_grid*m_grid*m_grid);
        reverse_order.clear();

        // we need to start the hilbert curve with a seed order 0,1,2,3,4,5,6,7
        unsigned int cell_order[8];
        for (unsigned int i = 0; i < 8; i++)
            cell_order[i] = i;
        generateTraversalOrder(0,0,0, m_grid, m_grid, cell_order, reverse_order);

        // access traversal order
        ArrayHandle<unsigned int> h_traversal_order(m_traversal_order, access_location::host, access_mode::overwrite);

        for (unsigned int i = 0; i < m_grid*m_grid*m_grid; i++)
            h_traversal_order.data[reverse_order[i]] = i;

        m_last_grid = m_grid;
        // store the last system dimension computed so we can be mindful if that ever changes
        m_last_dim = m_sysdef->getNDimensions();
        }

    // sanity checks
    assert(m_gpu_particle_bins.getNumElements() >= m_pdata->getN());

    // access arrays
    ArrayHandle<Scalar4> d_pos(m_pdata->getPositions(), access_location::device, access_mode::read);
    ArrayHandle<unsigned int> d_gpu_particle_bins(m_gpu_particle_bins, access_location::device, access_mode::overwrite);
    ArrayHandle<unsigned int> d_gpu_sort_order(m_gpu_sort_order, access_location::device, access_mode::overwrite);
    ArrayHandle<unsigned int> d_traversal_order(m_traversal_order, access_location::device, access_mode::read);

    // put the particles in the bins and sort
    gpu_generate_sorted_order(m_pdata->getN(),
        d_pos.data,
        d_gpu_particle_bins.data,
        d_traversal_order.data,
        m_grid,
        d_gpu_sort_order.data,
        box,
        m_sysdef->getNDimensions() == 2,
        m_mgpu_context);

    if (m_exec_conf->isCUDAErrorCheckingEnabled()) CHECK_CUDA_ERROR();
    }

void SFCPackUpdaterGPU::applySortOrder()
    {
    assert(m_pdata);
    assert(m_gpu_sort_order.getNumElements() >= m_pdata->getN());

        {
        // access alternate arrays to write to
        ArrayHandle<Scalar4> d_pos_alt(m_pdata->getAltPositions(), access_location::device, access_mode::overwrite);
        ArrayHandle<Scalar4> d_vel_alt(m_pdata->getAltVelocities(), access_location::device, access_mode::overwrite);
        ArrayHandle<Scalar3> d_accel_alt(m_pdata->getAltAccelerations(), access_location::device, access_mode::overwrite);
        ArrayHandle<Scalar> d_charge_alt(m_pdata->getAltCharges(), access_location::device, access_mode::overwrite);
        ArrayHandle<Scalar> d_diameter_alt(m_pdata->getAltDiameters(), access_location::device, access_mode::overwrite);
        ArrayHandle<int3> d_image_alt(m_pdata->getAltImages(), access_location::device, access_mode::overwrite);
        ArrayHandle<unsigned int> d_body_alt(m_pdata->getAltBodies(), access_location::device, access_mode::overwrite);
        ArrayHandle<unsigned int> d_tag_alt(m_pdata->getAltTags(), access_location::device, access_mode::overwrite);
        ArrayHandle<Scalar4> d_orientation_alt(m_pdata->getAltOrientationArray(), access_location::device, access_mode::overwrite);

        ArrayHandle<Scalar> d_net_virial_alt(m_pdata->getAltNetVirial(), access_location::device, access_mode::overwrite);
        ArrayHandle<Scalar4> d_net_force_alt(m_pdata->getAltNetForce(), access_location::device, access_mode::overwrite);
        ArrayHandle<Scalar4> d_net_torque_alt(m_pdata->getAltNetTorqueArray(), access_location::device, access_mode::overwrite);

        // access live particle data to read from
        ArrayHandle<Scalar4> d_pos(m_pdata->getPositions(), access_location::device, access_mode::read);
        ArrayHandle<Scalar4> d_vel(m_pdata->getVelocities(), access_location::device, access_mode::read);
        ArrayHandle<Scalar3> d_accel(m_pdata->getAccelerations(), access_location::device, access_mode::read);
        ArrayHandle<Scalar> d_charge(m_pdata->getCharges(), access_location::device, access_mode::read);
        ArrayHandle<Scalar> d_diameter(m_pdata->getDiameters(), access_location::device, access_mode::read);
        ArrayHandle<int3> d_image(m_pdata->getImages(), access_location::device, access_mode::read);
        ArrayHandle<unsigned int> d_body(m_pdata->getBodies(), access_location::device, access_mode::read);
        ArrayHandle<unsigned int> d_tag(m_pdata->getTags(), access_location::device, access_mode::read);
        ArrayHandle<Scalar4> d_orientation(m_pdata->getOrientationArray(), access_location::device, access_mode::read);

        ArrayHandle<Scalar> d_net_virial(m_pdata->getNetVirial(), access_location::device, access_mode::read);
        ArrayHandle<Scalar4> d_net_force(m_pdata->getNetForce(), access_location::device, access_mode::read);
        ArrayHandle<Scalar4> d_net_torque(m_pdata->getNetTorqueArray(), access_location::device, access_mode::read);

        // access rtags
        ArrayHandle<unsigned int> d_rtag(m_pdata->getRTags(), access_location::device, access_mode::readwrite);

        // access sorted order
        ArrayHandle<unsigned int> d_gpu_sort_order(m_gpu_sort_order, access_location::device, access_mode::read);

        // apply sorted order and re-build rtags
        gpu_apply_sorted_order(m_pdata->getN(),
            d_gpu_sort_order.data,
            d_pos.data,
            d_pos_alt.data,
            d_vel.data,
            d_vel_alt.data,
            d_accel.data,
            d_accel_alt.data,
            d_charge.data,
            d_charge_alt.data,
            d_diameter.data,
            d_diameter_alt.data,
            d_image.data,
            d_image_alt.data,
            d_body.data,
            d_body_alt.data,
            d_tag.data,
            d_tag_alt.data,
            d_orientation.data,
            d_orientation_alt.data,
            d_net_virial.data,
            d_net_virial_alt.data,
            d_net_force.data,
            d_net_force_alt.data,
            d_net_torque.data,
            d_net_torque_alt.data,
            d_rtag.data);

        if (m_exec_conf->isCUDAErrorCheckingEnabled()) CHECK_CUDA_ERROR();
        }

    // make alternate arrays current
    m_pdata->swapPositions();
    m_pdata->swapVelocities();
    m_pdata->swapAccelerations();
    m_pdata->swapCharges();
    m_pdata->swapDiameters();
    m_pdata->swapImages();
    m_pdata->swapBodies();
    m_pdata->swapTags();
    m_pdata->swapOrientations();
    m_pdata->swapNetVirial();
    m_pdata->swapNetForce();
    m_pdata->swapNetTorque();
    }

void export_SFCPackUpdaterGPU()
    {
    class_<SFCPackUpdaterGPU, bases<SFCPackUpdater>, boost::shared_ptr<SFCPackUpdaterGPU>, boost::noncopyable>
    ("SFCPackUpdaterGPU", init< boost::shared_ptr<SystemDefinition> >())
    ;
    }

#ifdef WIN32
#pragma warning( pop )
#endif

#endif
