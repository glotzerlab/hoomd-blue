// Copyright (c) 2009-2019 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.


// Maintainer: jglaser

/*! \file SFCPackUpdaterGPU.cc
    \brief Defines the SFCPackUpdaterGPU class
*/

#ifdef ENABLE_CUDA


#include "SFCPackUpdaterGPU.h"
#include "SFCPackUpdaterGPU.cuh"

#include <math.h>
#include <stdexcept>
#include <algorithm>
#include <fstream>
#include <iostream>

using namespace std;
namespace py = pybind11;

//! Constructor
/*! \param sysdef System to perform sorts on
 */
SFCPackUpdaterGPU::SFCPackUpdaterGPU(std::shared_ptr<SystemDefinition> sysdef)
        : SFCPackUpdater(sysdef)
    {
    m_exec_conf->msg->notice(5) << "Constructing SFCPackUpdaterGPU" << endl;

    // perform lots of sanity checks
    assert(m_pdata);

    GlobalArray<unsigned int> gpu_sort_order(m_pdata->getMaxN(), m_exec_conf);
    m_gpu_sort_order.swap(gpu_sort_order);
    TAG_ALLOCATION(m_gpu_sort_order);

    GlobalArray<unsigned int> gpu_particle_bins(m_pdata->getMaxN(), m_exec_conf);
    m_gpu_particle_bins.swap(gpu_particle_bins);
    TAG_ALLOCATION(m_gpu_particle_bins);

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

        ArrayHandle<Scalar4> d_angmom_alt(m_pdata->getAltAngularMomentumArray(), access_location::device, access_mode::overwrite);
        ArrayHandle<Scalar3> d_inertia_alt(m_pdata->getAltMomentsOfInertiaArray(), access_location::device, access_mode::overwrite);
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
        ArrayHandle<Scalar4> d_angmom(m_pdata->getAngularMomentumArray(), access_location::device, access_mode::read);
        ArrayHandle<Scalar3> d_inertia(m_pdata->getMomentsOfInertiaArray(), access_location::device, access_mode::read);

        ArrayHandle<Scalar> d_net_virial(m_pdata->getNetVirial(), access_location::device, access_mode::read);
        ArrayHandle<Scalar4> d_net_force(m_pdata->getNetForce(), access_location::device, access_mode::read);
        ArrayHandle<Scalar4> d_net_torque(m_pdata->getNetTorqueArray(), access_location::device, access_mode::read);

        // access rtags
        ArrayHandle<unsigned int> d_rtag(m_pdata->getRTags(), access_location::device, access_mode::readwrite);

        // access sorted order
        ArrayHandle<unsigned int> d_gpu_sort_order(m_gpu_sort_order, access_location::device, access_mode::read);

        // apply sorted order and re-build rtags
        gpu_apply_sorted_order(m_pdata->getN(),
            m_pdata->getNGhosts(),
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
            d_angmom.data,
            d_angmom_alt.data,
            d_inertia.data,
            d_inertia_alt.data,
            d_net_virial.data,
            d_net_virial_alt.data,
            m_pdata->getNetVirial().getPitch(),
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
    m_pdata->swapAngularMomenta();
    m_pdata->swapMomentsOfInertia();
    m_pdata->swapNetVirial();
    m_pdata->swapNetForce();
    m_pdata->swapNetTorque();
    }

void export_SFCPackUpdaterGPU(py::module& m)
    {
    py::class_<SFCPackUpdaterGPU, std::shared_ptr<SFCPackUpdaterGPU> >(m,"SFCPackUpdaterGPU",py::base<SFCPackUpdater>())
    .def(py::init< std::shared_ptr<SystemDefinition> >())
    ;
    }


#endif
