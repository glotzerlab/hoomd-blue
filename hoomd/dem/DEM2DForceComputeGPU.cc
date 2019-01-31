// Copyright (c) 2009-2019 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.

// Maintainer: mspells

/*! \file DEM2DForceComputeGPU.cc
  \brief Defines the DEM2DForceComputeGPU class
*/

#ifdef WIN32
#pragma warning( push )
#pragma warning( disable : 4103 4244 )
#endif

#ifdef ENABLE_CUDA

#include "DEM2DForceComputeGPU.h"
#include "cuda_runtime.h"

#include <stdexcept>

#include <hoomd/extern/pybind/include/pybind11/pybind11.h>

using namespace std;

/*! \param sysdef System to compute forces on
  \param nlist Neighborlist to use for computing the forces
  \param r_cut Cutoff radius beyond which the force is 0
  \param potential Global potential parameters for the compute

  \post memory is allocated with empty shape vectors

  \note The DEM2DForceComputeGPU does not own the Neighborlist, the caller should
  delete the neighborlist when done.
*/

template<typename Real, typename Real2, typename Real4, typename Potential>
DEM2DForceComputeGPU<Real, Real2, Real4, Potential>::DEM2DForceComputeGPU(
    std::shared_ptr<SystemDefinition> sysdef,
    std::shared_ptr<NeighborList> nlist,
    Scalar r_cut, Potential potential)
    : DEM2DForceCompute<Real, Real4, Potential>(sysdef, nlist, r_cut, potential),
    m_vertices(this->m_pdata->getNTypes(), this->m_exec_conf),
    m_num_shape_vertices(this->m_pdata->getNTypes(), this->m_exec_conf)
    {
    // can't run on the GPU if there aren't any GPUs in the execution configuration
    if (!this->m_exec_conf->isCUDAEnabled())
        {
        this->m_exec_conf->msg->error() << "Creating a DEM2DForceComputeGPU with no GPU in the execution configuration" << endl;
        throw std::runtime_error("Error initializing DEM2DForceComputeGPU");
        }

    m_tuner.reset(new Autotuner(32, 1024, 32, 5, 100000, "dem_2d", this->m_exec_conf));
    }

/*! Destructor. */
template<typename Real, typename Real2, typename Real4, typename Potential>
DEM2DForceComputeGPU<Real, Real2, Real4, Potential>::~DEM2DForceComputeGPU()
    {
    }

/*! setParams: set the vertices for a numeric particle type from a python list.
  \param type Particle type index
  \param vertices Python list of 2D vertices specifying a polygon
*/
template<typename Real, typename Real2, typename Real4, typename Potential>
void DEM2DForceComputeGPU<Real, Real2, Real4, Potential>::setParams(unsigned int type,
    const pybind11::list &vertices)
    {
    DEM2DForceCompute<Real, Real4, Potential>::setParams(type, vertices);
    createGeometry();
    }

/*! \post The DEM2D forces are computed for the given timestep on the GPU.
  The neighborlist's compute method is called to ensure that it is up to date
  before forces are computed.
  \param timestep Current time step of the simulation

  Calls gpu_compute_dem2d_forces to do the dirty work.
*/
template<typename Real, typename Real2, typename Real4, typename Potential>
void DEM2DForceComputeGPU<Real, Real2, Real4, Potential>::computeForces(unsigned int timestep)
    {
    typedef DEMEvaluator<Real, Real4, Potential> Evaluator;
    // start by updating the neighborlist
    this->m_nlist->compute(timestep);

    // start the profile
    if (this->m_prof) this->m_prof->push(this->m_exec_conf, "DEM2D pair");

    // The GPU implementation CANNOT handle a half neighborlist, error out now
    bool third_law = this->m_nlist->getStorageMode() == NeighborList::half;
    if (third_law)
        {
        this->m_exec_conf->msg->error() << "DEM2DForceComputeGPU cannot handle a half neighborlist" << endl;
        throw runtime_error("Error computing forces in DEM2DForceComputeGPU");
        }

    size_t threadsPerParticle(2*this->maxVertices());
    size_t particlesPerBlock(m_tuner->getParam()/threadsPerParticle);
    // cap the block size so we don't have forces for one particle
    // spread among multiple blocks
    particlesPerBlock = min(particlesPerBlock,
        (size_t)(this->m_exec_conf->dev_prop.maxThreadsPerBlock)/threadsPerParticle);
    // don't use too many registers (~86 per thread)
    particlesPerBlock = min(particlesPerBlock,
        (size_t)(this->m_exec_conf->dev_prop.regsPerBlock/threadsPerParticle/86));
    // we better calculate for at least one particle per block
    particlesPerBlock = max(particlesPerBlock, (size_t)1);

    // access the neighbor list, which just selects the neighborlist into the device's memory, copying
    // it there if needed
    ArrayHandle<unsigned int> d_n_neigh(this->m_nlist->getNNeighArray(), access_location::device, access_mode::read);
    ArrayHandle<unsigned int> d_nlist(this->m_nlist->getNListArray(), access_location::device, access_mode::read);
    ArrayHandle<unsigned int> d_head_list(this->m_nlist->getHeadList(), access_location::device, access_mode::read);

    // access the particle data
    ArrayHandle<Scalar4> d_pos(this->m_pdata->getPositions(), access_location::device, access_mode::read);
    ArrayHandle<Scalar4> d_quat(this->m_pdata->getOrientationArray(), access_location::device, access_mode::read);
    ArrayHandle<Scalar>  d_diam(this->m_pdata->getDiameters(), access_location::device, access_mode::read);
    ArrayHandle<Scalar4>  d_velocity(this->m_pdata->getVelocities(), access_location::device, access_mode::read);
    ArrayHandle<Real2> d_vertices(m_vertices, access_location::device, access_mode::read);
    ArrayHandle<unsigned int> d_num_shape_vertices(m_num_shape_vertices, access_location::device, access_mode::read);
    BoxDim box = this->m_pdata->getBox();

    ArrayHandle<Scalar4> d_force(this->m_force,access_location::device,access_mode::overwrite);
    ArrayHandle<Scalar4> d_torque(this->m_torque,access_location::device,access_mode::overwrite);
    ArrayHandle<Scalar> d_virial(this->m_virial,access_location::device,access_mode::overwrite);

    // if there is no work to do, exit early
    if (this->m_pdata->getN() == 0) return;

    // run the kernel on all GPUs in parallel
    m_tuner->begin();
    gpu_compute_dem2d_forces<Real, Real2, Real4, Evaluator>(
        d_force.data,
        d_torque.data,
        d_virial.data,
        this->m_virial.getPitch(),
        this->m_pdata->getN(),
        this->m_pdata->getNGhosts(),
        d_pos.data,
        d_quat.data,
        d_vertices.data,
        d_num_shape_vertices.data,
        d_diam.data,
        d_velocity.data,
        numVertices(),
        box,
        d_n_neigh.data,
        d_nlist.data,
        d_head_list.data,
        this->m_evaluator,
        this->m_r_cut * this->m_r_cut,
        this->m_shapes.size(),
        particlesPerBlock, this->maxVertices());
    if (this->m_exec_conf->isCUDAErrorCheckingEnabled())
        CHECK_CUDA_ERROR();
    m_tuner->end();

    Scalar avg_neigh = this->m_nlist->estimateNNeigh();
    int64_t n_calc = int64_t(avg_neigh * this->m_pdata->getN());
    int64_t mem_transfer = this->m_pdata->getN() * (4 + 16 + 20) + n_calc * (4 + 16);
    int64_t flops = n_calc * (3+12+5+2+3+11+3+8+7);
    if (this->m_prof) this->m_prof->pop(this->m_exec_conf, flops, mem_transfer);
    }

/*!
  createGeometry: Update the device-side list of vertices and vertex indices.
*/
template<typename Real, typename Real2, typename Real4, typename Potential>
void DEM2DForceComputeGPU<Real, Real2, Real4, Potential>::createGeometry()
    {
    const size_t nVerts(numVertices());

    // resize the geometry arrays if necessary
    if(m_vertices.getNumElements() != nVerts)
        m_vertices.resize(nVerts);

    if(this->m_shapes.size() != m_num_shape_vertices.getNumElements())
        m_num_shape_vertices.resize(this->m_shapes.size());

    ArrayHandle<Real2> h_vertices(m_vertices, access_location::host,
        access_mode::overwrite);
    ArrayHandle<unsigned int> h_num_shape_vertices(
        m_num_shape_vertices, access_location::host, access_mode::overwrite);

    for(size_t i(0), k(0); i < this->m_shapes.size(); ++i)
        {
        h_num_shape_vertices.data[i] = this->m_shapes[i].size();
        for(size_t j(0); j < this->m_shapes[i].size(); ++j, ++k)
            {
            h_vertices.data[k].x = this->m_shapes[i][j].x;
            h_vertices.data[k].y = this->m_shapes[i][j].y;
            }
        }
    }

/*!
  numVertices: Returns the total number of vertices for all shapes
  in the system.
*/
template<typename Real, typename Real2, typename Real4, typename Potential>
size_t DEM2DForceComputeGPU<Real, Real2, Real4, Potential>::numVertices() const
    {
    size_t result(0);

    for(typename std::vector<std::vector<vec2<Real> > >::const_iterator shapeIter(this->m_shapes.begin());
        shapeIter != this->m_shapes.end(); ++shapeIter)
        result += shapeIter->size() ? shapeIter->size(): 1;

    return result;
    }

/*!
  maxVertices: returns the maximum number of vertices among all shapes
  in the system.
*/
template<typename Real, typename Real2, typename Real4, typename Potential>
size_t DEM2DForceComputeGPU<Real, Real2, Real4, Potential>::maxVertices() const
    {
    size_t result(1);

    for(typename std::vector<std::vector<vec2<Real> > >::const_iterator shapeIter(this->m_shapes.begin());
        shapeIter != this->m_shapes.end(); ++shapeIter)
        result = max(result, shapeIter->size());

    return result;
    }

#endif

#ifdef WIN32
#pragma warning( pop )
#endif
