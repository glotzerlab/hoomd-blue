// Copyright (c) 2009-2019 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.

// Maintainer: mspells

/*! \file DEM3DForceComputeGPU.cc
  \brief Defines the DEM3DForceComputeGPU class
*/

#ifdef WIN32
#pragma warning( push )
#pragma warning( disable : 4103 4244 )
#endif

#ifdef ENABLE_CUDA

#include "DEM3DForceComputeGPU.h"
#include "DEM3DForceGPU.cuh"
#include "cuda_runtime.h"

#include <stdexcept>

#include <hoomd/extern/pybind/include/pybind11/pybind11.h>

using namespace std;

/*! \param sysdef System to compute forces on
  \param nlist Neighborlist to use for computing the forces
  \param r_cut Cutoff radius beyond which the force is 0
  \param potential Global potential parameters for the compute

  \post memory is allocated with empty shape vectors

  \note The DEM3DForceComputeGPU does not own the Neighborlist, the caller should
  delete the neighborlist when done.
*/
template<typename Real, typename Real4, typename Potential>
DEM3DForceComputeGPU<Real, Real4, Potential>::DEM3DForceComputeGPU(std::shared_ptr<SystemDefinition> sysdef,
    std::shared_ptr<NeighborList> nlist,
    Real r_cut, Potential potential)
    : DEM3DForceCompute<Real, Real4, Potential>(sysdef, nlist, r_cut, potential)
    {
    // can't run on the GPU if there aren't any GPUs in the execution configuration
    if (!this->m_exec_conf->isCUDAEnabled())
        {
        this->m_exec_conf->msg->error() << "Creating a DEM3DForceComputeGPU with no GPU in the execution configuration" << endl;
        throw std::runtime_error("Error initializing DEM3DForceComputeGPU");
        }

#if SINGLE_PRECISION
    int cudaVersion(0);
    cudaRuntimeGetVersion(&cudaVersion);
    if (this->m_exec_conf->dev_prop.major == 5 &&
        this->m_exec_conf->dev_prop.minor == 2 &&
        cudaVersion <= 7050)
        {
        this->m_exec_conf->msg->warning() << "3D DEM in single precision is "
            "known to exhibit a compiler bug with cuda < 8.0 on "
            "SM 5.2 cards! Undefined behavior may result." << endl;
        }
#endif

    m_tuner.reset(new Autotuner(32, 1024, 32, 5, 100000, "dem_3d", this->m_exec_conf));
    }

/*! Destructor. */
template<typename Real, typename Real4, typename Potential>
DEM3DForceComputeGPU<Real, Real4, Potential>::~DEM3DForceComputeGPU()
    {
    }

/*!  maxGPUThreads: returns the maximum number of GPU threads
  (2*vertices + edges) that will be needed among all shapes in the
  system.
*/
template<typename Real, typename Real4, typename Potential>
size_t DEM3DForceComputeGPU<Real, Real4, Potential>::maxGPUThreads() const
    {
    typedef std::vector<std::vector<unsigned int> >::const_iterator FaceIter;
    typedef std::vector<unsigned int>::const_iterator VertIter;

    size_t maxGPUThreads(0);

    for(size_t i(0); i < this->m_facesVec.size(); ++i)
        {
        std::set<std::pair<unsigned int, unsigned int> > edges;
        for(FaceIter faceIter(this->m_facesVec[i].begin());
            faceIter != this->m_facesVec[i].end(); ++faceIter)
            {
            for(VertIter vertIter(faceIter->begin());
                (vertIter + 1) != faceIter->end(); ++vertIter)
                {
                unsigned int smaller(*vertIter < *(vertIter + 1)? *vertIter: *(vertIter + 1));
                unsigned int larger(*vertIter < *(vertIter + 1)? *(vertIter + 1): *vertIter);
                std::pair<unsigned int, unsigned int> edge(smaller, larger);
                edges.insert(edge);
                }
            unsigned int smaller(faceIter->back() < faceIter->front()? faceIter->back(): faceIter->front());
            unsigned int larger(faceIter->back() < faceIter->front()? faceIter->front(): faceIter->back());
            std::pair<unsigned int, unsigned int> edge(smaller, larger);
            edges.insert(edge);
            }
        maxGPUThreads = max(maxGPUThreads, 2*this->m_shapes[i].size() + edges.size());
        }

    return maxGPUThreads;
    }

/*! \post The DEM3D forces are computed for the given timestep on the GPU.
  The neighborlist's compute method is called to ensure that it is up to date
  before forces are computed.
  \param timestep Current time step of the simulation

  Calls gpu_compute_dem3d_forces to do the dirty work.
*/
template<typename Real, typename Real4, typename Potential>
void DEM3DForceComputeGPU<Real, Real4, Potential>::computeForces(unsigned int timestep)
    {
    // start by updating the neighborlist
    this->m_nlist->compute(timestep);

    // start the profile
    if (this->m_prof) this->m_prof->push(this->m_exec_conf, "DEM3D pair");

    // The GPU implementation CANNOT handle a half neighborlist, error out now
    bool third_law = this->m_nlist->getStorageMode() == NeighborList::half;
    if (third_law)
        {
        this->m_exec_conf->msg->error() << "DEM3DForceComputeGPU cannot handle a half neighborlist" << endl;
        throw runtime_error("Error computing forces in DEM3DForceComputeGPU");
        }

    size_t threadsPerParticle(this->maxGPUThreads());
    size_t particlesPerBlock(m_tuner->getParam()/threadsPerParticle);
    // cap the block size so we don't have forces for one particle
    // spread among multiple blocks
    particlesPerBlock = min(particlesPerBlock,
        (size_t)(this->m_exec_conf->dev_prop.maxThreadsPerBlock)/threadsPerParticle);
    // don't use too many registers (~145 per thread)
    particlesPerBlock = min(particlesPerBlock,
        (size_t)(this->m_exec_conf->dev_prop.regsPerBlock/threadsPerParticle/145));
    // we better calculate for at least one particle per block
    particlesPerBlock = max(particlesPerBlock, (size_t)1);

    // access the neighbor list, which just selects the neighborlist into the device's memory, copying
    // it there if needed
    ArrayHandle<unsigned int> d_n_neigh(this->m_nlist->getNNeighArray(), access_location::device, access_mode::read);
    ArrayHandle<unsigned int> d_nlist(this->m_nlist->getNListArray(), access_location::device, access_mode::read);
    ArrayHandle<unsigned int> d_head_list(this->m_nlist->getHeadList(), access_location::device, access_mode::read);

    // access the particle data
    ArrayHandle<Real4> d_pos(this->m_pdata->getPositions(), access_location::device, access_mode::read);
    ArrayHandle<Real4> d_quat(this->m_pdata->getOrientationArray(), access_location::device, access_mode::read);
    ArrayHandle<Scalar> d_diam(this->m_pdata->getDiameters(), access_location::device, access_mode::read);
    ArrayHandle<Real4> d_velocity(this->m_pdata->getVelocities(), access_location::device, access_mode::read);

    // GPU array handles
    ArrayHandle<unsigned int> d_nextFaceVert(this->m_nextFaceVert, access_location::device, access_mode::read);
    ArrayHandle<unsigned int> d_realVertIndex(this->m_realVertIndex, access_location::device, access_mode::read);
    ArrayHandle<unsigned int> d_nextFace(this->m_nextFace, access_location::device, access_mode::read);
    ArrayHandle<unsigned int> d_firstFaceVert(this->m_firstFaceVert, access_location::device, access_mode::read);
    ArrayHandle<unsigned int> d_firstTypeVert(this->m_firstTypeVert, access_location::device,
        access_mode::read);
    ArrayHandle<unsigned int> d_numTypeVerts(this->m_numTypeVerts, access_location::device,
        access_mode::read);
    ArrayHandle<unsigned int> d_firstTypeEdge(this->m_firstTypeEdge, access_location::device,
        access_mode::read);
    ArrayHandle<unsigned int> d_numTypeEdges(this->m_numTypeEdges, access_location::device,
        access_mode::read);
    ArrayHandle<unsigned int> d_numTypeFaces(this->m_numTypeFaces, access_location::device,
        access_mode::read);
    ArrayHandle<unsigned int> d_vertexConnectivity(this->m_vertexConnectivity, access_location::device,
        access_mode::read);
    ArrayHandle<unsigned int> d_edges(this->m_edges, access_location::device,
        access_mode::read);
    ArrayHandle<Real4> d_verts(this->m_verts, access_location::device, access_mode::read);
    ArrayHandle<Real> d_faceRcutSq(this->m_faceRcutSq, access_location::device,
        access_mode::read);
    ArrayHandle<Real> d_edgeRcutSq(this->m_edgeRcutSq, access_location::device,
        access_mode::read);
    BoxDim box = this->m_pdata->getBox();

    ArrayHandle<Real4> d_force(this->m_force,access_location::device,access_mode::overwrite);
    ArrayHandle<Real4> d_torque(this->m_torque,access_location::device,access_mode::overwrite);
    ArrayHandle<Real> d_virial(this->m_virial,access_location::device,access_mode::overwrite);

    // if there is no work to do, exit early
    if (this->m_pdata->getN() == 0) return;

    m_tuner->begin();
    gpu_compute_dem3d_forces<Real, Real4, DEMEvaluator<Real, Real4, Potential> >(
        d_force.data, d_torque.data,
        d_virial.data, this->m_virial.getPitch(),
        this->m_pdata->getN(), this->m_pdata->getNGhosts(), d_pos.data, d_quat.data,
        d_nextFace.data, d_firstFaceVert.data, d_nextFaceVert.data,
        d_realVertIndex.data, d_verts.data, d_diam.data, d_velocity.data,
        this->maxGPUThreads(), this->maxVertices(),
        this->numFaces(), this->numDegenerateVerts(),
        this->numVertices(), this->numEdges(),
        this->m_pdata->getNTypes(), box, d_n_neigh.data, d_nlist.data,
        d_head_list.data, this->m_evaluator, this->m_r_cut * this->m_r_cut,
        particlesPerBlock, d_firstTypeVert.data, d_numTypeVerts.data,
        d_firstTypeEdge.data, d_numTypeEdges.data, d_numTypeFaces.data,
        d_vertexConnectivity.data, d_edges.data);

    if (this->m_exec_conf->isCUDAErrorCheckingEnabled())
        CHECK_CUDA_ERROR();
    m_tuner->end();

    Scalar avg_neigh = this->m_nlist->estimateNNeigh();
    int64_t n_calc = int64_t(avg_neigh * this->m_pdata->getN());
    int64_t mem_transfer = this->m_pdata->getN() * (4 + 16 + 20) + n_calc * (4 + 16);
    int64_t flops = n_calc * (3+12+5+2+3+11+3+8+7);
    if (this->m_prof) this->m_prof->pop(this->m_exec_conf, flops, mem_transfer);
    }

#endif

#ifdef WIN32
#pragma warning( pop )
#endif
