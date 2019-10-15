// Copyright (c) 2009-2019 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.

// Maintainer: mspells

#include "hoomd/ForceCompute.h"
#include "hoomd/md/NeighborList.h"

#include <hoomd/extern/pybind/include/pybind11/pybind11.h>
#include <memory>

#include "DEMEvaluator.h"
#include "hoomd/GSDShapeSpecWriter.h"

/*! \file DEM3DForceCompute.h
  \brief Declares the DEM3DForceCompute class
*/

#ifdef NVCC
#error This header cannot be compiled by nvcc
#endif

#ifndef __DEM3DFORCECOMPUTE_H__
#define __DEM3DFORCECOMPUTE_H__

//! Computes DEM 3D forces on each particle
/*! The total pair force is summed for each particle when compute() is called. Forces are only summed between
  neighboring particles with a separation distance less than \c r_cut. A NeighborList must be provided
  to identify these neighbors. Calling compute() in this class will in turn result in a call to the
  NeighborList's compute() to make sure that the neighbor list is up to date.

  Usage: Construct a DEM3DForceCompute, providing it an already constructed ParticleData and NeighborList.
  Then set parameters for all possible pairs of types by calling setParams.

  Forces can be computed directly by calling compute() and then retrieved with a call to acquire(), but
  a more typical usage will be to add the force compute to NVEUpdater or NVTUpdater.

  Basics of how geometry of particles is stored:
  - Each face of a shape has a unique index among all shapes
  - Each vertex has a unique "degenerate vertex" index among all shapes

  The following arrays are created to assist in iterating over geometry:
  - face->next face in the shape (circularly linked index list)
  - face->first degenerate vertex in the face
  - degenerate vertex->next degenerate vertex in the face (circularly linked index list)
  - degenerate vertex->real vertex index
  - type index->first real vertex of type index
  - type index->number of vertices
  - type index->first edge of type index
  - type index->number of edges in type
  - (2*edge index)->first real vertex index in edge, (2*edge index + 1)->second real vertex in edge
  - real vertex index->vertex (3D point)

  Implementation details:
  - The first face of the type with type index i is stored at index i within the face->next face array
  - Faces in a shape and vertices in a face use a circularly linked index structure
  - Vertices (3D points) are stored consecutively for a shape
  - Edges (pairs of vertex indices) are stored consecutively for a shape

  \ingroup computes
*/
template<typename Real, typename Real4, typename Potential>
class DEM3DForceCompute : public ForceCompute
    {
    public:
        //! Constructs the compute
        DEM3DForceCompute(std::shared_ptr<SystemDefinition> sysdef,
            std::shared_ptr<NeighborList> nlist,
            Real r_cut, Potential potential);

        //! Destructor
        virtual ~DEM3DForceCompute();

        //! Set the vertices for a particle
        virtual void setParams(unsigned int type,
            const pybind11::list &pyVertices,
            const pybind11::list &pyFaces);

        virtual void setRcut(Real r_cut) {m_r_cut = r_cut;}

        //! Returns a list of log quantities this compute calculates
        virtual std::vector< std::string > getProvidedLogQuantities();

        //! Calculates the requested log value and returns it
        virtual Real getLogValue(const std::string& quantity, unsigned int timestep);

        //! Find the total number of vertices in the current set of shapes
        size_t numVertices() const;

        //! Find the maximum number of vertices among all shapes
        size_t maxVertices() const;

        //! Find the maximum number of vertices in the current set of shapes
        size_t numFaces() const;

        //! Find the number of unique edges in the shape
        size_t numEdges() const;

        //! Find the total number of degenerate vertices for all shapes
        size_t numDegenerateVerts() const;

        void connectDEMGSDShapeSpec(std::shared_ptr<GSDDumpWriter> writer);

        int slotWriteDEMGSDShapeSpec(gsd_handle& handle) const;

        std::string getTypeShape(const std::vector<vec3<Real>> &verts, const Real &radius) const;

        std::string encodeVertices(const std::vector<vec3<Real>> &verts) const;

        std::vector<std::string> getTypeShapeMapping(const std::vector<std::vector<vec3<Real>>> &verts, const Real &radius) const;

        pybind11::list getTypeShapesPy();

    #ifdef ENABLE_MPI
        //! Get requested ghost communication flags
        virtual CommFlags getRequestedCommFlags(unsigned int timestep)
            {
            // by default, only request positions
            CommFlags flags(0);
            flags[comm_flag::orientation] = 1;
            flags[comm_flag::net_torque] = 1; // only used with rigid bodies

            flags |= ForceCompute::getRequestedCommFlags(timestep);
            return flags;
            }
    #endif

        //! Returns true because we compute the torque
        virtual bool isAnisotropic()
            {
            return true;
            }

    protected:
        std::shared_ptr<NeighborList> m_nlist;    //!< The neighborlist to use for the computation
        Real m_r_cut;         //!< Cutoff radius beyond which the force is set to 0
        DEMEvaluator<Real, Real4, Potential> m_evaluator; //!< Object holding parameters and computation method for the potential
        GPUArray<unsigned int> m_nextFace; //! face->next face
        GPUArray<unsigned int> m_firstFaceVert; //!< face->first vertex
        GPUArray<unsigned int> m_nextFaceVert; //!< vertex->next vertex in the given face
        GPUArray<unsigned int> m_realVertIndex; //!< vertex->real vertex
        GPUArray<unsigned int> m_firstTypeVert; //!< type->first real vertex index
        GPUArray<unsigned int> m_numTypeVerts; //!< type->number of vertices
        GPUArray<unsigned int> m_firstTypeEdge; //!< type->first edge in pair
        GPUArray<unsigned int> m_numTypeEdges; //!< type->number of edges
        GPUArray<unsigned int> m_numTypeFaces; //!< type->number of faces
        GPUArray<unsigned int> m_vertexConnectivity; //!< real vertex index->number of times it appears in an edge
        GPUArray<unsigned int> m_edges; //!< 2*edge->first real vert, 2*edge+1->second real vert in edge
        GPUArray<Real> m_faceRcutSq; //!< face index->rcut*rcut
        GPUArray<Real> m_edgeRcutSq; //!< edge index->rcut*rcut
        GPUArray<Real4> m_verts; //! Vertices for each real index
        std::vector<std::vector<vec3<Real> > > m_shapes; //!< Vertices for each type
        std::vector<std::vector<std::vector<unsigned int> > > m_facesVec; //!< Faces for each type

        //! Re-send the list of vertices and links to the GPU
        void createGeometry();

        //! Actually compute the forces
        virtual void computeForces(unsigned int timestep);
    };

#include "DEM3DForceCompute.cc"

#endif
