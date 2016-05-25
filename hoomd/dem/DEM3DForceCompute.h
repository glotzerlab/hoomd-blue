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

// Maintainer: mspells

#include "hoomd/ForceCompute.h"
#include "hoomd/md/NeighborList.h"

#include <boost/python.hpp>
#include <boost/shared_ptr.hpp>

#include "DEMEvaluator.h"

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
        DEM3DForceCompute(boost::shared_ptr<SystemDefinition> sysdef,
                          boost::shared_ptr<NeighborList> nlist,
                          Real r_cut, Potential potential);

        //! Destructor
        virtual ~DEM3DForceCompute();

        //! Set the vertices for a particle
        virtual void setParams(unsigned int type,
                               const boost::python::list &pyVertices,
                               const boost::python::list &pyFaces);

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

        #ifdef ENABLE_MPI
        //! Get requested ghost communication flags
        virtual CommFlags getRequestedCommFlags(unsigned int timestep)
        {
            // by default, only request positions
            CommFlags flags(0);
            flags[comm_flag::orientation] = 1;

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
        boost::shared_ptr<NeighborList> m_nlist;    //!< The neighborlist to use for the computation
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
        std::vector<std::vector<vec3<Real> > > m_vertsVec; //!< Vertices for each type
        std::vector<std::vector<std::vector<unsigned int> > > m_facesVec; //!< Faces for each type

        //! Re-send the list of vertices and links to the GPU
        void createGeometry();

        //! Actually compute the forces
        virtual void computeForces(unsigned int timestep);
    };

#include "DEM3DForceCompute.cc"

#endif
