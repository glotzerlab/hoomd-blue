// Copyright (c) 2009-2019 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.

// Maintainer(s): mspells, rmarson

#ifdef WIN32
#pragma warning( push )
#pragma warning( disable : 4103 4244 )
#endif

#include "DEM3DForceCompute.h"

#include <hoomd/extern/pybind/include/pybind11/pybind11.h>


#include <stdexcept>
#include <utility>
#include <set>

#ifdef ENABLE_OPENMP
#include <omp.h>
#endif

/*! \file DEM3DForceCompute.cc
  \brief Defines the DEM3DForceCompute class
*/

using namespace std;

/*! \param sysdef System to compute forces on
  \param nlist Neighborlist to use for computing the forces
  \param r_cut Cutoff radius beyond which the force is 0
  \param potential Global potential parameters for the interaction
  \post memory is allocated.
*/
template<typename Real, typename Real4, typename Potential>
DEM3DForceCompute<Real, Real4, Potential>::DEM3DForceCompute(
    std::shared_ptr<SystemDefinition> sysdef,
    std::shared_ptr<NeighborList> nlist,
    Real r_cut, Potential potential)
    : ForceCompute(sysdef), m_nlist(nlist), m_r_cut(r_cut),
      m_evaluator(potential), m_nextFace(0, this->m_exec_conf),
      m_firstFaceVert(0, this->m_exec_conf), m_nextFaceVert(0, this->m_exec_conf),
      m_realVertIndex(0, this->m_exec_conf), m_firstTypeVert(0, this->m_exec_conf),
      m_numTypeVerts(0, this->m_exec_conf), m_firstTypeEdge(0, this->m_exec_conf),
      m_numTypeEdges(0, this->m_exec_conf), m_numTypeFaces(0, this->m_exec_conf),
      m_vertexConnectivity(0, this->m_exec_conf), m_edges(0, this->m_exec_conf),
      m_faceRcutSq(0, this->m_exec_conf), m_edgeRcutSq(0, this->m_exec_conf),
      m_verts(0, this->m_exec_conf), m_shapes(), m_facesVec()
    {
    m_exec_conf->msg->notice(5) << "Constructing DEM3DForceCompute" << endl;

    assert(m_pdata);
    assert(m_nlist);

    if (r_cut < 0.0)
        {
        m_exec_conf->msg->error() << "dem: Negative r_cut makes no sense" << endl;
        throw runtime_error("Error initializing DEM3DForceCompute");
        }
    }

template<typename Real, typename Real4, typename Potential>
void DEM3DForceCompute<Real, Real4, Potential>::connectDEMGSDShapeSpec(std::shared_ptr<GSDDumpWriter> writer)
    {
    typedef hoomd::detail::SharedSignalSlot<int(gsd_handle&)> SlotType;
    auto func = std::bind(&DEM3DForceCompute<Real, Real4, Potential>::slotWriteDEMGSDShapeSpec, this, std::placeholders::_1);
    std::shared_ptr<hoomd::detail::SignalSlot> pslot(new SlotType(writer->getWriteSignal(), func));
    addSlot(pslot);
    }

template<typename Real, typename Real4, typename Potential>
int DEM3DForceCompute<Real, Real4, Potential>::slotWriteDEMGSDShapeSpec(gsd_handle& handle) const
    {
    GSDShapeSpecWriter shapespec(m_exec_conf);
    m_exec_conf->msg->notice(10) << "DEM3DForceCompute writing particle shape information to GSD file in chunk: " << shapespec.getName() << std::endl;
    int retval = shapespec.write(handle, this->getTypeShapeMapping(m_shapes, m_evaluator.getRadius()));
    return retval;
    }

template<typename Real, typename Real4, typename Potential>
std::string DEM3DForceCompute<Real, Real4, Potential>::getTypeShape(const std::vector<vec3<Real>> &verts, const Real &radius) const
    {
    std::ostringstream shapedef;
    unsigned int nverts = verts.size();
    if (nverts == 1)
        {
        shapedef << "{\"type\": \"Sphere\", " << "\"diameter\": " << Real(2)*radius << "}";
        }
    else if (nverts == 2)
        {
        throw std::runtime_error("Shape definition not supported for 2-vertex polyhedra");
        }
    else
        {
        shapedef <<  "{\"type\": \"ConvexPolyhedron\", " << "\"rounding_radius\": " << radius <<
                    ", \"vertices\": "  << encodeVertices(verts) << "}";
        }
    return shapedef.str();
    }

template<typename Real, typename Real4, typename Potential>
std::string DEM3DForceCompute<Real, Real4, Potential>::encodeVertices(const std::vector<vec3<Real>> &verts) const
    {
    std::ostringstream vertstr;
    unsigned int nverts = verts.size();
    vertstr << "[";
    for (unsigned int i = 0; i < nverts-1; i++)
        {
        vertstr << "[" << verts[i].x << ", " << verts[i].y << ", " << verts[i].z << "], ";
        }
    vertstr << "[" << verts[nverts-1].x << ", " << verts[nverts-1].y << ", " << verts[nverts-1].z  << "]" << "]";
    return vertstr.str();
    }

template<typename Real, typename Real4, typename Potential>
std::vector<std::string> DEM3DForceCompute<Real, Real4, Potential>::getTypeShapeMapping(const std::vector<std::vector<vec3<Real>>> &verts, const Real &radius) const
    {
    std::vector<std::string> type_shape_mapping(verts.size());
    for (unsigned int i = 0; i < type_shape_mapping.size(); i++)
        {
        type_shape_mapping[i] = this->getTypeShape(verts[i], radius);
        }
    return type_shape_mapping;
    }

template<typename Real, typename Real4, typename Potential>
pybind11::list DEM3DForceCompute<Real, Real4, Potential>::getTypeShapesPy()
    {
    std::vector<std::string> type_shape_mapping = this->getTypeShapeMapping(m_shapes, m_evaluator.getRadius());
    pybind11::list type_shapes;
    for (unsigned int i = 0; i < type_shape_mapping.size(); i++)
        type_shapes.append(type_shape_mapping[i]);
    return type_shapes;
    }

/*! Destructor. */
template<typename Real, typename Real4, typename Potential>
DEM3DForceCompute<Real, Real4, Potential>::~DEM3DForceCompute()
    {
    m_exec_conf->msg->notice(5) << "Destroying DEM3DForceCompute" << endl;
    }

/*! setParams: set the vertices for a numeric particle type from a python list.
  \param type Particle type index
  \param pyVertices Python list of 3D vertices specifying a polygon
  \param pyFaces Python list of lists of vertex indices, one for each faces
*/
template<typename Real, typename Real4, typename Potential>
void DEM3DForceCompute<Real, Real4, Potential>::setParams(
    unsigned int type, const pybind11::list &pyVertices, const pybind11::list &pyFaces)
    {
    if (type >= m_pdata->getNTypes())
        {
        m_exec_conf->msg->error() <<
            "dem: Trying to set params for a non existent type! " << type << endl;
        throw runtime_error("Error setting parameters in DEM3DForceCompute");
        }

    for(int i(type - m_shapes.size()); i >= 0; --i)
        {
        m_shapes.push_back(vector<vec3<Real> >(0));
        m_facesVec.push_back(vector<vector<unsigned int> >(0));
        }

    // build a vector of points
    vector<vec3<Real> > points;

    for(size_t i(0); i < (size_t) pybind11::len(pyVertices); ++i)
        {
        const pybind11::tuple pyPoint = pybind11::cast<pybind11::tuple>(pyVertices[i]);

        if(pybind11::len(pyPoint) != 3)
            throw runtime_error("Non-3D vertex given for DEM3DForceCompute::setParams");

        const Real x = pybind11::cast<Real>(pyPoint[0]);
        const Real y = pybind11::cast<Real>(pyPoint[1]);
        const Real z = pybind11::cast<Real>(pyPoint[2]);
        const vec3<Real> point(x, y, z);
        points.push_back(point);
        }

    //build a vector of vectors of unsigned ints for each face
    vector<vector<unsigned int> > faces;
    vector<vector<unsigned int> > NxtFaces;

    for(size_t i(0); i < (size_t) pybind11::len(pyFaces); ++i)
        {
        vector<unsigned int> face;
        vector<unsigned int> NxtFace;
        pybind11::list pyFaces_i = pybind11::cast<pybind11::list>(pyFaces[i]);

        for(size_t j(0); j + 1 < (size_t) pybind11::len(pyFaces_i); ++j)
            {
            face.push_back(pybind11::cast<unsigned int>(pyFaces_i[j]));
            }

        face.push_back(pybind11::cast<unsigned int>(pyFaces_i[pybind11::len(pyFaces[i]) - 1]));

        faces.push_back(face);
        }

    m_shapes[type] = points;
    m_facesVec[type] = faces;

    createGeometry();
    }

/*!
  createGeometry: Update the device-side list of vertices and vertex indices.
*/
template<typename Real, typename Real4, typename Potential>
void DEM3DForceCompute<Real, Real4, Potential>::createGeometry()
    {
    const size_t nVerts(numVertices());
    const size_t nDegVerts(numDegenerateVerts());
    const size_t nFaces(numFaces());
    const size_t nEdges(numEdges());
    const size_t nTypes(m_pdata->getNTypes());

    if(m_facesVec.size() != nTypes || m_shapes.size() != nTypes)
        return;

    // resize the geometry arrays if necessary
    if(m_verts.getNumElements() != nVerts)
        m_verts.resize(nVerts);

    if(m_nextFaceVert.getNumElements() != nDegVerts)
        m_nextFaceVert.resize(nDegVerts);

    if(m_realVertIndex.getNumElements() != nDegVerts)
        m_realVertIndex.resize(nDegVerts);

    if(m_nextFace.getNumElements() != nFaces)
        m_nextFace.resize(nFaces);

    if(m_firstFaceVert.getNumElements() != nFaces)
        m_firstFaceVert.resize(nFaces);

    if(m_faceRcutSq.getNumElements() != nFaces)
        m_faceRcutSq.resize(nFaces);

    if(m_firstTypeVert.getNumElements() != nTypes)
        m_firstTypeVert.resize(nTypes);

    if(m_numTypeVerts.getNumElements() != nTypes)
        m_numTypeVerts.resize(nTypes);

    if(m_firstTypeEdge.getNumElements() != nTypes)
        m_firstTypeEdge.resize(nTypes);

    if(m_numTypeEdges.getNumElements() != nTypes)
        m_numTypeEdges.resize(nTypes);

    if(m_numTypeFaces.getNumElements() != nTypes)
        m_numTypeFaces.resize(nTypes);

    if(m_vertexConnectivity.getNumElements() != nVerts)
        m_vertexConnectivity.resize(nVerts);

    if(m_edgeRcutSq.getNumElements() != nEdges)
        m_edgeRcutSq.resize(nEdges);

    if(m_edges.getNumElements() != 2*nEdges)
        m_edges.resize(2*nEdges);

    ArrayHandle<Real4> h_verts(m_verts, access_location::host,
        access_mode::overwrite);
    ArrayHandle<Real> h_faceRcutSq(m_faceRcutSq, access_location::host,
        access_mode::overwrite);
    ArrayHandle<Real> h_edgeRcutSq(m_edgeRcutSq, access_location::host,
        access_mode::overwrite);
    ArrayHandle<unsigned int> h_nextFaceVert(m_nextFaceVert, access_location::host,
        access_mode::overwrite);
    ArrayHandle<unsigned int> h_realVertIndex(m_realVertIndex, access_location::host,
        access_mode::overwrite);
    ArrayHandle<unsigned int> h_nextFace(m_nextFace, access_location::host,
        access_mode::overwrite);
    ArrayHandle<unsigned int> h_firstFaceVert(m_firstFaceVert, access_location::host,
        access_mode::overwrite);
    ArrayHandle<unsigned int> h_firstTypeVert(m_firstTypeVert, access_location::host,
        access_mode::overwrite);
    ArrayHandle<unsigned int> h_numTypeVerts(m_numTypeVerts, access_location::host,
        access_mode::overwrite);
    ArrayHandle<unsigned int> h_firstTypeEdge(m_firstTypeEdge, access_location::host,
        access_mode::overwrite);
    ArrayHandle<unsigned int> h_numTypeEdges(m_numTypeEdges, access_location::host,
        access_mode::overwrite);
    ArrayHandle<unsigned int> h_numTypeFaces(m_numTypeFaces, access_location::host,
        access_mode::overwrite);
    ArrayHandle<unsigned int> h_vertexConnectivity(m_vertexConnectivity, access_location::host,
        access_mode::overwrite);
    ArrayHandle<unsigned int> h_edges(m_edges, access_location::host,
        access_mode::overwrite);

    // iterate over shapes to build GPU Arrays m_verts,
    // m_firstTypeVert, and m_numTypeVerts
    for(size_t i(0), j(0); i < m_shapes.size(); ++i)
        {
        h_firstTypeVert.data[i] = j;
        h_numTypeVerts.data[i] = m_shapes[i].size();

        for(size_t k(0); k < m_shapes[i].size(); ++j, ++k)
            {
            const vec3<Real> point(m_shapes[i][k]);
            h_verts.data[j] = vec_to_scalar4(point, 0);
            }
        }

    // build m_nextFace
    for(size_t shapeIdx(0), faceCount(0);
        shapeIdx < m_facesVec.size(); ++shapeIdx)
        {
        if(m_facesVec[shapeIdx].size() > 1)
            {
            h_nextFace.data[shapeIdx] = faceCount + nTypes;

            // We already account for two faces above and below here,
            // so only do N-2 iterations in this loop; basically, just
            // iterate to the end-1 instead of the end
            for(size_t faceIdx(2); faceIdx < m_facesVec[shapeIdx].size(); ++faceIdx, ++faceCount)
                h_nextFace.data[faceCount + nTypes] = faceCount + nTypes + 1;

            // close the loop
            h_nextFace.data[faceCount + nTypes] = shapeIdx;
            ++faceCount;
            }
        else
            h_nextFace.data[shapeIdx] = shapeIdx;
        }

    // build m_firstFaceVert
    for(size_t shapeIdx(0), vertCount(0);
        shapeIdx < m_facesVec.size(); ++shapeIdx)
        {
        for(size_t faceIdx(shapeIdx), vecIdx(0);
            vecIdx < m_facesVec[shapeIdx].size();
            faceIdx = h_nextFace.data[faceIdx], ++vecIdx)
            {
            h_firstFaceVert.data[faceIdx] = vertCount;
            vertCount += m_facesVec[shapeIdx][vecIdx].size();
            }
        }

    // build m_nextFaceVert
    for(size_t shapeIdx(0), vertCount(0);
        shapeIdx < m_facesVec.size(); ++shapeIdx)
        {
        for(size_t faceIdx(0); faceIdx < m_facesVec[shapeIdx].size(); ++faceIdx)
            {
            // first vertex in the shape, which we'll loop around to
            const size_t firstVert(vertCount);

            for(size_t vertIdx(1); vertIdx < m_facesVec[shapeIdx][faceIdx].size();
                ++vertIdx, ++vertCount)
                h_nextFaceVert.data[vertCount] = vertCount + 1;

            // close the loop
            h_nextFaceVert.data[vertCount] = firstVert;
            ++vertCount;
            }
        }

    // build m_realVertIndex
    for(size_t shapeIdx(0), vertCount(0), vertTypeOffset(0);
        shapeIdx < m_facesVec.size(); ++shapeIdx)
        {
        for(size_t faceIdx(0); faceIdx < m_facesVec[shapeIdx].size(); ++faceIdx)
            {
            for(size_t vertIdx(0); vertIdx < m_facesVec[shapeIdx][faceIdx].size();
                ++vertIdx, ++vertCount)
                h_realVertIndex.data[vertCount] = vertTypeOffset + m_facesVec[shapeIdx][faceIdx][vertIdx];
            }
        vertTypeOffset += m_shapes[shapeIdx].size();
        }

    // build m_firstTypeEdge, m_numTypeEdges, m_edges, and m_vertexConnectivity
    memset(h_vertexConnectivity.data, 0, sizeof(unsigned int)*nVerts);
    for(size_t shapeIdx(0), edgeCount(0), vertTypeOffset(0);
        shapeIdx < m_facesVec.size(); ++shapeIdx)
        {
        typedef std::pair<size_t, size_t> edge;
        std::set<edge> edges;

        // build the edges found in this shape
        for(size_t faceIdx(0); faceIdx < m_facesVec[shapeIdx].size(); ++faceIdx)
            {
            for(size_t vertIdx(0); vertIdx + 1 < m_facesVec[shapeIdx][faceIdx].size(); ++vertIdx)
                {
                size_t first(vertTypeOffset + m_facesVec[shapeIdx][faceIdx][vertIdx]);
                size_t second(vertTypeOffset + m_facesVec[shapeIdx][faceIdx][vertIdx + 1]);

                if(second < first)
                    std::swap(first, second);
                if(first != second)
                    edges.insert(edge(first, second));
                }

            size_t first(vertTypeOffset + m_facesVec[shapeIdx][faceIdx].back());
            size_t second(vertTypeOffset + m_facesVec[shapeIdx][faceIdx].front());

            if(second < first)
                std::swap(first, second);
            if(first != second)
                edges.insert(edge(first, second));
            }
        vertTypeOffset += m_shapes[shapeIdx].size();

        // fill the GPUArrays
        h_firstTypeEdge.data[shapeIdx] = edgeCount;

        for(std::set<edge>::const_iterator edgeIter(edges.begin());
            edgeIter != edges.end(); ++edgeIter, ++edgeCount)
            {
            h_edges.data[2*edgeCount] = edgeIter->first;
            h_edges.data[2*edgeCount + 1] = edgeIter->second;

            ++h_vertexConnectivity.data[edgeIter->first];
            ++h_vertexConnectivity.data[edgeIter->second];
            }

        h_numTypeEdges.data[shapeIdx] = edges.size();
        }

    // build m_numTypeFaces
    for(size_t shapeIdx(0); shapeIdx < m_facesVec.size(); ++shapeIdx)
        {
        const unsigned int faceSize(m_facesVec[shapeIdx].size());
        h_numTypeFaces.data[shapeIdx] = faceSize;
        }
    }

/*!
  numVertices: Returns the total number of vertices for all shapes
  in the system.
*/
template<typename Real, typename Real4, typename Potential>
size_t DEM3DForceCompute<Real, Real4, Potential>::numVertices() const
    {
    size_t result(0);

    for(typename std::vector<std::vector<vec3<Real> > >::const_iterator shapeIter(this->m_shapes.begin());
        shapeIter != this->m_shapes.end(); ++shapeIter)
        result += shapeIter->size() ? shapeIter->size(): 1;

    return result;
    }

/*!
  maxVertices: Returns the maximum number of vertices among all shapes
  in the system.
*/
template<typename Real, typename Real4, typename Potential>
size_t DEM3DForceCompute<Real, Real4, Potential>::maxVertices() const
    {
    size_t result(1);

    for(typename std::vector<std::vector<vec3<Real> > >::const_iterator shapeIter(this->m_shapes.begin());
        shapeIter != this->m_shapes.end(); ++shapeIter)
        result = max(result, shapeIter->size());

    return result;
    }

/*!
  numFaces: returns the number of faces among all shapes
  in the system.
*/
template<typename Real, typename Real4, typename Potential>
size_t DEM3DForceCompute<Real, Real4, Potential>::numFaces() const
    {
    size_t result(0);

    for(typename std::vector<std::vector<std::vector<unsigned int> > >::const_iterator shapeIter(this->m_facesVec.begin());
        shapeIter != this->m_facesVec.end(); ++shapeIter)
        result += shapeIter->size() ? shapeIter->size(): 1;

    return result;
    }

/*!
  numTypeEdges: returns the number of edges among all shapes
  in the system.
*/
template<typename Real, typename Real4, typename Potential>
size_t DEM3DForceCompute<Real, Real4, Potential>::numEdges() const
    {
    typedef std::vector<std::vector<std::vector<unsigned int> > >::const_iterator ShapeIter;
    typedef std::vector<std::vector<unsigned int> >::const_iterator FaceIter;
    typedef std::vector<unsigned int>::const_iterator VertIter;

    size_t result(0);

    for(ShapeIter shapeIter(this->m_facesVec.begin());
        shapeIter != this->m_facesVec.end(); ++shapeIter)
        {
        std::set<std::pair<unsigned int, unsigned int> > edges;
        for(FaceIter faceIter(shapeIter->begin());
            faceIter != shapeIter->end(); ++faceIter)
            {
            for(VertIter vertIter(faceIter->begin());
                (vertIter + 1) != faceIter->end(); ++vertIter)
                {
                unsigned int smaller(*vertIter < *(vertIter + 1)? *vertIter: *(vertIter + 1));
                unsigned int larger(*vertIter < *(vertIter + 1)? *(vertIter + 1): *vertIter);
                std::pair<unsigned int, unsigned int> edge(smaller, larger);
                if(smaller != larger)
                    edges.insert(edge);
                }
            unsigned int smaller(faceIter->back() < faceIter->front()? faceIter->back(): faceIter->front());
            unsigned int larger(faceIter->back() < faceIter->front()? faceIter->front(): faceIter->back());
            std::pair<unsigned int, unsigned int> edge(smaller, larger);
            if(smaller != larger)
                edges.insert(edge);
            }
        result += edges.size();
        }

    return result;
    }

/*!
  numDegenerateVerts: Returns the total number of degenerate vertices for all shapes
  in the system.
*/
template<typename Real, typename Real4, typename Potential>
size_t DEM3DForceCompute<Real, Real4, Potential>::numDegenerateVerts() const
    {
    size_t result(0);

    for(typename std::vector<std::vector<std::vector<unsigned int> > >::const_iterator shapeIter(this->m_facesVec.begin());
        shapeIter != this->m_facesVec.end(); ++shapeIter)
        {
        for(typename std::vector<std::vector<unsigned int> >::const_iterator faceIter(shapeIter->begin());
            faceIter != shapeIter->end(); ++faceIter)
            result += faceIter->size() ? faceIter->size(): 1;
        }

    return result;
    }

/*! DEM3DForceCompute provides
  - \c pair_dem_energy
*/
template<typename Real, typename Real4, typename Potential>
std::vector< std::string > DEM3DForceCompute<Real, Real4, Potential>::getProvidedLogQuantities()
    {
    vector<string> list;
    list.push_back("pair_dem_energy");
    return list;
    }

template<typename Real, typename Real4, typename Potential>
Real DEM3DForceCompute<Real, Real4, Potential>::getLogValue(const std::string& quantity, unsigned int timestep)
    {
    if (quantity == string("pair_dem_energy"))
        {
        compute(timestep);
        return calcEnergySum();
        }
    else
        {
        m_exec_conf->msg->error() << "dem: " << quantity << " is not a valid log quantity" << endl;
        throw runtime_error("Error getting log value");
        }
    }

/*! \post The DEM3D forces are computed for the given timestep. The neighborlist's
  compute method is called to ensure that it is up to date.

  \param timestep specifies the current time step of the simulation
*/
template<typename Real, typename Real4, typename Potential>
void DEM3DForceCompute<Real, Real4, Potential>::computeForces(unsigned int timestep)
    {
    // start by updating the neighborlist
    m_nlist->compute(timestep);

    // start the profile for this compute
    if (m_prof) m_prof->push("DEM3D pair");

    // grab handles for particle data
    ArrayHandle<Scalar4> h_force(m_force,access_location::host, access_mode::overwrite);
    ArrayHandle<Scalar4> h_torque(m_torque,access_location::host, access_mode::overwrite);
    ArrayHandle<Scalar> h_virial(m_virial,access_location::host, access_mode::overwrite);
    const unsigned int virial_pitch = m_virial.getPitch();

    // there are enough other checks on the input data: but it doesn't hurt to be safe
    assert(h_force.data);
    assert(h_torque.data);
    assert(h_virial.data);

    // Zero data for force calculation.
    memset((void*)h_force.data,0,sizeof(Scalar4)*m_force.getNumElements());
    memset((void*)h_torque.data,0,sizeof(Scalar4)*m_torque.getNumElements());
    memset((void*)h_virial.data,0,sizeof(Scalar)*m_virial.getNumElements());

    // depending on the neighborlist settings, we can take advantage of newton's third law
    // to reduce computations at the cost of memory access complexity: set that flag now
    bool third_law = m_nlist->getStorageMode() == NeighborList::half;

    // access the neighbor list
    ArrayHandle<unsigned int> h_n_neigh(m_nlist->getNNeighArray(), access_location::host, access_mode::read);
    ArrayHandle<unsigned int> h_nlist(m_nlist->getNListArray(), access_location::host, access_mode::read);
    ArrayHandle<unsigned int> h_head_list(m_nlist->getHeadList(), access_location::host, access_mode::read);

    // access the particle data
    ArrayHandle< Scalar4 > h_pos(m_pdata->getPositions(), access_location::host, access_mode::read);
    ArrayHandle<Scalar4> h_velocity(m_pdata->getVelocities(), access_location::host, access_mode::read);
    ArrayHandle<Scalar4> h_orientation(m_pdata->getOrientationArray(), access_location::host,access_mode::read);
    ArrayHandle<Scalar> h_diameter(m_pdata->getDiameters(), access_location::host,access_mode::read);
    // sanity check
    assert(h_pos.data != NULL);

    // GPU array handles
    ArrayHandle<Real4> h_verts(m_verts, access_location::host,
        access_mode::read);
    ArrayHandle<Real> h_faceRcutSq(m_faceRcutSq, access_location::host,
        access_mode::read);
    ArrayHandle<Real> h_edgeRcutSq(m_edgeRcutSq, access_location::host,
        access_mode::read);
    ArrayHandle<unsigned int> h_nextFaceVert(m_nextFaceVert, access_location::host,
        access_mode::read);
    ArrayHandle<unsigned int> h_realVertIndex(m_realVertIndex, access_location::host,
        access_mode::read);
    ArrayHandle<unsigned int> h_nextFace(m_nextFace, access_location::host,
        access_mode::read);
    ArrayHandle<unsigned int> h_firstFaceVert(m_firstFaceVert, access_location::host,
        access_mode::read);
    ArrayHandle<unsigned int> h_firstTypeVert(m_firstTypeVert, access_location::host,
        access_mode::read);
    ArrayHandle<unsigned int> h_numTypeVerts(m_numTypeVerts, access_location::host,
        access_mode::read);
    ArrayHandle<unsigned int> h_firstTypeEdge(m_firstTypeEdge, access_location::host,
        access_mode::read);
    ArrayHandle<unsigned int> h_numTypeEdges(m_numTypeEdges, access_location::host,
        access_mode::read);
    ArrayHandle<unsigned int> h_numTypeFaces(m_numTypeFaces, access_location::host,
        access_mode::read);
    ArrayHandle<unsigned int> h_vertexConnectivity(m_vertexConnectivity, access_location::host,
        access_mode::read);
    ArrayHandle<unsigned int> h_edges(m_edges, access_location::host,
        access_mode::read);


    // get a local copy of the simulation box too
    const BoxDim& box = m_pdata->getBox();

    // create a temporary copy of r_cut squared
    Scalar r_cut_sq = m_r_cut * m_r_cut;

    // tally up the number of forces calculated
    int64_t n_calc = 0;

    // for each particle
    for (unsigned int i = 0; i < m_pdata->getN(); i++)
        {
        // access the particle's position and type (MEM TRANSFER: 4 scalars)
        vec3<Scalar> pi(h_pos.data[i].x, h_pos.data[i].y, h_pos.data[i].z);
        quat<Scalar> quati(h_orientation.data[i]);
        unsigned int typei = __scalar_as_int(h_pos.data[i].w);
        // sanity check
        assert(typei < m_pdata->getNTypes());

        // initialize current particle force, potential energy, and virial to 0
        vec3<Real> fi;
        vec3<Real> ti;
        Real pei(0);
        Real viriali[6];
        for (int k = 0; k < 6; k++)
            viriali[k] = 0.0;

        // If the evaluator needs the diameters of the particles to evaluate, grab particle_i's here
        // MEM TRANSFER (1 scalar)
        Scalar di;
        if (Potential::needsDiameter())
            {
            di = h_diameter.data[i];
            }

        vec3<Scalar> vi;
        if(Potential::needsVelocity())
            vi = vec3<Scalar>(h_velocity.data[i]);

        // loop over all of the neighbors of this particle
        const unsigned int myHead = h_head_list.data[i];
        const unsigned int size = (unsigned int)h_n_neigh.data[i];
        for (unsigned int j = 0; j < size; j++)
            {
            // increment our calculation counter
            n_calc++;

            // access the index of this neighbor (MEM TRANSFER: 1 scalar)
            unsigned int k = h_nlist.data[myHead + j];
            // sanity check
            assert(k < m_pdata->getN() + m_pdata->getNGhosts());

            // calculate dr (MEM TRANSFER: 3 scalars / FLOPS: 3)
            vec3<Scalar> pj(h_pos.data[k].x, h_pos.data[k].y, h_pos.data[k].z);
            quat<Scalar> quatj(h_orientation.data[k]);
            vec3<Scalar> dxScalar(pj - pi);

            // access the type of the neighbor particle (MEM TRANSFER: 1 scalar
            unsigned int typej = __scalar_as_int(h_pos.data[k].w);
            // sanity check
            assert(typej < m_pdata->getNTypes());

            // apply periodic boundary conditions (FLOPS: 9 (worst case: first branch is missed, the 2nd is taken and the add is done)
            dxScalar = vec3<Scalar>(box.minImage(vec_to_scalar3(dxScalar)));
            const vec3<Real> dx(dxScalar);

            // If the evaluator needs the diameters of the particles to evaluate, grab particle_j's and
            // pass in the diameters of the particles here
            // MEM TRANSFER (1 scalar)
            Scalar dj;
            if (Potential::needsDiameter())
                {
                dj = h_diameter.data[k];
                m_evaluator.setDiameter(di,dj);
                }

            if(Potential::needsVelocity())
                m_evaluator.setVelocity(vi - vec3<Scalar>(h_velocity.data[k]));

            // start computing the force
            // calculate r squared (FLOPS: 5)
            Real rsq = dot(dx, dx);

            // only compute the force if the particles are closer than the cutoff (FLOPS: 1)
            if (m_evaluator.withinCutoff(rsq,r_cut_sq))
                {
                // local forces and torques for particles i and j
                vec3<Real> forceij, forceji;
                vec3<Real> torqueij, torqueji;
                Real potentialij(0);

                // iterate over each vertex in particle i
                for(size_t vertIndex(0); vertIndex < h_numTypeVerts.data[typei]; ++vertIndex)
                    {
                    const vec3<Real> vertex0(
                        rotate(quati, vec3<Real>(h_verts.data[h_firstTypeVert.data[typei] + vertIndex])));

                    // iterate over each face in particle j
                    size_t faceIndex(typej);
                    if(h_numTypeFaces.data[typej] > 0)
                        {
                        do
                            {
                            m_evaluator.vertexFace(dx, vertex0, quatj,
                                h_verts.data,
                                h_realVertIndex.data,
                                h_nextFaceVert.data,
                                h_firstFaceVert.data[faceIndex],
                                potentialij,
                                forceij, torqueij,
                                forceji, torqueji);
                            faceIndex = h_nextFace.data[faceIndex];
                            }
                        while(faceIndex != typej);
                        }
                    // no faces; is it a spherocylinder?
                    else if(h_numTypeEdges.data[typej] > 0)
                        {
                        // iterate over all edges of j
                        for(size_t edgej(0); edgej < h_numTypeEdges.data[typej]; ++edgej)
                            {
                            vec3<Real> p10(h_verts.data[h_edges.data[2*(edgej + h_firstTypeEdge.data[typej])]]);
                            vec3<Real> p11(h_verts.data[h_edges.data[2*(edgej + h_firstTypeEdge.data[typej]) + 1]]);
                            p10 = rotate(quatj, p10);
                            p11 = rotate(quatj, p11);

                            m_evaluator.vertexEdge(dx, vertex0, p10, p11,
                                potentialij, forceij, torqueij,
                                forceji, torqueji);
                            }
                        }
                    // no edges either; must be a sphere
                    else
                        {
                        // all pairs of vertices
                        for(size_t vertj(0); vertj < h_numTypeVerts.data[typej]; ++vertj)
                            {
                            vec3<Real> vertex1(h_verts.data[h_firstTypeVert.data[typej] + vertj]);
                            vertex1 = rotate(quatj, vertex1);

                            m_evaluator.vertexVertex(dx, vertex0, dx + vertex1,
                                potentialij, forceij, torqueij,
                                forceji, torqueji);
                            }
                        }
                    }

                // iterate over each vertex in particle j
                for(size_t vertIndex(0); vertIndex < h_numTypeVerts.data[typej]; ++vertIndex)
                    {
                    const vec3<Real> vertex0(
                        rotate(quatj, vec3<Real>(h_verts.data[h_firstTypeVert.data[typej] + vertIndex])));

                    // iterate over each face in particle i
                    size_t faceIndex(typei);
                    if(h_numTypeFaces.data[typei] > 0)
                        {
                        do
                            {
                            m_evaluator.vertexFace(-dx, vertex0, quati,
                                h_verts.data,
                                h_realVertIndex.data,
                                h_nextFaceVert.data,
                                h_firstFaceVert.data[faceIndex],
                                potentialij,
                                forceji, torqueji,
                                forceij, torqueij);
                            faceIndex = h_nextFace.data[faceIndex];
                            }
                        while(faceIndex != typei);
                        }
                    // no faces; is it a spherocylinder?
                    else if(h_numTypeEdges.data[typei] > 0)
                        {
                        // iterate over all edges of i
                        for(size_t edgei(0); edgei < h_numTypeEdges.data[typei]; ++edgei)
                            {
                            vec3<Real> p10(h_verts.data[h_edges.data[2*(edgei + h_firstTypeEdge.data[typei])]]);
                            vec3<Real> p11(h_verts.data[h_edges.data[2*(edgei + h_firstTypeEdge.data[typei]) + 1]]);
                            p10 = rotate(quati, p10);
                            p11 = rotate(quati, p11);

                            m_evaluator.vertexEdge(-dx, vertex0, p10, p11,
                                potentialij, forceji, torqueji,
                                forceij, torqueij);
                            }
                        }
                    // if it is a sphere, the vertex/vertex check was
                    // done above while iterating over vertices in
                    // particle i so we don't need another one here
                    }

                // iterate over all pairs of edges
                for(size_t edgei(0); edgei < h_numTypeEdges.data[typei]; ++edgei)
                    {
                    vec3<Real> p00(h_verts.data[h_edges.data[2*(edgei + h_firstTypeEdge.data[typei])]]);
                    vec3<Real> p01(h_verts.data[h_edges.data[2*(edgei + h_firstTypeEdge.data[typei]) + 1]]);
                    p00 = rotate(quati, p00);
                    p01 = rotate(quati, p01);

                    // iterate over all edges of j
                    for(size_t edgej(0); edgej < h_numTypeEdges.data[typej]; ++edgej)
                        {
                        vec3<Real> p10(h_verts.data[h_edges.data[2*(edgej + h_firstTypeEdge.data[typej])]]);
                        vec3<Real> p11(h_verts.data[h_edges.data[2*(edgej + h_firstTypeEdge.data[typej]) + 1]]);
                        p10 = rotate(quatj, p10);
                        p11 = rotate(quatj, p11);

                        m_evaluator.edgeEdge(dx, p00, p01, dx + p10, dx + p11, potentialij, forceij, torqueij, forceji, torqueji);
                        }
                    }

                // compute the pair energy and virial (FLOPS: 6)
                Real pair_virial[6];
                pair_virial[0] = -Real(0.5) * dx.x * forceij.x;
                pair_virial[1] = -Real(0.5) * dx.y * forceij.x;
                pair_virial[2] = -Real(0.5) * dx.z * forceij.x;
                pair_virial[3] = -Real(0.5) * dx.y * forceij.y;
                pair_virial[4] = -Real(0.5) * dx.z * forceij.y;
                pair_virial[5] = -Real(0.5) * dx.z * forceij.z;

                // Scale potential energy by half for pairwise contribution
                potentialij *= Real(0.5);

                // add the force, potential energy and virial to the particle i
                // (FLOPS: 8)
                fi += forceij;
                ti += torqueij;
                pei += potentialij;
                viriali[0] += pair_virial[0];
                viriali[1] += pair_virial[1];
                viriali[2] += pair_virial[2];
                viriali[3] += pair_virial[3];
                viriali[4] += pair_virial[4];
                viriali[5] += pair_virial[5];

                // add the force to particle j if we are using the third law (MEM TRANSFER: 10 scalars / FLOPS: 8)
                if (third_law && k < m_pdata->getN())
                    {
                    h_force.data[k].x  += forceji.x;
                    h_force.data[k].y  += forceji.y;
                    h_force.data[k].z  += forceji.z;
                    h_force.data[k].w  += potentialij;
                    h_torque.data[k].x += torqueji.x;
                    h_torque.data[k].y += torqueji.y;
                    h_torque.data[k].z += torqueji.z;
                    h_virial.data[0*virial_pitch + k] += pair_virial[0];
                    h_virial.data[1*virial_pitch + k] += pair_virial[1];
                    h_virial.data[2*virial_pitch + k] += pair_virial[2];
                    h_virial.data[3*virial_pitch + k] += pair_virial[3];
                    h_virial.data[4*virial_pitch + k] += pair_virial[4];
                    h_virial.data[5*virial_pitch + k] += pair_virial[5];
                    }
                }

            }

        // finally, increment the force, potential energy and virial for particle i
        // (MEM TRANSFER: 10 scalars / FLOPS: 5)
        h_force.data[i].x  += fi.x;
        h_force.data[i].y  += fi.y;
        h_force.data[i].z  += fi.z;
        h_force.data[i].w  += pei;
        h_torque.data[i].x += ti.x;
        h_torque.data[i].y += ti.y;
        h_torque.data[i].z += ti.z;
        h_virial.data[0*virial_pitch + i] += viriali[0];
        h_virial.data[1*virial_pitch + i] += viriali[1];
        h_virial.data[2*virial_pitch + i] += viriali[2];
        h_virial.data[3*virial_pitch + i] += viriali[3];
        h_virial.data[4*virial_pitch + i] += viriali[4];
        h_virial.data[5*virial_pitch + i] += viriali[5];
        }

    int64_t flops = m_pdata->getN() * 5 + n_calc * (3+5+9+1+14+6+8);
    if (third_law) flops += n_calc * 8;
    int64_t mem_transfer = m_pdata->getN() * (5+4+10)*sizeof(Real) + n_calc * (1+3+1)*sizeof(Real);
    if (third_law) mem_transfer += n_calc*10*sizeof(Real);
    if (m_prof) m_prof->pop(flops, mem_transfer);
    }

#ifdef WIN32
#pragma warning( pop )
#endif
