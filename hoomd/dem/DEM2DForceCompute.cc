// Copyright (c) 2009-2019 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.

// Maintainer: mspells

#ifdef WIN32
#pragma warning( push )
#pragma warning( disable : 4103 4244 )
#endif

#include "DEM2DForceCompute.h"
#include <hoomd/extern/pybind/include/pybind11/pybind11.h>

#include <stdexcept>

#ifdef ENABLE_OPENMP
#include <omp.h>
#endif

/*! \file DEM2DForceCompute.cc
  \brief Defines the DEM2DForceCompute class
*/

using namespace std;

/*! \param sysdef System to compute forces on
  \param nlist Neighborlist to use for computing the forces
  \param r_cut Cutoff radius beyond which the force is 0
  \param potential Global potential parameters for the compute
  \post memory is allocated
*/
template<typename Real, typename Real4, typename Potential>
DEM2DForceCompute<Real, Real4, Potential>::DEM2DForceCompute(
    std::shared_ptr<SystemDefinition> sysdef,
    std::shared_ptr<NeighborList> nlist,
    Real r_cut, Potential potential)
    : ForceCompute(sysdef), m_nlist(nlist), m_r_cut(r_cut),
      m_evaluator(potential), m_shapes()
    {
    m_exec_conf->msg->notice(5) << "Constructing DEM2DForceCompute" << endl;

    assert(m_pdata);
    assert(m_nlist);

    if (r_cut < 0.0)
        {
        m_exec_conf->msg->error() << "dem: Negative r_cut makes no sense" << endl;
        throw runtime_error("Error initializing DEM2DForceCompute");
        }
    }

/*! Destructor. */
template<typename Real, typename Real4, typename Potential>
DEM2DForceCompute<Real, Real4, Potential>::~DEM2DForceCompute()
    {
    m_exec_conf->msg->notice(5) << "Destroying DEM2DForceCompute" << endl;
    }

template<typename Real, typename Real4, typename Potential>
void DEM2DForceCompute<Real, Real4, Potential>::connectDEMGSDShapeSpec(std::shared_ptr<GSDDumpWriter> writer)
    {
    typedef hoomd::detail::SharedSignalSlot<int(gsd_handle&)> SlotType;
    auto func = std::bind(&DEM2DForceCompute<Real, Real4, Potential>::slotWriteDEMGSDShapeSpec, this, std::placeholders::_1);
    std::shared_ptr<hoomd::detail::SignalSlot> pslot(new SlotType(writer->getWriteSignal(), func));
    addSlot(pslot);
    }

template<typename Real, typename Real4, typename Potential>
int DEM2DForceCompute<Real, Real4, Potential>::slotWriteDEMGSDShapeSpec(gsd_handle& handle) const
    {
    GSDShapeSpecWriter shapespec(m_exec_conf);
    m_exec_conf->msg->notice(10) << "DEM3DForceCompute writing particle shape information to GSD file in chunk: " << shapespec.getName() << std::endl;
    int retval = shapespec.write(handle, this->getTypeShapeMapping(m_shapes, m_evaluator.getRadius()));
    return retval;
    }

template<typename Real, typename Real4, typename Potential>
std::string DEM2DForceCompute<Real, Real4, Potential>::getTypeShape(const std::vector<vec2<Real>> &verts, const Real &radius) const
    {
    std::ostringstream shapedef;
    unsigned int nverts = verts.size();
    if (nverts == 1)
        {
        shapedef << "{\"type\": \"Sphere\", " << "\"diameter\": " << Real(2)*radius << "}";
        }
    else if (nverts == 2)
        {
        throw std::runtime_error("Shape definition not supported for 2-vertex polygons");
        }
    else
        {
        shapedef << "{\"type\": \"Polygon\", " << "\"rounding_radius\": " << radius <<
                    ", \"vertices\": "  << encodeVertices(verts) << "}";
        }
    return shapedef.str();
    }

template<typename Real, typename Real4, typename Potential>
std::string DEM2DForceCompute<Real, Real4, Potential>::encodeVertices(const std::vector<vec2<Real>> &verts) const
    {
    std::ostringstream vertstr;
    unsigned int nverts = verts.size();
    vertstr << "[";
    for (unsigned int i = 0; i < nverts-1; i++)
        {
        vertstr << "[" << verts[i].x << ", " << verts[i].y << "], ";
        }
    vertstr << "[" << verts[nverts-1].x << ", " << verts[nverts-1].y << "]" << "]";
    return vertstr.str();
    }

template<typename Real, typename Real4, typename Potential>
std::vector<std::string> DEM2DForceCompute<Real, Real4, Potential>::getTypeShapeMapping(const std::vector<std::vector<vec2<Real>>> &verts, const Real &radius) const
    {
    std::vector<std::string> type_shape_mapping(verts.size());
    for (unsigned int i = 0; i < type_shape_mapping.size(); i++)
        {
        type_shape_mapping[i] = this->getTypeShape(verts[i], radius);
        }
    return type_shape_mapping;
    }

template<typename Real, typename Real4, typename Potential>
pybind11::list DEM2DForceCompute<Real, Real4, Potential>::getTypeShapesPy()
    {
    std::vector<std::string> type_shape_mapping = this->getTypeShapeMapping(m_shapes, m_evaluator.getRadius());
    pybind11::list type_shapes;
    for (unsigned int i = 0; i < type_shape_mapping.size(); i++)
        type_shapes.append(type_shape_mapping[i]);
    return type_shapes;
    }

/*! setParams: set the vertices for a numeric particle type from a python list.
  \param type Particle type index
  \param vertices Python list of 2D vertices specifying a polygon
*/
template<typename Real, typename Real4, typename Potential>
void DEM2DForceCompute<Real, Real4, Potential>::setParams(
    unsigned int type, const pybind11::list &vertices)
    {
    if (type >= m_pdata->getNTypes())
        {
        m_exec_conf->msg->error() <<
            "dem: Trying to set params for a non existent type! " << type << endl;
        throw runtime_error("Error setting parameters in DEM2DForceCompute");
        }

    for(int i(type - m_shapes.size()); i >= 0; --i)
        m_shapes.push_back(vector<vec2<Real> >(0));

    // build a vector of points
    vector<vec2<Real> > points;

    for(size_t i(0); i < (size_t) pybind11::len(vertices); i++)
        {
        const pybind11::tuple pyPoint = pybind11::cast<pybind11::tuple>(vertices[i]);

        if(pybind11::len(pyPoint) != 2)
            throw runtime_error("Non-2D vertex given for DEM2DForceCompute::setParams");

        const Real x = pybind11::cast<Real>(pyPoint[0]);
        const Real y = pybind11::cast<Real>(pyPoint[1]);
        const vec2<Real> point(x, y);
        points.push_back(point);
        }

    m_shapes[type] = points;
    }

/*! DEM2DForceCompute provides
  - \c pair_dem_energy
*/
template<typename Real, typename Real4, typename Potential>
std::vector< std::string > DEM2DForceCompute<Real, Real4, Potential>::getProvidedLogQuantities()
    {
    vector<string> list;
    list.push_back("pair_dem_energy");
    return list;
    }

template<typename Real, typename Real4, typename Potential>
Real DEM2DForceCompute<Real, Real4, Potential>::getLogValue(const std::string& quantity, unsigned int timestep)
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

/*! \post The DEM2D forces are computed for the given timestep. The neighborlist's
  compute method is called to ensure that it is up to date.

  \param timestep specifies the current time step of the simulation
*/
template<typename Real, typename Real4, typename Potential>
void DEM2DForceCompute<Real, Real4, Potential>::computeForces(unsigned int timestep)
    {
    // start by updating the neighborlist
    m_nlist->compute(timestep);

    // start the profile for this compute
    if (m_prof) m_prof->push("DEM2D pair");

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
        vec2<Real> fi;
        Real ti(0), pei(0);
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

        // Make a local copy for the rotated vertices for particle i
        vector<vec2<Real> > vertices_i(m_shapes[typei]);
        for(typename vector<vec2<Real> >::iterator vertIter(vertices_i.begin());
            vertIter != vertices_i.end(); ++vertIter)
            *vertIter = rotate(quati, *vertIter);

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
            assert(k < m_pdata->getN());

            // calculate dr (MEM TRANSFER: 3 scalars / FLOPS: 3)
            vec3<Scalar> pj(h_pos.data[k].x, h_pos.data[k].y, 0);
            quat<Scalar> quatj(h_orientation.data[k]);
            vec3<Scalar> dx3(pj - pi);

            // access the type of the neighbor particle (MEM TRANSFER: 1 scalar
            unsigned int typej = __scalar_as_int(h_pos.data[k].w);
            // sanity check
            assert(typej < m_pdata->getNTypes());

            // apply periodic boundary conditions (FLOPS: 9 (worst case: first branch is missed, the 2nd is taken and the add is done)
            dx3 = vec3<Scalar>(box.minImage(vec_to_scalar3(dx3)));
            vec2<Real> dx(dx3.x, dx3.y);

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
            Scalar rsq = dot(dx, dx);

            // only compute the force if the particles are closer than the cutoff (FLOPS: 1)
            if (m_evaluator.withinCutoff(rsq,r_cut_sq))
                {
                // local forces and torques for particles i and j
                vec2<Real> forceij, forceji;
                Real torqueij(0), torqueji(0), potentialij(0);

                // Make a local copy for the rotated vertices for particle j
                vector<vec2<Real> > vertices_j(m_shapes[typej]);
                for(typename vector<vec2<Real> >::iterator vertIter(vertices_j.begin());
                    vertIter != vertices_j.end(); ++vertIter)
                    *vertIter = rotate(quatj, *vertIter);

                // Iterate over each vertex of particle i, if particle j has any edges
                if (vertices_j.size()>1)
                    {
                    for(typename vector<vec2<Real> >::const_iterator viIter(vertices_i.begin());
                        viIter != vertices_i.end(); ++viIter)
                        {
                        // iterate over each edge of particle j
                        for(typename vector<vec2<Real> >::const_iterator vjIter(vertices_j.begin());
                            vjIter + 1 != vertices_j.end(); ++vjIter)
                            {
                            m_evaluator.vertexEdge(dx, *viIter, *vjIter, *(vjIter + 1),
                                potentialij, forceij, torqueij,
                                forceji, torqueji);
                            }
                        // evaluate for the last edge, but only if we
                        // didn't just evaluate that edge (i.e. the
                        // shape isn't a spherocylinder)
                        if(vertices_j.size() > 2)
                            m_evaluator.vertexEdge(dx, *viIter, vertices_j.back(), vertices_j.front(),
                                potentialij, forceij, torqueij,
                                forceji, torqueji);
                        }
                    }
                // iterate over each vertex of particle j, if vi has any edges
                if (vertices_i.size()>1)
                    {
                    for(typename vector<vec2<Real> >::const_iterator vjIter(vertices_j.begin());
                        vjIter != vertices_j.end(); ++vjIter)
                        {
                        // iterate over each edge of particle i
                        for(typename vector<vec2<Real> >::const_iterator viIter(vertices_i.begin());
                            viIter + 1 != vertices_i.end(); ++viIter)
                            {
                            m_evaluator.vertexEdge(-dx, *vjIter, *viIter, *(viIter + 1),
                                potentialij, forceji, torqueji,
                                forceij, torqueij);
                            }
                        // evaluate for the last edge, but only if we
                        // didn't just evaluate that edge (i.e. the
                        // shape isn't a spherocylinder)
                        if(vertices_i.size() > 2)
                            m_evaluator.vertexEdge(-dx, *vjIter, vertices_i.back(), vertices_i.front(),
                                potentialij, forceji, torqueji,
                                forceij, torqueij);
                        }
                    }
                // if i doesn't have any edges and j doesn't have any
                // edges, both are disks
                else if(vertices_j.size() <= 1)
                    {
                    m_evaluator.vertexVertex(dx, vertices_i[0], dx + vertices_j[0],
                        potentialij, forceij, torqueij,
                        forceji, torqueji);
                    }

                // compute the pair energy and virial (FLOPS: 6)
                Scalar pair_virial[6];

                pair_virial[0] = -Scalar(0.5) * dx.x * forceij.x;
                pair_virial[1] = -Scalar(0.5) * dx.y * forceij.x;
                pair_virial[3] = -Scalar(0.5) * dx.y * forceij.y;

                // Scale potential energy by half for pairwise contribution
                potentialij *= Scalar(0.5);

                // add the force, potential energy and virial to the particle i
                // (FLOPS: 8)
                fi += forceij;
                ti += torqueij;
                pei += potentialij;
                viriali[0] += pair_virial[0];
                viriali[1] += pair_virial[1];
                viriali[3] += pair_virial[3];

                // add the force to particle j if we are using the third law (MEM TRANSFER: 10 scalars / FLOPS: 8)
                if (third_law && k < m_pdata->getN())
                    {
                    h_force.data[k].x  += forceji.x;
                    h_force.data[k].y  += forceji.y;
                    h_force.data[k].w  += potentialij;
                    h_torque.data[k].z += torqueji;
                    h_virial.data[0*virial_pitch + k] += pair_virial[0];
                    h_virial.data[1*virial_pitch + k] += pair_virial[1];
                    h_virial.data[3*virial_pitch + k] += pair_virial[3];
                    }
                }

            }

        // finally, increment the force, potential energy and virial for particle i
        // (MEM TRANSFER: 10 scalars / FLOPS: 5)
        h_force.data[i].x  += fi.x;
        h_force.data[i].y  += fi.y;
        h_force.data[i].w  += pei;
        h_torque.data[i].z += ti;
        h_virial.data[0*virial_pitch + i] += viriali[0];
        h_virial.data[1*virial_pitch + i] += viriali[1];
        h_virial.data[3*virial_pitch + i] += viriali[3];
        }

    int64_t flops = m_pdata->getN() * 5 + n_calc * (3+5+9+1+14+6+8);
    if (third_law) flops += n_calc * 8;
    int64_t mem_transfer = m_pdata->getN() * (5+4+10)*sizeof(Scalar) + n_calc * (1+3+1)*sizeof(Scalar);
    if (third_law) mem_transfer += n_calc*10*sizeof(Scalar);
    if (m_prof) m_prof->pop(flops, mem_transfer);
    }

#ifdef WIN32
#pragma warning( pop )
#endif
