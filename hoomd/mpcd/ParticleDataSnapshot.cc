// Copyright (c) 2009-2017 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.

// Maintainer: mphoward

/*!
 * \file ParticleDataSnapshot.cc
 * \brief Definition of MPCD ParticleDataSnapshot
 */

#include "ParticleDataSnapshot.h"
#include "hoomd/extern/num_util.h"

mpcd::ParticleDataSnapshot::ParticleDataSnapshot()
    : size(0), mass(1.0)
    {}

/*!
 * \param N Number of particles in the snapshot
 */
mpcd::ParticleDataSnapshot::ParticleDataSnapshot(unsigned int N)
    : size(N), mass(1.0)
    {
    resize(N);
    }

/*!
 * \param N Number of particles in the snapshot
 */
void mpcd::ParticleDataSnapshot::resize(unsigned int N)
    {
    position.resize(N, vec3<Scalar>(0.0, 0.0, 0.0));
    velocity.resize(N, vec3<Scalar>(0.0, 0.0, 0.0));
    type.resize(N, 0);

    size = N;
    }

/*!
 * \returns True if particle data is valid
 *
 * Checks that all particle data arrays have the correct size and
 * that a valid type mapping exists.
 */
bool mpcd::ParticleDataSnapshot::validate() const
    {
    if (position.size() != size || velocity.size() != size || type.size() != size)
        {
        return false;
        }

    // validate the type map
    // the type map must not be empty, and every type must fall in the range of known types
    if (size > 0)
        {
        if (type_mapping.size() == 0) return false;
        for (unsigned int i=0; i < size; ++i)
            {
            if (type[i] >= type_mapping.size()) return false;
            }
        }

    return true;
    }

/*!
 * \param nx Number of times to replicate along x
 * \param ny Number of times to replicate along y
 * \param nz Number of times to replicate along z
 * \param old_box Old box dimensions
 * \param new_box Dimensions of replicated box
 */
void mpcd::ParticleDataSnapshot::replicate(unsigned int nx,
                                           unsigned int ny,
                                           unsigned int nz,
                                           const BoxDim& old_box,
                                           const BoxDim& new_box)

    {
    assert(nx > 0);
    assert(ny > 0);
    assert(nz > 0);

    const unsigned int old_size = size;

    resize(old_size*nx*ny*nz);

    for (unsigned int i = 0; i < old_size; ++i)
        {
        // unwrap position of particle i in old box using image flags
        vec3<Scalar> p = position[i];
        vec3<Scalar> f = old_box.makeFraction(p);

        unsigned int j = 0;
        for (unsigned int l = 0; l < nx; ++l)
            {
            for (unsigned int m = 0; m < ny; ++m)
                {
                for (unsigned int n = 0; n < nz; ++n)
                    {
                    Scalar3 f_new;
                    // replicate particle
                    f_new.x = f.x/(Scalar)nx + (Scalar)l/(Scalar)nx;
                    f_new.y = f.y/(Scalar)ny + (Scalar)m/(Scalar)ny;
                    f_new.z = f.z/(Scalar)nz + (Scalar)n/(Scalar)nz;

                    unsigned int k = j*old_size + i;

                    // coordinates in new box
                    Scalar3 q = new_box.makeCoordinates(f_new);
                    int3 image = make_int3(0,0,0);
                    new_box.wrap(q,image);

                    position[k] = vec3<Scalar>(q);
                    velocity[k] = velocity[i];
                    type[k] = type[i];
                    ++j;
                    } // n
                } // m
            } // l
        } // i
    }

pybind11::object mpcd::ParticleDataSnapshot::getPosition()
    {
    std::vector<intp> dims(2);
    dims[0] = position.size();
    dims[1] = 3;
    return pybind11::object(num_util::makeNumFromData((Scalar*)&position[0], dims), false);
    }

pybind11::object mpcd::ParticleDataSnapshot::getVelocity()
    {
    std::vector<intp> dims(2);
    dims[0] = velocity.size();
    dims[1] = 3;
    return pybind11::object(num_util::makeNumFromData((Scalar*)&velocity[0], dims), false);
    }

pybind11::object mpcd::ParticleDataSnapshot::getType()
    {
    return pybind11::object(num_util::makeNumFromData(&type[0], type.size()), false);
    }

pybind11::list mpcd::ParticleDataSnapshot::getTypeNames()
    {
    pybind11::list py_types;
    for (unsigned int i=0; i < type_mapping.size(); ++i)
        {
        py_types.append(pybind11::str(type_mapping[i]));
        }
    return py_types;
    }

/*!
 * \param types Python list of strings to set as type names
 */
void mpcd::ParticleDataSnapshot::setTypeNames(pybind11::list types)
    {
    type_mapping.resize(len(types));
    for (unsigned int i=0; i < len(types); ++i)
        {
        type_mapping[i] = pybind11::cast<std::string>(types[i]);
        }
    }

/*!
 * \param m Python module to export to
 */
void mpcd::detail::export_ParticleDataSnapshot(pybind11::module& m)
    {
    pybind11::class_< mpcd::ParticleDataSnapshot, std::shared_ptr<mpcd::ParticleDataSnapshot> >(m, "MPCDParticleDataSnapshot")
    .def(pybind11::init<unsigned int>())
    .def_property_readonly("position", &mpcd::ParticleDataSnapshot::getPosition, pybind11::return_value_policy::take_ownership)
    .def_property_readonly("velocity", &mpcd::ParticleDataSnapshot::getVelocity, pybind11::return_value_policy::take_ownership)
    .def_property_readonly("typeid", &mpcd::ParticleDataSnapshot::getType, pybind11::return_value_policy::take_ownership)
    .def_readwrite("mass", &mpcd::ParticleDataSnapshot::mass)
    .def_property("types", &mpcd::ParticleDataSnapshot::getTypeNames, &mpcd::ParticleDataSnapshot::setTypeNames)
    .def_readonly("N", &mpcd::ParticleDataSnapshot::size)
    .def("resize", &mpcd::ParticleDataSnapshot::resize)
    .def("replicate", &mpcd::ParticleDataSnapshot::replicate)
    ;
    }
