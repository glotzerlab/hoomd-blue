// Copyright (c) 2009-2019 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.

// Maintainer: joaander



#include "Initializers.h"
#include "SnapshotSystemData.h"

#include <stdlib.h>

#include <iostream>
#include <cassert>
#include <stdexcept>

using namespace std;

#include <hoomd/extern/pybind/include/pybind11/pybind11.h>

/*! \file Initializers.cc
    \brief Defines a few initializers for setting up ParticleData instances
*/

////////////////////////////////////////////////////////////////////////////////
// Simple Cubic Initializer

/*! An \c M x \c M x \c M array of particles is going to be created \c spacing units
    apart from each other.
    \param M Number of particles along a side of the cube
    \param spacing Separation between particles
    \param type_name Name of the particle type to create
*/
SimpleCubicInitializer::SimpleCubicInitializer(unsigned int M, Scalar spacing, const std::string &type_name)
    : m_M(M), m_spacing(spacing), box(M * spacing), m_type_name(type_name)
    {
    }

/*! initialize a snapshot with a cubic crystal */
std::shared_ptr< SnapshotSystemData<Scalar> > SimpleCubicInitializer::getSnapshot() const
    {
    std::shared_ptr< SnapshotSystemData<Scalar> > snapshot(new SnapshotSystemData<Scalar>());
    snapshot->global_box = box;

    SnapshotParticleData<Scalar>& pdata = snapshot->particle_data;
    unsigned int num_particles = m_M * m_M * m_M;
    pdata.resize(num_particles);

    Scalar3 lo = box.getLo();

    // just do a simple triple for loop to fill the space
    unsigned int c = 0;
    for (unsigned int k = 0; k < m_M; k++)
        {
        for (unsigned int j = 0; j < m_M; j++)
            {
            for (unsigned int i = 0; i < m_M; i++)
                {
                pdata.pos[c].x = i * m_spacing + lo.x;
                pdata.pos[c].y = j * m_spacing + lo.y;
                pdata.pos[c].z = k * m_spacing + lo.z;
                c++;
                }
            }
        }

    pdata.type_mapping.push_back(m_type_name);
    return snapshot;
    }

/////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////

/*! \param N number of particles to create
    \param phi_p Packing fraction of particles in the box
    \param min_dist Minimum distance two particles will be placed apart
    \param type_name Name of the particle type to create
    \note assumes particles have a diameter of 1
*/
RandomInitializer::RandomInitializer(unsigned int N, Scalar phi_p, Scalar min_dist, const std::string &type_name)
    : m_N(N), m_phi_p(phi_p), m_min_dist(min_dist), m_type_name(type_name)
    {
    // sanity checks
    if (N == 0)
        {
        throw runtime_error("RandomInitializer: Cannot generate 0 particles");
        }
    if (phi_p <= 0)
        {
        throw runtime_error("RandomInitializer: phi_p <= 0 doesn't make sense");
        }
    if (min_dist < 0)
        {
        throw runtime_error("RandomInitializer: min_dist <= 0 doesn't make sense");
        }

    Scalar L = pow(Scalar(M_PI/6.0)*Scalar(N) / phi_p, Scalar(1.0/3.0));
    m_box = BoxDim(L);
    }

/*! \param seed Random seed to set
    Two RandomInitializers with the same random seen should produce the same
    particle positions.

    \warning setSeed is guaranteed to work properly if and only if
    there are no methods that might call random() called between
    the setSeed and the construction of the ParticleData.
*/
void RandomInitializer::setSeed(unsigned int seed)
    {
    srand(seed);
    }

/*  \post \a N particles are randomly placed in the box
    \note An exception is thrown if too many tries are made to find a spot where
        min_dist can be satisfied.
*/
std::shared_ptr< SnapshotSystemData<Scalar> > RandomInitializer::getSnapshot() const
    {
    std::shared_ptr< SnapshotSystemData<Scalar> > snapshot(new SnapshotSystemData<Scalar>());
    snapshot->global_box = m_box;

    SnapshotParticleData<Scalar>& pdata = snapshot->particle_data;
    pdata.resize(m_N);

    Scalar L = m_box.getL().x;
    for (unsigned int i = 0; i < m_N; i++)
        {
        // generate random particles until we find a suitable one meeting the min_dist
        // criteria
        bool done = false;
        unsigned int tries = 0;
        Scalar x,y,z;
        while (!done)
            {
            //Hack to fix compilation error
            x = Scalar((rand())/Scalar(RAND_MAX) - 0.5)*L;
            y = Scalar((rand())/Scalar(RAND_MAX) - 0.5)*L;
            z = Scalar((rand())/Scalar(RAND_MAX) - 0.5)*L;
            // assume we are done unless we are not
            done = true;
            // only do the minimum distance check if the minimum distance is non-zero
            if (m_min_dist > 1e-6)
                {
                for (unsigned int j = 0; j < i; j++)
                    {
                    Scalar dx = pdata.pos[j].x - x;
                    if (dx < -L/Scalar(2.0))
                        dx += L;
                    if (dx > L/Scalar(2.0))
                        dx -= L;

                    Scalar dy = pdata.pos[j].y - y;
                    if (dy < -L/Scalar(2.0))
                        dy += L;
                    if (dy > L/Scalar(2.0))
                        dy -= L;

                    Scalar dz = pdata.pos[j].z - z;
                    if (dz < -L/Scalar(2.0))
                        dz += L;
                    if (dz > L/Scalar(2.0))
                        dz -= L;

                    Scalar dr2 = dx*dx + dy*dy + dz*dz;
                    if (dr2 <= m_min_dist * m_min_dist)
                        done = false;
                    }
                }
            tries++;
            if (tries > m_N*100)
                {
                throw runtime_error("Unable to find location for particle after trying many times");
                }
            }

        pdata.pos[i] = vec3<Scalar>(x,y,z);
        }

    pdata.type_mapping.push_back(m_type_name);
    return snapshot;
    }
