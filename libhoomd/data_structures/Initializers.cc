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
// Maintainer: joaander

#ifdef WIN32
#pragma warning( push )
#pragma warning( disable : 4244 )
#endif

#include <stdlib.h>

#include <iostream>
#include <cassert>
#include <stdexcept>

using namespace std;

#include <boost/python.hpp>
using namespace boost::python;

#include "Initializers.h"
#include "WallData.h"


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

/*! \return Number of particles that will be created
*/
unsigned int SimpleCubicInitializer::getNumParticles() const
    {
    return m_M * m_M * m_M;
    }

/*! \return Box dimensions that just fit the particles on their lattice
*/
BoxDim SimpleCubicInitializer::getBox() const
    {
    return box;
    }


/*! initialize a snapshot with a cubic crystal */
void SimpleCubicInitializer::initSnapshot(SnapshotParticleData &snapshot) const
    {
    assert(snapshot.size == getNumParticles());
    Scalar3 lo = box.getLo();
    
    // just do a simple triple for loop to fill the space
    unsigned int c = 0;
    for (unsigned int k = 0; k < m_M; k++)
        {
        for (unsigned int j = 0; j < m_M; j++)
            {
            for (unsigned int i = 0; i < m_M; i++)
                {
                snapshot.pos[c].x = i * m_spacing + lo.x;
                snapshot.pos[c].y = j * m_spacing + lo.y;
                snapshot.pos[c].z = k * m_spacing + lo.z;
                c++;
                }
            }
        }

    snapshot.type_mapping.push_back(m_type_name);
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
        cerr << endl << "***Error! RandomInitializer: Cannot generate 0 particles" << endl << endl;
        throw runtime_error("Error initializing RandomInitializer");
        }
    if (phi_p <= 0)
        {
        cerr << endl << "***Error! RandomInitializer: phi_p <= 0 doesn't make sense" << endl << endl;
        throw runtime_error("Error initializing RandomInitializer");
        }
    if (min_dist < 0)
        {
        cerr << endl << "***Error! RandomInitializer: min_dist <= 0 doesn't make sense" << endl << endl;
        throw runtime_error("Error initializing RandomInitializer");
        }
        
    Scalar L = pow(Scalar(M_PI/6.0)*Scalar(N) / phi_p, Scalar(1.0/3.0));
    m_box = BoxDim(L);
    }

/*! \return Number of particles that will be initialized
*/
unsigned int RandomInitializer::getNumParticles() const
    {
    return m_N;
    }

/*! \return Box particles will be placed in
*/
BoxDim RandomInitializer::getBox() const
    {
    return m_box;
    }

/*! \param seed Random seed to set
    Two RandomInitializers with the same random seen should produce the same
    particle positions.

    \warning setSeed is guarunteed to work properly if and only if
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
void RandomInitializer::initSnapshot(SnapshotParticleData& snapshot) const
    {
    assert(snapshot.size == m_N);

    Scalar L = m_box.getL().x;
    for (unsigned int i = 0; i < m_N; i++)
        {
        // generate random particles until we find a suitable one meating the min_dist
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
                    Scalar dx = snapshot.pos[j].x - x;
                    if (dx < -L/Scalar(2.0))
                        dx += L;
                    if (dx > L/Scalar(2.0))
                        dx -= L;
                        
                    Scalar dy = snapshot.pos[j].y - y;
                    if (dy < -L/Scalar(2.0))
                        dy += L;
                    if (dy > L/Scalar(2.0))
                        dy -= L;
                        
                    Scalar dz = snapshot.pos[j].z - z;
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
                cerr << endl 
                     << "***Error! RandomInitializer: Unable to find location for particle after trying many times"
                     << endl << endl;
                throw runtime_error("Unable to init system in RandomInitializer");
                }
            }

        snapshot.pos[i] = make_scalar3(x,y,z);
        }

    snapshot.type_mapping.push_back(m_type_name);
    }

/////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////

/*! \param N number of particles to create
    \param phi_p Packing fraction of particles in the box
    \param min_dist Minimum distance two particles will be placed apart
    \param wall_buffer Distance from the edge of the box to place the walls
    \param type_name Name of the particle type to create
    \note assumes particles have a diameter of 1
*/
RandomInitializerWithWalls::RandomInitializerWithWalls(unsigned int N,
                                                       Scalar phi_p,
                                                       Scalar min_dist,
                                                       Scalar wall_buffer,
                                                       const std::string &type_name)
    : RandomInitializer(N, phi_p, min_dist, type_name), m_wall_buffer(wall_buffer)
    {
    Scalar L = pow(Scalar(M_PI/6.0)*Scalar(N) / phi_p, Scalar(1.0/3.0));
    // artificially shrink the box dimensions by 10% so that the super class doesn't put
    // particles too close to the walls
    m_box = BoxDim(L*Scalar(0.9));
    // save the real box for specifying the walls
    m_real_box = BoxDim(L);
    }

RandomInitializerWithWalls::~RandomInitializerWithWalls()
    {
    }

//! Returns the box the particles will sit in
BoxDim RandomInitializerWithWalls::getBox() const
    {
    // the real box dimensions we need to return need to be increased by m_wall_buffer*2
    Scalar L = m_real_box.getL().x + m_wall_buffer*2;
    return BoxDim(L);
    }

/*! \param wall_data Shared pointer to the WallData to initialize
    Walls are created on all 6 sides of the box, spaced in from the edge by a distance of \a wall_buffer
    specified in the constructor.
*/
void RandomInitializerWithWalls::initWallData(boost::shared_ptr<WallData> wall_data) const
    {
    Scalar3 lo = m_real_box.getLo();
    Scalar3 hi = m_real_box.getHi();
    // add all walls
    // left
    wall_data->addWall(Wall(lo.x, 0.0, 0.0, 1.0, 0.0, 0.0));
    // right
    wall_data->addWall(Wall(hi.x, 0.0, 0.0, -1.0, 0.0, 0.0));
    // bottom
    wall_data->addWall(Wall(0.0, lo.y, 0.0, 0.0, 1.0, 0.0));
    // top
    wall_data->addWall(Wall(0.0, hi.y, 0.0, 0.0, -1.0, 0.0));
    // front
    wall_data->addWall(Wall(0.0, 0.0, lo.z, 0.0, 0.0, 1.0));
    // back
    wall_data->addWall(Wall(0.0, 0.0, hi.z, 0.0, 0.0, -1.0));
    }


void export_SimpleCubicInitializer()
    {
    class_< SimpleCubicInitializer, bases<ParticleDataInitializer> >
        ("SimpleCubicInitializer", init<unsigned int, Scalar, string>())
    ;
    // no need to .def methods, they are all inherited
    }

void export_RandomInitializer()
    {
    class_< RandomInitializer, bases<ParticleDataInitializer> >
        ("RandomInitializer", init<unsigned int, Scalar, Scalar, string>())
    ;
    // no need to .def methods, they are all inherited
    }

void export_RandomInitializerWithWalls()
    {
    class_< RandomInitializerWithWalls, bases<ParticleDataInitializer> >
        ("RandomInitializerWithWalls", init<unsigned int, Scalar, Scalar, Scalar, string>())
    ;
    // no need to .def methods, they are all inherited
    }

#ifdef WIN32
#pragma warning( pop )
#endif

