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

/*! \file Initializers.h
    \brief Declares a few initializers for setting up ParticleData instances
*/

#ifdef NVCC
#error This header cannot be compiled by nvcc
#endif

#include "ParticleData.h"

#ifndef __INITIALIZERS_H__
#define __INITIALIZERS_H__

//! Forward declaration of SnapshotSystemData
class SnapshotSystemData;

//! Inits a ParticleData with a simple cubic array of particles
/*! A number of particles along each axis are specified along with a spacing
    between particles. This initializer only generates a single particle type.
    \ingroup data_structs
*/
class SimpleCubicInitializer 
    {
    public:
        //! Set the parameters
        SimpleCubicInitializer(unsigned int M, Scalar spacing, const std::string &type_name);
        //! Empty Destructor
        virtual ~SimpleCubicInitializer() { }
        
        //! initializes a snapshot with the particle data
        virtual void initSnapshot(SnapshotSystemData &snapshot) const;
        
    private:
        unsigned int m_M;   //!< Number of particles wide to make the box
        Scalar m_spacing;   //!< Spacing between particles
        BoxDim box;         //!< Precalculated box
        std::string m_type_name;    //!< Name of the particle type created
    };

//! Inits a ParticleData with randomly placed particles in a cube
/*! A minimum distance parameter is provided so that particles are never
    placed too close together. This initializer only generates a single particle
    type.
*/
class RandomInitializer 
    {
    public:
        //! Set the parameters
        RandomInitializer(unsigned int N, Scalar phi_p, Scalar min_dist, const std::string &type_name);
        //! Empty Destructor
        virtual ~RandomInitializer() { }
        
        //! initializes a snapshot with the particle data
        virtual void initSnapshot(SnapshotSystemData &snapshot) const;
        
        //! Sets the random seed to use in the generation
        void setSeed(unsigned int seed);
        
    protected:
        unsigned int m_N;           //!< Number of particles to generate
        Scalar m_phi_p;             //!< Packing fraction to generate the particles at
        Scalar m_min_dist;          //!< Minimum distance to separate particles by
        BoxDim m_box;               //!< Box to put the particles in
        std::string m_type_name;    //!< Name of the particle type created
    };


//! Creates a random particle system with walls defined on all 6 faces of the cube
/*! A \a wall_buffer argument is specified in the the call to the constructor which shifts the edge of the
    simulation box out that distance from the walls.
*/
class RandomInitializerWithWalls : public RandomInitializer
    {
    public:
        //! Set the parameters
        RandomInitializerWithWalls(unsigned int N, Scalar phi_p, Scalar min_dist, Scalar wall_buffer, const std::string &type_name);
        //! Empty Destructor
        virtual ~RandomInitializerWithWalls() ;

        //! initializes a snapshot with the particle data
        virtual void initSnapshot(SnapshotSystemData &snapshot) const;
 
    protected:
        Scalar m_wall_buffer;   //!< Buffer distance between the wall and the edge of the box
        BoxDim m_real_box;      //!< Stores the actual dimensions of the box where the walls are defined
        
    };

//! Exports the SimpleCubicInitializer class to python
void export_SimpleCubicInitializer();
//! Exports the RandomInitializer class to python
void export_RandomInitializer();
//! Exports the RandomInitializerWithWalls class to python
void export_RandomInitializerWithWalls();

#endif

