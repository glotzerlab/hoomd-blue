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

/*! \file WallData.h
    \brief Contains declarations for WallData.
 */

#include <math.h>
#include "ParticleData.h"

#include <boost/utility.hpp>

#ifndef __WALLDATA_H__
#define __WALLDATA_H__

//! Simple structure representing a single wall
/*! Walls are represented by an origin and a unit length normal.
    \ingroup data_structs
*/
struct Wall
    {
    //! Constructor
    /*! \param ox Origin x-component
        \param oy Origin y-component
        \param oz Origin z-component
        \param nx Origin x-component
        \param ny Normal y-component
        \param nz Normal z-component
    */
    Wall(Scalar ox=0.0, Scalar oy=0.0, Scalar oz=0.0, Scalar nx=1.0, Scalar ny=0.0, Scalar nz=0.0)
            : origin_x(ox), origin_y(oy), origin_z(oz)
        {
        // normalize nx,ny,nz
        Scalar len = sqrt(nx*nx + ny*ny + nz*nz);
        normal_x = nx / len;
        normal_y = ny / len;
        normal_z = nz / len;
        }
        
    Scalar origin_x;    //!< x-component of the origin
    Scalar origin_y;    //!< y-component of the origin
    Scalar origin_z;    //!< z-component of the origin
    
    Scalar normal_x;    //!< x-component of the normal
    Scalar normal_y;    //!< y-component of the normal
    Scalar normal_z;    //!< z-component of the normal
    };

//! Stores information about all the walls defined in the simulation
/*! WallData is responsible for storing all of the walls in the simulation.
    Walls are specified by the Wall struct and any number can be added.

    On the CPU, walls can be accessed with getWall()

    An optimized data structure for the GPU will be written later.
    It will most likely take the form of a 2D texture.
    \ingroup data_structs
*/
class WallData : boost::noncopyable
    {
    public:
        //! Creates an empty structure with no walls
        WallData() : m_walls() {}
        
        //! Get the number of walls in the data
        /*! \return Number of walls
        */
        unsigned int getNumWalls() const
            {
            return (unsigned int)m_walls.size();
            }
            
        //! Access a specific wall
        /*! \param idx Index of the wall to retrieve
            \return Wall stored at index \a idx
        */
        const Wall& getWall(unsigned int idx) const
            {
            assert(idx < m_walls.size());
            return m_walls[idx];
            }
            
        //! Adds a wall to the data structure
        void addWall(const Wall& wall);
        
    private:
        //! Storage for the walls
        std::vector<Wall> m_walls;
    };

#endif

