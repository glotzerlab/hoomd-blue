/*
Highly Optimized Object-oriented Many-particle Dynamics -- Blue Edition
(HOOMD-blue) Open Source Software License Copyright 2008-2011 Ames Laboratory
Iowa State University and The Regents of the University of Michigan All rights
reserved.

HOOMD-blue may contain modifications ("Contributions") provided, and to which
copyright is held, by various Contributors who have granted The Regents of the
University of Michigan the right to modify and/or distribute such Contributions.

Redistribution and use of HOOMD-blue, in source and binary forms, with or
without modification, are permitted, provided that the following conditions are
met:

* Redistributions of source code must retain the above copyright notice, this
list of conditions, and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright notice, this
list of conditions, and the following disclaimer in the documentation and/or
other materials provided with the distribution.

* Neither the name of the copyright holder nor the names of HOOMD-blue's
contributors may be used to endorse or promote products derived from this
software without specific prior written permission.

Disclaimer

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDER AND CONTRIBUTORS ``AS IS''
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, AND/OR
ANY WARRANTIES THAT THIS SOFTWARE IS FREE OF INFRINGEMENT ARE DISCLAIMED.

IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT,
INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE
OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF
ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

// Maintainer: joaander

/*! \file SFCPackUpdater.h
    \brief Declares the SFCPackUpdater class
*/

#ifdef WIN32
#pragma warning( push )
#pragma warning( disable : 4103 )
#endif

#include <boost/shared_ptr.hpp>
#include <boost/signals.hpp>
#include <vector>
#include <utility>

#include "Updater.h"
#include "NeighborList.h"

#ifndef __SFCPACK_UPDATER_H__
#define __SFCPACK_UPDATER_H__

//! Sort the particles
/*! Impelements an algorithm that reorders particles in the ParticleData so that particles
    near each other in space become near each other in memory. This transformation improves
    cache locality in almost every other calculation in HOOMD, such as LJForceCompute,
    HarmonicBondForceCompute, and NeighborListBinned, to name a few. As particles move
    through time, they will tend to unsort themselves at a rate depending on how diffusive
    the simulation is. Tests preformed on a Leannard-Jones liquid simulation at a temperature of 1.2
    showed that performing the sort every 1,000 time steps is sufficient to maintain the
    benifits of the sort without significant overhead. Less diffusive systems can easily increase
    that value to 2,000 or more.

    Usage:<br>
    Constructe the SFCPackUpdater, attaching it to the ParticleData. The grid size is automatically set to reasonable 
    defaults, which is as high as it can possibly go without consuming a significant amount of memory. The grid
    dimension can be changed by calling setGrid().

    Implementation details:<br>
    The rearranging is done by computing bins for the particles, and then ordering the particles based on the order in
    which those bins appear along a hilbert curve. It is very efficient, even when the box size changes often as the
    grid dimension is kept constant.

    \ingroup updaters
*/
class SFCPackUpdater : public Updater
    {
    public:
        //! Constructor
        SFCPackUpdater(boost::shared_ptr<SystemDefinition> sysdef);
        
        //! Take one timestep forward
        virtual void update(unsigned int timestep);
        
        //! Set the grid dimension
        /*! \param grid New grid dimension to set
            \note It is automatically rounded up to the nearest power of 2
        */
        void setGrid(unsigned int grid)
            {
            m_grid = (unsigned int)pow(2.0, ceil(log(double(grid)) / log(2.0)));;
            }
            
    private:
        unsigned int m_grid;        //!< Grid dimension to use
        unsigned int m_last_grid;   //!< The last value of MMax
        unsigned int m_last_dim;    //!< Check the last dimension we ran at
        
        std::vector< std::pair<unsigned int, unsigned int> > m_particle_bins;    //!< Binned particles
        
        std::vector< unsigned int > m_traversal_order;      //!< Generated traversal order of bins
        std::vector<unsigned int> m_sort_order;             //!< Generated sort order of the particles
        
        //! Helper function that actually performs the sort
        void getSortedOrder2D();
        //! Helper function that actually performs the sort
        void getSortedOrder3D();
        
        //! Apply the sorted order to the particle data
        void applySortOrder();

        //! Write traversal order out for visualization
        void writeTraversalOrder(const std::string& fname, const vector< unsigned int >& reverse_order);
        
    };

//! Export the SFCPackUpdater class to python
void export_SFCPackUpdater();

#endif

#ifdef WIN32
#pragma warning( pop )
#endif

