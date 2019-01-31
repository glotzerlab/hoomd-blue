// Copyright (c) 2009-2019 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.


// Maintainer: joaander

/*! \file SFCPackUpdater.h
    \brief Declares the SFCPackUpdater class
*/

#ifdef NVCC
#error This header cannot be compiled by nvcc
#endif

#include "Updater.h"
#include "GPUVector.h"

#include <memory>
#include <vector>
#include <utility>
#include <hoomd/extern/pybind/include/pybind11/pybind11.h>

#ifndef __SFCPACK_UPDATER_H__
#define __SFCPACK_UPDATER_H__

//! Sort the particles
/*! Implements an algorithm that reorders particles in the ParticleData so that particles
    near each other in space become near each other in memory. This transformation improves
    cache locality in almost every other calculation in HOOMD, such as LJForceCompute,
    HarmonicBondForceCompute, and NeighborListBinned, to name a few. As particles move
    through time, they will tend to unsort themselves at a rate depending on how diffusive
    the simulation is. Tests preformed on a Lennard-Jones liquid simulation at a temperature of 1.2
    showed that performing the sort every 1,000 time steps is sufficient to maintain the
    benefits of the sort without significant overhead. Less diffusive systems can easily increase
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
class PYBIND11_EXPORT SFCPackUpdater : public Updater
    {
    public:
        //! Constructor
        SFCPackUpdater(std::shared_ptr<SystemDefinition> sysdef);

        //! Destructor
        virtual ~SFCPackUpdater();

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

    protected:
        unsigned int m_grid;        //!< Grid dimension to use
        unsigned int m_last_grid;   //!< The last value of MMax
        unsigned int m_last_dim;    //!< Check the last dimension we ran at
        GPUArray< unsigned int > m_traversal_order;      //!< Generated traversal order of bins

        //! Helper function that actually performs the sort
        virtual void getSortedOrder2D();
        //! Helper function that actually performs the sort
        virtual void getSortedOrder3D();

        //! Apply the sorted order to the particle data
        virtual void applySortOrder();

        //! Helper function to generate traversal order
        static void generateTraversalOrder(int i, int j, int k, int w, int Mx, unsigned int cell_order[8], std::vector< unsigned int > &traversal_order);

        //! Write traversal order out for visualization
        void writeTraversalOrder(const std::string& fname, const std::vector< unsigned int >& reverse_order);

        //! Reallocate internal arrays
        virtual void reallocate();

    private:
        std::vector<unsigned int> m_sort_order;             //!< Generated sort order of the particles
        std::vector< std::pair<unsigned int, unsigned int> > m_particle_bins;    //!< Binned particles

   };

//! Export the SFCPackUpdater class to python
void export_SFCPackUpdater(pybind11::module& m);

#endif
