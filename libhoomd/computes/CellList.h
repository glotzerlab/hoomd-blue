/*
Highly Optimized Object-oriented Many-particle Dynamics -- Blue Edition
(HOOMD-blue) Open Source Software License Copyright 2008, 2009 Ames Laboratory
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

// $Id$
// $URL$
// Maintainer: joaander

#ifdef WIN32
#pragma warning( push )
#pragma warning( disable : 4103 )
#endif

#include <boost/shared_ptr.hpp>
#include <boost/signals.hpp>
#include "GPUArray.h"

#include "Compute.h"

/*! \file CellList.h
    \brief Declares the CellList class
*/

#ifndef __CELLLIST_H__
#define __CELLLIST_H__

//! Computes a cell list from the particles in the system
/*! \b Overview:
	Cell lists are useful data structures when working with locality queries on particles. The most notable useage of
	cell lists in HOOMD is as an auxilliary data structure used when building the neighbor list. The data layout design
	decisions for CellList were made to optimize the performance of the neighbor list build as deteremined by
	microbenchmarking. However, CellList is written as generally as possible so that it can be used throughout the code
	in other locations where a cell list is needed.
	
	A cell list defines a set of cuboid cells that completely covers the simulation box. A single nominal width is given
	which is then rounded up to fit an integral number of cells across each dimension. A given particle belongs in one
	and only one cell, the one that contains the particle's x,y,z position.
	
	<b>Data storage:</b>
	
	All data is stored in GPUArrays for access on the host and device.
	
	 * The \c cell_size array lists the number of members in each cell.
	 * The \c xyzf array contains Scalar4 elements each of which holds the x,y,z coordinates of the particle and a flag.
	   That flag can optionally be the particle index or its charge.
	 * The \c tdb array contains Scalar4 elements each of which holds the type, diameter, and body of the particle.
	   It is only computed is requested to reduce the computation time when it is not needed.
	 * The cell_adj array lists indices of adjacent cells. A specified radius (3,5,7,...) of cells is included in the
	   list.
	
	A given cell cuboid with x,y,z indices of i,j,k has a unique cell index. This index can be obtained from the Index3D
	object returned by ____
	\code
	Index3D cell_indexer = ___
	cidx = cell_indexer(i,j,k)
	\endcode
	
	The other arrays are 2D (or 4D if you include i,j,k) in nature. They can be indexed with the appropriate Index2D
	from ____ or ____. 
	\code
	Index2D cell_list_indexer = ____
	Index2D cell_adj_indexer = ____
	\endcode
	
	 * <code>cell_size[cidx]</code> is the number of particles in the cell with index \c cidx
	 * \c xyzf is Ncells x Nmax and <code>xyzf[cell_list_indexer(offset,cidx)]</code> is the data stored for particle
	 \c offset in cell \c cidx (\c offset can vary from 0 to <code>cell_size[cidx]-1</code>)
	 * \c tbd is structured identically to \c xyzf
	 * <code>cell_adj[cell_adj_indexer(offset,cidx)]</code> is the cell index for neighboring cell \c offset to \c cidx.
	   \c offset can vary from 9 to (radius*2+1)^3-1 (typically 26 with radius 1)
	 
	<b>Parameters:</b>
	 * \c width - minimum width of a cell in any x,y,z direction
	 * \c radius - integer radius of cells to generate in \c cell_adj (1,2,3,4,...)
	 * \c max_cells - maximum number of cells to allocate
	 
	After a set call is made to adjust a parameter, changes do not take effect until the next call to compute().
*/
class CellList : public Compute
	{
	public:
        CellList(boost::shared_ptr<SystemDefinition> sysdef);
		
		void setNominalWidth(Scalar width)
			{
			m_nominalWidth = width;
			m_params_changed = true;
			}
		
		void setRadius(unsigned int radius)
			{
			m_radius = radius;
			m_params_changed = true;
			}
		
		void setMaxCells(unsigned int max_cells)
			{
			m_max_cells = max_cells;
			m_params_changed = true;
			}
		
		const Index3D& getCellIndexer()
			{
			return m_cell_indexer;
			}
		
		const Index2D& getCellListIndexer()
			{
			return m_cell_list_indexer;
			}
		
		const Index2D& getCellAdjIndexer()
			{
			return m_cell_adj_indexer;
			}
		
		const GPUArray<unsigned int>& getCellSizeArray()
			{
			return m_cell_size;
			}
		
		const GPUArray<unsigned int>& getCellAdjArray()
			{
			return m_cell_adj;
			}
		
		const GPUArray<Scalar4>& getXYZFArray()
			{
			return m_xyzf;
			}
		
		const GPUArray<Scalar4>& getTDBArray()
			{
			return m_tdb;
			}
			
		void setComputeTDB(bool compute_tdb)
			{
			m_compute_tdb = compute_tdb;
			}
		
		void setFlagCharge()
			{
			m_flag_charge = true;
			}
		
		void setFlagIndex()
			{
			m_flag_index = false;
			}
		
		void compute(unsigned int timestep);
		
	private:
		// user specified parameters
		Scalar m_nominal_width;			//!< Minimum width of cell in any direction
		unsigned int m_radius;			//!< Radius of adjacency bins to list
		unsigned int m_max_cells;		//!< Maximum number of cells to allocate
		bool m_compute_tdb;				//!< true if the tdb list should be computed
		bool m_flag_charge;				//!< true if the flag should be set to the charge, it will be index otherwise
		
		// parameters determined by initialize
		Scalar3 m_width;				//!< Actual width
		Index3D m_cell_indexer;			//!< Indexes cells from i,j,k
		Index2D m_cell_list_indexer;	//!< Indexes elements in the cell list
		Index2D m_cell_adj_indexer;		//!< Indexes elements in the cell adjacency list
		bool m_params_changed;			//!< Set to true when parameters are changed
		
		// values computed by Compute
		GPUArray<unsigned int> m_cell_size;	//!< Number of members in each cell
		GPUArray<unsigned int> m_cell_adj;	//!< Cell adjacency list
		GPUArray<Scalar4> m_xyzf;			//!< Cell list with position and flags
		GPUArray<Scalar4> m_tdb;			//!< Cell list with type,body,diameter
		
		void initialize();
	}

#endif
