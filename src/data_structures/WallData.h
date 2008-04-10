/*
Highly Optimized Object-Oriented Molecular Dynamics (HOOMD) Open
Source Software License
Copyright (c) 2008 Ames Laboratory Iowa State University
All rights reserved.

Redistribution and use of HOOMD, in source and binary forms, with or
without modification, are permitted, provided that the following
conditions are met:

* Redistributions of source code must retain the above copyright notice,
this list of conditions and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright
notice, this list of conditions and the following disclaimer in the
documentation and/or other materials provided with the distribution.

* Neither the name of the copyright holder nor the names HOOMD's
contributors may be used to endorse or promote products derived from this
software without specific prior written permission.

Disclaimer

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDER AND
CONTRIBUTORS ``AS IS''  AND ANY EXPRESS OR IMPLIED WARRANTIES,
INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY
AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. 

IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS  BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF
THE POSSIBILITY OF SUCH DAMAGE.
*/

// $Id$
// $URL$

/*! \file ParticleData.h
 	\brief Contains declarations for WallData.
 */

//! 
struct WallDataArrays 
	{
	//! The x coordinate for the origin of the planes
	vector<Scalar> ox;
	//! The y coordinate for the origin of the planes
	vector<Scalar> oy;
	//! The z coordinate for the origin of the planes
	vector<Scalar> oz;
				
	//! The x direction for the normal of the planes
	vector<Scalar> nx;
	//! The y direction for the normal of the planes
	vector<Scalar> ny;
	//! The z direction for the normal of the planes
	vector<Scalar> nz;
				
	//! constructs an empty structure
	WallDataArrays();
	};


//! Class for storing information about wall forces in simulation
/*!

	\ingroup data_structs
*/
class WallData {

	public:
		//! Creates useless walls
		WallData() : m_walls() {}
		//! Creates walls surrounding a box
		/*!	Walls are created on all six sides of the box.
			The offset parameter moves the walls into the 
			box to prevent edge cases from causing headaches.

			The default of 0.3 was chosen because it is half the size of a blue molecule
			in a simulation.

			
		*/
		WallData(BoxDim box, Scalar offset = 0.3) : m_walls(box, offset) {}
		//! Destructor, yay!
		~WallData() {}
		//! Get the struct containing all of the data about the walls
		WallDataArrays getWallArrays() { return m_walls; }
		//! Returns the number of walls contained in the walldata
		unsigned int getNumWalls() { return m_walls.numWalls; }
		//! Adds a wall to a simulation.
		/*!	This is mostly for debugging purposes
		*/
		void addWall(Scalar ox_p, Scalar oy_p, Scalar oz_p, Scalar nx_p, Scalar ny_p, Scalar nz_p);
	private:
		WallDataArrays m_walls;
};
