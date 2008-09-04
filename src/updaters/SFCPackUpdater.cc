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

/*! \file SFCPackUpdater.cc
	\brief Defines the SFCPackUpdater class
*/

#ifdef WIN32
#pragma warning( push )
#pragma warning( disable : 4103 4244 )
#endif

#ifdef USE_PYTHON
#include <boost/python.hpp>
using namespace boost::python;
#endif

#include <math.h>
#include <stdexcept>

#include "SFCPackUpdater.h"

using namespace std;

/*! \param pdata Particle data to perform sorts on
	\param bin_width Maximum width of bins to place particles in
	\note bin_width will be dynamically decreased to reach a power of two grid size
 */
SFCPackUpdater::SFCPackUpdater(boost::shared_ptr<ParticleData> pdata, Scalar bin_width)
	: Updater(pdata), m_bin_width(bin_width), m_lastMmax(0)
	{
	// perform lots of sanity checks
	assert(m_pdata);
	
	if (m_bin_width < 0.01)
		{
		cerr << endl << "***Error! Bin width in SFCPackUpdater much too small" << endl << endl;
		throw runtime_error("Error initializing SFCPackUpdater");
		}

	m_sort_order.resize(m_pdata->getN());
	}
	
/*! Performs the sort. 
	\note The neighborlist is set to a forced recompute on the next call to compute
		overriding the normal rules when compute is called multiple times in a single 
		time step
	\note In an updater list, this sort should be done first, before anyone else
		gets ahold of the particle data
		
	\param timestep Current timestep of the simulation
 */	
void SFCPackUpdater::update(unsigned int timestep)
	{
	if (m_prof)
		m_prof->push("SFCPack");

	// figure out the sort order we need to apply
	getSortedOrder();
	// apply that sort order to the particles
	applySortOrder();

	if (m_prof)
		m_prof->pop();

	m_pdata->notifyParticleSort();
	}
	
void SFCPackUpdater::applySortOrder()
	{
	if (m_prof)
		m_prof->push("Apply");
	
	assert(m_pdata);
	assert(m_sort_order.size() == m_pdata->getN());
	const ParticleDataArrays& arrays = m_pdata->acquireReadWrite();
	
	// construct a temporary holding array for the sorted data
	Scalar *scal_tmp = new Scalar[m_pdata->getN()];
	
	// sort x
	for (unsigned int i = 0; i < arrays.nparticles; i++)
		scal_tmp[i] = arrays.x[m_sort_order[i]];
	for (unsigned int i = 0; i < arrays.nparticles; i++)
		arrays.x[i] = scal_tmp[i];

	// sort y
	for (unsigned int i = 0; i < arrays.nparticles; i++)
		scal_tmp[i] = arrays.y[m_sort_order[i]];
	for (unsigned int i = 0; i < arrays.nparticles; i++)
		arrays.y[i] = scal_tmp[i];
		
	// sort z
	for (unsigned int i = 0; i < arrays.nparticles; i++)
		scal_tmp[i] = arrays.z[m_sort_order[i]];
	for (unsigned int i = 0; i < arrays.nparticles; i++)
		arrays.z[i] = scal_tmp[i];		
		
	// sort vx
	for (unsigned int i = 0; i < arrays.nparticles; i++)
		scal_tmp[i] = arrays.vx[m_sort_order[i]];
	for (unsigned int i = 0; i < arrays.nparticles; i++)
		arrays.vx[i] = scal_tmp[i];

	// sort vy
	for (unsigned int i = 0; i < arrays.nparticles; i++)
		scal_tmp[i] = arrays.vy[m_sort_order[i]];
	for (unsigned int i = 0; i < arrays.nparticles; i++)
		arrays.vy[i] = scal_tmp[i];
		
	// sort vz
	for (unsigned int i = 0; i < arrays.nparticles; i++)
		scal_tmp[i] = arrays.vz[m_sort_order[i]];
	for (unsigned int i = 0; i < arrays.nparticles; i++)
		arrays.vz[i] = scal_tmp[i];
		
	// sort ax
	for (unsigned int i = 0; i < arrays.nparticles; i++)
		scal_tmp[i] = arrays.ax[m_sort_order[i]];
	for (unsigned int i = 0; i < arrays.nparticles; i++)
		arrays.ax[i] = scal_tmp[i];

	// sort ay
	for (unsigned int i = 0; i < arrays.nparticles; i++)
		scal_tmp[i] = arrays.ay[m_sort_order[i]];
	for (unsigned int i = 0; i < arrays.nparticles; i++)
		arrays.ay[i] = scal_tmp[i];
		
	// sort az
	for (unsigned int i = 0; i < arrays.nparticles; i++)
		scal_tmp[i] = arrays.az[m_sort_order[i]];
	for (unsigned int i = 0; i < arrays.nparticles; i++)
		arrays.az[i] = scal_tmp[i];

	// sort charge
	for (unsigned int i = 0; i < arrays.nparticles; i++)
		scal_tmp[i] = arrays.charge[m_sort_order[i]];
	for (unsigned int i = 0; i < arrays.nparticles; i++)
		arrays.charge[i] = scal_tmp[i];

	// sort type
	unsigned int *int_tmp = new unsigned int[m_pdata->getN()];
	for (unsigned int i = 0; i < arrays.nparticles; i++)
		int_tmp[i] = arrays.type[m_sort_order[i]];
	for (unsigned int i = 0; i < arrays.nparticles; i++)
		arrays.type[i] = int_tmp[i];

	// sort tag
	for (unsigned int i = 0; i < arrays.nparticles; i++)
		int_tmp[i] = arrays.tag[m_sort_order[i]];
	for (unsigned int i = 0; i < arrays.nparticles; i++)
		arrays.tag[i] = int_tmp[i];
		
	// rebuild rtag
	for (unsigned int i = 0; i < arrays.nparticles; i++)
		arrays.rtag[arrays.tag[i]] = i;	
	
	delete[] scal_tmp;
	delete[] int_tmp;
	
	m_pdata->release();

	if (m_prof)
		m_prof->pop();
	}
	
// this function will generate a Z-order traversal through the 3d array Mx by Mx x Mx
// it is done recursively through an octree subdivision (supporting power-of-2 cubic dimensions only for simplicity at this stage)
// w on the first call should be equal to Mx
// i,j,k on the first call should be 0

// as it recurses down, w will be decreased appropriately
// traversal_order will be filled out with unique values i*Mx*Mx + j*Mx + k for the coordinates
// i,j,k in the Z-order
/*static void generateTraversalOrder(int i, int j, int k, int w, int Mx, vector< unsigned int > &traversal_order)
	{
	if (w == 1)
		{
		// handle base case
		traversal_order.push_back(i*Mx*Mx + j*Mx + k);
		}
	else
		{
		// handle arbitrary case, split the box into 8 sub boxes
		w = w / 2;
		generateTraversalOrder(i,j,k,w,Mx,traversal_order);
		generateTraversalOrder(i,j,k+w,w,Mx,traversal_order);
		generateTraversalOrder(i,j+w,k,w,Mx,traversal_order);
		generateTraversalOrder(i,j+w,k+w,w,Mx,traversal_order);

		generateTraversalOrder(i+w,j,k,w,Mx,traversal_order);
		generateTraversalOrder(i+w,j,k+w,w,Mx,traversal_order);
		generateTraversalOrder(i+w,j+w,k,w,Mx,traversal_order);
		generateTraversalOrder(i+w,j+w,k+w,w,Mx,traversal_order);
		}
	}*/

//! x walking table for the hilbert curve
static int istep[] = {0, 0, 0, 0, 1, 1, 1, 1};
//! y walking table for the hilbert curve
static int jstep[] = {0, 0, 1, 1, 1, 1, 0, 0};
//! z walking table for the hilbert curve
static int kstep[] = {0, 1, 1, 0, 0, 1, 1, 0};


//! Helper function for recursive hilbert curve generation
/*! \param in Input sequence
	\returns output sequence permuted by rule 1
*/	
static vector<unsigned int> permute1(const vector<unsigned int>& in)
	{
	assert(in.size() == 8);
	
	vector<unsigned int> result;
	result.push_back(in[0]);
	result.push_back(in[3]);
	result.push_back(in[4]);
	result.push_back(in[7]);
	result.push_back(in[6]);
	result.push_back(in[5]);
	result.push_back(in[2]);
	result.push_back(in[1]);
	
	return result;
	}

//! Helper function for recursive hilbert curve generation
/*! \param in Input sequence
	\returns output sequence permuted by rule 2
*/
static vector<unsigned int> permute2(const vector<unsigned int>& in)
	{
	assert(in.size() == 8);
	
	vector<unsigned int> result;
	result.push_back(in[0]);
	result.push_back(in[7]);
	result.push_back(in[6]);
	result.push_back(in[1]);
	result.push_back(in[2]);
	result.push_back(in[5]);
	result.push_back(in[4]);
	result.push_back(in[3]);
	
	return result;
	}

//! Helper function for recursive hilbert curve generation
/*! \param in Input sequence
	\returns output sequence permuted by rule 3
*/
static vector<unsigned int> permute3(const vector<unsigned int>& in)
	{
	return permute2(in);
	}

//! Helper function for recursive hilbert curve generation
/*! \param in Input sequence
	\returns output sequence permuted by rule 4
*/
static vector<unsigned int> permute4(const vector<unsigned int>& in)
	{
	assert(in.size() == 8);
	
	vector<unsigned int> result;
	result.push_back(in[2]);
	result.push_back(in[3]);
	result.push_back(in[0]);
	result.push_back(in[1]);
	result.push_back(in[6]);
	result.push_back(in[7]);
	result.push_back(in[4]);
	result.push_back(in[5]);
	
	return result;
	}

//! Helper function for recursive hilbert curve generation
/*! \param in Input sequence
	\returns output sequence permuted by rule 5
*/
static vector<unsigned int> permute5(const vector<unsigned int>& in)
	{
	return permute4(in);
	}

//! Helper function for recursive hilbert curve generation
/*! \param in Input sequence
	\returns output sequence permuted by rule 6
*/
static vector<unsigned int> permute6(const vector<unsigned int>& in)
	{
	assert(in.size() == 8);
	
	vector<unsigned int> result;
	result.push_back(in[4]);
	result.push_back(in[3]);
	result.push_back(in[2]);
	result.push_back(in[5]);
	result.push_back(in[6]);
	result.push_back(in[1]);
	result.push_back(in[0]);
	result.push_back(in[7]);
	
	return result;
	}

//! Helper function for recursive hilbert curve generation
/*! \param in Input sequence
	\returns output sequence permuted by rule 7
*/
static vector<unsigned int> permute7(const vector<unsigned int>& in)
	{
	return permute6(in);
	}

//! Helper function for recursive hilbert curve generation
/*! \param in Input sequence
	\returns output sequence permuted by rule 8
*/
static vector<unsigned int> permute8(const vector<unsigned int>& in)
	{
	assert(in.size() == 8);
	
	vector<unsigned int> result;
	result.push_back(in[6]);
	result.push_back(in[5]);
	result.push_back(in[2]);
	result.push_back(in[1]);
	result.push_back(in[0]);
	result.push_back(in[3]);
	result.push_back(in[4]);
	result.push_back(in[7]);
	
	return result;
	}

//! Helper function for recursive hilbert curve generation
/*! \param in Input sequence
	\param p permutation rule to apply
	\returns output sequence permuted by rule \a p
*/
static vector<unsigned int> permute(int p, const vector<unsigned int>& in)
	{
	switch (p)
		{
		case 0:
			return permute1(in);
			break;
		case 1:
			return permute2(in);
			break;
		case 2:
			return permute3(in);
			break;
		case 3:
			return permute4(in);
			break;
		case 4:
			return permute5(in);
			break;
		case 5:
			return permute6(in);
			break;
		case 6:
			return permute7(in);
			break;
		case 7:
			return permute8(in);
			break;
		default:	
	  		assert(false);
			return vector<unsigned int>();
		}
	}

//! recursive function for generating hilbert curve traversal order
/*! \param i Current x coordinate in grid
	\param j Current y coordinate in grid
	\param k Current z coordinate in grid
	\param w Number of grid cells wide at the current recursion level
	\param Mx Width of the entire grid (it is cubic, same width in all 3 directions)
	\param cell_order Current permutation order to traverse cells along
	\param travseral_order Traversal order to build up
	\pre \a traversal_order.size() == 0
	\pre Initial call should be with \a i = \a j = \a k = 0, \a w = \w Mx, \a cell_order = (0,1,2,3,4,5,6,7,8) 
	\post traversal order contains the grid index (i*Mx*Mx + j*Mx + k) of each grid point
		listed in the order of the hilbert curve
*/
static void generateTraversalOrder(int i, int j, int k, int w, int Mx, vector<unsigned int> cell_order, vector< unsigned int > &traversal_order)
	{
	if (w == 1)
		{
		// handle base case
		traversal_order.push_back(i*Mx*Mx + j*Mx + k);
		}
	else
		{
		// handle arbitrary case, split the box into 8 sub boxes
		w = w / 2;
		
		// we ned to handle each sub box in the order defined by cell order
		assert(cell_order.size() == 8);
		for (int m = 0;	m < 8; m++)
			{
			unsigned int cur_cell = cell_order[m];
			int ic = i + w * istep[cur_cell];
			int jc = j + w * jstep[cur_cell];
			int kc = k + w * kstep[cur_cell];
			
			vector<unsigned int> child_cell_order = permute(m, cell_order);
			generateTraversalOrder(ic,jc,kc,w,Mx, child_cell_order, traversal_order);
			}
		}
	}

void SFCPackUpdater::getSortedOrder()
	{
	if (m_prof)
		m_prof->push("Sort");
	
	// start by checking the saneness of some member variables
	assert(m_pdata);
	assert(m_sort_order.size() == m_pdata->getN());
	
	// calculate the bin dimensions
	const BoxDim& box = m_pdata->getBox();
	assert(box.xhi > box.xlo && box.yhi > box.ylo && box.zhi > box.zlo);	
	unsigned int Mx = (unsigned int)((box.xhi - box.xlo) / (m_bin_width));
	unsigned int My = (unsigned int)((box.yhi - box.ylo) / (m_bin_width));
	unsigned int Mz = (unsigned int)((box.zhi - box.zlo) / (m_bin_width));
	if (Mx == 0)
		Mx = 1;
	if (My == 0)
		My = 1;
	if (Mz == 0)
		Mz = 1;
		
	// now, we need to play a game here: the octtree traversal only works with cubes
	// and only works if Mx is a power of 2. So, choose the largest of the 3 dimesions and 
	// make them all the same
	unsigned int Mmax = Mx;
	if (My > Mmax)
		Mmax = My;
	if (Mz > Mmax)
		Mz = Mmax;
	
	// round up to the nearest power of 2
	Mmax = (unsigned int)pow(2.0, ceil(log(double(Mmax)) / log(2.0)));

	// make even bin dimensions
	Scalar binx = (box.xhi - box.xlo) / Scalar(Mmax);
	Scalar biny = (box.yhi - box.ylo) / Scalar(Mmax);
	Scalar binz = (box.zhi - box.zlo) / Scalar(Mmax);

	// precompute scale factors to eliminate division in inner loop
	Scalar scalex = Scalar(1.0) / binx;
	Scalar scaley = Scalar(1.0) / biny;
	Scalar scalez = Scalar(1.0) / binz;

	// reallocate memory arrays if Mmax changed
	// also regenerate the traversal order
	if (m_lastMmax != Mmax)
		{
		if (m_prof)
			{
			m_prof->pop();
			m_prof->push("Curve gen");
			}

		m_bins.resize(Mmax*Mmax*Mmax);

		// generate the traversal order
		m_traversal_order.resize(Mmax*Mmax*Mmax);
		m_traversal_order.clear();

		// we need to start the hilbert curve with a seed order 0,1,2,3,4,5,6,7
		vector<unsigned int> cell_order(8);
		for (unsigned int i = 0; i < 8; i++)
			cell_order[i] = i;	
		generateTraversalOrder(0,0,0, Mmax, Mmax, cell_order, m_traversal_order);

		m_lastMmax = Mmax;

		if (m_prof)
			{
			m_prof->pop();
			m_prof->push("Sort");
			}
		}
		
	// sanity checks
	assert(m_bins.size() == Mmax*Mmax*Mmax);
	assert(m_traversal_order.size() == Mmax*Mmax*Mmax);

	// need to clear the bins
	for (unsigned int bin = 0; bin < Mmax*Mmax*Mmax; bin++)
		m_bins[bin].clear();

	// put the particles in the bins
	ParticleDataArraysConst arrays = m_pdata->acquireReadOnly();
	// for each particle
	for (unsigned int n = 0; n < arrays.nparticles; n++)
		{
		// find the bin each particle belongs in
		unsigned int ib = (unsigned int)((arrays.x[n]-box.xlo)*scalex);
		unsigned int jb = (unsigned int)((arrays.y[n]-box.ylo)*scaley);
		unsigned int kb = (unsigned int)((arrays.z[n]-box.zlo)*scalez);

		// need to handle the case where the particle is exactly at the box hi
		if (ib == Mmax)
			ib = 0;
		if (jb == Mmax)
			jb = 0;
		if (kb == Mmax)
			kb = 0;

		// sanity check
		assert(ib < (unsigned int)(Mmax) && jb < (unsigned int)(Mmax) && kb < (unsigned int)(Mmax));

		// record its bin
		unsigned int bin = ib*(Mmax*Mmax) + jb * Mmax + kb;

		m_bins[bin].push_back(n);
		}
	m_pdata->release();
		
	// now, loop through the bins and produce the sort order
	int cur = 0;
	
	for (unsigned int i = 0; i < Mmax*Mmax*Mmax; i++)
		{
		unsigned int bin = m_traversal_order[i];
				
		for (unsigned int j = 0; j < m_bins[bin].size(); j++)
			{
			m_sort_order[cur++] = m_bins[bin][j];
			}
		}
	
	if (m_prof)
		m_prof->pop();
	}

#ifdef USE_PYTHON
void export_SFCPackUpdater()
	{
	class_<SFCPackUpdater, boost::shared_ptr<SFCPackUpdater>, bases<Updater>, boost::noncopyable>
		("SFCPackUpdater", init< boost::shared_ptr<ParticleData>, Scalar >())
		.def("setBinWidth", &SFCPackUpdater::setBinWidth)
		;
	}
#endif

#ifdef WIN32
#pragma warning( pop )
#endif

