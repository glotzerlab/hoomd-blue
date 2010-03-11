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

/*! \file SFCPackUpdater.cc
    \brief Defines the SFCPackUpdater class
*/

#ifdef WIN32
#pragma warning( push )
#pragma warning( disable : 4103 4244 )
#endif

#include <boost/python.hpp>
using namespace boost::python;

#include <math.h>
#include <stdexcept>
#include <algorithm>

#include "SFCPackUpdater.h"

using namespace std;

/*! \param sysdef System to perform sorts on
 */
SFCPackUpdater::SFCPackUpdater(boost::shared_ptr<SystemDefinition> sysdef)
        : Updater(sysdef), m_last_grid(0), m_last_dim(0)
    {
    // perform lots of sanity checks
    assert(m_pdata);
    
    m_sort_order.resize(m_pdata->getN());
    m_particle_bins.resize(m_pdata->getN());
    
    // set the default grid
    // Grid dimension must always be a power of 2 and determines the memory usage for m_traversal_order
    // To prevent massive overruns of the memory, always use 256 for 3d and 4096 for 2d
    if (m_sysdef->getNDimensions() == 2)
        m_grid = 4096;
    else
        m_grid = 256;
    }

/*! Performs the sort.
    \note In an updater list, this sort should be done first, before anyone else
    gets ahold of the particle data

    \param timestep Current timestep of the simulation
 */
void SFCPackUpdater::update(unsigned int timestep)
    {
    if (m_prof) m_prof->push("SFCPack");
    
    // figure out the sort order we need to apply
    if (m_sysdef->getNDimensions() == 2)
        getSortedOrder2D();
    else
        getSortedOrder3D();
    
    // apply that sort order to the particles
    applySortOrder();
    
    if (m_prof) m_prof->pop();
    
    m_pdata->notifyParticleSort();
    }

void SFCPackUpdater::applySortOrder()
    {
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
        
    // sort mass
    for (unsigned int i = 0; i < arrays.nparticles; i++)
        scal_tmp[i] = arrays.mass[m_sort_order[i]];
    for (unsigned int i = 0; i < arrays.nparticles; i++)
        arrays.mass[i] = scal_tmp[i];
        
    // sort diameter
    for (unsigned int i = 0; i < arrays.nparticles; i++)
        scal_tmp[i] = arrays.diameter[m_sort_order[i]];
    for (unsigned int i = 0; i < arrays.nparticles; i++)
        arrays.diameter[i] = scal_tmp[i];
        
    // sort ix
    int *int_tmp = new int[m_pdata->getN()];
    for (unsigned int i = 0; i < arrays.nparticles; i++)
        int_tmp[i] = arrays.ix[m_sort_order[i]];
    for (unsigned int i = 0; i < arrays.nparticles; i++)
        arrays.ix[i] = int_tmp[i];
        
    for (unsigned int i = 0; i < arrays.nparticles; i++)
        int_tmp[i] = arrays.iy[m_sort_order[i]];
    for (unsigned int i = 0; i < arrays.nparticles; i++)
        arrays.iy[i] = int_tmp[i];
        
    for (unsigned int i = 0; i < arrays.nparticles; i++)
        int_tmp[i] = arrays.iz[m_sort_order[i]];
    for (unsigned int i = 0; i < arrays.nparticles; i++)
        arrays.iz[i] = int_tmp[i];
        
    // sort type
    unsigned int *uint_tmp = new unsigned int[m_pdata->getN()];
    for (unsigned int i = 0; i < arrays.nparticles; i++)
        uint_tmp[i] = arrays.type[m_sort_order[i]];
    for (unsigned int i = 0; i < arrays.nparticles; i++)
        arrays.type[i] = uint_tmp[i];
        
    // sort tag
    for (unsigned int i = 0; i < arrays.nparticles; i++)
        uint_tmp[i] = arrays.tag[m_sort_order[i]];
    for (unsigned int i = 0; i < arrays.nparticles; i++)
        arrays.tag[i] = uint_tmp[i];
        
    // rebuild rtag
    for (unsigned int i = 0; i < arrays.nparticles; i++)
        arrays.rtag[arrays.tag[i]] = i;
        
    delete[] scal_tmp;
    delete[] uint_tmp;
    delete[] int_tmp;
    
    m_pdata->release();
    }

//! x walking table for the hilbert curve
static int istep[] = {0, 0, 0, 0, 1, 1, 1, 1};
//! y walking table for the hilbert curve
static int jstep[] = {0, 0, 1, 1, 1, 1, 0, 0};
//! z walking table for the hilbert curve
static int kstep[] = {0, 1, 1, 0, 0, 1, 1, 0};


//! Helper function for recursive hilbert curve generation
/*! \param result Output sequence to be permuted by rule 1
    \param in Input sequence
*/
static void permute1(unsigned int result[8], const unsigned int in[8])
    {
    result[0] = in[0];
    result[1] = in[3];
    result[2] = in[4];
    result[3] = in[7];
    result[4] = in[6];
    result[5] = in[5];
    result[6] = in[2];
    result[7] = in[1];
    }

//! Helper function for recursive hilbert curve generation
/*! \param result Output sequence to be permuted by rule 2
    \param in Input sequence
*/
static void permute2(unsigned int result[8], const unsigned int in[8])
    {
    result[0] = in[0];
    result[1] = in[7];
    result[2] = in[6];
    result[3] = in[1];
    result[4] = in[2];
    result[5] = in[5];
    result[6] = in[4];
    result[7] = in[3];
    }

//! Helper function for recursive hilbert curve generation
/*! \param result Output sequence to be permuted by rule 3
    \param in Input sequence
*/
static void permute3(unsigned int result[8], const unsigned int in[8])
    {
    permute2(result, in);
    }

//! Helper function for recursive hilbert curve generation
/*! \param result Output sequence to be permuted by rule 4
    \param in Input sequence
*/
static void permute4(unsigned int result[8], const unsigned int in[8])
    {
    result[0] = in[2];
    result[1] = in[3];
    result[2] = in[0];
    result[3] = in[1];
    result[4] = in[6];
    result[5] = in[7];
    result[6] = in[4];
    result[7] = in[5];
    }

//! Helper function for recursive hilbert curve generation
/*! \param result Output sequence to be permuted by rule 5
    \param in Input sequence
*/
static void permute5(unsigned int result[8], const unsigned int in[8])
    {
    permute4(result, in);
    }

//! Helper function for recursive hilbert curve generation
/*! \param result Output sequence to be permuted by rule 6
    \param in Input sequence
*/
static void permute6(unsigned int result[8], const unsigned int in[8])
    {
    result[0] = in[4];
    result[1] = in[3];
    result[2] = in[2];
    result[3] = in[5];
    result[4] = in[6];
    result[5] = in[1];
    result[6] = in[0];
    result[7] = in[7];
    }

//! Helper function for recursive hilbert curve generation
/*! \param result Output sequence to be permuted by rule 7
    \param in Input sequence
*/
static void permute7(unsigned int result[8], const unsigned int in[8])
    {
    permute6(result, in);
    }

//! Helper function for recursive hilbert curve generation
/*! \param result Output sequence to be permuted by rule 8
    \param in Input sequence
*/
static void permute8(unsigned int result[8], const unsigned int in[8])
    {
    result[0] = in[6];
    result[1] = in[5];
    result[2] = in[2];
    result[3] = in[1];
    result[4] = in[0];
    result[5] = in[3];
    result[6] = in[4];
    result[7] = in[7];
    }

//! Helper function for recursive hilbert curve generation
/*! \param result Output sequence to be permuted by rule \a p-1
    \param in Input sequence
    \param p permutation rule to apply
*/
void permute(unsigned int result[8], const unsigned int in[8], int p)
    {
    switch (p)
        {
        case 0:
            permute1(result, in);
            break;
        case 1:
            permute2(result, in);
            break;
        case 2:
            permute3(result, in);
            break;
        case 3:
            permute4(result, in);
            break;
        case 4:
            permute5(result, in);
            break;
        case 5:
            permute6(result, in);
            break;
        case 6:
            permute7(result, in);
            break;
        case 7:
            permute8(result, in);
            break;
        default:
            assert(false);
        }
    }

//! recursive function for generating hilbert curve traversal order
/*! \param i Current x coordinate in grid
    \param j Current y coordinate in grid
    \param k Current z coordinate in grid
    \param w Number of grid cells wide at the current recursion level
    \param Mx Width of the entire grid (it is cubic, same width in all 3 directions)
    \param cell_order Current permutation order to traverse cells along
    \param traversal_order Traversal order to build up
    \pre \a traversal_order.size() == 0
    \pre Initial call should be with \a i = \a j = \a k = 0, \a w = \a Mx, \a cell_order = (0,1,2,3,4,5,6,7,8)
    \post traversal order contains the grid index (i*Mx*Mx + j*Mx + k) of each grid point
        listed in the order of the hilbert curve
*/
static void generateTraversalOrder(int i, int j, int k, int w, int Mx, unsigned int cell_order[8], vector< unsigned int > &traversal_order)
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
        for (int m = 0; m < 8; m++)
            {
            unsigned int cur_cell = cell_order[m];
            int ic = i + w * istep[cur_cell];
            int jc = j + w * jstep[cur_cell];
            int kc = k + w * kstep[cur_cell];
            
            unsigned int child_cell_order[8];
            permute(child_cell_order, cell_order, m);
            generateTraversalOrder(ic,jc,kc,w,Mx, child_cell_order, traversal_order);
            }
        }
    }

void SFCPackUpdater::getSortedOrder2D()
    {
    // start by checking the saneness of some member variables
    assert(m_pdata);
    assert(m_sort_order.size() == m_pdata->getN());
    
    // make even bin dimensions
    const BoxDim& box = m_pdata->getBox();
    assert(box.xhi > box.xlo && box.yhi > box.ylo && box.zhi > box.zlo);
    Scalar binx = (box.xhi - box.xlo) / Scalar(m_grid);
    Scalar biny = (box.yhi - box.ylo) / Scalar(m_grid);
    
    // precompute scale factors to eliminate division in inner loop
    Scalar scalex = Scalar(1.0) / binx;
    Scalar scaley = Scalar(1.0) / biny;
    
    // put the particles in the bins
    ParticleDataArraysConst arrays = m_pdata->acquireReadOnly();
    // for each particle
    for (unsigned int n = 0; n < arrays.nparticles; n++)
        {
        // find the bin each particle belongs in
        unsigned int ib = (unsigned int)((arrays.x[n]-box.xlo)*scalex) % m_grid;
        unsigned int jb = (unsigned int)((arrays.y[n]-box.ylo)*scaley) % m_grid;
        
        // record its bin
        unsigned int bin = ib*m_grid + jb;
        
        m_particle_bins[n] = std::pair<unsigned int, unsigned int>(bin, n);
        }
    m_pdata->release();
    
    // sort the tuples
    sort(m_particle_bins.begin(), m_particle_bins.end());
    
    // translate the sorted order
    for (unsigned int j = 0; j < m_pdata->getN(); j++)
        {
        m_sort_order[j] = m_particle_bins[j].second;
        }
    }

void SFCPackUpdater::getSortedOrder3D()
    {
    // start by checking the saneness of some member variables
    assert(m_pdata);
    assert(m_sort_order.size() == m_pdata->getN());
    
    // make even bin dimensions
    const BoxDim& box = m_pdata->getBox();
    assert(box.xhi > box.xlo && box.yhi > box.ylo && box.zhi > box.zlo);
    Scalar binx = (box.xhi - box.xlo) / Scalar(m_grid);
    Scalar biny = (box.yhi - box.ylo) / Scalar(m_grid);
    Scalar binz = (box.zhi - box.zlo) / Scalar(m_grid);
    
    // precompute scale factors to eliminate division in inner loop
    Scalar scalex = Scalar(1.0) / binx;
    Scalar scaley = Scalar(1.0) / biny;
    Scalar scalez = Scalar(1.0) / binz;
    
    // reallocate memory arrays if m_grid changed
    // also regenerate the traversal order
    if (m_last_grid != m_grid || m_last_dim != 3)
        {
        if (m_grid > 256)
            {
            unsigned int mb = m_grid*m_grid*m_grid*4 / 1024 / 1024;
            cout << endl;
            cout << "***Warning! sorter is about to allocate a very large amount of memory (" << mb << "MB)"
                 << " and may crash." << endl;
            cout << "            Reduce the amount of memory allocated to prevent this by decreasing the " << endl;
            cout << "            grid dimension (i.e. sorter.set_params(grid=128) ) or by disabling it " << endl;
            cout << "            ( sorter.disable() ) before beginning the run()." << endl << endl;
            }

        // generate the traversal order
        m_traversal_order.resize(m_grid*m_grid*m_grid);
        m_traversal_order.clear();
        vector< unsigned int > reverse_order(m_grid*m_grid*m_grid);
        reverse_order.clear();
        
        // we need to start the hilbert curve with a seed order 0,1,2,3,4,5,6,7
        unsigned int cell_order[8];
        for (unsigned int i = 0; i < 8; i++)
            cell_order[i] = i;
        generateTraversalOrder(0,0,0, m_grid, m_grid, cell_order, reverse_order);
        
        for (unsigned int i = 0; i < m_grid*m_grid*m_grid; i++)
            m_traversal_order[reverse_order[i]] = i;
        
        m_last_grid = m_grid;
        // store the last system dimension computed so we can be mindful if that ever changes
        m_last_dim = m_sysdef->getNDimensions();
        }
        
    // sanity checks
    assert(m_particle_bins.size() == m_pdata->getN());
    assert(m_traversal_order.size() == m_grid*m_grid*m_grid);
    
    // put the particles in the bins
    ParticleDataArraysConst arrays = m_pdata->acquireReadOnly();
    // for each particle
    for (unsigned int n = 0; n < arrays.nparticles; n++)
        {
        // find the bin each particle belongs in
        unsigned int ib = (unsigned int)((arrays.x[n]-box.xlo)*scalex) % m_grid;
        unsigned int jb = (unsigned int)((arrays.y[n]-box.ylo)*scaley) % m_grid;
        unsigned int kb = (unsigned int)((arrays.z[n]-box.zlo)*scalez) % m_grid;
        
        // record its bin
        unsigned int bin = ib*(m_grid*m_grid) + jb * m_grid + kb;
        
        m_particle_bins[n] = std::pair<unsigned int, unsigned int>(m_traversal_order[bin], n);
        }
    m_pdata->release();
    
    // sort the tuples
    sort(m_particle_bins.begin(), m_particle_bins.end());
    
    // translate the sorted order
    for (unsigned int j = 0; j < m_pdata->getN(); j++)
        {
        m_sort_order[j] = m_particle_bins[j].second;
        }
    }

void export_SFCPackUpdater()
    {
    class_<SFCPackUpdater, boost::shared_ptr<SFCPackUpdater>, bases<Updater>, boost::noncopyable>
    ("SFCPackUpdater", init< boost::shared_ptr<SystemDefinition> >())
    .def("setGrid", &SFCPackUpdater::setGrid)
    ;
    }

#ifdef WIN32
#pragma warning( pop )
#endif

