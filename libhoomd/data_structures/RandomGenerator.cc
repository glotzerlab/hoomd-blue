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

* All publications based on HOOMD-blue, including any reports or published
results obtained, in whole or in part, with HOOMD-blue, will acknowledge its use
according to the terms posted at the time of submission on:
http://codeblue.umich.edu/hoomd-blue/citations.html

* Any electronic documents citing HOOMD-Blue will link to the HOOMD-Blue website
at: http://codeblue.umich.edu/hoomd-blue/.

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
#pragma warning( disable : 4103 4244 4267)
#endif

#include "RandomGenerator.h"

#include <cassert>
#include <stdexcept>

#include <boost/python.hpp>
#include <boost/python/suite/indexing/vector_indexing_suite.hpp>
using namespace boost::python;

using namespace std;

// windows defines a macro min and max
#undef min
#undef max

/*! \file RandomGenerator.cc
    \brief Contains definitions for RandomGenerator and related classes.
 */

/*! \param n_particles Number of particles that will be generated
    \param box Box the particles are generated in
    \param radii Mapping of particle types to their minimum separation radius

    After construction, all data structure are set to defaults and particles are ready to be placed.
*/
GeneratedParticles::GeneratedParticles(unsigned int n_particles,
                                       const BoxDim& box,
                                       const std::map< std::string, Scalar >& radii) 
    : m_particles(n_particles), m_box(box), m_radii(radii)
    {
    // sanity checks
    assert(n_particles > 0);
    assert(box.xhi > box.xlo && box.yhi > box.ylo && box.zhi > box.zlo);
    assert(m_radii.size() > 0);
    
    // find the maximum particle radius
    Scalar max_radius = Scalar(0.0);
    map<string, Scalar>::iterator itr;
    for (itr = m_radii.begin(); itr != m_radii.end(); ++itr)
        {
        Scalar r = itr->second;
        if (r > max_radius)
            max_radius=r;
        }
        
    // target a bin size of 7.0 * max_radius
    // the requirement is really only 2, 7 is to save memory
    Scalar target_size = Scalar(7.0)*max_radius;
    // cap the size to a minimum of 2 to prevent small sizes from blowing up the memory uszge
    if (target_size < Scalar(2.0))
        target_size = Scalar(2.0);
    
    // calculate the particle binning
    m_Mx = (int)((m_box.xhi - m_box.xlo) / (target_size));
    m_My = (int)((m_box.yhi - m_box.ylo) / (target_size));
    m_Mz = (int)((m_box.zhi - m_box.zlo) / (target_size));
    if (m_Mx == 0)
        m_Mx = 1;
    if (m_My == 0)
        m_My = 1;
    if (m_Mz == 0)
        m_Mz = 1;

    if (m_Mx > 100 || m_My > 100 || m_Mz > 100)
        {
        cout << "***Warning! random generator is about to allocate a very large amount of memory and may crash." << endl << endl;
        }
        
    // make even bin dimensions
    Scalar binx = (m_box.xhi - m_box.xlo) / Scalar(m_Mx);
    Scalar biny = (m_box.yhi - m_box.ylo) / Scalar(m_My);
    Scalar binz = (m_box.zhi - m_box.zlo) / Scalar(m_Mz);
    
    // precompute scale factors to eliminate division in inner loop
    m_scalex = Scalar(1.0) / binx;
    m_scaley = Scalar(1.0) / biny;
    m_scalez = Scalar(1.0) / binz;
    
    // setup the memory arrays
    m_bins.resize(m_Mx*m_My*m_Mz);
    }

/*! \param p Particle under consideration
    \returns true If \a p will not overlap any existing particles
    The position of \a p is checked against all nearby particles that have already been placed with place().
    If all distances are greater than the radius of p plus the radius of the compared particle, true is
    returned. If there is any overlap, false is returned.
*/
bool GeneratedParticles::canPlace(const particle& p)
    {
    // begin with an error check that p.type is actually in the radius map
    if (m_radii.count(p.type) == 0)
        {
        cerr << endl << "***Error! Radius not set for particle in RandomGenerator" << endl << endl;
        throw runtime_error("Error placing particle");
        }
        
    // first, map the particle back into the box
    Scalar x = p.x;
    Scalar lx = m_box.xhi - m_box.xlo;
    if (x > m_box.xhi)
        x -= lx*(((int)((x-m_box.xhi)/lx))+1);
    else if (x < m_box.xlo)
        x += lx*(((int)((m_box.xlo-x)/lx))+1);
        
    Scalar y = p.y;
    Scalar ly = m_box.yhi - m_box.ylo;
    if (y > m_box.yhi)
        y -= ly*(((int)((y-m_box.yhi)/ly))+1);
    else if (y < m_box.ylo)
        y += ly*(((int)((m_box.ylo-y)/ly))+1);

    Scalar z = p.z;
    Scalar lz = m_box.zhi - m_box.zlo;
    if (z > m_box.zhi)
        z -= lz*(((int)((z-m_box.zhi)/lz))+1);
    else if (z < m_box.zlo)
        z += lz*(((int)((m_box.zlo-z)/lz))+1);
        
    // determine the bin the particle is in
    int ib = (int)((x-m_box.xlo)*m_scalex);
    int jb = (int)((y-m_box.ylo)*m_scaley);
    int kb = (int)((z-m_box.zlo)*m_scalez);
    
    // need to handle the case where the particle is exactly at the box hi
    if (ib == m_Mx)
        ib = 0;
    if (jb == m_My)
        jb = 0;
    if (kb == m_Mz)
        kb = 0;
        
    // sanity check
    assert(0<= ib && ib < m_Mx && 0 <= jb && jb < m_My && 0<=kb && kb < m_Mz);
    
    // loop over all neighboring bins in (cur_ib, cur_jb, cur_kb)
    for (int cur_ib = ib - 1; cur_ib <= ib+1; cur_ib++)
        {
        for (int cur_jb = jb - 1; cur_jb <= jb+1; cur_jb++)
            {
            for (int cur_kb = kb - 1; cur_kb <= kb + 1; cur_kb++)
                {
                // generate box-wrapped bin coordinates in (cmp_ib, cmp_jb, cmp_kb), cmp is for compare
                int cmp_ib = cur_ib;
                if (cmp_ib < 0)
                    cmp_ib += m_Mx;
                if (cmp_ib >= m_Mx)
                    cmp_ib -= m_Mx;
                    
                int cmp_jb = cur_jb;
                if (cmp_jb < 0)
                    cmp_jb += m_My;
                if (cmp_jb >= m_My)
                    cmp_jb -= m_My;
                    
                int cmp_kb = cur_kb;
                if (cmp_kb < 0)
                    cmp_kb += m_Mz;
                if (cmp_kb >= m_Mz)
                    cmp_kb -= m_Mz;

                assert(cmp_ib >= 0 && cmp_ib < m_Mx && cmp_jb >=0 && cmp_jb < m_My && cmp_kb >= 0 && cmp_kb < m_Mz);
                int cmp_bin = cmp_ib*(m_My*m_Mz) + cmp_jb * m_Mz + cmp_kb;
                
                // check all particles in that bin
                const vector<unsigned int> &bin_list = m_bins[cmp_bin];
                for (unsigned int i = 0; i < bin_list.size(); i++)
                    {
                    // compare particles
                    const particle& p_cmp = m_particles[bin_list[i]];
                    
                    Scalar min_dist = m_radii[p.type] + m_radii[p_cmp.type];

                    // map p_cmp into box
                    Scalar cmp_x = p_cmp.x;
                    if (cmp_x > m_box.xhi)
                        cmp_x -= lx*(((int)((cmp_x-m_box.xhi)/lx))+1);
                    else if (x < m_box.xlo)
                        cmp_x += lx*(((int)((m_box.xlo-cmp_x)/lx))+1);
        
                    Scalar cmp_y = p_cmp.y;
                    if (y > m_box.yhi)
                        cmp_y -= ly*(((int)((cmp_y-m_box.yhi)/ly))+1);
                    else if (y < m_box.ylo)
                        cmp_y += ly*(((int)((m_box.ylo-cmp_y)/ly))+1);

                    Scalar cmp_z = p_cmp.z;
                    if (cmp_z > m_box.zhi)
                        cmp_z -= lz*(((int)((cmp_z-m_box.zhi)/lz))+1);
                    else if (cmp_z < m_box.zlo)
                        cmp_z += lz*(((int)((m_box.zlo-cmp_z)/lz))+1);

                    // minimum image convention for dx
                    Scalar dx = x - cmp_x;
                    if (dx > lx/2.)
                        dx -= lx;
                    else if (dx <= -lx/2.)
                        dx += lx;
                        
                    Scalar dy = y - cmp_y;
                    if (dy > ly/2.)
                        dy -= ly;
                    else if (dy <= -ly/2.)
                        dy += ly;
                        
                    Scalar dz = z - cmp_z;
                    if (dz > lz/2.)
                        dz -= lz;
                    else if (dz <= -lz/2.)
                        dz += lz;
                        
                    if (dx*dx + dy*dy + dz*dz < min_dist)
                        return false;
                    }
                }
            }
        }
    return true;
    }



/*! \param p Particle to place
    \param idx Index to place it at

    \note It is an error to place a particle at the same idx more than once unless undoPlace() is
    called before each subsequent place(). This error will not be detected.
*/
void GeneratedParticles::place(const particle& p, unsigned int idx)
    {
    assert(idx < m_particles.size());
    
    // begin with an error check that p.type is actually in the radius map
    if (m_radii.count(p.type) == 0)
        {
        cerr << endl << "***Error! Radius not set for particle in RandomGenerator" << endl << endl;
        throw runtime_error("Error placing particle");
        }
        
    // first, map the particle back into the box
    Scalar x = p.x;
    Scalar lx = m_box.xhi - m_box.xlo;
    int ix = 0;
    if (x > m_box.xhi)
        {
        ix=(((int)((x-m_box.xhi)/lx))+1);
        x -= lx*ix;
        }
    else if (x < m_box.xlo)
        {
        ix=-(((int)((m_box.xlo-x)/lx))+1);
        x -= lx*ix;
        }

    Scalar y = p.y;
    Scalar ly = m_box.yhi - m_box.ylo;
    int iy = 0;
    if (y > m_box.yhi)
        {
        iy=(((int)((y-m_box.yhi)/ly))+1);
        y -= ly*iy;
        }
    else if (y < m_box.ylo)
        {
        iy=-(((int)((m_box.ylo-y)/ly))+1);
        y -= ly*iy;
        }
        
    Scalar z = p.z;
    Scalar lz = m_box.zhi - m_box.zlo;
    int iz = 0;
    if (z > m_box.zhi)
        {
        iz=(((int)((z-m_box.zhi)/lz))+1);
        z -= lz*iz;
        }
    else if (z < m_box.zlo)
        {
        iz=-(((int)((m_box.zlo-z)/lz))+1);
        z -= lz*iz;
        }

    // set the particle data
    m_particles[idx].x = x;
    m_particles[idx].y = y;
    m_particles[idx].z = z;
    m_particles[idx].ix = ix;
    m_particles[idx].iy = iy;
    m_particles[idx].iz = iz;
    m_particles[idx].type = p.type;
    
    // determine the bin the particle is in
    int ib = (int)((x-m_box.xlo)*m_scalex);
    int jb = (int)((y-m_box.ylo)*m_scaley);
    int kb = (int)((z-m_box.zlo)*m_scalez);
    
    // need to handle the case where the particle is exactly at the box hi
    if (ib == m_Mx)
        ib = 0;
    if (jb == m_My)
        jb = 0;
    if (kb == m_Mz)
        kb = 0;
        
    // sanity check
    assert(ib >= 0 && ib < m_Mx && jb >=0 && jb < m_My && kb >= 0 && kb < m_Mz);
    
    // add it to the bin
    int bin = ib*(m_My*m_Mz) + jb * m_Mz + kb;
    m_bins[bin].push_back(idx);
    }


/*! \param idx Index of the particle to remove
    If a particle was placed by place() and it is later determined that it needs to be moved,
    the caller must undo the placement with undoPlace() before replacing the particle with place().
*/
void GeneratedParticles::undoPlace(unsigned int idx)
    {
    assert(idx < m_particles.size());
    
    particle p = m_particles[idx];
    // first, map the particle back into the box
    Scalar x = p.x;
    Scalar lx = m_box.xhi - m_box.xlo;
    int ix = 0;
    if (x > m_box.xhi)
        {
        ix=(((int)((x-m_box.xhi)/lx))+1);
        x -= lx*ix;
        }
    else if (x < m_box.xlo)
        {
        ix=-(((int)((m_box.xlo-x)/lx))+1);
        x -= lx*ix;
        }

    Scalar y = p.y;
    Scalar ly = m_box.yhi - m_box.ylo;
    int iy = 0;
    if (y > m_box.yhi)
        {
        iy=(((int)((y-m_box.yhi)/ly))+1);
        y -= ly*iy;
        }
    else if (y < m_box.ylo)
        {
        iy=-(((int)((m_box.ylo-y)/ly))+1);
        y -= ly*iy;
        }
        
    Scalar z = p.z;
    Scalar lz = m_box.zhi - m_box.zlo;
    int iz = 0;
    if (z > m_box.zhi)
        {
        iz=(((int)((z-m_box.zhi)/lz))+1);
        z -= lz*iz;
        }
    else if (z < m_box.zlo)
        {
        iz=-(((int)((m_box.zlo-z)/lz))+1);
        z -= lz*iz;
        }

    // set the particle data
    m_particles[idx].x = x;
    m_particles[idx].y = y;
    m_particles[idx].z = z;
    m_particles[idx].ix = ix;
    m_particles[idx].iy = iy;
    m_particles[idx].iz = iz;
    m_particles[idx].type = p.type;
    
    // determine the bin the particle is in
    int ib = (int)((x-m_box.xlo)*m_scalex);
    int jb = (int)((y-m_box.ylo)*m_scaley);
    int kb = (int)((z-m_box.zlo)*m_scalez);
    
    // need to handle the case where the particle is exactly at the box hi
    if (ib == m_Mx)
        ib = 0;
    if (jb == m_My)
        jb = 0;
    if (kb == m_Mz)
        kb = 0;
        
    // sanity check
    assert(ib < m_Mx && jb < m_My && kb < m_Mz);
    
    // remove it from the bin
    int bin = ib*(m_My*m_Mz) + jb * m_Mz + kb;
    vector<unsigned int> &bin_list = m_bins[bin];
    vector<unsigned int>::iterator itr;
    for (itr = bin_list.begin(); itr != bin_list.end(); ++itr)
        {
        if (*itr == idx)
            {
            bin_list.erase(itr);
            break;
            }
        }
    }

/*! \param a Tag of the first particle in the bond
    \param b Tag of the second particle in the bond
    \param type Type of the bond

    Adds a bond between particles with tags \a a and \a b of type \a type
*/
void GeneratedParticles::addBond(unsigned int a, unsigned int b, const std::string& type)
    {
    m_bonds.push_back(bond(a,b, type));
    }

/*! \param box Box dimensions to generate in
    \param seed Random number generator seed
*/
RandomGenerator::RandomGenerator(const BoxDim& box, unsigned int seed) : m_box(box), m_seed(seed)
    {
    }

unsigned int RandomGenerator::getNumParticles() const
    {
    return (unsigned int)m_data.m_particles.size();
    }

unsigned int RandomGenerator::getNumParticleTypes() const
    {
    return (unsigned int)m_type_mapping.size();
    }

BoxDim RandomGenerator::getBox() const
    {
    return m_box;
    }


void RandomGenerator::initArrays(const ParticleDataArrays &pdata) const
    {
    for (unsigned int i = 0; i < pdata.nparticles; i++)
        {
        pdata.x[i] = m_data.m_particles[i].x;
        pdata.y[i] = m_data.m_particles[i].y;
        pdata.z[i] = m_data.m_particles[i].z;
        pdata.ix[i] = m_data.m_particles[i].ix;
        pdata.iy[i] = m_data.m_particles[i].iy;
        pdata.iz[i] = m_data.m_particles[i].iz;
        pdata.type[i] = m_data.m_particles[i].type_id;
        
        pdata.tag[i] = i;
        pdata.rtag[i] = i;
        }
    }

std::vector<std::string> RandomGenerator::getTypeMapping() const
    {
    return m_type_mapping;
    }

/*! \return Number of bond types generated
*/
unsigned int RandomGenerator::getNumBondTypes() const
    {
    return m_bond_type_mapping.size();
    }

/*! \param bond_data Shared pointer to the BondData to be initialized
    Adds all generated bonds to the BondData
*/
void RandomGenerator::initBondData(boost::shared_ptr<BondData> bond_data) const
    {
    // loop through all the bonds and add a bond for each
    for (unsigned int i = 0; i < m_data.m_bonds.size(); i++)
        bond_data->addBond(Bond(m_data.m_bonds[i].type_id, m_data.m_bonds[i].tag_a, m_data.m_bonds[i].tag_b));
        
    bond_data->setBondTypeMapping(m_bond_type_mapping);
    }

/*! \param type Name of the particle type to set the radius for
    \param radius Radius to set
*/
void RandomGenerator::setSeparationRadius(string type, Scalar radius)
    {
    m_radii[type] = radius;
    }

/*! \param repeat Number of copies of this generator to create in the box
    \param generator Smart pointer to the generator to use
*/
void RandomGenerator::addGenerator(unsigned int repeat, boost::shared_ptr<ParticleGenerator> generator)
    {
    m_generator_repeat.push_back(repeat);
    m_generators.push_back(generator);
    }

/*! \pre setSeparationRadius has been called for all particle types that will be generated
    \pre addGenerator has been called for all desired generators
*/
void RandomGenerator::generate()
    {
    // sanity check
    assert(m_radii.size() > 0);
    assert(m_generators.size() > 0);
    assert(m_generators.size() == m_generator_repeat.size());
    
    // count the number of particles
    unsigned int n_particles = 0;
    for (unsigned int i = 0; i < m_generators.size(); i++)
        n_particles += m_generator_repeat[i] * m_generators[i]->getNumToGenerate();
        
    // setup data structures
    m_data = GeneratedParticles(n_particles, m_box, m_radii);
    
    // start the random number generator
    boost::mt19937 rnd;
    rnd.seed(boost::mt19937::result_type(m_seed));
    
    // perform the generation
    unsigned int start_idx = 0;
    for (unsigned int i = 0; i < m_generators.size(); i++)
        {
        for (unsigned int j = 0; j < m_generator_repeat[i]; j++)
            {
            m_generators[i]->generateParticles(m_data, rnd, start_idx);
            start_idx += m_generators[i]->getNumToGenerate();
            }
        }
        
    // get the type id of all particles
    for (unsigned int i = 0; i < m_data.m_particles.size(); i++)
        {
        m_data.m_particles[i].type_id = getTypeId(m_data.m_particles[i].type);
        }
        
    // walk through all the bonds and assign ids
    for (unsigned int i = 0; i < m_data.m_bonds.size(); i++)
        m_data.m_bonds[i].type_id = getBondTypeId(m_data.m_bonds[i].type);
    }

/*! \param name Name to get type id of
    If \a name has already been added, this returns the type index of that name.
    If \a name has not yet been added, it is added to the list and the new id is returned.
*/
unsigned int RandomGenerator::getTypeId(const std::string& name)
    {
    // search for the type mapping
    for (unsigned int i = 0; i < m_type_mapping.size(); i++)
        {
        if (m_type_mapping[i] == name)
            return i;
        }
    // add a new one if it is not found
    m_type_mapping.push_back(name);
    return (unsigned int)m_type_mapping.size()-1;
    }

/*! \param name Name to get type id of
    If \a name has already been added, this returns the type index of that name.
    If \a name has not yet been added, it is added to the list and the new id is returned.
*/
unsigned int RandomGenerator::getBondTypeId(const std::string& name)
    {
    // search for the type mapping
    for (unsigned int i = 0; i < m_bond_type_mapping.size(); i++)
        {
        if (m_bond_type_mapping[i] == name)
            return i;
        }
    // add a new one if it is not found
    m_bond_type_mapping.push_back(name);
    return (unsigned int)m_bond_type_mapping.size()-1;
    }

//! Helper function to generate a [0..1] float
/*! \param rnd Random number generator to use
*/
static Scalar random01(boost::mt19937& rnd)
    {
    unsigned int val = rnd();
    
    double val01 = ((double)val - (double)rnd.min()) / ( (double)rnd.max() - (double)rnd.min() );
    return Scalar(val01);
    }

/////////////////////////////////////////////////////////////////////////////////////////
// PolymerParticleGenerator
/*! \param bond_len Bond length to generate
    \param types Vector of type names. One element per bead of the polymer.
    \param bond_a List of the first particle in each bond
    \param bond_b List of the 2nd particle in each bond
    \param bond_type List of the bond type names for each bond
    \param max_attempts The maximum number of attempts to place each particle

    A bonded pair of paritlces is \a bond_a[i] bonded to \a bond_b[i], with 0 being the first particle in the polymer.
    Hence, the sizes of \a bond_a and \a bond_b \b must be the same.
*/
PolymerParticleGenerator::PolymerParticleGenerator(Scalar bond_len, const std::vector<std::string>& types, const std::vector<unsigned int>& bond_a, const std::vector<unsigned int>& bond_b, const std::vector<string>& bond_type, unsigned int max_attempts)
        : m_bond_len(bond_len), m_types(types), m_bond_a(bond_a), m_bond_b(bond_b), m_bond_type(bond_type), m_max_attempts(max_attempts)
    {
    assert(m_types.size() > 0);
    assert(m_max_attempts > 0);
    assert(bond_len > Scalar(0.0));
    assert(m_bond_a.size() == m_bond_b.size());
    assert(m_bond_a.size() == m_bond_type.size());
    }

/*! \param particles Data to place particles in
    \param rnd Random number generator
    \param start_idx Index to start generating particles at
*/
void PolymerParticleGenerator::generateParticles(GeneratedParticles& particles, boost::mt19937& rnd, unsigned int start_idx)
    {
    const BoxDim& box = particles.getBox();
    
    GeneratedParticles::particle p;
    p.type = m_types[0];
    
    
    // make a maximum of m_max_attempts tries to generate the polymer
    for (unsigned int attempt = 0; attempt < m_max_attempts; attempt++)
        {
        // generate the position of the first particle
        p.x = box.xlo + random01(rnd) * (box.xhi - box.xlo);
        p.y = box.ylo + random01(rnd) * (box.yhi - box.ylo);
        p.z = box.zlo + random01(rnd) * (box.zhi - box.zlo);
        
        // see if we can place the particle
        if (!particles.canPlace(p))
            continue;  // try again if we cannot
            
        // place the particle
        particles.place(p, start_idx);
        
        if (generateNextParticle(particles, rnd, 1, start_idx, p))
            {
            // success! we are done
            // create the bonds for this polymer now (polymers are simply linear for now)
            for (unsigned int i = 0; i < m_bond_a.size(); i++)
                {
                particles.addBond(start_idx+m_bond_a[i], start_idx + m_bond_b[i], m_bond_type[i]);
                }
            return;
            }
            
        // failure, rollback
        particles.undoPlace(start_idx);
        cout << "Notice: Polymer generator is trying particle " << start_idx << " again" << endl;
        }
        
    // we've failed to place a polymer, this is an unrecoverable error
    cerr << endl << "***Error! The polymer generator failed to place a polymer, the system is too dense or the separation radii are set too high" << endl << endl;
    throw runtime_error("Error generating polymer system");
    }

/*! \param particles Data to place particles in
    \param rnd Random number generator
    \param i Index of the bead in the polymer to place
    \param start_idx Index to start generating particles at
    \param prev_particle Previous particle placed

    \returns true When all particles in the polymer > i are able to be placed
*/
bool PolymerParticleGenerator::generateNextParticle(GeneratedParticles& particles, boost::mt19937& rnd, unsigned int i, unsigned int start_idx,  const GeneratedParticles::particle& prev_particle)
    {
    // handle stopping condition
    if (i == m_types.size())
        return true;
        
    GeneratedParticles::particle p;
    p.type = m_types[i];
    
    // make a maximum of m_max_attempts tries to generate the polymer
    for (unsigned int attempt = 0; attempt < m_max_attempts; attempt++)
        {
        // generate a vector to move by to get to the next polymer bead
        Scalar r = m_bond_len;
        
        Scalar dy = Scalar(2.0 * random01(rnd) - 1.0);
        Scalar phi = Scalar(2.0 * M_PI*random01(rnd));
        Scalar dx = sin(phi) * cos(asin(dy));
        Scalar dz = cos(phi) * cos(asin(dy));
        
        p.x = prev_particle.x + r*dx;
        p.y = prev_particle.y + r*dy;
        p.z = prev_particle.z + r*dz;
        
        // see if we can place the particle
        if (!particles.canPlace(p))
            continue;  // try again if we cannot
            
        // place the particle
        particles.place(p, start_idx+i);
        
        if (generateNextParticle(particles, rnd, i+1, start_idx, p))
            {
            // success! we are done
            return true;
            }
            
        // failure, rollback
        particles.undoPlace(start_idx+i);
        }
        
    // we've tried and we've failed
    return false;
    }

//! Helper class to allow boost.python wrapping of ParticleGenerator
class ParticleGeneratorWrap : public ParticleGenerator, public wrapper<ParticleGenerator>
    {
    public:
        //! Calls overidden ParticleGenerator::getNumToGenerate()
        unsigned int getNumToGenerate()
            {
            return this->get_override("getNumToGenerate")();
            }
            
        //! Calls overidden ParticleGenerator::generateParticles()
        /*! \param particles Place generated particles here after a GeneratedParticles::canPlace() check
            \param rnd Random number benerator to use
            \param start_idx Starting index to generate particles at
            Derived classes must implement this method. RandomGenerator will
            call it to generate the particles. Particles should be placed at indices
            \a start_idx, \a start_idx + 1, ... \a start_idx + getNumToGenerate()-1
        */
        void generateParticles(GeneratedParticles& particles, boost::mt19937& rnd, unsigned int start_idx)
            {
            this->get_override("generateParticle")(particles, rnd, start_idx);
            }
    };


void export_RandomGenerator()
    {
    class_<std::vector<string> >("std_vector_string")
    .def(vector_indexing_suite<std::vector<string> >())
    ;
    
    class_< RandomGenerator, bases<ParticleDataInitializer> >("RandomGenerator", init<const BoxDim&, unsigned int>())
    // virtual methods from ParticleDataInitializer are inherited
    .def("setSeparationRadius", &RandomGenerator::setSeparationRadius)
    .def("addGenerator", &RandomGenerator::addGenerator)
    .def("generate", &RandomGenerator::generate)
    ;
    
    class_< ParticleGeneratorWrap, boost::shared_ptr<ParticleGeneratorWrap>, boost::noncopyable >("ParticleGenerator", init<>())
    // no methods exposed to python
    ;
    
    class_< PolymerParticleGenerator, boost::shared_ptr<PolymerParticleGenerator>, bases<ParticleGenerator>, boost::noncopyable >("PolymerParticleGenerator", init< Scalar, const std::vector<std::string>&, std::vector<unsigned int>&, std::vector<unsigned int>&, std::vector<string>&, unsigned int >())
    // all methods are internal C++ methods
    ;
    }

#ifdef WIN32
#pragma warning( pop )
#endif

