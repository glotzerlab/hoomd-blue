// Copyright (c) 2009-2019 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.


// Maintainer: joaander
#include "RandomGenerator.h"

#include <cassert>
#include <stdexcept>

namespace py = pybind11;
using namespace std;

// windows defines a macro min and max
#undef min
#undef max

/*! \file RandomGenerator.cc
    \brief Contains definitions for RandomGenerator and related classes.
 */

/*! \param exec_conf The execution configuration used for messaging
    \param n_particles Number of particles that will be generated
    \param box Box the particles are generated in
    \param radii Mapping of particle types to their minimum separation radius

    After construction, all data structure are set to defaults and particles are ready to be placed.
*/
GeneratedParticles::GeneratedParticles(std::shared_ptr<const ExecutionConfiguration> exec_conf,
                                       unsigned int n_particles,
                                       const BoxDim& box,
                                       const std::map< std::string, Scalar >& radii)
    : m_exec_conf(exec_conf), m_particles(n_particles), m_box(box), m_radii(radii)
    {
    // sanity checks
    assert(n_particles > 0);
    assert(m_radii.size() > 0);
    Scalar3 L = box.getNearestPlaneDistance();

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
    // cap the size to a minimum of 2 to prevent small sizes from blowing up the memory usage
    if (target_size < Scalar(2.0))
        target_size = Scalar(2.0);

    // calculate the particle binning
    m_Mx = (int)(L.x / (target_size));
    m_My = (int)(L.y / (target_size));
    m_Mz = (int)(L.z / (target_size));
    if (m_Mx == 0)
        m_Mx = 1;
    if (m_My == 0)
        m_My = 1;
    if (m_Mz == 0)
        m_Mz = 1;

    if (m_Mx > 100 || m_My > 100 || m_Mz > 100)
        {
        m_exec_conf->msg->warning() << "Random generator is about to allocate a very large amount of memory and may crash." << endl << endl;
        }

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
        m_exec_conf->msg->error() << endl << "Radius not set for particle in RandomGenerator" << endl << endl;
        throw runtime_error("Error placing particle");
        }

    // first, map the particle back into the box
    Scalar3 pos = make_scalar3(p.x,p.y,p.z);
    int3 img = m_box.getImage(pos);
    int3 negimg = make_int3(-img.x, -img.y, -img.z);
    pos = m_box.shift(pos, negimg);

    // determine the bin the particle is in
    Scalar3 f = m_box.makeFraction(pos);
    int ib = (int)(f.x*m_Mx);
    int jb = (int)(f.y*m_My);
    int kb = (int)(f.z*m_Mz);

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
                    Scalar3 cmp_pos = make_scalar3(p_cmp.x, p_cmp.y, p_cmp.z);

                    int3 img = m_box.getImage(pos);
                    int3 negimg = make_int3(-img.x, -img.y, -img.z);
                    cmp_pos = m_box.shift(cmp_pos,negimg);

                    Scalar3 dx = pos - cmp_pos;
                    // minimum image convention for dx
                    dx = m_box.minImage(dx);

                    if (dot(dx,dx) < min_dist*min_dist)
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
        m_exec_conf->msg->error() << endl << "Radius not set for particle in RandomGenerator" << endl << endl;
        throw runtime_error("Error placing particle");
        }

    // first, map the particle back into the box
    Scalar3 pos = make_scalar3(p.x,p.y,p.z);
    int3 img = m_box.getImage(pos);
    int3 negimg = make_int3(-img.x,-img.y,-img.z);
    pos = m_box.shift(pos,negimg);

    // set the particle data
    m_particles[idx].x = pos.x;
    m_particles[idx].y = pos.y;
    m_particles[idx].z = pos.z;
    m_particles[idx].ix = img.x;
    m_particles[idx].iy = img.y;
    m_particles[idx].iz = img.z;
    m_particles[idx].type = p.type;

    // determine the bin the particle is in
    Scalar3 f =m_box.makeFraction(pos);
    int ib = (int)(f.x*m_Mx);
    int jb = (int)(f.y*m_My);
    int kb = (int)(f.z*m_Mz);

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
    int3 img = make_int3(0,0,0);
    Scalar3 pos = make_scalar3(p.x,p.y,p.z);

    m_box.wrap(pos,img);

    // set the particle data
    m_particles[idx].x = pos.x;
    m_particles[idx].y = pos.y;
    m_particles[idx].z = pos.z;
    m_particles[idx].ix = img.x;
    m_particles[idx].iy = img.y;
    m_particles[idx].iz = img.z;
    m_particles[idx].type = p.type;

    // determine the bin the particle is in
    Scalar3 f = m_box.makeFraction(pos);
    int ib = (int)(f.x*m_Mx);
    int jb = (int)(f.y*m_My);
    int kb = (int)(f.z*m_Mz);

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

/*! \param exec_conf Execution configuration
    \param box Box dimensions to generate in
    \param seed Random number generator seed
    \param dimensions Number of dimensions in the simulation box
*/
RandomGenerator::RandomGenerator(std::shared_ptr<const ExecutionConfiguration> exec_conf,
                                 const BoxDim& box,
                                 unsigned int seed,
                                 unsigned int dimensions)
    : m_exec_conf(exec_conf),
      m_box(box),
      m_seed(seed),
      m_dimensions(dimensions)
    {
    }

/*! initializes a snapshot->with the internally stored copy of the particle and bond data */
std::shared_ptr< SnapshotSystemData<Scalar> > RandomGenerator::getSnapshot() const
    {
    // create a snapshot
    std::shared_ptr< SnapshotSystemData<Scalar> > snapshot(new SnapshotSystemData<Scalar>());

    // only execute on rank 0
    if (m_exec_conf->getRank()) return snapshot;

    // initialize box dimensions
    snapshot->global_box = m_box;

    // initialize particle data
    SnapshotParticleData<Scalar>& pdata_snap = snapshot->particle_data;

    unsigned int nparticles = m_data.m_particles.size();
    pdata_snap.resize(nparticles);

    for (unsigned int i = 0; i < nparticles; i++)
        {
        pdata_snap.pos[i] = vec3<Scalar>(m_data.m_particles[i].x, m_data.m_particles[i].y, m_data.m_particles[i].z);
        pdata_snap.image[i] = make_int3(m_data.m_particles[i].ix, m_data.m_particles[i].iy, m_data.m_particles[i].iz);
        pdata_snap.type[i] = m_data.m_particles[i].type_id;
        }

    pdata_snap.type_mapping = m_type_mapping;

    // initialize bonds
    BondData::Snapshot& bdata_snap = snapshot->bond_data;
    bdata_snap.resize(m_data.m_bonds.size());
    for (unsigned int i = 0; i < m_data.m_bonds.size(); i++)
        {
        BondData::members_t bond;
        bond.tag[0] = m_data.m_bonds[i].tag_a; bond.tag[1] = m_data.m_bonds[i].tag_b;
        bdata_snap.groups[i] = bond;
        bdata_snap.type_id[i] = m_data.m_bonds[i].type_id;
        }

    bdata_snap.type_mapping = m_bond_type_mapping;

    snapshot->dimensions = m_dimensions;

    return snapshot;
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
void RandomGenerator::addGenerator(unsigned int repeat, std::shared_ptr<ParticleGenerator> generator)
    {
    m_generator_repeat.push_back(repeat);
    m_generators.push_back(generator);
    }

/*! \pre setSeparationRadius has been called for all particle types that will be generated
    \pre addGenerator has been called for all desired generators
*/
void RandomGenerator::generate()
    {
    // only execute on rank 0
    if (m_exec_conf->getRank()) return;

    // sanity check
    assert(m_radii.size() > 0);
    assert(m_generators.size() > 0);
    assert(m_generators.size() == m_generator_repeat.size());

    // count the number of particles
    unsigned int n_particles = 0;
    for (unsigned int i = 0; i < m_generators.size(); i++)
        n_particles += m_generator_repeat[i] * m_generators[i]->getNumToGenerate();

    // setup data structures
    m_data = GeneratedParticles(m_exec_conf, n_particles, m_box, m_radii);

    // start the random number generator
    std::mt19937 rnd;
    rnd.seed(std::mt19937::result_type(m_seed));

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
static Scalar random01(std::mt19937& rnd)
    {
    unsigned int val = rnd();

    double val01 = ((double)val - (double)rnd.min()) / ( (double)rnd.max() - (double)rnd.min() );
    return Scalar(val01);
    }

/////////////////////////////////////////////////////////////////////////////////////////
// PolymerParticleGenerator
/*! \param exec_conf Execution configuration used for messaging
    \param bond_len Bond length to generate
    \param types Vector of type names. One element per bead of the polymer.
    \param bond_a List of the first particle in each bond
    \param bond_b List of the 2nd particle in each bond
    \param bond_type List of the bond type names for each bond
    \param max_attempts The maximum number of attempts to place each particle
    \param dimensions Number of dimensions in the system

    A bonded pair of particles is \a bond_a[i] bonded to \a bond_b[i], with 0 being the first particle in the polymer.
    Hence, the sizes of \a bond_a and \a bond_b \b must be the same.
*/
PolymerParticleGenerator::PolymerParticleGenerator(std::shared_ptr<const ExecutionConfiguration> exec_conf,
                                                   Scalar bond_len,
                                                   const std::vector<std::string>& types,
                                                   const std::vector<unsigned int>& bond_a,
                                                   const std::vector<unsigned int>& bond_b,
                                                   const std::vector<string>& bond_type,
                                                   unsigned int max_attempts,
                                                   unsigned int dimensions)
        : m_exec_conf(exec_conf), m_bond_len(bond_len), m_types(types), m_bond_a(bond_a), m_bond_b(bond_b),
          m_bond_type(bond_type), m_max_attempts(max_attempts), m_dimensions(dimensions)
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
void PolymerParticleGenerator::generateParticles(GeneratedParticles& particles, std::mt19937& rnd, unsigned int start_idx)
    {
    const BoxDim& box = particles.getBox();

    GeneratedParticles::particle p;
    p.type = m_types[0];

    // make a maximum of m_max_attempts tries to generate the polymer
    for (unsigned int attempt = 0; attempt < m_max_attempts; attempt++)
        {
        // generate the position of the first particle
        Scalar3 f = make_scalar3(random01(rnd),random01(rnd),random01(rnd));
        if (m_dimensions == 2)
            f.z = 0;
        Scalar3 pos = box.makeCoordinates(f);
        p.x = pos.x;
        p.y = pos.y;
        p.z = pos.z;

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
        m_exec_conf->msg->notice(2) << "Polymer generator is trying particle " << start_idx << " again" << endl;
        }

    // we've failed to place a polymer, this is an unrecoverable error
    m_exec_conf->msg->error() << endl << "The polymer generator failed to place a polymer, the system is too dense or the separation radii are set too high" << endl << endl;
    throw runtime_error("Error generating polymer system");
    }

/*! \param particles Data to place particles in
    \param rnd Random number generator
    \param i Index of the bead in the polymer to place
    \param start_idx Index to start generating particles at
    \param prev_particle Previous particle placed

    \returns true When all particles in the polymer > i are able to be placed
*/
bool PolymerParticleGenerator::generateNextParticle(GeneratedParticles& particles, std::mt19937& rnd, unsigned int i, unsigned int start_idx,  const GeneratedParticles::particle& prev_particle)
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

        Scalar dy, dx, dz;
        if (m_dimensions==3)
            {
            dy = Scalar(2.0 * random01(rnd) - 1.0);
            Scalar phi = Scalar(2.0 * M_PI*random01(rnd));
            dx = sin(phi) * cos(asin(dy));
            dz = cos(phi) * cos(asin(dy));
            }
        else
            {
            Scalar phi = Scalar(2.0 * M_PI*random01(rnd));
            dx = cos(phi);
            dy = sin(phi);
            dz = 0;
            }

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


void export_RandomGenerator(py::module& m)
    {
    py::class_< RandomGenerator >(m,"RandomGenerator")
    .def(py::init<std::shared_ptr<const ExecutionConfiguration>, const BoxDim&, unsigned int, unsigned int>())
    .def("setSeparationRadius", &RandomGenerator::setSeparationRadius)
    .def("addGenerator", &RandomGenerator::addGenerator)
    .def("generate", &RandomGenerator::generate)
    .def("getSnapshot", &RandomGenerator::getSnapshot)
    ;

    py::class_< ParticleGenerator, std::shared_ptr<ParticleGenerator> >(m,"ParticleGenerator")
    .def(py::init<>())
    // no methods exposed to python
    ;

    py::class_< PolymerParticleGenerator, std::shared_ptr<PolymerParticleGenerator> >(m,"PolymerParticleGenerator",py::base<ParticleGenerator>())
    .def(py::init< std::shared_ptr<const ExecutionConfiguration>, Scalar, const std::vector<std::string>&, std::vector<unsigned int>&, std::vector<unsigned int>&, std::vector<string>&, unsigned int, unsigned int >())
    // all methods are internal C++ methods
    ;
    }
