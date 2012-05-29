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

/*! \file RemoveParticlesUpdater.cc
    \brief Defines the RemoveParticlesUpdater class
*/

#ifdef WIN32
#pragma warning( push )
#pragma warning( disable : 4103 4244 )
#endif

#include <boost/python.hpp>
using namespace boost::python;

#include "RemoveParticlesUpdater.h"

using namespace std;

/*! \param sysdef System to remove particles from
    \param groupA Primary group of particles on which this updater will act
    \param groupB Secondary group of particles on which the updater will act
    \param thermo Compute for the thermodynamic quantities of this group
    \param nlist Neighbor list from which bound atoms will be found
    \param r_cut Cutoff radius for removing neighbors
    \param filename File to which removed particles will be recorded
*/
RemoveParticlesUpdater::RemoveParticlesUpdater(boost::shared_ptr<SystemDefinition> sysdef,
                                               boost::shared_ptr<ParticleGroup> groupA,
											   boost::shared_ptr<ParticleGroup> groupB,
                                               boost::shared_ptr<ComputeThermo> thermo,
                                               boost::shared_ptr<NeighborList> nlist,
											   Scalar r_cut,
                                               std::string filename)
    : Updater(sysdef), m_groupA(groupA), m_groupB(groupB), m_thermo(thermo), m_nlist(nlist), m_rcut(r_cut),
	m_filename(filename), num_particles(groupA->getNumMembers())
{
    assert(m_pdata);

	m_exec_conf->msg->notice(5) << "Contructing RemoveParticlesUpdater" << endl;
}

/*! Perform the necessary computations to remove particles from the system
    \param timestep Current time step of the simulation
*/
void RemoveParticlesUpdater::update(unsigned int timestep)
{
    if (m_prof) m_prof->push("RemoveParticles");

    m_nlist->compute(timestep);
    ArrayHandle<unsigned int> h_n_neigh(m_nlist->getNNeighArray(), access_location::host, access_mode::read);
    ArrayHandle<unsigned int> h_nlist(m_nlist->getNListArray(), access_location::host, access_mode::read);
    Index2D nli = m_nlist->getNListIndexer();

    // access the particle data for writing on the CPU
	assert(m_pdata);
	ArrayHandle<Scalar4> h_vel(m_pdata->getVelocities(), access_location::host, access_mode::readwrite);
    ArrayHandle<Scalar3> h_accel(m_pdata->getAccelerations(), access_location::host, access_mode::readwrite);
    ArrayHandle<Scalar4> h_pos(m_pdata->getPositions(), access_location::host, access_mode::readwrite);
    ArrayHandle<int3> h_image(m_pdata->getImages(), access_location::host, access_mode::readwrite);
	ArrayHandle< unsigned int > h_tag(m_pdata->getTags(), access_location::host, access_mode::readwrite);

    // store the group size of groupA
    const unsigned int group_size = m_groupA->getNumMembers();
    if (group_size == 0) return;

    // store the box dimensions
    const BoxDim& box = m_pdata->getBox();

    // store the inert ID
    unsigned int inertID = m_pdata->getTypeByName("inert");

    for (unsigned int group_idx = 0; group_idx < group_size; group_idx++)
    {
        unsigned int i = m_groupA->getMemberIndex(group_idx);

        Scalar temp_mass = Scalar(0.0);
        Scalar velx = Scalar(0.0);
        Scalar vely = Scalar(0.0);
        Scalar velz = Scalar(0.0);
		// Scalar posx = Scalar(0.0);
		// Scalar posy = Scalar(0.0);
		// Scalar posz = Scalar(0.0);
        std::stringstream species;
        std::string particle_type;

        // check if the particle passed a z boundary
        if (h_image.data[i].z != 0)
        {
            // decrement the degrees of freedom
            --num_particles;

            // add atom's velocity to molecule velocity
            velx += h_vel.data[i].w * h_vel.data[i].x;
            vely += h_vel.data[i].w * h_vel.data[i].y;
            velz += h_vel.data[i].w * h_vel.data[i].z;

			// add atom's position to molecule position
			// posx += h_vel.data[i].w * ( h_pos.data[i].x + h_image.data[i].x * Lx );
			// posy += h_vel.data[i].w * ( h_pos.data[i].y + h_image.data[i].y * Ly );
			// posz += h_vel.data[i].w * ( h_pos.data[i].z + h_image.data[i].z * Lz );

            // record the sputtered products
            particle_type = m_pdata->getNameByType(m_pdata->getType(h_tag.data[i]));
            temp_mass += h_vel.data[i].w;
			species << particle_type << h_tag.data[i];

            // render the particle inert
            h_image.data[i].z = 0;
			m_pdata->setType(h_tag.data[i], inertID);
            h_accel.data[i].x = Scalar(0.0);
            h_accel.data[i].y = Scalar(0.0);
            h_accel.data[i].z = Scalar(0.0);
            h_vel.data[i].x = Scalar(0.0);
            h_vel.data[i].y = Scalar(0.0);
            h_vel.data[i].z = Scalar(0.0);

            // loops through particle's neighbors
            const unsigned int nlist_size_i = (unsigned int) h_n_neigh.data[i];
            for (unsigned int j = 0; j < nlist_size_i; j++)
            {
                // access the index of j
                unsigned int jj = h_nlist.data[nli(i,j)];
                assert(jj < m_pdata->getN());

				// compute the distance between i and j
				// Scalar3 dx = h_pos.data[i] - h_pos.data[jj];
				// dx = box.minImage(dx);

				const bool isInert = ( m_pdata->getType(h_tag.data[jj]) == inertID );
                if ( (!isInert) && ( m_groupA->isMember(jj) ) )
                {

                    // decrement the degrees of freedom
                    --num_particles;

                    // add atom's velocity to molecule velocity
                    velx += h_vel.data[jj].w * h_vel.data[jj].x;
                    vely += h_vel.data[jj].w * h_vel.data[jj].x;
                    velz += h_vel.data[jj].w * h_vel.data[jj].x;

					// add atom's position to molecule position
					// posx += h_vel.data[jj].w * ( h_pos.data[jj].x + h_image.data[jj].x * Lx );
					// posy += h_vel.data[jj].w * ( h_pos.data[jj].y + h_image.data[jj].y * Ly );
					// posz += h_vel.data[jj].w * ( h_pos.data[jj].z + h_image.data[jj].z * Lz );

                    // record the sputtered products
                    particle_type = m_pdata->getNameByType( m_pdata->getType(h_tag.data[jj]) );
                    temp_mass += h_vel.data[jj].w;
					species << ":" << particle_type << h_tag.data[jj];

                    // render the particle inert
                    h_image.data[jj].z = 0;
					m_pdata->setType(h_tag.data[jj], inertID);
                    h_accel.data[jj].x = Scalar(0.0);
                    h_accel.data[jj].y = Scalar(0.0);
                    h_accel.data[jj].z = Scalar(0.0);
                    h_vel.data[jj].x = Scalar(0.0);
                    h_vel.data[jj].y = Scalar(0.0);
                    h_vel.data[jj].z = Scalar(0.0);

					// loops through secondary particle's neighbors
					// const unsigned int nlist_size_j = (unsigned int) h_n_neigh.data[jj];
                    // for (unsigned int k = 0; k < nlist_size_j; k++)
                    // {
                        // // access the index of k
                        // unsigned int kk = h_nlist.data[nli(j,k)];
                        // assert(kk < m-pdata->getN());

                        // if ( ( kk != i ) && ( arrays.type[kk] != inertID ) && ( m_groupA->isMember(kk) ) )
                        // {
                            // // decrement the number of degrees of freedom
                            // --num_particles;

                            // // add atoms velocity to molecule velocity
                            // velx += arrays.mass[kk] * arrays.vx[kk];
                            // vely += arrays.mass[kk] * arrays.vy[kk];
                            // velz += arrays.mass[kk] * arrays.vz[kk];

							// // add atom's position to molecule position
							// posx += arrays.mass[kk] * ( arrays.x[kk] + arrays.ix[kk] * Lx );
							// posy += arrays.mass[kk] * ( arrays.y[kk] + arrays.iy[kk] * Ly );
							// posz += arrays.mass[kk] * ( arrays.z[kk] + arrays.iz[kk] * Lz );

                            // // record the sputtered products
                            // particle_type = m_pdata->getNameByType(arrays.type[kk]);
                            // temp_mass += arrays.mass[kk];
							// species << ":" << particle_type << arrays.tag[kk];

                            // arrays.iz[kk] = 0;
                            // arrays.type[kk] = inertID;
                            // arrays.ax[kk] = Scalar(0.0);
                            // arrays.ay[kk] = Scalar(0.0);
                            // arrays.az[kk] = Scalar(0.0);
                            // arrays.vx[kk] = Scalar(0.0);
                            // arrays.vy[kk] = Scalar(0.0);
                            // arrays.vz[kk] = Scalar(0.0);
                        // }
                    // }
                }
            }

            // compute the outgoing trajectory of the particle
            velx /= temp_mass;
            vely /= temp_mass;
            velz /= temp_mass;
			// posx /= temp_mass;
			// posy /= temp_mass;
			// posz /= temp_mass;
            Scalar xy = sqrt(velx * velx + vely * vely);
            Scalar phi = atan2(xy, velz) * 180.0 / M_PI;
            Scalar theta = atan2(vely, velx) * 180.0 / M_PI;
			Scalar kinetic_energy = 0.5 * temp_mass * ( velz * velz + xy * xy );

            // output the sputtered products
            ofstream sputter_output;
            sputter_output.open(m_filename.c_str(), ios::app);
            sputter_output << "  " << timestep << "\t" << species.str() << "\t";
			sputter_output << phi << "\t" << theta << "\t" << kinetic_energy << std::endl;
			// sputter_output << posx << "\t" << posy << "\t" << posz << std::endl;
            sputter_output.close();
        }
    }

    // store the group size of groupB
    const unsigned int group_sizeB = m_groupB->getNumMembers();
    if ( group_sizeB == 0 ) return;
    
    for (unsigned int group_idx = 0; group_idx < group_sizeB; group_idx++)
    {
        unsigned int i = m_groupB->getMemberIndex(group_idx);

        Scalar velx = Scalar(0.0);
        Scalar vely = Scalar(0.0);
        Scalar velz = Scalar(0.0);
		// Scalar posx = Scalar(0.0);
		// Scalar posy = Scalar(0.0);
		// Scalar posz = Scalar(0.0);
		std::stringstream species;
        std::string particle_type;

	const bool isInert = ( m_pdata->getType(h_tag.data[i]) == inertID );
	if ( ( h_image.data[i].z != 0 || h_pos.data[i].z < m_rcut ) && (!isInert) )
	{
	    // record the atom's velocity
	    velx = h_vel.data[i].x;
	    vely = h_vel.data[i].y;
	    velz = h_vel.data[i].z;

	    // add atom's position to molecule position
	    // posx += h_pos.data[i].x + h_image.data[i].x * Lx;
	    // posy += h_pos.data[i].y + h_image.data[i].y * Ly;
	    // posz += h_pos.data[i].z + h_image.data[i].z * Lz;

	    // record the atom's species
	    particle_type = m_pdata->getNameByType(m_pdata->getType(h_tag.data[i]));
	    species << particle_type << h_tag.data[i];

	    // render the particle inert
	    h_image.data[i].z = 0;
		m_pdata->setType(h_tag.data[i], inertID);
	    h_accel.data[i].x = Scalar(0.0);
	    h_accel.data[i].y = Scalar(0.0);
	    h_accel.data[i].z = Scalar(0.0);
	    h_vel.data[i].x = Scalar(0.0);
	    h_vel.data[i].y = Scalar(0.0);
	    h_vel.data[i].z = Scalar(0.0);

		// compute the outgoing trajectory of the particle
		Scalar xy = sqrt(velx * velx + vely * vely);
		Scalar phi = atan2(xy, velz) * 180.0 / M_PI;
		Scalar theta = atan2(vely, velx) * 180.0 / M_PI;
		Scalar kinetic_energy = 0.5 * h_vel.data[i].w * ( velz * velz + xy * xy );

		// output the sputtered products
		ofstream sputter_output;
		sputter_output.open(m_filename.c_str(), ios::app);
		sputter_output << "  " << timestep << "\t" << species.str() << "\t";
	    sputter_output << phi << "\t" << theta << "\t" << kinetic_energy << std::endl;
	    // sputter_output << posx << "\t" << posy << "\t" << posz << std::endl;
		sputter_output.close();
	}
    }

    if (m_prof) m_prof->pop();

    // now we must change the number of degrees of freedom in the system
    if (num_particles > 1)
        m_thermo->setNDOF(3 * (num_particles - 1));
    else
        m_thermo->setNDOF(1);
}

void export_RemoveParticlesUpdater()
{
    class_<RemoveParticlesUpdater, boost::shared_ptr<RemoveParticlesUpdater>, bases<Updater>, boost::noncopyable>
    ("RemoveParticlesUpdater", init< boost::shared_ptr<SystemDefinition>,
                                     boost::shared_ptr<ParticleGroup>,
				     boost::shared_ptr<ParticleGroup>,
                                     boost::shared_ptr<ComputeThermo>,
                                     boost::shared_ptr<NeighborList>,
				     Scalar,
                                     std::string
                                     >())
    ;
}

#ifdef WIN32
#pragma warning( pop )
#endif
