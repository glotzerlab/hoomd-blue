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

#ifdef WIN32
#pragma warning( push )
#pragma warning( disable : 4103 4244 4267)
#endif

#include "SystemDefinition.h"

#include <boost/python.hpp>
using namespace boost::python;

/*! \post All shared pointers contained in SystemDefinition are NULL
*/
SystemDefinition::SystemDefinition()
	{
	}

/*! \param N Number of particles to allocate
	\param box Initial box particles are in
	\param n_types Number of particle types to set
	\param n_bond_types Number of bond types to create
	\param exec_conf The ExecutionConfiguration HOOMD is to be run on
	
	Creating SystemDefinition with this constructor results in 
	 - ParticleData constructed with the arguments \a N, \a box, \a n_types, and \a exec_conf. 
	 - BondData constructed with the arguments \a n_bond_types
	 - All other data structures are default constructed.
*/
SystemDefinition::SystemDefinition(unsigned int N, const BoxDim &box, unsigned int n_types, unsigned int n_bond_types, unsigned int n_angle_types, unsigned int n_dihedral_types, unsigned int n_improper_types, const ExecutionConfiguration& exec_conf)
	{
	m_particle_data = boost::shared_ptr<ParticleData>(new ParticleData(N, box, n_types, exec_conf));
	m_bond_data = boost::shared_ptr<BondData>(new BondData(m_particle_data, n_bond_types));
	m_wall_data = boost::shared_ptr<WallData>(new WallData());
	
	// only initialize the rigid body data if we are not running on multiple GPUs
	// this is a temporary hack only while GPUArray doesn't support multiple GPUs
	#ifdef ENABLE_CUDA
	if (exec_conf.gpu.size() <= 1)
	#endif
		m_rigid_data = boost::shared_ptr<RigidData>(new RigidData(m_particle_data));
		
	m_angle_data = boost::shared_ptr<AngleData>(new AngleData(m_particle_data, n_angle_types));
	m_dihedral_data = boost::shared_ptr<DihedralData>(new DihedralData(m_particle_data, n_dihedral_types));
	m_improper_data = boost::shared_ptr<DihedralData>(new DihedralData(m_particle_data, n_improper_types));
	}

/*! Calls the initializer's members to determine the number of particles, box size and then
	uses it to fill out the position and velocity data.
	\param init Initializer to use
	\param exec_conf Execution configuration to run on
	
	\b TEMPORARY!!!!! Initializers are planned to be rewritten
*/
SystemDefinition::SystemDefinition(const ParticleDataInitializer& init, const ExecutionConfiguration& exec_conf)
	{
	m_particle_data = boost::shared_ptr<ParticleData>(new ParticleData(init, exec_conf));
	
	m_bond_data = boost::shared_ptr<BondData>(new BondData(m_particle_data, init.getNumBondTypes()));
	init.initBondData(m_bond_data);
	
	m_wall_data = boost::shared_ptr<WallData>(new WallData());
	init.initWallData(m_wall_data);
	
	// only initialize the rigid body data if we are not running on multiple GPUs
	// this is a temporary hack only while GPUArray doesn't support multiple GPUs	
	#ifdef ENABLE_CUDA
	if (exec_conf.gpu.size() <= 1)
	#endif
		m_rigid_data = boost::shared_ptr<RigidData>(new RigidData(m_particle_data));
		
	m_angle_data = boost::shared_ptr<AngleData>(new AngleData(m_particle_data, init.getNumAngleTypes()));
	init.initAngleData(m_angle_data);
	
	m_dihedral_data = boost::shared_ptr<DihedralData>(new DihedralData(m_particle_data, init.getNumDihedralTypes()));
	init.initDihedralData(m_dihedral_data);
	
	m_improper_data = boost::shared_ptr<DihedralData>(new DihedralData(m_particle_data, init.getNumImproperTypes()));
	init.initImproperData(m_improper_data);
	}

/*! Initialize required data before runs
 
*/
int SystemDefinition::init()
{
	// initialize rigid bodies
	m_rigid_data->initializeData();
	
	return 1;
}


/*! Write restart file

*/
void SystemDefinition::writeRestart(unsigned int timestep)
{
	BoxDim box = m_particle_data->getBox();

	char file_name[100];
	sprintf(file_name, "restart_%d.txt", timestep);
	FILE *fp = fopen(file_name, "w");
	
	Scalar Lx = box.xhi - box.xlo;
    Scalar Ly = box.yhi - box.ylo;
    Scalar Lz = box.zhi - box.zlo;

	// Particles
	ParticleDataArrays arrays = m_particle_data->acquireReadWrite();
	fprintf(fp, "%d\n", arrays.nparticles);
	fprintf(fp, "%d\n", m_particle_data->getNTypes());
	fprintf(fp, "%d\n", m_bond_data->getNBondTypes());
	fprintf(fp, "%f\t%f\t%f\n", Lx, Ly, Lz);
	for (unsigned int i = 0; i < arrays.nparticles; i++)
		{
		fprintf(fp, "%d\t%d\t%f\t%f\t%f\t%f\t%f\t%f\n", arrays.type[i], arrays.body[i], 
				arrays.x[i], arrays.y[i], arrays.z[i], 
				arrays.vx[i], arrays.vy[i], arrays.vz[i]);
		}	
	
	m_particle_data->release();
	
	// Rigid bodies
	unsigned int n_bodies = m_rigid_data->getNumBodies();

	fprintf(fp, "%d\n", n_bodies);
	if (n_bodies <= 0) 
		{
		fclose(fp);
		return;
		}
	
	
	{
	ArrayHandle<Scalar> body_mass_handle(m_rigid_data->getBodyMass(), access_location::host, access_mode::read);
	ArrayHandle<unsigned int> body_size_handle(m_rigid_data->getBodySize(), access_location::host, access_mode::read);
	ArrayHandle<Scalar4> moment_inertia_handle(m_rigid_data->getMomentInertia(), access_location::host, access_mode::read);
		
	ArrayHandle<Scalar4> com_handle(m_rigid_data->getCOM(), access_location::host, access_mode::read);	
	ArrayHandle<Scalar4> angvel_handle(m_rigid_data->getAngVel(), access_location::host, access_mode::read);
	
	ArrayHandle<Scalar4> orientation_handle(m_rigid_data->getOrientation(), access_location::host, access_mode::read);
	ArrayHandle<Scalar4> ex_space_handle(m_rigid_data->getExSpace(), access_location::host, access_mode::read);
	ArrayHandle<Scalar4> ey_space_handle(m_rigid_data->getEySpace(), access_location::host, access_mode::read);
	ArrayHandle<Scalar4> ez_space_handle(m_rigid_data->getEzSpace(), access_location::host, access_mode::read);

	ArrayHandle<Scalar4> particle_pos_handle(m_rigid_data->getParticlePos(), access_location::host, access_mode::read);
	unsigned int particle_pos_pitch = m_rigid_data->getParticlePos().getPitch();
		
	for (unsigned int body = 0; body < n_bodies; body++)
		{
		fprintf(fp, "%f\t%f\t%f\n", moment_inertia_handle.data[body].x, moment_inertia_handle.data[body].y, moment_inertia_handle.data[body].z);
		fprintf(fp, "%f\t%f\t%f\n", com_handle.data[body].x, com_handle.data[body].y, com_handle.data[body].z);
		
		fprintf(fp, "%f\t%f\t%f\t%f\n", orientation_handle.data[body].x, orientation_handle.data[body].y, orientation_handle.data[body].z, orientation_handle.data[body].w);
		fprintf(fp, "%f\t%f\t%f\n", ex_space_handle.data[body].x, ex_space_handle.data[body].y, ex_space_handle.data[body].z);
		fprintf(fp, "%f\t%f\t%f\n", ey_space_handle.data[body].x, ey_space_handle.data[body].y, ey_space_handle.data[body].z);
		fprintf(fp, "%f\t%f\t%f\n", ez_space_handle.data[body].x, ez_space_handle.data[body].y, ez_space_handle.data[body].z);
		
		unsigned int len = body_size_handle.data[body];
		for (unsigned int j = 0; j < len; j++)
			{
			unsigned int localidx = body * particle_pos_pitch + j;
			fprintf(fp, "%f\t%f\t%f\n", particle_pos_handle.data[localidx].x, particle_pos_handle.data[localidx].y, particle_pos_handle.data[localidx].z);
			}
		}
		
	}
	
	fclose(fp);
}

/*! Read restart file

*/
void SystemDefinition::readRestart(const std::string& file_name)
{
	ParticleDataArrays arrays = m_particle_data->acquireReadWrite();

	unsigned int nparticles, natomtypes, nbondtypes;
			
	FILE *fp = fopen(file_name.c_str(), "r");
	fscanf(fp, "%d\n", &nparticles);
	fscanf(fp, "%d\n", &natomtypes);
	fscanf(fp, "%d\n", &nbondtypes);

	if (nparticles != arrays.nparticles || natomtypes != m_particle_data->getNTypes() || nbondtypes != m_bond_data->getNBondTypes())
		{
		printf("Restart file does not match!\n");
		return;
		}

	double Lx, Ly, Lz;
	BoxDim box;
	fscanf(fp, "%lf\t%lf\t%lf\n", &Lx, &Ly, &Lz);
	box.xlo = -0.5 * Lx;
	box.xhi = 0.5 * Lx;
	box.ylo = -0.5 * Ly;
	box.yhi = 0.5 * Ly;
	box.zlo = -0.5 * Lz;
	box.zhi = 0.5 * Lz;
	
	m_particle_data->setBox(box);
	for (unsigned int i = 0; i < nparticles; i++)
		{
		fscanf(fp, "%d\t%d\t%f\t%f\t%f\t%f\t%f\t%f\n", &arrays.type[i], &arrays.body[i], 
				&arrays.x[i], &arrays.y[i], &arrays.z[i], 
				&arrays.vx[i], &arrays.vy[i], &arrays.vz[i]);
		}	
		
	m_particle_data->release();
	
	// Rigid bodies
	unsigned int n_bodies = m_rigid_data->getNumBodies();
	fscanf(fp, "%d\n", &n_bodies);
	if (n_bodies <= 0) 
		{
		fclose(fp);
		return;
		}
		
	{
	ArrayHandle<unsigned int> body_size_handle(m_rigid_data->getBodySize(), access_location::host, access_mode::read);
	ArrayHandle<Scalar4> moment_inertia_handle(m_rigid_data->getMomentInertia(), access_location::host, access_mode::readwrite);
	
	ArrayHandle<Scalar4> com_handle(m_rigid_data->getCOM(), access_location::host, access_mode::readwrite);	
	ArrayHandle<Scalar4> orientation_handle(m_rigid_data->getOrientation(), access_location::host, access_mode::readwrite);
	ArrayHandle<Scalar4> ex_space_handle(m_rigid_data->getExSpace(), access_location::host, access_mode::readwrite);
	ArrayHandle<Scalar4> ey_space_handle(m_rigid_data->getEySpace(), access_location::host, access_mode::readwrite);
	ArrayHandle<Scalar4> ez_space_handle(m_rigid_data->getEzSpace(), access_location::host, access_mode::readwrite);
	
	ArrayHandle<Scalar4> particle_pos_handle(m_rigid_data->getParticlePos(), access_location::host, access_mode::readwrite);
	unsigned int particle_pos_pitch = m_rigid_data->getParticlePos().getPitch();
	
	for (unsigned int body = 0; body < n_bodies; body++)
		{
		fscanf(fp, "%f\t%f\t%f\n", &moment_inertia_handle.data[body].x, &moment_inertia_handle.data[body].y, &moment_inertia_handle.data[body].z);
		fscanf(fp, "%f\t%f\t%f\n", &com_handle.data[body].x, &com_handle.data[body].y, &com_handle.data[body].z);
		
		fscanf(fp, "%f\t%f\t%f\t%f\n", &orientation_handle.data[body].x, &orientation_handle.data[body].y, &orientation_handle.data[body].z, &orientation_handle.data[body].w);
		fscanf(fp, "%f\t%f\t%f\n", &ex_space_handle.data[body].x, &ex_space_handle.data[body].y, &ex_space_handle.data[body].z);
		fscanf(fp, "%f\t%f\t%f\n", &ey_space_handle.data[body].x, &ey_space_handle.data[body].y, &ey_space_handle.data[body].z);
		fscanf(fp, "%f\t%f\t%f\n", &ez_space_handle.data[body].x, &ez_space_handle.data[body].y, &ez_space_handle.data[body].z);
		
		unsigned int len = body_size_handle.data[body];
		for (unsigned int j = 0; j < len; j++)
			{
			unsigned int localidx = body * particle_pos_pitch + j;
			fscanf(fp, "%f\t%f\t%f\n", &particle_pos_handle.data[localidx].x, &particle_pos_handle.data[localidx].y, &particle_pos_handle.data[localidx].z);
			}

		}
	}	
	
	fclose(fp);
	
}

void export_SystemDefinition()
	{
	class_<SystemDefinition, boost::shared_ptr<SystemDefinition> >("SystemDefinition", init<>())
		.def(init<unsigned int, const BoxDim&, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, const ExecutionConfiguration&>())
		.def(init<unsigned int, const BoxDim&, unsigned int>())
		.def(init<const ParticleDataInitializer&, const ExecutionConfiguration&>())
		.def("getParticleData", &SystemDefinition::getParticleData)
		.def("getBondData", &SystemDefinition::getBondData)
		.def("getAngleData", &SystemDefinition::getAngleData)
		.def("getDihedralData", &SystemDefinition::getDihedralData)
		.def("getImproperData", &SystemDefinition::getImproperData)
		.def("getWallData", &SystemDefinition::getWallData)
		.def("getRigidData", &SystemDefinition::getRigidData)
		.def("getRigidData", &SystemDefinition::writeRestart)
		.def("getRigidData", &SystemDefinition::readRestart)
		;
	}

#ifdef WIN32
#pragma warning( pop )
#endif
