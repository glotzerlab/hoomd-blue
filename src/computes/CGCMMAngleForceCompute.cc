
#ifdef WIN32
#pragma warning( push )
#pragma warning( disable : 4244 )
#endif

#include <boost/python.hpp>
using namespace boost::python;

#include "CGCMMAngleForceCompute.h"

#include <iostream>
#include <sstream>
#include <stdexcept>
#include <math.h>

using namespace std;

/*! \param SMALL a relatively small number
*/
#define SMALL 0.001f

/*! \file CGCMMAngleForceCompute.cc
	\brief Contains code for the CGCMMAngleForceCompute class
*/

/*! \param pdata Particle data to compute forces on
	\post Memory is allocated, and forces are zeroed.
*/
CGCMMAngleForceCompute::CGCMMAngleForceCompute(boost::shared_ptr<ParticleData> pdata) :	ForceCompute(pdata),
	m_K(NULL), m_t_0(NULL), m_eps(NULL), m_sigma(NULL), m_rcut(NULL), m_cg_type(NULL)
	{
	// access the angle data for later use
	m_CGCMMangle_data = m_pdata->getAngleData();
	
	// check for some silly errors a user could make 
	if (m_CGCMMangle_data->getNAngleTypes() == 0)
		{
		cout << endl << "***Error! No CGCMMangle types specified" << endl << endl;
		throw runtime_error("Error initializing CGCMMAngleForceCompute");
		}
		
	// allocate the parameters
	m_K = new Scalar[m_CGCMMangle_data->getNAngleTypes()];
	m_t_0 = new Scalar[m_CGCMMangle_data->getNAngleTypes()];
	m_eps = new Scalar[m_CGCMMangle_data->getNAngleTypes()];
	m_sigma = new Scalar[m_CGCMMangle_data->getNAngleTypes()];
	m_rcut = new Scalar[m_CGCMMangle_data->getNAngleTypes()];
	m_cg_type = new unsigned int[m_CGCMMangle_data->getNAngleTypes()];
	
	// zero parameters
	memset(m_K, 0, sizeof(Scalar) * m_CGCMMangle_data->getNAngleTypes());
	memset(m_t_0, 0, sizeof(Scalar) * m_CGCMMangle_data->getNAngleTypes());
	memset(m_eps, 0, sizeof(Scalar) * m_CGCMMangle_data->getNAngleTypes());
	memset(m_sigma, 0, sizeof(Scalar) * m_CGCMMangle_data->getNAngleTypes());
	memset(m_rcut, 0, sizeof(Scalar) * m_CGCMMangle_data->getNAngleTypes());
	memset(m_cg_type, 0, sizeof(unsigned int) * m_CGCMMangle_data->getNAngleTypes());

        prefact[0] = 0.0;
        prefact[1] = 6.75;
        prefact[2] = 2.59807621135332;
        prefact[3] = 4.0;

        cgPow1[0]  = 0.0;
        cgPow1[1]  = 9.0;
        cgPow1[2]  = 12.0;
        cgPow1[3]  = 12.0;

        cgPow2[0]  = 0.0;
        cgPow2[1]  = 6.0;
        cgPow2[2]  = 4.0;
        cgPow2[3]  = 6.0;

	}
	
CGCMMAngleForceCompute::~CGCMMAngleForceCompute()
	{
	delete[] m_K;
	delete[] m_t_0;
        delete[] m_cg_type;
        delete[] m_eps;
        delete[] m_sigma;
        delete[] m_rcut;
        m_K = NULL;
        m_t_0 = NULL;
        m_cg_type = NULL;
        m_eps = NULL;
        m_sigma = NULL;
        m_rcut = NULL;
	}

/*! \param type Type of the angle to set parameters for
	\param K Stiffness parameter for the force computation
	\param t_0 Equilibrium angle in radians for the force computation
        \param cg_type the type of course grained angle
        \param eps the epsilon parameter for the 1-3 repulsion term
        \param sigma the sigma parameter for the 1-3 repulsion term
	
	Sets parameters for the potential of a particular angle type
*/
void CGCMMAngleForceCompute::setParams(unsigned int type, Scalar K, Scalar t_0, unsigned int cg_type, Scalar eps, Scalar sigma)
	{
	// make sure the type is valid
	if (type >= m_CGCMMangle_data->getNAngleTypes())
		{
		cout << endl << "***Error! Invalid CGCMMangle typee specified" << endl << endl;
		throw runtime_error("Error setting parameters in CGCMMAngleForceCompute");
		}
	
        const float myPow1 = cgPow1[cg_type];
        const float myPow2 = cgPow2[cg_type];

        Scalar my_rcut = sigma*exp(1.0f/(myPow1-myPow2)*log(myPow1/myPow2));

	m_K[type] = K;
	m_t_0[type] = t_0;
        m_cg_type[type] = cg_type;
        m_eps[type] = eps;
        m_sigma[type] = sigma;
        m_rcut[type] = my_rcut;

	// check for some silly errors a user could make 
        if (cg_type < 0 || cg_type > 3)
                cout << "***Warning! Unrecognized cg_type specified for harmonic CGCMMangle" << endl;
	if (K <= 0)
		cout << "***Warning! K <= 0 specified for harmonic CGCMMangle" << endl;
	if (t_0 <= 0)
		cout << "***Warning! t_0 <= 0 specified for harmonic CGCMMangle" << endl;
	if (eps <= 0)
		cout << "***Warning! eps <= 0 specified for harmonic CGCMMangle" << endl;
 	if (sigma <= 0)
		cout << "***Warning! sigma <= 0 specified for harmonic CGCMMangle" << endl;
  
	}

/*! CGCMMAngleForceCompute provides
	- \c harmonic_energy
*/
std::vector< std::string > CGCMMAngleForceCompute::getProvidedLogQuantities()
	{
	vector<string> list;
	list.push_back("CGCMMangle_harmonic_energy");
	return list;
	}

/*! \param quantity Name of the quantity to get the log value of
	\param timestep Current time step of the simulation
*/
Scalar CGCMMAngleForceCompute::getLogValue(const std::string& quantity, unsigned int timestep)
	{
	if (quantity == string("CGCMMangle_harmonic_energy"))
		{
		compute(timestep);
		return calcEnergySum();
		}
	else
		{
		cerr << endl << "***Error! " << quantity << " is not a valid log quantity for CGCMMAngleForceCompute" << endl << endl;
		throw runtime_error("Error getting log value");
		}
	}	

/*! Actually perform the force computation
	\param timestep Current time step
 */
void CGCMMAngleForceCompute::computeForces(unsigned int timestep)
 	{
	if (m_prof) m_prof->push("CGCMMAngle");

 	assert(m_pdata);
 	// access the particle data arrays
	ParticleDataArraysConst arrays = m_pdata->acquireReadOnly();
	// there are enough other checks on the input data: but it doesn't hurt to be safe
	assert(m_fx);
	assert(m_fy);
	assert(m_fz);
	assert(m_pe);
	assert(arrays.x);
	assert(arrays.y);
	assert(arrays.z);

	// get a local copy of the simulation box too
	const BoxDim& box = m_pdata->getBox();
	// sanity check
	assert(box.xhi > box.xlo && box.yhi > box.ylo && box.zhi > box.zlo);

	// precalculate box lenghts
	Scalar Lx = box.xhi - box.xlo;
	Scalar Ly = box.yhi - box.ylo;
	Scalar Lz = box.zhi - box.zlo;
	Scalar Lx2 = Lx / Scalar(2.0);
	Scalar Ly2 = Ly / Scalar(2.0);
	Scalar Lz2 = Lz / Scalar(2.0);

        // allocate forces
        Scalar fab[3], fcb[3];
        Scalar fac;

        Scalar eac;
        Scalar vacX,vacY,vacZ;
	
	// need to start from a zero force
	// MEM TRANSFER: 5*N Scalars
	memset((void*)m_fx, 0, sizeof(Scalar) * m_pdata->getN());
	memset((void*)m_fy, 0, sizeof(Scalar) * m_pdata->getN());
	memset((void*)m_fz, 0, sizeof(Scalar) * m_pdata->getN());
	memset((void*)m_pe, 0, sizeof(Scalar) * m_pdata->getN());
	memset((void*)m_virial, 0, sizeof(Scalar) * m_pdata->getN());
	
	// for each of the angles
	const unsigned int size = (unsigned int)m_CGCMMangle_data->getNumAngles(); 
	for (unsigned int i = 0; i < size; i++)
		{
		// lookup the tag of each of the particles participating in the angle
		const Angle& angle = m_CGCMMangle_data->getAngle(i);
		assert(angle.a < m_pdata->getN());
		assert(angle.b < m_pdata->getN());
		assert(angle.c < m_pdata->getN());
				
		// transform a, b, and c into indicies into the particle data arrays
		// MEM TRANSFER: 6 ints
		unsigned int idx_a = arrays.rtag[angle.a];
		unsigned int idx_b = arrays.rtag[angle.b];
		unsigned int idx_c = arrays.rtag[angle.c];
		assert(idx_a < m_pdata->getN());
		assert(idx_b < m_pdata->getN());
		assert(idx_c < m_pdata->getN());

		// calculate d\vec{r}
		// MEM_TRANSFER: 18 Scalars / FLOPS 9
		Scalar dxab = arrays.x[idx_a] - arrays.x[idx_b];
		Scalar dyab = arrays.y[idx_a] - arrays.y[idx_b];
		Scalar dzab = arrays.z[idx_a] - arrays.z[idx_b];

		Scalar dxcb = arrays.x[idx_c] - arrays.x[idx_b];
		Scalar dycb = arrays.y[idx_c] - arrays.y[idx_b];
		Scalar dzcb = arrays.z[idx_c] - arrays.z[idx_b];

		Scalar dxac = arrays.x[idx_a] - arrays.x[idx_c]; // used for the 1-3 JL interaction
		Scalar dyac = arrays.y[idx_a] - arrays.y[idx_c];
		Scalar dzac = arrays.z[idx_a] - arrays.z[idx_c];

		// if the a->b vector crosses the box, pull it back
		// (total FLOPS: 27 (worst case: first branch is missed, the 2nd is taken and the add is done, for each))


		if (dxab >= Lx2)
			dxab -= Lx;
		else
		if (dxab < -Lx2)
			dxab += Lx;
		
		if (dyab >= Ly2)
			dyab -= Ly;
		else
		if (dyab < -Ly2)
			dyab += Ly;
		
		if (dzab >= Lz2)
			dzab -= Lz;
		else
		if (dzab < -Lz2)
			dzab += Lz;

		// if the b->c vector crosses the box, pull it back
		if (dxcb >= Lx2)
			dxcb -= Lx;
		else
		if (dxcb < -Lx2)
			dxcb += Lx;
		
		if (dycb >= Ly2)
			dycb -= Ly;
		else
		if (dycb < -Ly2)
			dycb += Ly;
		
		if (dzcb >= Lz2)
			dzcb -= Lz;
		else
		if (dzcb < -Lz2)
			dzcb += Lz;

		// if the a->c vector crosses the box, pull it back
		if (dxac >= Lx2)
			dxac -= Lx;
		else
		if (dxac < -Lx2)
			dxac += Lx;
		
		if (dyac >= Ly2)
			dyac -= Ly;
		else
		if (dyac < -Ly2)
			dyac += Ly;
		
		if (dzac >= Lz2)
			dzac -= Lz;
		else
		if (dzac < -Lz2)
			dzac += Lz;


		// sanity check
		assert((dxab >= box.xlo && dxab < box.xhi) && (dxcb >= box.xlo && dxcb < box.xhi) && (dxac >= box.xlo && dxac < box.xhi));
		assert((dyab >= box.ylo && dyab < box.yhi) && (dycb >= box.ylo && dycb < box.yhi) && (dyac >= box.ylo && dyac < box.yhi));
		assert((dzab >= box.zlo && dzab < box.zhi) && (dzcb >= box.zlo && dzcb < box.zhi) && (dzac >= box.zlo && dzac < box.zhi));

		// on paper, the formula turns out to be: F = K*\vec{r} * (r_0/r - 1)
		// FLOPS: 14 / MEM TRANSFER: 2 Scalars


		// FLOPS: 42 / MEM TRANSFER: 6 Scalars
                Scalar rsqab = dxab*dxab+dyab*dyab+dzab*dzab;
                Scalar rab = sqrt(rsqab);
                Scalar rsqcb = dxcb*dxcb+dycb*dycb+dzcb*dzcb;
                Scalar rcb = sqrt(rsqcb);
                Scalar rsqac = dxac*dxac+dyac*dyac+dzac*dzac;
                Scalar rac = sqrt(rsqac);

                Scalar c_abbc = dxab*dxcb+dyab*dycb+dzab*dzcb;
                c_abbc /= rab*rcb;

                if (c_abbc > 1.0) c_abbc = 1.0;
                if (c_abbc < -1.0) c_abbc = -1.0;

                Scalar s_abbc = sqrt(1.0 - c_abbc*c_abbc);
                if (s_abbc < SMALL) s_abbc = SMALL;
                s_abbc = 1.0/s_abbc;

                //////////////////////////////////////////
                // THIS CODE DOES THE 1-3 LJ repulsions //
                //////////////////////////////////////////////////////////////////////////////
                fac = 0.0f;
                eac = 0.0f;
                vacX = vacY = vacZ = 0.0f;
                if (rac < m_rcut[angle.type])
                {
                  const unsigned int cg_type = m_cg_type[angle.type];
                  const float cg_pow1 = cgPow1[cg_type];
                  const float cg_pow2 = cgPow1[cg_type];
                  const float cg_pref = prefact[cg_type];

                  const float cg_ratio = m_sigma[angle.type]/rac;
                  const float cg_eps   = m_eps[angle.type];

                  fac = cg_pref*cg_eps / rsqac * (cg_pow1*pow(cg_ratio,cg_pow1) - cg_pow2*pow(cg_ratio,cg_pow2));
                  eac = cg_eps + cg_pref*cg_eps * (pow(cg_ratio,cg_pow1) - pow(cg_ratio,cg_pow2));

                  vacX = fac * dxac*dxac;
                  vacY = fac * dyac*dyac;
                  vacZ = fac * dzac*dzac;

                }
                //////////////////////////////////////////////////////////////////////////////

                // actually calculate the force
                Scalar dth = acos(c_abbc) - m_t_0[angle.type];
                Scalar tk = m_K[angle.type]*dth;
                
                Scalar a = -2.0 * tk * s_abbc;
                Scalar a11 = a*c_abbc/rsqab;
                Scalar a12 = -a / (rab*rcb);              
                Scalar a22 = a*c_abbc / rsqcb;

                fab[0] = a11*dxab + a12*dxcb;
                fab[1] = a11*dyab + a12*dycb;
                fab[2] = a11*dzab + a12*dzcb; 

                fcb[0] = a22*dxcb + a12*dxab;
                fcb[1] = a22*dycb + a12*dyab; 
                fcb[2] = a22*dzcb + a12*dzab; 

                // compute 1/3 of the energy, 1/3 for each atom in the angle
                Scalar angle_eng = (tk*dth + eac)*Scalar(1.0/6.0);

                // do we really need a virial here for harmonic angles?
                // ... if not, this may be wrong...
                Scalar vx = dxab*fab[0] + dxcb*fcb[0] + vacX; 
                Scalar vy = dyab*fab[1] + dycb*fcb[1] + vacY;
                Scalar vz = dzab*fab[2] + dzcb*fcb[2] + vacZ;

                Scalar angle_virial = Scalar(1.0/6.0)*(vx + vy + vz);

                // Now, apply the force to each individual atom a,b,c, and accumlate the energy/virial
		m_fx[idx_a] += fab[0] + fac*dxac;
		m_fy[idx_a] += fab[1] + fac*dyac;
		m_fz[idx_a] += fab[2] + fac*dzac;
                m_pe[idx_a] += angle_eng;
                m_virial[idx_a] += angle_virial;

		m_fx[idx_b] -= fab[0] + fcb[0];
		m_fy[idx_b] -= fab[1] + fcb[1];
		m_fz[idx_b] -= fab[2] + fcb[2];
                m_pe[idx_b] += angle_eng;
                m_virial[idx_b] += angle_virial;

		m_fx[idx_c] += fcb[0] - fac*dxac;
		m_fy[idx_c] += fcb[1] - fac*dyac;
		m_fz[idx_c] += fcb[2] - fac*dzac;
                m_pe[idx_c] += angle_eng;
                m_virial[idx_c] += angle_virial;


		}

	m_pdata->release();

	#ifdef ENABLE_CUDA
	// the data is now only up to date on the CPU
	m_data_location = cpu;
	#endif

        // ALL TIMING STUFF HAS BEEN COMMENTED OUT... if you uncomment, re-count all memtransfers and flops
	//int64_t flops = size*(3 + 9 + 14 + 2 + 16);
	//int64_t mem_transfer = m_pdata->getN() * 5 * sizeof(Scalar) + size * ( (4)*sizeof(unsigned int) + (6+2+20)*sizeof(Scalar) );
	//if (m_prof) m_prof->pop(flops, mem_transfer);
	}
	
void export_CGCMMAngleForceCompute()
	{
	class_<CGCMMAngleForceCompute, boost::shared_ptr<CGCMMAngleForceCompute>, bases<ForceCompute>, boost::noncopyable >
		("CGCMMAngleForceCompute", init< boost::shared_ptr<ParticleData> >())
		.def("setParams", &CGCMMAngleForceCompute::setParams)
		;
	}

#ifdef WIN32
#pragma warning( pop )
#endif
