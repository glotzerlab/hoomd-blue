// Copyright (c) 2009-2019 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.


// Maintainer: joaander

#include "TwoStepRATTLEBD.h"
#include "hoomd/VectorMath.h"
#include "QuaternionMath.h"
#include "hoomd/HOOMDMath.h"

#include "hoomd/RandomNumbers.h"
#include "hoomd/RNGIdentifiers.h"
using namespace hoomd;

inline Scalar maxNorm(Scalar3 vec, Scalar resid)
    {
    Scalar vec_norm = sqrt(dot(vec,vec));
    Scalar abs_resid = fabs(resid);
    if ( vec_norm > abs_resid) return vec_norm;
    else return abs_resid;
    }

#ifdef ENABLE_MPI
#include "hoomd/HOOMDMPI.h"
#endif

namespace py = pybind11;
using namespace std;

/*! \file TwoStepRATTLEBD.h
    \brief Contains code for the TwoStepRATTLEBD class
    \Warning NDOF is still 3*(N_part-1) and not 2*(N_part-1)!!! Has to be considered in thermodynamic quantities calculations.
*/

/*! \param sysdef SystemDefinition this method will act on. Must not be NULL.
    \param group The group of particles this integration method is to work on
    \param manifold The manifold describing the constraint during the RATTLE integration method
    \param T Temperature set point as a function of time
    \param seed Random seed to use in generating random numbers
    \param use_lambda If true, gamma=lambda*diameter, otherwise use a per-type gamma via setGamma()
    \param lambda Scale factor to convert diameter to gamma
    \param noiseless_t If set true, there will be no translational noise (random force)
    \param noiseless_r If set true, there will be no rotational noise (random torque)
    \param eta Tolerance for the RATTLE iteration algorithm
*/
TwoStepRATTLEBD::TwoStepRATTLEBD(std::shared_ptr<SystemDefinition> sysdef,
                           std::shared_ptr<ParticleGroup> group,
                           std::shared_ptr<Manifold> manifold,
                           std::shared_ptr<Variant> T,
                           unsigned int seed,
                           bool use_lambda,
                           Scalar lambda,
                           bool noiseless_t,
                           bool noiseless_r,
                           Scalar eta
                           )
  : TwoStepLangevinBase(sysdef, group, T, seed, use_lambda, lambda), m_manifold(manifold),
    m_noiseless_t(noiseless_t), m_noiseless_r(noiseless_r), m_eta(eta)
    {
    m_exec_conf->msg->notice(5) << "Constructing TwoStepRATTLEBD" << endl;

    unsigned int group_size = m_group->getNumMembers();

    GPUArray<Scalar3> tmp_f_brownian(group_size, m_exec_conf);

    m_f_brownian.swap(tmp_f_brownian);

    ArrayHandle<Scalar3> h_f_brownian(m_f_brownian, access_location::host);

    for (unsigned int i = 0; i < group_size; i++)
        {
        h_f_brownian.data[i].x = 0;
        h_f_brownian.data[i].y = 0;
        h_f_brownian.data[i].z = 0;
        }
    }

TwoStepRATTLEBD::~TwoStepRATTLEBD()
    {
    m_exec_conf->msg->notice(5) << "Destroying TwoStepRATTLEBD" << endl;
    }

/*! \param timestep Current time step
    \post Particle positions are moved forward to timestep+1

    The integration method here is from the book "The Langevin and Generalised Langevin Approach to the Dynamics of
    Atomic, Polymeric and Colloidal Systems", chapter 6.
*/
void TwoStepRATTLEBD::integrateStepOne(unsigned int timestep)
    {
    unsigned int group_size = m_group->getNumMembers();

    const Scalar currentTemp = m_T->getValue(timestep);

    // profile this step
    if (m_prof)
        m_prof->push("BD step 1");

    const GlobalArray< Scalar4 >& net_force = m_pdata->getNetForce();
    ArrayHandle<Scalar4> h_vel(m_pdata->getVelocities(), access_location::host, access_mode::readwrite);
    ArrayHandle<Scalar4> h_pos(m_pdata->getPositions(), access_location::host, access_mode::readwrite);
    ArrayHandle<int3> h_image(m_pdata->getImages(), access_location::host, access_mode::readwrite);
    ArrayHandle<unsigned int> h_rtag(m_pdata->getRTags(), access_location::host, access_mode::read);

    ArrayHandle<Scalar4> h_net_force(net_force, access_location::host, access_mode::read);
    ArrayHandle<Scalar3> h_f_brownian(m_f_brownian, access_location::host, access_mode::read);
    ArrayHandle<Scalar> h_gamma(m_gamma, access_location::host, access_mode::read);
    ArrayHandle<Scalar> h_diameter(m_pdata->getDiameters(), access_location::host, access_mode::read);

    ArrayHandle<Scalar3> h_gamma_r(m_gamma_r, access_location::host, access_mode::read);
    ArrayHandle<Scalar4> h_orientation(m_pdata->getOrientationArray(), access_location::host, access_mode::readwrite);
    ArrayHandle<Scalar4> h_torque(m_pdata->getNetTorqueArray(), access_location::host, access_mode::readwrite);

    ArrayHandle<Scalar4> h_angmom(m_pdata->getAngularMomentumArray(), access_location::host, access_mode::readwrite);
    ArrayHandle<Scalar3> h_inertia(m_pdata->getMomentsOfInertiaArray(), access_location::host, access_mode::read);

    const BoxDim& box = m_pdata->getBox();

    // perform the first half step
    // r(t+deltaT) = r(t) + (Fc(t) + Fr)*deltaT/gamma
    // iterative: r(t+deltaT) = r(t+deltaT) - J^(-1)*residual
    // v(t+deltaT) = random distribution consistent with T
    for (unsigned int group_idx = 0; group_idx < group_size; group_idx++)
        {
        unsigned int ptag = m_group->getMemberTag(group_idx);
        unsigned int j = h_rtag.data[ptag];

        // Initialize the RNG
        RandomGenerator rng(RNGIdentifier::TwoStepBD, m_seed, ptag, timestep);

        Scalar gamma;
        if (m_use_lambda)
            gamma = m_lambda*h_diameter.data[j];
        else
            {
            unsigned int type = __scalar_as_int(h_pos.data[j].w);
            gamma = h_gamma.data[type];
            }
        Scalar deltaT_gamma = m_deltaT/gamma;

	    Scalar dx = (h_net_force.data[j].x + h_f_brownian.data[group_idx].x) * deltaT_gamma;
	    Scalar dy = (h_net_force.data[j].y + h_f_brownian.data[group_idx].y) * deltaT_gamma;
	    Scalar dz = (h_net_force.data[j].z + h_f_brownian.data[group_idx].z) * deltaT_gamma;

        h_pos.data[j].x += dx;
        h_pos.data[j].y += dy;
        h_pos.data[j].z += dz;

        // particles may have been moved slightly outside the box by the above steps, wrap them back into place
        box.wrap(h_pos.data[j], h_image.data[j]);

        // rotational random force and orientation quaternion updates
        if (m_aniso)
            {
            unsigned int type_r = __scalar_as_int(h_pos.data[j].w);
            Scalar3 gamma_r = h_gamma_r.data[type_r];
            if (gamma_r.x > 0 || gamma_r.y > 0 || gamma_r.z > 0)
                {
                vec3<Scalar> p_vec;
                quat<Scalar> q(h_orientation.data[j]);
                vec3<Scalar> t(h_torque.data[j]);
                vec3<Scalar> I(h_inertia.data[j]);

                bool x_zero, y_zero, z_zero;
                x_zero = (I.x < EPSILON); y_zero = (I.y < EPSILON); z_zero = (I.z < EPSILON);

                Scalar3 sigma_r = make_scalar3(fast::sqrt(Scalar(2.0)*gamma_r.x*currentTemp/m_deltaT),
                                               fast::sqrt(Scalar(2.0)*gamma_r.y*currentTemp/m_deltaT),
                                               fast::sqrt(Scalar(2.0)*gamma_r.z*currentTemp/m_deltaT));
                if (m_noiseless_r)
                    sigma_r = make_scalar3(0,0,0);

                // original Gaussian random torque
                // Gaussian random distribution is preferred in terms of preserving the exact math
                vec3<Scalar> bf_torque;
                bf_torque.x = NormalDistribution<Scalar>(sigma_r.x)(rng);
                bf_torque.y = NormalDistribution<Scalar>(sigma_r.y)(rng);
                bf_torque.z = NormalDistribution<Scalar>(sigma_r.z)(rng);

                if (x_zero) bf_torque.x = 0;
                if (y_zero) bf_torque.y = 0;
                if (z_zero) bf_torque.z = 0;

                // use the d_invamping by gamma_r and rotate back to lab frame
                // Notes For the Future: take special care when have anisotropic gamma_r
                // if aniso gamma_r, first rotate the torque into particle frame and divide the different gamma_r
                // and then rotate the "angular velocity" back to lab frame and integrate
                bf_torque = rotate(q, bf_torque);

                // do the integration for quaternion
                q += Scalar(0.5) * m_deltaT * ((t + bf_torque) / vec3<Scalar>(gamma_r)) * q ;
                q = q * (Scalar(1.0) / slow::sqrt(norm2(q)));
                h_orientation.data[j] = quat_to_scalar4(q);

                // draw a new random ang_mom for particle j in body frame
                p_vec.x = NormalDistribution<Scalar>(fast::sqrt(currentTemp * I.x))(rng);
                p_vec.y = NormalDistribution<Scalar>(fast::sqrt(currentTemp * I.y))(rng);
                p_vec.z = NormalDistribution<Scalar>(fast::sqrt(currentTemp * I.z))(rng);
                if (x_zero) p_vec.x = 0;
                if (y_zero) p_vec.y = 0;
                if (z_zero) p_vec.z = 0;

                // !! Note this isn't well-behaving in 2D,
                // !! because may have effective non-zero ang_mom in x,y

                // store ang_mom quaternion
                quat<Scalar> p = Scalar(2.0) * q * p_vec;
                h_angmom.data[j] = quat_to_scalar4(p);
                }
            }
        }
    //exit(0);
    // done profiling
    if (m_prof)
        m_prof->pop();
    }

/*! \param timestep Current time step
*/
void TwoStepRATTLEBD::integrateStepTwo(unsigned int timestep)
    {
    // there is no step 2 in Brownian dynamics.
    }


void TwoStepRATTLEBD::IncludeRATTLEForce(unsigned int timestep)
    {

    unsigned int group_size = m_group->getNumMembers();

    const Scalar currentTemp = m_T->getValue(timestep);

    const GlobalArray< Scalar4 >& net_force = m_pdata->getNetForce();
    const GlobalArray<Scalar>&  net_virial = m_pdata->getNetVirial();
    ArrayHandle<Scalar4> h_vel(m_pdata->getVelocities(), access_location::host, access_mode::read);
    ArrayHandle<Scalar4> h_pos(m_pdata->getPositions(), access_location::host, access_mode::read);
    ArrayHandle<unsigned int> h_rtag(m_pdata->getRTags(), access_location::host, access_mode::read);

    ArrayHandle<Scalar4> h_net_force(net_force, access_location::host, access_mode::readwrite);
    ArrayHandle<Scalar> h_net_virial(net_virial, access_location::host, access_mode::readwrite);
    ArrayHandle<Scalar3> h_f_brownian(m_f_brownian, access_location::host, access_mode::readwrite);
    ArrayHandle<Scalar> h_gamma(m_gamma, access_location::host, access_mode::read);
    ArrayHandle<Scalar> h_diameter(m_pdata->getDiameters(), access_location::host, access_mode::read);

    unsigned int net_virial_pitch = net_virial.getPitch();

    unsigned int maxiteration = 10;

    // perform the first half step
    // r(t+deltaT) = r(t) + (Fc(t) + Fr)*deltaT/gamma
    // iterative: r(t+deltaT) = r(t+deltaT) - J^(-1)*residual
    // v(t+deltaT) = random distribution consistent with T
    for (unsigned int group_idx = 0; group_idx < group_size; group_idx++)
        {
        unsigned int ptag = m_group->getMemberTag(group_idx);
        unsigned int j = h_rtag.data[ptag];

        // Initialize the RNG
        RandomGenerator rng(RNGIdentifier::TwoStepBD, m_seed, ptag, timestep);

        Scalar gamma;
        if (m_use_lambda)
            gamma = m_lambda*h_diameter.data[j];
        else
            {
            unsigned int type = __scalar_as_int(h_pos.data[j].w);
            gamma = h_gamma.data[type];
            }
        Scalar deltaT_gamma = m_deltaT/gamma;


	    Scalar3 next_pos;
	    next_pos.x = h_pos.data[j].x;
	    next_pos.y = h_pos.data[j].y;
	    next_pos.z = h_pos.data[j].z;

	    Scalar3 normal = m_manifold->derivative(next_pos);

        // draw a new random velocity for particle j
        Scalar mass =  h_vel.data[j].w;
        Scalar sigma1 = fast::sqrt(currentTemp/mass);
        NormalDistribution<Scalar> norm(sigma1);

        Scalar3 vec_rand;
        vec_rand.x =norm(rng);
        vec_rand.y =norm(rng);
        vec_rand.z =norm(rng);

        Scalar norm_normal = 1.0/fast::sqrt(normal.x*normal.x+normal.y*normal.y+normal.z*normal.z);

        normal.x = norm_normal*normal.x;
        normal.y = norm_normal*normal.y;
        normal.z = norm_normal*normal.z;

        Scalar rand_norm = vec_rand.x*normal.x+ vec_rand.y*normal.y + vec_rand.z*normal.z;
        vec_rand.x -= rand_norm*normal.x;
        vec_rand.y -= rand_norm*normal.y;
        vec_rand.z -= rand_norm*normal.z;

        h_vel.data[j].x = vec_rand.x;
        h_vel.data[j].y = vec_rand.y;
        h_vel.data[j].z = vec_rand.z;


        Scalar rx, ry, rz, coeff;

        if(currentTemp > 0)
	        {
	    	// compute the random force
	    	UniformDistribution<Scalar> uniform(Scalar(-1), Scalar(1));
	    	rx = uniform(rng);
	    	ry = uniform(rng);
	    	rz = uniform(rng);

	    	Scalar3 proj = normal;
	    	Scalar proj_norm = 1.0/slow::sqrt(proj.x*proj.x+proj.y*proj.y+proj.z*proj.z);
	    	proj.x *= proj_norm;
	    	proj.y *= proj_norm;
	    	proj.z *= proj_norm;

	    	Scalar proj_r = rx*proj.x + ry*proj.y + rz*proj.z;

	    	rx = rx - proj_r*proj.x;
	    	ry = ry - proj_r*proj.y;
	    	rz = rz - proj_r*proj.z;

	    	// compute the bd force (the extra factor of 3 is because <rx^2> is 1/3 in the uniform -1,1 distribution
	    	// it is not the dimensionality of the system
	    	coeff = fast::sqrt(Scalar(6.0)*currentTemp/deltaT_gamma);
	    	if (m_noiseless_t)
	    	    coeff = Scalar(0.0);
 	        }
	    else
	        {
        	rx = 0;
           	ry = 0;
           	rz = 0;
          	coeff = 0;
	        }

        Scalar Fr_x = rx*coeff;
        Scalar Fr_y = ry*coeff;
        Scalar Fr_z = rz*coeff;

            // update position
	    Scalar mu = 0.0;
            
	    Scalar inv_alpha = -deltaT_gamma;
	    inv_alpha = Scalar(1.0)/inv_alpha;


	    Scalar3 residual;
	    Scalar resid;
	    unsigned int iteration = 0;

	    do
	    {
	        iteration++;
	        residual.x = h_pos.data[j].x - next_pos.x + (h_net_force.data[j].x + Fr_x - mu*normal.x) * deltaT_gamma;
	        residual.y = h_pos.data[j].y - next_pos.y + (h_net_force.data[j].y + Fr_y - mu*normal.y) * deltaT_gamma;
	        residual.z = h_pos.data[j].z - next_pos.z + (h_net_force.data[j].z + Fr_z - mu*normal.z) * deltaT_gamma;
	        resid = m_manifold->implicit_function(next_pos);

            Scalar3 next_normal = m_manifold->derivative(next_pos);

	        Scalar nndotr = dot(next_normal,residual);
	        Scalar nndotn = dot(next_normal,normal);
	        Scalar beta = (resid + nndotr)/nndotn;

            next_pos.x = next_pos.x - beta*normal.x + residual.x;   
            next_pos.y = next_pos.y - beta*normal.y + residual.y;   
            next_pos.z = next_pos.z - beta*normal.z + residual.z;
	        mu = mu - beta*inv_alpha;
	    
	    } while (maxNorm(residual,resid) > m_eta && iteration < maxiteration );

	    h_net_force.data[j].x -= mu*normal.x;
	    h_net_force.data[j].y -= mu*normal.y;
	    h_net_force.data[j].z -= mu*normal.z;

        h_net_virial.data[0*net_virial_pitch+j] -= mu*normal.x*h_pos.data[j].x;
        h_net_virial.data[1*net_virial_pitch+j] -= 0.5*mu*(normal.y*h_pos.data[j].x + normal.x*h_pos.data[j].y);
        h_net_virial.data[2*net_virial_pitch+j] -= 0.5*mu*(normal.z*h_pos.data[j].x + normal.x*h_pos.data[j].z);
        h_net_virial.data[3*net_virial_pitch+j] -= mu*normal.y*h_pos.data[j].y;
        h_net_virial.data[4*net_virial_pitch+j] -= 0.5*mu*(normal.y*h_pos.data[j].z + normal.z*h_pos.data[j].y);
        h_net_virial.data[5*net_virial_pitch+j] -= mu*normal.z*h_pos.data[j].z;

	    h_f_brownian.data[group_idx].x = Fr_x;
	    h_f_brownian.data[group_idx].y = Fr_y;
	    h_f_brownian.data[group_idx].z = Fr_z;

        }
    }

/*! \param query_group Group over which to count (translational) degrees of freedom.
    A majority of the integration methods add D degrees of freedom per particle in \a query_group that is also in the
    group assigned to the method. Hence, the base class IntegrationMethodTwoStep will implement that counting.
    Derived classes can override if needed.
*/
unsigned int TwoStepRATTLEBD::getNDOF(std::shared_ptr<ParticleGroup> query_group)
    {
    // get the size of the intersection between query_group and m_group
    unsigned int intersect_size = ParticleGroup::groupIntersection(query_group, m_group)->getNumMembersGlobal();

    return ( m_sysdef->getNDimensions() - 1 ) * intersect_size;
    }

void export_TwoStepRATTLEBD(py::module& m)
    {
    py::class_<TwoStepRATTLEBD, std::shared_ptr<TwoStepRATTLEBD> >(m, "TwoStepRATTLEBD", py::base<TwoStepLangevinBase>())
    .def(py::init< std::shared_ptr<SystemDefinition>,
                            std::shared_ptr<ParticleGroup>,
			    std::shared_ptr<Manifold>,
                            std::shared_ptr<Variant>,
                            unsigned int,
                            bool,
                            Scalar,
                            bool,
                            bool,
			    Scalar>())
        ;
    }
