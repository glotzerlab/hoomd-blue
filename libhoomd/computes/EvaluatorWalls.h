/*
Highly Optimized Object-oriented Many-particle Dynamics -- Blue Edition
(HOOMD-blue) Open Source Software License Copyright 2009-2015 The Regents of
the University of Michigan All rights reserved.

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

* All publications and presentations based on HOOMD-blue, including any reports
or published results obtained, in whole or in part, with HOOMD-blue, will
acknowledge its use according to the terms posted at the time of submission on:
http://codeblue.umich.edu/hoomd-blue/citations.html

* Any electronic documents citing HOOMD-Blue will link to the HOOMD-Blue website:
http://codeblue.umich.edu/hoomd-blue/

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

// Maintainer: jproc

/*! \file EvaluatorWalls.h
    \brief Executes an external field potential of several evaluator types for each wall in the system.
 */

#ifndef __EVALUATOR_WALLS_H__
#define __EVALUATOR_WALLS_H__

// need to declare these class methods with __device__ qualifiers when building in nvcc
// DEVICE is __host__ __device__ when included in nvcc and blank when included into the host compiler
#ifdef NVCC
#define DEVICE __device__
#else
#define DEVICE
#endif

// sets the max numbers for each wall geometry type
// if modified, the same number should be modified in the python module
#define MAX_N_SWALLS 20
#define MAX_N_CWALLS 20
#define MAX_N_PWALLS 60
// doesn't work without being in a boost module
// boost::python::scope().attr("_max_n_sphere_walls") = MAX_N_SWALLS;
// boost::python::scope().attr("_max_n_cylinder_walls") = MAX_N_CWALLS;
// boost::python::scope().attr("_max_n_plane_walls") = MAX_N_PWALLS;

#include "Compute.h"
#include <boost/shared_ptr.hpp>
#include <boost/python.hpp>
#include <boost/bind.hpp>
#include "HOOMDMath.h"
#include "VectorMath.h"
#include "WallData.h"

struct wall_type{
	SphereWall 		Spheres[MAX_N_SWALLS];
	CylinderWall 	Cylinders[MAX_N_CWALLS];
	PlaneWall 		Planes[MAX_N_PWALLS];
	unsigned int 	numSpheres;
	unsigned int 	numCylinders;
	unsigned int 	numPlanes;

};

template<class evaluator>
class EvaluatorWalls
	{
	public:
        typedef struct{
        	typename evaluator::param_type params;
			Scalar rcutsq;
        	Scalar ronsq;
        } param_type;

		typedef wall_type field_type;

		DEVICE EvaluatorWalls(Scalar3 pos, const BoxDim& box, const param_type& p, const field_type& f) : m_pos(pos), m_field(f), m_params(p)
			{
			}

		//! Test if evaluator needs Diameter
		DEVICE static bool needsDiameter()
			{
			return evaluator::needsDiameter();
			}
		//! Accept the optional diameter value
		/*! \param di Diameter of particle i
		*/
		DEVICE void setDiameter(Scalar diameter)
			{
			di = diameter;
			}

		DEVICE void evalForceEnergyAndVirial(Scalar3& F, Scalar& energy, Scalar* virial)
			{
			F.x = Scalar(0.0);
            F.y = Scalar(0.0);
            F.z = Scalar(0.0);
            energy = Scalar(0.0);
            for (unsigned int i = 0; i < 6; i++)
                virial[i] = Scalar(0.0);

			// convert type as little as possible
			vec3<Scalar> position = vec3<Scalar>(m_pos);
			vec3<Scalar> dxv;
			// initialize virial
			bool energy_shift = true; //Forces V(r) at r_cut to be continuous

			for (unsigned int k = 0; k < m_field.numSpheres; k++)
				{
				dxv = vecInsPtToWall(m_field.Spheres[k], position);
				Scalar3 dx = -vec_to_scalar3(dxv);

				// calculate r_ij squared (FLOPS: 5)
				Scalar rsq = dot(dx, dx);
				if (rsq > m_params.ronsq)
					{
					// compute the force and potential energy
					Scalar force_divr = Scalar(0.0);
					Scalar pair_eng = Scalar(0.0);
					evaluator eval(rsq, m_params.rcutsq, m_params.params);
					if (evaluator::needsDiameter())
						eval.setDiameter(di, Scalar(0.0));
					bool evaluated = eval.evalForceAndEnergy(force_divr, pair_eng, energy_shift);

					if (evaluated)
						{

						//Scalar force_div2r = force_divr; // removing half since the other "particle" won't be represented * Scalar(0.5);
						// add the force, potential energy and virial to the particle i
						// (FLOPS: 8)
						F += dx*force_divr;
						energy += pair_eng; // removing half since the other "particle" won't be represented * Scalar(0.5);
						virial[0] += force_divr*dx.x*dx.x;
						virial[1] += force_divr*dx.x*dx.y;
						virial[2] += force_divr*dx.x*dx.z;
						virial[3] += force_divr*dx.y*dx.y;
						virial[4] += force_divr*dx.y*dx.z;
						virial[5] += force_divr*dx.z*dx.z;
						}
					}
				}
			for (unsigned int k = 0; k < m_field.numCylinders; k++)
				{
				dxv = vecInsPtToWall(m_field.Cylinders[k], position);
				Scalar3 dx = -vec_to_scalar3(dxv);

				// calculate r_ij squared (FLOPS: 5)
				Scalar rsq = dot(dx, dx);
				if (rsq > m_params.ronsq)
					{
					// compute the force and potential energy
					Scalar force_divr = Scalar(0.0);
					Scalar pair_eng = Scalar(0.0);
					evaluator eval(rsq, m_params.rcutsq, m_params.params);
					if (evaluator::needsDiameter())
						eval.setDiameter(di, Scalar(0.0));
					bool evaluated = eval.evalForceAndEnergy(force_divr, pair_eng, energy_shift);

					if (evaluated)
						{

						//Scalar force_div2r = force_divr; // removing half since the other "particle" won't be represented * Scalar(0.5);
						// add the force, potential energy and virial to the particle i
						// (FLOPS: 8)
						F += dx*force_divr;
						energy += pair_eng; // removing half since the other "particle" won't be represented * Scalar(0.5);
						virial[0] += force_divr*dx.x*dx.x;
						virial[1] += force_divr*dx.x*dx.y;
						virial[2] += force_divr*dx.x*dx.z;
						virial[3] += force_divr*dx.y*dx.y;
						virial[4] += force_divr*dx.y*dx.z;
						virial[5] += force_divr*dx.z*dx.z;
						}
					}
				}
			for (unsigned int k = 0; k < m_field.numPlanes; k++)
				{
				dxv = vecInsPtToWall(m_field.Planes[k], position);
				Scalar3 dx = -vec_to_scalar3(dxv);

				// calculate r_ij squared (FLOPS: 5)
				Scalar rsq = dot(dx, dx);
				if (rsq > m_params.ronsq)
					{
					// compute the force and potential energy
					Scalar force_divr = Scalar(0.0);
					Scalar pair_eng = Scalar(0.0);
					evaluator eval(rsq, m_params.rcutsq, m_params.params);
					if (evaluator::needsDiameter())
						eval.setDiameter(di, Scalar(0.0));
					bool evaluated = eval.evalForceAndEnergy(force_divr, pair_eng, energy_shift);

					if (evaluated)
						{

						//Scalar force_div2r = force_divr; // removing half since the other "particle" won't be represented * Scalar(0.5);
						// add the force, potential energy and virial to the particle i
						// (FLOPS: 8)
						F += dx*force_divr;
						energy += pair_eng; // removing half since the other "particle" won't be represented * Scalar(0.5);
						virial[0] += force_divr*dx.x*dx.x;
						virial[1] += force_divr*dx.x*dx.y;
						virial[2] += force_divr*dx.x*dx.z;
						virial[3] += force_divr*dx.y*dx.y;
						virial[4] += force_divr*dx.y*dx.z;
						virial[5] += force_divr*dx.z*dx.z;
						}
					}
				}
			};

        #ifndef NVCC
        //! Get the name of this potential
        /*! \returns The potential name. Must be short and all lowercase, as this is the name energies will be logged as
            via analyze.log.
        */
        static std::string getName()
            {
            return std::string("walls_") + evaluator::getName();
            }
        #endif

    protected:
        Scalar3 		m_pos;                //!< particle position
        field_type 		m_field;			  //!< contains all information about the walls.
        param_type 		m_params;
		Scalar			di;
	};

template < class evaluator >
typename EvaluatorWalls<evaluator>::param_type make_wall_params(typename evaluator::param_type p, Scalar rcutsq, Scalar ronsq)
	{
	typename EvaluatorWalls<evaluator>::param_type params;
	params.params = p;
	params.rcutsq = rcutsq;
	params.ronsq = ronsq;
	return params;
	const boost::shared_ptr<ParticleData> m_pdata;      //!< The particle data this compute is associated with
	}

template< class evaluator >
void export_wall_params_helpers()
	{
	class_<typename EvaluatorWalls<evaluator>::param_type , boost::shared_ptr<typename EvaluatorWalls<evaluator>::param_type> >((EvaluatorWalls<evaluator>::getName()+"_params").c_str(), init<>())
		.def_readwrite("params", &EvaluatorWalls<evaluator>::param_type::params)
		.def_readwrite("ronsq", &EvaluatorWalls<evaluator>::param_type::ronsq)
		.def_readwrite("rcutsq", &EvaluatorWalls<evaluator>::param_type::rcutsq)
		;
	def(std::string("make_"+EvaluatorWalls<evaluator>::getName()+"_params").c_str(), &make_wall_params<evaluator>);
	}

template < class evaluator >
void export_PotentialExternalWall(const std::string& name)
	{
	export_PotentialExternal< PotentialExternal<EvaluatorWalls<evaluator> > >(name);
	export_wall_params_helpers<evaluator>();
	}

wall_type make_wall_field_params(boost::python::object walls)
	{
		wall_type w;
		w.numSpheres = boost::python::extract<unsigned int>(walls.attr("num_spheres"));
		w.numCylinders = boost::python::extract<unsigned int>(walls.attr("num_cylinders"));
		w.numPlanes = boost::python::extract<unsigned int>(walls.attr("num_planes"));
		for(unsigned int i = 0; i < w.numSpheres; i++)
			{
			Scalar 	r = boost::python::extract<Scalar>(walls.attr("spheres")[i].attr("r"));
			Scalar3 origin =boost::python::extract<Scalar3>(walls.attr("spheres")[i].attr("origin"));
			bool 	inside =boost::python::extract<bool>(walls.attr("spheres")[i].attr("inside"));
			w.Spheres[i] = SphereWall(r, origin, inside);
			}
		for(unsigned int i = 0; i < w.numCylinders; i++)
			{
			Scalar 	r = boost::python::extract<Scalar>(walls.attr("cylinders")[i].attr("r"));
			Scalar3 origin =boost::python::extract<Scalar3>(walls.attr("cylinders")[i].attr("origin"));
			Scalar3 axis =boost::python::extract<Scalar3>(walls.attr("cylinders")[i].attr("axis"));
			bool 	inside =boost::python::extract<bool>(walls.attr("cylinders")[i].attr("inside"));
			w.Cylinders[i] = CylinderWall(r, origin, axis, inside);
			}
		for(unsigned int i = 0; i < w.numPlanes; i++)
			{
			Scalar3 origin =boost::python::extract<Scalar3>(walls.attr("planes")[i].attr("origin"));
			Scalar3 normal =boost::python::extract<Scalar3>(walls.attr("planes")[i].attr("normal"));
			w.Planes[i] = PlaneWall(origin, normal);
			}
		return w;
	}

void export_wall_field_helpers()
	{
	class_< wall_type, boost::shared_ptr<wall_type> >( "wall_type", init<>());
	def("make_wall_field_params", &make_wall_field_params);
	}


#endif //__EVALUATOR__WALLS_H__
