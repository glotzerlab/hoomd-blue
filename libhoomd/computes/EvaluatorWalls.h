#ifndef __EVALUATOR_WALLS_H__
#define __EVALUATOR_WALLS_H__

// need to declare these class methods with __device__ qualifiers when building in nvcc
// DEVICE is __host__ __device__ when included in nvcc and blank when included into the host compiler
#ifdef NVCC
#define DEVICE __device__
#else
#define DEVICE
#endif

#define MAX_NUM_WALLS 500

#include "HOOMDMath.h"
#include "VectorMath.h"
#include "WallData.h"


template<class evaluator>
class EvaluatorWalls : public ForceCompute
	{
	public:
        typedef struct {
        	Scalar rcutsq;
        	Scalar ronsq;
        	typename evaluator::param_type params;
        } param_type;
		typedef struct {
			SphereWall m_Spheres[20];
			CylinderWall m_Cylinders[20];
			PlaneWall m_Planes[MAX_NUM_WALLS - 40];
		} field_type;

		DEVICE EvaluatorWalls(Scalar3 m_pos, unsigned int idx, const BoxDim& m_box, const param_type& params, const field_type& field)
			{
			vec3<Scalar> dx;
			}

		DEVICE inline vec3<Scalar> wall_eval_dist(const SphereWall& wall, const vec3<Scalar>& position, const BoxDim& box)
		    {
		    // vec3<Scalar> t = position - box_origin;
		    // box.minImage(t);
		    // t-=wall.origin;
		    // vec3<Scalar> shifted_pos(t);
		    // Scalar rxyz_sq = shifted_pos.x*shifted_pos.x + shifted_pos.y*shifted_pos.y + shifted_pos.z*shifted_pos.z;
		    // Scalar r = wall.r - sqrt(rxyz_sq);
			vec3<Scalar> t = position;
			box.minImage(t);
			t-=wall.origin;
			vec3<Scalar> shifted_pos(t);
			Scalar rxyz = sqrt(dot(shifted_pos,shifted_pos));
			if (((rxyz < wall.r) && wall.inside) || (rxyz > wall.r) && !(wall.inside)) {
				t *= wall.r/rxyz;
				vec3<Scalar> dx = t - shifted_pos;
				return dx;
			}
			else{
				return vec3<Scalar>(0.0,0.0,0.0);
			}
		    };

		DEVICE inline vec3<Scalar> wall_eval_dist(const CylinderWall& wall, const vec3<Scalar>& position, const BoxDim& box)
		    {
		    // vec3<Scalar> t = position - box_origin;
		    // box.minImage(t);
		    // t-=wall.origin;
		    // vec3<Scalar> shifted_pos=rotate(wall.q_reorientation,t);
		    // Scalar rxy_sq= shifted_pos.x*shifted_pos.x + shifted_pos.y*shifted_pos.y;
		    // Scalar r = wall.r - sqrt(rxy_sq);
			vec3<Scalar> t = position;
	        box.minImage(t);
	        t-=wall.origin;
	        vec3<Scalar> shifted_pos = rotate(wall.q_reorientation,t);
			shifted_pos.z = 0;
	        Scalar rxy = sqrt(dot(shifted_pos,shifted_pos));
			if (((rxy < wall.r) && wall.inside) || (rxy > wall.r) && !(wall.inside)) {
		        t = (wall.r / rxy) * shifted_pos;
		        vec3<Scalar> dx = t - shifted_pos;
				dx = rotate(conj(wall.q_reorientation),dx);
				return dx;
			}
			else{
				return vec3<Scalar>(0.0,0.0,0.0);
			}
		    };

		DEVICE inline vec3<Scalar> wall_eval_dist(const PlaneWall& wall, const vec3<Scalar>& position, const BoxDim& box)
		    {
		    // vec3<Scalar> t = position - box_origin;
		    // box.minImage(t);
		    // Scalar r =dot(wall.normal,t)-dot(wall.normal,wall.origin);
			vec3<Scalar> t = position;
			box.minImage(t);
			Scalar wall_dist = dot(wall.normal,t) - dot(wall.normal,wall.origin);
			if (wall_dist > 0.0) {
				vec3<Scalar> dx = wall_dist * wall.normal;
				return dx;
			}
			else {
				return vec3<Scalar>(0.0,0.0,0.0);
			}
			};

		DEVICE void evalForceEnergyAndVirial(Scalar3& F, Scalar& energy, Scalar* virial)
			{

			ArrayHandle<Scalar> h_diameter(m_pdata->getDiameters(), access_location::host, access_mode::read);
			ArrayHandle<Scalar> h_charge(m_pdata->getCharges(), access_location::host, access_mode::read);

			// access diameter and charge (if needed)
			Scalar di = Scalar(0.0);
			Scalar qi = Scalar(0.0);
			if (evaluator::needsDiameter())
				di = h_diameter.data[idx];
			if (evaluator::needsCharge())
				qi = h_charge.data[idx];

			// convert type as little as possible
			vec3<Scalar> position = vec3<Scalar>(m_pos);
			vec3<Scalar> dxv;

			// initialize virial
			bool energy_shift = false;
			for (unsigned int k = 0; k < WallDataNew::getNumSphereWalls(); k++)
				{
				dxv = wall_eval_dist(field.m_Spheres[k], position, m_box);
				Scalar3 dx = vec_to_scalar3(dxv);

				// calculate r_ij squared (FLOPS: 5)
	            Scalar rsq = dot(dx, dx);
	            
	            if (rsq > params.ronsq)
		            {
		            // compute the force and potential energy
		            Scalar force_divr = Scalar(0.0);
		            Scalar pair_eng = Scalar(0.0);
		            evaluator eval(rsq, params.rcutsq, params.params);
		            if (evaluator::needsDiameter())
		                eval.setDiameter(di, 0.0);
		            if (evaluator::needsCharge())
		                eval.setCharge(qi, 0.0);

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
			for (unsigned int k = 0; k < WallDataNew::getNumCylinderWalls(); k++)
				{
				dxv = wall_eval_dist(field.m_Cylinders[k], position, m_box);
				Scalar3 dx = vec_to_scalar3(dxv);

				// calculate r_ij squared (FLOPS: 5)
	            Scalar rsq = dot(dx, dx);
	            
	            if (rsq > params.ronsq)
		            {
		            // compute the force and potential energy
		            Scalar force_divr = Scalar(0.0);
		            Scalar pair_eng = Scalar(0.0);
		            evaluator eval(rsq, params.rcutsq, params.params);
		            if (evaluator::needsDiameter())
		                eval.setDiameter(di, 0.0);
		            if (evaluator::needsCharge())
		                eval.setCharge(qi, 0.0);

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
			for (unsigned int k = 0; k < WallDataNew::getNumPlaneWalls(); k++)
				{
				dxv = wall_eval_dist(field.m_Planes[k], position, m_box);
				Scalar3 dx = vec_to_scalar3(dxv);

				// calculate r_ij squared (FLOPS: 5)
	            Scalar rsq = dot(dx, dx);
	            
	            if (rsq > params.ronsq)
		            {
		            // compute the force and potential energy
		            Scalar force_divr = Scalar(0.0);
		            Scalar pair_eng = Scalar(0.0);
		            evaluator eval(rsq, params.rcutsq, params.params);
		            if (evaluator::needsDiameter())
		                eval.setDiameter(di, 0.0);
		            if (evaluator::needsCharge())
		                eval.setCharge(qi, 0.0);

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
        Scalar3 m_pos;                //!< particle position
        BoxDim m_box;                 //!< box dimensions
        unsigned int idx;
        field_type field;
        param_type params;
        GPUArray<param_type> m_params;        //!< Array of per-type parameters

	};

#endif //__EVALUATOR__WALLS_H__
