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

template<class evaluator>
class EvaluatorWalls : public ForceCompute
	{
	public:
        typedef typename evaluator::param_type param_type;
		typedef struct {
			SphereWall[20];
			CylinderWall[20];
			PlaneWall[MAX_NUM_WALLS - 40];
		} field_type;
        
        //! Set the pair parameters for a single type pair
        virtual void setParams(unsigned int typ1, unsigned int typ2, const param_type& param);
        //! Set the rcut for a single type pair
        virtual void setRcut(unsigned int typ1, unsigned int typ2, Scalar rcut);
        //! Set ron for a single type pair
        virtual void setRon(unsigned int typ1, unsigned int typ2, Scalar ron);
        
	DEVICE inline Scalar wall_eval_rsq<SphereWall>(const SphereWall& wall, const vec3<Scalar>& position, const vec3<Scalar>& box_origin, const BoxDim& box)
	    {
	    vec3<Scalar> t = position - box_origin;
	    box.minImage(t);
	    t-=wall.origin;
	    vec3<Scalar> shifted_pos(t);
	    Scalar rxyz_sq = shifted_pos.x*shifted_pos.x + shifted_pos.y*shifted_pos.y + shifted_pos.z*shifted_pos.z;
	    Scalar r = wall.r - sqrt(rxyz_sq);
	    Scalar r_sq = r * r;
	    return r_sq;
	    };

	DEVICE inline Scalar wall_dist_eval<CylinderWall>(const CylinderWall& wall, const vec3<Scalar>& position, const vec3<Scalar>& box_origin, const BoxDim& box)
	    {
	    vec3<Scalar> t = position - box_origin;
	    box.minImage(t);
	    t-=wall.origin;
	    vec3<Scalar> shifted_pos=rotate(wall.q_reorientation,t);
	    Scalar rxy_sq= shifted_pos.x*shifted_pos.x + shifted_pos.y*shifted_pos.y;
	    Scalar r = wall.r - sqrt(rxy_sq);
	    Scalar r_sq = r * r;
	    return r_sq;
	    };

	DEVICE inline Scalar wall_dist_eval<PlaneWall>(const PlaneWall& wall, const vec3<Scalar>& position, const vec3<Scalar>& box_origin, const BoxDim& box)
	    {
	    vec3<Scalar> t = position - box_origin;
	    box.minImage(t);
	    Scalar r =dot(wall.normal,t)-dot(wall.normal,wall.origin);
	    Scalar r_sq = r * r;
	    return r_sq;
		};

    protected:
        Scalar3 m_pos;                //!< particle position
        BoxDim m_box;                 //!< box dimensions
	};

#endif __EVALUATOR__WALLS_H__
