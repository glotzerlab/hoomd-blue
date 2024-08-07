// Copyright (c) 2009-2022 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

/*! \file EvaluatorWalls.h
    \brief Executes an external field potential of several evaluator types for each wall in the
   system.
 */

#pragma once

#ifndef __HIPCC__
#include <pybind11/pybind11.h>
#include <string>
#endif

#include "WallData.h"
#include "hoomd/BoxDim.h"
#include "hoomd/HOOMDMath.h"
#include "hoomd/VectorMath.h"

#undef DEVICE
#ifdef __HIPCC__
#define DEVICE __device__
#define HOOMD_PYBIND11_EXPORT
#else
#define DEVICE
#define HOOMD_PYBIND11_EXPORT PYBIND11_EXPORT
#endif

// sets the max numbers for each wall geometry type
const unsigned int MAX_N_SWALLS = 20;
const unsigned int MAX_N_CWALLS = 20;
const unsigned int MAX_N_COWALLS = 20;
const unsigned int MAX_N_PWALLS = 60;

namespace hoomd
    {
namespace md
    {
struct HOOMD_PYBIND11_EXPORT wall_type
    {
    unsigned int numSpheres; // these data types come first, since the structs are aligned already
    unsigned int numCylinders;
    unsigned int numCones;
    unsigned int numPlanes;
    SphereWall Spheres[MAX_N_SWALLS];
    CylinderWall Cylinders[MAX_N_CWALLS];
    ConeWall Cones[MAX_N_COWALLS];
    PlaneWall Planes[MAX_N_PWALLS];

    wall_type() : numSpheres(0), numCylinders(0), numCones(0), numPlanes(0) { }

    // The following methods are to test the ArrayView<> templated class.

    unsigned int getNumSpheres()
        {
        return numSpheres;
        }

    SphereWall& getSphere(size_t index)
        {
        return Spheres[index];
        }

    unsigned int getNumCylinders()
        {
        return numCylinders;
        }

    CylinderWall& getCylinder(size_t index)
        {
        return Cylinders[index];
        }

    unsigned int getNumCones()
        {
        return numCones;
        }

    CylinderWall& getCone(size_t index)
        {
        return Cones[index];
        }

    unsigned int& getNumPlanes()
        {
        return numPlanes;
        }

    PlaneWall& getPlane(size_t index)
        {
        return Planes[index];
        }
    };

/// Function for getting the force direction for particle on the wall with r_extrap
DEVICE inline Scalar3 onWallForceDirection(const vec3<Scalar>& position, const SphereWall& wall)
    {
    // We need a direction for the force and since r == 0 then the distance from
    // the origin is the Sphere's radius.
    return vec_to_scalar3(wall.origin - position) / wall.r;
    }

/// Function for getting the force direction for particle on the wall with r_extrap
DEVICE inline Scalar3 onWallForceDirection(const vec3<Scalar>& position, const CylinderWall& wall)
    {
    auto s = rotate(wall.quatAxisToZRot, wall.origin - position);
    s.z = 0.0;
    return vec_to_scalar3(rotate(conj(wall.quatAxisToZRot), s) / wall.r);
    }

/// Function for getting the force direction for particle on the wall with r_extrap
DEVICE inline Scalar3 onWallForceDirection(const vec3<Scalar>& position, const ConeWall& wall)
    {
    // auto s = rotate(wall.quatAxisToZRot, wall.origin - position);
    // s.z = 0.0;
    vec3<Scalar> rotatingaxis(-s.y,s.x,0);
    rotatingaxis = rotatingaxis * fast::rsqrt(s.y * s.y + s.x * s.x);
    quat<Scalar> quatSideToZRot(fast::cos(wall.angle/2), - fast::sin(wall.angle/2) * rotatingaxis.x, fast::sin(wall.angle/2) * rotatingaxis.y, 0);
    // quat<Scalar> quatSideToZRot = quat<Scalar>::fromAxisAngle(rotatingaxis, wall.angle);
    quat<Scalar> quatCombined = wall.quatAxisToZRot * quatAxisToZRot;
    auto s = rotate(quatCombined, position);

    return vec_to_scalar3(rotate(conj(wall.quatAxisToZRot), s) / wall.r);
    return vec_to_scalar3(rotate(conj(quatCombined), s) / wall.r);
    }

/// Function for getting the force direction for particle on the wall with r_extrap
DEVICE inline Scalar3 onWallForceDirection(const PlaneWall& wall)
    {
    return vec_to_scalar3(-wall.normal);
    }

//! Applys a wall force from all walls in the field parameter
/*! \ingroup computes
 */
template<class evaluator> class EvaluatorWalls
    {
    public:
    struct param_type
        {
        typename evaluator::param_type params;
        Scalar rcutsq;
        Scalar rextrap;

#ifndef __HIPCC__
        param_type(pybind11::object param_dict)
            : params(param_dict), rcutsq(pow(param_dict["r_cut"].cast<Scalar>(), 2)),
              rextrap(param_dict["r_extrap"].cast<Scalar>())
            {
            }

        pybind11::object toPython()
            {
            auto py_params = params.asDict();
            py_params["r_cut"] = sqrt(rcutsq);
            py_params["r_extrap"] = rextrap;
            return std::move(py_params);
            }
#endif
        };

    typedef wall_type field_type;

    //! Constructs the external wall potential evaluator
    DEVICE EvaluatorWalls(Scalar3 pos, const BoxDim& box, const param_type& p, const field_type& f)
        : m_pos(pos), m_field(f), m_params(p)
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

    //! Charges not supported by walls evals
    DEVICE static bool needsCharge()
        {
        return evaluator::needsCharge();
        }

    //! Declares additional virial contributions are needed for the external field
    DEVICE static bool requestFieldVirialTerm()
        {
        return false; // volume change dependence is not currently defined
        }

    //! Accept the optional charge value
    /*! \param qi Charge of particle i
    Walls charge currently assigns a charge of 0 to the walls. It is however unused by implemented
    potentials.
    */
    DEVICE void setCharge(Scalar charge)
        {
        qi = charge;
        }

    DEVICE inline void callEvaluator(Scalar3& F, Scalar& energy, const Scalar3 drv)
        {
        Scalar rsq = dot(drv, drv);

        // compute the force and potential energy
        Scalar force_divr = Scalar(0.0);
        Scalar pair_eng = Scalar(0.0);
        evaluator eval(rsq, m_params.rcutsq, m_params.params);
        if (evaluator::needsDiameter())
            eval.setDiameter(di, Scalar(0.0));
        if (evaluator::needsCharge())
            eval.setCharge(qi, Scalar(0.0));

        bool evaluated = eval.evalForceAndEnergy(force_divr, pair_eng, true);

        if (evaluated)
            {
// correctly result in a 0 force in this case
#ifdef __HIPCC__
            if (!isfinite(force_divr))
#else
            if (!std::isfinite(force_divr))
#endif
                {
                force_divr = Scalar(0.0);
                pair_eng = Scalar(0.0);
                }
            // add the force and potential energy to the particle i
            F += drv * force_divr;
            energy += pair_eng; // removing half since the other "particle" won't be represented *
                                // Scalar(0.5);
            }
        }

    DEVICE inline void extrapEvaluator(Scalar3& F,
                                       Scalar& energy,
                                       const Scalar3 drv,
                                       const Scalar rextrapsq,
                                       const Scalar r)
        {
        // compute the force and potential energy
        Scalar force_divr = Scalar(0.0);
        Scalar pair_eng = Scalar(0.0);

        evaluator eval(rextrapsq, m_params.rcutsq, m_params.params);
        if (evaluator::needsDiameter())
            eval.setDiameter(di, Scalar(0.0));
        if (evaluator::needsCharge())
            eval.setCharge(qi, Scalar(0.0));

        bool evaluated = eval.evalForceAndEnergy(force_divr, pair_eng, true);

        if (evaluated)
            {
            pair_eng = pair_eng + force_divr * m_params.rextrap * r;
            force_divr *= m_params.rextrap / r;
// correctly result in a 0 force in this case
#ifdef __HIPCC__
            if (!isfinite(force_divr))
#else
            if (!std::isfinite(force_divr))
#endif
                {
                force_divr = Scalar(0.0);
                pair_eng = Scalar(0.0);
                }
            F += drv * force_divr;
            energy += pair_eng;
            }
        }

    //! Generates force and energy from standard evaluators using wall geometry functions
    DEVICE void evalForceEnergyAndVirial(Scalar3& F, Scalar& energy, Scalar* virial)
        {
        F.x = Scalar(0.0);
        F.y = Scalar(0.0);
        F.z = Scalar(0.0);
        energy = Scalar(0.0);
        // initialize virial
        for (unsigned int i = 0; i < 6; i++)
            virial[i] = Scalar(0.0);

        // convert type as little as possible
        vec3<Scalar> position = vec3<Scalar>(m_pos);
        Scalar3 drv;
        bool in_active_space = false;
        if (m_params.rextrap > 0.0) // extrapolated mode
            {
            Scalar rextrapsq = m_params.rextrap * m_params.rextrap;
            Scalar rsq;
            for (unsigned int k = 0; k < m_field.numSpheres; k++)
                {
                drv = distVectorWallToPoint(m_field.Spheres[k], position, in_active_space);
                rsq = dot(drv, drv);
                if (in_active_space && rsq >= rextrapsq)
                    {
                    callEvaluator(F, energy, drv);
                    }
                // Need to use extrapolated potential
                else
                    {
                    Scalar r = fast::sqrt(rsq);
                    // Normalize distance vectors
                    if (rsq == 0.0)
                        {
                        in_active_space = true; // just in case
                        drv = onWallForceDirection(position, m_field.Spheres[k]);
                        }
                    else
                        {
                        drv *= 1 / r;
                        }
                    // Recompute r and distance vector in terms of r_extrap
                    r = in_active_space ? m_params.rextrap - r : m_params.rextrap + r;
                    drv *= in_active_space ? r : -r;
                    extrapEvaluator(F, energy, drv, rextrapsq, r);
                    }
                }
            vec3<Scalar> intermediate_distance_vector;
            for (unsigned int k = 0; k < m_field.numCylinders; k++)
                {
                drv = distVectorWallToPoint(m_field.Cylinders[k], position, in_active_space);
                rsq = dot(drv, drv);
                if (in_active_space && rsq >= rextrapsq)
                    {
                    callEvaluator(F, energy, drv);
                    }
                else
                    {
                    Scalar r = fast::sqrt(rsq);
                    if (rsq == 0.0)
                        {
                        in_active_space = true; // just in case
                        drv = onWallForceDirection(position, m_field.Cylinders[k]);
                        }
                    else
                        {
                        drv *= 1 / r;
                        }
                    r = (in_active_space) ? m_params.rextrap - r : m_params.rextrap + r;
                    drv *= (in_active_space) ? r : -r;
                    extrapEvaluator(F, energy, drv, rextrapsq, r);
                    }
                }
            // vec3<Scalar> intermediate_distance_vector;
            for (unsigned int k = 0; k < m_field.numCones; k++)
                {
                drv = distVectorWallToPoint(m_field.Cones[k], position, in_active_space);
                rsq = dot(drv, drv);
                if (in_active_space && rsq >= rextrapsq)
                    {
                    callEvaluator(F, energy, drv);
                    }
                else
                    {
                    Scalar r = fast::sqrt(rsq);
                    if (rsq == 0.0)
                        {
                        in_active_space = true; // just in case
                        drv = onWallForceDirection(position, m_field.Cones[k]);
                        }
                    else
                        {
                        drv *= 1 / r;
                        }
                    r = (in_active_space) ? m_params.rextrap - r : m_params.rextrap + r;
                    drv *= (in_active_space) ? r : -r;
                    extrapEvaluator(F, energy, drv, rextrapsq, r);
                    }
                }
            for (unsigned int k = 0; k < m_field.numPlanes; k++)
                {
                drv = distVectorWallToPoint(m_field.Planes[k], position, in_active_space);
                rsq = dot(drv, drv);
                if (in_active_space && rsq >= rextrapsq)
                    {
                    callEvaluator(F, energy, drv);
                    }
                else
                    {
                    Scalar r = fast::sqrt(rsq);
                    if (rsq == 0.0)
                        {
                        in_active_space = true; // just in case
                        drv = onWallForceDirection(m_field.Planes[k]);
                        }
                    else
                        {
                        drv *= 1 / r;
                        }
                    r = (in_active_space) ? m_params.rextrap - r : m_params.rextrap + r;
                    drv *= (in_active_space) ? r : -r;
                    extrapEvaluator(F, energy, drv, rextrapsq, r);
                    }
                }
            }
        else // normal mode
            {
            for (unsigned int k = 0; k < m_field.numSpheres; k++)
                {
                drv = distVectorWallToPoint(m_field.Spheres[k], position, in_active_space);
                if (in_active_space)
                    {
                    callEvaluator(F, energy, drv);
                    }
                }
            for (unsigned int k = 0; k < m_field.numCylinders; k++)
                {
                drv = distVectorWallToPoint(m_field.Cylinders[k], position, in_active_space);
                if (in_active_space)
                    {
                    callEvaluator(F, energy, drv);
                    }
                }
            for (unsigned int k = 0; k < m_field.numCones; k++)
                {
                drv = distVectorWallToPoint(m_field.Cones[k], position, in_active_space);
                if (in_active_space)
                    {
                    callEvaluator(F, energy, drv);
                    }
                }
            for (unsigned int k = 0; k < m_field.numPlanes; k++)
                {
                drv = distVectorWallToPoint(m_field.Planes[k], position, in_active_space);
                if (in_active_space)
                    {
                    callEvaluator(F, energy, drv);
                    }
                }
            }

        // evaluate virial
        virial[0] = F.x * m_pos.x;
        virial[1] = F.x * m_pos.y;
        virial[2] = F.x * m_pos.z;
        virial[3] = F.y * m_pos.y;
        virial[4] = F.y * m_pos.z;
        virial[5] = F.z * m_pos.z;
        }

#ifndef __HIPCC__
    //! Get the name of this potential
    /*! \returns The potential name.
     */
    static std::string getName()
        {
        return std::string("wall_") + evaluator::getName();
        }
#endif

    protected:
    Scalar3 m_pos;             //!< particle position
    const field_type& m_field; //!< contains all information about the walls.
    param_type m_params;
    Scalar di;
    Scalar qi;
    };

    } // end namespace md
    } // end namespace hoomd

#undef HOOMD_PYBIND11_EXPORT
