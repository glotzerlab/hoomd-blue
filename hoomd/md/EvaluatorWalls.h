// Copyright (c) 2009-2021 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.

// Maintainer: jproc

/*! \file EvaluatorWalls.h
    \brief Executes an external field potential of several evaluator types for each wall in the
   system.
 */

#pragma once

#ifndef __HIPCC__
#include "hoomd/ArrayView.h"
#include <functional>
#include <memory>
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
#define PY
#else
#define DEVICE
#define PY PYBIND11_EXPORT
#endif

// sets the max numbers for each wall geometry type
const unsigned int MAX_N_SWALLS = 20;
const unsigned int MAX_N_CWALLS = 20;
const unsigned int MAX_N_PWALLS = 60;

struct PY wall_type
    {
    unsigned int numSpheres; // these data types come first, since the structs are aligned already
    unsigned int numCylinders;
    unsigned int numPlanes;
    SphereWall Spheres[MAX_N_SWALLS];
    CylinderWall Cylinders[MAX_N_CWALLS];
    PlaneWall Planes[MAX_N_PWALLS];

    wall_type() : numSpheres(0), numCylinders(0), numPlanes(0) { }

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

    unsigned int& getNumPlanes()
        {
        return numPlanes;
        }

    PlaneWall& getPlane(size_t index)
        {
        return Planes[index];
        }
    };

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

    DEVICE inline void callEvaluator(Scalar3& F, Scalar& energy, const vec3<Scalar> drv)
        {
        Scalar3 dr = -vec_to_scalar3(drv);
        Scalar rsq = dot(dr, dr);

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
            F += dr * force_divr;
            energy += pair_eng; // removing half since the other "particle" won't be represented *
                                // Scalar(0.5);
            }
        }

    DEVICE inline void extrapEvaluator(Scalar3& F,
                                       Scalar& energy,
                                       const vec3<Scalar> drv,
                                       const Scalar rextrapsq,
                                       const Scalar r)
        {
        Scalar3 dr = -vec_to_scalar3(drv);
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
            // add the force and potential energy to the particle i
            pair_eng
                = pair_eng
                  + force_divr * m_params.rextrap * r; // removing half since the other "particle"
                                                       // won't be represented * Scalar(0.5);
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
            // add the force and potential energy to the particle i
            F += dr * force_divr;
            energy += pair_eng; // removing half since the other "particle" won't be represented *
                                // Scalar(0.5);
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
        vec3<Scalar> drv;
        bool inside = false;        // keeps compiler from complaining
        if (m_params.rextrap > 0.0) // extrapolated mode
            {
            Scalar rextrapsq = m_params.rextrap * m_params.rextrap;
            Scalar rsq;
            for (unsigned int k = 0; k < m_field.numSpheres; k++)
                {
                drv = vecPtToWall(m_field.Spheres[k], position, inside);
                rsq = dot(drv, drv);
                if (inside && rsq >= rextrapsq)
                    {
                    callEvaluator(F, energy, drv);
                    }
                else
                    {
                    Scalar r = fast::sqrt(rsq);
                    if (rsq == 0.0)
                        {
                        inside = true; // just in case
                        drv = (position - m_field.Spheres[k].origin) / m_field.Spheres[k].r;
                        }
                    else
                        {
                        drv *= 1 / r;
                        }
                    r = (inside) ? m_params.rextrap - r : m_params.rextrap + r;
                    drv *= (inside) ? r : -r;
                    extrapEvaluator(F, energy, drv, rextrapsq, r);
                    }
                }
            for (unsigned int k = 0; k < m_field.numCylinders; k++)
                {
                drv = vecPtToWall(m_field.Cylinders[k], position, inside);
                rsq = dot(drv, drv);
                if (inside && rsq >= rextrapsq)
                    {
                    callEvaluator(F, energy, drv);
                    }
                else
                    {
                    Scalar r = fast::sqrt(rsq);
                    if (rsq == 0.0)
                        {
                        inside = true; // just in case
                        drv = rotate(m_field.Cylinders[k].quatAxisToZRot,
                                     position - m_field.Cylinders[k].origin);
                        drv.z = 0.0;
                        drv = rotate(conj(m_field.Cylinders[k].quatAxisToZRot), drv)
                              / m_field.Cylinders[k].r;
                        }
                    else
                        {
                        drv *= 1 / r;
                        }
                    r = (inside) ? m_params.rextrap - r : m_params.rextrap + r;
                    drv *= (inside) ? r : -r;
                    extrapEvaluator(F, energy, drv, rextrapsq, r);
                    }
                }
            for (unsigned int k = 0; k < m_field.numPlanes; k++)
                {
                drv = vecPtToWall(m_field.Planes[k], position, inside);
                rsq = dot(drv, drv);
                if (inside && rsq >= rextrapsq)
                    {
                    callEvaluator(F, energy, drv);
                    }
                else
                    {
                    Scalar r = fast::sqrt(rsq);
                    if (rsq == 0.0)
                        {
                        inside = true; // just in case
                        drv = m_field.Planes[k].normal;
                        }
                    else
                        {
                        drv *= 1 / r;
                        }
                    r = (inside) ? m_params.rextrap - r : m_params.rextrap + r;
                    drv *= (inside) ? r : -r;
                    extrapEvaluator(F, energy, drv, rextrapsq, r);
                    }
                }
            }
        else // normal mode
            {
            for (unsigned int k = 0; k < m_field.numSpheres; k++)
                {
                drv = vecPtToWall(m_field.Spheres[k], position, inside);
                if (inside)
                    {
                    callEvaluator(F, energy, drv);
                    }
                }
            for (unsigned int k = 0; k < m_field.numCylinders; k++)
                {
                drv = vecPtToWall(m_field.Cylinders[k], position, inside);
                if (inside)
                    {
                    callEvaluator(F, energy, drv);
                    }
                }
            for (unsigned int k = 0; k < m_field.numPlanes; k++)
                {
                drv = vecPtToWall(m_field.Planes[k], position, inside);
                if (inside)
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

#ifndef __HIPCC__
void export_wall_field(pybind11::module& m)
    {
    // Export the necessary array_view types to enable access in Python
    export_array_view<SphereWall>(m, "SphereArray");
    export_array_view<CylinderWall>(m, "CylinderArray");
    export_array_view<PlaneWall>(m, "PlaneArray");

    pybind11::class_<wall_type, std::shared_ptr<wall_type>>(m, "WallCollection")
        .def("_unsafe_create",
             []() -> std::shared_ptr<wall_type> { return std::make_shared<wall_type>(); })
        // The different get_*_list methods use array_view's (see hoomd/ArrayView.h for more info)
        // callback to ensure that the way_type object's sizes remain correct even during
        // modification.
        .def("get_sphere_list",
             [](wall_type& wall_list)
             {
                 return make_array_view(
                     &wall_list.Spheres[0],
                     MAX_N_SWALLS,
                     wall_list.numSpheres,
                     std::function<void(const array_view<SphereWall>*)>(
                         [&wall_list](const array_view<SphereWall>* view) -> void
                         { wall_list.numSpheres = static_cast<unsigned int>(view->size); }));
             })
        .def("get_cylinder_list",
             [](wall_type& wall_list)
             {
                 return make_array_view(
                     &wall_list.Cylinders[0],
                     MAX_N_CWALLS,
                     wall_list.numCylinders,
                     std::function<void(const array_view<CylinderWall>*)>(
                         [&wall_list](const array_view<CylinderWall>* view) -> void
                         { wall_list.numCylinders = static_cast<unsigned int>(view->size); }));
             })
        .def("get_plane_list",
             [](wall_type& wall_list)
             {
                 return make_array_view(
                     &wall_list.Planes[0],
                     MAX_N_PWALLS,
                     wall_list.numPlanes,
                     std::function<void(const array_view<PlaneWall>*)>(
                         [&wall_list](const array_view<PlaneWall>* view) -> void
                         { wall_list.numPlanes = static_cast<unsigned int>(view->size); }));
             })
        // These functions are not necessary for the Python interface but allow for more ready
        // testing of the array_view class and this exporting.
        .def("get_sphere", &wall_type::getSphere)
        .def("get_cylinder", &wall_type::getCylinder)
        .def("get_plane", &wall_type::getPlane)
        .def_property_readonly("num_spheres", &wall_type::getNumSpheres)
        .def_property_readonly("num_cylinders", &wall_type::getNumCylinders)
        .def_property_readonly("num_planes", &wall_type::getNumPlanes);
    }
#endif
