// Copyright (c) 2009-2020 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.


// Maintainer: pschoenhoefer

#include "SphereManifold.h"

namespace py = pybind11;

using namespace std;

/*! \file SphereManifold.cc
    \brief Contains code for the SphereManifold class
*/

/*!
    \param r radius of the sphere
    \param P position of the sphere
*/
SphereManifold::SphereManifold(std::shared_ptr<SystemDefinition> sysdef,
                               Scalar r,
                               Scalar3 P)
  : Manifold(sysdef), m_r(r), m_P(P) 
       {
    m_exec_conf->msg->notice(5) << "Constructing SphereManifold" << endl;
    m_surf = manifold_enum::sphere;
    validate();
       }

SphereManifold::~SphereManifold() 
       {
    m_exec_conf->msg->notice(5) << "Destroying SphereManifold" << endl;
       }

        //! Return the value of the implicit surface function of the sphere.
        /*! \param point The position to evaluate the function.
        */
Scalar SphereManifold::implicit_function(Scalar3 point)
       {
       Scalar3 delta = point - m_P;
       return dot(delta, delta) - m_r*m_r;
       }

       //! Return the gradient of the constraint.
       /*! \param point The location to evaluate the gradient.
       */
Scalar3 SphereManifold::derivative(Scalar3 point)
       {
       Scalar3 delta = point - m_P;
       return 2*delta;
       }

void SphereManifold::validate()
    {
    BoxDim box = m_pdata->getGlobalBox();
    Scalar3 lo = box.getLo();
    Scalar3 hi = box.getHi();

    if (m_P.x + m_r > hi.x || m_P.x - m_r < lo.x ||
        m_P.y + m_r > hi.y || m_P.y - m_r < lo.y ||
        m_P.z + m_r > hi.z || m_P.z - m_r < lo.z)
        {
        m_exec_conf->msg->error() << "manifold.Sphere: Sphere manifold is outside of the box. Constrained particle positions may be incorrect "
             << endl;
        throw std::runtime_error("Error during Sphere manifold.");
        }
    }

//! Exports the SphereManifold class to python
void export_SphereManifold(pybind11::module& m)
    {
    py::class_< SphereManifold, Manifold, std::shared_ptr<SphereManifold> >(m, "SphereManifold")
    .def(py::init< std::shared_ptr<SystemDefinition>,Scalar, Scalar3 >())
    .def("implicit_function", &SphereManifold::implicit_function)
    .def("derivative", &SphereManifold::derivative)
    .def("returnL", &SphereManifold::returnL)
    .def("returnR", &SphereManifold::returnL)
    ;
    }
