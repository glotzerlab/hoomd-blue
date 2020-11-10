// Copyright (c) 2009-2020 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.


// Maintainer: pschoenhoefer

#include "CylinderManifold.h"

namespace py = pybind11;

using namespace std;

/*! \file CylinderManifold.cc
    \brief Contains code for the CylinderManifold class
*/

/*!
    \param r radius of the sphere
    \param P position of the sphere
*/
CylinderManifold::CylinderManifold(std::shared_ptr<SystemDefinition> sysdef,
                               Scalar r,
                               Scalar3 P)
  : Manifold(sysdef), m_r(r) 
       {
    m_exec_conf->msg->notice(5) << "Constructing CylinderManifold" << endl;
    m_surf = manifold_enum::cylinder;
    validate();
       }

CylinderManifold::~CylinderManifold() 
       {
    m_exec_conf->msg->notice(5) << "Destroying CylinderManifold" << endl;
       }

        //! Return the value of the implicit surface function of the sphere.
        /*! \param point The position to evaluate the function.
        */
Scalar CylinderManifold::implicit_function(Scalar3 point)
       {
       Scalar3 delta = point - m_P;
       return delta.x*delta.x + delta.y*delta.y - m_r*m_r;
       }

       //! Return the gradient of the constraint.
       /*! \param point The location to evaluate the gradient.
       */
Scalar3 CylinderManifold::derivative(Scalar3 point)
       {
       Scalar3 delta = point - m_P;
       delta.z = 0;
       return 2*delta;
       }

void CylinderManifold::validate()
    {
    BoxDim box = m_pdata->getGlobalBox();
    Scalar3 lo = box.getLo();
    Scalar3 hi = box.getHi();

    if (m_P.x + m_r > hi.x || m_P.x - m_r < lo.x ||
        m_P.y + m_r > hi.y || m_P.y - m_r < lo.y)
        {
        m_exec_conf->msg->error() << "manifold.Cylinder: Cylinder manifold is outside of the box. Constrained particle positions may be incorrect"
             << endl;
        throw std::runtime_error("Error during Cylinder manifold.");
        }
    }

//! Exports the CylinderManifold class to python
void export_CylinderManifold(pybind11::module& m)
    {
    py::class_< CylinderManifold, Manifold, std::shared_ptr<CylinderManifold> >(m, "CylinderManifold")
    .def(py::init< std::shared_ptr<SystemDefinition>,Scalar, Scalar3 >())
    .def("implicit_function", &CylinderManifold::implicit_function)
    .def("derivative", &CylinderManifold::derivative)
    .def("returnL", &CylinderManifold::returnL)
    .def("returnR", &CylinderManifold::returnR)
    ;
    }
