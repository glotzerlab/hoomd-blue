// Copyright (c) 2009-2020 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.


// Maintainer: pschoenhoefer

#include "EllipsoidManifold.h"

namespace py = pybind11;

using namespace std;

/*! \file EllipsoidManifold.cc
    \brief Contains code for the EllipsoidManifold class
*/

/*!
    \param r radius of the sphere
    \param P position of the sphere
*/
EllipsoidManifold::EllipsoidManifold(std::shared_ptr<SystemDefinition> sysdef,
                               Scalar a,
                               Scalar b,
                               Scalar c,
                               Scalar3 P)
  : Manifold(sysdef), m_inva2(Scalar(1.0)/(a*a)), m_invb2(Scalar(1.0)/(b*b)), m_invc2(Scalar(1.0)/(c*c)), m_P(P) 
       {
    m_exec_conf->msg->notice(5) << "Constructing EllipsoidManifold" << endl;
    m_surf = manifold_enum::sphere;
    validate();
       }

EllipsoidManifold::~EllipsoidManifold() 
       {
    m_exec_conf->msg->notice(5) << "Destroying EllipsoidManifold" << endl;
       }

        //! Return the value of the implicit surface function of the sphere.
        /*! \param point The position to evaluate the function.
        */
Scalar EllipsoidManifold::implicit_function(Scalar3 point)
       {
       Scalar3 delta = point - m_P;
       return delta.x*delta.x*m_inva2 + delta.y*delta.y*m_invb2 + delta.z*delta.z*m_invc2 - 1;
       }

       //! Return the gradient of the constraint.
       /*! \param point The location to evaluate the gradient.
       */
Scalar3 EllipsoidManifold::derivative(Scalar3 point)
       {
       Scalar3 delta = point - m_P;
       delta.x *= 2*m_inva2;
       delta.y *= 2*m_invb2;
       delta.z *= 2*m_invc2;
       return delta;
       }

void EllipsoidManifold::validate()
    {
    BoxDim box = m_pdata->getGlobalBox();
    Scalar3 lo = box.getLo();
    Scalar3 hi = box.getHi();
    Scalar a = Scalar(1.0)/fast::sqrt(m_inva2);
    Scalar b = Scalar(1.0)/fast::sqrt(m_invb2);
    Scalar c = Scalar(1.0)/fast::sqrt(m_invc2);

    if (m_P.x + a > hi.x || m_P.x - a < lo.x ||
        m_P.y + b > hi.y || m_P.y - b < lo.y ||
        m_P.z + c > hi.z || m_P.z - c < lo.z)
        {
        m_exec_conf->msg->error() << "manifold.Ellipsoid: Ellipsoid manifold is outside of the box. Constrained particle positions may be incorrect"
             << endl;
        throw std::runtime_error("Error during Ellipsoid manifold.");
        }
    }

//! Exports the EllipsoidManifold class to python
void export_EllipsoidManifold(pybind11::module& m)
    {
    py::class_< EllipsoidManifold, Manifold, std::shared_ptr<EllipsoidManifold> >(m, "EllipsoidManifold")
    .def(py::init< std::shared_ptr<SystemDefinition>,Scalar, Scalar, Scalar, Scalar3 >())
    .def("implicit_function", &EllipsoidManifold::implicit_function)
    .def("derivative", &EllipsoidManifold::derivative)
    .def("returnL", &EllipsoidManifold::returnL)
    ;
    }
