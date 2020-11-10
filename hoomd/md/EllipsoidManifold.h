// Copyright (c) 2009-2020 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.


// Maintainer: pschoenhoefer

#include "hoomd/Manifold.h"

/*! \file EllipsoidManifold.h
    \brief Declares the implicit function of a sphere.
*/

#ifdef NVCC
#error This header cannot be compiled by nvcc
#endif

#include <pybind11/pybind11.h>

#ifndef __ELLIPSOID_MANIFOLD_H__
#define __ELLIPSOID_MANIFOLD_H__

//! Defines the geometry of a manifold.
class PYBIND11_EXPORT EllipsoidManifold : public Manifold
    {
    public:
        //! Constructs the compute
        /*! \param a The length of axis in x direction.
            \param b The length of axis in y direction.
            \param c The length of axis in z direction.
            \param P The location of the sphere.
        */
        EllipsoidManifold(std::shared_ptr<SystemDefinition> sysdef,
                  Scalar a, 
                  Scalar b, 
                  Scalar c, 
                  Scalar3 P=make_scalar3(0,0,0));

        //! Destructor
        virtual ~EllipsoidManifold();

        //! Return the value of the implicit surface function of the sphere.
        /*! \param point The position to evaluate the function.
        */
        Scalar implicit_function(Scalar3 point);

        //! Return the gradient of the implicit function/normal vector.
        /*! \param point The location to evaluate the gradient.
        */
        Scalar3 derivative(Scalar3 point);

	Scalar3 returnL(){return m_P;};

	Scalar3 returnR(){return make_scalar3(m_inva2, m_invb2, m_invc2);};

    protected:
        Scalar m_inva2; //! The inverse of x-axis length of the ellipsoid.
        Scalar m_invb2; //! The inverse of x-axis length of the ellipsoid.
        Scalar m_invc2; //! The inverse of x-axis length of the ellipsoid.
        Scalar3 m_P; //! The center of the sphere.

    private:
        //! Validate that the sphere is in the box and all particles are very near the constraint
        void validate();
    };

//! Exports the EllipsoidManifold class to python
void export_EllipsoidManifold(pybind11::module& m);

#endif
