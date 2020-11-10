// Copyright (c) 2009-2020 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.


// Maintainer: pschoenhoefer

#include "hoomd/Manifold.h"

/*! \file TPMSManifold.h
    \brief Declares the implicit function of a sphere.
*/

#ifdef NVCC
#error This header cannot be compiled by nvcc
#endif

#include <pybind11/pybind11.h>

#ifndef __TPMS_MANIFOLD_H__
#define __TPMS_MANIFOLD_H__

//! Defines the geometry of triply periodic minimal surface.
class PYBIND11_EXPORT TPMSManifold : public Manifold
    {
    public:
        //! Constructs the compute
        /*! \param surf Defines the specific triply periodic minimal surface
            \param Nx The number of unitcells in x-direction
            \param Ny The number of unitcells in y-direction
            \param Nz The number of unitcells in z-direction
        */
        TPMSManifold(std::shared_ptr<SystemDefinition> sysdef,
                  std::string surf, 
                 unsigned int Nx,
                  unsigned int Ny,
                  unsigned int Nz);

        //! Destructor
        virtual ~TPMSManifold();

        //! Return the value of the implicit surface function of the sphere.
        /*! \param point The position to evaluate the function.
        */
        Scalar implicit_function(Scalar3 point);

        //! Return the gradient of the implicit function/normal vector.
        /*! \param point The location to evaluate the gradient.
        */
        Scalar3 derivative(Scalar3 point);

        Scalar3 returnL();

    protected:
        Scalar m_Nx; //! number of unit cells in x direction
        Scalar m_Ny; //! number of unit cells in x direction
        Scalar m_Nz; //! number of unit cells in x direction

    private:
        //! setting up the TPMS properly
        void setup();
	Scalar Lx;
	Scalar Ly;
	Scalar Lz;
	
    };

//! Exports the TPMSManifold class to python
void export_TPMSManifold(pybind11::module& m);

#endif
