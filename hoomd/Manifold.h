// Copyright (c) 2009-2020 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.


// Maintainer: pschoenhoefer

#include "SystemDefinition.h"
#include "ParticleGroup.h"
#include "Profiler.h"
#include "ManifoldSurfaces.h"

/*! \file Manifold.h
    \brief Declares a class that defines a differentiable manifold.
*/

#ifdef NVCC
#error This header cannot be compiled by nvcc
#endif

#include <pybind11/pybind11.h>

#ifndef __MANIFOLD_H__
#define __MANIFOLD_H__

//! Defines the geometry of a manifold.
class PYBIND11_EXPORT Manifold
    {
    public:
        //! Constructs the compute. Does nothing in base class.
        Manifold(std::shared_ptr<SystemDefinition> sysdef);
        virtual ~Manifold() {}

        //! Sets the profiler for the manifold to use
        void setProfiler(std::shared_ptr<Profiler> prof);

        //! Return the value of the implicit surface function describing the manifold F(x,y,z)=0.
        /*! \param point The location to evaluate the implicit surface function.
        */
        virtual Scalar implicit_function(Scalar3 point) {return 0;}

        //! Return the derivative of the implicit function/normal vector
        /*! \param point The position to evaluate the derivative.
        */
        virtual Scalar3 derivative(Scalar3 point) {return make_scalar3(0, 0, 0);}

	virtual Scalar3 returnL() {return make_scalar3(0, 0, 0);}

	virtual Scalar3 returnR() {return make_scalar3(0, 0, 0);}

	virtual manifold_enum::surf returnSurf() {return m_surf;}

#ifdef ENABLE_MPI
        //! Set the communicator to use
        /*! \param comm MPI communication class
         */
        void setCommunicator(std::shared_ptr<Communicator> comm)
            {
            assert(comm);
            m_comm = comm;
            }
#endif
    protected:
        const std::shared_ptr<SystemDefinition> m_sysdef; //!< The system definition this method is associated with
        const std::shared_ptr<ParticleData> m_pdata;      //!< The particle data this method is associated with
        std::shared_ptr<Profiler> m_prof;                 //!< The profiler this method is to use
        std::shared_ptr<const ExecutionConfiguration> m_exec_conf; //!< Stored shared ptr to the execution configuration

        manifold_enum::surf m_surf; //! determines the specific manifold

#ifdef ENABLE_MPI
        std::shared_ptr<Communicator> m_comm;             //!< The communicator to use for MPI
#endif
    };

//! Exports the Manifold class to python
void export_Manifold(pybind11::module& m);

#endif
