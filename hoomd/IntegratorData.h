// Copyright (c) 2009-2019 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.


// Maintainer: joaander

/*! \file IntegratorData.h
    \brief Contains declarations for IntegratorData.
 */

#ifdef NVCC
#error This header cannot be compiled by nvcc
#endif

#ifndef __INTEGRATORDATA_H__
#define __INTEGRATORDATA_H__

#include "ParticleData.h"
#include <string>

#include "HOOMDMPI.h"

//! Stores integrator variables
/*! The integration state is necessary for exact restarts.  Extended systems
    integrators in the spirit of Nose-Hoover store the positions, velocities,
    etc. of the fictitious variables.  Other integrators store a random number
    seed.
    \ingroup data_structs
*/
struct PYBIND11_EXPORT IntegratorVariables
    {
    std::string type;                   //!<The type of integrator (NVT, NPT, etc.)
    std::vector<Scalar> variable;       //!<Variables that define the integration state
    };

#ifdef ENABLE_MPI
namespace cereal
    {
    //! Serialization of IntegratorVariables
    template<class Archive>
    void serialize(Archive & ar, IntegratorVariables & iv, const unsigned int version)
        {
        // serialize both members
        ar & iv.type;
        ar & iv.variable;
        }
    }
#endif

//! Stores all integrator variables in the simulation
/*! IntegratorData keeps track of the parameters for all of the integrators
    defined in the simulation, so that they can be saved and reloaded from data files.

    Each integrator must register with IntegratorData by calling registerIntegrator(), which returns an unsigned int
    to be used to access the variables for that integrator. The same sequence of registerIntegrator() calls will produce
    the same set of handles, so they can be used to read existing state values after loading data from a file.

    The state of current registered integrators is reset when a new IntegratorData is constructed. This is consistent
    with the above use-case, as the construction of a new IntegratorData means the construction of a new SystemData, and
    hence a new series of constructed Integrators, which will then re-register.

    \ingroup data_structs
*/
class PYBIND11_EXPORT IntegratorData
    {
    public:
        //! Constructs an empty list with no integrator variables
        IntegratorData() : m_num_registered(0) {}

        //! Constructs an IntegratorData from a given set of IntegratorVariables
        IntegratorData(const std::vector<IntegratorVariables>& variables)
            : m_num_registered(0)
            {
            m_integrator_variables = variables;
            }

        //! Destructor
        ~IntegratorData() {}

        //! Register an integrator (should occur during integrator construction)
        unsigned int registerIntegrator();

        //! Get the number of integrator variables
        /*! \return Number of integrator variables present
        */
        unsigned int getNumIntegrators() const
            {
            return (unsigned int)m_integrator_variables.size();
            }

        //! Load a number of integrator variables
        /*! \param n Number of variables to load
            When loading from a file, a given number of integrator variables must be preloaded without registering them.
            This method does that. After calling load(n), setIntegratorVariables() can be called for \a i up to \a n-1
        */
        void load(unsigned int n)
            {
            m_integrator_variables.resize(n);
            }

        //! Get a collection of integrator variables
        /*! \param i access integrator variables for integrator i
        */
        const IntegratorVariables& getIntegratorVariables(unsigned int i) const
            {
            assert(i < m_integrator_variables.size()); return m_integrator_variables[i];
            }

        //! Set a collection of integrator variables
        /*! \param i set integrator variables for integrator i
            \param v Variables to set
        */
        void setIntegratorVariables(unsigned int i, const IntegratorVariables& v)
            {
            assert(i < m_integrator_variables.size()); m_integrator_variables[i] = v;
            }

    private:
        unsigned int m_num_registered;                                  //!< Number of integrators that have registered
        std::vector<IntegratorVariables> m_integrator_variables;        //!< List of the integrator variables defined

    };

//! Exports IntegratorData to python
void export_IntegratorData();

#endif
