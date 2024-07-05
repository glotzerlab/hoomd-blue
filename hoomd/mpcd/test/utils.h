// Copyright (c) 2009-2024 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

#ifndef MPCD_TEST_UTILS_H_
#define MPCD_TEST_UTILS_H_

#include "hoomd/mpcd/CellThermoCompute.h"

namespace hoomd
    {
//! Request object for all available thermo flags
class AllThermoRequest
    {
    public:
    //! Constructor
    /*!
     * \param thermo Thermo compute to supply flags to.
     *
     * \post This object is connected to \a thermo.
     */
    AllThermoRequest(std::shared_ptr<mpcd::CellThermoCompute> thermo) : m_thermo(thermo)
        {
        if (m_thermo)
            m_thermo->getFlagsSignal().connect<AllThermoRequest, &AllThermoRequest::operator()>(
                this);
        }

    //! Destructor
    /*!
     * \post This object is disconnected from its compute.
     */
    ~AllThermoRequest()
        {
        if (m_thermo)
            m_thermo->getFlagsSignal().disconnect<AllThermoRequest, &AllThermoRequest::operator()>(
                this);
        }

    //! Flag request operator
    /*!
     * \returns ThermoFlags with all bits set.
     */
    mpcd::detail::ThermoFlags operator()() const
        {
        mpcd::detail::ThermoFlags flags(0xffffffff);
        return flags;
        }

    private:
    std::shared_ptr<mpcd::CellThermoCompute> m_thermo;
    };

    } // end namespace hoomd

#endif // MPCD_TEST_UTILS_H_
