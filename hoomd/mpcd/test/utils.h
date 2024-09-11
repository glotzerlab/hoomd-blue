// Copyright (c) 2009-2024 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

#ifndef MPCD_TEST_UTILS_H_
#define MPCD_TEST_UTILS_H_

#include "hoomd/mpcd/CellList.h"
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

Scalar3 scale(const Scalar3& ref_pos, std::shared_ptr<BoxDim> ref_box, std::shared_ptr<BoxDim> box)
    {
    return box->makeCoordinates(ref_box->makeFraction(ref_pos));
    }

Scalar4 scale(const Scalar4& ref_pos, std::shared_ptr<BoxDim> ref_box, std::shared_ptr<BoxDim> box)
    {
    const Scalar3 pos = scale(make_scalar3(ref_pos.x, ref_pos.y, ref_pos.z), ref_box, box);
    return make_scalar4(pos.x, pos.y, pos.z, ref_pos.w);
    }

vec3<Scalar>
scale(const vec3<Scalar>& ref_pos, std::shared_ptr<BoxDim> ref_box, std::shared_ptr<BoxDim> box)
    {
    return vec3<Scalar>(scale(vec_to_scalar3(ref_pos), ref_box, box));
    }

unsigned int make_local_cell(std::shared_ptr<mpcd::CellList> cl, int ix, int iy, int iz)
    {
    const int3 cell = cl->getLocalCell(make_int3(ix, iy, iz));
    return cl->getCellIndexer()(cell.x, cell.y, cell.z);
    }

    } // end namespace hoomd

#endif // MPCD_TEST_UTILS_H_
