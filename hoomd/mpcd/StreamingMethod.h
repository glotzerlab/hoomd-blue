// Copyright (c) 2009-2024 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

/*!
 * \file mpcd/StreamingMethod.h
 * \brief Declaration of mpcd::StreamingMethod
 */

#ifndef MPCD_STREAMING_METHOD_H_
#define MPCD_STREAMING_METHOD_H_

#ifdef __HIPCC__
#error This header cannot be compiled by nvcc
#endif

#include "CellList.h"
#include "ExternalField.h"
#include "hoomd/Autotuned.h"
#include "hoomd/GPUPolymorph.h"
#include "hoomd/SystemDefinition.h"

#include <pybind11/pybind11.h>

namespace hoomd
    {
namespace mpcd
    {
//! MPCD streaming method
/*!
 * This method implements the base version of ballistic propagation of MPCD
 * particles.
 */
class PYBIND11_EXPORT StreamingMethod : public Autotuned
    {
    public:
    //! Constructor
    StreamingMethod(std::shared_ptr<SystemDefinition> sysdef,
                    unsigned int cur_timestep,
                    unsigned int period,
                    int phase);
    //! Destructor
    virtual ~StreamingMethod();

    //! Implementation of the streaming rule
    virtual void stream(uint64_t timestep) { }

    //! Peek if the next step requires streaming
    virtual bool peekStream(uint64_t timestep) const;

    //! Change the timestep
    /*!
     * \param deltaT Fundamental HOOMD integration timestep
     *
     * The streaming step size is set to period * deltaT so that each time
     * the streaming operation is called, the particles advance across the
     * full MPCD interval.
     */
    virtual void setDeltaT(Scalar deltaT)
        {
        m_mpcd_dt = Scalar(m_period) * deltaT;
        }

    //! Get the timestep
    Scalar getDeltaT() const
        {
        return m_mpcd_dt;
        }

    //! Set the external field
    void setField(std::shared_ptr<hoomd::GPUPolymorph<mpcd::ExternalField>> field)
        {
        m_field = field;
        }

    //! Remove the external field
    void removeField()
        {
        m_field.reset();
        }

    //! Set the period of the streaming method
    void setPeriod(unsigned int cur_timestep, unsigned int period);

    //! Set the cell list used for collisions
    virtual void setCellList(std::shared_ptr<mpcd::CellList> cl)
        {
        m_cl = cl;
        }

    protected:
    std::shared_ptr<SystemDefinition> m_sysdef;                //!< HOOMD system definition
    std::shared_ptr<hoomd::ParticleData> m_pdata;              //!< HOOMD particle data
    std::shared_ptr<mpcd::ParticleData> m_mpcd_pdata;          //!< MPCD particle data
    std::shared_ptr<mpcd::CellList> m_cl;                      //!< MPCD cell list
    std::shared_ptr<const ExecutionConfiguration> m_exec_conf; //!< Execution configuration

    Scalar m_mpcd_dt;         //!< Integration time step
    unsigned int m_period;    //!< Number of MD timesteps between streaming steps
    uint64_t m_next_timestep; //!< Timestep next streaming step should be performed

    std::shared_ptr<hoomd::GPUPolymorph<mpcd::ExternalField>> m_field; //!< External field

    //! Check if streaming should occur
    virtual bool shouldStream(uint64_t timestep);
    };

namespace detail
    {
//! Export mpcd::StreamingMethod to python
void export_StreamingMethod(pybind11::module& m);
    } // end namespace detail
    } // end namespace mpcd
    } // end namespace hoomd
#endif // MPCD_STREAMING_METHOD_H_
