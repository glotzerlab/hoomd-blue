// Copyright (c) 2009-2022 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

/*! \file MuellerPlatheFlowGPU.h

    \brief Declares a class to exchange velocities of
           different spatial region, to create a flow.
           GPU accelerated version.
*/

#ifdef ENABLE_HIP
// Above this line shared constructs can be declared.
#ifndef __HIPCC__
#include "MuellerPlatheFlow.h"
#include "hoomd/Autotuner.h"
#include "hoomd/ParticleGroup.h"
#include "hoomd/Updater.h"
#include "hoomd/Variant.h"
#include <pybind11/pybind11.h>

#include <cfloat>
#include <memory>

#ifndef __MUELLER_PLATHE_FLOW_GPU_H__
#define __MUELLER_PLATHE_FLOW_GPU_H__

namespace hoomd
    {
namespace md
    {
//! By exchanging velocities based on their spatial position a flow is created. GPU accelerated
/*! \ingroup computes
 */
class MuellerPlatheFlowGPU : public MuellerPlatheFlow
    {
    public:
    //! Constructs the compute
    //!
    //! \param direction Indicates the normal direction of the slabs.
    //! \param N_slabs Number of total slabs in the simulation box.
    //! \param min_slabs Index of slabs, where the min velocity is searched.
    //! \param max_slabs Index of slabs, where the max velocity is searched.
    //! \note N_slabs should be a multiple of the DomainDecomposition boxes in that direction.
    //! If it is not, the number is rescaled and the user is informed.
    MuellerPlatheFlowGPU(std::shared_ptr<SystemDefinition> sysdef,
                         std::shared_ptr<ParticleGroup> group,
                         std::shared_ptr<Variant> flow_target,
                         std::string slab_direction_str,
                         std::string flow_direction_str,
                         const unsigned int N_slabs,
                         const unsigned int min_slab,
                         const unsigned int max_slab,
                         Scalar flow_epsilon);

    //! Destructor
    virtual ~MuellerPlatheFlowGPU(void);

    //! Set autotuner parameters
    /*! \param enable Enable/disable autotuning
        \param period period (approximate) in time steps when returning occurs
    */
    virtual void setAutotunerParams(bool enable, unsigned int period)
        {
        MuellerPlatheFlow::setAutotunerParams(enable, period);
        m_tuner->setPeriod(period);
        m_tuner->setEnabled(enable);
        }

    protected:
    std::unique_ptr<Autotuner> m_tuner; //!< Autotuner for block size

    virtual void searchMinMaxVelocity(void);
    virtual void updateMinMaxVelocity(void);
    };

namespace detail
    {
//! Exports the MuellerPlatheFlow class to python
void export_MuellerPlatheFlowGPU(pybind11::module& m);

    } // end namespace detail
    } // end namespace md
    } // end namespace hoomd

#endif // __HIPCC__
#endif //__MUELLER_PLATHE_FLOW_GPU_H__
#endif // ENABLE_HIP
