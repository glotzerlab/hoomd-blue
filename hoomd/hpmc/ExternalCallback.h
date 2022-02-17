// Copyright (c) 2009-2022 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

#ifndef _EXTERNAL_CALLBACK_H_
#define _EXTERNAL_CALLBACK_H_

/*! \file ExternalCallback.h
    \brief Declaration of ExternalCallback base class
*/

#include "hoomd/Compute.h"
#include "hoomd/SnapshotSystemData.h"
#include "hoomd/VectorMath.h"

#include "ExternalField.h"

#ifndef __HIPCC__
#include <pybind11/pybind11.h>
#endif

namespace hoomd
    {
namespace hpmc
    {
template<class Shape>
class __attribute__((visibility("hidden"))) ExternalCallback : public ExternalFieldMono<Shape>
    {
    public:
    ExternalCallback(std::shared_ptr<SystemDefinition> sysdef, pybind11::object energy_function)
        : ExternalFieldMono<Shape>(sysdef), callback(energy_function)
        {
#ifdef ENABLE_MPI
        if (this->m_pdata->getDomainDecomposition())
            {
            throw std::runtime_error("ExternalCallback doesn't support MPI.");
            }
#endif
        }

    ~ExternalCallback() { }

    //! Compute Boltzmann weight exp(-U) of current configuration
    Scalar calculateBoltzmannWeight(uint64_t timestep)
        {
        auto snap = takeSnapshot();
        double energy = getEnergy(snap);
        return exp(-energy);
        }

    //! Compute DeltaU = Unew-Uold
    /*! \param position_old_arg Old (local) positions
        \param orientation_old_arg Old (local) orientations
        \param box_old Old (global) box
     */
    double calculateDeltaE(uint64_t timestep,
                           const Scalar4* const position_old_arg,
                           const Scalar4* const orientation_old_arg,
                           const BoxDim& box_old)
        {
        auto snap = takeSnapshot();
        double energy_new = getEnergy(snap);

        // update snapshot with old configuration
        // FIXME: this will not work in MPI, we will have to broadcast to root and modify snapshot
        // there
        snap->global_box = std::make_shared<BoxDim>(box_old);
        unsigned int N = this->m_pdata->getN();
        ArrayHandle<unsigned int> h_tag(this->m_pdata->getTags(),
                                        access_location::host,
                                        access_mode::read);
        for (unsigned int i = 0; i < N; ++i)
            {
            unsigned int tag = h_tag.data[i];
            auto snap_it = snap->map.find(tag);
            assert(snap_it != snap->map.end());
            unsigned int snap_idx = snap_it->second;
            snap->particle_data.pos[snap_idx] = vec3<Scalar>(position_old_arg[i]);
            if (orientation_old_arg != NULL)
                snap->particle_data.orientation[snap_idx] = quat<Scalar>(orientation_old_arg[i]);
            }
        double energy_old = getEnergy(snap);
        return energy_new - energy_old;
        }

    // does nothing
    void compute(uint64_t timestep) { }

    // Compute the energy difference for a proposed move on a single particle
    double energydiff(uint64_t timestep,
                      const unsigned int& index,
                      const vec3<Scalar>& position_old,
                      const Shape& shape_old,
                      const vec3<Scalar>& position_new,
                      const Shape& shape_new)
        {
        // find index in snapshot
        unsigned int tag;
            {
            ArrayHandle<unsigned int> h_tag(this->m_pdata->getTags(),
                                            access_location::host,
                                            access_mode::read);
            tag = h_tag.data[index];
            }

        auto snap = takeSnapshot();
        auto snap_it = snap->map.find(tag);
        assert(snap_it != snap->map.end());
        unsigned int snap_idx = snap_it->second;

        // update snapshot with old configuration
        snap->particle_data.pos[snap_idx] = position_old;
        snap->particle_data.orientation[snap_idx] = shape_old.orientation;
        double energy_old = getEnergy(snap);

        // update snapshot with new configuration
        snap->particle_data.pos[snap_idx] = position_new;
        snap->particle_data.orientation[snap_idx] = shape_new.orientation;
        double energy_new = getEnergy(snap);

        return energy_new - energy_old;
        }

    protected:
    // Take a snapshot of the particle data (only)
    std::shared_ptr<SnapshotSystemData<Scalar>> takeSnapshot()
        {
        return this->m_sysdef->template takeSnapshot<Scalar>();
        }

    double getEnergy(std::shared_ptr<SnapshotSystemData<Scalar>> snap)
        {
        double e = 0.0;
        if (!callback.is(pybind11::none()))
            {
            pybind11::object rv = callback(snap);
            try
                {
                e = pybind11::cast<Scalar>(rv);
                }
            catch (const std::exception& e)
                {
                throw std::runtime_error("Expected a scalar (energy/kT) as return value.");
                }
            }
        return e;
        }

    private:
    pybind11::object callback; //! The python callback
    };

namespace detail
    {
template<class Shape> void export_ExternalCallback(pybind11::module& m, const std::string& name)
    {
    pybind11::class_<ExternalCallback<Shape>,
                     ExternalFieldMono<Shape>,
                     std::shared_ptr<ExternalCallback<Shape>>>(m, name.c_str())
        .def(pybind11::init<std::shared_ptr<SystemDefinition>, pybind11::object>());
    }

    }  // end namespace detail
    }  // namespace hpmc
    }  // end namespace hoomd
#endif // _EXTERNAL_FIELD_LATTICE_H_
