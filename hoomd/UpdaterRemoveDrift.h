// Copyright (c) 2009-2024 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

/*! \file UpdaterRemoveDrift.h
    \brief Declares an updater that removes the average drift from the particles
*/

// inclusion guard
#ifndef _REMOVE_DRIFT_UPDATER_H_
#define _REMOVE_DRIFT_UPDATER_H_

#include "hoomd/Updater.h"

#ifndef __HIPCC__
#include <pybind11/pybind11.h>
#endif

namespace hoomd
    {
/** This updater removes the average particle drift from the reference positions.
 * The minimum image convention is applied to each particle displacement from the
 * reference configuration before averaging over N_particles. The particles are
 * wrapped back into the box after the mean drift is substracted.
 */
class UpdaterRemoveDrift : public Updater
    {
    public:
    //! Constructor
    UpdaterRemoveDrift(std::shared_ptr<SystemDefinition> sysdef,
                       std::shared_ptr<Trigger> trigger,
                       pybind11::array_t<double> ref_positions)
        : Updater(sysdef, trigger)
        {
        setReferencePositions(ref_positions);
        }

    //! Set reference positions from a (N_particles, 3) numpy array
    void setReferencePositions(const pybind11::array_t<double> ref_pos)
        {
        if (ref_pos.ndim() != 2)
            {
            throw std::runtime_error("The array must be of shape (N_particles, 3).");
            }

        const size_t N_particles = ref_pos.shape(0);
        const size_t dim = ref_pos.shape(1);
        if (N_particles != this->m_pdata->getNGlobal() || dim != 3)
            {
            throw std::runtime_error("The array must be of shape (N_particles, 3).");
            }
        const double* rawdata = static_cast<const double*>(ref_pos.data());
        m_ref_positions.resize(m_pdata->getNGlobal());
        for (size_t i = 0; i < N_particles; i++)
            {
            const size_t array_index = i * 3;
            this->m_ref_positions[i] = vec3<Scalar>(rawdata[array_index],
                                                    rawdata[array_index + 1],
                                                    rawdata[array_index + 2]);
            }

#ifdef ENABLE_MPI
        if (this->m_pdata->getDomainDecomposition())
            {
            bcast(m_ref_positions, 0, m_exec_conf->getMPICommunicator());
            }
#endif
        }

    //! Get reference positions as a (N_particles, 3) numpy array
    pybind11::array_t<Scalar> getReferencePositions() const
        {
        std::vector<size_t> dims(2);
        dims[0] = this->m_ref_positions.size();
        dims[1] = 3;
        // the cast from vec3<Scalar>* to Scalar* is safe since vec3 is tightly packed without any
        // padding. This also makes a copy so, modifications of this array do not effect the
        // original reference positions.
        const auto reference_array = pybind11::array_t<Scalar>(
            dims,
            reinterpret_cast<const Scalar*>(this->m_ref_positions.data()));
        // This is necessary to expose the array in a read only fashion through C++
        reinterpret_cast<pybind11::detail::PyArray_Proxy*>(reference_array.ptr())->flags
            &= ~pybind11::detail::npy_api::NPY_ARRAY_WRITEABLE_;
        return reference_array;
        }

    //! Take one timestep forward
    virtual void update(uint64_t timestep)
        {
        ArrayHandle<Scalar4> h_postype(this->m_pdata->getPositions(),
                                       access_location::host,
                                       access_mode::readwrite);
        ArrayHandle<unsigned int> h_tag(this->m_pdata->getTags(),
                                        access_location::host,
                                        access_mode::read);
        ArrayHandle<int3> h_image(this->m_pdata->getImages(),
                                  access_location::host,
                                  access_mode::readwrite);
        const BoxDim box = this->m_pdata->getGlobalBox();
        const vec3<Scalar> origin(this->m_pdata->getOrigin());
        vec3<Scalar> rshift;
        rshift.x = rshift.y = rshift.z = 0.0f;

        for (unsigned int i = 0; i < this->m_pdata->getN(); i++)
            {
            unsigned int tag_i = h_tag.data[i];
            // read in the current position and orientation
            vec3<Scalar> postype_i = vec3<Scalar>(h_postype.data[i]) - origin;
            int3 tmp_image = make_int3(0, 0, 0);
            box.wrap(postype_i, tmp_image);
            const vec3<Scalar> dr = postype_i - m_ref_positions[tag_i];
            rshift += vec3<Scalar>(box.minImage(vec_to_scalar3(dr)));
            }

#ifdef ENABLE_MPI
        if (this->m_pdata->getDomainDecomposition())
            {
            Scalar r[3] = {rshift.x, rshift.y, rshift.z};
            MPI_Allreduce(MPI_IN_PLACE,
                          &r[0],
                          3,
                          MPI_HOOMD_SCALAR,
                          MPI_SUM,
                          m_exec_conf->getMPICommunicator());
            rshift.x = r[0];
            rshift.y = r[1];
            rshift.z = r[2];
            }
#endif

        rshift /= Scalar(this->m_pdata->getNGlobal());

        for (unsigned int i = 0; i < this->m_pdata->getN(); i++)
            {
            // read in the current position and orientation
            Scalar4 postype_i = h_postype.data[i];
            const vec3<Scalar> r_i = vec3<Scalar>(postype_i);
            h_postype.data[i] = vec_to_scalar4(r_i - rshift, postype_i.w);
            box.wrap(h_postype.data[i], h_image.data[i]);
            }
        }

    protected:
    std::vector<vec3<Scalar>> m_ref_positions;
    };

namespace detail
    {
/// Export the UpdaterRemoveDrift to python
void export_UpdaterRemoveDrift(pybind11::module& m)
    {
    pybind11::class_<UpdaterRemoveDrift, Updater, std::shared_ptr<UpdaterRemoveDrift>>(
        m,
        "UpdaterRemoveDrift")
        .def(pybind11::init<std::shared_ptr<SystemDefinition>,
                            std::shared_ptr<Trigger>,
                            pybind11::array_t<double>>())
        .def_property("reference_positions",
                      &UpdaterRemoveDrift::getReferencePositions,
                      &UpdaterRemoveDrift::setReferencePositions);
    }

    } // end namespace detail

    } // end namespace hoomd

#endif // _REMOVE_DRIFT_UPDATER_H_
