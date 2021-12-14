// Copyright (c) 2009-2021 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.

#ifndef _EXTERNAL_FIELD_LATTICE_H_
#define _EXTERNAL_FIELD_LATTICE_H_

/*! \file ExternalField.h
    \brief Declaration of ExternalField base class
*/

#include "hoomd/Compute.h"
#include "hoomd/HOOMDMPI.h"
#include "hoomd/VectorMath.h"

#include "ExternalField.h"

#ifndef __HIPCC__
#include <pybind11/pybind11.h>
#endif

namespace hoomd
    {
namespace hpmc
    {

template<class Shape> class ExternalFieldLattice : public ExternalFieldMono<Shape>
    {
    using ExternalFieldMono<Shape>::m_pdata;
    using ExternalFieldMono<Shape>::m_exec_conf;
    using ExternalFieldMono<Shape>::m_sysdef;

    public:
    //! Constructor
    ExternalFieldLattice(std::shared_ptr<SystemDefinition> sysdef,
                         pybind11::array_t<double> r0,
                         Scalar k,
                         pybind11::array_t<double> q0,
                         Scalar q,
                         pybind11::array_t<double> symRotations)
        : ExternalFieldMono<Shape>(sysdef), m_k_translational(k), m_k_rotational(q)
        {
        setReferencePositions(r0);
        setReferenceOrientations(q0);
        setSymmetricallyEquivalentOrientations(symRotations); // TODO: check for identity?

        // connect updateMemberTags() method to maximum particle number change signal
        m_pdata->getGlobalParticleNumberChangeSignal()
            .template connect<ExternalFieldLattice,
                              &ExternalFieldLattice::slotGlobalParticleNumChange>(this);
        } // end constructor

    //! Destructor
    ~ExternalFieldLattice()
        {
        if (m_pdata)
            {
            m_pdata->getGlobalParticleNumberChangeSignal()
                .template disconnect<ExternalFieldLattice,
                                     &ExternalFieldLattice::slotGlobalParticleNumChange>(this);
            }
        } // end destructor

    //! Set reference positions from a (N_particles, 3) numpy array
    void setReferencePositions(const pybind11::array_t<double> ref_pos)
        {
        m_lattice_positions.resize(m_pdata->getNGlobal());
        if (m_exec_conf->getRank() == 0)
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
            for (size_t i = 0; i < min(N_particles, m_pdata->getNGlobal()); i++)
                {
                const size_t array_index = i * 3;
                this->m_lattice_positions[i] = vec3<Scalar>(rawdata[array_index],
                                                            rawdata[array_index + 1],
                                                            rawdata[array_index + 2]);
                }
            }

#ifdef ENABLE_MPI
        if (this->m_pdata->isDomainDecomposed())
            {
            bcast(m_lattice_positions, 0, m_exec_conf->getMPICommunicator());
            }
#endif
        } // end setReferencePositions

    //! Set reference orientations from a (N_particles, 4) numpy array
    void setReferenceOrientations(const pybind11::array_t<double> ref_ors)
        {
        m_lattice_orientations.resize(m_pdata->getNGlobal());
        if (m_exec_conf->getRank() == 0)
            {
            if (ref_ors.ndim() != 2)
                {
                throw std::runtime_error("The array must be of shape (N_particles, 4).");
                }

            const size_t N_particles = ref_ors.shape(0);
            const size_t dim = ref_ors.shape(1);
            if (N_particles != this->m_pdata->getNGlobal() || dim != 4)
                {
                throw std::runtime_error("The array must be of shape (N_particles, 4).");
                }
            const double* rawdata = static_cast<const double*>(ref_ors.data());
            for (size_t i = 0; i < N_particles; i++)
                {
                const size_t array_index = i * 4;
                this->m_lattice_orientations[i] = quat<Scalar>(rawdata[array_index],
                                                               vec3<Scalar>(rawdata[array_index + 1],
                                                                            rawdata[array_index + 2],
                                                                            rawdata[array_index + 3]));
                }
            }

#ifdef ENABLE_MPI
        if (this->m_pdata->isDomainDecomposed())
            {
            bcast(m_lattice_orientations, 0, m_exec_conf->getMPICommunicator());
            }
#endif
        } // end setReferenceOrientations

    //! Set symmetrically equivalent orientations from a (N_symmetry, 4) numpy array
    void
    setSymmetricallyEquivalentOrientations(const pybind11::array_t<double> equivalent_quaternions)
        {
        if (equivalent_quaternions.ndim() != 2)
            {
            throw std::runtime_error("The array must be of shape (N_sym, 4).");
            }

        const size_t N_sym = equivalent_quaternions.shape(0); // Number of equivalent orientations
        const size_t dim = equivalent_quaternions.shape(1);
        if (dim != 4)
            {
            throw std::runtime_error("The array must be of shape (N_sym, 4).");
            }
        const double* rawdata = static_cast<const double*>(equivalent_quaternions.data());
        m_symmetry.resize(N_sym);
        for (size_t i = 0; i < N_sym; i++)
            {
            const size_t array_index = i * 4;
            this->m_symmetry[i] = quat<Scalar>(rawdata[array_index],
                                               vec3<Scalar>(rawdata[array_index + 1],
                                                            rawdata[array_index + 2],
                                                            rawdata[array_index + 3]));
            }

#ifdef ENABLE_MPI
        if (this->m_pdata->isDomainDecomposed())
            {
            bcast(m_symmetry, 0, m_exec_conf->getMPICommunicator());
            }
#endif
        } // end setSymmetricallyEquivalentOrientations

    //! Get lattice positions as a (N_particles, 3) numpy array
    pybind11::array_t<Scalar> getReferencePositions() const
        {
        std::vector<size_t> dims(2);
        dims[0] = this->m_lattice_positions.size();
        dims[1] = 3;
        // the cast from vec3<Scalar>* to Scalar* is safe since vec3 is tightly packed without any
        // padding. This also makes a copy so, modifications of this array do not effect the
        // original reference positions.
        const auto reference_array = pybind11::array_t<Scalar>(
            dims,
            reinterpret_cast<const Scalar*>(this->m_lattice_positions.data()));
        // This is necessary to expose the array in a read only fashion through C++
        reinterpret_cast<pybind11::detail::PyArray_Proxy*>(reference_array.ptr())->flags
            &= ~pybind11::detail::npy_api::NPY_ARRAY_WRITEABLE_;
        return reference_array;
        }

    //! Get lattice orientations as a (N_particles, 4) numpy array
    pybind11::array_t<Scalar> getReferenceOrientations() const
        {
        std::vector<size_t> dims(2);
        dims[0] = this->m_lattice_orientations.size();
        dims[1] = 4;
        // the cast from vec3<Scalar>* to Scalar* is safe since vec3 is tightly packed without any
        // padding. This also makes a copy so, modifications of this array do not effect the
        // original reference positions.
        const auto reference_array = pybind11::array_t<Scalar>(
            dims,
            reinterpret_cast<const Scalar*>(this->m_lattice_orientations.data()));
        // This is necessary to expose the array in a read only fashion through C++
        reinterpret_cast<pybind11::detail::PyArray_Proxy*>(reference_array.ptr())->flags
            &= ~pybind11::detail::npy_api::NPY_ARRAY_WRITEABLE_;
        return reference_array;
        }

    //! Get symmetrically equivalent orientations as a (N_particles, 4) numpy array
    pybind11::array_t<Scalar> getSymmetricallyEquivalentOrientations() const
        {
        std::vector<size_t> dims(2);
        dims[0] = this->m_symmetry.size();
        dims[1] = 4;
        // the cast from vec3<Scalar>* to Scalar* is safe since vec3 is tightly packed without any
        // padding. This also makes a copy so, modifications of this array do not effect the
        // original reference positions.
        const auto reference_array
            = pybind11::array_t<Scalar>(dims,
                                        reinterpret_cast<const Scalar*>(this->m_symmetry.data()));
        // This is necessary to expose the array in a read only fashion through C++
        reinterpret_cast<pybind11::detail::PyArray_Proxy*>(reference_array.ptr())->flags
            &= ~pybind11::detail::npy_api::NPY_ARRAY_WRITEABLE_;
        return reference_array;
        }

    //! Setter for translational spring constant
    void setKTranslational(Scalar k_translational)
        {
        m_k_translational = k_translational;
        }

    //! Getter for translational spring constant
    Scalar getKTranslational()
        {
        return m_k_translational;
        }

    //! Setter for rotational spring constant
    void setKRotational(Scalar k_rotational)
        {
        m_k_rotational = k_rotational;
        }

    //! Getter for rotational spring constant
    Scalar getKRotational()
        {
        return m_k_rotational;
        }

    //! Helper function to be called when particles are added/removed
    void slotGlobalParticleNumChange()
        {
        if (m_lattice_positions.size() != this->m_pdata->getNGlobal()
            || m_lattice_orientations.size() != this->m_pdata->getNGlobal())
            {
            throw std::runtime_error("Number of particles no longer equals number of lattice "
                                     "points in ExternalFieldLattice.");
            }
        }

    /** Calculate the change in energy for trial moves
     *
     * This function currently ignores any information associated with box changes.
     * However, this function only gets called in the box updater, so it should not ignore box
     * changes. But why would you have a lattice field and a box updater active in the same
     * simulation?
     */
    double calculateDeltaE(const Scalar4* const position_old_arg, // why is this a Scalar4?
                           const Scalar4* const orientation_old_arg,
                           const BoxDim* const box_old_arg)
        {
        ArrayHandle<Scalar4> h_pos(m_pdata->getPositions(),
                                   access_location::host,
                                   access_mode::readwrite);
        ArrayHandle<Scalar4> h_orient(m_pdata->getOrientationArray(),
                                      access_location::host,
                                      access_mode::readwrite);
        const Scalar4* const position_new = h_pos.data; // current positions from system definition
        const Scalar4* const orientation_new
            = h_orient.data; // current orientations from system definition
        // const BoxDim* const box_new = &m_pdata->getGlobalBox();
        const Scalar4 *position_old = position_old_arg, *orientation_old = orientation_old_arg;
        // const BoxDim* box_old = box_old_arg;
        if (!position_old)
            position_old = position_new;
        if (!orientation_old)
            orientation_old = orientation_new;
        /*if (!box_old)
            box_old = box_new;

        Scalar curVolume = m_pdata->getBox().getVolume();
        Scalar newVolume = box_new->getVolume();
        Scalar oldVolume = box_old->getVolume();
        Scalar scaleOld = pow((oldVolume / curVolume), Scalar(1.0 / 3.0));
        Scalar scaleNew = pow((newVolume / curVolume), Scalar(1.0 / 3.0));
        */

        double dE = 0.0;
        for (unsigned int i = 0; i < m_pdata->getN(); i++)
            {
            Scalar old_E
                = calcE(i, vec3<Scalar>(*(position_old + i)), quat<Scalar>(*(orientation_old + i)));
            // scaleOld);
            Scalar new_E
                = calcE(i, vec3<Scalar>(*(position_new + i)), quat<Scalar>(*(orientation_new + i)));
            // scaleNew);
            dE += new_E - old_E;
            }

#ifdef ENABLE_MPI
        if (this->m_pdata->isDomainDecomposed())
            {
            MPI_Allreduce(MPI_IN_PLACE,
                          &dE,
                          1,
                          MPI_HOOMD_SCALAR,
                          MPI_SUM,
                          m_exec_conf->getMPICommunicator());
            }
#endif

        return dE;
        }

    /** This function gets called during the update step in the MC integrator
     *
     * What purpose does this function serve?
     */
    Scalar getEnergy(uint64_t timestep)
        {
        Scalar energy = Scalar(0.0);
        // access particle data and system box
        ArrayHandle<Scalar4> h_postype(m_pdata->getPositions(),
                                       access_location::host,
                                       access_mode::read);
        ArrayHandle<Scalar4> h_orient(m_pdata->getOrientationArray(),
                                      access_location::host,
                                      access_mode::read);
        for (unsigned int i = 0; i < m_pdata->getN(); i++)
            {
            vec3<Scalar> position(h_postype.data[i]);
            quat<Scalar> orientation(h_orient.data[i]);
            energy += calcE(i, position, orientation);
            }

#ifdef ENABLE_MPI
        if (this->m_pdata->isDomainDecomposed())
            {
            MPI_Allreduce(MPI_IN_PLACE,
                          &energy,
                          1,
                          MPI_HOOMD_SCALAR,
                          MPI_SUM,
                          m_exec_conf->getMPICommunicator());
            }
#endif

        return energy;
        } // end getEnergy(uin64_t)

    //! Calculate the change in energy from moving a single particle with tag = index
    double energydiff(const unsigned int& index,
                      const vec3<Scalar>& position_old,
                      const Shape& shape_old,
                      const vec3<Scalar>& position_new,
                      const Shape& shape_new)
        {
        double old_U = calcE(index, position_old, shape_old),
               new_U = calcE(index, position_new, shape_new);
        return new_U - old_U;
        }

    protected:
    //! Calculate the energy associated with the deviation of a single particle from its reference
    //! position
    Scalar
    calcE_trans(const unsigned int& index, const vec3<Scalar>& position, const Scalar& scale = 1.0)
        {
        ArrayHandle<unsigned int> h_tags(m_pdata->getTags(),
                                         access_location::host,
                                         access_mode::read);
        int3 dummy = make_int3(0, 0, 0);
        vec3<Scalar> origin(m_pdata->getOrigin());
        const BoxDim& box = this->m_pdata->getGlobalBox();
        vec3<Scalar> r0 = m_lattice_positions[h_tags.data[index]];
        r0 *= scale;
        Scalar3 t = vec_to_scalar3(position - origin);
        box.wrap(t, dummy);
        vec3<Scalar> shifted_pos(t);
        vec3<Scalar> dr = vec3<Scalar>(box.minImage(vec_to_scalar3(r0 - position + origin)));
        return m_k_translational * dot(dr, dr);
        }

    //! Calculate the energy associated with the deviation of a single particle from its reference
    //! orientation
    Scalar calcE_rot(const unsigned int& index, const quat<Scalar>& orientation)
        {
        assert(m_symmetry.size());
        ArrayHandle<unsigned int> h_tags(m_pdata->getTags(),
                                         access_location::host,
                                         access_mode::read);
        quat<Scalar> q0 = m_lattice_orientations[h_tags.data[index]];
        Scalar dqmin = 0.0;
        for (size_t i = 0; i < m_symmetry.size(); i++)
            {
            quat<Scalar> equiv_orientation = orientation * m_symmetry[i];
            quat<Scalar> dq = q0 - equiv_orientation;
            dqmin = (i == 0) ? norm2(dq) : fmin(dqmin, norm2(dq));
            }
        return m_k_rotational * dqmin;
        }

    Scalar calcE_rot(const unsigned int& index, const Shape& shape)
        {
        if (!shape.hasOrientation())
            return Scalar(0.0);

        return calcE_rot(index, shape.orientation);
        }

    /** Calculate the total energy associated with the deviation of a single particle from its ref.
     * pos. and orientation
     *
     * This function _should_ only be used for logging purposes and not for calculating move
     * acceptance criteria, since it's the energy difference that matters for the latter.
     */
    Scalar calcE(const unsigned int& index,
                 const vec3<Scalar>& position,
                 const quat<Scalar>& orientation,
                 const Scalar& scale = 1.0)
        {
        Scalar energy = 0.0;
        energy += calcE_trans(index, position, scale);
        energy += calcE_rot(index, orientation);
        return energy;
        }

    Scalar calcE(const unsigned int& index,
                 const vec3<Scalar>& position,
                 const Shape& shape,
                 const Scalar& scale = 1.0)
        {
        return calcE(index, position, shape.orientation, scale);
        }

    private:
    std::vector<vec3<Scalar>> m_lattice_positions;    // reference positions
    std::vector<quat<Scalar>> m_lattice_orientations; // reference orientations
    std::vector<quat<Scalar>> m_symmetry;             // symmetry-equivalent orientations
    Scalar m_k_translational;                         // translational spring constant
    Scalar m_k_rotational;                            // rotational spring constant
    };

namespace detail
    {
template<class Shape> void export_LatticeField(pybind11::module& m, std::string name)
    {
    pybind11::class_<ExternalFieldLattice<Shape>,
                     ExternalFieldMono<Shape>,
                     std::shared_ptr<ExternalFieldLattice<Shape>>>(m, name.c_str())
        .def(pybind11::init<std::shared_ptr<SystemDefinition>,
                            pybind11::array_t<double>,
                            Scalar,
                            pybind11::array_t<double>,
                            Scalar,
                            pybind11::array_t<double>>())
        .def_property("reference_positions",
                      &ExternalFieldLattice<Shape>::getReferencePositions,
                      &ExternalFieldLattice<Shape>::setReferencePositions)
        .def_property("reference_orientations",
                      &ExternalFieldLattice<Shape>::getReferenceOrientations,
                      &ExternalFieldLattice<Shape>::setReferenceOrientations)
        .def_property("k_translational",
                      &ExternalFieldLattice<Shape>::getKTranslational,
                      &ExternalFieldLattice<Shape>::setKTranslational)
        .def_property("k_rotational",
                      &ExternalFieldLattice<Shape>::getKRotational,
                      &ExternalFieldLattice<Shape>::setKRotational)
        .def_property("symmetries",
                      &ExternalFieldLattice<Shape>::getSymmetricallyEquivalentOrientations,
                      &ExternalFieldLattice<Shape>::setSymmetricallyEquivalentOrientations)
        .def("getEnergy", &ExternalFieldLattice<Shape>::getEnergy);
    }

void export_LatticeFields(pybind11::module& m);

    } // end namespace detail
    } // namespace hpmc
    } // end namespace hoomd

#endif // _EXTERNAL_FIELD_LATTICE_H_
