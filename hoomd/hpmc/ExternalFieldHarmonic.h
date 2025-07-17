// Copyright (c) 2009-2025 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

#pragma once

#include "hoomd/Compute.h"
#include "hoomd/VectorMath.h"
#include <hoomd/Variant.h>

#include "ExternalPotential.h"

#ifndef __HIPCC__
#include <pybind11/pybind11.h>
#endif

namespace hoomd
    {
namespace hpmc
    {

class ExternalFieldHarmonic : public ExternalPotential
    {
    public:
    //! Constructor
    ExternalFieldHarmonic(std::shared_ptr<SystemDefinition> sysdef,
                          pybind11::array_t<double> r0,
                          std::shared_ptr<Variant> k,
                          pybind11::array_t<double> q0,
                          std::shared_ptr<Variant> q,
                          pybind11::array_t<double> symRotations)
        : ExternalPotential(sysdef), m_k_translational(k), m_k_rotational(q)
        {
        setReferencePositions(r0);
        setReferenceOrientations(q0);
        setSymmetricallyEquivalentOrientations(symRotations); // TODO: check for identity?

        // connect updateMemberTags() method to maximum particle number change signal
        this->m_pdata->getGlobalParticleNumberChangeSignal()
            .template connect<ExternalFieldHarmonic,
                              &ExternalFieldHarmonic::slotGlobalParticleNumChange>(this);
        } // end constructor

    //! Destructor
    virtual ~ExternalFieldHarmonic()
        {
        if (this->m_pdata)
            {
            this->m_pdata->getGlobalParticleNumberChangeSignal()
                .template disconnect<ExternalFieldHarmonic,
                                     &ExternalFieldHarmonic::slotGlobalParticleNumChange>(this);
            }
        } // end destructor

    //! Set reference positions from a (N_particles, 3) numpy array
    void setReferencePositions(const pybind11::array_t<double> ref_pos)
        {
        m_reference_positions.resize(this->m_pdata->getNGlobal());
        if (this->m_exec_conf->getRank() == 0)
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
            for (size_t i = 0; i < N_particles * 3; i += 3)
                {
                this->m_reference_positions[i / 3]
                    = vec3<Scalar>(rawdata[i], rawdata[i + 1], rawdata[i + 2]);
                }
            }

#ifdef ENABLE_MPI
        if (this->m_sysdef->isDomainDecomposed())
            {
            bcast(m_reference_positions, 0, m_exec_conf->getMPICommunicator());
            }
#endif
        } // end setReferencePositions

    //! Set reference orientations from a (N_particles, 4) numpy array
    void setReferenceOrientations(const pybind11::array_t<double> ref_ors)
        {
        m_reference_orientations.resize(this->m_pdata->getNGlobal());
        if (this->m_exec_conf->getRank() == 0)
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
            for (size_t i = 0; i < N_particles * 4; i += 4)
                {
                this->m_reference_orientations[i / 4]
                    = quat<Scalar>(rawdata[i],
                                   vec3<Scalar>(rawdata[i + 1], rawdata[i + 2], rawdata[i + 3]));
                }
            }

#ifdef ENABLE_MPI
        if (this->m_sysdef->isDomainDecomposed())
            {
            bcast(m_reference_orientations, 0, m_exec_conf->getMPICommunicator());
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
        for (size_t i = 0; i < N_sym * 4; i += 4)
            {
            this->m_symmetry[i / 4]
                = quat<Scalar>(rawdata[i],
                               vec3<Scalar>(rawdata[i + 1], rawdata[i + 2], rawdata[i + 3]));
            }

#ifdef ENABLE_MPI
        if (this->m_sysdef->isDomainDecomposed())
            {
            bcast(m_symmetry, 0, m_exec_conf->getMPICommunicator());
            }
#endif
        } // end setSymmetricallyEquivalentOrientations

    //! Get reference positions as a (N_particles, 3) numpy array
    pybind11::array_t<Scalar> getReferencePositions() const
        {
        std::vector<size_t> dims(2);
        dims[0] = this->m_reference_positions.size();
        dims[1] = 3;
        // the cast from vec3<Scalar>* to Scalar* is safe since vec3 is tightly packed without any
        // padding. This also makes a copy so, modifications of this array do not effect the
        // original reference positions.
        const auto reference_array = pybind11::array_t<Scalar>(
            dims,
            reinterpret_cast<const Scalar*>(this->m_reference_positions.data()));
        // This is necessary to expose the array in a read only fashion through C++
        reinterpret_cast<pybind11::detail::PyArray_Proxy*>(reference_array.ptr())->flags
            &= ~pybind11::detail::npy_api::NPY_ARRAY_WRITEABLE_;
        return reference_array;
        }

    //! Get reference orientations as a (N_particles, 4) numpy array
    pybind11::array_t<Scalar> getReferenceOrientations() const
        {
        std::vector<size_t> dims(2);
        dims[0] = this->m_reference_orientations.size();
        dims[1] = 4;
        // the cast from vec3<Scalar>* to Scalar* is safe since vec3 is tightly packed without any
        // padding. This also makes a copy so, modifications of this array do not effect the
        // original reference positions.
        const auto reference_array = pybind11::array_t<Scalar>(
            dims,
            reinterpret_cast<const Scalar*>(this->m_reference_orientations.data()));
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
    void setKTranslational(const std::shared_ptr<Variant>& k_translational)
        {
        m_k_translational = k_translational;
        }

    //! Getter for translational spring constant
    std::shared_ptr<Variant> getKTranslational()
        {
        return m_k_translational;
        }

    //! Setter for rotational spring constant
    void setKRotational(const std::shared_ptr<Variant>& k_rotational)
        {
        m_k_rotational = k_rotational;
        }

    //! Getter for rotational spring constant
    std::shared_ptr<Variant> getKRotational()
        {
        return m_k_rotational;
        }

    //! Helper function to be called when particles are added/removed
    void slotGlobalParticleNumChange()
        {
        if (m_reference_positions.size() != this->m_pdata->getNGlobal()
            || m_reference_orientations.size() != this->m_pdata->getNGlobal())
            {
            throw std::runtime_error("Number of particles no longer equals number of reference "
                                     "points in ExternalFieldHarmonic.");
            }
        }

    protected:
    //! Calculate the energy associated with the deviation of a single particle from its reference
    //! position
    Scalar calcE_trans(uint64_t timestep, const unsigned int& tag, const vec3<Scalar>& position)
        {
        const BoxDim box = this->m_pdata->getGlobalBox();
        vec3<Scalar> r0 = m_reference_positions[tag];
        vec3<Scalar> dr = vec3<Scalar>(box.minImage(vec_to_scalar3(r0 - position)));
        Scalar k = (*m_k_translational)(timestep);
        return Scalar(0.5) * k * dot(dr, dr);
        }

    //! Calculate the energy associated with the deviation of a single particle from its reference
    //! orientation
    Scalar calcE_rot(uint64_t timestep, const unsigned int& tag, const quat<Scalar>& orientation)
        {
        assert(m_symmetry.size());
        quat<Scalar> q0 = m_reference_orientations[tag];
        Scalar dqmin = 0.0;
        for (size_t i = 0; i < m_symmetry.size(); i++)
            {
            quat<Scalar> equiv_orientation = orientation * m_symmetry[i];
            quat<Scalar> dq = q0 - equiv_orientation;
            dqmin = (i == 0) ? norm2(dq) : fmin(dqmin, norm2(dq));
            }
        Scalar k = (*m_k_rotational)(timestep);
        return Scalar(0.5) * k * dqmin;
        }

    /** Evaluate the lattice constraint energy of a single particle.
     */
    virtual LongReal particleEnergyImplementation(uint64_t timestep,
                                                  unsigned int tag_i,
                                                  unsigned int type_i,
                                                  const vec3<LongReal>& r_i,
                                                  const quat<LongReal>& q_i,
                                                  LongReal charge_i,
                                                  Trial trial)
        {
        Scalar energy = 0.0;
        energy += calcE_trans(timestep, tag_i, r_i);
        energy += calcE_rot(timestep, tag_i, q_i);
        return energy;
        }

    private:
    std::vector<vec3<Scalar>> m_reference_positions;    // reference positions
    std::vector<quat<Scalar>> m_reference_orientations; // reference orientations
    std::vector<quat<Scalar>> m_symmetry;               // symmetry-equivalent orientations
    std::shared_ptr<Variant> m_k_translational;         // translational spring constant
    std::shared_ptr<Variant> m_k_rotational;            // rotational spring constant
    };

namespace detail
    {
void export_ExternalHarmonicField(pybind11::module& m)
    {
    pybind11::class_<ExternalFieldHarmonic,
                     ExternalPotential,
                     std::shared_ptr<ExternalFieldHarmonic>>(m, "PotentialExternalHarmonic")
        .def(pybind11::init<std::shared_ptr<SystemDefinition>,
                            pybind11::array_t<double>,
                            std::shared_ptr<Variant>,
                            pybind11::array_t<double>,
                            std::shared_ptr<Variant>,
                            pybind11::array_t<double>>())
        .def_property("reference_positions",
                      &ExternalFieldHarmonic::getReferencePositions,
                      &ExternalFieldHarmonic::setReferencePositions)
        .def_property("reference_orientations",
                      &ExternalFieldHarmonic::getReferenceOrientations,
                      &ExternalFieldHarmonic::setReferenceOrientations)
        .def_property("k_translational",
                      &ExternalFieldHarmonic::getKTranslational,
                      &ExternalFieldHarmonic::setKTranslational)
        .def_property("k_rotational",
                      &ExternalFieldHarmonic::getKRotational,
                      &ExternalFieldHarmonic::setKRotational)
        .def_property("symmetries",
                      &ExternalFieldHarmonic::getSymmetricallyEquivalentOrientations,
                      &ExternalFieldHarmonic::setSymmetricallyEquivalentOrientations);
    }
    } // end namespace detail
    } // namespace hpmc
    } // end namespace hoomd
