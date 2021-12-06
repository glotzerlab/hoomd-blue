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
    ExternalFieldLattice(std::shared_ptr<SystemDefinition> sysdef,
                         pybind11::list r0,
                         Scalar k,
                         pybind11::list q0,
                         Scalar q,
                         pybind11::list symRotations)
        : ExternalFieldMono<Shape>(sysdef), m_k_translational(k), m_k_rotational(q), m_energy(0.0)
        {
        setReferences(r0, q0);

        std::vector<Scalar4> rots;
        python_list_to_vector_scalar4(symRotations, rots);
        bool identityFound = false;
        quat<Scalar> identity(1, vec3<Scalar>(0, 0, 0));
        Scalar tol = 1e-5;
        for (size_t i = 0; i < rots.size(); i++)
            {
            quat<Scalar> qi(rots[i]);
            identityFound = !identityFound ? norm2(qi - identity) < tol : identityFound;
            m_symmetry.push_back(qi);
            }
        if (!identityFound) // ensure that the identity rotation is provided.
            {
            m_symmetry.push_back(identity);
            }
        }

    double calculateDeltaE(const Scalar4* const position_old_arg,
                           const Scalar4* const orientation_old_arg,
                           const BoxDim* const box_old_arg)
        {
        // TODO: rethink the formatting a bit.
        ArrayHandle<Scalar4> h_pos(m_pdata->getPositions(),
                                   access_location::host,
                                   access_mode::readwrite);
        ArrayHandle<Scalar4> h_orient(m_pdata->getOrientationArray(),
                                      access_location::host,
                                      access_mode::readwrite);
        const Scalar4* const position_new = h_pos.data;
        const Scalar4* const orientation_new = h_orient.data;
        const BoxDim* const box_new = &m_pdata->getGlobalBox();
        const Scalar4 *position_old = position_old_arg, *orientation_old = orientation_old_arg;
        const BoxDim* box_old = box_old_arg;
        if (!position_old)
            position_old = position_new;
        if (!orientation_old)
            orientation_old = orientation_new;
        if (!box_old)
            box_old = box_new;

        Scalar curVolume = m_box.getVolume();
        Scalar newVolume = box_new->getVolume();
        Scalar oldVolume = box_old->getVolume();
        Scalar scaleOld = pow((oldVolume / curVolume), Scalar(1.0 / 3.0));
        Scalar scaleNew = pow((newVolume / curVolume), Scalar(1.0 / 3.0));

        double dE = 0.0;
        for (unsigned int i = 0; i < m_pdata->getN(); i++)
            {
            Scalar old_E = calcE(i,
                                 vec3<Scalar>(*(position_old + i)),
                                 quat<Scalar>(*(orientation_old + i)),
                                 scaleOld);
            Scalar new_E = calcE(i,
                                 vec3<Scalar>(*(position_new + i)),
                                 quat<Scalar>(*(orientation_new + i)),
                                 scaleNew);
            dE += new_E - old_E;
            }

#ifdef ENABLE_MPI
        if (this->m_pdata->getDomainDecomposition())
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

    void compute(uint64_t timestep)
        {
        if (!this->shouldCompute(timestep))
            {
            return;
            }
        m_energy = Scalar(0.0);
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
            m_energy += calcE(i, position, orientation);
            }

#ifdef ENABLE_MPI
        if (this->m_pdata->getDomainDecomposition())
            {
            MPI_Allreduce(MPI_IN_PLACE,
                          &m_energy,
                          1,
                          MPI_HOOMD_SCALAR,
                          MPI_SUM,
                          m_exec_conf->getMPICommunicator());
            }
#endif

        }

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

    /**
     * Set the reference positions and orientations
     *
     * Passing an empty list to either argument causes that member to remain unchanged
     */
    void setReferences(const pybind11::list& r0, const pybind11::list& q0)
        {
        unsigned int ndim = m_sysdef->getNDimensions();
        std::vector<Scalar3> lattice_positions;
        std::vector<Scalar> pos_buffer;
        std::vector<Scalar4> lattice_orientations;
        std::vector<Scalar> ors_buffer;
#ifdef ENABLE_MPI
        unsigned int pos_size = 0, ors_size = 0;  // should this be a size_t?

        if (this->m_exec_conf->isRoot())
            {
            python_list_to_vector_scalar3(r0, lattice_positions, ndim);
            python_list_to_vector_scalar4(q0, lattice_orientations);
            pos_size = (unsigned int)lattice_positions.size();
            ors_size = (unsigned int)lattice_orientations.size();
            }
        if (this->m_pdata->getDomainDecomposition())
            {
            if (pos_size)
                {
                pos_buffer.resize(3 * pos_size, 0.0);
                for (size_t i = 0; i < pos_size; i++)
                    {
                    pos_buffer[3 * i] = lattice_positions[i].x;
                    pos_buffer[3 * i + 1] = lattice_positions[i].y;
                    pos_buffer[3 * i + 2] = lattice_positions[i].z;
                    }
                }
            if (ors_size)
                {
                ors_buffer.resize(4 * ors_size, 0.0);
                for (size_t i = 0; i < ors_size; i++)
                    {
                    ors_buffer[4 * i] = lattice_orientations[i].x;
                    ors_buffer[4 * i + 1] = lattice_orientations[i].y;
                    ors_buffer[4 * i + 2] = lattice_orientations[i].z;
                    ors_buffer[4 * i + 3] = lattice_orientations[i].w;
                    }
                }
            MPI_Bcast(&pos_size, 1, MPI_UNSIGNED, 0, m_exec_conf->getMPICommunicator());
            if (pos_size)
                {
                if (!pos_buffer.size())
                    pos_buffer.resize(3 * pos_size, 0.0);
                MPI_Bcast(&pos_buffer.front(),
                          3 * pos_size,
                          MPI_HOOMD_SCALAR,
                          0,
                          m_exec_conf->getMPICommunicator());
                if (!lattice_positions.size())
                    {
                    lattice_positions.resize(pos_size, make_scalar3(0.0, 0.0, 0.0));
                    for (size_t i = 0; i < pos_size; i++)
                        {
                        lattice_positions[i].x = pos_buffer[3 * i];
                        lattice_positions[i].y = pos_buffer[3 * i + 1];
                        lattice_positions[i].z = pos_buffer[3 * i + 2];
                        }
                    }
                }
            MPI_Bcast(&ors_size, 1, MPI_UNSIGNED, 0, m_exec_conf->getMPICommunicator());
            if (ors_size)
                {
                if (!ors_buffer.size())
                    ors_buffer.resize(4 * ors_size, 0.0);
                MPI_Bcast(&ors_buffer.front(),
                          4 * ors_size,
                          MPI_HOOMD_SCALAR,
                          0,
                          m_exec_conf->getMPICommunicator());
                if (!lattice_orientations.size())
                    {
                    lattice_orientations.resize(ors_size, make_scalar4(0, 0, 0, 0));
                    for (size_t i = 0; i < ors_size; i++)
                        {
                        lattice_orientations[i].x = ors_buffer[4 * i];
                        lattice_orientations[i].y = ors_buffer[4 * i + 1];
                        lattice_orientations[i].z = ors_buffer[4 * i + 2];
                        lattice_orientations[i].w = ors_buffer[4 * i + 3];
                        }
                    }
                }
            }

#else  // ENABLE_MPI == false
        python_list_to_vector_scalar3(r0, lattice_positions, ndim);
        python_list_to_vector_scalar4(q0, lattice_orientations);
#endif  // end if ENABLE_MPI

        if (lattice_positions.size())
            m_lattice_positions.setReferences(lattice_positions.begin(),
                                              lattice_positions.end(),
                                              m_pdata,
                                              m_exec_conf);

        if (lattice_orientations.size())
            m_lattice_orientations.setReferences(lattice_orientations.begin(),
                                                 lattice_orientations.end(),
                                                 m_pdata,
                                                 m_exec_conf);
        }  // end void setReferences

    void setKTranslational(Scalar k_translational)
        {
        m_k_translational = k_translational;
        }

    Scalar getKTranslational()
        {
        return m_k_translational;
        }

    void setKRotational(Scalar k_rotational)
        {
        m_k_rotational = k_rotational;
        }

    Scalar getKRotational()
        {
        return m_k_rotational;
        }

    Scalar getEnergy(uint64_t timestep)
        {
        compute(timestep);
        return m_energy;
        }

    protected:
    // These could be a little redundant. think about this more later.
    Scalar calcE_trans(const unsigned int& index, const vec3<Scalar>& position, const Scalar& scale = 1.0)
        {
        ArrayHandle<unsigned int> h_tags(m_pdata->getTags(),
                                         access_location::host,
                                         access_mode::read);
        int3 dummy = make_int3(0, 0, 0);
        vec3<Scalar> origin(m_pdata->getOrigin());
        const BoxDim& box = this->m_pdata->getGlobalBox();
        vec3<Scalar> r0(m_lattice_positions.getReference(h_tags.data[index]));
        r0 *= scale;
        Scalar3 t = vec_to_scalar3(position - origin);
        box.wrap(t, dummy);
        vec3<Scalar> shifted_pos(t);
        vec3<Scalar> dr = vec3<Scalar>(box.minImage(vec_to_scalar3(r0 - position + origin)));
        return m_k_translational * dot(dr, dr);
        }

    Scalar calcE_rot(const unsigned int& index, const quat<Scalar>& orientation)
        {
        assert(m_symmetry.size());
        ArrayHandle<unsigned int> h_tags(m_pdata->getTags(),
                                         access_location::host,
                                         access_mode::read);
        quat<Scalar> q0(m_lattice_orientations.getReference(h_tags.data[index]));
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
    Scalar calcE(const unsigned int& index,
                 const vec3<Scalar>& position,
                 const quat<Scalar>& orientation,
                 const Scalar& scale = 1.0)
        {
        Scalar energy = 0.0;
        if (m_lattice_positions.isValid())
            {
            energy += calcE_trans(index, position, scale);
            }
        if (m_lattice_orientations.isValid())
            {
            energy += calcE_rot(index, orientation);
            }
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
    std::vector<vec3<Scalar>> m_lattice_positions;     // reference positions
    std::vector<quat<Scalar>> m_lattice_orientations;  // reference orientations
    std::vector<quat<Scalar>> m_symmetry;              // symmetry-equivalent orientations
    Scalar m_k_translational;                          // translational spring constant
    Scalar m_k_rotational;                             // rotational spring constant
    Scalar m_energy;                                   // total energy from previous timestep
    };

namespace detail
    {
template<class Shape> void export_LatticeField(pybind11::module& m, std::string name)
    {
    pybind11::class_<ExternalFieldLattice<Shape>,
                     ExternalFieldMono<Shape>,
                     std::shared_ptr<ExternalFieldLattice<Shape>>>(m, name.c_str())
        .def(pybind11::init<std::shared_ptr<SystemDefinition>,
                            pybind11::list,
                            Scalar,
                            pybind11::list,
                            Scalar,
                            pybind11::list>())
        .def("setReferences", &ExternalFieldLattice<Shape>::setReferences)
        .def("setParams", &ExternalFieldLattice<Shape>::setParams)
        .def("reset", &ExternalFieldLattice<Shape>::reset)
        .def("clearPositions", &ExternalFieldLattice<Shape>::clearPositions)
        .def("clearOrientations", &ExternalFieldLattice<Shape>::clearOrientations)
        .def("getEnergy", &ExternalFieldLattice<Shape>::getEnergy)
        .def("getAvgEnergy", &ExternalFieldLattice<Shape>::getAvgEnergy)
        .def("getSigma", &ExternalFieldLattice<Shape>::getSigma);
    }

void export_LatticeFields(pybind11::module& m);

    } // end namespace detail
    } // namespace hpmc
    } // end namespace hoomd

#endif // _EXTERNAL_FIELD_LATTICE_H_
