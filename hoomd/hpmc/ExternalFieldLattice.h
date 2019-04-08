// Copyright (c) 2009-2019 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.

#ifndef _EXTERNAL_FIELD_LATTICE_H_
#define _EXTERNAL_FIELD_LATTICE_H_

/*! \file ExternalField.h
    \brief Declaration of ExternalField base class
*/

#include "hoomd/Compute.h"
#include "hoomd/Saru.h"
#include "hoomd/VectorMath.h"
#include "hoomd/HOOMDMPI.h"
#include "hoomd/ParticleGroup.h"

#include "ExternalField.h"

#ifndef NVCC
#include <hoomd/extern/pybind/include/pybind11/pybind11.h>
#endif

namespace hpmc
{
/*
For simplicity and consistency both the positional and orientational versions of
the external field will take in a list of either positions or orientations that
are the reference values. the i-th reference point will correspond to the particle
with tag i.
*/
inline void python_list_to_vector_scalar3(
        const pybind11::list& r0,
        std::vector<Scalar3>& ret,
        unsigned int ndim)
    {
    // validate input type and rank
    pybind11::ssize_t n = pybind11::len(r0);
    ret.resize(n);
    for ( pybind11::ssize_t i=0; i<n; i++)
        {
        // make sure dimensions of system and positions match
        pybind11::ssize_t d = pybind11::len(r0[i]);
        pybind11::list r0_tuple = pybind11::cast<pybind11::list >(r0[i]);
        if (d < ndim)
            {
            throw std::runtime_error(
                    "dimension of the list does not match the dimension of the simulation.");
            }

        Scalar x = pybind11::cast<Scalar>(r0_tuple[0]);
        Scalar y = pybind11::cast<Scalar>(r0_tuple[1]);
        Scalar z = 0.0;
        if (d == 3)
            {
            z = pybind11::cast<Scalar>(r0_tuple[2]);
            }
        ret[i] = make_scalar3(x, y, z);
        }
    }  // end python_list_to_vector_scalar3()

inline void python_list_to_vector_scalar4(const pybind11::list& r0, std::vector<Scalar4>& ret)
    {
    // validate input type and rank
    pybind11::ssize_t n = pybind11::len(r0);
    ret.resize(n);
    for (pybind11::ssize_t i=0; i<n; i++)
        {
        pybind11::list r0_tuple = pybind11::cast<pybind11::list >(r0[i]);
        ret[i] = make_scalar4(pybind11::cast<Scalar>(r0_tuple[0]),
                              pybind11::cast<Scalar>(r0_tuple[1]),
                              pybind11::cast<Scalar>(r0_tuple[2]),
                              pybind11::cast<Scalar>(r0_tuple[3]));
        }
    }


template< class ScalarType >
class LatticeReferenceList
    {
    public:
        /* Default constructor
         *
         **/
        LatticeReferenceList() : m_N(0) {}

        /*  Constructor that sets the reference positions
         *
         **/
        template<class InputIterator >
        LatticeReferenceList(
                InputIterator first,
                InputIterator last,
                const std::shared_ptr<ParticleData> pdata,
                std::shared_ptr<const ExecutionConfiguration> exec_conf)
            {
            initialize(first, last, pdata, exec_conf);
            }

        /*  Default destructor
         *
         **/
        ~LatticeReferenceList() {}

        template <class InputIterator>
        void initialize(InputIterator first,
                InputIterator last,
                const std::shared_ptr<ParticleData> pdata,
                std::shared_ptr<const ExecutionConfiguration> exec_conf)
            {
            m_N = std::distance(first, last);
            if (m_N > 0)
                {
                setReferences(first, last, pdata, exec_conf);
                }
            }

        const ScalarType& getReference(const unsigned int& tag)
            {
            ArrayHandle<ScalarType> h_ref(m_reference, access_location::host, access_mode::read);
            return h_ref.data[tag];
            }

        const GPUArray< ScalarType >& getReferenceArray()
            {
            return m_reference;
            }

        template <class InputIterator>
        void setReferences(
                InputIterator first,
                InputIterator last,
                const std::shared_ptr<ParticleData> pdata,
                std::shared_ptr<const ExecutionConfiguration> exec_conf)
            {
            size_t numPoints = std::distance(first, last);

            // early exit if reference list is 0 elements in length
            if (!numPoints)
                {
                clear();
                return;
                }

            if (!exec_conf || !pdata || pdata->getNGlobal() != numPoints)
                {
                if (exec_conf)
                    {
                    exec_conf->msg->error()
                    << "Check pointers and initialization list"
                    << std::endl;
                    }
                throw std::runtime_error("Error setting LatticeReferenceList");
                }

            m_N = numPoints;
            GPUArray<ScalarType> temp(numPoints, exec_conf);
            { // scope the copy
            ArrayHandle<ScalarType> h_temp(temp, access_location::host, access_mode::overwrite);
            // now copy and swap the data.
            std::copy(first, last, h_temp.data);
            }
            m_reference.swap(temp);
            }  // end LatticeReferenceList::setReferences()

        void scale(const Scalar& s)
            {
            ArrayHandle<ScalarType> h_ref(
                    m_reference, access_location::host, access_mode::readwrite);
            for(unsigned int i = 0; i < m_N; i++)
                {
                h_ref.data[i].x *= s;
                h_ref.data[i].y *= s;
                h_ref.data[i].z *= s;
                }
            }

        void clear()
            {
            m_N = 0;
            GPUArray<ScalarType> nullArray;
            m_reference.swap(nullArray);
            }

        bool isValid()
            {
            return m_N != 0 && !m_reference.isNull();
            }

    private:
        GPUArray<ScalarType> m_reference;
        unsigned int         m_N;  /// number of particles in the system (change for groups)
    };  // end class LatticeReferenceList


#define LATTICE_ENERGY_LOG_NAME                 "lattice_energy"
#define LATTICE_ENERGY_AVG_LOG_NAME             "lattice_energy_pp_avg"
#define LATTICE_ENERGY_SIGMA_LOG_NAME           "lattice_energy_pp_sigma"
#define LATTICE_TRANS_SPRING_CONSTANT_LOG_NAME  "lattice_translational_spring_constant"
#define LATTICE_ROTAT_SPRING_CONSTANT_LOG_NAME  "lattice_rotational_spring_constant"
#define LATTICE_NUM_SAMPLES_LOG_NAME            "lattice_num_samples"

template< class Shape>
class ExternalFieldLattice : public ExternalFieldMono<Shape>
    {
    using ExternalFieldMono<Shape>::m_pdata;
    using ExternalFieldMono<Shape>::m_exec_conf;
    using ExternalFieldMono<Shape>::m_sysdef;
    public:
        ExternalFieldLattice(std::shared_ptr<SystemDefinition> sysdef,
                             std::shared_ptr<ParticleGroup> group,
                             pybind11::list r0,
                             Scalar k,
                             pybind11::list q0,
                             Scalar q,
                             pybind11::list symRotations)
                             : ExternalFieldMono<Shape>(sysdef),
                               m_k(k),
                               m_q(q),
                               m_Energy(0),
                               m_group(group)
            {
            // add to provided quantities
            m_ProvidedQuantities.push_back(LATTICE_ENERGY_LOG_NAME);
            m_ProvidedQuantities.push_back(LATTICE_ENERGY_AVG_LOG_NAME);
            m_ProvidedQuantities.push_back(LATTICE_ENERGY_SIGMA_LOG_NAME);
            m_ProvidedQuantities.push_back(LATTICE_TRANS_SPRING_CONSTANT_LOG_NAME);
            m_ProvidedQuantities.push_back(LATTICE_ROTAT_SPRING_CONSTANT_LOG_NAME);
            m_ProvidedQuantities.push_back(LATTICE_NUM_SAMPLES_LOG_NAME);

            // Connect to the BoxChange signal
            m_box = m_pdata->getBox();
            m_pdata->getBoxChangeSignal().template connect<
                ExternalFieldLattice<Shape>,
                &ExternalFieldLattice<Shape>::scaleReferencePoints>(this);
            setReferences(r0, q0);

            // build up list of equivalent orientations
            std::vector<Scalar4> rots;
            python_list_to_vector_scalar4(symRotations, rots);
            bool identityFound = false;
            quat<Scalar> identity(1, vec3<Scalar>(0, 0, 0));
            Scalar tol = 1e-5;
            for(size_t i = 0; i < rots.size(); i++)
                {
                quat<Scalar> qi(rots[i]);
                identityFound = !identityFound ? norm2(qi-identity) < tol : identityFound;
                m_symmetry.push_back(qi);
                }
            if (!identityFound)  // ensure that the identity rotation is provided
                {
                m_symmetry.push_back(identity);
                }
            reset(0);  // initializes all of the energy logging parameters
            }  // end ExternalFieldLattice::ExternalFieldLattice()

        ~ExternalFieldLattice()
            {
            // Disconnect from the BoxChange signal
            m_pdata->getBoxChangeSignal().template disconnect<
                ExternalFieldLattice<Shape>,
                &ExternalFieldLattice<Shape>::scaleReferencePoints>(this);
            }

        // why is this zero? this feels wrong, esp. considering we have an energy
        // associated with configurations with this external field
        Scalar calculateBoltzmannWeight(unsigned int timestep)
            {
            return 0.0;
            }

        //! Calculate energy difference between new and old configurations
        double calculateDeltaE(const Scalar4 * const position_old_arg,
                               const Scalar4 * const orientation_old_arg,
                               const BoxDim * const box_old_arg
                               )
            {
            // TODO: rethink the formatting a bit
            ArrayHandle<Scalar4> h_pos(
                    m_pdata->getPositions(), access_location::host, access_mode::readwrite);
            ArrayHandle<Scalar4> h_orient(
                    m_pdata->getOrientationArray(), access_location::host, access_mode::readwrite);
            const Scalar4 * const position_new = h_pos.data;
            const Scalar4 * const orientation_new = h_orient.data;
            const BoxDim * const box_new = &m_pdata->getGlobalBox();

            // copy arguments into new arrays
            const Scalar4 * position_old = position_old_arg;
            const Scalar4 * orientation_old = orientation_old_arg;
            const BoxDim * box_old = box_old_arg;

            // if arrays are empty, fill with "new" ones
            if (!position_old)
                position_old = position_new;
            if (!orientation_old)
                orientation_old = orientation_new;
            if (!box_old)
                box_old = box_new;

            Scalar curVolume = m_box.getVolume();
            Scalar newVolume = box_new->getVolume();
            Scalar oldVolume = box_old->getVolume();
            Scalar scaleOld = pow((oldVolume/curVolume), Scalar(1.0/3.0));
            Scalar scaleNew = pow((newVolume/curVolume), Scalar(1.0/3.0));

            double dE = 0.0;
            // TODO: only loop over particles in the group
            for(size_t i = 0; i < m_pdata->getN(); i++)
                {
                if (!m_group->isMember(i))
                    continue;

                Scalar old_E = calcE(
                        i,
                        vec3<Scalar>(*(position_old+i)),
                        quat<Scalar>(*(orientation_old+i)),
                        scaleOld);
                Scalar new_E = calcE(
                        i,
                        vec3<Scalar>(*(position_new+i)),
                        quat<Scalar>(*(orientation_new+i)),
                        scaleNew);
                dE += new_E - old_E;
                }

            #ifdef ENABLE_MPI
            if (this->m_pdata->getDomainDecomposition())
                {
                MPI_Allreduce(
                        MPI_IN_PLACE,
                        &dE,
                        1,
                        MPI_HOOMD_SCALAR,
                        MPI_SUM,
                        m_exec_conf->getMPICommunicator());
                }
            #endif

            return dE;
            }  // end ExternalFieldLattice::calculateDeltaE()

        void compute(unsigned int timestep)
            {
            // early exit if we don't need to calculate this at this timestep
            if (!this->shouldCompute(timestep))
                {
                return;
                }

            m_Energy = Scalar(0.0);
            // access particle data and system box
            ArrayHandle<Scalar4> h_postype(
                    m_pdata->getPositions(), access_location::host, access_mode::read);
            ArrayHandle<Scalar4> h_orient(
                    m_pdata->getOrientationArray(), access_location::host, access_mode::read);
            // TODO: only loop over particles in group
            for(size_t i = 0; i < m_pdata->getN(); i++)
                {
                if (!m_group->isMember(i))
                    continue;

                vec3<Scalar> position(h_postype.data[i]);
                quat<Scalar> orientation(h_orient.data[i]);
                m_Energy += calcE(i, position, orientation);
                }

            #ifdef ENABLE_MPI
            if (this->m_pdata->getDomainDecomposition())
                {
                MPI_Allreduce(
                        MPI_IN_PLACE,
                        &m_Energy,
                        1,
                        MPI_HOOMD_SCALAR,
                        MPI_SUM,
                        m_exec_conf->getMPICommunicator());
                }
            #endif

            // Kahan/compensation summation of energy per particle and (energy per particle)**2
            // See https://en.wikipedia.org/wiki/Kahan_summation_algorithm
            // energy_per <--> input[i]
            Scalar energy_per = m_Energy / Scalar(m_group->getNumMembersGlobal());
            m_EnergySum_y    = energy_per - m_EnergySum_c;
            m_EnergySum_t    = m_EnergySum + m_EnergySum_y;
            m_EnergySum_c    = (m_EnergySum_t - m_EnergySum) - m_EnergySum_y;
            m_EnergySum      = m_EnergySum_t;

            Scalar energy_sq_per = energy_per*energy_per;
            m_EnergySqSum_y    = energy_sq_per - m_EnergySqSum_c;
            m_EnergySqSum_t    = m_EnergySqSum + m_EnergySqSum_y;
            m_EnergySqSum_c    = (m_EnergySqSum_t-m_EnergySqSum) - m_EnergySqSum_y;
            m_EnergySqSum      = m_EnergySqSum_t;
            m_num_samples++;
            }  // end ExternalFieldLattice::compute()

        double energydiff(
                const unsigned int& index,
                const vec3<Scalar>& position_old,
                const Shape& shape_old,
                const vec3<Scalar>& position_new,
                const Shape& shape_new)
            {
            if (!m_group->isMember(index))
                return 0;

            double old_U = calcE(index, position_old, shape_old);
            double new_U = calcE(index, position_new, shape_new);
            return new_U - old_U;
            }  // end ExternalFieldLattice::energydiff()

        void setReferences(const pybind11::list& r0, const pybind11::list& q0)
            {
            // initialize some arrays to hold things
            unsigned int ndim = m_sysdef->getNDimensions();
            std::vector<Scalar3> lattice_positions;
            std::vector<Scalar> pbuffer;
            std::vector<Scalar4> lattice_orientations;
            std::vector<Scalar> qbuffer;
            #ifdef ENABLE_MPI
            unsigned int psz = 0, qsz = 0;  // length of particle and orientation lists
            if (this->m_exec_conf->isRoot())
                {
                python_list_to_vector_scalar3(r0, lattice_positions, ndim);
                python_list_to_vector_scalar4(q0, lattice_orientations);
                psz = lattice_positions.size();
                qsz = lattice_orientations.size();
                }
            if (this->m_pdata->getDomainDecomposition())
                {
                // fill in the position and orientation buffers
                if (psz)
                    {
                    pbuffer.resize(3*psz, 0.0);
                    for(size_t i = 0; i < psz; i++)
                        {
                        pbuffer[3*i] = lattice_positions[i].x;
                        pbuffer[3*i+1] = lattice_positions[i].y;
                        pbuffer[3*i+2] = lattice_positions[i].z;
                        }
                    }
                if (qsz)
                    {
                    qbuffer.resize(4*qsz, 0.0);
                    for(size_t i = 0; i < qsz; i++)
                        {
                        qbuffer[4*i] = lattice_orientations[i].x;
                        qbuffer[4*i+1] = lattice_orientations[i].y;
                        qbuffer[4*i+2] = lattice_orientations[i].z;
                        qbuffer[4*i+3] = lattice_orientations[i].w;
                        }
                    }

                // broadcast particle array size to all ranks
                MPI_Bcast(&psz, 1, MPI_UNSIGNED, 0, m_exec_conf->getMPICommunicator());
                if (psz)
                    {
                    if (!pbuffer.size())
                        pbuffer.resize(3*psz, 0.0);
                    // broadcast particle position array to all ranks
                    MPI_Bcast(
                            &pbuffer.front(),
                            3*psz,
                            MPI_HOOMD_SCALAR,
                            0,
                            m_exec_conf->getMPICommunicator());
                    if (!lattice_positions.size())
                        {
                        lattice_positions.resize(psz, make_scalar3(0.0, 0.0, 0.0));
                        for(size_t i = 0; i < psz; i++)
                            {
                            lattice_positions[i].x = pbuffer[3*i];
                            lattice_positions[i].y = pbuffer[3*i+1];
                            lattice_positions[i].z = pbuffer[3*i+2];
                            }
                        }
                    }  // end if (psz)

                // broadcast particle orientation array size to all ranks
                MPI_Bcast(&qsz, 1, MPI_UNSIGNED, 0, m_exec_conf->getMPICommunicator());
                if (qsz)
                    {
                    if (!qbuffer.size())
                        qbuffer.resize(4*qsz, 0.0);
                    // broadcast orientation array to all ranks
                    MPI_Bcast(&qbuffer.front(),
                            4*qsz,
                            MPI_HOOMD_SCALAR,
                            0,
                            m_exec_conf->getMPICommunicator());
                    if (!lattice_orientations.size())
                        {
                        lattice_orientations.resize(qsz, make_scalar4(0, 0, 0, 0));
                        for(size_t i = 0; i < qsz; i++)
                            {
                            lattice_orientations[i].x = qbuffer[4*i];
                            lattice_orientations[i].y = qbuffer[4*i+1];
                            lattice_orientations[i].z = qbuffer[4*i+2];
                            lattice_orientations[i].w = qbuffer[4*i+3];
                            }
                        }
                    }
                }  // end if (this->m_pdata->getDomainDecomposition())

            #else
            // if not using mpi, can just use these functions to set reference
            // positions/orientations
            python_list_to_vector_scalar3(r0, lattice_positions, ndim);
            python_list_to_vector_scalar4(q0, lattice_orientations);
            #endif

            // set reference positions for the lattice position/orientations
            if (lattice_positions.size())
                {
                m_latticePositions.setReferences(
                        lattice_positions.begin(),
                        lattice_positions.end(),
                        m_pdata,
                        m_exec_conf);
                }

            if (lattice_orientations.size())
                {
                m_latticeOrientations.setReferences(
                        lattice_orientations.begin(),
                        lattice_orientations.end(),
                        m_pdata,
                        m_exec_conf);
                }
            }  // end ExternalFieldLattice::setReferences()

        void clearPositions() { m_latticePositions.clear(); }

        void clearOrientations() { m_latticeOrientations.clear(); }

        void scaleReferencePoints()
            {
            BoxDim newBox = m_pdata->getBox();
            Scalar newVol = newBox.getVolume();
            Scalar lastVol = m_box.getVolume();
            Scalar scale;
            Scalar ndim = Scalar(this->m_sysdef->getNDimensions());
            scale = pow((newVol/lastVol), Scalar(1.0/ndim));
            m_latticePositions.scale(scale);
            m_box = newBox;
            }

        //! Returns a list of log quantities this compute calculates
        std::vector< std::string > getProvidedLogQuantities()
            {
            return m_ProvidedQuantities;
            }

        //! Calculates the requested log value and returns it
        Scalar getLogValue(const std::string& quantity, unsigned int timestep)
            {
            compute(timestep);

            if (quantity == LATTICE_ENERGY_LOG_NAME)
                {
                return m_Energy;
                }
            else if (quantity == LATTICE_ENERGY_AVG_LOG_NAME)
                {
                return getAvgEnergy(timestep);
                }
            else if (quantity == LATTICE_ENERGY_SIGMA_LOG_NAME)
                {
                return getSigma(timestep);
                }
            else if (quantity == LATTICE_TRANS_SPRING_CONSTANT_LOG_NAME)
                {
                return m_k;
                }
            else if (quantity == LATTICE_ROTAT_SPRING_CONSTANT_LOG_NAME)
                {
                return m_q;
                }
            else if (quantity == LATTICE_NUM_SAMPLES_LOG_NAME)
                {
                return m_num_samples;
                }
            else
                {
                m_exec_conf->msg->error()
                    << "field.lattice_field: "
                    << quantity
                    << " is not a valid log quantity"
                    << std::endl;
                throw std::runtime_error("Error getting log value");
                }
            }  // end ExternalFieldLattice::getLogValue()

        void setParams(Scalar k, Scalar q)
            {
            m_k = k;
            m_q = q;
            }

        const GPUArray< Scalar3 >& getReferenceLatticePositions()
            {
            return m_latticePositions.getReferenceArray();
            }

        const GPUArray< Scalar4 >& getReferenceLatticeOrientations()
            {
            return m_latticeOrientations.getReferenceArray();
            }

        void reset(unsigned int)  // TODO: remove the timestep
            {
            m_EnergySum = m_EnergySum_y = m_EnergySum_t = m_EnergySum_c = Scalar(0.0);
            m_EnergySqSum = m_EnergySqSum_y = m_EnergySqSum_t = m_EnergySqSum_c = Scalar(0.0);
            m_num_samples = 0;
            }

        Scalar getEnergy(unsigned int timestep)
            {
            compute(timestep);
            return m_Energy;
            }

        //! Energy per particle averaged over the number of computes()s called
        Scalar getAvgEnergy(unsigned int timestep)
            {
            compute(timestep);
            if (!m_num_samples)
                return 0.0;
            return m_EnergySum / double(m_num_samples);
            }

        //! Standard deviation of the energy per particle over the number of times compute() is
        // called
        Scalar getSigma(unsigned int timestep)
            {
            compute(timestep);
            if (!m_num_samples)
                return 0.0;
            Scalar first_moment = m_EnergySum / double(m_num_samples);
            Scalar second_moment = m_EnergySqSum / double(m_num_samples);
            return sqrt(second_moment - (first_moment*first_moment));
            }

    protected:
        // These could be a little redundant. think about this more later.
        // They definetely _feel_ redundant
        Scalar calcE_trans(
                const unsigned int& index,
                const vec3<Scalar>& position,
                const Scalar& scale = 1.0)
            {
            if (!m_group->isMember(index))
                return 0.0;
            ArrayHandle<unsigned int> h_tags(
                    m_pdata->getTags(), access_location::host, access_mode::read);
            int3 dummy = make_int3(0,0,0);
            vec3<Scalar> origin(m_pdata->getOrigin());
            const BoxDim& box = this->m_pdata->getGlobalBox();
            vec3<Scalar> r0(m_latticePositions.getReference(h_tags.data[index]));
            r0 *= scale;
            Scalar3 t = vec_to_scalar3(position - origin);
            box.wrap(t, dummy);
            vec3<Scalar> shifted_pos(t);
            vec3<Scalar> dr = vec3<Scalar>(box.minImage(vec_to_scalar3(r0 - position + origin)));
            return m_k*dot(dr,dr);
            }

        Scalar calcE_rot(const unsigned int& index, const quat<Scalar>& orientation)
            {
            if (!m_group->isMember(index))
                return 0.0;
            assert(m_symmetry.size());
            ArrayHandle<unsigned int> h_tags(
                    m_pdata->getTags(), access_location::host, access_mode::read);
            quat<Scalar> q0(m_latticeOrientations.getReference(h_tags.data[index]));
            Scalar dqmin = 0.0;
            for(size_t i = 0; i < m_symmetry.size(); i++)
                {
                quat<Scalar> equiv_orientation = orientation*m_symmetry[i];
                quat<Scalar> dq = q0 - equiv_orientation;
                dqmin = (i == 0) ? norm2(dq) : fmin(dqmin, norm2(dq));
                }
            return m_q*dqmin;
            }
        Scalar calcE_rot(const unsigned int& index, const Shape& shape)
            {
            if (!m_group->isMember(index))
                return 0.0;
            if (!shape.hasOrientation())
                return Scalar(0.0);

            return calcE_rot(index, shape.orientation);
            }

        //! Calculate the energy associated with the configuration
        Scalar calcE(
                const unsigned int& index,
                const vec3<Scalar>& position,
                const quat<Scalar>& orientation,
                const Scalar& scale = 1.0)
            {
            // exit early if particle not in group
            if (!m_group->isMember(index))
                return 0.0;

            Scalar energy = 0.0;
            if (m_latticePositions.isValid())
                {
                energy += calcE_trans(index, position, scale);
                }
            if (m_latticeOrientations.isValid())
                {
                energy += calcE_rot(index, orientation);
                }
            return energy;
            }  // end ExternalFieldLattice::calcE()

        Scalar calcE(const unsigned int& index,
                const vec3<Scalar>& position,
                const Shape& shape,
                const Scalar& scale = 1.0)
            {
            return calcE(index, position, shape.orientation, scale);
            }

    private:
        Scalar  m_k;        // translational spring constant
        Scalar  m_q;        // rotational spring constant
        Scalar  m_Energy;   // total energy of the last computed timestep

        // group to apply external field to
        std::shared_ptr<ParticleGroup> m_group;

        // positions of the lattice
        LatticeReferenceList<Scalar3> m_latticePositions;

        // orientation of the lattice particles
        LatticeReferenceList<Scalar4> m_latticeOrientations;

        // quaternions in the symmetry group of the shape
        std::vector< quat<Scalar> > m_symmetry;

        // Terms for Kahan summation of energy; all are on a per-particle basis
        Scalar  m_EnergySum;
        Scalar  m_EnergySum_y;
        Scalar  m_EnergySum_t;
        Scalar  m_EnergySum_c;
        Scalar  m_EnergySqSum;
        Scalar  m_EnergySqSum_y;
        Scalar  m_EnergySqSum_t;
        Scalar  m_EnergySqSum_c;

        // other stuff
        unsigned int              m_num_samples;   // no. times compute() has been called
        std::vector<std::string>  m_ProvidedQuantities;
        BoxDim                    m_box;
    };  // end class ExternalFieldLattice

template<class Shape>
void export_LatticeField(pybind11::module& m, std::string name)
    {
   pybind11::class_<
       ExternalFieldLattice<Shape>,
       std::shared_ptr< ExternalFieldLattice<Shape> > >(
               m,
               name.c_str(),
               pybind11::base< ExternalFieldMono<Shape> >())
    .def(pybind11::init< std::shared_ptr<SystemDefinition>,
         std::shared_ptr<ParticleGroup>,
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
    .def("getSigma", &ExternalFieldLattice<Shape>::getSigma)
    ;
    }

void export_LatticeFields(pybind11::module& m);

} // namespace hpmc

#endif // _EXTERNAL_FIELD_LATTICE_H_
