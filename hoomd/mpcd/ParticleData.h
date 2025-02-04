// Copyright (c) 2009-2024 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

#ifndef MPCD_PARTICLE_DATA_H_
#define MPCD_PARTICLE_DATA_H_

/*!
 * \file mpcd/ParticleData.h
 * \brief Declaration of mpcd::ParticleData
 */

#ifdef __HIPCC__
#error This header cannot be compiled by nvcc
#endif

#include "ParticleDataSnapshot.h"
#include "ParticleDataUtilities.h"

#ifdef ENABLE_HIP
#include "ParticleData.cuh"
#ifdef ENABLE_MPI
#include "hoomd/Autotuner.h"
#endif // ENABLE_MPI
#endif // ENABLE_HIP

#include "hoomd/Autotuned.h"
#include "hoomd/BoxDim.h"
#include "hoomd/DomainDecomposition.h"
#include "hoomd/ExecutionConfiguration.h"
#include "hoomd/GPUArray.h"
#include "hoomd/GPUFlags.h"
#include "hoomd/GPUVector.h"

#include "hoomd/extern/nano-signal-slot/nano_signal_slot.hpp"

// pybind11
#include <pybind11/pybind11.h>

namespace hoomd
    {
namespace mpcd
    {
//! Stores MPCD particle data
/*!
 * MPCD particles are characterized by position, velocity, and mass. We assume all
 * particles have the same mass. The data is laid out as follows:
 * - position + type in array of Scalar4
 * - velocity + cell index in array of Scalar4
 * - tag in array of unsigned int
 *
 * Unlike the standard ParticleData, a reverse tag mapping is not currently maintained
 * in order to save local memory. (That is, it is possible to read the tag of a local particle,
 * but it is not possible to efficiently find the local particle that has a given
 * tag.)
 *
 * The MPCD cell index is stored with the velocity because most MPCD operations
 * are based on around the velocity and cell. For details of what the cell means,
 * refer to the mpcd::CellList.
 *
 * \todo Because the local cell index changes with position, a signal will be put
 * in place to indicate when the cached cell index is still valid.
 *
 * \todo Likewise, MPCD benefits from sorting data into cell order, so a signal
 * needs to be put in place when the ordering changes.
 *
 * \todo Likewise, a signal should be incorporated to indicate when particles are
 * added or removed locally, as is the case during particle migration.
 *
 * \ingroup data_structs
 */
class PYBIND11_EXPORT ParticleData : public Autotuned
    {
    public:
    //! Number constructor
    ParticleData(unsigned int N,
                 std::shared_ptr<const BoxDim> local_box,
                 Scalar kT,
                 unsigned int seed,
                 unsigned int ndimensions,
                 std::shared_ptr<ExecutionConfiguration> exec_conf,
                 std::shared_ptr<DomainDecomposition> decomposition
                 = std::shared_ptr<DomainDecomposition>());

    //! Snapshot constructor
    ParticleData(const mpcd::ParticleDataSnapshot& snapshot,
                 std::shared_ptr<const BoxDim> global_box,
                 std::shared_ptr<const ExecutionConfiguration> exec_conf,
                 std::shared_ptr<DomainDecomposition> decomposition
                 = std::shared_ptr<DomainDecomposition>());

    //! Destructor
    ~ParticleData();

    //! Initialize the MPCD particle data from a snapshot
    void initializeFromSnapshot(const mpcd::ParticleDataSnapshot& snapshot,
                                std::shared_ptr<const BoxDim> global_box);

    //! Default initialize the MPCD particle data per rank
    void initializeRandom(unsigned int N,
                          std::shared_ptr<const BoxDim> local_box,
                          Scalar kT,
                          unsigned int seed,
                          unsigned int ndimensions);

    //! Take a snapshot of the MPCD particle data
    void takeSnapshot(mpcd::ParticleDataSnapshot& snapshot,
                      std::shared_ptr<const BoxDim> global_box) const;

    //! \name accessor methods
    //@{
    //! Get number of MPCD particles on the rank
    unsigned int getN() const
        {
        return m_N;
        }

    //! Get the number of MPCD virtual particles on this rank
    unsigned int getNVirtual() const
        {
        return m_N_virtual;
        }

    //! Get global number of MPCD particles
    unsigned int getNGlobal() const
        {
        return m_N_global;
        }

    //! Get the global number of virtual MPCD particles
    /*!
     * This method requires a collective reduction in MPI simulations. The caller is responsible for
     * caching the returned value for performance if necessary.
     */
    unsigned int getNVirtualGlobal() const
        {
#ifdef ENABLE_MPI
        if (m_exec_conf->getNRanks() > 1)
            {
            unsigned int N_virtual_global = m_N_virtual;
            MPI_Allreduce(MPI_IN_PLACE,
                          &N_virtual_global,
                          1,
                          MPI_UNSIGNED,
                          MPI_SUM,
                          m_exec_conf->getMPICommunicator());
            return N_virtual_global;
            }
        else
#endif // ENABLE_MPI
            {
            return m_N_virtual;
            }
        }

    //! Get number of MPCD particle types
    unsigned int getNTypes() const
        {
        return (unsigned int)m_type_mapping.size();
        }

    //! Get the type-name mapping
    const std::vector<std::string>& getTypeNames() const
        {
        return m_type_mapping;
        }

    //! Get the type index by its name
    unsigned int getTypeByName(const std::string& name) const;

    //! Get the name of a type by its index
    std::string getNameByType(unsigned int type) const;

    //! Get array of MPCD particle positions
    const GPUArray<Scalar4>& getPositions() const
        {
        return m_pos;
        }

    //! Get array of MPCD particle velocities
    const GPUArray<Scalar4>& getVelocities() const
        {
        return m_vel;
        }

    //! Get array of MPCD particle tags
    const GPUArray<unsigned int>& getTags() const
        {
        return m_tag;
        }

    //! Get particle mass
    Scalar getMass() const
        {
        return m_mass;
        }

    //! Set particle mass
    void setMass(Scalar mass);

    //! Get the position of the particle on the local rank
    Scalar3 getPosition(unsigned int idx) const;

    //! Get the type of the particle on the local rank
    unsigned int getType(unsigned int idx) const;

    //! Get the velocity of the particle on the local rank
    Scalar3 getVelocity(unsigned int idx) const;

    //! Get the tag of the particle on the local rank
    unsigned int getTag(unsigned int idx) const;

    //@}

    //! \name swap methods
    //@{
    //! Get alternate array of MPCD particle positions
    const GPUArray<Scalar4>& getAltPositions() const
        {
        return m_pos_alt;
        }

    //! Swap out alternate MPCD particle position array
    void swapPositions()
        {
        m_pos.swap(m_pos_alt);
        }

    //! Get alternate array of MPCD particle velocities
    const GPUArray<Scalar4>& getAltVelocities() const
        {
        return m_vel_alt;
        }

    //! Swap out alternate MPCD particle velocity array
    void swapVelocities()
        {
        m_vel.swap(m_vel_alt);
        }

    //! Get alternate array of MPCD particle tags
    const GPUArray<unsigned int>& getAltTags() const
        {
        return m_tag_alt;
        }

    //! Swap out alternate MPCD particle tags
    void swapTags()
        {
        m_tag.swap(m_tag_alt);
        }
    //@}

    //! \name signal methods
    //@{
    //! Mark the cell value cached in the last element of the velocity as valid
    void validateCellCache()
        {
        m_valid_cell_cache = true;
        }
    //! Mark the cell value cached in the last element of the velocity as invalid
    void invalidateCellCache()
        {
        m_valid_cell_cache = false;
        }
    //! Check if the cell value cached in the last element of the velocity is valid
    /*!
     * \returns True if the cache is valid, false otherwise
     */
    bool checkCellCache() const
        {
        return m_valid_cell_cache;
        }

    //! Signature for particle sort signal
    typedef Nano::Signal<
        void(uint64_t timestep, const GPUArray<unsigned int>&, const GPUArray<unsigned int>&)>
        SortSignal;

    //! Get the sort signal
    /*!
     * \returns A sort signal that subscribers can attach a callback to for sorting
     *          their own per-particle data.
     */
    SortSignal& getSortSignal()
        {
        return m_sort_signal;
        }

    //! Notify subscribers of a particle sort
    /*!
     * \param timestep Timestep that the sorting occurred
     * \param order Mapping of sorted particle indexes onto old particle indexes
     * \param rorder Mapping of old particle indexes onto sorted particle indexes
     *
     * This method notifies the subscribers of the sort occurring at \a timestep.
     * Subscribers may choose to use \a order and \a rorder to reorder their
     * per-particle data immediately, or delay the sort until their next call.
     */
    void notifySort(uint64_t timestep,
                    const GPUArray<unsigned int>& order,
                    const GPUArray<unsigned int>& rorder)
        {
        m_sort_signal.emit(timestep, order, rorder);
        }

    //! Notify subscribers of a particle sort
    /*!
     * \param timestep Timestep that the sorting occurred
     *
     * This method notifies the subscribers of the sort occurring at \a timestep.
     * Subscribers are not given the updated particle order.
     */
    void notifySort(uint64_t timestep)
        {
        GPUArray<unsigned int> order, rorder;
        m_sort_signal.emit(timestep, order, rorder);
        }
    //@}

    //! \name virtual particle methods
    //@{
    //! Get the signal for the number of virtual particles changing
    /*!
     * \returns A signal that notifies subscribers when the number of virtual
     *          particles changes. This includes addition and removal of particles.
     */
    Nano::Signal<void()>& getNumVirtualSignal()
        {
        return m_virtual_signal;
        }

    //! Notify subscribers that the number of virtual particles has changed
    void notifyNumVirtual()
        {
        m_virtual_signal.emit();
        }

    //! Allocate memory for virtual particles
    unsigned int addVirtualParticles(unsigned int N);

    //! Remove all virtual particles
    /*!
     * \post The virtual particle counter is reset to zero.
     *
     * The memory associated with the previous virtual particle allocation is not freed
     * since the array growth is amortized in allocateVirtualParticles.
     */
    void removeVirtualParticles()
        {
        const unsigned int old_N_virtual = m_N_virtual;
        m_N_virtual = 0;

        // only notify of a change if there were virtual particles that have now been removed
        if (old_N_virtual != 0)
            notifyNumVirtual();
        }
    //@}

#ifdef ENABLE_MPI
    //! \name communication methods
    //@{

    //! Pack particle data into a buffer
    void removeParticles(GPUVector<mpcd::detail::pdata_element>& out,
                         unsigned int mask,
                         uint64_t timestep);

    //! Add new local particles
    void addParticles(const GPUVector<mpcd::detail::pdata_element>& in,
                      unsigned int mask,
                      uint64_t timestep);

#ifdef ENABLE_HIP
    //! Pack particle data into a buffer (GPU version)
    void removeParticlesGPU(GPUVector<mpcd::detail::pdata_element>& out,
                            unsigned int mask,
                            uint64_t timestep);

    //! Add new local particles (GPU version)
    void addParticlesGPU(const GPUVector<mpcd::detail::pdata_element>& in,
                         unsigned int mask,
                         uint64_t timestep);
#endif // ENABLE_HIP

    //! Get the MPCD particle communication flags
    const GPUArray<unsigned int>& getCommFlags() const
        {
        return m_comm_flags;
        }

    //! Get the alternate MPCD particle communication flags
    const GPUArray<unsigned int>& getAltCommFlags() const
        {
        return m_comm_flags_alt;
        }

    //! Swap out alternate MPCD communication flags
    void swapCommFlags()
        {
        m_comm_flags.swap(m_comm_flags_alt);
        }

//@}
#endif // ENABLE_MPI

    private:
    unsigned int m_N;         //!< Number of MPCD particles
    unsigned int m_N_virtual; //!< Number of virtual MPCD particles
    unsigned int m_N_global;  //!< Total number of MPCD particles
    unsigned int m_N_max;     //!< Maximum number of MPCD particles arrays can hold

    std::shared_ptr<const ExecutionConfiguration> m_exec_conf; //!< GPU execution configuration
    std::shared_ptr<DomainDecomposition> m_decomposition;      //!< Domain decomposition

    GPUArray<Scalar4> m_pos;                 //!< MPCD particle positions plus type
    GPUArray<Scalar4> m_vel;                 //!< MPCD particle velocities plus cell list id
    Scalar m_mass;                           //!< MPCD particle mass
    GPUArray<unsigned int> m_tag;            //!< MPCD particle tags
    std::vector<std::string> m_type_mapping; //!< Type name mapping
#ifdef ENABLE_MPI
    GPUArray<unsigned int> m_comm_flags; //!< MPCD particle communication flags
#endif                                   // ENABLE_MPI

    GPUArray<Scalar4> m_pos_alt;      //!< Alternate position array
    GPUArray<Scalar4> m_vel_alt;      //!< Alternate velocity array
    GPUArray<unsigned int> m_tag_alt; //!< Alternate tag array
#ifdef ENABLE_MPI
    GPUArray<unsigned int> m_comm_flags_alt; //!< Alternate communication flags
    GPUArray<unsigned int> m_remove_ids;     //!< Partitioned indexes of particles to keep
#ifdef ENABLE_HIP
    GPUArray<unsigned char> m_remove_flags; //!< Temporary flag to mark keeping particle
    GPUFlags<unsigned int> m_num_remove;    //!< Number of particles to remove

    std::shared_ptr<Autotuner<1>> m_mark_tuner;   //!< Tuner for marking particles
    std::shared_ptr<Autotuner<1>> m_remove_tuner; //!< Tuner for removing particles
    std::shared_ptr<Autotuner<1>> m_add_tuner;    //!< Tuner for adding particles
#endif                                            // ENABLE_HIP
#endif                                            // ENABLE_MPI

    bool m_valid_cell_cache;               //!< Flag for validity of cell cache
    SortSignal m_sort_signal;              //!< Signal triggered when particles are sorted
    Nano::Signal<void()> m_virtual_signal; //!< Signal for number of virtual particles changing

    //! Check for a valid snapshot
    bool checkSnapshot(const mpcd::ParticleDataSnapshot& snapshot);

    //! Check if all particles lie within the box
    bool checkInBox(const mpcd::ParticleDataSnapshot& snapshot, std::shared_ptr<const BoxDim> box);

    //! Set the global number of particles (for parallel simulations)
    void setNGlobal(unsigned int nglobal);

    //! Allocate data arrays
    void allocate(unsigned int N_max);

    //! Reallocate data arrays
    void reallocate(unsigned int N_max);

    const static float resize_factor; //!< Amortized growth factor the data arrays
    //! Resize the data
    void resize(unsigned int N);

#ifdef ENABLE_MPI
    //! Setup MPI
    void setupMPI(std::shared_ptr<DomainDecomposition> decomposition);
#endif // ENABLE_MPI
    };
    } // end namespace mpcd
    } // end namespace hoomd
#endif // MPCD_PARTICLE_DATA_H_
