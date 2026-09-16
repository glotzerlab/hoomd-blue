// Copyright (c) 2009-2026 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

/*!
 * \file mpcd/CellList.h
 * \brief Declaration of mpcd::CellList
 */

#ifndef MPCD_CELL_LIST_H_
#define MPCD_CELL_LIST_H_

#ifdef __HIPCC__
#error This header cannot be compiled by nvcc
#endif

#include "CellThermoTypes.h"
#include "CommunicatorUtilities.h"
#include "ParticleData.h"

#include "hoomd/Compute.h"
#include "hoomd/GPUFlags.h"
#include "hoomd/ParticleGroup.h"

#include "hoomd/extern/nano-signal-slot/nano_signal_slot.hpp"
#include <pybind11/pybind11.h>

#include <array>

namespace hoomd
    {
namespace mpcd
    {
//! Computes the MPCD cell list on the CPU
class PYBIND11_EXPORT CellList : public Compute
    {
    public:
    //! Constructor by size (deprecated)
    CellList(std::shared_ptr<SystemDefinition> sysdef, Scalar cell_size, bool shift);

    //! Constructor by dimension
    CellList(std::shared_ptr<SystemDefinition> sysdef, const uint3& global_cell_dim, bool shift);

    //! Destructor
    virtual ~CellList();

    //! Build the cell list
    void compute(uint64_t timestep) override;

    //! Sizes the cell list based on the box
    void computeDimensions();

    //! Start autotuning kernel launch parameters
    void startAutotuning() override;

    //! Check if kernel autotuning is complete
    bool isAutotuningComplete() override;

    //! Get the number of particles per cell
    const GPUArray<unsigned int>& getCellSizeArray() const
        {
        return m_cell_np;
        }

    //! Get the cell velocities from the last call to compute
    const GPUArray<double4>& getCellVelocities() const
        {
        return m_cell_vel;
        }

    //! Get the cell energies from the last call to compute
    const GPUArray<double>& getCellEnergies() const
        {
        return m_cell_energy;
        }

    //! Get the cell temperature from the last call to compute
    const GPUArray<double>& getCellTemperature() const
        {
        return m_cell_temp;
        }

    //! Get the total number of cells in the list
    const unsigned int getNCells() const
        {
        return m_cell_indexer.getNumElements();
        }

    //! Get the cell indexer
    const Index3D& getCellIndexer() const
        {
        return m_cell_indexer;
        }

    //! Get the global cell indexer
    const Index3D& getGlobalCellIndexer() const
        {
        return m_global_cell_indexer;
        }

    //! Get the number of cells in each dimension
    const uint3& getDim() const
        {
        return m_cell_dim;
        }

    //! Get the global number of cells in each dimension
    const uint3& getGlobalDim() const
        {
        return m_global_cell_dim;
        }

    void setGlobalDim(const uint3& global_cell_dim);

    const int3& getOriginIndex() const
        {
        return m_origin_idx;
        }

    //! Obtain the local cell index corresponding to a global cell
    const int3 getLocalCell(const int3& global);

    //! Obtain the global cell corresponding to local cell
    const int3 getGlobalCell(const int3& local);

    //! Wrap a cell into a global cell
    const int3 wrapGlobalCell(const int3& cell);

    //! Get the MPCD cell size (deprecated)
    Scalar3 getCellSize();

    //! Set the MPCD cell size (deprecated)
    void setCellSize(Scalar cell_size);

    //! Get the box that is covered by the cell list
    /*!
     * In MPI simulations, this results in a calculation of the cell list
     * dimension. In non-MPI simulations, the box is returned.
     */
    const BoxDim getCoverageBox()
        {
#ifdef ENABLE_MPI
        computeDimensions();
        return m_cover_box;
#else
        return m_pdata->getBox();
#endif // ENABLE_MPI
        }

#ifdef ENABLE_MPI
    //! Set the number of extra communication cells
    void setNExtraCells(unsigned int num_extra)
        {
        m_num_extra = num_extra;
        m_needs_compute_dim = true;
        }

    //! Get the number of extra communication cells
    unsigned int getNExtraCells() const
        {
        return m_num_extra;
        }

    //! Get the number of communication cells on each face of the box
    const std::array<unsigned int, 6>& getNComm() const
        {
        return m_num_comm;
        }

    //! Check if communication is occurring along a direction
    bool isCommunicating(mpcd::detail::face dir);
#endif // ENABLE_MPI

    //! Get whether grid shifting is enabled
    bool isGridShifting() const
        {
        return m_enable_grid_shift;
        }

    //! Toggle the grid shifting on or off
    /*!
     * \param enable_grid_shift Flag to enable grid shifting if true
     */
    void enableGridShifting(bool enable_grid_shift)
        {
        m_enable_grid_shift = enable_grid_shift;
        if (!m_enable_grid_shift)
            {
            setGridShift(make_scalar3(0, 0, 0));
            }
        }

    //! Get the maximum permitted grid shift (fractional coordinates)
    const Scalar3 getMaxGridShift()
        {
        computeDimensions();
        return m_max_grid_shift;
        }

    // Get the grid shift vector (fractional coordinates)
    const Scalar3& getGridShift() const
        {
        return m_grid_shift;
        }

    //! Set the grid shift vector (fractional coordinates)
    void setGridShift(const Scalar3& shift)
        {
        const Scalar3 max_grid_shift = getMaxGridShift();
        if (std::fabs(shift.x) > max_grid_shift.x || std::fabs(shift.y) > max_grid_shift.y
            || std::fabs(shift.z) > max_grid_shift.z)
            {
            throw std::runtime_error("MPCD grid shift out of range");
            }

        m_grid_shift = shift;
        }

    //! Generates the random grid shift vector
    void drawGridShift(uint64_t timestep);

    //! Gets the group of particles that is coupled to the MPCD solvent through the collision step
    std::shared_ptr<ParticleGroup> getEmbeddedGroup() const
        {
        return m_embed_group;
        }

    //! Sets a group of particles that is coupled to the MPCD solvent through the collision step
    void setEmbeddedGroup(std::shared_ptr<ParticleGroup> embed_group)
        {
        if (embed_group != m_embed_group)
            {
            m_embed_group = embed_group;
            m_force_compute = true;
            }
        }

    //! Gets the cell id array for the embedded particles
    const GPUArray<unsigned int>& getEmbeddedGroupCellIds() const
        {
        return m_embed_cell_ids;
        }

    //! Get the signal for dimensions changing
    /*!
     * \returns A signal that subscribers can attach to be notified that the
     *          cell list dimensions have been updated.
     */
    Nano::Signal<void()>& getSizeChangeSignal()
        {
        return m_dim_signal;
        }
    //! Get the net momentum of the particles from the last call to compute
    Scalar3 getNetMomentum()
        {
        if (m_needs_net_reduce)
            computeNetProperties();
        ArrayHandle<double> h_net_properties(m_net_properties,
                                             access_location::host,
                                             access_mode::read);
        const Scalar3 net_momentum
            = make_scalar3(h_net_properties.data[mpcd::detail::thermo_index::momentum_x],
                           h_net_properties.data[mpcd::detail::thermo_index::momentum_y],
                           h_net_properties.data[mpcd::detail::thermo_index::momentum_z]);
        return net_momentum;
        }

    //! Get the net energy of the particles from the last call to compute
    Scalar getNetEnergy()
        {
        if (!m_flags[mpcd::detail::thermo_options::energy])
            {
            m_exec_conf->msg->error()
                << "Energy requested from CellList, but was not computed." << std::endl;
            throw std::runtime_error("Net cell energy not available");
            }
        if (m_needs_net_reduce)
            computeNetProperties();
        ArrayHandle<double> h_net_properties(m_net_properties,
                                             access_location::host,
                                             access_mode::read);
        return h_net_properties.data[mpcd::detail::thermo_index::energy];
        }

    //! Get the average cell temperature from the last call to compute
    Scalar getTemperature()
        {
        if (!m_flags[mpcd::detail::thermo_options::energy])
            {
            m_exec_conf->msg->error()
                << "Temperature requested from CellList, but was not computed." << std::endl;
            throw std::runtime_error("Net cell temperature not available");
            }
        if (m_needs_net_reduce)
            computeNetProperties();
        ArrayHandle<double> h_net_properties(m_net_properties,
                                             access_location::host,
                                             access_mode::read);
        return h_net_properties.data[mpcd::detail::thermo_index::temperature];
        }

    //! Get the signal for requested thermo flags
    /*!
     * \returns A signal that subscribers can attach a callback to in order
     *          to request certain data.
     *
     * For performance reasons, the CellList should be able to
     * supply many related cell-level quantities from a single kernel launch.
     * However, sometimes these quantities are not needed, and it is better
     * to skip calculating them. Subscribing classes can optionally request
     * some of these quantities via a callback return mpcd::detail::ThermoFlags
     * with the appropriate bits set.
     */
    Nano::Signal<mpcd::detail::ThermoFlags()>& getFlagsSignal()
        {
        return m_flag_signal;
        }

    protected:
    std::shared_ptr<mpcd::ParticleData> m_mpcd_pdata; //!< MPCD particle data
    std::shared_ptr<ParticleGroup> m_embed_group;     //!< Embedded particles

    bool m_enable_grid_shift; //!< Flag to enable grid shifting
    Scalar3 m_grid_shift;     //!< Amount to shift particle positions when computing cell list
    Scalar3 m_max_grid_shift; //!< Maximum amount grid can be shifted in any direction

    //! Allocates internal data arrays
    virtual void reallocate();

    uint3 m_cell_dim;              //!< Number of cells in each direction
    uint3 m_global_cell_dim;       //!< Number of cells in each direction of global simulation box
    Scalar3 m_global_cell_dim_inv; //!< Inverse of number of cells in each direction of global box
    Index3D m_cell_indexer;        //!< Indexer from 3D into cell list 1D
    Index3D m_global_cell_indexer; //!< Indexer from 3D into 1D for global cell indexes
    GPUVector<unsigned int> m_cell_np;        //!< Number of particles per cell
    GPUVector<unsigned int> m_embed_cell_ids; //!< Cell ids of the embedded particles
    GPUFlags<uint3> m_conditions; //!< Detect conditions that might fail building cell list

    int3 m_origin_idx; //!< Origin as a global index

    GPUVector<double4> m_cell_vel;     //!< Average velocity of a cell + cell mass
    GPUVector<double> m_cell_energy;   //!< Kinetic energy
    GPUVector<double> m_cell_temp;     //!< Unscaled temperature
    GPUArray<double> m_net_properties; //!< Scalar properties of the system
    bool m_needs_net_reduce;           //!< Flag if a net reduction is necessary

    Nano::Signal<mpcd::detail::ThermoFlags()> m_flag_signal; //!< Signal for requested flags
    mpcd::detail::ThermoFlags m_flags;                       //!< Requested thermo flags

#ifdef ENABLE_MPI
    unsigned int m_num_extra;               //!< Number of extra cells to communicate over
    std::array<unsigned int, 6> m_num_comm; //!< Number of cells to communicate on each face
    BoxDim m_cover_box;                     //!< Box covered by the cell list

    //! Determine if embedded particles require migration
    virtual bool needsEmbedMigrate(uint64_t timestep);
#endif // ENABLE_MPI

    //! Updates the requested optional flags
    void updateFlags();

    //! Check the condition flags
    bool checkConditions();

    //! Reset the conditions array
    void resetConditions();

    //! Builds the cell list and handles cell list memory
    virtual void buildCellList();

    //! Do final cell property calculation
    virtual void finishComputeProperties();

    //! Compute the net properties of all the cells
    virtual void computeNetProperties();

    private:
    bool m_needs_compute_dim; //!< True if the dimensions need to be (re-)computed
    //! Slot for box resizing
    void slotBoxChanged()
        {
        m_needs_compute_dim = true;
        }

    Nano::Signal<void()> m_dim_signal; //!< Signal for dimensions changing
    //! Notify subscribers that dimensions have changed
    void notifySizeChange()
        {
        m_dim_signal.emit();
        }

    bool m_particles_sorted; //!< True if any embedded particles have been sorted
    //! Slot for particle sorting
    void slotSorted()
        {
        m_particles_sorted = true;
        }

    bool m_virtual_change; //!< True if the number of virtual particles has changed
    //! Slot for the number of virtual particles changing
    void slotNumVirtual()
        {
        m_virtual_change = true;
        }

    //! Update global simulation box and check that cell list is compatible with it
    void updateGlobalBox();

#ifdef ENABLE_MPI
    std::shared_ptr<DomainDecomposition> m_decomposition;
#endif // ENABLE_MPI
    };
    } // end namespace mpcd
    } // end namespace hoomd
#endif // MPCD_CELL_LIST_H_
